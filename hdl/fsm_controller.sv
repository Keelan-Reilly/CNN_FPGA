//------------------------------------------------------------------------------
// Module: fsm_controller
//
// Overview
//   Top-level pipeline controller for a CNN inference datapath. The controller
//   sequences the stages (CONV → RELU → POOL → FLAT → DENSE → TX) so that only
//   one stage is active at a time. Each stage is started with a single-cycle
//   start pulse and is considered complete when its corresponding *_done flag
//   asserts.
//
// Handshake model
//   • Inputs:
//       - frame_loaded: indicates a complete input frame is available.
//       - *_done:       stage completion strobes/levels (assumed synchronous).
//       - tx_ready:     prediction ready for transmit (e.g., argmax done).
//       - tx_busy:      UART transmitter busy (prevents premature tx_start).
//   • Outputs:
//       - *_start: 1-cycle pulses to launch each stage.
//       - busy:    high while processing a frame end-to-end.
//
// Sequencing
//   IDLE  : wait for frame_loaded, then pulse conv_start.
//   CONV  : wait conv_done, then pulse relu_start.
//   RELU  : wait relu_done, then pulse pool_start.
//   POOL  : wait pool_done, then advance to FLAT.
//   FLAT  : wait flat_done, then pulse dense_start.
//   DENSE : wait dense_done, then advance to TX.
//   TX    : wait tx_ready AND tx_busy deasserted, then pulse tx_start.
//   WAIT  : one-cycle buffer (tail) then return to IDLE.
//
// Notes
//   • Start pulses are deasserted by default every cycle; they are asserted
//     only on the transition into the corresponding stage.
//   • If any *_done is level-high for multiple cycles, the FSM will still only
//     issue a single start pulse for the next stage because it advances state
//     immediately when the condition is met.
//------------------------------------------------------------------------------

module fsm_controller (
    input  logic clk,
    input  logic reset,

    // Trigger to start the pipeline (frame has been loaded via UART)
    input  logic frame_loaded,

    // Done flags from stages
    input  logic conv_done,
    input  logic relu_done,
    input  logic pool_done,
    input  logic flat_done,
    input  logic dense_done,
    input  logic tx_ready,   // e.g. argmax_done
    input  logic tx_busy,    // UART TX busy

    // One-cycle start pulses to stages
    output logic conv_start,
    output logic relu_start,
    output logic pool_start,
    output logic dense_start,
    output logic tx_start,

    // Status
    output logic busy
);

    typedef enum logic [2:0] {
        IDLE,    // waiting for a new frame
        CONV,    // convolution running
        RELU,    // ReLU running
        POOL,    // pooling running
        FLAT,    // flattening (performed outside this module)
        DENSE,   // dense layer running
        TX,      // waiting to transmit prediction
        WAIT     // one-cycle tail state before returning to IDLE
    } state_t;

    state_t state;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            busy  <= 1'b0;
            {conv_start, relu_start, pool_start, dense_start, tx_start} <= '0;
        end else begin
            // Default: no stage starts unless explicitly pulsed below
            {conv_start, relu_start, pool_start, dense_start, tx_start} <= '0;

            unique case (state)

                IDLE: begin
                    busy <= 1'b0;
                    if (frame_loaded) begin
                        busy       <= 1'b1;
                        conv_start <= 1'b1;
                        state      <= CONV;
                    end
                end

                CONV: begin
                    if (conv_done) begin
                        relu_start <= 1'b1;
                        state      <= RELU;
                    end
                end

                RELU: begin
                    if (relu_done) begin
                        pool_start <= 1'b1;
                        state      <= POOL;
                    end
                end

                POOL: begin
                    if (pool_done) begin
                        state <= FLAT;
                    end
                end

                FLAT: begin
                    if (flat_done) begin
                        dense_start <= 1'b1;
                        state       <= DENSE;
                    end
                end

                DENSE: begin
                    if (dense_done) begin
                        state <= TX;
                    end
                end

                TX: begin
                    if (tx_ready && !tx_busy) begin
                        tx_start <= 1'b1;
                        state    <= WAIT;
                    end
                end

                WAIT: begin
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule
