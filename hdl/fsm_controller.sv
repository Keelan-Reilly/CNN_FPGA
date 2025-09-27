//----------------------------------------------------------------------
// Module: fsm_controller
// Description:
//   This module implements a finite state machine (FSM) to control the
//   sequential execution of stages in the CNN processing pipeline.
//   It coordinates the following stages: convolution, ReLU activation,
//   pooling, flattening, dense layer, and argmax/UART transmission.
//   The controller waits for each stage to signal completion (`*_done`),
//   then issues a single-cycle start pulse (`*_start`) to the next stage.
//   It ensures only one stage runs at a time, preventing overlap, and
//   checks the UART transmission busy signal (`tx_busy`) to avoid early
//   transmission. The pipeline begins when `frame_loaded` is asserted,
//   and a busy flag indicates when processing is active.
//----------------------------------------------------------------------


module fsm_controller (
    input  logic clk, reset,

    // Trigger to start the whole pipeline (after UART frame load)
    input  logic frame_loaded,

    // Done flags from each stage
    input  logic conv_done,
    input  logic relu_done,
    input  logic pool_done,
    input  logic flat_done,
    input  logic dense_done,
    input  logic tx_ready,   // argmax_done
    input  logic tx_busy,    // UART TX busy flag

    // Start pulses (1-cycle each) to kick off stages
    output logic conv_start,
    output logic relu_start,
    output logic pool_start,
    output logic dense_start,
    output logic tx_start,

    // Status
    output logic busy         // high while pipeline is active
);

    // FSM states — each corresponds to waiting for a stage to finish
    typedef enum logic [2:0] {
        IDLE,   // waiting for new frame
        CONV,   // convolution running
        RELU,   // relu running
        POOL,   // pooling running
        FLAT,   // flattening (handled in top_level)
        DENSE,  // dense layer running
        TX,     // waiting to transmit prediction
        WAIT    // small buffer state before going idle
    } state_t;
    state_t state;

    //==================================================================
    // FSM: advance through pipeline stages in order
    //==================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            {conv_start, relu_start, pool_start, dense_start, tx_start} <= '0;
            busy <= 1'b0;
        end else begin
            // Default: no stage triggered unless explicitly set
            {conv_start, relu_start, pool_start, dense_start, tx_start} <= '0;

            unique case (state)
              //--------------------------------------------------------
              // IDLE — wait for frame to be loaded
              //--------------------------------------------------------
              IDLE: begin
                  busy <= 1'b0;
                  if (frame_loaded) begin
                      busy       <= 1'b1;
                      conv_start <= 1'b1;   // kick off conv
                      state      <= CONV;
                  end
              end

              //--------------------------------------------------------
              // CONV — wait until convolution finishes
              //--------------------------------------------------------
              CONV:  if (conv_done) begin
                          relu_start <= 1'b1; // start relu
                          state      <= RELU;
                     end

              //--------------------------------------------------------
              // RELU — wait until relu finishes
              //--------------------------------------------------------
              RELU:  if (relu_done) begin
                          pool_start <= 1'b1; // start pooling
                          state      <= POOL;
                     end

              //--------------------------------------------------------
              // POOL — flattening is done outside this FSM
              //--------------------------------------------------------
              POOL:  if (pool_done) state <= FLAT;

              //--------------------------------------------------------
              // FLAT — wait until flattening finishes, then start dense
              //--------------------------------------------------------
              FLAT:  if (flat_done) begin
                          dense_start <= 1'b1;
                          state       <= DENSE;
                     end

              //--------------------------------------------------------
              // DENSE — wait until dense finishes
              //--------------------------------------------------------
              DENSE: if (dense_done) state <= TX;

              //--------------------------------------------------------
              // TX — wait until argmax result ready and UART not busy
              //--------------------------------------------------------
              TX:    if (tx_ready && !tx_busy) begin
                          tx_start <= 1'b1; // trigger UART transmission
                          state    <= WAIT;
                     end

              //--------------------------------------------------------
              // WAIT — buffer cycle before returning to idle
              //--------------------------------------------------------
              WAIT:  state <= IDLE;

              default: state <= IDLE;
            endcase
        end
    end
endmodule
