// fsm_controller.sv
module fsm_controller (
    input  logic clk, reset,

    // triggers
    input  logic frame_loaded,     // 28x28 received

    // stage done flags
    input  logic conv_done, 
    input  logic relu_done, 
    input  logic pool_done, 
    input  logic flat_done,        // <-- NEW: from flatten block in top_level
    input  logic dense_done, 
    input  logic tx_ready,         // argmax_done

    // stage starts (1-cycle pulses)
    output logic conv_start, 
    output logic relu_start, 
    output logic pool_start, 
    output logic dense_start, 
    output logic tx_start,

    output logic busy
);
    typedef enum logic [2:0] {IDLE, CONV, RELU, POOL, FLAT, DENSE, TX, WAIT} state_t;
    state_t state;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            {conv_start, relu_start, pool_start, dense_start, tx_start} <= '0;
            busy <= 1'b0;
        end else begin
            // default: no start pulses
            {conv_start, relu_start, pool_start, dense_start, tx_start} <= '0;

            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    if (frame_loaded) begin
                        busy <= 1'b1;
                        conv_start <= 1'b1;
                        state <= CONV;
                    end
                end

                CONV:  if (conv_done) begin relu_start <= 1'b1; state <= RELU; end
                RELU:  if (relu_done) begin pool_start <= 1'b1; state <= POOL; end
                POOL:  if (pool_done)                          state <= FLAT;   // wait for flatten
                FLAT:  if (flat_done) begin dense_start <= 1'b1; state <= DENSE; end
                DENSE: if (dense_done)                          state <= TX;
                TX:    if (tx_ready) begin tx_start <= 1'b1;    state <= WAIT; end
                WAIT:  state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule