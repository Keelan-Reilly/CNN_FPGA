//======================================================================
// relu.sv — ReLU Activation Layer
//----------------------------------------------------------------------
// What this module does:
//   • Applies ReLU (Rectified Linear Unit) activation to all inputs.
//   • For each element: output = max(0, input).
//   • On `start`, processes the whole feature map once and pulses `done`.
//
// Key points:
//   • Input:  in_feature[channel][row][col] (signed).
//   • Output: same size as input; negative values set to zero.
//   • FSM steps through channels, rows, and columns in sequence.
//======================================================================

module relu #(
    parameter DATA_WIDTH = 16,  // signed width of input/output values
    parameter FRAC_BITS  = 7,   // fractional bits (not used in logic, for consistency)
    parameter CHANNELS   = 8,   // number of channels
    parameter IMG_SIZE   = 28   // height = width of feature maps
)(
    input  logic clk, reset, start, // clock, reset, control
    input  logic signed [DATA_WIDTH-1:0]
           in_feature [0:CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1],
    output logic signed [DATA_WIDTH-1:0]
           out_feature[0:CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1],
    output logic done
);

    // FSM states
    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;

    // Loop indices: channel, row, column
    integer c, r, q;

    //==================================================================
    // FSM: cycles through all pixels, applies ReLU, writes output
    //==================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; done <= 0; c <= 0; r <= 0; q <= 0;
        end else case (state)

            //----------------------------------------------------------
            // IDLE — wait for start
            //----------------------------------------------------------
            IDLE: begin
                done <= 0;
                if (start) begin
                    c <= 0; r <= 0; q <= 0;
                    state <= RUN;
                end
            end

            //----------------------------------------------------------
            // RUN — apply ReLU: negative values clamped to 0
            //----------------------------------------------------------
            RUN: begin
                logic signed [DATA_WIDTH-1:0] v = in_feature[c][r][q];
                out_feature[c][r][q] <= v[DATA_WIDTH-1] ? '0 : v;

                // Advance spatial/channel counters
                if (q == IMG_SIZE-1) begin
                    q <= 0;
                    if (r == IMG_SIZE-1) begin
                        r <= 0;
                        if (c == CHANNELS-1) state <= FINISH;
                        else c <= c + 1;
                    end else r <= r + 1;
                end else q <= q + 1;
            end

            //----------------------------------------------------------
            // FINISH — pulse done and return to IDLE
            //----------------------------------------------------------
            FINISH: begin
                done <= 1;
                state <= IDLE;
            end
        endcase
    end
endmodule