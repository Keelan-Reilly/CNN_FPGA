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
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS  = 7,
    parameter int CHANNELS   = 8,
    parameter int IMG_SIZE   = 28
)(
    input  logic clk, reset, start,
    input  logic signed [DATA_WIDTH-1:0] in_feature_flat  [0:CHANNELS*IMG_SIZE*IMG_SIZE-1],
    output logic signed [DATA_WIDTH-1:0] out_feature_flat [0:CHANNELS*IMG_SIZE*IMG_SIZE-1],
    output logic done
);
    localparam int H = IMG_SIZE, W = IMG_SIZE;
    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;
    integer c, r, q;

    function automatic int idx3(input int ch, input int row, input int col);
        return (ch*H + row)*W + col;
    endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; done <= 0; c <= 0; r <= 0; q <= 0;
        end else case (state)
            IDLE: begin
                done <= 0;
                if (start) begin
                    c <= 0; r <= 0; q <= 0;
                    state <= RUN;
                end
            end
            RUN: begin
                logic signed [DATA_WIDTH-1:0] v = in_feature_flat[idx3(c,r,q)];
                out_feature_flat[idx3(c,r,q)] <= v[DATA_WIDTH-1] ? '0 : v;

                if (q == W-1) begin
                    q <= 0;
                    if (r == H-1) begin
                        r <= 0;
                        if (c == CHANNELS-1) state <= FINISH;
                        else c <= c + 1;
                    end else r <= r + 1;
                end else q <= q + 1;
            end
            FINISH: begin
                done <= 1;
                state <= IDLE;
            end
        endcase
    end
endmodule
