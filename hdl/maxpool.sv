//======================================================================
// maxpool.sv — 2×2 Max-Pooling Layer
//----------------------------------------------------------------------
// What this module does:
//   • Takes feature maps of size IN_SIZE×IN_SIZE for each channel.
//   • Applies 2×2 max pooling (POOL=2 by default).
//   • Produces downsampled feature maps of size (IN_SIZE/2)×(IN_SIZE/2).
//   • On `start`, processes the entire input once and pulses `done` when finished.
//
// Key points:
//   • Input:  in_feature[channel][row][col].
//   • Output: out_feature[channel][row][col] with reduced spatial size.
//   • For each 2×2 block, the maximum value is chosen.
//   • FSM steps through all channels, rows, and columns systematically.
//======================================================================

module maxpool #(
    parameter int DATA_WIDTH = 16,
    parameter int CHANNELS   = 8,
    parameter int IN_SIZE    = 28,
    parameter int POOL       = 2
)(
    input  logic clk, reset, start,
    input  logic signed [DATA_WIDTH-1:0] in_feature_flat  [0:CHANNELS*IN_SIZE*IN_SIZE-1],
    output logic signed [DATA_WIDTH-1:0] out_feature_flat [0:CHANNELS*(IN_SIZE/POOL)*(IN_SIZE/POOL)-1],
    output logic done
);
    localparam int OUT_SIZE = IN_SIZE/POOL;
    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;
    integer ch, r, q;

    function automatic int idx3(
        input int ch_i, input int row_i, input int col_i,
        input int H_i,  input int W_i
    );
        return (ch_i*H_i + row_i)*W_i + col_i;
    endfunction

    function automatic logic signed [DATA_WIDTH-1:0] max2(
        input logic signed [DATA_WIDTH-1:0] a,b
    ); return (a>b)?a:b; endfunction

    function automatic logic signed [DATA_WIDTH-1:0] max4(
        input logic signed [DATA_WIDTH-1:0] a,b,x,y
    ); return max2(max2(a,b), max2(x,y)); endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
            state<=IDLE; done<=0; ch<=0; r<=0; q<=0;
        end else case (state)
            IDLE: begin
                done<=0;
                if (start) begin
                    ch<=0; r<=0; q<=0;
                    state<=RUN;
                end
            end
            RUN: begin
                out_feature_flat[idx3(ch, r, q, OUT_SIZE, OUT_SIZE)] <= max4(
                    in_feature_flat[idx3(ch, 2*r,   2*q,   IN_SIZE, IN_SIZE)],
                    in_feature_flat[idx3(ch, 2*r,   2*q+1, IN_SIZE, IN_SIZE)],
                    in_feature_flat[idx3(ch, 2*r+1, 2*q,   IN_SIZE, IN_SIZE)],
                    in_feature_flat[idx3(ch, 2*r+1, 2*q+1, IN_SIZE, IN_SIZE)]
                );

                if (q==OUT_SIZE-1) begin
                    q<=0;
                    if (r==OUT_SIZE-1) begin
                        r<=0;
                        if (ch==CHANNELS-1) state<=FINISH;
                        else ch<=ch+1;
                    end else r<=r+1;
                end else q<=q+1;
            end
            FINISH: begin
                done<=1;
                state<=IDLE;
            end
        endcase
    end
endmodule
