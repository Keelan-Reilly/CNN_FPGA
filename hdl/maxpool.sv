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
    parameter int DATA_WIDTH = 16,  // bits per element
    parameter int CHANNELS   = 8,   // number of channels
    parameter int IN_SIZE    = 28,  // input feature map size
    parameter int POOL       = 2    // pooling factor (default = 2×2)
)(
    input  logic clk, reset, start, // clock, reset, control
    input  logic signed [DATA_WIDTH-1:0]
           in_feature [0:CHANNELS-1][0:IN_SIZE-1][0:IN_SIZE-1],
    output logic signed [DATA_WIDTH-1:0]
           out_feature[0:CHANNELS-1][0:(IN_SIZE/POOL)-1][0:(IN_SIZE/POOL)-1],
    output logic done
);

    localparam int OUT_SIZE = IN_SIZE/POOL;

    // FSM states
    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;

    // Loop indices: channel, row, column
    integer ch, r, q;

    // Helper function: compute max of 4 values (used for 2×2 pooling window)
    function automatic logic signed [DATA_WIDTH-1:0] max4(
        input logic signed [DATA_WIDTH-1:0] a,b,x,y
    );
        logic signed [DATA_WIDTH-1:0] m1 = (a>b)?a:b;
        logic signed [DATA_WIDTH-1:0] m2 = (x>y)?x:y;
        return (m1>m2)?m1:m2;
    endfunction

    //==================================================================
    // FSM: cycles through each 2×2 block, computes max, writes output
    //==================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            state<=IDLE; done<=0; ch<=0; r<=0; q<=0;
        end else case (state)

            //----------------------------------------------------------
            // IDLE — wait for start
            //----------------------------------------------------------
            IDLE: begin
                done<=0;
                if (start) begin
                    ch<=0; r<=0; q<=0;
                    state<=RUN;
                end
            end

            //----------------------------------------------------------
            // RUN — compute max over current 2×2 block
            //----------------------------------------------------------
            RUN: begin
                out_feature[ch][r][q] <= max4(
                    in_feature[ch][2*r][2*q],
                    in_feature[ch][2*r][2*q+1],
                    in_feature[ch][2*r+1][2*q],
                    in_feature[ch][2*r+1][2*q+1]
                );

                // Advance spatial/channel counters
                if (q==OUT_SIZE-1) begin
                    q<=0;
                    if (r==OUT_SIZE-1) begin
                        r<=0;
                        if (ch==CHANNELS-1) state<=FINISH;
                        else ch<=ch+1;
                    end else r<=r+1;
                end else q<=q+1;
            end

            //----------------------------------------------------------
            // FINISH — pulse done and return to IDLE
            //----------------------------------------------------------
            FINISH: begin
                done<=1;
                state<=IDLE;
            end
        endcase
    end
endmodule