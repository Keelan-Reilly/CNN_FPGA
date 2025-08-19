// maxpool.sv
module maxpool #(
    parameter DATA_WIDTH = 16,
    parameter CHANNELS   = 8,
    parameter IN_SIZE    = 28,
    parameter POOL       = 2
)(
    input  logic clk, reset, start,
    input  logic signed [DATA_WIDTH-1:0] in_feature [0:CHANNELS-1][0:IN_SIZE-1][0:IN_SIZE-1],
    output logic signed [DATA_WIDTH-1:0] out_feature[0:CHANNELS-1][0:(IN_SIZE/POOL)-1][0:(IN_SIZE/POOL)-1],
    output logic done
);
    localparam OUT_SIZE = IN_SIZE/POOL;
    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;
    integer c, r, q;

    function automatic logic signed [DATA_WIDTH-1:0] max4(
        input logic signed [DATA_WIDTH-1:0] a,b,c,d
    );
        logic signed [DATA_WIDTH-1:0] m1 = (a>b)?a:b;
        logic signed [DATA_WIDTH-1:0] m2 = (c>d)?c:d;
        max4 = (m1>m2)?m1:m2;
    endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
            state<=IDLE; done<=0; c<=0; r<=0; q<=0;
        end else case (state)
            IDLE: begin done<=0; if (start) begin c<=0; r<=0; q<=0; state<=RUN; end end
            RUN: begin
                out_feature[c][r][q] <= max4(
                    in_feature[c][2*r][2*q],
                    in_feature[c][2*r][2*q+1],
                    in_feature[c][2*r+1][2*q],
                    in_feature[c][2*r+1][2*q+1]
                );
                if (q==OUT_SIZE-1) begin
                    q<=0;
                    if (r==OUT_SIZE-1) begin
                        r<=0;
                        if (c==CHANNELS-1) state<=FINISH; else c<=c+1;
                    end else r<=r+1;
                end else q<=q+1;
            end
            FINISH: begin done<=1; state<=IDLE; end
        endcase
    end
endmodule