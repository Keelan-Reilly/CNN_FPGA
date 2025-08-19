// relu.sv
module relu #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS  = 7,
    parameter CHANNELS   = 8,
    parameter IMG_SIZE   = 28
)(
    input  logic clk, reset, start,
    input  logic signed [DATA_WIDTH-1:0] in_feature [0:CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1],
    output logic signed [DATA_WIDTH-1:0] out_feature[0:CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1],
    output logic done
);
    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;
    integer c, r, q;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; done <= 0; c <= 0; r <= 0; q <= 0;
        end else case (state)
            IDLE: begin
                done <= 0;
                if (start) begin c <= 0; r <= 0; q <= 0; state <= RUN; end
            end
            RUN: begin
                logic signed [DATA_WIDTH-1:0] v = in_feature[c][r][q];
                out_feature[c][r][q] <= v[DATA_WIDTH-1] ? '0 : v; // clamp negatives to 0
                if (q == IMG_SIZE-1) begin
                    q <= 0;
                    if (r == IMG_SIZE-1) begin
                        r <= 0;
                        if (c == CHANNELS-1) begin state <= FINISH; end
                        else c <= c + 1;
                    end else r <= r + 1;
                end else q <= q + 1;
            end
            FINISH: begin done <= 1; state <= IDLE; end
        endcase
    end
endmodule