// dense.sv
module dense #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS  = 7,
    parameter IN_DIM     = 1568, // 8*14*14
    parameter OUT_DIM    = 10
)(
    input  logic clk, reset, start,
    input  logic signed [DATA_WIDTH-1:0] in_vec   [0:IN_DIM-1],
    input  logic signed [DATA_WIDTH-1:0] weights  [0:OUT_DIM-1][0:IN_DIM-1], // [o][i]
    input  logic signed [DATA_WIDTH-1:0] biases   [0:OUT_DIM-1],
    output logic signed [DATA_WIDTH-1:0] out_vec  [0:OUT_DIM-1],
    output logic done
);
    localparam ACCW = DATA_WIDTH*2 + $clog2(IN_DIM);
    typedef enum logic [1:0] {IDLE, ACCUM, NEXT, FINISH} state_t;
    state_t state;

    integer o, i;
    logic signed [ACCW-1:0] acc;
    logic signed [2*DATA_WIDTH-1:0] prod;

    function automatic logic signed [DATA_WIDTH-1:0] scale_sat(input logic signed [ACCW-1:0] v);
        logic signed [ACCW-1:0] s = v >>> FRAC_BITS;
        logic signed [DATA_WIDTH-1:0] mx = {1'b0, {(DATA_WIDTH-1){1'b1}}};
        logic signed [DATA_WIDTH-1:0] mn = {1'b1, {(DATA_WIDTH-1){1'b0}}};
        if (s > mx) return mx;
        else if (s < mn) return mn;
        else return s[DATA_WIDTH-1:0];
    endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
            state<=IDLE; done<=0; o<=0; i<=0; acc<='0;
        end else case (state)
            IDLE: begin
                done<=0;
                if (start) begin 
                    o<=0; 
                    i<=0; 
                    acc <= ({{(ACCW-DATA_WIDTH){biases[0][DATA_WIDTH-1]}}, biases[0]}) <<< FRAC_BITS; 
                    state <= ACCUM;
                end
            end
            ACCUM: begin
                prod = in_vec[i] * weights[o][i];
                acc  = acc + prod;
                if (i == IN_DIM-1) state <= NEXT;
                else i <= i + 1;
            end
            NEXT: begin
                out_vec[o] <= scale_sat(acc);
                if (o == OUT_DIM-1) begin state<=FINISH; end
                else begin
                    o <= o + 1; i <= 0;
                    acc <= ({{(ACCW-DATA_WIDTH){biases[o+1][DATA_WIDTH-1]}}, biases[o+1]}) <<< FRAC_BITS;
                    state <= ACCUM;
                end
            end
            FINISH: begin done<=1; state<=IDLE; end
        endcase
    end
endmodule