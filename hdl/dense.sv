//======================================================================
// dense.sv — Fully-connected (dense) layer for FPGA CNN accelerator
//----------------------------------------------------------------------
// What this module does:
//   • Takes a flat input vector of length IN_DIM and computes OUT_DIM outputs.
//   • Each output is: dot_product(in_vec, weights[o][:]) + bias[o].
//   • Uses fixed-point maths; results are scaled back and clipped to the
//     DATA_WIDTH range.
//   • You pulse `start` to run one full pass; `done` pulses when all
//     OUT_DIM outputs have been written.
//
// How it runs internally:
//   • A small FSM outputs one by one.
//   • For each output index `o`, it accumulates in_vec[i] * weights[o][i]
//     over all i, starting from the bias.
//   • After the last multiply-add for that output, it scales, saturates,
//     writes out_vec[o], then moves to the next output (or finishes).
//======================================================================
(* keep_hierarchy = "yes" *)
module dense #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS  = 7,
    parameter int IN_DIM     = 1568,
    parameter int OUT_DIM    = 10,
    parameter string WEIGHTS_FILE = "fc1_weights.mem",
    parameter string BIASES_FILE  = "fc1_biases.mem"
)(
    input  logic clk,
    input  logic reset,
    input  logic start,

    input  logic signed [DATA_WIDTH-1:0] in_vec  [0:IN_DIM-1],
    output logic signed [DATA_WIDTH-1:0] out_vec [0:OUT_DIM-1],

    output logic done
);
    localparam int ACCW = DATA_WIDTH*2 + $clog2(IN_DIM);
    typedef enum logic [1:0] {IDLE, ACCUM, WRITE, FINISH} state_t;
    state_t state;

    integer o, i;
    logic signed [ACCW-1:0]         acc;
    (* use_dsp = "yes" *) logic signed [2*DATA_WIDTH-1:0] prod;

    localparam logic signed [DATA_WIDTH-1:0] S_MAX = (1 <<< (DATA_WIDTH-1)) - 1;
    localparam logic signed [DATA_WIDTH-1:0] S_MIN = - (1 <<< (DATA_WIDTH-1));

    function automatic int w_idx(input int oo, input int ii);
        return oo*IN_DIM + ii;
    endfunction

    localparam int W_DEPTH = OUT_DIM*IN_DIM;
    (* rom_style = "block", ram_style = "block" *) logic signed [DATA_WIDTH-1:0] W [0:W_DEPTH-1];
    (* rom_style = "block", ram_style = "block" *) logic signed [DATA_WIDTH-1:0] B [0:OUT_DIM-1];

    initial begin
        $readmemh(WEIGHTS_FILE, W);
        $readmemh(BIASES_FILE,  B);
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; done <= 1'b0;
            o <= 0; i <= 0; acc <= '0; prod <= '0;
        end else begin
            done <= 1'b0;
            unique case (state)
              IDLE: if (start) begin
                        o <= 0; i <= 0;
                        acc <= ({{(ACCW-DATA_WIDTH){B[0][DATA_WIDTH-1]}}, B[0]}) <<< FRAC_BITS;
                        state <= ACCUM;
                    end
              ACCUM: begin
                        prod = in_vec[i] * W[w_idx(o,i)];
                        acc  = acc + prod;
                        if (i == IN_DIM-1) state <= WRITE;
                        else               i <= i + 1;
                     end
              WRITE: begin
                        logic signed [ACCW-1:0]       shifted;
                        logic signed [DATA_WIDTH-1:0] res;
                        shifted = acc >>> FRAC_BITS;
                        if (shifted > S_MAX)      res <= S_MAX;
                        else if (shifted < S_MIN) res <= S_MIN;
                        else                      res <= shifted[DATA_WIDTH-1:0];
                        out_vec[o] <= res;

                        if (o == OUT_DIM-1) begin
                            state <= FINISH;
                        end else begin
                            o <= o + 1; i <= 0;
                            acc <= ({{(ACCW-DATA_WIDTH){B[o+1][DATA_WIDTH-1]}}, B[o+1]}) <<< FRAC_BITS;
                            state <= ACCUM;
                        end
                     end
              FINISH: begin
                        done <= 1'b1;
                        state <= IDLE;
                     end
            endcase
        end
    end
endmodule
