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

module dense #(

    // Fixed-point format and geometry
    parameter DATA_WIDTH = 16,  // signed word length for inputs/weights/biases/output
    parameter FRAC_BITS  = 7,   // fractional bits in Q(FRAC_BITS) format
    parameter IN_DIM     = 1568, // input length (e.g., 8*14*14 after conv/pool)
    parameter OUT_DIM    = 10    // number of output neurons/classes
)(
    // Clock / control
    input  logic clk,            // rising-edge clock
    input  logic reset,          // synchronous active-high reset
    input  logic start,          // pulse to begin one full dense pass

    // Inputs
    input  logic signed [DATA_WIDTH-1:0] in_vec   [0:IN_DIM-1],                 // Q(FRAC_BITS)
    input  logic signed [DATA_WIDTH-1:0] weights  [0:OUT_DIM-1][0:IN_DIM-1],    // Q(FRAC_BITS) [o][i]
    input  logic signed [DATA_WIDTH-1:0] biases   [0:OUT_DIM-1],                // Q(FRAC_BITS)

    // Outputs
    output logic signed [DATA_WIDTH-1:0] out_vec  [0:OUT_DIM-1],                // Q(FRAC_BITS)
    output logic done                     // pulses high for one cycle at completion
);

    // ACCW: widened accumulator to safely hold sum of IN_DIM products
    localparam ACCW = DATA_WIDTH*2 + $clog2(IN_DIM);

    // FSM states
    typedef enum logic [1:0] {IDLE, ACCUM, WRITE, FINISH} state_t;
    state_t state;            

    integer o, i;             // output index and input index
    logic signed [ACCW-1:0]         acc;   // accumulator (bias + dot product)
    logic signed [2*DATA_WIDTH-1:0] prod;  // single product in_vec[i]*weights[o][i]

    // Saturation bounds for final write-back
    localparam logic signed [DATA_WIDTH-1:0] S_MAX = (1 <<< (DATA_WIDTH-1)) - 1;
    localparam logic signed [DATA_WIDTH-1:0] S_MIN = - (1 <<< (DATA_WIDTH-1));

    // Saturation statistics for debugging/analysis
    logic [31:0] sat_pos_cnt, sat_neg_cnt;

    //==================================================================
    // Sequential process: FSM + MAC + writeback
    //==================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            // Clear state, flags, counters, and stats
            state <= IDLE; done <= 1'b0;
            o <= 0; i <= 0; acc <= '0; prod <= '0;
            sat_pos_cnt <= 0; sat_neg_cnt <= 0;

        end else begin
            // Default: `done` only pulses when entering FINISH
            done <= 1'b0;

            unique case (state)

              //----------------------------------------------------------
              // IDLE — wait for `start`, seed indices and accumulator
              //----------------------------------------------------------
              IDLE: if (start) begin
                        o <= 0;         // first output neuron
                        i <= 0;         // start dot product at in_vec[0]
                        // Preload accumulator with bias[o] in accumulator scale:
                        // sign-extend to ACCW, then << FRAC_BITS
                        acc <= ({{(ACCW-DATA_WIDTH){biases[0][DATA_WIDTH-1]}}, biases[0]}) <<< FRAC_BITS;
                        state <= ACCUM;
                    end

              //----------------------------------------------------------
              // ACCUM — one multiply-add per cycle over i = 0..IN_DIM-1
              //----------------------------------------------------------
              ACCUM: begin
                        // Form current product; blocking so `prod` is usable immediately
                        prod = in_vec[i] * weights[o][i];

                        // Add into accumulator.
                        // NOTE: This uses a *blocking* assignment to `acc` so that the
                        // next statement in this cycle would see the updated value.
                        acc  = acc + prod;

                        // Advance input index or move on to WRITE when last element is done
                        if (i == IN_DIM-1) state <= WRITE;
                        else               i     <= i + 1;
                     end

              //----------------------------------------------------------
              // WRITE — scale back to Q(FRAC_BITS), saturate, store out[o]
              //----------------------------------------------------------
              WRITE: begin
                        logic signed [ACCW-1:0]       shifted;
                        logic signed [DATA_WIDTH-1:0] res;

                        // Convert from accumulator scale back to Q(FRAC_BITS)
                        shifted = acc >>> FRAC_BITS;

                        // Saturate into DATA_WIDTH range and track clips
                        if (shifted > S_MAX) begin
                            res <= S_MAX; sat_pos_cnt <= sat_pos_cnt + 1;
                        end else if (shifted < S_MIN) begin
                            res <= S_MIN; sat_neg_cnt <= sat_neg_cnt + 1;
                        end else begin
                            res <= shifted[DATA_WIDTH-1:0];
                        end

                        // Commit the output neuron
                        out_vec[o] <= res;

                        // Either finish or set up for the next output index
                        if (o == OUT_DIM-1) begin
                            state <= FINISH;
                        end else begin
                            o   <= o + 1;    // next neuron
                            i   <= 0;        // restart input index
                            // Preload acc with next bias in accumulator scale
                            acc <= ({{(ACCW-DATA_WIDTH){biases[o+1][DATA_WIDTH-1]}}, biases[o+1]}) <<< FRAC_BITS;
                            state <= ACCUM;
                        end
                      end

              //----------------------------------------------------------
              // FINISH — pulse `done` and return to IDLE
              //----------------------------------------------------------
              FINISH: begin
                        done  <= 1'b1;
                        state <= IDLE;
                      end
            endcase
        end
    end
endmodule