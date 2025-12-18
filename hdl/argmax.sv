//------------------------------------------------------------------------------
// Module: argmax
//
// Overview
//   Finds the index of the maximum element (argmax) in a vector of signed
//   values. The design performs one signed comparison per cycle using a
//   simple FSM, updating the “best so far” value and index as it scans.
//
// Operation
//   • On `start`, seed the running best with vec[0] and begin scanning at i=1.
//   • Each cycle in RUN compares vec[i] against bestv:
//       - if vec[i] > bestv, update bestv and besti
//   • After comparing the final element vec[DIM-1], latch the result to `idx`
//     and pulse `done` for one cycle.
//
// Notes
//   • Signed comparison is used throughout.
//   • The internal loop index `i` is an integer (control-only), while the
//     exported index is `IDXW` bits wide.
//   • Assumes DIM >= 1. For DIM == 1, the output is always 0.
//------------------------------------------------------------------------------

(* keep_hierarchy = "yes" *)
module argmax #(
    parameter int DATA_WIDTH = 16,                             // element bit width (signed)
    parameter int DIM        = 10,                             // vector length
    parameter int IDXW       = (DIM <= 1) ? 1 : $clog2(DIM)     // index width
)(
    input  logic clk,
    input  logic reset,
    input  logic start,

    input  logic signed [DATA_WIDTH-1:0] vec [0:DIM-1],         // input vector
    output logic [IDXW-1:0]              idx,                   // argmax index
    output logic                         done                   // one-cycle done pulse
);

    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;

    integer i;                                                  // scan pointer (control)
    logic signed [DATA_WIDTH-1:0] bestv;                        // best value so far
    logic [IDXW-1:0]              besti;                        // best index so far

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            done  <= 1'b0;
            idx   <= '0;

            i     <= 0;
            besti <= '0;
            bestv <= $signed({1'b1, {(DATA_WIDTH-1){1'b0}}});   // most negative value

        end else begin
            done <= 1'b0;                                       // default

            unique case (state)

                // ------------------------------ IDLE ------------------------------
                IDLE: begin
                    if (start) begin
                        bestv <= vec[0];                        // seed with first element
                        besti <= '0;
                        i     <= 1;                             // next element to compare
                        state <= (DIM <= 1) ? FINISH : RUN;      // handle DIM==1 cleanly
                    end
                end

                // ------------------------------ RUN -------------------------------
                RUN: begin
                    if (vec[i] > bestv) begin
                        bestv <= vec[i];
                        besti <= i[IDXW-1:0];
                    end

                    if (i == DIM-1) begin
                        state <= FINISH;                         // last compare complete
                    end else begin
                        i <= i + 1;                              // advance scan
                    end
                end

                // ----------------------------- FINISH -----------------------------
                FINISH: begin
                    idx  <= besti;
                    done <= 1'b1;                                // pulse
                    state<= IDLE;
                end

                default: state <= IDLE;

            endcase
        end
    end

endmodule
