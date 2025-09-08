//======================================================================
// argmax.sv — Find index of maximum value in a vector
//----------------------------------------------------------------------
// What this module does:
//   • Takes a vector of DIM signed values.
//   • Finds which index holds the largest value (argmax).
//   • On `start`, it scans through the vector once and, when done,
//     outputs the index of the maximum and pulses `done`.
//
// Key points:
//   • Input:  vec[0..DIM-1] — signed values (fixed-point or integer).
//   • Output: idx = index of the largest element.
//   • Control: assert `start` while IDLE; `done` pulses when complete.
//   • Uses a simple FSM to walk through the vector one element per cycle.
//
// How it runs internally:
//   • IDLE   : Wait for `start`. Initialise "best value" = vec[0].
//   • RUN    : For each element, compare with the current best; update
//              best value/index if larger.
//   • FINISH : Output the best index, pulse `done`, and return to IDLE.
//======================================================================

(* keep_hierarchy = "yes" *)
module argmax #(
    parameter int DATA_WIDTH = 16,
    parameter int DIM        = 10,
    parameter int IDXW       = (DIM <= 1) ? 1 : $clog2(DIM)
)(
    input  logic clk, reset, start,
    input  logic signed [DATA_WIDTH-1:0] vec [0:DIM-1],
    output logic [IDXW-1:0] idx,
    output logic done
);
    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;
    integer i;
    logic signed [DATA_WIDTH-1:0] bestv;
    logic [IDXW-1:0]              besti;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; done <= 1'b0; idx <= '0;
            i <= 0; besti <= '0;
            bestv <= $signed({1'b1, {(DATA_WIDTH-1){1'b0}}});
        end else begin
            case (state)
              IDLE: begin
                  done <= 1'b0;
                  if (start) begin
                      i <= 1; besti <= '0; bestv <= vec[0];
                      state <= RUN;
                  end
              end
              RUN: begin
                  if (vec[i] > bestv) begin
                      bestv <= vec[i];
                      besti <= i[IDXW-1:0];
                  end
                  if (i == DIM-1) state <= FINISH;
                  else            i <= i + 1;
              end
              FINISH: begin
                  idx  <= besti;
                  done <= 1'b1;
                  state<= IDLE;
              end
            endcase
        end
    end
endmodule
