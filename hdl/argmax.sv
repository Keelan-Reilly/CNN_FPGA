//------------------------------------------------------------------------------
// Module: argmax
// Description:
//   Scans a vector of signed values and returns the index of the largest
//   element (argmax). Uses a small FSM that does one comparison per cycle.
//   On start: seed "best" with vec[0], then walk vec[1..DIM-1], updating the
//   best value/index. Pulses `done` when the scan completes.
//------------------------------------------------------------------------------
(* keep_hierarchy = "yes" *)
module argmax #(
    parameter int DATA_WIDTH = 16,                             // bit width for each element in the input vector.
    parameter int DIM        = 10,                             // number of elements in the input vector.
    parameter int IDXW       = (DIM <= 1) ? 1 : $clog2(DIM)    // bit width for the output index.
)(
    input  logic clk, reset, start,
    input  logic signed [DATA_WIDTH-1:0] vec [0:DIM-1],        // Input vector (array of signed values).
    output logic [IDXW-1:0] idx,                               // Output: index of maximum element (0..DIM-1).
    output logic done
);
    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;

    integer i;
    logic signed [DATA_WIDTH-1:0] bestv;  // Best value found so far
    logic [IDXW-1:0]              besti;  // Index of best value found so far

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; 
            done <= 1'b0; 
            idx <= '0;
            i <= 0; 
            besti <= '0;
            bestv <= $signed({1'b1, {(DATA_WIDTH-1){1'b0}}});   // Smallest possible signed value 1000...0
        end else begin
            case (state)

              IDLE: begin
                  done <= 1'b0;
                  if (start) begin
                      i <= 1;                   // Start scanning at vec[1]
                      besti <= '0;              // seed best index = 0
                      bestv <= vec[0];          // Seed best value with vec[0]
                      state <= RUN;
                  end
              end

              // Compare the current element to the best so far (signed compare):
              RUN: begin
                  if (vec[i] > bestv) begin
                      bestv <= vec[i];              // Update best value.
                      besti <= i[IDXW-1:0];         // Update best index (truncated to IDXW bits).
                  end
                  // Advance the scan; if last element reached, move to FINISH:
                  if (i == DIM-1) state <= FINISH;   // When we've compared vec[DIM-1], finish
                  else            i <= i + 1;        // Otherwise, increment i for next cycle.
              end

              FINISH: begin
                  idx  <= besti;                   // Output the best index found.  
                  done <= 1'b1;
                  state<= IDLE;
              end
            endcase
        end
    end
endmodule
