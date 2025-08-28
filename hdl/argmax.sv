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

module argmax #(

    parameter int DATA_WIDTH = 16, // width of each input element
    parameter int DIM        = 10, // number of elements in input vector
    // width of output index (minimum #bits needed to address DIM elements)
    parameter int IDXW       = (DIM <= 1) ? 1 : $clog2(DIM)
)(
    // Clock / control
    input  logic clk, reset, start,  // clk, synchronous reset, and start pulse

    // Inputs
    input  logic signed [DATA_WIDTH-1:0] vec [0:DIM-1], // input vector

    // Outputs
    output logic [IDXW-1:0] idx,   // index of the maximum element
    output logic done              // pulses high for one cycle at completion
);

    // FSM states
    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;

    integer i;                                    // loop counter
    logic signed [DATA_WIDTH-1:0] bestv;          // current best value
    logic [IDXW-1:0]              besti;          // current best index

    //==================================================================
    // Sequential process: FSM
    //==================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            // Reset all state and outputs
            state <= IDLE; done <= 1'b0; idx <= '0;
            i <= 0; besti <= '0;
            // Initialise bestv to most-negative possible value
            bestv <= $signed({1'b1, {(DATA_WIDTH-1){1'b0}}});

        end else begin
            case (state)

              //--------------------------------------------------------
              // IDLE — Wait for start; seed best value with vec[0]
              //--------------------------------------------------------
              IDLE: begin
                  done <= 1'b0;
                  if (start) begin
                      i     <= 1;          // begin at second element
                      besti <= '0;         // current best index = 0
                      bestv <= vec[0];     // current best value = vec[0]
                      state <= RUN;
                  end
              end

              //--------------------------------------------------------
              // RUN — Step through vec[1..DIM-1], update best if larger
              //--------------------------------------------------------
              RUN: begin
                  if (vec[i] > bestv) begin
                      bestv <= vec[i];           // new best value
                      besti <= i[IDXW-1:0];      // new best index
                  end
                  // Either finished scanning or increment to next index
                  if (i == DIM-1) state <= FINISH;
                  else            i     <= i + 1;
              end

              //--------------------------------------------------------
              // FINISH — Output result and pulse done
              //--------------------------------------------------------
              FINISH: begin
                  idx  <= besti;  // latch result
                  done <= 1'b1;   // signal completion
                  state<= IDLE;   // return to idle
              end
            endcase
        end
    end
endmodule