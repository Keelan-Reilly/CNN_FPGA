//------------------------------------------------------------------------------
//  Argmax Module
//  -------------
//  This module scans a vector of signed values and determines the index of the
//  maximum element (argmax). Upon receiving a start signal, it iterates through
//  the input vector, compares each value, and tracks the largest value found.
//  When the scan is complete, it outputs the index of the maximum value and
//  pulses a 'done' signal to indicate completion. The design uses a simple
//  finite state machine (FSM) to control the process, ensuring one comparison
//  per clock cycle.
//------------------------------------------------------------------------------

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
            state <= IDLE; 
            done <= 1'b0; 
            idx <= '0;
            i <= 0; 
            besti <= '0;
            bestv <= $signed({1'b1, {(DATA_WIDTH-1){1'b0}}});
        end else begin
            case (state)
              IDLE: begin
                  done <= 1'b0;
                  if (start) begin
                      i <= 1; 
                      besti <= '0; 
                      bestv <= vec[0];
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
