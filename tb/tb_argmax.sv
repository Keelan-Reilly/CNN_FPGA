`timescale 1ns/1ps
`default_nettype none

module tb_argmax;                               // Testbench for argmax
  localparam int DW=16;                         // Data width for each element in the input vector.
  localparam int DIM=6;                         // Number of elements in the input vector.
  localparam int IDXW=(DIM<=1)?1:$clog2(DIM);   // Index width for the output.

  logic clk=0, reset=1, start; // declare signals
  always #5 clk=~clk;
  initial begin 
    repeat(4) @(posedge clk); // hold reset high for a few cycle
    reset=0;                  // then release
  end

  logic signed [DW-1:0] vec [0:DIM-1];    // input vector (array of signed values). [0:DIM-1] for easy indexing
  logic [IDXW-1:0] idx;                   // index of the maximum element 
  logic done;     

  // Instantiate the DUT
  argmax #(
    .DATA_WIDTH(DW), 
    .DIM(DIM), 
    .IDXW(IDXW)
  ) dut (
    .clk(clk),
    .reset(reset),
    .start(start),
    .vec(vec),
    .idx(idx),
    .done(done)
  );

  // Loads a test vector into DUT, pulses start, waits for done, and checks output index against expected.
  task automatic run_case(
    input logic signed [DW-1:0] a [0:DIM-1],   // test input vector
    input int exp_idx);                        // expected output index

    int g;                                     // general counter
    for (int i=0;i<DIM;i++) vec[i]=a[i];       // load input vector into DUT

    // pulse start high for one cycle
    start<=1; 
    @(posedge clk); 
    start<=0;

  // wait until done or timeout
    g=0; 
    while(!done && g<1000) begin 
        @(posedge clk); 
        g++;
    end
    if (!done) $fatal(1,"argmax timeout");                                         // Timeout check    
    if (idx!==exp_idx[IDXW-1:0]) $error("argmax got %0d exp %0d", idx, exp_idx);   // Check output
    else $display("PASS: argmax -> %0d", idx);
  endtask

  // Main initial block: define test vector and call run_case
  initial begin

    // Declare some test vectors
    logic signed [DW-1:0] a0 [0:DIM-1];
    logic signed [DW-1:0] a1 [0:DIM-1];
    logic signed [DW-1:0] a2 [0:DIM-1];

    // Assign values (SystemVerilog array literals)
    a0 = '{-3, -1, -7, -1, -2, -9}; // Max at index 1 (tie with index 3, but argmax picks first)
    a1 = '{0, 5, 2, 5, 1, 4};       // Max at index 1 (tie with index 3, but argmax picks first)
    a2 = '{1, 2, 3, 9, 8, 0};       // Max at index 3

    // wait until reset is released plus 2 cycles
    @(negedge reset); 
    repeat(2) @(posedge clk);

    // Run the test cases
    run_case(a0, 1);
    run_case(a1, 1);
    run_case(a2, 3);
    $display("All argmax tests done.");
    $finish;
  end
endmodule
