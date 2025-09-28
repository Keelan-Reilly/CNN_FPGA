`timescale 1ns/1ps
`default_nettype none

module tb_argmax;
  localparam int DW=16, DIM=6, IDXW=(DIM<=1)?1:$clog2(DIM);

  logic clk=0, reset=1, start; 
  always #5 clk=~clk;
  initial begin 
    repeat(4) @(posedge clk); reset=0; 
  end

  logic signed [DW-1:0] vec [0:DIM-1];
  logic [IDXW-1:0] idx;
  logic done;

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

  task automatic run_case(input logic signed [DW-1:0] a [0:DIM-1], input int exp_idx);
    int g;
    for (int i=0;i<DIM;i++) vec[i]=a[i];

    start<=1; 
    @(posedge clk); 
    start<=0;

    g=0; 
    while(!done && g<1000) begin 
        @(posedge clk); 
        g++;
    end
    if (!done) $fatal(1,"argmax timeout");
    if (idx!==exp_idx[IDXW-1:0]) $error("argmax got %0d exp %0d", idx, exp_idx);
    else $display("PASS: argmax -> %0d", idx);
  endtask

  initial begin
    // Declarations first
    logic signed [DW-1:0] a0 [0:DIM-1];
    logic signed [DW-1:0] a1 [0:DIM-1];
    logic signed [DW-1:0] a2 [0:DIM-1];

    // Now assignments with array-patterns (SV)
    a0 = '{-3, -1, -7, -1, -2, -9}; // tie @1 & 3 -> first max wins
    a1 = '{0, 5, 2, 5, 1, 4};        // tie @1 & 3 -> expect 1
    a2 = '{1, 2, 3, 9, 8, 0};        // clear max @3

    // Then your event controls and calls
    @(negedge reset); repeat(2) @(posedge clk);
    run_case(a0, 1);
    run_case(a1, 1);
    run_case(a2, 3);
    $display("All argmax tests done.");
    $finish;
  end
endmodule
