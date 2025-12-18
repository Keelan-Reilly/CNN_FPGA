`timescale 1ns/1ps
`default_nettype none

module tb_maxpool;
  localparam int DW=16, C=1, IN_SIZE=4, POOL=2;
  localparam int OUT_SIZE=IN_SIZE/POOL;

  logic clk=0, reset=1; always #5 clk=~clk;
  initial begin repeat(4) @(posedge clk); reset=0; end

  // CONV buffer source (1-cycle latency)
  localparam int CONV_SZ = C*IN_SIZE*IN_SIZE;
  localparam int CONV_AW = (CONV_SZ<=1)?1:$clog2(CONV_SZ);
  logic [CONV_AW-1:0] conv_addr;
  logic conv_en;
  logic signed [DW-1:0] conv_q;

  logic [CONV_AW-1:0] conv_addr_q;
  logic conv_en_q;
  logic signed [DW-1:0] conv_mem [0:CONV_SZ-1];

  function int lin3(input int ch,input int r,input int c, input int H, input int W);
    return (ch*H + r)*W + c;
  endfunction

  initial begin
    // Fill 4x4 with 0..15
    for (int r=0;r<IN_SIZE;r++)
      for (int c=0;c<IN_SIZE;c++)
        conv_mem[lin3(0,r,c,IN_SIZE,IN_SIZE)] = r*IN_SIZE + c;
  end

  always_ff @(posedge clk) begin
    conv_addr_q <= conv_addr;
    conv_en_q   <= conv_en;
  end

  always_comb begin
    conv_q = (conv_en_q) ? conv_mem[conv_addr_q] : '0;
  end

  // POOL buffer capture
  localparam int POOL_SZ = C*OUT_SIZE*OUT_SIZE;
  localparam int POOL_AW = (POOL_SZ<=1)?1:$clog2(POOL_SZ);
  logic [POOL_AW-1:0] pool_addr;
  logic pool_en, pool_we;
  logic signed [DW-1:0] pool_d;
  logic signed [DW-1:0] pool_mem [0:POOL_SZ-1];

  always_ff @(posedge clk) begin
    if (pool_en && pool_we) pool_mem[pool_addr] <= pool_d;
  end

  logic start, done;

  maxpool #(
    .DATA_WIDTH(DW), .CHANNELS(C), .IN_SIZE(IN_SIZE), .POOL(POOL)
  ) dut (
    .clk, .reset, .start,
    .conv_addr, .conv_en, .conv_q,
    .pool_addr, .pool_en, .pool_we, .pool_d,
    .done
  );

  // Golden 2x2 = max of each 2x2 block
  initial begin
    int guard;
    int errs;
    @(negedge reset); repeat(2) @(posedge clk);
    start<=1; @(posedge clk); start<=0;

    guard=0; while(!done && guard<10000) begin @(posedge clk); guard++; end
    if (!done) $fatal(1,"maxpool timeout");

    // Expected: [[5,7],[13,15]] in row-major
    errs=0;
    if (pool_mem[0]!==5 ) begin $display("p[0] got %0d exp 5",  pool_mem[0]); errs++; end
    if (pool_mem[1]!==7 ) begin $display("p[1] got %0d exp 7",  pool_mem[1]); errs++; end
    if (pool_mem[2]!==13) begin $display("p[2] got %0d exp 13", pool_mem[2]); errs++; end
    if (pool_mem[3]!==15) begin $display("p[3] got %0d exp 15", pool_mem[3]); errs++; end

    if (errs==0) $display("PASS: maxpool 4x4→2x2 correct.");
    else         $error("FAIL: maxpool mismatches=%0d", errs);
    $finish;
  end
endmodule
