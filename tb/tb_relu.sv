`timescale 1ns/1ps
`default_nettype none

// 1 cycle ahead, but easier fix than others
module tb_relu;
  localparam int DW=16, C=1, SZ=8; // IMG_SIZE=8 → 64 elements

  logic clk=0, reset=1; always #5 clk=~clk;
  initial begin repeat(4) @(posedge clk); reset=0; end

  localparam int N = C*SZ*SZ;
  localparam int AW = (N<=1)?1:$clog2(N);

  // Backing memory for CONV buffer (single array, emulate dual-port behavior)
  logic signed [DW-1:0] mem [0:N-1];

  // Seed with negatives, zeros, positives
  initial begin
    for (int i=0;i<N;i++) begin
      mem[i] = (i%3==0) ? -i : (i%3==1) ? 0 : i;
    end
  end

  // Port A (read)
  logic [AW-1:0] conv_r_addr; logic conv_r_en;
  logic signed [DW-1:0] conv_r_q;
  // Port B (write)
  logic [AW-1:0] conv_w_addr; logic conv_w_en, conv_w_we;
  logic signed [DW-1:0] conv_w_d;

  always_ff @(posedge clk) begin
    if (conv_r_en) conv_r_q <= mem[conv_r_addr];
    if (conv_w_en && conv_w_we) mem[conv_w_addr] <= conv_w_d;
  end

  logic start, done;

  relu #(.DATA_WIDTH(DW), .CHANNELS(C), .IMG_SIZE(SZ)) dut (
    .clk, .reset, .start,
    .conv_r_addr, .conv_r_en, .conv_r_q,
    .conv_w_addr, .conv_w_en, .conv_w_we, .conv_w_d,
    .done
  );

  initial begin
    int guard;
    int errs;
    @(negedge reset); repeat(2) @(posedge clk);
    start<=1; @(posedge clk); start<=0;

    guard=0; while(!done && guard<100000) begin @(posedge clk); guard++; end
    if (!done) $fatal(1,"relu timeout");

    // Check: all negatives clamped to 0, zeros/positives unchanged
    errs=0;
    for (int i=0;i<N;i++) begin
      logic signed [DW-1:0] v0 = (i%3==0) ? -i : (i%3==1) ? 0 : i;
      logic signed [DW-1:0] exp = (v0<0) ? 0 : v0;
      if (mem[i] !== exp) begin
        $display("relu mismatch @%0d: got %0d exp %0d", i, mem[i], exp); errs++;
      end
    end
    if (errs==0) $display("PASS: relu in-place OK.");
    else         $error("FAIL: relu mismatches=%0d", errs);
    $finish;
  end
endmodule

