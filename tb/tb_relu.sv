`timescale 1ns/1ps
`default_nettype none

// ============================================================
// Dual-port BRAM model with configurable READ latency (Port A).
// - Port A: r_en + r_addr sampled on clk, r_q updates after LAT cycles.
// - Port B: synchronous write (w_en && w_we) on clk.
// - When no valid read response, r_q holds last value.
// - Internal mem[] is accessible via hierarchical reference: bramX.mem[i]
// ============================================================
module tb_dualport_bram #(
  parameter int DW    = 16,
  parameter int DEPTH = 64,
  parameter int LAT   = 1,
  parameter int AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
  input  logic                   clk,

  // Port A (read)
  input  logic                   r_en,
  input  logic [AW-1:0]          r_addr,
  output logic signed [DW-1:0]   r_q,

  // Port B (write)
  input  logic                   w_en,
  input  logic                   w_we,
  input  logic [AW-1:0]          w_addr,
  input  logic signed [DW-1:0]   w_d
);

  logic signed [DW-1:0] mem [0:DEPTH-1];

  localparam int P = (LAT < 0) ? 0 : LAT;

  logic [AW-1:0] addr_pipe [0:P];
  logic          v_pipe    [0:P];

  integer k;

  initial begin
    r_q = '0;
    for (k=0; k<DEPTH; k++) mem[k] = '0;
    for (k=0; k<=$high(addr_pipe); k++) begin
      addr_pipe[k] = '0;
      v_pipe[k]    = 1'b0;
    end
  end

  always_ff @(posedge clk) begin
    addr_pipe[0] <= r_addr;
    v_pipe[0]    <= r_en;

    for (k=1; k<=$high(addr_pipe); k++) begin
      addr_pipe[k] <= addr_pipe[k-1];
      v_pipe[k]    <= v_pipe[k-1];
    end

    if (P == 0) begin
      if (r_en) r_q <= mem[r_addr];
    end else begin
      if (v_pipe[P]) r_q <= mem[addr_pipe[P]];
    end

    if (w_en && w_we) begin
      mem[w_addr] <= w_d;
    end
  end

endmodule


// ============================================================
// Assumes DUT timing:
//   read strobe @ t
//   write @ t+4
// so compare against stage [4] of the monitor pipe.
// ============================================================
module relu_monitor #(
  parameter int AW  = 6,
  parameter int DW  = 16,
  parameter int N   = 64,
  parameter string TAG = "DUT"
)(
  input  logic                   clk,
  input  logic                   reset,

  input  logic                   start,
  input  logic                   done,

  input  logic                   r_en,
  input  logic [AW-1:0]          r_addr,
  input  logic signed [31:0]     exp_in,   // expected data for this read (computed in TB)

  input  logic                   w_en,
  input  logic                   w_we,
  input  logic [AW-1:0]          w_addr,
  input  logic signed [DW-1:0]   w_d,

  output int                     reads,
  output int                     writes
);

  // 5-deep pipe [0..4]
  logic [31:0]        rd_addr_pipe0, rd_addr_pipe1, rd_addr_pipe2, rd_addr_pipe3, rd_addr_pipe4;
  logic signed [31:0] exp_pipe0,     exp_pipe1,     exp_pipe2,     exp_pipe3,     exp_pipe4;
  logic               v0,            v1,            v2,            v3,            v4;

  int done_cycle_count;

  task automatic clear_all();
    reads = 0;
    writes = 0;

    rd_addr_pipe0 = '0; rd_addr_pipe1 = '0; rd_addr_pipe2 = '0; rd_addr_pipe3 = '0; rd_addr_pipe4 = '0;
    exp_pipe0     = '0; exp_pipe1     = '0; exp_pipe2     = '0; exp_pipe3     = '0; exp_pipe4     = '0;
    v0            = 1'b0; v1          = 1'b0; v2          = 1'b0; v3          = 1'b0; v4          = 1'b0;

    done_cycle_count = 0;
  endtask

  initial begin
    clear_all();
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      clear_all();
    end else begin
      if (start) begin
        clear_all();
      end

      // shift (unrolled)
      rd_addr_pipe4 <= rd_addr_pipe3;
      rd_addr_pipe3 <= rd_addr_pipe2;
      rd_addr_pipe2 <= rd_addr_pipe1;
      rd_addr_pipe1 <= rd_addr_pipe0;

      exp_pipe4     <= exp_pipe3;
      exp_pipe3     <= exp_pipe2;
      exp_pipe2     <= exp_pipe1;
      exp_pipe1     <= exp_pipe0;

      v4            <= v3;
      v3            <= v2;
      v2            <= v1;
      v1            <= v0;
      v0            <= 1'b0;

      // read observe
      if (r_en) begin
        if (r_addr !== (reads[AW-1:0])) begin
          $error("[%0t][%s] BAD_READ_ADDR got=%0d exp=%0d", $time, TAG, r_addr, reads);
        end
        rd_addr_pipe0 <= r_addr;
        exp_pipe0     <= exp_in;
        v0            <= 1'b1;
        reads         <= reads + 1;
      end

      // write observe: match the read from 4 cycles earlier (pre-shift stage 3)
      if (w_en && w_we) begin
        if (!v3) begin
          $error("[%0t][%s] WRITE_WITHOUT_PRIOR_READ addr=%0d", $time, TAG, w_addr);
        end else begin
          if (w_addr !== rd_addr_pipe3[AW-1:0]) begin
            $error("[%0t][%s] BAD_WRITE_ADDR got=%0d exp=%0d",
                  $time, TAG, w_addr, rd_addr_pipe3[AW-1:0]);
          end
          if (w_d !== exp_pipe3[DW-1:0]) begin
            $error("[%0t][%s] BAD_WRITE_DATA @%0d got=%0d exp=%0d",
                  $time, TAG, w_addr, w_d, exp_pipe3[DW-1:0]);
          end
        end
        writes <= writes + 1;
      end

      // done must be 1-cycle pulse
      if (done) begin
        done_cycle_count <= done_cycle_count + 1;
        if (done_cycle_count > 0) $error("[%0t][%s] done not 1-cycle", $time, TAG);
      end else begin
        done_cycle_count <= 0;
      end
    end
  end

endmodule


// ============================================================
// "Big" ReLU testbench (Verilator-friendly)
// ============================================================
module tb_relu;

  // ------------------------ clock/reset ------------------------
  logic clk=1'b0, reset=1'b1;
  always #5 clk = ~clk;

  initial begin
    $dumpfile("wave_relu.vcd");
    $dumpvars(0, tb_relu);
    repeat (4) @(posedge clk);
    reset = 1'b0;
  end

  // golden helper (32-bit domain)
  function automatic logic signed [31:0] relu32(input logic signed [31:0] x);
    return (x < 0) ? 0 : x;
  endfunction

  // ============================================================
  // Instance 0: DW=16, C=1, SZ=8  => N=64
  // ============================================================
  localparam int DW0=16, C0=1, SZ0=8;
  localparam int N0 = C0*SZ0*SZ0;
  localparam int AW0 = (N0<=1)?1:$clog2(N0);
  typedef logic [AW0-1:0]         addr0_t;
  typedef logic signed [DW0-1:0]  data0_t;

  logic start0, done0;
  addr0_t r_addr0, w_addr0;
  logic r_en0, w_en0, w_we0;
  data0_t r_q0, w_d0;

  data0_t init0[0:N0-1];
  data0_t gold0[0:N0-1];

  tb_dualport_bram #(.DW(DW0), .DEPTH(N0), .LAT(1), .AW(AW0)) bram0 (
    .clk,
    .r_en(r_en0), .r_addr(r_addr0), .r_q(r_q0),
    .w_en(w_en0), .w_we(w_we0), .w_addr(w_addr0), .w_d(w_d0)
  );

  relu #(.DATA_WIDTH(DW0), .CHANNELS(C0), .IMG_SIZE(SZ0)) dut0 (
    .clk, .reset, .start(start0),
    .conv_r_addr(r_addr0), .conv_r_en(r_en0), .conv_r_q(r_q0),
    .conv_w_addr(w_addr0), .conv_w_en(w_en0), .conv_w_we(w_we0), .conv_w_d(w_d0),
    .done(done0)
  );

  logic signed [31:0] exp0;
  always_comb exp0 = relu32($signed(bram0.mem[r_addr0]));

  int reads0, writes0;
  relu_monitor #(.AW(AW0), .DW(DW0), .N(N0), .TAG("DUT0")) mon0 (
    .clk, .reset,
    .start(start0), .done(done0),
    .r_en(r_en0), .r_addr(r_addr0), .exp_in(exp0),
    .w_en(w_en0), .w_we(w_we0), .w_addr(w_addr0), .w_d(w_d0),
    .reads(reads0), .writes(writes0)
  );

  // ============================================================
  // Instance 1: DW=16, C=2, SZ=6 => N=72
  // ============================================================
  localparam int DW1=16, C1=2, SZ1=6;
  localparam int N1 = C1*SZ1*SZ1;
  localparam int AW1 = (N1<=1)?1:$clog2(N1);
  typedef logic [AW1-1:0]         addr1_t;
  typedef logic signed [DW1-1:0]  data1_t;

  logic start1, done1;
  addr1_t r_addr1, w_addr1;
  logic r_en1, w_en1, w_we1;
  data1_t r_q1, w_d1;

  data1_t init1[0:N1-1];
  data1_t gold1[0:N1-1];

  tb_dualport_bram #(.DW(DW1), .DEPTH(N1), .LAT(1), .AW(AW1)) bram1 (
    .clk,
    .r_en(r_en1), .r_addr(r_addr1), .r_q(r_q1),
    .w_en(w_en1), .w_we(w_we1), .w_addr(w_addr1), .w_d(w_d1)
  );

  relu #(.DATA_WIDTH(DW1), .CHANNELS(C1), .IMG_SIZE(SZ1)) dut1 (
    .clk, .reset, .start(start1),
    .conv_r_addr(r_addr1), .conv_r_en(r_en1), .conv_r_q(r_q1),
    .conv_w_addr(w_addr1), .conv_w_en(w_en1), .conv_w_we(w_we1), .conv_w_d(w_d1),
    .done(done1)
  );

  logic signed [31:0] exp1;
  always_comb exp1 = relu32($signed(bram1.mem[r_addr1]));

  int reads1, writes1;
  relu_monitor #(.AW(AW1), .DW(DW1), .N(N1), .TAG("DUT1")) mon1 (
    .clk, .reset,
    .start(start1), .done(done1),
    .r_en(r_en1), .r_addr(r_addr1), .exp_in(exp1),
    .w_en(w_en1), .w_we(w_we1), .w_addr(w_addr1), .w_d(w_d1),
    .reads(reads1), .writes(writes1)
  );

  // ============================================================
  // Instance 2: DW=12, C=3, SZ=4 => N=48
  // ============================================================
  localparam int DW2=12, C2=3, SZ2=4;
  localparam int N2 = C2*SZ2*SZ2;
  localparam int AW2 = (N2<=1)?1:$clog2(N2);
  typedef logic [AW2-1:0]         addr2_t;
  typedef logic signed [DW2-1:0]  data2_t;

  logic start2, done2;
  addr2_t r_addr2, w_addr2;
  logic r_en2, w_en2, w_we2;
  data2_t r_q2, w_d2;

  data2_t init2[0:N2-1];
  data2_t gold2[0:N2-1];

  tb_dualport_bram #(.DW(DW2), .DEPTH(N2), .LAT(1), .AW(AW2)) bram2 (
    .clk,
    .r_en(r_en2), .r_addr(r_addr2), .r_q(r_q2),
    .w_en(w_en2), .w_we(w_we2), .w_addr(w_addr2), .w_d(w_d2)
  );

  relu #(.DATA_WIDTH(DW2), .CHANNELS(C2), .IMG_SIZE(SZ2)) dut2 (
    .clk, .reset, .start(start2),
    .conv_r_addr(r_addr2), .conv_r_en(r_en2), .conv_r_q(r_q2),
    .conv_w_addr(w_addr2), .conv_w_en(w_en2), .conv_w_we(w_we2), .conv_w_d(w_d2),
    .done(done2)
  );

  logic signed [31:0] exp2;
  always_comb exp2 = relu32($signed(bram2.mem[r_addr2]));

  int reads2, writes2;
  relu_monitor #(.AW(AW2), .DW(DW2), .N(N2), .TAG("DUT2")) mon2 (
    .clk, .reset,
    .start(start2), .done(done2),
    .r_en(r_en2), .r_addr(r_addr2), .exp_in(exp2),
    .w_en(w_en2), .w_we(w_we2), .w_addr(w_addr2), .w_d(w_d2),
    .reads(reads2), .writes(writes2)
  );

  // ============================================================
  // Initialisation patterns
  // ============================================================
  task automatic init_pattern_seq_mix_0();
    for (int i=0;i<N0;i++) begin
      if ((i % 7) == 0)      init0[i] = data0_t'(-$signed(i));
      else if ((i % 7) == 1) init0[i] = data0_t'(0);
      else                   init0[i] = data0_t'($signed(i + 3));
    end
  endtask

  task automatic init_pattern_checker_1();
    int ch, idx, r, c;
    bit chk;
    for (int i=0;i<N1;i++) begin
      ch  = i / (SZ1*SZ1);
      idx = i % (SZ1*SZ1);
      r   = idx / SZ1;
      c   = idx % SZ1;
      chk = bit'(((r ^ c) & 1) != 0);
      if (ch==0) init1[i] = chk ? data1_t'(-1) : data1_t'(2);
      else       init1[i] = chk ? data1_t'(-16'sd32768) : data1_t'(16'sd32767);
    end
  endtask

  task automatic init_pattern_prng_2(int seed);
    int s, v;
    s = seed;
    for (int i=0;i<N2;i++) begin
      s ^= (s << 13);
      s ^= (s >> 17);
      s ^= (s << 5);
      v = (s % 257) - 128;
      init2[i] = data2_t'(v);
    end
  endtask

  task automatic load_all_mems();
    for (int i=0;i<N0;i++) bram0.mem[i] = init0[i];
    for (int i=0;i<N1;i++) bram1.mem[i] = init1[i];
    for (int i=0;i<N2;i++) bram2.mem[i] = init2[i];
  endtask

  task automatic compute_all_golden();
    for (int i=0;i<N0;i++) gold0[i] = data0_t'(relu32($signed(init0[i])));
    for (int i=0;i<N1;i++) gold1[i] = data1_t'(relu32($signed(init1[i])));
    for (int i=0;i<N2;i++) gold2[i] = data2_t'(relu32($signed(init2[i])));
  endtask

  // ============================================================
  // Runner
  // ============================================================
  task automatic run_one(
    input string name,
    input int which,
    input int guard_max
  );
    int guard;
    int errs;

    if (which==0) begin start0=1; @(posedge clk); start0=0; end
    else if (which==1) begin start1=1; @(posedge clk); start1=0; end
    else begin start2=1; @(posedge clk); start2=0; end

    guard = 0;
    while (guard < guard_max) begin
      @(posedge clk);
      guard++;
      if ((which==0 && done0) || (which==1 && done1) || (which==2 && done2)) break;
    end
    if (guard >= guard_max) $fatal(1,"[%s] TIMEOUT waiting done", name);

    @(posedge clk);
    if ((which==0 && done0) || (which==1 && done1) || (which==2 && done2))
      $error("[%s] done is not a 1-cycle pulse (stayed high)", name);

    errs = 0;
    if (which==0) begin
      if (reads0  != N0) begin $display("[%s] BAD_READ_COUNT got=%0d exp=%0d",  name, reads0,  N0); errs++; end
      if (writes0 != N0) begin $display("[%s] BAD_WRITE_COUNT got=%0d exp=%0d", name, writes0, N0); errs++; end
      for (int i=0;i<N0;i++) if (bram0.mem[i] !== gold0[i]) begin
        $display("[%s] MISMATCH mem[%0d] got=%0d exp=%0d (init=%0d)", name, i, bram0.mem[i], gold0[i], init0[i]);
        errs++;
      end
    end else if (which==1) begin
      if (reads1  != N1) begin $display("[%s] BAD_READ_COUNT got=%0d exp=%0d",  name, reads1,  N1); errs++; end
      if (writes1 != N1) begin $display("[%s] BAD_WRITE_COUNT got=%0d exp=%0d", name, writes1, N1); errs++; end
      for (int i=0;i<N1;i++) if (bram1.mem[i] !== gold1[i]) begin
        $display("[%s] MISMATCH mem[%0d] got=%0d exp=%0d (init=%0d)", name, i, bram1.mem[i], gold1[i], init1[i]);
        errs++;
      end
    end else begin
      if (reads2  != N2) begin $display("[%s] BAD_READ_COUNT got=%0d exp=%0d",  name, reads2,  N2); errs++; end
      if (writes2 != N2) begin $display("[%s] BAD_WRITE_COUNT got=%0d exp=%0d", name, writes2, N2); errs++; end
      for (int i=0;i<N2;i++) if (bram2.mem[i] !== gold2[i]) begin
        $display("[%s] MISMATCH mem[%0d] got=%0d exp=%0d (init=%0d)", name, i, bram2.mem[i], gold2[i], init2[i]);
        errs++;
      end
    end

    if (errs==0) $display("[%s] PASS", name);
    else         $fatal(1,"[%s] FAIL errs=%0d", name, errs);
  endtask

  // ============================================================
  // Main
  // ============================================================
  initial begin
    start0 = 0; start1 = 0; start2 = 0;

    @(negedge reset);
    repeat (2) @(posedge clk);

    // CASE A
    init_pattern_seq_mix_0();
    init_pattern_checker_1();
    init_pattern_prng_2(32'h00C0_FFEE);
    compute_all_golden();
    load_all_mems();

    run_one("RELU_CASE_A_DW16_C1_SZ8", 0, 200000);
    run_one("RELU_CASE_A_DW16_C2_SZ6", 1, 200000);
    run_one("RELU_CASE_A_DW12_C3_SZ4", 2, 200000);

    // CASE B
    for (int i=0;i<N0;i++) init0[i] = data0_t'((i[0]) ? -$signed(i+5) : $signed(i));
    for (int i=0;i<N1;i++) begin
      int ch, idx;
      ch  = i / (SZ1*SZ1);
      idx = i % (SZ1*SZ1);
      init1[i] = data1_t'($signed(idx) - 10 + (ch ? 3 : -3));
    end
    init_pattern_prng_2(32'h1234_5678);

    compute_all_golden();
    load_all_mems();

    run_one("RELU_CASE_B_ALTNEG_DW16_C1_SZ8", 0, 200000);
    run_one("RELU_CASE_B_CROSSZERO_DW16_C2_SZ6", 1, 200000);
    run_one("RELU_CASE_B_PRNG2_DW12_C3_SZ4", 2, 200000);

    $display("ALL RELU TESTS PASSED");
    $finish;
  end

endmodule

`default_nettype wire
