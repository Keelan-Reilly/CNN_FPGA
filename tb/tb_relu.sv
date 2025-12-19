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

  function automatic logic signed [31:0] relu32(input logic signed [31:0] x);
    return (x < 0) ? 0 : x;
  endfunction

  // ============================================================
  // Per-instance monitor state
  //
  // DUT timing (with Verilator-safe extra wait):
  //   read strobe @ t
  //   capture conv_r_q @ t+3
  //   write @ t+4
  // => read->write latency = 4 cycles => check stage [4]
  // ============================================================
  typedef struct {
    int  reads;
    int  writes;
    bit  started;

    logic [31:0]        rd_addr_pipe [0:4];
    logic signed [31:0] exp_d_pipe   [0:4];
    bit                v_pipe       [0:4];

    bit  done_seen;
    int  done_cycle_count;
  } mon_t;

  task automatic mon_reset(ref mon_t m);
    m.reads            = 0;
    m.writes           = 0;
    m.started          = 0;
    m.done_seen        = 0;
    m.done_cycle_count = 0;
    for (int i=0;i<5;i++) begin
      m.rd_addr_pipe[i] = '0;
      m.exp_d_pipe[i]   = '0;
      m.v_pipe[i]       = 1'b0;
    end
  endtask

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
  // Monitors
  // ============================================================
  mon_t m0, m1, m2;

  task automatic mon_shift5(ref mon_t m);
    for (int k=4;k>0;k--) begin
      m.rd_addr_pipe[k] <= m.rd_addr_pipe[k-1];
      m.exp_d_pipe[k]   <= m.exp_d_pipe[k-1];
      m.v_pipe[k]       <= m.v_pipe[k-1];
    end
    m.v_pipe[0] <= 1'b0;
  endtask

  task automatic mon_step_common(
    ref mon_t m,
    input bit start,
    input bit done,
    input bit r_en,
    input int unsigned r_addr_u,
    input int signed  exp_from_mem,
    input bit w_en,
    input bit w_we,
    input int unsigned w_addr_u,
    input int signed  w_d_s,
    input string tag
  );
    if (reset) begin
      mon_reset(m);
    end else begin
      if (start) begin
        mon_reset(m);
        m.started = 1;
      end

      mon_shift5(m);

      if (r_en) begin
        if (r_addr_u !== m.reads) begin
          $error("[%0t][%s] BAD_READ_ADDR got=%0d exp=%0d", $time, tag, r_addr_u, m.reads);
        end
        m.rd_addr_pipe[0] <= r_addr_u;
        m.exp_d_pipe[0]   <= exp_from_mem;
        m.v_pipe[0]       <= 1'b1;
        m.reads++;
      end

      // WRITE must match read from 4 cycles earlier => stage [4]
      if (w_en && w_we) begin
        if (!m.v_pipe[4]) begin
          $error("[%0t][%s] WRITE_WITHOUT_PRIOR_READ addr=%0d", $time, tag, w_addr_u);
        end else begin
          if (w_addr_u !== m.rd_addr_pipe[4]) begin
            $error("[%0t][%s] BAD_WRITE_ADDR got=%0d exp=%0d", $time, tag, w_addr_u, m.rd_addr_pipe[4]);
          end
          if (w_d_s !== $signed(m.exp_d_pipe[4])) begin
            $error("[%0t][%s] BAD_WRITE_DATA @%0d got=%0d exp=%0d",
                   $time, tag, w_addr_u, w_d_s, m.exp_d_pipe[4]);
          end
        end
        m.writes++;
      end

      if (done) begin
        m.done_seen = 1;
        m.done_cycle_count++;
        if (m.done_cycle_count > 1) $error("[%0t][%s] done not 1-cycle", $time, tag);
      end
      if (!done) m.done_cycle_count = 0;
    end
  endtask

  task automatic mon_step_0();
    mon_step_common(
      m0,
      start0, done0,
      r_en0,  int'(r_addr0),
      relu32($signed(bram0.mem[r_addr0])),
      w_en0, w_we0,
      int'(w_addr0),
      int'($signed(w_d0)),
      "DUT0"
    );
  endtask

  task automatic mon_step_1();
    mon_step_common(
      m1,
      start1, done1,
      r_en1,  int'(r_addr1),
      relu32($signed(bram1.mem[r_addr1])),
      w_en1, w_we1,
      int'(w_addr1),
      int'($signed(w_d1)),
      "DUT1"
    );
  endtask

  task automatic mon_step_2();
    mon_step_common(
      m2,
      start2, done2,
      r_en2,  int'(r_addr2),
      relu32($signed(bram2.mem[r_addr2])),
      w_en2, w_we2,
      int'(w_addr2),
      int'($signed(w_d2)),
      "DUT2"
    );
  endtask

  always_ff @(posedge clk) begin
    mon_step_0();
    mon_step_1();
    mon_step_2();
  end

  // ============================================================
  // Runner (unchanged)
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
      if (m0.reads  != N0) begin $display("[%s] BAD_READ_COUNT got=%0d exp=%0d",  name, m0.reads,  N0); errs++; end
      if (m0.writes != N0) begin $display("[%s] BAD_WRITE_COUNT got=%0d exp=%0d", name, m0.writes, N0); errs++; end
      for (int i=0;i<N0;i++) if (bram0.mem[i] !== gold0[i]) begin
        $display("[%s] MISMATCH mem[%0d] got=%0d exp=%0d (init=%0d)", name, i, bram0.mem[i], gold0[i], init0[i]);
        errs++;
      end
    end else if (which==1) begin
      if (m1.reads  != N1) begin $display("[%s] BAD_READ_COUNT got=%0d exp=%0d",  name, m1.reads,  N1); errs++; end
      if (m1.writes != N1) begin $display("[%s] BAD_WRITE_COUNT got=%0d exp=%0d", name, m1.writes, N1); errs++; end
      for (int i=0;i<N1;i++) if (bram1.mem[i] !== gold1[i]) begin
        $display("[%s] MISMATCH mem[%0d] got=%0d exp=%0d (init=%0d)", name, i, bram1.mem[i], gold1[i], init1[i]);
        errs++;
      end
    end else begin
      if (m2.reads  != N2) begin $display("[%s] BAD_READ_COUNT got=%0d exp=%0d",  name, m2.reads,  N2); errs++; end
      if (m2.writes != N2) begin $display("[%s] BAD_WRITE_COUNT got=%0d exp=%0d", name, m2.writes, N2); errs++; end
      for (int i=0;i<N2;i++) if (bram2.mem[i] !== gold2[i]) begin
        $display("[%s] MISMATCH mem[%0d] got=%0d exp=%0d (init=%0d)", name, i, bram2.mem[i], gold2[i], init2[i]);
        errs++;
      end
    end

    if (errs==0) $display("[%s] PASS", name);
    else         $fatal(1,"[%s] FAIL errs=%0d", name, errs);
  endtask

  // ============================================================
  // Main (unchanged)
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
