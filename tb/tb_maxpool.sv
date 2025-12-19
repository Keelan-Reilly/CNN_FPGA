`timescale 1ns/1ps
`default_nettype none

// ============================================================
// BRAM-like featuremap source with configurable edge-to-edge latency
//   - When en is pulsed, address is sampled.
//   - After LAT cycles, q updates to mem[addr].
//   - When no valid response, q holds last value.
// ============================================================
module tb_fmap_bram #(
  parameter int DW    = 16,
  parameter int DEPTH = 16,
  parameter int LAT   = 1,
  parameter int AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
  input  logic clk,
  input  logic en,
  input  logic [AW-1:0] addr,
  input  logic signed [DW-1:0] mem [0:DEPTH-1],
  output logic signed [DW-1:0] q
);
  localparam int P = (LAT < 1) ? 1 : LAT;

  logic [AW-1:0] addr_pipe [0:P];
  logic          v_pipe    [0:P];

  integer k;

  initial begin
    q = '0;
    for (k=0; k<=$high(addr_pipe); k++) begin
      addr_pipe[k] = '0;
      v_pipe[k]    = 1'b0;
    end
  end

  always_ff @(posedge clk) begin
    addr_pipe[0] <= addr;
    v_pipe[0]    <= en;

    for (k=1; k<=$high(addr_pipe); k++) begin
      addr_pipe[k] <= addr_pipe[k-1];
      v_pipe[k]    <= v_pipe[k-1];
    end

    if (LAT == 0) begin
      if (en) q <= mem[addr];
    end else begin
      if (v_pipe[LAT]) q <= mem[addr_pipe[LAT]];
    end
  end
endmodule


// ============================================================
// Maxpool "contract + golden" testbench
//   - Multiple DUT instances with different BRAM_LAT and data patterns
//   - Checks:
//       * handshake (start/done pulse)
//       * conv read contract (en pulses + address sequence)
//       * golden maxpool output match
//       * write contract (pool_en/pool_we count + address sequence)
// ============================================================
module tb_maxpool;

  // ------------------------ clock/reset ------------------------
  logic clk=1'b0, reset=1'b1;
  always #5 clk = ~clk;

  initial begin
    $dumpfile("wave_maxpool.vcd");
    $dumpvars(0, tb_maxpool);
    repeat (4) @(posedge clk);
    reset = 1'b0;
  end

  // ------------------------ params ------------------------
  localparam int DW      = 16;
  localparam int POOL    = 2;

  localparam int C       = 2;
  localparam int IN_SIZE = 4;
  localparam int OUT_SIZE= IN_SIZE/POOL;

  localparam int CONV_SZ = C*IN_SIZE*IN_SIZE;
  localparam int POOL_SZ = C*OUT_SIZE*OUT_SIZE;

  localparam int CONV_AW = (CONV_SZ<=1)?1:$clog2(CONV_SZ);
  localparam int POOL_AW = (POOL_SZ<=1)?1:$clog2(POOL_SZ);

  typedef logic [CONV_AW-1:0] conv_addr_t;
  typedef logic [POOL_AW-1:0] pool_addr_t;

  // CHW linear index helper (matches your module doc: CHW-linear)
  function automatic int lin_chw(input int ch, input int r, input int c);
    return (ch*IN_SIZE + r)*IN_SIZE + c;
  endfunction

  function automatic int lin_pool(input int ch, input int r, input int c);
    return (ch*OUT_SIZE + r)*OUT_SIZE + c;
  endfunction

  // ============================================================
  // Case input featuremaps
  // ============================================================
  logic signed [DW-1:0] conv_mem0 [0:CONV_SZ-1];
  logic signed [DW-1:0] conv_mem1 [0:CONV_SZ-1];
  logic signed [DW-1:0] conv_mem2 [0:CONV_SZ-1];

  initial begin
    // CASE0: channel0 = 0..15, channel1 = 100..115
    for (int r=0; r<IN_SIZE; r++) begin
      for (int c=0; c<IN_SIZE; c++) begin
        conv_mem0[lin_chw(0,r,c)] = (r*IN_SIZE + c);
        conv_mem0[lin_chw(1,r,c)] = 100 + (r*IN_SIZE + c);
      end
    end

    // CASE1: mixed negatives to catch signed compare mistakes
    // channel0: [-1,-2,-3,-4; ...], channel1: some alternating
    for (int r=0; r<IN_SIZE; r++) begin
      for (int c=0; c<IN_SIZE; c++) begin
        conv_mem1[lin_chw(0,r,c)] = - (r*IN_SIZE + c + 1);
        conv_mem1[lin_chw(1,r,c)] = ( (r+c) & 1 ) ? 16'sd7 : -16'sd8;
      end
    end

    // CASE2: strictly decreasing per 2x2 window to validate address order
    // channel0: 1000 - idx, channel1: 2000 - 2*idx
    for (int r=0; r<IN_SIZE; r++) begin
      for (int c=0; c<IN_SIZE; c++) begin
        int idx = r*IN_SIZE + c;
        conv_mem2[lin_chw(0,r,c)] = 16'sd1000 - idx;
        conv_mem2[lin_chw(1,r,c)] = 16'sd2000 - 2*idx;
      end
    end
  end

  // ============================================================
  // Golden compute
  // ============================================================
  task automatic compute_golden(
    input  logic signed [DW-1:0] in  [0:CONV_SZ-1],
    output logic signed [DW-1:0] out [0:POOL_SZ-1]
  );
    for (int ch=0; ch<C; ch++) begin
      for (int r=0; r<OUT_SIZE; r++) begin
        for (int c=0; c<OUT_SIZE; c++) begin
          logic signed [DW-1:0] x0, x1, x2, x3, m0, m1;
          x0 = in[lin_chw(ch, 2*r,   2*c  )];
          x1 = in[lin_chw(ch, 2*r,   2*c+1)];
          x2 = in[lin_chw(ch, 2*r+1, 2*c  )];
          x3 = in[lin_chw(ch, 2*r+1, 2*c+1)];
          m0 = (x0 > x1) ? x0 : x1;
          m1 = (x2 > x3) ? x2 : x3;
          out[lin_pool(ch,r,c)] = (m0 > m1) ? m0 : m1;
        end
      end
    end
  endtask

  task automatic dump_pool(input string tag, input logic signed [DW-1:0] v [0:POOL_SZ-1]);
    $display("\n--- %s ---", tag);
    for (int ch=0; ch<C; ch++) begin
      $display("ch=%0d", ch);
      for (int r=0; r<OUT_SIZE; r++) begin
        for (int c=0; c<OUT_SIZE; c++) begin
          $write(" %0d", v[lin_pool(ch,r,c)]);
        end
        $write("\n");
      end
    end
    $display("--- end %s ---\n", tag);
  endtask

  // ============================================================
  // Instantiate 3 DUTs with different BRAM_LAT
  // ============================================================
  localparam int BRAM_LAT0 = 1;
  localparam int BRAM_LAT1 = 2;
  localparam int BRAM_LAT2 = 0;

  // ---- DUT0 signals ----
  conv_addr_t conv_addr0; logic conv_en0; logic signed [DW-1:0] conv_q0;
  pool_addr_t pool_addr0; logic pool_en0, pool_we0; logic signed [DW-1:0] pool_d0;
  logic start0, done0;

  // ---- DUT1 signals ----
  conv_addr_t conv_addr1; logic conv_en1; logic signed [DW-1:0] conv_q1;
  pool_addr_t pool_addr1; logic pool_en1, pool_we1; logic signed [DW-1:0] pool_d1;
  logic start1, done1;

  // ---- DUT2 signals ----
  conv_addr_t conv_addr2; logic conv_en2; logic signed [DW-1:0] conv_q2;
  pool_addr_t pool_addr2; logic pool_en2, pool_we2; logic signed [DW-1:0] pool_d2;
  logic start2, done2;

  tb_fmap_bram #(.DW(DW), .DEPTH(CONV_SZ), .LAT(BRAM_LAT0), .AW(CONV_AW))
    bram0 (.clk, .en(conv_en0), .addr(conv_addr0), .mem(conv_mem0), .q(conv_q0));

  tb_fmap_bram #(.DW(DW), .DEPTH(CONV_SZ), .LAT(BRAM_LAT1), .AW(CONV_AW))
    bram1 (.clk, .en(conv_en1), .addr(conv_addr1), .mem(conv_mem1), .q(conv_q1));

  tb_fmap_bram #(.DW(DW), .DEPTH(CONV_SZ), .LAT(BRAM_LAT2), .AW(CONV_AW))
    bram2 (.clk, .en(conv_en2), .addr(conv_addr2), .mem(conv_mem2), .q(conv_q2));

  // Output memories (captured writes)
  logic signed [DW-1:0] pool_mem0 [0:POOL_SZ-1];
  logic signed [DW-1:0] pool_mem1 [0:POOL_SZ-1];
  logic signed [DW-1:0] pool_mem2 [0:POOL_SZ-1];

  always_ff @(posedge clk) begin
    if (pool_en0 && pool_we0) pool_mem0[pool_addr0] <= pool_d0;
    if (pool_en1 && pool_we1) pool_mem1[pool_addr1] <= pool_d1;
    if (pool_en2 && pool_we2) pool_mem2[pool_addr2] <= pool_d2;
  end

  maxpool #(.DATA_WIDTH(DW), .CHANNELS(C), .IN_SIZE(IN_SIZE), .POOL(POOL)) dut0 (
    .clk, .reset, .start(start0),
    .conv_addr(conv_addr0), .conv_en(conv_en0), .conv_q(conv_q0),
    .pool_addr(pool_addr0), .pool_en(pool_en0), .pool_we(pool_we0), .pool_d(pool_d0),
    .done(done0)
  );

  maxpool #(.DATA_WIDTH(DW), .CHANNELS(C), .IN_SIZE(IN_SIZE), .POOL(POOL)) dut1 (
    .clk, .reset, .start(start1),
    .conv_addr(conv_addr1), .conv_en(conv_en1), .conv_q(conv_q1),
    .pool_addr(pool_addr1), .pool_en(pool_en1), .pool_we(pool_we1), .pool_d(pool_d1),
    .done(done1)
  );

  maxpool #(.DATA_WIDTH(DW), .CHANNELS(C), .IN_SIZE(IN_SIZE), .POOL(POOL)) dut2 (
    .clk, .reset, .start(start2),
    .conv_addr(conv_addr2), .conv_en(conv_en2), .conv_q(conv_q2),
    .pool_addr(pool_addr2), .pool_en(pool_en2), .pool_we(pool_we2), .pool_d(pool_d2),
    .done(done2)
  );

  // ============================================================
  // Monitors: conv read + pool write contracts
  // ============================================================
  typedef struct packed {
    int conv_reads;
    int pool_writes;
    int exp_conv_addr;
    int exp_pool_addr;
    bit started;
  } mon_t;

  mon_t m0, m1, m2;

  task automatic mon_reset(ref mon_t m);
    m.conv_reads    = 0;
    m.pool_writes   = 0;
    m.exp_conv_addr = 0;
    m.exp_pool_addr = 0;
    m.started       = 0;
  endtask

  // Expected conv address sequence emitted by THIS RTL:
  // For each channel:
  //   windows in pooled order (r=0..OUT-1, c=0..OUT-1)
  //   for each window: base, base+1, base+IN, base+IN+1
  function automatic int base_for_window(input int ch, input int pr, input int pc);
    // CHW: base = ch*IN*IN + (2*pr)*IN + (2*pc)
    return ch*(IN_SIZE*IN_SIZE) + (2*pr)*IN_SIZE + (2*pc);
  endfunction

  task automatic mon_step(
    input string tag,
    ref mon_t m,
    input logic rst,
    input logic start,
    input logic conv_en,
    input conv_addr_t conv_addr,
    input logic pool_en,
    input logic pool_we,
    input pool_addr_t pool_addr
  );
    if (rst) begin
      mon_reset(m);
    end else begin
      if (start) begin
        mon_reset(m);
        m.started = 1;
      end

      if (conv_en) begin
        int win;
        int phase;
        int ch_i, pr, pc;
        int b;
        int exp;

        win   = m.conv_reads / 4;
        phase = m.conv_reads % 4;

        ch_i = win / (OUT_SIZE*OUT_SIZE);
        pr   = (win % (OUT_SIZE*OUT_SIZE)) / OUT_SIZE;
        pc   = (win % (OUT_SIZE*OUT_SIZE)) % OUT_SIZE;

        b = base_for_window(ch_i, pr, pc);

        case (phase)
          0: exp = b;
          1: exp = b + 1;
          2: exp = b + IN_SIZE;
          3: exp = b + IN_SIZE + 1;
          default: exp = b;
        endcase

        if (conv_addr !== conv_addr_t'(exp)) begin
          $error("[%0t][%s] BAD_CONV_ADDR: got=%0d exp=%0d (read#=%0d win=%0d phase=%0d ch=%0d pr=%0d pc=%0d)",
                $time, tag, conv_addr, exp, m.conv_reads, win, phase, ch_i, pr, pc);
        end

        m.conv_reads++;
      end

      if (pool_en && pool_we) begin
        if (pool_addr !== pool_addr_t'(m.pool_writes)) begin
          $error("[%0t][%s] BAD_POOL_ADDR: got=%0d exp=%0d (write#=%0d)",
                 $time, tag, pool_addr, m.pool_writes, m.pool_writes);
        end
        m.pool_writes++;
      end
    end
  endtask

  always_ff @(posedge clk) begin
    mon_step("DUT0", m0, reset, start0, conv_en0, conv_addr0, pool_en0, pool_we0, pool_addr0);
    mon_step("DUT1", m1, reset, start1, conv_en1, conv_addr1, pool_en1, pool_we1, pool_addr1);
    mon_step("DUT2", m2, reset, start2, conv_en2, conv_addr2, pool_en2, pool_we2, pool_addr2);
  end

  // ============================================================
  // Run case: start, wait done, compare vs golden and contracts
  // ============================================================
  task automatic run_case(
    input string name,
    input int which,           // 0/1/2
    input int guard_max
  );
    int guard, errs;
    int exp_reads, exp_writes;
    logic signed [DW-1:0] gold [0:POOL_SZ-1];
    logic signed [DW-1:0] got  [0:POOL_SZ-1];

    if (which==0) compute_golden(conv_mem0, gold);
    else if (which==1) compute_golden(conv_mem1, gold);
    else compute_golden(conv_mem2, gold);

    // start pulse
    if (which==0) begin start0<=1; @(posedge clk); start0<=0; end
    else if (which==1) begin start1<=1; @(posedge clk); start1<=0; end
    else begin start2<=1; @(posedge clk); start2<=0; end

    // wait done
    guard = 0;
    while (guard < guard_max) begin
      @(posedge clk);
      guard++;
      if ((which==0 && done0) || (which==1 && done1) || (which==2 && done2)) break;
    end
    if (guard >= guard_max) $fatal(1,"[%s] TIMEOUT waiting done", name);

    // done should be 1-cycle pulse
    @(posedge clk);
    if ((which==0 && done0) || (which==1 && done1) || (which==2 && done2))
      $error("[%s] done is not a 1-cycle pulse", name);

    // snapshot outputs
    for (int i=0;i<POOL_SZ;i++) begin
      if (which==0) got[i] = pool_mem0[i];
      else if (which==1) got[i] = pool_mem1[i];
      else got[i] = pool_mem2[i];
    end

    // compare
    errs = 0;
    for (int i=0;i<POOL_SZ;i++) begin
      if (got[i] !== gold[i]) begin
        $display("[%s] MISMATCH idx=%0d got=%0d exp=%0d", name, i, got[i], gold[i]);
        errs++;
      end
    end

    // contract counts
    exp_reads  = POOL_SZ * 4;
    exp_writes = POOL_SZ;

    if (which==0) begin
      if (m0.conv_reads  != exp_reads)  begin $display("[%s] BAD_READ_COUNT got=%0d exp=%0d",  name, m0.conv_reads,  exp_reads);  errs++; end
      if (m0.pool_writes != exp_writes) begin $display("[%s] BAD_WRITE_COUNT got=%0d exp=%0d", name, m0.pool_writes, exp_writes); errs++; end
    end else if (which==1) begin
      if (m1.conv_reads  != exp_reads)  begin $display("[%s] BAD_READ_COUNT got=%0d exp=%0d",  name, m1.conv_reads,  exp_reads);  errs++; end
      if (m1.pool_writes != exp_writes) begin $display("[%s] BAD_WRITE_COUNT got=%0d exp=%0d", name, m1.pool_writes, exp_writes); errs++; end
    end else begin
      if (m2.conv_reads  != exp_reads)  begin $display("[%s] BAD_READ_COUNT got=%0d exp=%0d",  name, m2.conv_reads,  exp_reads);  errs++; end
      if (m2.pool_writes != exp_writes) begin $display("[%s] BAD_WRITE_COUNT got=%0d exp=%0d", name, m2.pool_writes, exp_writes); errs++; end
    end

    if (errs==0) begin
      $display("[%s] PASS (reads=%0d writes=%0d)", name, exp_reads, exp_writes);
    end else begin
      dump_pool({name," GOT"},  got);
      dump_pool({name," GOLD"}, gold);
      $fatal(1,"[%s] FAIL errs=%0d", name, errs);
    end
  endtask

  // ============================================================
  // Main
  // ============================================================
  initial begin
    start0 = 0; start1 = 0; start2 = 0;

    @(negedge reset);
    repeat (2) @(posedge clk);

    run_case("CASE0_SEQ_LAT1", 0, 20000);
    run_case("CASE1_NEG_LAT2", 1, 20000);
    run_case("CASE2_DEC_LAT0", 2, 20000);

    $display("ALL MAXPOOL TESTS PASSED");
    $finish;
  end

endmodule

`default_nettype wire
