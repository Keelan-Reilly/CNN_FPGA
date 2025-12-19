`timescale 1ns/1ps
`default_nettype none

// ============================================================
// BRAM-like vector source with configurable edge-to-edge latency
//   - When en is pulsed, the address is sampled.
//   - After LAT cycles, q updates to mem[addr].
//   - When no valid response that cycle, q holds last value.
// ============================================================
module tb_vec_bram #(
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

  // Pipeline arrays: make sure they exist even when LAT==0
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
// Dense "contract + golden" testbench
//   - Multiple DUT instances with different BRAM_LAT + weight sets
//   - DUT_LAT = BRAM_LAT + 1 (because MAC samples on posedge)
// ============================================================
module tb_dense;

  // ------------------------ clock/reset ------------------------
  logic clk=1'b0, reset=1'b1;
  always #5 clk = ~clk; // 100MHz

  initial begin
    $dumpfile("wave_dense.vcd");
    $dumpvars(0, tb_dense);
    repeat (4) @(posedge clk);
    reset = 1'b0;
  end

  // ------------------------ common params ------------------------
  localparam int DW        = 16;
  localparam int FRAC_BITS = 0;
  localparam int POST_SHIFT= 0;

  localparam int IN_DIM  = 8;
  localparam int OUT_DIM = 5;

  localparam int IN_AW   = (IN_DIM<=1)?1:$clog2(IN_DIM);
  localparam int W_DEPTH = IN_DIM*OUT_DIM;

  typedef logic [IN_AW-1:0] in_addr_t;

  // Saturation bounds in DW
  localparam logic signed [DW-1:0] S_MAX = (1 <<< (DW-1)) - 1;     //  32767
  localparam logic signed [DW-1:0] S_MIN = - (1 <<< (DW-1));       // -32768

  // ACCW match DUT formula
  localparam int ACCW = DW*2 + $clog2(IN_DIM) + 2;

  // ------------------------ helper funcs ------------------------
  function automatic int w_idx(input int o, input int i);
    return o*IN_DIM + i;
  endfunction

  function automatic logic signed [ACCW-1:0] sext_dw_to_accw(input logic signed [DW-1:0] x);
    return $signed({{(ACCW-DW){x[DW-1]}}, x});
  endfunction

  function automatic logic signed [ACCW-1:0] sext_prod_to_accw(input logic signed [2*DW-1:0] p);
    return $signed({{(ACCW-2*DW){p[2*DW-1]}}, p});
  endfunction

  function automatic logic signed [DW-1:0] clamp_dw_from_accw(input logic signed [ACCW-1:0] v);
    if      (v > sext_dw_to_accw(S_MAX)) return S_MAX;
    else if (v < sext_dw_to_accw(S_MIN)) return S_MIN;
    else                                 return v[DW-1:0];
  endfunction

  // Compute golden output vector for a given x[], W/B mem images.
  task automatic compute_golden(
    input  logic signed [DW-1:0] x   [0:IN_DIM-1],
    input  logic signed [DW-1:0] Wm  [0:W_DEPTH-1],
    input  logic signed [DW-1:0] Bm  [0:OUT_DIM-1],
    output logic signed [DW-1:0] y   [0:OUT_DIM-1],
    output int sat_p,
    output int sat_n
  );
    int o, i;
    logic signed [ACCW-1:0] acc;
    logic signed [2*DW-1:0] p;
    logic signed [ACCW-1:0] shifted;
    logic signed [DW-1:0]   res;

    sat_p = 0;
    sat_n = 0;

    for (o=0;o<OUT_DIM;o++) begin
      acc = sext_dw_to_accw(Bm[o]) <<< FRAC_BITS;
      for (i=0;i<IN_DIM;i++) begin
        p   = $signed(x[i]) * $signed(Wm[w_idx(o,i)]);
        acc = acc + sext_prod_to_accw(p);
      end

      shifted = acc >>> (FRAC_BITS + POST_SHIFT);
      res     = clamp_dw_from_accw(shifted);
      y[o]    = res;

      if (res === S_MAX) sat_p++;
      if (res === S_MIN) sat_n++;
    end
  endtask

  task automatic dump_vec(input string tag, input logic signed [DW-1:0] v [0:OUT_DIM-1]);
    $display("\n--- %s ---", tag);
    for (int o=0;o<OUT_DIM;o++) $display("  y[%0d] = %0d", o, v[o]);
    $display("--- end %s ---\n", tag);
  endtask

  // ============================================================
  // Build 3 cases worth of weight/bias files (time 0)
  // ============================================================
  initial begin : MAKE_MEM_FILES
    integer f;
    int o,i;

    // ---------- CASE0 ----------
    f = $fopen("tb_fc_w_case0.mem","w");
    // o0
    $fdisplay(f,"%0h",16'sd1); for (i=1;i<IN_DIM;i++) $fdisplay(f,"%0h",16'sd0);
    // o1
    $fdisplay(f,"%0h",16'sd0); $fdisplay(f,"%0h",16'sd1); $fdisplay(f,"%0h",16'sd1);
    for (i=3;i<IN_DIM;i++) $fdisplay(f,"%0h",16'sd0);
    // o2
    for (i=0;i<IN_DIM;i++) $fdisplay(f,"%0h", (i==3)?16'sd2:16'sd0);
    // o3
    for (i=0;i<IN_DIM;i++) $fdisplay(f,"%0h", (i<4)?16'sd1:16'sd0);
    // o4
    $fdisplay(f,"%0h",-16'sd1); for (i=1;i<IN_DIM;i++) $fdisplay(f,"%0h",16'sd0);
    $fclose(f);

    f = $fopen("tb_fc_b_case0.mem","w");
    $fdisplay(f,"%0h",16'sd2);
    $fdisplay(f,"%0h",-16'sd1);
    $fdisplay(f,"%0h",16'sd3);
    $fdisplay(f,"%0h",16'sd0);
    $fdisplay(f,"%0h",16'sd5);
    $fclose(f);

    // ---------- CASE1 (+SAT) ----------
    f = $fopen("tb_fc_w_case1.mem","w");
    for (o=0;o<OUT_DIM;o++) for (i=0;i<IN_DIM;i++) $fdisplay(f,"%0h",16'sd32767);
    $fclose(f);
    f = $fopen("tb_fc_b_case1.mem","w");
    for (o=0;o<OUT_DIM;o++) $fdisplay(f,"%0h",16'sd32767);
    $fclose(f);

    // ---------- CASE2 (-SAT) ----------
    f = $fopen("tb_fc_w_case2.mem","w");
    for (o=0;o<OUT_DIM;o++) for (i=0;i<IN_DIM;i++) $fdisplay(f,"%0h",-16'sd32768);
    $fclose(f);
    f = $fopen("tb_fc_b_case2.mem","w");
    for (o=0;o<OUT_DIM;o++) $fdisplay(f,"%0h",-16'sd32768);
    $fclose(f);

    #1; // ensure files exist before any DUT $readmemh
  end

  // ============================================================
  // Case-specific input vectors
  // ============================================================
  logic signed [DW-1:0] x_case0 [0:IN_DIM-1];
  logic signed [DW-1:0] x_case1 [0:IN_DIM-1];
  logic signed [DW-1:0] x_case2 [0:IN_DIM-1];

  initial begin
    x_case0[0]=10; x_case0[1]=3; x_case0[2]=7; x_case0[3]=4;
    x_case0[4]=1;  x_case0[5]=2; x_case0[6]=0; x_case0[7]=-6;

    for (int i=0;i<IN_DIM;i++) x_case1[i]=16'sd32767;
    for (int i=0;i<IN_DIM;i++) x_case2[i]=16'sd32767;
  end

  // ============================================================
  // Latencies
  // ============================================================
  localparam int BRAM_LAT0 = 1;
  localparam int BRAM_LAT1 = 2;
  localparam int BRAM_LAT2 = 0;

  localparam int DUT_LAT0  = BRAM_LAT0 + 1;
  localparam int DUT_LAT1  = BRAM_LAT1 + 1;
  localparam int DUT_LAT2  = BRAM_LAT2 + 1;

  // ---- DUT0 signals ----
  in_addr_t in_addr0;
  logic     in_en0;
  logic signed [DW-1:0] in_q0;
  logic signed [DW-1:0] out0 [0:OUT_DIM-1];
  logic start0, done0;

  // ---- DUT1 signals ----
  in_addr_t in_addr1;
  logic     in_en1;
  logic signed [DW-1:0] in_q1;
  logic signed [DW-1:0] out1 [0:OUT_DIM-1];
  logic start1, done1;

  // ---- DUT2 signals ----
  in_addr_t in_addr2;
  logic     in_en2;
  logic signed [DW-1:0] in_q2;
  logic signed [DW-1:0] out2 [0:OUT_DIM-1];
  logic start2, done2;

  // BRAM models
  tb_vec_bram #(.DW(DW), .DEPTH(IN_DIM), .LAT(BRAM_LAT0), .AW(IN_AW))
    bram0 (.clk, .en(in_en0), .addr(in_addr0), .mem(x_case0), .q(in_q0));

  tb_vec_bram #(.DW(DW), .DEPTH(IN_DIM), .LAT(BRAM_LAT1), .AW(IN_AW))
    bram1 (.clk, .en(in_en1), .addr(in_addr1), .mem(x_case1), .q(in_q1));

  tb_vec_bram #(.DW(DW), .DEPTH(IN_DIM), .LAT(BRAM_LAT2), .AW(IN_AW))
    bram2 (.clk, .en(in_en2), .addr(in_addr2), .mem(x_case2), .q(in_q2));

  // DUTs
  dense #(
    .DATA_WIDTH(DW), .FRAC_BITS(FRAC_BITS),
    .IN_DIM(IN_DIM), .OUT_DIM(OUT_DIM),
    .POST_SHIFT(POST_SHIFT),
    .WEIGHTS_FILE("tb_fc_w_case0.mem"),
    .BIASES_FILE ("tb_fc_b_case0.mem"),
    .LAT(DUT_LAT0),
    .DBG_ENABLE(0)
  ) dut0 (
    .clk, .reset, .start(start0),
    .in_addr(in_addr0), .in_en(in_en0), .in_q(in_q0),
    .out_vec(out0), .done(done0)
  );

  dense #(
    .DATA_WIDTH(DW), .FRAC_BITS(FRAC_BITS),
    .IN_DIM(IN_DIM), .OUT_DIM(OUT_DIM),
    .POST_SHIFT(POST_SHIFT),
    .WEIGHTS_FILE("tb_fc_w_case1.mem"),
    .BIASES_FILE ("tb_fc_b_case1.mem"),
    .LAT(DUT_LAT1),
    .DBG_ENABLE(0)
  ) dut1 (
    .clk, .reset, .start(start1),
    .in_addr(in_addr1), .in_en(in_en1), .in_q(in_q1),
    .out_vec(out1), .done(done1)
  );

  dense #(
    .DATA_WIDTH(DW), .FRAC_BITS(FRAC_BITS),
    .IN_DIM(IN_DIM), .OUT_DIM(OUT_DIM),
    .POST_SHIFT(POST_SHIFT),
    .WEIGHTS_FILE("tb_fc_w_case2.mem"),
    .BIASES_FILE ("tb_fc_b_case2.mem"),
    .LAT(DUT_LAT2),
    .DBG_ENABLE(0)
  ) dut2 (
    .clk, .reset, .start(start2),
    .in_addr(in_addr2), .in_en(in_en2), .in_q(in_q2),
    .out_vec(out2), .done(done2)
  );

  // ============================================================
  // Protocol monitors
  // ============================================================
  typedef struct packed {
    int reads;
    int exp_i;
    int exp_o;
    bit started;
    bit done_seen;
  } mon_t;

  mon_t m0, m1, m2;

  task automatic mon_reset(ref mon_t m);
    m.reads     = 0;
    m.exp_i     = 0;
    m.exp_o     = 0;
    m.started   = 0;
    m.done_seen = 0;
  endtask

  task automatic mon_step(
    input string tag,
    ref mon_t m,
    input logic rst,
    input logic start,
    input logic done,
    input logic en,
    input in_addr_t addr
  );
    if (rst) begin
      mon_reset(m);
    end else begin
      if (start) begin
        m.started   = 1;
        m.exp_i     = 0;
        m.exp_o     = 0;
        m.reads     = 0;
        m.done_seen = 0;
      end

      if (done) m.done_seen = 1;

      if (en) begin
        m.reads++;

        // Address contract: expect addr == exp_i
        if (addr !== in_addr_t'(m.exp_i)) begin
          $error("[%0t][%s] BAD_ADDR: got %0d exp %0d (exp_o=%0d exp_i=%0d reads=%0d)",
                 $time, tag, addr, m.exp_i, m.exp_o, m.exp_i, m.reads);
        end

        if (m.exp_i == IN_DIM-1) begin
          m.exp_i = 0;
          m.exp_o++;
        end else begin
          m.exp_i++;
        end
      end
    end
  endtask

  always_ff @(posedge clk) begin
    mon_step("DUT0", m0, reset, start0, done0, in_en0, in_addr0);
    mon_step("DUT1", m1, reset, start1, done1, in_en1, in_addr1);
    mon_step("DUT2", m2, reset, start2, done2, in_en2, in_addr2);
  end

  // ============================================================
  // Run a case
  // ============================================================
  task automatic run_case(
    input string name,
    input int    which,                 // 0/1/2
    input int    guard_max
  );
    int guard;
    int errs;
    int sat_p_exp, sat_n_exp;
    int sat_p_got, sat_n_got;

    logic signed [DW-1:0] gold [0:OUT_DIM-1];
    logic signed [DW-1:0] Wimg [0:W_DEPTH-1];
    logic signed [DW-1:0] Bimg [0:OUT_DIM-1];
    logic signed [DW-1:0] y_hw [0:OUT_DIM-1];

    if (which==0) begin
      $readmemh("tb_fc_w_case0.mem", Wimg);
      $readmemh("tb_fc_b_case0.mem", Bimg);
      compute_golden(x_case0, Wimg, Bimg, gold, sat_p_exp, sat_n_exp);
      start0 <= 1'b1; @(posedge clk); start0 <= 1'b0;
    end else if (which==1) begin
      $readmemh("tb_fc_w_case1.mem", Wimg);
      $readmemh("tb_fc_b_case1.mem", Bimg);
      compute_golden(x_case1, Wimg, Bimg, gold, sat_p_exp, sat_n_exp);
      start1 <= 1'b1; @(posedge clk); start1 <= 1'b0;
    end else begin
      $readmemh("tb_fc_w_case2.mem", Wimg);
      $readmemh("tb_fc_b_case2.mem", Bimg);
      compute_golden(x_case2, Wimg, Bimg, gold, sat_p_exp, sat_n_exp);
      start2 <= 1'b1; @(posedge clk); start2 <= 1'b0;
    end

    guard = 0;
    while (guard < guard_max) begin
      @(posedge clk);
      guard++;
      if ((which==0 && done0) || (which==1 && done1) || (which==2 && done2)) break;
    end
    if (guard >= guard_max) $fatal(1, "[%s] TIMEOUT waiting done (guard=%0d)", name, guard);

    // done should be a 1-cycle pulse
    @(posedge clk);
    if ((which==0 && done0) || (which==1 && done1) || (which==2 && done2))
      $error("[%s] done is not a 1-cycle pulse", name);

    sat_p_got = 0; sat_n_got = 0;
    for (int o=0;o<OUT_DIM;o++) begin
      if (which==0) y_hw[o] = out0[o];
      else if (which==1) y_hw[o] = out1[o];
      else y_hw[o] = out2[o];

      if (y_hw[o] === S_MAX) sat_p_got++;
      if (y_hw[o] === S_MIN) sat_n_got++;
    end

    errs = 0;
    for (int o=0;o<OUT_DIM;o++) begin
      if (y_hw[o] !== gold[o]) begin
        $display("[%s] MISMATCH o=%0d got=%0d exp=%0d", name, o, y_hw[o], gold[o]);
        errs++;
      end
    end

    if (sat_p_got != sat_p_exp || sat_n_got != sat_n_exp) begin
      $display("[%s] SAT_COUNT got(+%0d,-%0d) exp(+%0d,-%0d)",
               name, sat_p_got, sat_n_got, sat_p_exp, sat_n_exp);
      errs++;
    end

    if (which==0) begin
      if (m0.reads != OUT_DIM*IN_DIM) begin
        $display("[%s] BAD_READ_COUNT got=%0d exp=%0d", name, m0.reads, OUT_DIM*IN_DIM);
        errs++;
      end
    end else if (which==1) begin
      if (m1.reads != OUT_DIM*IN_DIM) begin
        $display("[%s] BAD_READ_COUNT got=%0d exp=%0d", name, m1.reads, OUT_DIM*IN_DIM);
        errs++;
      end
    end else begin
      if (m2.reads != OUT_DIM*IN_DIM) begin
        $display("[%s] BAD_READ_COUNT got=%0d exp=%0d", name, m2.reads, OUT_DIM*IN_DIM);
        errs++;
      end
    end

    if (errs==0) begin
      $display("[%s] PASS (reads=%0d, sat +%0d -%0d)",
               name,
               (which==0)?m0.reads:((which==1)?m1.reads:m2.reads),
               sat_p_got, sat_n_got);
    end else begin
      dump_vec({name," HW"},   y_hw);
      dump_vec({name," GOLD"}, gold);
      $fatal(1, "[%s] FAIL errs=%0d", name, errs);
    end
  endtask

  // ============================================================
  // Test sequence
  // ============================================================
  initial begin : MAIN
    start0 = 0; start1 = 0; start2 = 0;

    @(negedge reset);
    repeat (2) @(posedge clk);

    run_case("CASE0_NORMAL_DUTLAT2", 0, 200000);
    run_case("CASE1_POSSAT_DUTLAT3", 1, 200000);
    run_case("CASE2_NEGSAT_DUTLAT1", 2, 200000);

    $display("ALL DENSE TESTS PASSED");
    $finish;
  end

endmodule

`default_nettype wire
