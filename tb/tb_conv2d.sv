`timescale 1ns/1ps
`default_nettype none

module tb_conv2d;

  // ----------------------------
  // Configuration
  // ----------------------------
  localparam int DATA_WIDTH = 16;
  localparam int FRAC_BITS  = 0;

  localparam int IC = 2;
  localparam int OC = 3;

  localparam int K  = 3;
  localparam int H  = 4;
  localparam int W  = 4;

  localparam int IF_SZ = IC*H*W;
  localparam int OF_SZ = OC*H*W;

  localparam int IF_AW = (IF_SZ<=1) ? 1 : $clog2(IF_SZ);
  localparam int OF_AW = (OF_SZ<=1) ? 1 : $clog2(OF_SZ);

  // Saturation bounds for DATA_WIDTH
  localparam logic signed [DATA_WIDTH-1:0] S_MAX = (1 <<< (DATA_WIDTH-1)) - 1;
  localparam logic signed [DATA_WIDTH-1:0] S_MIN = - (1 <<< (DATA_WIDTH-1));

  // ----------------------------
  // Clock/reset
  // ----------------------------
  logic clk = 1'b0;
  logic reset = 1'b1;

  initial begin
    $dumpfile("wave.vcd");
    $dumpvars(0, tb_conv2d);
    forever #5 clk = ~clk;
  end

  initial begin
    repeat (4) @(posedge clk);
    reset = 1'b0;
  end

  // ----------------------------
  // Helper: linear indexing CHW
  // ----------------------------
  function automatic int lin3(input int ch, input int r, input int c, input int HH, input int WW);
    return (ch*HH + r)*WW + c;
  endfunction

  // ----------------------------
  // Weight/bias generation
  // ----------------------------
  localparam int W_DEPTH = OC*IC*K*K;

  logic signed [DATA_WIDTH-1:0] tbW [0:W_DEPTH-1];
  logic signed [DATA_WIDTH-1:0] tbB [0:OC-1];

  function automatic int w_addr(input int oc_i, input int ic_i, input int kr_i, input int kc_i);
    return oc_i*(IC*K*K) + ic_i*(K*K) + kr_i*K + kc_i;
  endfunction

  initial begin : make_weight_bias_files
    integer fW, fB;
    int a;

    // biases = 0
    for (int oc_i=0; oc_i<OC; oc_i++) tbB[oc_i] = '0;

    // weights
    for (int oc_i=0; oc_i<OC; oc_i++) begin
      for (int ic_i=0; ic_i<IC; ic_i++) begin
        for (int kr_i=0; kr_i<K; kr_i++) begin
          for (int kc_i=0; kc_i<K; kc_i++) begin
            a = w_addr(oc_i, ic_i, kr_i, kc_i);

            if (oc_i == 0) begin
              tbW[a] = $signed((oc_i+1)*11 + (ic_i+1)*7 + kr_i*3 + kc_i);
            end else if (oc_i == 1) begin
              tbW[a] = $signed(16'sd3000);
            end else begin
              tbW[a] = $signed(-16'sd3000);
            end
          end
        end
      end
    end

    fW = $fopen("tb_conv_w.mem","w");
    if (fW==0) $fatal(1, "TB: cannot open tb_conv_w.mem for write");
    for (int i=0; i<W_DEPTH; i++) $fdisplay(fW, "%0h", tbW[i]);
    $fclose(fW);

    fB = $fopen("tb_conv_b.mem","w");
    if (fB==0) $fatal(1, "TB: cannot open tb_conv_b.mem for write");
    for (int i=0; i<OC; i++) $fdisplay(fB, "%0h", tbB[i]);
    $fclose(fB);
  end

  // ----------------------------
  // IFMAP memory model
  // ----------------------------
  logic signed [DATA_WIDTH-1:0] ifmem [0:IF_SZ-1];

  initial begin : init_ifmap
    int idx;
    int v;
    idx = 0;
    for (int ic_i=0; ic_i<IC; ic_i++) begin
      for (int r=0; r<H; r++) begin
        for (int c=0; c<W; c++) begin
          v = (r*W + c + 1);
          if (ic_i == 0) ifmem[idx] = $signed(v);
          else           ifmem[idx] = $signed(-v);
          idx++;
        end
      end
    end
  end

  // DUT IF interface
  logic [IF_AW-1:0] if_addr;
  logic             if_en;
  logic signed [DATA_WIDTH-1:0] if_q;

  // 1-cycle BRAM
  logic [IF_AW-1:0]             if_addr_q;
  logic signed [DATA_WIDTH-1:0] if_q_reg;

  always_ff @(posedge clk) begin
    if (if_en) begin
      if_addr_q <= if_addr;
      if_q_reg  <= ifmem[if_addr];
    end
  end
  assign if_q = if_q_reg;

  // ----------------------------
  // Capture OFMAP writes
  // ----------------------------
  logic [OF_AW-1:0]              conv_addr;
  logic                          conv_en, conv_we;
  logic signed [DATA_WIDTH-1:0]  conv_d;

  logic signed [DATA_WIDTH-1:0]  ofmem [0:OF_SZ-1];
  bit                            wrote [0:OF_SZ-1];

  int write_count;
  bit done_seen;

  // ----------------------------
  // Start/done
  // ----------------------------
  logic start, done;

  // ----------------------------
  // DUT
  // ----------------------------
  conv2d #(
    .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
    .IN_CHANNELS(IC), .OUT_CHANNELS(OC),
    .KERNEL(K), .IMG_SIZE(H),
    .WEIGHTS_FILE("tb_conv_w.mem"),
    .BIASES_FILE ("tb_conv_b.mem")
  ) dut (
    .clk(clk),
    .reset(reset),
    .start(start),
    .if_addr(if_addr),
    .if_en(if_en),
    .if_q(if_q),
    .conv_addr(conv_addr),
    .conv_en(conv_en),
    .conv_we(conv_we),
    .conv_d(conv_d),
    .done(done)
  );

  // Write capture + protocol assertions
  always_ff @(posedge clk) begin
    if (reset) begin
      write_count <= 0;
      done_seen   <= 1'b0;
    end else begin
      if (done) done_seen <= 1'b1;

      if (conv_en && conv_we) begin
        if (^conv_addr === 1'bX) $fatal(1, "[%0t] WRITE: conv_addr is X", $time);
        if (^conv_d    === 1'bX) $fatal(1, "[%0t] WRITE: conv_d is X", $time);

        if (conv_addr >= OF_SZ)  $fatal(1, "[%0t] WRITE: conv_addr OOR %0d", $time, conv_addr);
        if (done_seen)           $fatal(1, "[%0t] WRITE after done! addr=%0d", $time, conv_addr);
        if (wrote[conv_addr])    $fatal(1, "[%0t] DUP WRITE addr=%0d", $time, conv_addr);

        wrote[conv_addr] <= 1'b1;
        ofmem[conv_addr] <= conv_d;
        write_count      <= write_count + 1;
      end
    end
  end

  // ----------------------------
  // Golden model
  // ----------------------------
  logic signed [DATA_WIDTH-1:0] gold [0:OF_SZ-1];

  function automatic logic signed [DATA_WIDTH-1:0] clamp_to_dw(input longint signed x);
    logic signed [DATA_WIDTH-1:0] y;
    begin
      if (x > $signed(S_MAX)) y = S_MAX;
      else if (x < $signed(S_MIN)) y = S_MIN;
      else y = x[DATA_WIDTH-1:0];
      return y;
    end
  endfunction

  task automatic compute_golden(output int sat_pos, output int sat_neg);
    int pad;
    longint signed acc;
    longint signed shifted;
    logic signed [DATA_WIDTH-1:0] y;
    int if_idx;
    int widx;

    sat_pos = 0;
    sat_neg = 0;
    pad = (K-1)/2;

    for (int oc_i=0; oc_i<OC; oc_i++) begin
      for (int r=0; r<H; r++) begin
        for (int c=0; c<W; c++) begin
          acc = 0;
          acc = $signed(tbB[oc_i]);

          for (int ic_i=0; ic_i<IC; ic_i++) begin
            for (int kr_i=0; kr_i<K; kr_i++) begin
              for (int kc_i=0; kc_i<K; kc_i++) begin
                int ir, icc;
                ir = r + kr_i - pad;
                icc = c + kc_i - pad;
                if (ir>=0 && ir<H && icc>=0 && icc<W) begin
                  if_idx = lin3(ic_i, ir, icc, H, W);
                  widx   = w_addr(oc_i, ic_i, kr_i, kc_i);
                  acc += $signed(ifmem[if_idx]) * $signed(tbW[widx]);
                end
              end
            end
          end

          if (FRAC_BITS == 0) shifted = acc;
          else                shifted = (acc >>> FRAC_BITS);

          y = clamp_to_dw(shifted);
          gold[lin3(oc_i, r, c, H, W)] = y;

          if (y == S_MAX) sat_pos++;
          if (y == S_MIN) sat_neg++;
        end
      end
    end
  endtask

  task automatic clear_outputs();
    for (int i=0; i<OF_SZ; i++) begin
      ofmem[i] = '0;
      wrote[i] = 1'b0;
    end
    write_count = 0;
    done_seen   = 1'b0;
  endtask

  task automatic dump_chan(
    input string tag,
    input int ch,
    input logic signed [DATA_WIDTH-1:0] arr [0:OF_SZ-1]
  );
    $display("\n--- %s (ch=%0d) ---", tag, ch);
    for (int r=0; r<H; r++) begin
      string line;
      line = "";
      for (int c=0; c<W; c++) begin
        int idx;
        idx = lin3(ch, r, c, H, W);
        line = {line, $sformatf("%0d%s", arr[idx], (c==W-1) ? "" : " ")};
      end
      $display("%s", line);
    end
    $display("--- end %s ---\n", tag);
  endtask

  task automatic check_results(input string run_tag, input int exp_sat_pos, input int exp_sat_neg);
    int errs;
    int got_sat_pos;
    int got_sat_neg;

    errs = 0;
    got_sat_pos = 0;
    got_sat_neg = 0;

    // coverage: every address written exactly once
    for (int i=0; i<OF_SZ; i++) begin
      if (!wrote[i]) begin
        $display("[%s] MISSING WRITE addr=%0d", run_tag, i);
        errs++;
      end
    end
    if (write_count != OF_SZ) begin
      $display("[%s] write_count mismatch: got %0d exp %0d", run_tag, write_count, OF_SZ);
      errs++;
    end

    // compare
    for (int oc_i=0; oc_i<OC; oc_i++) begin
      for (int r=0; r<H; r++) begin
        for (int c=0; c<W; c++) begin
          int idx;
          idx = lin3(oc_i, r, c, H, W);
          if (ofmem[idx] !== gold[idx]) begin
            $display("[%s] MISMATCH oc=%0d r=%0d c=%0d got=%0d exp=%0d",
                     run_tag, oc_i, r, c, ofmem[idx], gold[idx]);
            errs++;
          end
        end
      end
    end

    // saturation counts (informational)
    for (int i=0; i<OF_SZ; i++) begin
      if (ofmem[i] == S_MAX) got_sat_pos++;
      if (ofmem[i] == S_MIN) got_sat_neg++;
    end
    if (got_sat_pos != exp_sat_pos || got_sat_neg != exp_sat_neg) begin
      $display("[%s] SAT COUNT NOTE: got (+%0d,-%0d) exp (+%0d,-%0d)",
               run_tag, got_sat_pos, got_sat_neg, exp_sat_pos, exp_sat_neg);
    end

    if (errs == 0) begin
      $display("[%s] PASS", run_tag);
    end else begin
      $display("[%s] FAIL errs=%0d", run_tag, errs);
      for (int oc_i=0; oc_i<OC; oc_i++) begin
        dump_chan({run_tag," HW"}, oc_i, ofmem);
        dump_chan({run_tag," GOLD"}, oc_i, gold);
      end
      $fatal;
    end
  endtask

  task automatic run_once(input string run_tag);
    int guard;

    clear_outputs();

    // start pulse
    start = 1'b0;
    repeat (2) @(posedge clk);
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;

    // illegal extra start while busy (should not break anything)
    repeat (7) @(posedge clk);
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;

    guard = 0;
    while (!done && guard < 200000) begin
      @(posedge clk);
      guard++;
    end
    if (!done) $fatal(1, "[%s] TIMEOUT waiting for done", run_tag);

    // ensure no trailing write after done
    @(posedge clk);
  endtask

  // ----------------------------
  // Main
  // ----------------------------
  initial begin : main
    int exp_sat_pos, exp_sat_neg;
    start = 1'b0;

    @(negedge reset);

    compute_golden(exp_sat_pos, exp_sat_neg);
    $display("TB: expected saturation counts: +SAT=%0d -SAT=%0d (OF_SZ=%0d)",
             exp_sat_pos, exp_sat_neg, OF_SZ);

    run_once("RUN1");
    check_results("RUN1", exp_sat_pos, exp_sat_neg);

    run_once("RUN2");
    check_results("RUN2", exp_sat_pos, exp_sat_neg);

    $display("ALL TESTS PASSED");
    $finish;
  end

endmodule

`default_nettype wire
