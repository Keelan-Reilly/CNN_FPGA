`timescale 1ns/1ps
`default_nettype none

// 1-cycle synchronous BRAM read model for IFMAP (matches conv2d timing)
module tb_conv2d;
  // Params (FRAC_BITS=0 -> pure integer math)
  localparam int DATA_WIDTH=16, FRAC_BITS=0;
  localparam int IC=1, OC=1, K=3, H=3, W=3;
  localparam int IF_SZ = IC*H*W;
  localparam int OF_SZ = OC*H*W;

  // Clock/reset (init + deassert)
  logic clk=1'b0, reset=1'b1;
  initial begin
    // VCD
    $dumpfile("wave.vcd");
    $dumpvars(0, tb_conv2d);

    fork
      forever #5 clk = ~clk;   // 100 MHz
    join_none

    repeat (4) @(posedge clk);
    reset = 1'b0;
  end

  // Make tiny weight/bias files before $readmemh runs in the DUT
  initial begin
    integer f;
    f = $fopen("tb_conv_w.mem","w");
      repeat (9) $fdisplay(f, "%0h", 16'd1); // 3x3 all-ones
    $fclose(f);
    f = $fopen("tb_conv_b.mem","w");
      $fdisplay(f, "%0h", 16'd0);
    $fclose(f);
  end

  // IFMAP memory (1-cycle read latency), initialized to values 1..9
  logic signed [DATA_WIDTH-1:0] ifmem [0:IF_SZ-1];
  initial begin
    int idx = 0;
    for (int r=0; r<H; r++) begin
      for (int c=0; c<W; c++) begin
        /* verilator lint_off WIDTHTRUNC */
        ifmem[idx] = $signed(idx+1); // narrow 32->16 intentionally
        /* verilator lint_on  WIDTHTRUNC */
        idx++;
      end
    end
  end

  // Wires to DUT (IFMAP side)
  localparam int IF_AW = (IF_SZ<=1)?1:$clog2(IF_SZ);
  logic [IF_AW-1:0] if_addr;
  logic             if_en;
  logic signed [DATA_WIDTH-1:0] if_q;

  // 1-cycle synchronous BRAM read model (output updates only on if_en)
  logic [IF_AW-1:0] if_addr_q;
  logic signed [DATA_WIDTH-1:0] if_q_reg;

  always_ff @(posedge clk) begin
    if (if_en) begin
      if_addr_q <= if_addr;          // captured for debug / visibility
      if_q_reg  <= ifmem[if_addr];   // data becomes valid next cycle (registered)
    end
  end

  assign if_q = if_q_reg;

    // debug after updates settle
    always @(posedge clk) if (if_en)
    $strobe("[%0t] TB  : if_en=1 addr=%0d -> if_q_next=%0d", $time, if_addr, ifmem[if_addr]);

  // Capture CONV writes
  localparam int OF_AW = (OF_SZ<=1)?1:$clog2(OF_SZ);
  logic [OF_AW-1:0]     conv_addr;
  logic                 conv_en, conv_we;
  logic signed [DATA_WIDTH-1:0] conv_d;


  logic signed [DATA_WIDTH-1:0] ofmem [0:OF_SZ-1];

  always_ff @(posedge clk) begin
    if (conv_en && conv_we) ofmem[conv_addr] <= conv_d;
  end

  // Start/done
  logic start, done;

  // DUT
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

  // Helpers
  function int lin3(input int ch,input int r,input int c, input int HH, input int WW);
    return (ch*HH + r)*WW + c;
  endfunction

  // ---- GOLDEN compute & print (time 0) ----
  logic signed [DATA_WIDTH-1:0] gold [0:OF_SZ-1];
  initial begin
    automatic int pad=(K-1)/2;
    #1; // let ifmem init
    for (int r=0;r<H;r++) begin
      for (int c=0;c<W;c++) begin
        int acc=0;
        for (int kr=0;kr<K;kr++) for (int kc=0;kc<K;kc++) begin
          int ir = r + kr - pad;
          int ic = c + kc - pad;
          if (ir>=0 && ir<H && ic>=0 && ic<W)
            acc += ifmem[lin3(0,ir,ic,H,W)];
        end
        /* verilator lint_off WIDTHTRUNC */
        gold[lin3(0,r,c,H,W)] = $signed(acc);
        /* verilator lint_on  WIDTHTRUNC */
      end
    end
    dump_matrix("GOLDEN REF", gold);
  end

  // Test sequence
  initial begin
    int guard;
    int errs;

    @(negedge reset);
    repeat(2) @(posedge clk);
    start = 1'b1; @(posedge clk); start = 1'b0;  // blocking in initial

    // Wait for completion (timeout guard)
    guard = 0;
    $display("[%0t] waiting for done...", $time);
    while (!done && guard < 10000) begin
      @(posedge clk);
      guard++;
    end
    $display("[%0t] done=%0b after %0d cycles", $time, done, guard);
    if (!done) begin
      $error("conv2d: timeout waiting for done");
      $finish;
    end

    // >>> Print the DUT output now that it's complete <<<
    dump_matrix("HW OUTPUT", ofmem);


    // Compare all pixels
    errs = 0;
    for (int r=0;r<H;r++) begin
      for (int c=0;c<W;c++) begin
        int idx = lin3(0,r,c,H,W);
        if (ofmem[idx] !== gold[idx]) begin
          $display("Mismatch at (%0d,%0d): got %0d, exp %0d", r,c, ofmem[idx], gold[idx]);
          errs++;
        end
      end
    end

    if (errs==0) $display("PASS: conv2d 3x3 all-ones kernel matches golden.");
    else         $error("FAIL: conv2d mismatches=%0d", errs);

    repeat (5) @(posedge clk); // give tracer a couple cycles
    $finish;
  end

  task automatic dump_matrix
  (
    input string tag,
    input logic signed [DATA_WIDTH-1:0] arr [0:OF_SZ-1]
  );
    $display("\n--- %s ---", tag);
    for (int r=0; r<H; r++) begin
      string line = "";
      for (int c=0; c<W; c++) begin
        int idx = lin3(0,r,c,H,W);
        line = {line, $sformatf("%0d%s", arr[idx], (c==W-1) ? "" : " ")};
      end
      $display("%s", line);
    end
    $display("--- end %s ---\n", tag);
  endtask


endmodule
