`timescale 1ns/1ps
`default_nettype none

module tb_dense;
  localparam int DW=16, FRAC_BITS=0, IN_DIM=4, OUT_DIM=3, POST_SHIFT=0;

  logic clk=0, reset=1; always #5 clk=~clk;
  initial begin repeat(4) @(posedge clk); reset=0; end

  // Create small weight/bias mem files
  initial begin
    integer f;
    // W = [ [1,0,0,0],
    //       [0,1,1,0],
    //       [0,0,0,2] ]
    f=$fopen("tb_fc_w.mem","w");
      $fdisplay(f,"%0h",16'd1); $fdisplay(f,"%0h",16'd0); $fdisplay(f,"%0h",16'd0); $fdisplay(f,"%0h",16'd0);
      $fdisplay(f,"%0h",16'd0); $fdisplay(f,"%0h",16'd1); $fdisplay(f,"%0h",16'd1); $fdisplay(f,"%0h",16'd0);
      $fdisplay(f,"%0h",16'd0); $fdisplay(f,"%0h",16'd0); $fdisplay(f,"%0h",16'd0); $fdisplay(f,"%0h",16'd2);
    $fclose(f);
    // B = [2, -1, 3]
    f=$fopen("tb_fc_b.mem","w");
      $fdisplay(f,"%0h",16'd2);
      $fdisplay(f, "%0h", -16'sd1);
      $fdisplay(f,"%0h",16'd3);
    $fclose(f);
  end

  // BRAM-like input vector (1-cycle latency)
  logic [($clog2(IN_DIM)>0)?$clog2(IN_DIM):1-1:0] in_addr;
  logic in_en;
  logic signed [DW-1:0] in_q;
  logic signed [DW-1:0] invec [0:IN_DIM-1];

  initial begin
    invec[0]=10; invec[1]=3; invec[2]=7; invec[3]=4; // x=[10,3,7,4]
  end

  always_ff @(posedge clk) begin
    if (in_en) begin
      in_q <= invec[in_addr];   // 1-cycle sync read, hold otherwise
    end
  end

  // Outputs
  logic signed [DW-1:0] out_vec [0:OUT_DIM-1];
  logic start, done;

  dense #(
    .DATA_WIDTH(DW), .FRAC_BITS(FRAC_BITS),
    .IN_DIM(IN_DIM), .OUT_DIM(OUT_DIM),
    .POST_SHIFT(POST_SHIFT),
    .WEIGHTS_FILE("tb_fc_w.mem"),
    .BIASES_FILE ("tb_fc_b.mem")
  ) dut (
    .clk, .reset, .start,
    .in_addr, .in_en, .in_q,
    .out_vec, .done
  );

  // Golden: y0=1*x0+2=12; y1=1*x1+1*x2-1=3+7-1=9; y2=2*x3+3=11
  initial begin
    int guard;
    int errs;
    @(negedge reset);
    repeat(2) @(posedge clk);
    start<=1; @(posedge clk); start<=0;

    guard=0; 
    while(!done && guard<10000) begin @(posedge clk); guard++; end
    if (!done) $fatal(1,"dense timeout");

    errs=0;
    if (out_vec[0]!==12) begin $display("y0 got %0d exp 12", out_vec[0]); errs++; end
    if (out_vec[1]!==9 ) begin $display("y1 got %0d exp 9",  out_vec[1]); errs++; end
    if (out_vec[2]!==11) begin $display("y2 got %0d exp 11", out_vec[2]); errs++; end

    if (errs==0) $display("PASS: dense small case correct.");
    else         $error("FAIL: dense mismatches=%0d", errs);
    $finish;
  end
endmodule
