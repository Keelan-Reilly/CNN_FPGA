// bram_sdp.sv : write port A, read port B (1-cycle read latency)
module bram_sdp #(
  parameter int DW    = 16,
  parameter int DEPTH = 1024,
  parameter int AW    = (DEPTH<=1) ? 1 : $clog2(DEPTH)
)(
  input  logic                  clk,
  // Port A: write
  input  logic                  a_en,
  input  logic                  a_we,
  input  logic [AW-1:0]         a_addr,
  input  logic signed [DW-1:0]  a_din,
  // Port B: read
  input  logic                  b_en,
  input  logic [AW-1:0]         b_addr,
  output logic signed [DW-1:0]  b_dout
);
  (* ram_style="block" *) logic signed [DW-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (a_en && a_we) mem[a_addr] <= a_din;
  end
  always_ff @(posedge clk) begin
    if (b_en) b_dout <= mem[b_addr];
  end
endmodule
