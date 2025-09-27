//------------------------------------------------------------------------------
// Module: bram_sdp
// Description: 
//   This module implements a simple dual-port Block RAM (BRAM) with separate
//   write and read ports. Port A is dedicated for write operations, while
//   Port B is used for read operations. The read port has a 1-cycle latency,
//   meaning data becomes available one clock cycle after the read address is
//   provided.
//------------------------------------------------------------------------------

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
