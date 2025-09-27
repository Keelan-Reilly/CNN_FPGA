//------------------------------------------------------------------------------
// Module: bram_tdp
// Description: 
//   This module implements a true dual-port Block RAM (BRAM) with both ports 
//   supporting independent read and write operations. Each port can access the 
//   memory simultaneously, allowing concurrent read/write or write/write 
//   operations. The design features a 1-cycle read latency, meaning data 
//   requested for reading is available on the following clock cycle.
//------------------------------------------------------------------------------
module bram_tdp #(
  parameter int DW    = 16,
  parameter int DEPTH = 1024,
  parameter int AW    = (DEPTH<=1) ? 1 : $clog2(DEPTH)
)(
  input  logic                  clk,
  // Port A
  input  logic                  a_en,
  input  logic                  a_we,
  input  logic [AW-1:0]         a_addr,
  input  logic signed [DW-1:0]  a_din,
  output logic signed [DW-1:0]  a_dout,
  // Port B
  input  logic                  b_en,
  input  logic                  b_we,
  input  logic [AW-1:0]         b_addr,
  input  logic signed [DW-1:0]  b_din,
  output logic signed [DW-1:0]  b_dout
);
  (* ram_style="block" *) logic signed [DW-1:0] mem [0:DEPTH-1];

  // Port A process
  always_ff @(posedge clk) begin
    if (a_en) begin
      if (a_we) mem[a_addr] <= a_din;
      a_dout <= mem[a_addr];   // 1-cycle latency
    end
  end

  // Port B process
  always_ff @(posedge clk) begin
    if (b_en) begin
      if (b_we) mem[b_addr] <= b_din;
      b_dout <= mem[b_addr];   // 1-cycle latency
    end
  end
endmodule
