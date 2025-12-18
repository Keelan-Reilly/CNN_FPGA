//------------------------------------------------------------------------------
// Module: bram_tdp
//
// Overview
//   True dual-port BRAM model with two fully independent ports (A and B).
//   Each port can perform a read and/or write in the same cycle, subject to
//   the usual “read-during-write” semantics of the underlying FPGA/BRAM.
//
// Behaviour
//   • Synchronous read: when *_en is asserted, *_dout updates on the clock edge
//     using the addressed memory location.
//   • Synchronous write: when *_en && *_we is asserted, *_din is written into
//     mem[*_addr] on the clock edge.
//   • Read latency: 1 cycle from asserting *_en with a stable address to seeing
//     the corresponding value at *_dout.
//
// Notes
//   • If a port reads and writes the same address in the same cycle, the value
//     seen on *_dout is implementation-dependent (read-first vs write-first).
//     This model performs a “read-first” style by assigning *_dout from mem[]
//     after the optional write in the same always_ff block.
//   • If both ports write the same address in the same cycle, the final stored
//     value is undefined (last assignment wins in simulation; hardware varies).
//------------------------------------------------------------------------------

module bram_tdp #(
  parameter int DW    = 16,
  parameter int DEPTH = 1024,
  parameter int AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
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

  (* ram_style = "block" *) logic signed [DW-1:0] mem [0:DEPTH-1];

  // Port A
  always_ff @(posedge clk) begin
    if (a_en) begin
      if (a_we) mem[a_addr] <= a_din;
      a_dout <= mem[a_addr];
    end
  end

  // Port B
  always_ff @(posedge clk) begin
    if (b_en) begin
      if (b_we) mem[b_addr] <= b_din;
      b_dout <= mem[b_addr];
    end
  end

endmodule
