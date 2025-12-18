//------------------------------------------------------------------------------
// Module: bram_sdp
//
// Overview
//   Simple dual-port BRAM model with separate write and read ports:
//     • Port A: write-only
//     • Port B: read-only
//
// Behaviour
//   • Write: when a_en && a_we is asserted, a_din is written to mem[a_addr]
//     on the rising edge.
//   • Read: when b_en is asserted, b_dout updates on the rising edge from
//     mem[b_addr] (1-cycle synchronous read latency).
//
// Notes
//   • If Port A writes an address in the same cycle Port B reads that address,
//     read-during-write behaviour is device-dependent (old vs new vs unknown).
//------------------------------------------------------------------------------

module bram_sdp #(
  parameter int DW    = 16,
  parameter int DEPTH = 1024,
  parameter int AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
  input  logic                  clk,

  // Port A (write)
  input  logic                  a_en,
  input  logic                  a_we,
  input  logic [AW-1:0]         a_addr,
  input  logic signed [DW-1:0]  a_din,

  // Port B (read)
  input  logic                  b_en,
  input  logic [AW-1:0]         b_addr,
  output logic signed [DW-1:0]  b_dout
);

  (* ram_style = "block" *) logic signed [DW-1:0] mem [0:DEPTH-1];

  // Write port
  always_ff @(posedge clk) begin
    if (a_en && a_we) begin
      mem[a_addr] <= a_din;
    end
  end

  // Read port (1-cycle latency)
  always_ff @(posedge clk) begin
    if (b_en) begin
      b_dout <= mem[b_addr];
    end
  end

endmodule
