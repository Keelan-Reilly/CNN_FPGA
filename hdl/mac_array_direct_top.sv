//----------------------------------------------------------------------
// Directly measurable baseline MAC-array slice.
//
// This top is intentionally small and isolated from the CNN pipeline.
// It implements a parameterized ARRAY_ROWS x ARRAY_COLS baseline spatial
// MAC array that runs a fixed synthetic K-depth workload so the existing
// Vivado flow can measure DSP/LUT/timing on a real MAC-array-style RTL
// block without requiring the full architecture family.
//----------------------------------------------------------------------

(* keep_hierarchy = "yes" *)
module mac_array_baseline_cell #(
  parameter int DATA_WIDTH = 16,
  parameter int ACC_WIDTH = 40,
  parameter int CELL_INDEX = 0
)(
  input  logic clk,
  input  logic reset,
  input  logic load_i,
  input  logic step_i,
  input  logic signed [DATA_WIDTH-1:0] seed_a_i,
  input  logic signed [DATA_WIDTH-1:0] seed_b_i,
  output logic signed [ACC_WIDTH-1:0] acc_o
);
  logic signed [DATA_WIDTH-1:0] a_reg;
  logic signed [DATA_WIDTH-1:0] b_reg;
  (* use_dsp = "yes" *) logic signed [ACC_WIDTH-1:0] acc_reg;

  localparam logic signed [DATA_WIDTH-1:0] A_INC = DATA_WIDTH'(CELL_INDEX + 1);
  localparam logic signed [DATA_WIDTH-1:0] B_DEC = DATA_WIDTH'((CELL_INDEX % 3) + 1);

  assign acc_o = acc_reg;

  always_ff @(posedge clk) begin
    if (reset) begin
      a_reg <= '0;
      b_reg <= '0;
      acc_reg <= '0;
    end else if (load_i) begin
      a_reg <= seed_a_i;
      b_reg <= seed_b_i;
      acc_reg <= '0;
    end else if (step_i) begin
      acc_reg <= acc_reg + ($signed(a_reg) * $signed(b_reg));
      a_reg <= a_reg + A_INC;
      b_reg <= b_reg - B_DEC;
    end
  end
endmodule

(* keep_hierarchy = "yes" *)
module mac_array_direct_top #(
  parameter int DATA_WIDTH = 16,
  parameter int ARRAY_ROWS = 4,
  parameter int ARRAY_COLS = 4,
  parameter int K_DEPTH = 16,
  parameter int CLK_FREQ_HZ = 100_000_000
)(
  input  logic clk,
  input  logic reset,
  input  logic start_i,
  output logic done_o,
  output logic [3:0] signature_o
);
  localparam int CELL_COUNT = ARRAY_ROWS * ARRAY_COLS;
  localparam int K_COUNTER_WIDTH = (K_DEPTH <= 1) ? 1 : $clog2(K_DEPTH + 1);
  localparam int ACC_WIDTH = (2 * DATA_WIDTH) + ((K_DEPTH <= 1) ? 1 : $clog2(K_DEPTH)) + 4;

  logic active;
  logic load_cells;
  logic step_cells;
  logic [K_COUNTER_WIDTH-1:0] k_count;
  logic signed [ACC_WIDTH-1:0] acc [0:CELL_COUNT-1];

  function automatic logic signed [DATA_WIDTH-1:0] init_a(input int idx);
    init_a = DATA_WIDTH'((idx % 7) + 1);
  endfunction

  function automatic logic signed [DATA_WIDTH-1:0] init_b(input int idx);
    init_b = DATA_WIDTH'(((idx * 2) % 11) + 3);
  endfunction

  function automatic logic [3:0] fold_signature(input logic signed [ACC_WIDTH-1:0] values [0:CELL_COUNT-1]);
    logic [3:0] folded;
    begin
      folded = 4'h0;
      for (int idx = 0; idx < CELL_COUNT; idx++) begin
        folded ^= values[idx][3:0];
      end
      fold_signature = folded;
    end
  endfunction

  genvar cell_idx;
  generate
    for (cell_idx = 0; cell_idx < CELL_COUNT; cell_idx++) begin : gen_cells
      mac_array_baseline_cell #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .CELL_INDEX(cell_idx)
      ) u_cell (
        .clk(clk),
        .reset(reset),
        .load_i(load_cells),
        .step_i(step_cells),
        .seed_a_i(init_a(cell_idx)),
        .seed_b_i(init_b(cell_idx)),
        .acc_o(acc[cell_idx])
      );
    end
  endgenerate

  assign load_cells = start_i && !active;
  assign step_cells = active && (k_count < K_COUNTER_WIDTH'(K_DEPTH));

  always_ff @(posedge clk) begin
    if (reset) begin
      active <= 1'b0;
      done_o <= 1'b0;
      signature_o <= 4'h0;
      k_count <= '0;
    end else begin
      done_o <= 1'b0;
      if (start_i && !active) begin
        active <= 1'b1;
        signature_o <= 4'h0;
        k_count <= '0;
      end else if (active) begin
        if (k_count < K_COUNTER_WIDTH'(K_DEPTH)) begin
          k_count <= k_count + 1'b1;
        end else begin
          active <= 1'b0;
          done_o <= 1'b1;
          signature_o <= fold_signature(acc);
        end
      end
    end
  end
endmodule
