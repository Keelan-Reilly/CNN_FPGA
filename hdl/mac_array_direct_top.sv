//----------------------------------------------------------------------
// Directly measurable MAC-array slice.
//
// This top is intentionally small and isolated from the CNN pipeline.
// It implements a parameterized ARRAY_ROWS x ARRAY_COLS logical MAC array
// with three small architecture modes:
//   - baseline: one MAC-style cell per logical grid position
//   - shared_lut_saving: one DSP-backed multiplier shared across each adjacent
//     column pair
//   - shared_dsp_reducing: one LUT-mapped multiplier shared across each
//     adjacent column pair
// All modes run the same fixed synthetic K-depth workload so the existing
// Vivado flow can measure small, honest baseline-vs-shared tradeoffs on real
// MAC-array-style RTL without requiring the full architecture family.
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
module mac_array_shared_pair_cell #(
  parameter int DATA_WIDTH = 16,
  parameter int ACC_WIDTH = 40,
  parameter int EVEN_CELL_INDEX = 0,
  parameter bit ODD_CELL_VALID = 1'b1,
  parameter int ODD_CELL_INDEX = 1
)(
  input  logic clk,
  input  logic reset,
  input  logic load_i,
  input  logic step_i,
  input  logic phase_i,
  input  logic signed [DATA_WIDTH-1:0] even_seed_a_i,
  input  logic signed [DATA_WIDTH-1:0] even_seed_b_i,
  input  logic signed [DATA_WIDTH-1:0] odd_seed_a_i,
  input  logic signed [DATA_WIDTH-1:0] odd_seed_b_i,
  output logic signed [ACC_WIDTH-1:0] even_acc_o,
  output logic signed [ACC_WIDTH-1:0] odd_acc_o
);
  logic signed [DATA_WIDTH-1:0] even_a_reg;
  logic signed [DATA_WIDTH-1:0] even_b_reg;
  logic signed [DATA_WIDTH-1:0] odd_a_reg;
  logic signed [DATA_WIDTH-1:0] odd_b_reg;
  logic signed [ACC_WIDTH-1:0] even_acc_reg;
  logic signed [ACC_WIDTH-1:0] odd_acc_reg;
  logic signed [DATA_WIDTH-1:0] shared_mul_a;
  logic signed [DATA_WIDTH-1:0] shared_mul_b;
  (* use_dsp = "yes" *) logic signed [ACC_WIDTH-1:0] shared_mul_result;

  localparam logic signed [DATA_WIDTH-1:0] EVEN_A_INC = DATA_WIDTH'(EVEN_CELL_INDEX + 1);
  localparam logic signed [DATA_WIDTH-1:0] EVEN_B_DEC = DATA_WIDTH'((EVEN_CELL_INDEX % 3) + 1);
  localparam logic signed [DATA_WIDTH-1:0] ODD_A_INC = DATA_WIDTH'(ODD_CELL_INDEX + 1);
  localparam logic signed [DATA_WIDTH-1:0] ODD_B_DEC = DATA_WIDTH'((ODD_CELL_INDEX % 3) + 1);

  assign even_acc_o = even_acc_reg;
  assign odd_acc_o = odd_acc_reg;
  assign shared_mul_a = phase_i ? odd_a_reg : even_a_reg;
  assign shared_mul_b = phase_i ? odd_b_reg : even_b_reg;
  assign shared_mul_result = $signed(shared_mul_a) * $signed(shared_mul_b);

  always_ff @(posedge clk) begin
    if (reset) begin
      even_a_reg <= '0;
      even_b_reg <= '0;
      odd_a_reg <= '0;
      odd_b_reg <= '0;
      even_acc_reg <= '0;
      odd_acc_reg <= '0;
    end else if (load_i) begin
      even_a_reg <= even_seed_a_i;
      even_b_reg <= even_seed_b_i;
      odd_a_reg <= odd_seed_a_i;
      odd_b_reg <= odd_seed_b_i;
      even_acc_reg <= '0;
      odd_acc_reg <= '0;
    end else if (step_i) begin
      if (!phase_i) begin
        even_acc_reg <= even_acc_reg + shared_mul_result;
        even_a_reg <= even_a_reg + EVEN_A_INC;
        even_b_reg <= even_b_reg - EVEN_B_DEC;
      end else if (ODD_CELL_VALID) begin
        odd_acc_reg <= odd_acc_reg + shared_mul_result;
        odd_a_reg <= odd_a_reg + ODD_A_INC;
        odd_b_reg <= odd_b_reg - ODD_B_DEC;
      end
    end
  end
endmodule

(* keep_hierarchy = "yes" *)
module mac_array_shared_pair_lutmul_cell #(
  parameter int DATA_WIDTH = 16,
  parameter int ACC_WIDTH = 40,
  parameter int EVEN_CELL_INDEX = 0,
  parameter bit ODD_CELL_VALID = 1'b1,
  parameter int ODD_CELL_INDEX = 1
)(
  input  logic clk,
  input  logic reset,
  input  logic load_i,
  input  logic step_i,
  input  logic phase_i,
  input  logic signed [DATA_WIDTH-1:0] even_seed_a_i,
  input  logic signed [DATA_WIDTH-1:0] even_seed_b_i,
  input  logic signed [DATA_WIDTH-1:0] odd_seed_a_i,
  input  logic signed [DATA_WIDTH-1:0] odd_seed_b_i,
  output logic signed [ACC_WIDTH-1:0] even_acc_o,
  output logic signed [ACC_WIDTH-1:0] odd_acc_o
);
  logic signed [DATA_WIDTH-1:0] even_a_reg;
  logic signed [DATA_WIDTH-1:0] even_b_reg;
  logic signed [DATA_WIDTH-1:0] odd_a_reg;
  logic signed [DATA_WIDTH-1:0] odd_b_reg;
  logic signed [ACC_WIDTH-1:0] even_acc_reg;
  logic signed [ACC_WIDTH-1:0] odd_acc_reg;
  logic signed [DATA_WIDTH-1:0] shared_mul_a;
  logic signed [DATA_WIDTH-1:0] shared_mul_b;
  (* use_dsp = "no" *) logic signed [ACC_WIDTH-1:0] shared_mul_result;

  localparam logic signed [DATA_WIDTH-1:0] EVEN_A_INC = DATA_WIDTH'(EVEN_CELL_INDEX + 1);
  localparam logic signed [DATA_WIDTH-1:0] EVEN_B_DEC = DATA_WIDTH'((EVEN_CELL_INDEX % 3) + 1);
  localparam logic signed [DATA_WIDTH-1:0] ODD_A_INC = DATA_WIDTH'(ODD_CELL_INDEX + 1);
  localparam logic signed [DATA_WIDTH-1:0] ODD_B_DEC = DATA_WIDTH'((ODD_CELL_INDEX % 3) + 1);

  function automatic logic signed [ACC_WIDTH-1:0] lut_multiply(
    input logic signed [DATA_WIDTH-1:0] lhs,
    input logic signed [DATA_WIDTH-1:0] rhs
  );
    logic signed [DATA_WIDTH:0] lhs_ext;
    logic signed [DATA_WIDTH:0] rhs_ext;
    logic [DATA_WIDTH:0] lhs_mag;
    logic [DATA_WIDTH:0] rhs_mag;
    logic [ACC_WIDTH-1:0] accum;
    begin
      lhs_ext = {lhs[DATA_WIDTH-1], lhs};
      rhs_ext = {rhs[DATA_WIDTH-1], rhs};
      if (lhs_ext[DATA_WIDTH]) begin
        lhs_mag = -lhs_ext;
      end else begin
        lhs_mag = lhs_ext;
      end
      if (rhs_ext[DATA_WIDTH]) begin
        rhs_mag = -rhs_ext;
      end else begin
        rhs_mag = rhs_ext;
      end
      accum = '0;
      for (int bit_idx = 0; bit_idx <= DATA_WIDTH; bit_idx++) begin
        if (rhs_mag[bit_idx]) begin
          accum = accum + ({{(ACC_WIDTH-(DATA_WIDTH+1)){1'b0}}, lhs_mag} <<< bit_idx);
        end
      end
      if (lhs_ext[DATA_WIDTH] ^ rhs_ext[DATA_WIDTH]) begin
        lut_multiply = -$signed(accum);
      end else begin
        lut_multiply = $signed(accum);
      end
    end
  endfunction

  assign even_acc_o = even_acc_reg;
  assign odd_acc_o = odd_acc_reg;
  assign shared_mul_a = phase_i ? odd_a_reg : even_a_reg;
  assign shared_mul_b = phase_i ? odd_b_reg : even_b_reg;
  assign shared_mul_result = lut_multiply(shared_mul_a, shared_mul_b);

  always_ff @(posedge clk) begin
    if (reset) begin
      even_a_reg <= '0;
      even_b_reg <= '0;
      odd_a_reg <= '0;
      odd_b_reg <= '0;
      even_acc_reg <= '0;
      odd_acc_reg <= '0;
    end else if (load_i) begin
      even_a_reg <= even_seed_a_i;
      even_b_reg <= even_seed_b_i;
      odd_a_reg <= odd_seed_a_i;
      odd_b_reg <= odd_seed_b_i;
      even_acc_reg <= '0;
      odd_acc_reg <= '0;
    end else if (step_i) begin
      if (!phase_i) begin
        even_acc_reg <= even_acc_reg + shared_mul_result;
        even_a_reg <= even_a_reg + EVEN_A_INC;
        even_b_reg <= even_b_reg - EVEN_B_DEC;
      end else if (ODD_CELL_VALID) begin
        odd_acc_reg <= odd_acc_reg + shared_mul_result;
        odd_a_reg <= odd_a_reg + ODD_A_INC;
        odd_b_reg <= odd_b_reg - ODD_B_DEC;
      end
    end
  end
endmodule

(* keep_hierarchy = "yes" *)
module mac_array_direct_top #(
  parameter int DATA_WIDTH = 16,
  parameter int ARRAY_ROWS = 4,
  parameter int ARRAY_COLS = 4,
  parameter int K_DEPTH = 16,
  parameter int CLK_FREQ_HZ = 100_000_000,
  parameter int ARCH_MODE = 0
)(
  input  logic clk,
  input  logic reset,
  input  logic start_i,
  output logic done_o,
  output logic [3:0] signature_o
);
  localparam int ARCH_MODE_BASELINE = 0;
  localparam int ARCH_MODE_SHARED_LUT_SAVING = 1;
  localparam int ARCH_MODE_SHARED_DSP_REDUCING = 2;
  localparam bit USE_SHARED = (ARCH_MODE != ARCH_MODE_BASELINE);
  localparam bit USE_LUT_MULTIPLIER = (ARCH_MODE == ARCH_MODE_SHARED_DSP_REDUCING);
  localparam int CELL_COUNT = ARRAY_ROWS * ARRAY_COLS;
  localparam int SHARED_COL_GROUPS = (ARRAY_COLS + 1) / 2;
  localparam int SHARED_PHYSICAL_CELL_COUNT = ARRAY_ROWS * SHARED_COL_GROUPS;
  localparam int WORK_CYCLES = USE_SHARED ? (2 * K_DEPTH) : K_DEPTH;
  localparam int K_COUNTER_WIDTH = (WORK_CYCLES <= 1) ? 1 : $clog2(WORK_CYCLES + 1);
  localparam int ACC_WIDTH = (2 * DATA_WIDTH) + ((K_DEPTH <= 1) ? 1 : $clog2(K_DEPTH)) + 4;

  logic active;
  logic load_cells;
  logic step_cells;
  logic [K_COUNTER_WIDTH-1:0] k_count;
  logic shared_phase;
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

  assign shared_phase = k_count[0];

  genvar cell_idx;
  generate
    if (!USE_SHARED) begin : gen_baseline_cells
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
    end else if (!USE_LUT_MULTIPLIER) begin : gen_shared_dsp_cells
      for (cell_idx = 0; cell_idx < SHARED_PHYSICAL_CELL_COUNT; cell_idx++) begin : gen_pairs
        localparam int ROW_IDX = cell_idx / SHARED_COL_GROUPS;
        localparam int COL_PAIR_IDX = cell_idx % SHARED_COL_GROUPS;
        localparam int EVEN_LOGICAL_IDX = (ROW_IDX * ARRAY_COLS) + (2 * COL_PAIR_IDX);
        localparam int ODD_LOGICAL_IDX = EVEN_LOGICAL_IDX + 1;
        localparam bit ODD_VALID = ((2 * COL_PAIR_IDX) + 1) < ARRAY_COLS;

        logic signed [ACC_WIDTH-1:0] even_acc;
        logic signed [ACC_WIDTH-1:0] odd_acc;

        mac_array_shared_pair_cell #(
          .DATA_WIDTH(DATA_WIDTH),
          .ACC_WIDTH(ACC_WIDTH),
          .EVEN_CELL_INDEX(EVEN_LOGICAL_IDX),
          .ODD_CELL_VALID(ODD_VALID),
          .ODD_CELL_INDEX(ODD_LOGICAL_IDX)
        ) u_pair_cell (
          .clk(clk),
          .reset(reset),
          .load_i(load_cells),
          .step_i(step_cells),
          .phase_i(shared_phase),
          .even_seed_a_i(init_a(EVEN_LOGICAL_IDX)),
          .even_seed_b_i(init_b(EVEN_LOGICAL_IDX)),
          .odd_seed_a_i(init_a(ODD_LOGICAL_IDX)),
          .odd_seed_b_i(init_b(ODD_LOGICAL_IDX)),
          .even_acc_o(even_acc),
          .odd_acc_o(odd_acc)
        );

        assign acc[EVEN_LOGICAL_IDX] = even_acc;
        if (ODD_VALID) begin : gen_valid_odd
          assign acc[ODD_LOGICAL_IDX] = odd_acc;
        end
      end
    end else begin : gen_shared_lutmul_cells
      for (cell_idx = 0; cell_idx < SHARED_PHYSICAL_CELL_COUNT; cell_idx++) begin : gen_pairs
        localparam int ROW_IDX = cell_idx / SHARED_COL_GROUPS;
        localparam int COL_PAIR_IDX = cell_idx % SHARED_COL_GROUPS;
        localparam int EVEN_LOGICAL_IDX = (ROW_IDX * ARRAY_COLS) + (2 * COL_PAIR_IDX);
        localparam int ODD_LOGICAL_IDX = EVEN_LOGICAL_IDX + 1;
        localparam bit ODD_VALID = ((2 * COL_PAIR_IDX) + 1) < ARRAY_COLS;

        logic signed [ACC_WIDTH-1:0] even_acc;
        logic signed [ACC_WIDTH-1:0] odd_acc;

        mac_array_shared_pair_lutmul_cell #(
          .DATA_WIDTH(DATA_WIDTH),
          .ACC_WIDTH(ACC_WIDTH),
          .EVEN_CELL_INDEX(EVEN_LOGICAL_IDX),
          .ODD_CELL_VALID(ODD_VALID),
          .ODD_CELL_INDEX(ODD_LOGICAL_IDX)
        ) u_pair_cell (
          .clk(clk),
          .reset(reset),
          .load_i(load_cells),
          .step_i(step_cells),
          .phase_i(shared_phase),
          .even_seed_a_i(init_a(EVEN_LOGICAL_IDX)),
          .even_seed_b_i(init_b(EVEN_LOGICAL_IDX)),
          .odd_seed_a_i(init_a(ODD_LOGICAL_IDX)),
          .odd_seed_b_i(init_b(ODD_LOGICAL_IDX)),
          .even_acc_o(even_acc),
          .odd_acc_o(odd_acc)
        );

        assign acc[EVEN_LOGICAL_IDX] = even_acc;
        if (ODD_VALID) begin : gen_valid_odd
          assign acc[ODD_LOGICAL_IDX] = odd_acc;
        end
      end
    end
  endgenerate

  assign load_cells = start_i && !active;
  assign step_cells = active && (k_count < K_COUNTER_WIDTH'(WORK_CYCLES));

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
        if (k_count < K_COUNTER_WIDTH'(WORK_CYCLES)) begin
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
