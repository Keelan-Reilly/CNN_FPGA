//----------------------------------------------------------------------
// Module: dense
//
// Overview
//   Fully-connected (matrix–vector) layer that computes:
//
//       y[o] = sat( ( B[o] + Σ_{i=0..IN_DIM-1} x[i] * W[o,i] ) >> (FRAC_BITS + POST_SHIFT) )
//
//   The design is intentionally simple: one input read + `DENSE_OUT_PAR` MACs
//   per cycle, driven by a small FSM. Inputs come from a synchronous BRAM-like source
//   with a configurable visibility latency `LAT` cycles from asserting
//   `in_en` to seeing a valid `in_q`.
//
// Data / addressing
//   • Weights are stored flattened by output neuron (row-major):
//       W[o,i] at address (o*IN_DIM + i)
//     A single input activation can be broadcast to multiple output lanes.
//   • Biases are indexed by neuron: B[o].
//
// Timing model (LAT-cycle input)
//   • READ:  drive in_addr, pulse in_en, latch up to `DENSE_OUT_PAR` weights
//   • WAIT:  count down so that `in_q` becomes valid at MAC
//   • MAC :  sample valid `in_q`, multiply by latched w_reg, accumulate
//     Optional repair mode can split this into MAC_MUL then MAC_ACC.
//
// FSM flow
//   IDLE
//     - Wait for `start`. Clear indices (o_base=0,i=0).
//     - Preload the first output-lane batch with bias terms.
//
//   READ
//     - Pulse `in_en` for x[i].
//     - Latch the current input's weights for each active output lane.
//     - Initialise wait counter (LAT-1). If LAT==0, skip WAIT.
//
//   WAIT
//     - Countdown to align with BRAM/testbench latency.
//
//   MAC
//     - Compute one product per active output lane and accumulate.
//     - Either advance i or, if i is last, move to WRITE.
//
//   WRITE
//     - Scale: arithmetic right shift by (FRAC_BITS + POST_SHIFT).
//     - Saturate/clamp each active lane and store into out_vec[o].
//     - Advance to the next output batch, or go to FINISH after the last batch.
//
//   FINISH
//     - Pulse `done` for one cycle and return to IDLE.
//
// Notes
//   • `FRAC_BITS` aligns bias/products into the accumulator domain.
//   • `DENSE_OUT_PAR` controls output-neuron parallelism and defaults to 1.
//   • `POST_SHIFT` provides extra post-accum scaling (e.g., dequant / gain).
//   • Set `LAT` to match the source of `in_q` (TB behaves like LAT=2).
//   • ACCW is sized to avoid overflow growth across IN_DIM taps.
//----------------------------------------------------------------------

(* keep_hierarchy = "yes" *)
module dense #(
  parameter int DATA_WIDTH = 16,
  parameter int FRAC_BITS  = 7,
  parameter int IN_DIM     = 1568,
  parameter int OUT_DIM    = 10,
  parameter int DENSE_OUT_PAR = 1,
  parameter int POST_SHIFT = 0,
  parameter string WEIGHTS_FILE = "fc1_weights.mem",
  parameter string BIASES_FILE  = "fc1_biases.mem",
  parameter int LAT = 1,
  // Dense-local timing experiment knob. Default preserves baseline behavior.
  parameter bit SPLIT_MAC_PIPELINE = 1'b0,

  // ---- debug knobs (non-functional) ----
  parameter bit DBG_ENABLE       = 1,
  parameter int DBG_MAX_PRINTS   = 200,
  parameter int DBG_HEAD_ONLY_O  = 0,
  parameter int DBG_HEAD_ONLY_ON = 0
)(
  input  logic clk,
  input  logic reset,
  input  logic start,

  // Input vector BRAM (LAT-cycle visibility latency)
  output logic [$clog2(IN_DIM)-1:0]     in_addr,
  output logic                          in_en,
  input  logic signed [DATA_WIDTH-1:0]  in_q,

  // Output vector
  output logic signed [DATA_WIDTH-1:0]  out_vec [0:OUT_DIM-1],

  output logic done
);

  // ---------- widths / addressing ----------
  localparam int ACCW    = DATA_WIDTH*2 + $clog2(IN_DIM) + 2;
  localparam int W_DEPTH = OUT_DIM*IN_DIM;
  localparam int AW      = (W_DEPTH<=1)?1:$clog2(W_DEPTH);
  localparam int IN_AW   = (IN_DIM  <=1)?1:$clog2(IN_DIM);
  localparam int PAR     = (DENSE_OUT_PAR < 1) ? 1 : DENSE_OUT_PAR;

  typedef logic [AW-1:0]    w_addr_t;
  typedef logic [IN_AW-1:0] in_addr_t;

  // ---------- ROMs ----------
  (* rom_style="block", ram_style="block" *)
  logic signed [DATA_WIDTH-1:0] W [0:W_DEPTH-1];
  (* rom_style="block", ram_style="block" *)
  logic signed [DATA_WIDTH-1:0] B [0:OUT_DIM-1];

  // ---------- state ----------
  typedef enum logic [2:0] {IDLE, READ, WAIT, MAC, MAC_MUL, MAC_ACC, WRITE, FINISH} state_t;
  state_t state;

  // ---------- indices / accum ----------
  integer o_base, i;
  (* use_dsp = "yes" *) logic signed [ACCW-1:0] acc [0:PAR-1];

  // ---------- weight address/pipe ----------
  logic signed [DATA_WIDTH-1:0]   w_reg [0:PAR-1];
  logic signed [ACCW-1:0]         prod_reg [0:PAR-1];

  // ---------- LAT wait counter ----------
  localparam int WAITW = (LAT <= 1) ? 1 : $clog2(LAT);
  logic [WAITW-1:0] wait_cnt;

  // ---------- saturation helpers ----------
  localparam logic signed [DATA_WIDTH-1:0] S_MAX  = (1 <<< (DATA_WIDTH-1)) - 1;
  localparam logic signed [DATA_WIDTH-1:0] S_MIN  = - (1 <<< (DATA_WIDTH-1));
  localparam logic signed [ACCW-1:0]       S_MAXX = {{(ACCW-DATA_WIDTH){S_MAX[DATA_WIDTH-1]}}, S_MAX};
  localparam logic signed [ACCW-1:0]       S_MINX = {{(ACCW-DATA_WIDTH){S_MIN[DATA_WIDTH-1]}}, S_MIN};

  function automatic logic signed [ACCW-1:0] bias_ext(input logic signed [DATA_WIDTH-1:0] b);
    return $signed({{(ACCW-DATA_WIDTH){b[DATA_WIDTH-1]}}, b}) <<< FRAC_BITS;
  endfunction

  function automatic logic signed [ACCW-1:0] prod_ext(
    input logic signed [DATA_WIDTH-1:0] x,
    input logic signed [DATA_WIDTH-1:0] w
  );
    logic signed [2*DATA_WIDTH-1:0] p;
    begin
      p = $signed(x) * $signed(w);
      prod_ext = {{(ACCW-2*DATA_WIDTH){p[2*DATA_WIDTH-1]}}, p};
    end
  endfunction

  function automatic logic signed [DATA_WIDTH-1:0] clamp_acc(
    input logic signed [ACCW-1:0] value
  );
    logic signed [ACCW-1:0] shifted_value;
    begin
      shifted_value = value >>> (FRAC_BITS + POST_SHIFT);
      if      (shifted_value > S_MAXX) clamp_acc = S_MAX;
      else if (shifted_value < S_MINX) clamp_acc = S_MIN;
      else                             clamp_acc = shifted_value[DATA_WIDTH-1:0];
    end
  endfunction

  function automatic bit lane_active(input int lane_idx, input int out_base);
    return (out_base + lane_idx) < OUT_DIM;
  endfunction

  function automatic w_addr_t weight_addr(input int out_idx, input int in_idx);
    return w_addr_t'((out_idx * IN_DIM) + in_idx);
  endfunction

  // ---------- file load ----------
  initial begin
`ifndef SYNTHESIS
    integer fdw, fdb, k, sumW, sumB;
    fdw=$fopen(WEIGHTS_FILE,"r"); if (fdw==0) $fatal(1,"%m: cannot open weights '%s'",WEIGHTS_FILE); else $fclose(fdw);
    fdb=$fopen(BIASES_FILE ,"r"); if (fdb==0) $fatal(1,"%m: cannot open biases  '%s'",BIASES_FILE ); else $fclose(fdb);
`endif
    $readmemh(WEIGHTS_FILE, W);
    $readmemh(BIASES_FILE , B);
`ifndef SYNTHESIS
    sumW=0; for (k=0;k<$size(W);k++) sumW+=W[k];
    sumB=0; for (k=0;k<$size(B);k++) sumB+=B[k];
    $display("%m: loaded %0d weights, %0d biases; sums: W=%0d B=%0d",
             $size(W), $size(B), sumW, sumB);
`endif
  end

  // ---------- debug infra ----------
  int cyc, dbg_lines;
  always_ff @(posedge clk) begin
    if (reset) cyc <= 0; else cyc <= cyc + 1;
  end
  function automatic bit dbg_ok_neuron();
    if (!DBG_ENABLE) return 0;
    if (!DBG_HEAD_ONLY_ON) return 1;
    return (o_base <= DBG_HEAD_ONLY_O) && (DBG_HEAD_ONLY_O < (o_base + PAR));
  endfunction

  assign in_en = (state == READ);

  `ifndef SYNTHESIS
    bit sim_quiet;
    initial sim_quiet = $test$plusargs("quiet");  // enable with +quiet
  `endif

  // =============================== FSM ===============================
  always_ff @(posedge clk) begin
    if (reset) begin
      state     <= IDLE;
      done      <= 1'b0;

      o_base <= 0;
      i      <= 0;

      in_addr <= '0;

      wait_cnt  <= '0;
      dbg_lines <= 0;

      for (int lane = 0; lane < PAR; lane++) begin
        acc[lane]   <= '0;
        w_reg[lane] <= '0;
        prod_reg[lane] <= '0;
      end

      for (int out_idx = 0; out_idx < OUT_DIM; out_idx++) begin
        out_vec[out_idx] <= '0;
      end

    end else begin
      done  <= 1'b0;

      unique case (state)

        // ------------------------------ IDLE -------------------------
        IDLE: if (start) begin
                o_base   <= 0;
                i        <= 0;

                in_addr  <= in_addr_t'(0);

                for (int lane = 0; lane < PAR; lane++) begin
                  if (lane_active(lane, 0)) acc[lane] <= bias_ext(B[lane]);
                  else                      acc[lane] <= '0;
                end

                if (dbg_ok_neuron() && dbg_lines < DBG_MAX_PRINTS)
                  $display("[%0t][IDLE ] start -> o_base=%0d i=%0d par=%0d in_addr=%0d",
                           $time,0,0,PAR,in_addr);

                state <= READ;
              end

        // ------------------------------ READ -------------------------
        READ: begin
                wait_cnt <= (LAT==0) ? '0 : (LAT-1);

                for (int lane = 0; lane < PAR; lane++) begin
                  if (lane_active(lane, o_base))
                    w_reg[lane] <= W[weight_addr(o_base + lane, i)];
                  else
                    w_reg[lane] <= '0;
                end

                if (dbg_ok_neuron() && dbg_lines < DBG_MAX_PRINTS)
                  $display("[%0t][READ ] o_base=%0d i=%0d in_addr=%0d",
                           $time,o_base,i,in_addr);

                state <= (LAT==0) ? (SPLIT_MAC_PIPELINE ? MAC_MUL : MAC) : WAIT;
              end

        // ------------------------------ WAIT -------------------------
        WAIT: begin
                if (dbg_ok_neuron() && dbg_lines < DBG_MAX_PRINTS)
                  $display("[%0t][WAIT ] o_base=%0d i=%0d  cnt=%0d", $time,o_base,i,wait_cnt);

                if (wait_cnt == 0) state <= SPLIT_MAC_PIPELINE ? MAC_MUL : MAC;
                else               wait_cnt <= wait_cnt - 1;
              end

        // ------------------------------ MAC --------------------------
        MAC: begin
                for (int lane = 0; lane < PAR; lane++) begin
                  if (lane_active(lane, o_base)) begin
                    if (dbg_ok_neuron() && dbg_lines < DBG_MAX_PRINTS)
                      $display("[%0t][MAC  ] o=%0d i=%0d in_q=%0d w=%0d prod=%0d acc_pre=%0d acc_post=%0d",
                               $time,o_base + lane,i,in_q,w_reg[lane],
                               prod_ext(in_q, w_reg[lane]),
                               acc[lane],acc[lane] + prod_ext(in_q, w_reg[lane]));

                    acc[lane] <= acc[lane] + prod_ext(in_q, w_reg[lane]);
                  end
                end

                if (i == IN_DIM-1) begin
                  state <= WRITE;
                end else begin
                  i        <= i + 1;
                  in_addr  <= in_addr_t'(i + 1);
                  state    <= READ;
                end

                if (dbg_ok_neuron() && dbg_lines < DBG_MAX_PRINTS)
                  dbg_lines <= dbg_lines + 1;
              end

        // ----------------------------- MAC_MUL -----------------------
        MAC_MUL: begin
                for (int lane = 0; lane < PAR; lane++) begin
                  if (lane_active(lane, o_base))
                    prod_reg[lane] <= prod_ext(in_q, w_reg[lane]);
                  else
                    prod_reg[lane] <= '0;
                end

                state <= MAC_ACC;
              end

        // ----------------------------- MAC_ACC -----------------------
        MAC_ACC: begin
                for (int lane = 0; lane < PAR; lane++) begin
                  if (lane_active(lane, o_base)) begin
                    if (dbg_ok_neuron() && dbg_lines < DBG_MAX_PRINTS)
                      $display("[%0t][MAC  ] o=%0d i=%0d in_q=%0d w=%0d prod=%0d acc_pre=%0d acc_post=%0d",
                               $time,o_base + lane,i,in_q,w_reg[lane],
                               prod_reg[lane],
                               acc[lane],acc[lane] + prod_reg[lane]);

                    acc[lane] <= acc[lane] + prod_reg[lane];
                  end
                end

                if (i == IN_DIM-1) begin
                  state <= WRITE;
                end else begin
                  i        <= i + 1;
                  in_addr  <= in_addr_t'(i + 1);
                  state    <= READ;
                end

                if (dbg_ok_neuron() && dbg_lines < DBG_MAX_PRINTS)
                  dbg_lines <= dbg_lines + 1;
              end

        // ------------------------------ WRITE ------------------------
        WRITE: begin
                for (int lane = 0; lane < PAR; lane++) begin
                  if (lane_active(lane, o_base)) begin
                    out_vec[o_base + lane] <= clamp_acc(acc[lane]);

                    if (DBG_ENABLE && dbg_lines < DBG_MAX_PRINTS)
                      $display("[%0t][WRITE] o=%0d acc=%0d >>%0d -> res=%0d",
                               $time,o_base + lane,acc[lane],(FRAC_BITS+POST_SHIFT),clamp_acc(acc[lane]));
                  end
                end

                if ((o_base + PAR) >= OUT_DIM) begin
                  state <= FINISH;
                end else begin
                  o_base   <= o_base + PAR;
                  i        <= 0;

                  in_addr  <= in_addr_t'(0);

                  for (int lane = 0; lane < PAR; lane++) begin
                    if (lane_active(lane, o_base + PAR))
                      acc[lane] <= bias_ext(B[o_base + PAR + lane]);
                    else
                      acc[lane] <= '0;
                  end

                  if (DBG_ENABLE && dbg_lines < DBG_MAX_PRINTS)
                    $display("[%0t][NEXT ] o_base->%0d", $time,o_base + PAR);

                  state <= READ;
                end
              end

        // ------------------------------ FINISH -----------------------
        FINISH: begin
          done  <= 1'b1;
        `ifndef SYNTHESIS
          if (DBG_ENABLE && !sim_quiet)
            $display("[%0t][DONE ] dense complete.", $time);
        `endif
          state <= IDLE;
        end

      endcase
    end
  end
endmodule
