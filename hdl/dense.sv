//----------------------------------------------------------------------
// Dense with parameterized input BRAM latency + concise debug taps
//   - Assert in_en in READ
//   - Wait LAT cycles (to match TB BRAM latency)
//   - MAC samples raw in_q exactly when valid
//----------------------------------------------------------------------
(* keep_hierarchy = "yes" *)
module dense #(
  parameter int DATA_WIDTH = 16,
  parameter int FRAC_BITS  = 7,
  parameter int IN_DIM     = 1568,
  parameter int OUT_DIM    = 10,
  parameter int POST_SHIFT = 2,
  parameter string WEIGHTS_FILE = "fc1_weights.mem",
  parameter string BIASES_FILE  = "fc1_biases.mem",

  // BRAM visibility latency (cycles) from in_en↑ to in_q valid
  // Your TB behaves like 2 cycles; set LAT=2 (default).
  parameter int LAT = 2,

  // ---- debug knobs ----
  parameter bit DBG_ENABLE       = 1,
  parameter int DBG_MAX_PRINTS   = 200,
  parameter int DBG_HEAD_ONLY_O  = 0,
  parameter int DBG_HEAD_ONLY_ON = 0
)(
  input  logic clk,
  input  logic reset,
  input  logic start,

  // Input vector BRAM (LAT-cycle visible latency)
  output logic [$clog2(IN_DIM)-1:0]     in_addr,
  output logic                          in_en,
  input  logic signed [DATA_WIDTH-1:0]  in_q,

  // Output vector
  output logic signed [DATA_WIDTH-1:0]  out_vec [0:OUT_DIM-1],

  output logic done
);

  // ---------- widths / addrs ----------
  localparam int ACCW = DATA_WIDTH*2 + $clog2(IN_DIM) + 2;

  localparam int W_DEPTH = OUT_DIM*IN_DIM;
  localparam int AW      = (W_DEPTH<=1)?1:$clog2(W_DEPTH);
  localparam int IN_AW   = (IN_DIM  <=1)?1:$clog2(IN_DIM);

  typedef logic [AW-1:0]    w_addr_t;
  typedef logic [IN_AW-1:0] in_addr_t;

  // ---------- ROMs ----------
  (* rom_style="block", ram_style="block" *)
  logic signed [DATA_WIDTH-1:0] W [0:W_DEPTH-1];
  (* rom_style="block", ram_style="block" *)
  logic signed [DATA_WIDTH-1:0] B [0:OUT_DIM-1];

  // ---------- state ----------
  typedef enum logic [2:0] {IDLE, READ, WAIT, MAC, WRITE, FINISH} state_t;
  state_t state;

  // ---------- indices / regs ----------
  integer o, i;                                  // o = output neuron, i = input index
  (* use_dsp = "yes" *) logic signed [ACCW-1:0]  acc;

  // weight pipeline (comb -> w_reg)
  w_addr_t                         w_base, w_addr_q;
  logic   signed [DATA_WIDTH-1:0]  w_reg;

  // wait counter for LAT cycles
  logic [$clog2((LAT>0)?LAT:1)-1:0] wait_cnt;

  // ---------- sat helpers ----------
  localparam logic signed [DATA_WIDTH-1:0] S_MAX  = (1 <<< (DATA_WIDTH-1)) - 1;
  localparam logic signed [DATA_WIDTH-1:0] S_MIN  = - (1 <<< (DATA_WIDTH-1));
  localparam logic signed [ACCW-1:0]       S_MAXX = {{(ACCW-DATA_WIDTH){S_MAX[DATA_WIDTH-1]}}, S_MAX};
  localparam logic signed [ACCW-1:0]       S_MINX = {{(ACCW-DATA_WIDTH){S_MIN[DATA_WIDTH-1]}}, S_MIN};

  function automatic logic signed [ACCW-1:0] bias_ext(input logic signed [DATA_WIDTH-1:0] b);
    return $signed({{(ACCW-DATA_WIDTH){b[DATA_WIDTH-1]}}, b}) <<< FRAC_BITS;
  endfunction

  // ---------- load weights/bias ----------
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

  // ---------- debug ----------
  int cyc, dbg_lines;
  always_ff @(posedge clk) begin
    if (reset) cyc <= 0; else cyc <= cyc + 1;
  end
  function automatic bit dbg_ok_neuron();
    if (!DBG_ENABLE) return 0;
    if (!DBG_HEAD_ONLY_ON) return 1;
    return (o == DBG_HEAD_ONLY_O);
  endfunction

  // ---------- FSM ----------
  always_ff @(posedge clk) begin
    if (reset) begin
      state   <= IDLE; done <= 1'b0;
      o <= 0; i <= 0; acc <= '0;
      in_en   <= 1'b0; in_addr <= '0;
      w_base  <= '0;   w_addr_q <= '0;
      w_reg   <= '0;
      wait_cnt<= '0;
      dbg_lines <= 0;
    end else begin
      done  <= 1'b0;
      in_en <= 1'b0; // default low unless in READ

      unique case (state)
        // Prime first element addresses
        IDLE: if (start) begin
                o        <= 0;
                i        <= 0;
                acc      <= bias_ext(B[0]);
                in_addr  <= in_addr_t'(0);
                w_base   <= w_addr_t'(0);
                w_addr_q <= w_addr_t'(0);
                if (dbg_ok_neuron() && dbg_lines<DBG_MAX_PRINTS)
                  $display("[%0t][IDLE ] start -> o=%0d i=%0d bias=%0d (acc=%0d) in_addr=%0d w_addr=%0d",
                           $time,o,i,B[0],bias_ext(B[0]),in_addr,w_addr_q);
                state <= READ;
              end

        // Issue read: assert in_en (BRAM will present in_q after LAT cycles), latch weight
        READ: begin
                in_en  <= 1'b1;
                w_reg  <= W[w_addr_q];
                wait_cnt <= (LAT==0)? '0 : (LAT-1);
                if (dbg_ok_neuron() && dbg_lines<DBG_MAX_PRINTS)
                  $display("[%0t][READ ] o=%0d i=%0d  in_addr=%0d  w_addr=%0d  (W=%0d)",
                           $time,o,i,in_addr,w_addr_q,W[w_addr_q]);
                state <= (LAT==0) ? MAC : WAIT;
              end

        // Wait exactly LAT-1 cycles so that on the next cycle in_q is valid
        WAIT: begin
                if (dbg_ok_neuron() && dbg_lines<DBG_MAX_PRINTS)
                  $display("[%0t][WAIT ] o=%0d i=%0d  cnt=%0d", $time,o,i,wait_cnt);
                if (wait_cnt==0) begin
                  state <= MAC;
                end else begin
                  wait_cnt <= wait_cnt - 1;
                end
              end

        // Multiply-accumulate; sample raw in_q now that it is valid
        MAC: begin
                automatic logic signed [2*DATA_WIDTH-1:0] p;
                automatic logic signed [ACCW-1:0]         acc_next;
                p        = $signed(in_q) * $signed(w_reg);
                acc_next = acc + {{(ACCW-2*DATA_WIDTH){p[2*DATA_WIDTH-1]}}, p};
                if (dbg_ok_neuron() && dbg_lines<DBG_MAX_PRINTS)
                  $display("[%0t][MAC  ] o=%0d i=%0d  in_q=%0d  w=%0d  prod=%0d  acc_pre=%0d  acc_post=%0d",
                           $time,o,i,in_q,w_reg,p,acc,acc_next);
                acc <= acc_next;

                if (i == IN_DIM-1) begin
                  state <= WRITE;
                end else begin
                  // queue next addresses (for next READ)
                  i        <= i + 1;
                  in_addr  <= in_addr_t'(i + 1);
                  w_addr_q <= w_base + w_addr_t'(i + 1);
                  state    <= READ;
                end
                if (dbg_ok_neuron() && dbg_lines<DBG_MAX_PRINTS) dbg_lines <= dbg_lines + 1;
              end

        // Output store
        WRITE: begin
                logic signed [ACCW-1:0]       shifted;
                logic signed [DATA_WIDTH-1:0] res;
                shifted = acc >>> (FRAC_BITS + POST_SHIFT);
                if      (shifted > S_MAXX) res = S_MAX;
                else if (shifted < S_MINX) res = S_MIN;
                else                        res = shifted[DATA_WIDTH-1:0];
                out_vec[o] <= res;
                if (DBG_ENABLE && dbg_lines<DBG_MAX_PRINTS)
                  $display("[%0t][WRITE] o=%0d  acc=%0d  >>%0d -> res=%0d",
                           $time,o,acc,(FRAC_BITS+POST_SHIFT),res);

                if (o == OUT_DIM-1) begin
                  state <= FINISH;
                end else begin
                  // advance to next neuron; prime first element’s addresses
                  o        <= o + 1;
                  i        <= 0;
                  acc      <= bias_ext(B[o+1]);
                  in_addr  <= in_addr_t'(0);
                  w_base   <= w_base + w_addr_t'(IN_DIM);
                  w_addr_q <= w_base + w_addr_t'(IN_DIM);
                  if (DBG_ENABLE && dbg_lines<DBG_MAX_PRINTS)
                    $display("[%0t][NEXT ] o->%0d  preload bias=%0d  acc=%0d  w_base=%0d",
                             $time,o+1,B[o+1],bias_ext(B[o+1]),w_base + w_addr_t'(IN_DIM));
                  state    <= READ;
                end
              end

        FINISH: begin
                 done  <= 1'b1;
                 if (DBG_ENABLE)
                   $display("[%0t][DONE ] dense complete.", $time);
                 state <= IDLE;
               end
      endcase
    end
  end
endmodule
