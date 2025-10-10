//----------------------------------------------------------------------
// Module: dense
//
// High-level:
//   Fully-connected (matrix–vector) layer with a tiny FSM that streams
//   one input per cycle and performs one MAC per cycle. The input vector
//   lives in a BRAM-like memory with a configurable visibility latency
//   (LAT): after we assert `in_en` in READ, the testbench/BRAM makes
//   `in_q` valid exactly LAT cycles later. We wait those cycles in WAIT,
//   then sample `in_q` in MAC, multiply by the latched weight for the
//   current output neuron, and accumulate. After IN_DIM elements, we
//   scale/saturate and write the result to `out_vec[o]`, then move to
//   the next output neuron.
//
// FSM flow:
//   IDLE  -> seed addresses, preload bias for o=0, jump to READ
//   READ  -> assert in_en for input[i], latch weight W[o,i], preset WAIT
//   WAIT  -> count down LAT-1 cycles so the next cycle sees valid in_q
//   MAC   -> sample in_q, multiply by latched weight, accumulate;
//            queue next addresses or go to WRITE at the end of row
//   WRITE -> scale/saturate acc, store into out_vec[o]; advance o
//   FINISH-> pulse done; return to IDLE
//
// Notes:
//   • Weights are stored linearly by output neuron, i.e. W[o, i] at
//     address o*IN_DIM + i. `w_base` is the base for the current o.
//   • `FRAC_BITS` aligns the bias/products (Q-format), `POST_SHIFT` gives
//     you an extra post-accum scaling (e.g., activation dequant).
//   • Set LAT to match your BRAM/testbench latency. Your TB behaves like
//     LAT=2: READ, WAIT, MAC samples in_q.
//----------------------------------------------------------------------

(* keep_hierarchy = "yes" *)
module dense #(
  parameter int DATA_WIDTH = 16,                   // Q-format data/weights/biases width
  parameter int FRAC_BITS  = 7,                    // bias alignment / product scaling
  parameter int IN_DIM     = 1568,                 // input vector length
  parameter int OUT_DIM    = 10,                   // number of output neurons
  parameter int POST_SHIFT = 2,                    // extra right shift after accumulation
  parameter string WEIGHTS_FILE = "fc1_weights.mem", // flattened W[o,i] = W[o*IN_DIM + i]
  parameter string BIASES_FILE  = "fc1_biases.mem",  // B[o]

  // BRAM visibility latency: cycles from in_en↑ (READ) to valid in_q
  parameter int LAT = 2,                           // your TB: LAT=2

  // ---- debug knobs (non-functional) ----
  parameter bit DBG_ENABLE       = 1,              // enable $display taps
  parameter int DBG_MAX_PRINTS   = 200,            // limit print lines
  parameter int DBG_HEAD_ONLY_O  = 0,              // neuron to focus on
  parameter int DBG_HEAD_ONLY_ON = 0               // 1: print only that neuron
)(
  input  logic clk,                                // clock
  input  logic reset,                              // synchronous reset
  input  logic start,                              // start pulse

  // Input vector BRAM (LAT-cycle visible latency)
  output logic [$clog2(IN_DIM)-1:0]     in_addr,   // input address i
  output logic                          in_en,     // input read strobe
  input  logic signed [DATA_WIDTH-1:0]  in_q,      // input data x[i] (valid after LAT)

  // Output vector (written directly as regs/array)
  output logic signed [DATA_WIDTH-1:0]  out_vec [0:OUT_DIM-1], // y[o]

  output logic done                                  // done pulse when all o complete
);

  // ---------- widths / addressing ----------
  localparam int ACCW = DATA_WIDTH*2 + $clog2(IN_DIM) + 2; // product width + growth + margin

  localparam int W_DEPTH = OUT_DIM*IN_DIM;                 // total weights
  localparam int AW      = (W_DEPTH<=1)?1:$clog2(W_DEPTH); // weight address width
  localparam int IN_AW   = (IN_DIM  <=1)?1:$clog2(IN_DIM); // input address width

  typedef logic [AW-1:0]    w_addr_t;                      // weight addr type
  typedef logic [IN_AW-1:0] in_addr_t;                     // input addr type

  // ---------- ROMs ----------
  (* rom_style="block", ram_style="block" *)
  logic signed [DATA_WIDTH-1:0] W [0:W_DEPTH-1];           // weight ROM
  (* rom_style="block", ram_style="block" *)
  logic signed [DATA_WIDTH-1:0] B [0:OUT_DIM-1];           // bias ROM

  // ---------- state ----------
  typedef enum logic [2:0] {IDLE, READ, WAIT, MAC, WRITE, FINISH} state_t;
  state_t state;                                           // FSM state register

  // ---------- indices / accum ----------
  integer o, i;                                            // o = neuron index, i = input index
  (* use_dsp = "yes" *) logic signed [ACCW-1:0]  acc;      // running sum for y[o]

  // ---------- weight address/pipe ----------
  w_addr_t                         w_base;                 // base = o*IN_DIM
  w_addr_t                         w_addr_q;               // queued addr = w_base + i
  logic   signed [DATA_WIDTH-1:0]  w_reg;                  // latched W[o,i] for MAC

  // ---------- LAT wait counter ----------
  logic [$clog2((LAT>0)?LAT:1)-1:0] wait_cnt;              // counts LAT-1 .. 0

  // ---------- saturation helpers ----------
  localparam logic signed [DATA_WIDTH-1:0] S_MAX  = (1 <<< (DATA_WIDTH-1)) - 1; // +max
  localparam logic signed [DATA_WIDTH-1:0] S_MIN  = - (1 <<< (DATA_WIDTH-1));   // -min
  localparam logic signed [ACCW-1:0]       S_MAXX = {{(ACCW-DATA_WIDTH){S_MAX[DATA_WIDTH-1]}}, S_MAX}; // extended
  localparam logic signed [ACCW-1:0]       S_MINX = {{(ACCW-DATA_WIDTH){S_MIN[DATA_WIDTH-1]}}, S_MIN}; // extended

  function automatic logic signed [ACCW-1:0] bias_ext(      // sign-extend bias and align by FRAC_BITS
    input logic signed [DATA_WIDTH-1:0] b
  );
    return $signed({{(ACCW-DATA_WIDTH){b[DATA_WIDTH-1]}}, b}) <<< FRAC_BITS;
  endfunction

  // ---------- file load ----------
  initial begin
  `ifndef SYNTHESIS
    integer fdw, fdb, k, sumW, sumB;                       // sim-time checks & checksum
    fdw=$fopen(WEIGHTS_FILE,"r"); if (fdw==0) $fatal(1,"%m: cannot open weights '%s'",WEIGHTS_FILE); else $fclose(fdw);
    fdb=$fopen(BIASES_FILE ,"r"); if (fdb==0) $fatal(1,"%m: cannot open biases  '%s'",BIASES_FILE ); else $fclose(fdb);
  `endif
    $readmemh(WEIGHTS_FILE, W);                             // load W
    $readmemh(BIASES_FILE , B);                             // load B
  `ifndef SYNTHESIS
    sumW=0; for (k=0;k<$size(W);k++) sumW+=W[k];            // tiny checksum for visibility
    sumB=0; for (k=0;k<$size(B);k++) sumB+=B[k];
    $display("%m: loaded %0d weights, %0d biases; sums: W=%0d B=%0d",
             $size(W), $size(B), sumW, sumB);
  `endif
  end

  // ---------- debug infra ----------
  int cyc, dbg_lines;                                       // cycle counter + line limiter
  always_ff @(posedge clk) begin
    if (reset) cyc <= 0; else cyc <= cyc + 1;
  end
  function automatic bit dbg_ok_neuron();                   // decide whether to print for this o
    if (!DBG_ENABLE) return 0;
    if (!DBG_HEAD_ONLY_ON) return 1;
    return (o == DBG_HEAD_ONLY_O);
  endfunction

  // =============================== FSM ===============================
  always_ff @(posedge clk) begin
    if (reset) begin
      // ---- reset all state/outputs ----
      state   <= IDLE; done <= 1'b0;                        // idle + clear done
      o <= 0; i <= 0; acc <= '0;                            // clear indices/acc
      in_en   <= 1'b0; in_addr <= '0;                       // input port idle
      w_base  <= '0;   w_addr_q <= '0;                      // weight addressing clear
      w_reg   <= '0;                                        // latched weight clear
      wait_cnt<= '0;                                        // counter clear
      dbg_lines <= 0;                                       // debug budget reset
    end else begin
      done  <= 1'b0;                                        // default: not done
      in_en <= 1'b0;                                        // in_en only high in READ

      unique case (state)
        // ------------------------------ IDLE -------------------------
        IDLE: if (start) begin
                o        <= 0;                              // first neuron
                i        <= 0;                              // first input index
                acc      <= bias_ext(B[0]);                 // preload bias for o=0
                in_addr  <= in_addr_t'(0);                  // queue addr for i=0
                w_base   <= w_addr_t'(0);                   // base = 0*IN_DIM
                w_addr_q <= w_addr_t'(0);                   // queued W addr = base + i
                if (dbg_ok_neuron() && dbg_lines<DBG_MAX_PRINTS)
                  $display("[%0t][IDLE ] start -> o=%0d i=%0d bias=%0d (acc=%0d) in_addr=%0d w_addr=%0d",
                           $time,o,i,B[0],bias_ext(B[0]),in_addr,w_addr_q);
                state <= READ;                              // proceed to READ
              end

        // ------------------------------ READ -------------------------
        READ: begin
                in_en  <= 1'b1;                             // assert read for x[i]
                w_reg  <= W[w_addr_q];                      // latch W[o,i] now
                wait_cnt <= (LAT==0)? '0 : (LAT-1);         // preset wait: LAT-1 cycles
                if (dbg_ok_neuron() && dbg_lines<DBG_MAX_PRINTS)
                  $display("[%0t][READ ] o=%0d i=%0d  in_addr=%0d  w_addr=%0d  (W=%0d)",
                           $time,o,i,in_addr,w_addr_q,W[w_addr_q]);
                state <= (LAT==0) ? MAC : WAIT;             // skip WAIT if no latency
              end

        // ------------------------------ WAIT -------------------------
        WAIT: begin
                if (dbg_ok_neuron() && dbg_lines<DBG_MAX_PRINTS)
                  $display("[%0t][WAIT ] o=%0d i=%0d  cnt=%0d", $time,o,i,wait_cnt);
                if (wait_cnt==0) begin
                  state <= MAC;                              // next cycle: in_q is valid
                end else begin
                  wait_cnt <= wait_cnt - 1;                  // count down
                end
              end

        // ------------------------------- MAC -------------------------
        MAC: begin
                // Multiply the *now-valid* in_q by the latched weight and accumulate.
                automatic logic signed [2*DATA_WIDTH-1:0] p;    // raw product
                automatic logic signed [ACCW-1:0]         acc_next; // widened sum
                p        = $signed(in_q) * $signed(w_reg);       // x[i] * W[o,i]
                acc_next = acc + {{(ACCW-2*DATA_WIDTH){p[2*DATA_WIDTH-1]}}, p}; // sign-extend add
                if (dbg_ok_neuron() && dbg_lines<DBG_MAX_PRINTS)
                  $display("[%0t][MAC  ] o=%0d i=%0d  in_q=%0d  w=%0d  prod=%0d  acc_pre=%0d  acc_post=%0d",
                           $time,o,i,in_q,w_reg,p,acc,acc_next);
                acc <= acc_next;                               // commit accumulation

                if (i == IN_DIM-1) begin                       // finished this neuron’s row?
                  state <= WRITE;                              // go write y[o]
                end else begin
                  // Queue next element addresses for the next READ.
                  i        <= i + 1;                           // advance input index
                  in_addr  <= in_addr_t'(i + 1);               // next x[i+1]
                  w_addr_q <= w_base + w_addr_t'(i + 1);       // next W[o,i+1]
                  state    <= READ;                            // back to READ
                end
                if (dbg_ok_neuron() && dbg_lines<DBG_MAX_PRINTS) dbg_lines <= dbg_lines + 1;
              end

        // ------------------------------ WRITE ------------------------
        WRITE: begin
                // Scale and saturate acc -> out_vec[o].
                logic signed [ACCW-1:0]       shifted;         // scaled accumulator
                logic signed [DATA_WIDTH-1:0] res;             // final clamped result
                shifted = acc >>> (FRAC_BITS + POST_SHIFT);    // total arithmetic right shift
                if      (shifted > S_MAXX) res = S_MAX;        // saturate high
                else if (shifted < S_MINX) res = S_MIN;        // saturate low
                else                        res = shifted[DATA_WIDTH-1:0]; // in-range result
                out_vec[o] <= res;                              // store y[o]
                if (DBG_ENABLE && dbg_lines<DBG_MAX_PRINTS)
                  $display("[%0t][WRITE] o=%0d  acc=%0d  >>%0d -> res=%0d",
                           $time,o,acc,(FRAC_BITS+POST_SHIFT),res);

                if (o == OUT_DIM-1) begin                       // last neuron?
                  state <= FINISH;                              // all done
                end else begin
                  // Advance to the next neuron and prime addresses/bias.
                  o        <= o + 1;                            // next o
                  i        <= 0;                                // reset i
                  acc      <= bias_ext(B[o+1]);                 // preload bias for new o
                  in_addr  <= in_addr_t'(0);                    // start from x[0]
                  w_base   <= w_base + w_addr_t'(IN_DIM);       // base += IN_DIM
                  w_addr_q <= w_base + w_addr_t'(IN_DIM);       // queued addr = new base
                  if (DBG_ENABLE && dbg_lines<DBG_MAX_PRINTS)
                    $display("[%0t][NEXT ] o->%0d  preload bias=%0d  acc=%0d  w_base=%0d",
                             $time,o+1,B[o+1],bias_ext(B[o+1]),w_base + w_addr_t'(IN_DIM));
                  state    <= READ;                             // loop back
                end
              end

        // ------------------------------ FINISH -----------------------
        FINISH: begin
                 done  <= 1'b1;                                 // pulse done
                 if (DBG_ENABLE)
                   $display("[%0t][DONE ] dense complete.", $time);
                 state <= IDLE;                                  // await next start
               end
      endcase
    end
  end
endmodule
