//======================================================================
// Module: conv2d
//
// Overview
//   Streaming 2-D convolution (SAME padding) over a CHW input feature map,
//   producing a CHW output feature map. The design performs one tap MAC
//   per cycle using a small FSM and an explicit pipeline that aligns:
//
//     • 1R synchronous BRAM read data (if_q)
//     • current tap weight (W_rom)
//     • per-tap validity (padding mask)
//
//   The implementation is written to be timing-robust on FPGA BRAMs with
//   1-cycle read latency and to avoid off-by-one errors at the final tap.
//
// Data layout
//   • IFMAP: CHW, linear index = (ic*H + r)*W + c
//   • OFMAP: CHW, linear index = (oc*H + r)*W + c
//   • Weights: flattened [oc][ic][kr][kc] in W_rom
//       addr = oc*(Cin*K*K) + ic*(K*K) + kr*K + kc
//
// FSM / pipeline (one tap per loop)
//   IDLE
//     - Wait for `start`. Initialise (oc,orow,ocol) and tap counters.
//     - Preload bias into accumulator; set weight base for oc=0.
//
//   READ
//     - Compute the input coordinates (ir,icol) for the current tap.
//     - Determine in-range (padding) and record pix_valid.
//     - If valid, assert if_en and drive if_addr for the BRAM read.
//     - Read the corresponding weight from W_rom into weight_reg.
//
//   WAIT
//     - One-cycle spacer to advance the pix_valid pipeline and keep the
//       control/data stages aligned.
//
//   PROD
//     - Move the selected weight into the MAC stage (weight_q).
//     - Advance pix_valid again to match the BRAM return timing.
//
//   CAP
//     - Capture the BRAM output (if_q) into if_q_q (now valid).
//     - Advance pix_valid so (if_q_q, weight_q, pix_valid_q3) align.
//
//   ACCUM
//     - Compute product for this tap:
//         prod = (pix_valid ? if_q_q : 0) * weight_q
//       then sign-extend to ACCW and accumulate.
//     - Advance kc/kr/ic (and w_addr) to the next tap.
//     - On the final tap of the pixel, compute the final scaled/clamped
//       result and stage it (res_q + conv_addr_q) for a clean 1-cycle write.
//
//   WRITE
//     - Drive conv_addr/conv_d and assert conv_en/conv_we for exactly one
//       cycle (“tail flush”) so the last tap is included and the BRAM write
//       strobes are clean.
//     - Advance (ocol,orow,oc). Reset tap counters and preload bias/weight
//       base for the next output pixel.
//
//   FINISH
//     - Pulse `done` for one cycle, then return to IDLE.
//
// Why WAIT/PROD/CAP exist
//   A synchronous 1R BRAM returns data one cycle after the address/en is
//   asserted. By explicitly staging:
//     READ  -> WAIT -> PROD -> CAP -> ACCUM
//   we guarantee the tap’s (data, weight, valid) are time-aligned when the
//   multiply-accumulate occurs, even when taps are invalid (padding).
//
// Notes
//   • Padding: out-of-bounds taps are treated as zero via pix_valid gating.
//   • Scaling: bias is pre-shifted by FRAC_BITS; the final accumulator is
//     right-shifted by FRAC_BITS before saturation/clamping to DATA_WIDTH.
//   • Throughput: one tap per cycle; latency per output pixel is
//     (Cin*K*K) * pipeline_depth + write overhead.
//======================================================================
(* keep_hierarchy = "yes" *)
module conv2d #(
    parameter int DATA_WIDTH      = 16,
    parameter int FRAC_BITS       = 0,
    parameter int IN_CHANNELS     = 1,
    parameter int OUT_CHANNELS    = 8,
    parameter int KERNEL          = 3,                          // assumed odd
    parameter int IMG_SIZE        = 28,
    parameter string WEIGHTS_FILE = "conv1_weights.mem",
    parameter string BIASES_FILE  = "conv1_biases.mem"
)(
    input  logic clk,
    input  logic reset,                                        // synchronous reset
    input  logic start,                                        // start pulse

    // IFMAP BRAM (read-only, 1-cycle latency)
    output logic [$clog2(IN_CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] if_addr,
    output logic                                              if_en,
    input  logic signed [DATA_WIDTH-1:0]                      if_q,

    // OFMAP BRAM (write-only)
    output logic [$clog2(OUT_CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_addr,
    output logic                                              conv_en,
    output logic                                              conv_we,
    output logic signed [DATA_WIDTH-1:0]                      conv_d,

    output logic done
);

    // -------------------- Geometry and addressing --------------------
    localparam int PAD    = (KERNEL-1)/2;
    localparam int HEIGHT = IMG_SIZE;
    localparam int WIDTH  = IMG_SIZE;

    localparam int IF_SZ  = IN_CHANNELS*HEIGHT*WIDTH;
    localparam int OF_SZ  = OUT_CHANNELS*HEIGHT*WIDTH;

    localparam int IF_AW  = (IF_SZ <= 1) ? 1 : $clog2(IF_SZ);
    localparam int OF_AW  = (OF_SZ <= 1) ? 1 : $clog2(OF_SZ);

    typedef logic [IF_AW-1:0] if_addr_t;
    typedef logic [OF_AW-1:0] of_addr_t;

    // Accumulator width: product width + log2(#taps) + headroom
    localparam int ACCW = DATA_WIDTH*2 + $clog2(KERNEL*KERNEL*IN_CHANNELS) + 2;

    // --------------------------- FSM enum ----------------------------
    typedef enum logic [2:0] {IDLE, READ, WAIT, PROD, CAP, ACCUM, WRITE, FINISH} state_t;
    state_t state;

    // ------------------------- Loop counters -------------------------
    integer oc, orow, ocol;
    integer ic, kr, kc;

    // ----------------------- Accumulator / MAC -----------------------
    (* use_dsp = "yes" *) logic signed [ACCW-1:0] acc;

    // ------------------------ Saturation bounds ----------------------
    localparam logic signed [DATA_WIDTH-1:0] S_MAX  = (1 <<< (DATA_WIDTH-1)) - 1;
    localparam logic signed [DATA_WIDTH-1:0] S_MIN  = - (1 <<< (DATA_WIDTH-1));
    localparam logic signed [ACCW-1:0]       S_MAXX = {{(ACCW-DATA_WIDTH){S_MAX[DATA_WIDTH-1]}}, S_MAX};
    localparam logic signed [ACCW-1:0]       S_MINX = {{(ACCW-DATA_WIDTH){S_MIN[DATA_WIDTH-1]}}, S_MIN};

    // -------------------------- ROM storage --------------------------
    localparam int W_DEPTH = OUT_CHANNELS*IN_CHANNELS*KERNEL*KERNEL;
    (* rom_style="block", ram_style="block" *) logic signed [DATA_WIDTH-1:0] W_rom [0:W_DEPTH-1];
    (* rom_style="block", ram_style="block" *) logic signed [DATA_WIDTH-1:0] B_rom [0:OUT_CHANNELS-1];

    // ---------------------- Weight addressing ------------------------
    localparam int           W_K2        = KERNEL*KERNEL;
    localparam int           W_OC_STRIDE = IN_CHANNELS*W_K2;
    localparam int           W_AW        = (W_DEPTH<=1)? 1 : $clog2(W_DEPTH);
    typedef logic [W_AW-1:0] w_addr_t;

    w_addr_t w_addr;
    w_addr_t w_base_oc;

    // ---------------------------- Helpers ----------------------------
    function automatic int lin3(input int ch, input int row, input int col, input int H, input int W);
        return (ch*H + row)*W + col;
    endfunction

    function automatic logic signed [ACCW-1:0] bias_ext(input logic signed [DATA_WIDTH-1:0] b);
        return $signed({{(ACCW-DATA_WIDTH){b[DATA_WIDTH-1]}}, b}) <<< FRAC_BITS;
    endfunction

    // --------------- Pipeline regs to align data/weight/valid --------
    logic                         pix_valid_q, pix_valid_q1, pix_valid_q2, pix_valid_q3;
    logic signed [DATA_WIDTH-1:0] if_q_q;
    logic signed [DATA_WIDTH-1:0] weight_reg;
    logic signed [DATA_WIDTH-1:0] weight_q;

    // -------------------- Tail-flush write staging -------------------
    logic   signed [DATA_WIDTH-1:0] res_q;
    of_addr_t                       conv_addr_q;

    // -------------------- Scratch for product/extend -----------------
    logic signed [2*DATA_WIDTH-1:0] prod_now;
    logic signed [ACCW-1:0]         prod_ext;

    // --------------------- ROM file loading (sim) --------------------
    initial begin
`ifndef SYNTHESIS
        integer i, sumW, sumB, fdw, fdb;
        fdw = $fopen(WEIGHTS_FILE, "r"); if (fdw==0) $fatal(1, "%m: cannot open weights '%s'", WEIGHTS_FILE); else $fclose(fdw);
        fdb = $fopen(BIASES_FILE,  "r"); if (fdb==0) $fatal(1, "%m: cannot open biases  '%s'", BIASES_FILE);  else $fclose(fdb);
`endif
        $readmemh(WEIGHTS_FILE, W_rom);
        $readmemh(BIASES_FILE,  B_rom);
`ifndef SYNTHESIS
        sumW = 0; sumB = 0;
        for (i=0;i<$size(W_rom);i++) sumW += W_rom[i];
        for (i=0;i<$size(B_rom);i++) sumB += B_rom[i];
        $display("%m: loaded %0d weights, %0d biases; sums: W=%0d B=%0d",
                 $size(W_rom), $size(B_rom), sumW, sumB);
`endif
    end

    // =============================== FSM =============================
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; done <= 1'b0;

            oc<=0; orow<=0; ocol<=0;
            ic<=0; kr<=0; kc<=0;

            acc <= '0;

            if_en     <= 1'b0;
            if_addr   <= '0;
            conv_en   <= 1'b0;
            conv_we   <= 1'b0;
            conv_addr <= '0;
            conv_d    <= '0;

            pix_valid_q  <= 1'b0;
            pix_valid_q1 <= 1'b0;
            pix_valid_q2 <= 1'b0;
            pix_valid_q3 <= 1'b0;

            if_q_q     <= '0;
            weight_reg <= '0;
            weight_q   <= '0;

            w_addr    <= '0;
            w_base_oc <= '0;

            res_q       <= '0;
            conv_addr_q <= '0;

        end else begin
            // default strobes
            done    <= 1'b0;
            if_en   <= 1'b0;
            conv_en <= 1'b0;
            conv_we <= 1'b0;

            unique case (state)

              IDLE: if (start) begin
                        oc<=0; orow<=0; ocol<=0;
                        ic<=0; kr<=0; kc<=0;

                        acc       <= bias_ext(B_rom[0]);
                        w_base_oc <= w_addr_t'(0);
                        w_addr    <= w_addr_t'(0);

                        state <= READ;
                    end

              READ: begin
                    int   ir   = orow + kr - PAD;
                    int   icol = ocol + kc - PAD;
                    logic in_range = (ir>=0 && ir<HEIGHT && icol>=0 && icol<WIDTH);

                    pix_valid_q <= in_range;

                    if (in_range) begin
                        if_addr <= if_addr_t'(lin3(ic, ir, icol, HEIGHT, WIDTH));
                        if_en   <= 1'b1;
                    end

                    weight_reg <= W_rom[w_addr];

                    state <= WAIT;
              end

              WAIT: begin
                    pix_valid_q1 <= pix_valid_q;
                    state <= PROD;
              end

              PROD: begin
                    weight_q     <= weight_reg;
                    pix_valid_q2 <= pix_valid_q1;
                    state <= CAP;
              end

              CAP: begin
                    if_q_q       <= if_q;
                    pix_valid_q3 <= pix_valid_q2;
                    state <= ACCUM;
              end

              ACCUM: begin
                    prod_now = (pix_valid_q3 ? if_q_q : '0) * weight_q;
                    prod_ext = {{(ACCW-2*DATA_WIDTH){prod_now[2*DATA_WIDTH-1]}}, prod_now};
                    acc <= acc + prod_ext;

                    if (kc == KERNEL-1) begin
                        kc <= 0;
                        if (kr == KERNEL-1) begin
                            kr <= 0;
                            if (ic == IN_CHANNELS-1) begin
                                logic signed [ACCW-1:0] shifted;
                                shifted = (acc + prod_ext) >>> FRAC_BITS;

                                if      (shifted > S_MAXX) res_q = S_MAX;
                                else if (shifted < S_MINX) res_q = S_MIN;
                                else                       res_q = shifted[DATA_WIDTH-1:0];

                                conv_addr_q <= of_addr_t'(lin3(oc, orow, ocol, HEIGHT, WIDTH));
                                state <= WRITE;
                            end else begin
                                ic     <= ic + 1;
                                w_addr <= w_addr + w_addr_t'(1);
                                state  <= READ;
                            end
                        end else begin
                            kr     <= kr + 1;
                            w_addr <= w_addr + w_addr_t'(1);
                            state  <= READ;
                        end
                    end else begin
                        kc     <= kc + 1;
                        w_addr <= w_addr + w_addr_t'(1);
                        state  <= READ;
                    end
              end

              WRITE: begin
                    conv_addr <= conv_addr_q;
                    conv_d    <= res_q;
                    conv_en   <= 1'b1;
                    conv_we   <= 1'b1;

                    if (ocol == WIDTH-1) begin
                        ocol <= 0;
                        if (orow == HEIGHT-1) begin
                            orow <= 0;
                            if (oc == OUT_CHANNELS-1) begin
                                state <= FINISH;
                            end else begin
                                oc        <= oc + 1;
                                ic<=0; kr<=0; kc<=0;
                                acc       <= bias_ext(B_rom[oc+1]);
                                w_base_oc <= w_base_oc + w_addr_t'(W_OC_STRIDE);
                                w_addr    <= w_base_oc + w_addr_t'(W_OC_STRIDE);
                                state     <= READ;
                            end
                        end else begin
                            orow <= orow + 1;
                            ic<=0; kr<=0; kc<=0;
                            acc    <= bias_ext(B_rom[oc]);
                            w_addr <= w_base_oc;
                            state  <= READ;
                        end
                    end else begin
                        ocol <= ocol + 1;
                        ic<=0; kr<=0; kc<=0;
                        acc    <= bias_ext(B_rom[oc]);
                        w_addr <= w_base_oc;
                        state  <= READ;
                    end
              end

              FINISH: begin
                    done  <= 1'b1;
                    state <= IDLE;
              end

            endcase
        end
    end
endmodule
