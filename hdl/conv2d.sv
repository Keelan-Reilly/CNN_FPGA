//======================================================================
// Module: conv2d
//
// High-level: timing-correct 2-D convolution with a small, explicit MAC
// pipeline that lines up a 1-read (1R) BRAM’s data with the current
// weight and a per-tap valid bit for zero-padding.
//
// Pipeline / FSM flow:
//   IDLE  : seed loop counters, preload bias and weight base for oc=0
//   READ  : compute ifmap address for the current tap, assert if_en,
//           and grab the matching weight from W_rom
//   WAIT  : single spacer to register pix_valid (keeps stages balanced)
//   PROD  : move the selected weight into the MAC stage; advance valid
//   CAP   : capture BRAM output if_q after the TB/BRAM has updated it
//   ACCUM : multiply (if valid), sign-extend the product, accumulate;
//           advance nested (kc,kr,ic) tap counters; at the last tap
//           compute the final clamped result and stage it for WRITE
//   WRITE : one-cycle "tail flush": drive conv_addr/conv_d with the
//           staged result, assert conv_en/conv_we; then advance (ocol,
//           orow, oc) and preload bias/weight base for the next pixel
//   FINISH: pulse done and return to IDLE
//
// Why WAIT/PROD/CAP are necessary:
//   • READ puts an address on the input BRAM and picks a weight.
//   • A 1R BRAM updates its output on the *next* clock; the TB also
//     updates a cycle later. CAP captures that updated if_q one cycle
//     after PROD moved the weight into place. With WAIT+PROD+CAP you
//     guarantee (if_q_q, weight_q, valid) are aligned at ACCUM.
//
// Weight addressing:
//   • W_rom is linear: [oc][ic][kr][kc] flattened.
//   • w_base_oc points to the first weight for the current oc.
//   • w_addr starts at w_base_oc and increments once per tap.
//
// Padding:
//   • For out-of-bounds taps, pix_valid_q* flags 0; ACCUM multiplies 0.
//
// Output write:
//   • WRITE is separate so the last tap’s product is included (no
//     off-by-one on the accumulator) and the BRAM write strobes are
//     clean for one cycle.
//======================================================================
(* keep_hierarchy = "yes" *)
module conv2d #(
    parameter int DATA_WIDTH    = 16,                         // Q-format data/weights width
    parameter int FRAC_BITS     = 0,                          // binary point for bias/output scaling
    parameter int IN_CHANNELS   = 1,                          // Cin
    parameter int OUT_CHANNELS  = 8,                          // Cout
    parameter int KERNEL        = 3,                          // K (assumed odd)
    parameter int IMG_SIZE      = 28,                         // H=W
    parameter string WEIGHTS_FILE = "conv1_weights.mem",      // hex weights
    parameter string BIASES_FILE  = "conv1_biases.mem"        // hex biases
)(
    input  logic clk,                                         // clock
    input  logic reset,                                       // sync reset
    input  logic start,                                       // start pulse

    // IFMAP BRAM (read-only)
    output logic [$clog2(IN_CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] if_addr, // read address
    output logic                                              if_en,   // read enable
    input  logic  signed [DATA_WIDTH-1:0]                     if_q,    // read data

    // CONV buffer BRAM (write-only)
    output logic [$clog2(OUT_CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_addr, // write address
    output logic                                              conv_en,   // port enable
    output logic                                              conv_we,   // write strobe
    output logic  signed [DATA_WIDTH-1:0]                     conv_d,    // write data

    output logic done                                         // done pulse at end
);
    // -------------------- Geometry and addressing --------------------
    localparam int PAD     = (KERNEL-1)/2;                    // SAME padding half-width
    localparam int HEIGHT  = IMG_SIZE;                        // H
    localparam int WIDTH   = IMG_SIZE;                        // W
    localparam int IF_SZ   = IN_CHANNELS*HEIGHT*WIDTH;        // total IFMAP elements
    localparam int OF_SZ   = OUT_CHANNELS*HEIGHT*WIDTH;       // total OFMAP elements

    localparam int IF_AW = (IF_SZ <= 1) ? 1 : $clog2(IF_SZ);  // IFMAP addr width
    localparam int OF_AW = (OF_SZ <= 1) ? 1 : $clog2(OF_SZ);  // OFMAP addr width
    typedef logic [IF_AW-1:0] if_addr_t;                      // IFMAP addr type
    typedef logic [OF_AW-1:0] of_addr_t;                      // OFMAP addr type

    // Accumulator width: product width + log2(#taps) + small headroom
    localparam int ACCW = DATA_WIDTH*2 + $clog2(KERNEL*KERNEL*IN_CHANNELS) + 2;

    // --------------------------- FSM enum ----------------------------
    typedef enum logic [2:0] {IDLE, READ, WAIT, PROD, CAP, ACCUM, WRITE, FINISH} state_t;
    state_t state;                                            // current FSM state

    // ------------------------- Loop counters -------------------------
    integer oc, orow, ocol;                                   // output channel,row,col
    integer ic, kr, kc;                                       // input channel,kernel r,c

    // ----------------------- Accumulator / MAC -----------------------
    (* use_dsp = "yes" *) logic signed [ACCW-1:0] acc;        // running sum for one output pixel

    // ------------------------ Saturation bounds ----------------------
    localparam logic signed [DATA_WIDTH-1:0] S_MAX  = (1 <<< (DATA_WIDTH-1)) - 1; // +max at DATA_WIDTH
    localparam logic signed [DATA_WIDTH-1:0] S_MIN  = - (1 <<< (DATA_WIDTH-1));   // -min at DATA_WIDTH
    localparam logic signed [ACCW-1:0]       S_MAXX = {{(ACCW-DATA_WIDTH){S_MAX[DATA_WIDTH-1]}}, S_MAX}; // sign-extend S_MAX
    localparam logic signed [ACCW-1:0]       S_MINX = {{(ACCW-DATA_WIDTH){S_MIN[DATA_WIDTH-1]}}, S_MIN}; // sign-extend S_MIN

    // -------------------------- ROM storage --------------------------
    localparam int W_DEPTH = OUT_CHANNELS*IN_CHANNELS*KERNEL*KERNEL;   // total weights
    (* rom_style = "block", ram_style = "block" *) logic signed [DATA_WIDTH-1:0] W_rom [0:W_DEPTH-1]; // weights
    (* rom_style = "block", ram_style = "block" *) logic signed [DATA_WIDTH-1:0] B_rom [0:OUT_CHANNELS-1]; // biases

    // ---------------------- Weight addressing ------------------------
    localparam int            W_K2        = KERNEL*KERNEL;             // kernel taps per (oc,ic)
    localparam int            W_OC_STRIDE = IN_CHANNELS*W_K2;          // weights per output channel
    localparam int            W_AW        = (W_DEPTH<=1)? 1 : $clog2(W_DEPTH); // weight addr width
    typedef logic [W_AW-1:0]  w_addr_t;                                 // weight addr type

    w_addr_t w_addr;                                                   // current tap address
    w_addr_t w_base_oc;                                                // start of current oc block

    // ---------------------------- Helpers ----------------------------
    function automatic int lin3(                                       // flatten (ch,row,col) -> linear
        input int ch, input int row, input int col, input int H, input int W
    );
        return (ch*H + row)*W + col;                                   // ch*H*W + row*W + col
    endfunction

    function automatic logic signed [ACCW-1:0] bias_ext(               // sign-extend + <<FRAC_BITS
        input logic signed [DATA_WIDTH-1:0] b
    );
        return $signed({{(ACCW-DATA_WIDTH){b[DATA_WIDTH-1]}}, b}) <<< FRAC_BITS;
    endfunction

    // --------------- Pipeline regs to align data/weight/valid --------
    logic                         pix_valid_q, pix_valid_q1, pix_valid_q2, pix_valid_q3; // valid shift reg
    logic signed [DATA_WIDTH-1:0] if_q_q;           // captured BRAM data (CAP stage)
    logic signed [DATA_WIDTH-1:0] weight_reg;       // weight from W_rom (READ stage)
    logic signed [DATA_WIDTH-1:0] weight_q;         // weight into MAC (PROD stage)

    // -------------------- Tail-flush write staging -------------------
    logic   signed [DATA_WIDTH-1:0] res_q;          // staged final pixel result
    of_addr_t                       conv_addr_q;    // staged output address

    // -------------------- Scratch for product/extend -----------------
    logic signed [2*DATA_WIDTH-1:0] prod_now;       // raw product (DATA_WIDTH*2)
    logic signed [ACCW-1:0]         prod_ext;       // sign-extended product to ACCW

    // --------------------- ROM file loading (sim) --------------------
    initial begin
    `ifndef SYNTHESIS
        integer i, sumW, sumB, fdw, fdb;                               // sim-only checks
        fdw = $fopen(WEIGHTS_FILE, "r"); if (fdw==0) $fatal(1, "%m: cannot open weights '%s'", WEIGHTS_FILE); else $fclose(fdw);
        fdb = $fopen(BIASES_FILE,  "r"); if (fdb==0) $fatal(1, "%m: cannot open biases  '%s'", BIASES_FILE);  else $fclose(fdb);
    `endif
        $readmemh(WEIGHTS_FILE, W_rom);                                 // load weights
        $readmemh(BIASES_FILE,  B_rom);                                 // load biases
    `ifndef SYNTHESIS
        sumW = 0; sumB = 0;                                             // tiny checksum / visibility
        for (i=0;i<$size(W_rom);i++) sumW += W_rom[i];
        for (i=0;i<$size(B_rom);i++) sumB += B_rom[i];
        $display("%m: loaded %0d weights, %0d biases; sums: W=%0d B=%0d",
                 $size(W_rom), $size(B_rom), sumW, sumB);
    `endif
    end

    // =============================== FSM =============================
    always_ff @(posedge clk) begin
        if (reset) begin
            // ----------- reset all state/output/pipeline regs --------
            state <= IDLE; done <= 1'b0;

            oc<=0; orow<=0; ocol<=0;                                   // output indices
            ic<=0; kr<=0; kc<=0;                                       // tap indices

            acc <= '0;                                                // clear MAC

            if_en    <= 1'b0;                                          // IFMAP port idle
            if_addr  <= '0;                                            // IFMAP addr zero
            conv_en  <= 1'b0;                                          // OFMAP port idle
            conv_we  <= 1'b0;                                          // no write
            conv_addr<= '0;                                            // OFMAP addr zero
            conv_d   <= '0;                                            // OFMAP data zero

            pix_valid_q  <= 1'b0;                                      // valid pipeline clear
            pix_valid_q1 <= 1'b0;
            pix_valid_q2 <= 1'b0;
            pix_valid_q3 <= 1'b0;

            if_q_q     <= '0;                                          // captured pixel clear
            weight_reg <= '0;                                          // weight pipe clear
            weight_q   <= '0;

            w_addr    <= '0;                                           // weight addr clear
            w_base_oc <= '0;                                           // weight base clear

            res_q       <= '0;                                         // staged result clear
            conv_addr_q <= '0;                                         // staged addr clear

        end else begin
            // --------- default outputs each cycle (deassert strobes) --
            done    <= 1'b0;
            if_en   <= 1'b0;
            conv_en <= 1'b0;
            conv_we <= 1'b0;

            unique case (state)

              // -------------------------- IDLE ------------------------
              IDLE: if (start) begin
                        oc<=0; orow<=0; ocol<=0;                       // start at (oc=0,r=0,c=0)
                        ic<=0; kr<=0; kc<=0;                           // first tap

                        acc        <= bias_ext(B_rom[0]);              // preload bias for oc=0
                        w_base_oc  <= w_addr_t'(0);                    // weight base for oc=0
                        w_addr     <= w_addr_t'(0);                    // tap address starts at base

                        state <= READ;                                 // proceed to first read
                    end

              // -------------------------- READ ------------------------
              READ: begin
                    int ir  = orow + kr - PAD;                         // input row being tapped
                    int icol= ocol + kc - PAD;                         // input col being tapped
                    logic in_range = (ir>=0 && ir<HEIGHT &&            // padding check
                                       icol>=0 && icol<WIDTH);

                    pix_valid_q <= in_range;                           // record if this tap is valid

                    if (in_range) begin
                        if_addr <= if_addr_t'( lin3(ic, ir, icol, HEIGHT, WIDTH) ); // set IFMAP addr
                        if_en   <= 1'b1;                               // pulse IFMAP read
                    end

                    weight_reg <= W_rom[w_addr];                       // fetch weight for this tap

                    state <= WAIT;                                     // spacer for valid pipeline
              end

              // -------------------------- WAIT ------------------------
              WAIT: begin
                    pix_valid_q1 <= pix_valid_q;                       // advance valid by 1
                    state <= PROD;                                     // move weight next
              end

              // -------------------------- PROD ------------------------
              PROD: begin
                    weight_q     <= weight_reg;                        // place weight in MAC stage
                    pix_valid_q2 <= pix_valid_q1;                      // advance valid by 1
                    state <= CAP;                                      // next: capture BRAM data
              end

              // --------------------------- CAP ------------------------
              CAP: begin
                    if_q_q       <= if_q;                              // capture BRAM output now valid
                    pix_valid_q3 <= pix_valid_q2;                      // advance valid to align with if_q_q
                    state <= ACCUM;                                    // ready for MAC
              end

              // -------------------------- ACCUM -----------------------
              ACCUM: begin
                    // Multiply by weight only if this tap is in range (padding -> 0)
                    prod_now = (pix_valid_q3 ? if_q_q : '0) * weight_q; // raw product
                    prod_ext = {{(ACCW-2*DATA_WIDTH){prod_now[2*DATA_WIDTH-1]}}, prod_now}; // sign-extend
                    acc <= acc + prod_ext;                             // accumulate into acc

                    // ---- Advance nested tap counters; decide next state ----
                    if (kc == KERNEL-1) begin                          // end of kernel column?
                        kc <= 0;                                       // wrap kc
                        if (kr == KERNEL-1) begin                      // end of kernel row?
                            kr <= 0;                                   // wrap kr
                            if (ic == IN_CHANNELS-1) begin             // end of input channels?
                                // Last tap for this output pixel: form final value NOW
                                logic signed [ACCW-1:0] shifted;       // scaled accumulator
                                shifted = (acc + prod_ext) >>> FRAC_BITS; // include this cycle’s prod

                                if      (shifted > S_MAXX) res_q = S_MAX; // saturate high
                                else if (shifted < S_MINX) res_q = S_MIN; // saturate low
                                else                      res_q = shifted[DATA_WIDTH-1:0]; // in-range

                                conv_addr_q <= of_addr_t'(              // compute output address
                                                   lin3(oc, orow, ocol, HEIGHT, WIDTH) );
                                state <= WRITE;                         // tail flush write next
                            end else begin
                                ic     <= ic + 1;                       // next input channel
                                w_addr <= w_addr + w_addr_t'(1);        // next tap address
                                state  <= READ;                         // another tap
                            end
                        end else begin
                            kr     <= kr + 1;                           // next kernel row
                            w_addr <= w_addr + w_addr_t'(1);            // next tap
                            state  <= READ;                             // another tap
                        end
                    end else begin
                        kc     <= kc + 1;                               // next kernel col
                        w_addr <= w_addr + w_addr_t'(1);                // next tap
                        state  <= READ;                                 // another tap
                    end
              end

              // -------------------------- WRITE -----------------------
              WRITE: begin
                    conv_addr <= conv_addr_q;                          // drive output addr
                    conv_d    <= res_q;                                // drive output data
                    conv_en   <= 1'b1;                                 // enable port
                    conv_we   <= 1'b1;                                 // strobe write

                    // ---- Advance output pixel indices; preload next bias/weights ----
                    if (ocol == WIDTH-1) begin                         // end of row?
                        ocol <= 0;                                      // wrap col
                        if (orow == HEIGHT-1) begin                     // end of image?
                            orow <= 0;                                  // wrap row
                            if (oc == OUT_CHANNELS-1) begin             // last output channel?
                                state <= FINISH;                        // all done
                            end else begin
                                oc        <= oc + 1;                    // next output channel
                                ic<=0; kr<=0; kc<=0;                    // reset tap counters
                                acc       <= bias_ext(B_rom[oc+1]);     // preload next bias
                                w_base_oc <= w_base_oc + w_addr_t'(W_OC_STRIDE); // advance oc base
                                w_addr    <= w_base_oc + w_addr_t'(W_OC_STRIDE); // reset tap addr
                                state     <= READ;                      // start next pixel
                            end
                        end else begin
                            orow <= orow + 1;                           // next row
                            ic<=0; kr<=0; kc<=0;                        // reset tap counters
                            acc    <= bias_ext(B_rom[oc]);              // same oc bias
                            w_addr <= w_base_oc;                        // reset tap addr for oc
                            state  <= READ;                             // next pixel
                        end
                    end else begin
                        ocol <= ocol + 1;                               // next column
                        ic<=0; kr<=0; kc<=0;                            // reset tap counters
                        acc    <= bias_ext(B_rom[oc]);                  // same oc bias
                        w_addr <= w_base_oc;                            // reset tap addr for oc
                        state  <= READ;                                 // next pixel
                    end
              end

              // -------------------------- FINISH ----------------------
              FINISH: begin
                    done  <= 1'b1;                                     // pulse done
                    state <= IDLE;                                     // wait for next start
              end

            endcase
        end
    end
endmodule
