//======================================================================
// Module: conv2d
// Description:
//   Implements a 2-D convolutional neural network layer.
//
//   Functionality:
//   • Streams an input feature map from BRAM and applies small convolution
//     kernels (weights) across it to generate output feature maps.
//   • For each output channel and spatial position, multiplies the local
//     input patch by the kernel weights, sums the products, and adds a bias.
//   • Handles zero-padding at the image boundaries so the kernel can be
//     applied everywhere.
//   • Produces one accumulated result per output pixel, which is then
//     scaled, saturated, and written into output BRAM.
//
//   Key points:
//   • This is a local dot-product: 
//       out[oc,r,c] = bias[oc] + Σ_ic,kr,kc ( W[oc,ic,kr,kc] * in[ic,r+kr,c+kc] )
//   • Weights are reused across the entire spatial map (weight sharing).
//   • State machine streams data through a simple MAC pipeline, reusing
//     the same multipliers over many positions.
//   • Padding, kernel looping, and channel looping are handled internally
//     by counters and address generation logic.
//======================================================================
(* keep_hierarchy = "yes" *)
module conv2d #(
    parameter int DATA_WIDTH   = 16,  // Bit width for activations/weights (fixed-point).
    parameter int FRAC_BITS    = 7,   // Number of fractional bits (for Q format; used on bias shift and output descaling).
    parameter int IN_CHANNELS  = 1,   // Number of input feature map channels.
    parameter int OUT_CHANNELS = 8,   // Number of output channels (number of kernels/filters).
    parameter int KERNEL       = 3,   // Kernel size (KERNEL x KERNEL).
    parameter int IMG_SIZE     = 28,  // Height/Width of (square) input/output feature map.
    parameter string WEIGHTS_FILE = "conv1_weights.mem",
    parameter string BIASES_FILE  = "conv1_biases.mem"
)(
    input  logic clk,
    input  logic reset,
    input  logic start,

    // IFMAP BRAM (read-only here)
    output logic [$clog2(IN_CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] if_addr,  // Address into input feature-map BRAM.
    output logic                                              if_en,   // Read-enable for IFMAP BRAM.
    input  logic  signed [DATA_WIDTH-1:0]                     if_q,    // Data returned from IFMAP BRAM.

    // CONV buffer BRAM (write-only here)
    output logic [$clog2(OUT_CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_addr,  // Address into output feature-map BRAM.
    output logic                                              conv_en,    // Enable write port.
    output logic                                              conv_we,    // Write-enable strobe.
    output logic  signed [DATA_WIDTH-1:0]                     conv_d,     // Data written to output BRAM.

    output logic done // Pulsed high when a full sweep is complete.
);
    localparam int PAD    = (KERNEL-1)/2;   // Implicit SAME padding (assumes odd kernel).
    localparam int HEIGHT = IMG_SIZE;       // Height of input/output feature map.
    localparam int WIDTH  = IMG_SIZE;       // Width of input/output feature map.
    localparam int IF_SZ  = IN_CHANNELS*HEIGHT*WIDTH;  // Total input feature map size.
    localparam int OF_SZ  = OUT_CHANNELS*HEIGHT*WIDTH; // Total output feature map size.
    localparam int IF_AW = $clog2(IF_SZ);     // Address width for input feature map.
    localparam int OF_AW = $clog2(OF_SZ);     // Address width for output feature map.
    typedef logic [IF_AW-1:0] if_addr_t;      // Type for input feature map addresses.   
    typedef logic [OF_AW-1:0] of_addr_t;      // Type for output feature map addresses.

    // Accumulator headroom
    localparam int ACCW = DATA_WIDTH*2 + $clog2(KERNEL*KERNEL*IN_CHANNELS) + 2;  // Width of accumulator: product width + growth for sum of (K^2 * Cin) terms + margin.

    typedef enum logic [2:0] {IDLE, READ, PROD, ACCUM, WRITE, FINISH} state_t;
    state_t state;

    integer oc, orow, ocol;  // Loop counters: output channel, output row, output col.
    integer ic, kr, kc;      // Loop counters: input channel, kernel row, kernel col.

    (* use_dsp = "yes" *) logic signed [ACCW-1:0]         acc;   // Accumulator for MACs; hint to map to DSP block(s).

    // Sign-extended saturation bounds in ACC width
    localparam logic signed [DATA_WIDTH-1:0] S_MAX = (1 <<< (DATA_WIDTH-1)) - 1;  // Max representable signed value at DATA_WIDTH. 2^(DATA_WIDTH-1) - 1 = 32768
    localparam logic signed [DATA_WIDTH-1:0] S_MIN = - (1 <<< (DATA_WIDTH-1));    // Min representable signed value at DATA_WIDTH. -2^(DATA_WIDTH-1) = -32768
    localparam logic signed [ACCW-1:0] S_MAXX = {{(ACCW-DATA_WIDTH){S_MAX[DATA_WIDTH-1]}}, S_MAX};  // S_MAX sign-extended to ACCW. Replication operator of signed bit
    localparam logic signed [ACCW-1:0] S_MINX = {{(ACCW-DATA_WIDTH){S_MIN[DATA_WIDTH-1]}}, S_MIN};  // S_MIN sign-extended to ACCW. Replication operator of signed bit

    // ROMs
    localparam int W_DEPTH = OUT_CHANNELS*IN_CHANNELS*KERNEL*KERNEL;   // Total number of weights.
    (* rom_style = "block", ram_style = "block" *) logic signed [DATA_WIDTH-1:0] W_rom [0:W_DEPTH-1];        // Weight ROM (initialised from file; infer block RAM/ROM).
    (* rom_style = "block", ram_style = "block" *) logic signed [DATA_WIDTH-1:0] B_rom [0:OUT_CHANNELS-1];   // Bias ROM per output channel.

    //lightweight linear weight addressing
    localparam int            W_K2        = KERNEL*KERNEL;                    // Kernel elements per channel pair.
    localparam int            W_OC_STRIDE = IN_CHANNELS*W_K2;                 // Weights per output channel (stride between oc blocks).
    localparam int            W_AW        = (W_DEPTH<=1)? 1 : $clog2(W_DEPTH);   // Address width for W_rom.
    typedef logic [W_AW-1:0]  w_addr_t;   // Typedef for weight address.
    w_addr_t w_addr;       // current weight address within current oc (advances each tap)
    w_addr_t w_base_oc;    // base address for current oc (start of the oc's weight block)

    integer fdw, fdb;     // File descriptor ints for $fopen checks (simulation-only).

    // Load weights/biases from file for simulation or Vivado synthesis

    initial begin
    `ifndef SYNTHESIS
        integer i; integer sumW; integer sumB;

        fdw = $fopen(WEIGHTS_FILE, "r");
        if (fdw == 0) $fatal(1, "%m: cannot open weights file '%s'", WEIGHTS_FILE);
        else $fclose(fdw);
        fdb = $fopen(BIASES_FILE, "r");
        if (fdb == 0) $fatal(1, "%m: cannot open biases file '%s'", BIASES_FILE);
        else $fclose(fdb);
    `endif

        $readmemh(WEIGHTS_FILE, W_rom);
        $readmemh(BIASES_FILE,  B_rom);

    `ifndef SYNTHESIS
        // Tiny checksum to verify non-zero data
        sumW = 0; sumB = 0;
        for (i = 0; i < $size(W_rom); i++) sumW = sumW + W_rom[i];
        for (i = 0; i < $size(B_rom); i++) sumB = sumB + B_rom[i];
        $display("%m: loaded %0d weights, %0d biases; sums: W=%0d B=%0d",
                $size(W_rom), $size(B_rom), sumW, sumB);
    `endif
    end

    // This function computes a linear address from 3D indices (channel, row, column) into a 1D address.
    function automatic int lin3(input int ch, input int row, input int col, input int H,  input int W);
        return (ch*H + row)*W + col;
    endfunction

      // This function sign-extends bias to ACCW bits and shifts left by FRAC_BITS to align w
    function automatic logic signed [ACCW-1:0] bias_ext(input logic signed [DATA_WIDTH-1:0] b);
        return $signed({{(ACCW-DATA_WIDTH){b[DATA_WIDTH-1]}}, b}) <<< FRAC_BITS;    // Sign-extend bias to ACCW then left-shift by FRAC_BITS (Q-format).
    endfunction

    // BRAM read pipeline helpers
    logic signed [DATA_WIDTH-1:0] if_q_q;  // Registered version of if_q (1-cycle delayed sample).
    logic pix_valid_q, pix_valid_q1;       // Valid flags delayed to match if_q_q timing.

    logic signed [2*DATA_WIDTH-1:0] prod_reg;  // Registered product (for debug/waves).
    logic signed [DATA_WIDTH-1:0] weight_reg;  // Registered weight before multiply.

    logic signed [DATA_WIDTH-1:0] weight_q;    // Weight registered into the PROD stage.
    integer ir, icc;                           // Current input row/col (for padding check). 

    // debug counter
    int cyc;
    always_ff @(posedge clk) begin
        if (reset) cyc <= 0; 
        else cyc <= cyc + 1;
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; 
            done <= 1'b0;
            oc<=0; orow<=0; ocol<=0;
            ic<=0; kr<=0; kc<=0;
            acc <= '0;
            if_en <= 1'b0; conv_en <= 1'b0; conv_we <= 1'b0;
            pix_valid_q <= 1'b0; 
            pix_valid_q1 <= 1'b0; 
            weight_q <= '0; 
            if_q_q <= '0;
            prod_reg <= '0;
            w_addr <= '0; 
            w_base_oc <= '0;
        end else begin         // Active (not reset)
            done   <= 1'b0;    // Default: done low (pulse only).
            if_en  <= 1'b0;    // Default: IFMAP port idle unless READ asserts.
            conv_en<= 1'b0;    // Default: output port idle.
            conv_we<= 1'b0;    // Default: no write.

            unique case (state)
              // Wait for `start` signal to begin processing
              IDLE: if (start) begin
                        oc<=0; orow<=0; ocol<=0;
                        ic<=0; kr<=0; kc<=0;
                        acc <= bias_ext(B_rom[0]);
                        w_base_oc <= w_addr_t'(0); 
                        w_addr <= w_addr_t'(0);
                        state <= READ;
                    end
              // Compute linear address for current (ic,ir,ic) tap. Request IFMAP BRAM read. Check padding.
              READ: begin : dbg_read
                    ir  = orow + kr - PAD;
                    icc = ocol + kc - PAD;

                    pix_valid_q <= (ir>=0 && ir<HEIGHT && icc>=0 && icc<WIDTH);
                    if ((ir>=0) && (ir<HEIGHT) && (icc>=0) && (icc<WIDTH)) begin
                        if_addr <= if_addr_t'( lin3(ic, ir, icc, HEIGHT, WIDTH) );
                        if_en   <= 1'b1;
                    end
                    weight_reg <= W_rom[w_addr]; // latch weight for MAC stage

                    // TAP: show what was just requested
                    if (cyc < 200) $display("[%0t][READ ] oc=%0d r=%0d c=%0d ic=%0d kr=%0d kc=%0d v=%0b if_addr=%0d w_addr=%0d w_now=%0d",
                                            $time, oc, orow, ocol, ic, kr, kc, pix_valid_q,
                                            if_addr, w_addr, W_rom[w_addr]);
                    state <= PROD;
                  end
              // align BRAM data with weight into regs for multiply
              PROD: begin : dbg_prod
                    if_q_q       <= if_q;         // Capture IFMAP data arriving from prior READ.
                    weight_q     <= weight_reg;   // Move weight into multiply stage register.
                    pix_valid_q1 <= pix_valid_q;  // Pipeline the valid bit to align with if_q_q, weight_q.
                    // TAP: show what was just latched
                    if (cyc < 200) $display("[%0t][PROD ] latched: v1=%0b if_q_q=%0d w_q=%0d (raw if_q=%0d w_reg=%0d)",
                          $time, pix_valid_q1, if_q_q, weight_q, if_q, weight_reg);

                    state       <= ACCUM;
                end

              // This is the convolution step; multiplies input pixel by weight and sums into accumulator.
              ACCUM: begin : dbg_accum
                    // product for *this* tap
                    logic signed [2*DATA_WIDTH-1:0] prod_now_local;
                    prod_now_local = (pix_valid_q1 ? if_q_q : '0) * weight_q; // Multiply if valid; else multiply 0 (implements zero-padding).

                    // Sign-extend product to ACCW and add to accumulator.
                    acc <= acc + {{(ACCW-2*DATA_WIDTH){prod_now_local[2*DATA_WIDTH-1]}}, prod_now_local};

                    prod_reg <= prod_now_local; // capture for debug/waves

                    if (cyc < 200) $display("[%0t][ACCUM] v1=%0b if_q_q=%0d w_q=%0d prod_now=%0d acc_pre=%0d",
                                            $time, pix_valid_q1, if_q_q, weight_q, prod_now_local, acc);

                    // Loop index updates
                    if (kc == KERNEL-1) begin
                        kc <= 0;
                        if (kr == KERNEL-1) begin
                            kr <= 0;
                            if (ic == IN_CHANNELS-1) begin
                                state <= WRITE;             // Done all taps for this output pixel; go write result.
                            end else begin
                                ic <= ic + 1;    // Next input channel
                                w_addr <= w_addr + w_addr_t'(1); // advance weight address
                                state <= READ;           // Start next tap read
                            end
                        end else begin
                            kr <= kr + 1;    // Next kernel row
                            w_addr <= w_addr + w_addr_t'(1); // advance weight address
                            state <= READ;           // Start next tap read
                        end
                    end else begin
                        kc <= kc + 1;  // Next kernel col
                        w_addr <= w_addr + w_addr_t'(1); // advance weight address
                        state <= READ;
                    end
              end

              // Write accumulated result to output BRAM with saturation and descaling; advance output indices.
              WRITE: begin : dbg_write
                    logic signed [ACCW-1:0]       shifted;  // shifted accumulator
                    logic signed [DATA_WIDTH-1:0] res;      // Final clamped result at DATA_WIDTH.

                    shifted = acc >>> FRAC_BITS;            // right shift to undo fixed-point accumulation scaling
                    if      (shifted > S_MAXX) res <= S_MAX;  // Clamp to max if overflow
                    else if (shifted < S_MINX) res <= S_MIN;  // Clamp to min if underflow
                    else                       res <= shifted[DATA_WIDTH-1:0];  // Take lower DATA_WIDTH bits if in range

                    conv_addr <= of_addr_t'( lin3(oc, orow, ocol, HEIGHT, WIDTH) );  // Compute output address
                    conv_d    <= res;                   // Data to write
                    conv_en   <= 1'b1; 
                    conv_we   <= 1'b1;

                    if (cyc < 400) $display("[%0t][WRITE] oc=%0d r=%0d c=%0d -> conv_addr=%0d conv_d=%0d acc=%0d",
                          $time, oc, orow, ocol, conv_addr, conv_d, acc);

                    // Advance output indices and prep next accumulation
                    if (ocol == WIDTH-1) begin
                        ocol <= 0;
                        if (orow == HEIGHT-1) begin
                            orow <= 0;
                            if (oc == OUT_CHANNELS-1) begin
                                state <= FINISH;
                            end else begin
                                oc  <= oc + 1;
                                ic<=0; kr<=0; kc<=0;
                                acc <= bias_ext(B_rom[oc+1]);                     // Load bias for next output channel
                                w_base_oc <= w_base_oc + w_addr_t'(W_OC_STRIDE);  // Advance base by one oc stride (skip current block).
                                w_addr    <= w_base_oc + w_addr_t'(W_OC_STRIDE);  // Set w_addr to start of next oc block
                                state <= READ;

                            end
                        end else begin
                            orow <= orow + 1;            // Next output row
                            ic<=0; kr<=0; kc<=0;         // Reset input coords
                            acc <= bias_ext(B_rom[oc]);  // Re-seed acc with same oc bias for new pixel.
                            w_addr <= w_base_oc;         // reset to start of current oc weights
                            state <= READ;
                        end
                    end else begin
                        ocol <= ocol + 1;           // Next output column
                        ic<=0; kr<=0; kc<=0;        // Reset inner loops for pixel
                        acc <= bias_ext(B_rom[oc]); // Re-seed acc with same oc bias for new pixel.
                        w_addr <= w_base_oc;        // reset to start of current oc weights
                        state <= READ;
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
