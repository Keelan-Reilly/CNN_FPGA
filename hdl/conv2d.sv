//======================================================================
// conv2d.sv — 2-D convolution layer (BRAM streaming version, lint-clean)
//======================================================================
(* keep_hierarchy = "yes" *)
module conv2d #(
    parameter int DATA_WIDTH   = 16,
    parameter int FRAC_BITS    = 7,
    parameter int IN_CHANNELS  = 1,
    parameter int OUT_CHANNELS = 8,
    parameter int KERNEL       = 3,
    parameter int IMG_SIZE     = 28,
    parameter string WEIGHTS_FILE = "conv1_weights.mem",
    parameter string BIASES_FILE  = "conv1_biases.mem"
)(
    input  logic clk,
    input  logic reset,
    input  logic start,

    // IFMAP BRAM (read-only here)
    output logic [$clog2(IN_CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] if_addr,
    output logic                                              if_en,
    input  logic  signed [DATA_WIDTH-1:0]                     if_q,

    // CONV buffer BRAM (write-only here)
    output logic [$clog2(OUT_CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_addr,
    output logic                                              conv_en,
    output logic                                              conv_we,
    output logic  signed [DATA_WIDTH-1:0]                     conv_d,

    output logic done
);
    localparam int PAD    = (KERNEL-1)/2;
    localparam int HEIGHT = IMG_SIZE;
    localparam int WIDTH  = IMG_SIZE;
    localparam int IF_SZ  = IN_CHANNELS*HEIGHT*WIDTH;
    localparam int OF_SZ  = OUT_CHANNELS*HEIGHT*WIDTH;
    localparam int IF_AW = $clog2(IF_SZ);
    localparam int OF_AW = $clog2(OF_SZ);
    typedef logic [IF_AW-1:0] if_addr_t;
    typedef logic [OF_AW-1:0] of_addr_t;

    // Accumulator headroom
    localparam int ACCW = DATA_WIDTH*2 + $clog2(KERNEL*KERNEL*IN_CHANNELS) + 2;

    typedef enum logic [2:0] {IDLE, READ, MAC, WRITE, FINISH} state_t;
    state_t state;

    integer oc, orow, ocol;
    integer ic, kr, kc;

    logic signed [ACCW-1:0]         acc;
    (* use_dsp = "yes" *) logic signed [2*DATA_WIDTH-1:0] prod;

    // Sign-extended saturation bounds in ACC width
    localparam logic signed [DATA_WIDTH-1:0] S_MAX = (1 <<< (DATA_WIDTH-1)) - 1;
    localparam logic signed [DATA_WIDTH-1:0] S_MIN = - (1 <<< (DATA_WIDTH-1));
    localparam logic signed [ACCW-1:0] S_MAXX = {{(ACCW-DATA_WIDTH){S_MAX[DATA_WIDTH-1]}}, S_MAX};
    localparam logic signed [ACCW-1:0] S_MINX = {{(ACCW-DATA_WIDTH){S_MIN[DATA_WIDTH-1]}}, S_MIN};

    // ROMs
    localparam int W_DEPTH = OUT_CHANNELS*IN_CHANNELS*KERNEL*KERNEL;
    (* rom_style = "block", ram_style = "block" *) logic signed [DATA_WIDTH-1:0] W_rom [0:W_DEPTH-1];
    (* rom_style = "block", ram_style = "block" *) logic signed [DATA_WIDTH-1:0] B_rom [0:OUT_CHANNELS-1];

    integer fdw, fdb;
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

    function automatic int lin3(input int ch, input int row, input int col,
                                input int H,  input int W);
        return (ch*H + row)*W + col;
    endfunction

    function automatic logic signed [ACCW-1:0] bias_ext(input logic signed [DATA_WIDTH-1:0] b);
        return $signed({{(ACCW-DATA_WIDTH){b[DATA_WIDTH-1]}}, b}) <<< FRAC_BITS;
    endfunction

    // BRAM read pipeline helpers
    logic pix_valid_q;
    logic signed [DATA_WIDTH-1:0] weight_reg;
    integer ir, icc;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; done <= 1'b0;
            oc<=0; orow<=0; ocol<=0;
            ic<=0; kr<=0; kc<=0;
            acc <= '0; prod <= '0;
            if_en <= 1'b0; conv_en <= 1'b0; conv_we <= 1'b0;
            pix_valid_q <= 1'b0;
        end else begin
            done   <= 1'b0;
            if_en  <= 1'b0;
            conv_en<= 1'b0;
            conv_we<= 1'b0;

            unique case (state)
              IDLE: if (start) begin
                        oc<=0; orow<=0; ocol<=0;
                        ic<=0; kr<=0; kc<=0;
                        acc <= bias_ext(B_rom[0]);
                        state <= READ;
                    end

              READ: begin
                    ir  = orow + kr - PAD;
                    icc = ocol + kc - PAD;

                    pix_valid_q <= (ir>=0 && ir<HEIGHT && icc>=0 && icc<WIDTH);
                    if ((ir>=0) && (ir<HEIGHT) && (icc>=0) && (icc<WIDTH)) begin
                        if_addr <= if_addr_t'( lin3(ic, ir, icc, HEIGHT, WIDTH) );
                        if_en   <= 1'b1;
                    end
                    weight_reg <= W_rom[((oc*IN_CHANNELS + ic)*KERNEL + kr)*KERNEL + kc];
                    state <= MAC;
                  end

              MAC: begin
                    // Use the pixel arriving from BRAM this cycle and the weight latched in READ
                    prod = (pix_valid_q ? if_q : '0) * weight_reg; // blocking temp
                    acc  <= acc + {{(ACCW-2*DATA_WIDTH){prod[2*DATA_WIDTH-1]}}, prod};

                    if (kc == KERNEL-1) begin
                        kc <= 0;
                        if (kr == KERNEL-1) begin
                            kr <= 0;
                            if (ic == IN_CHANNELS-1) state <= WRITE;
                            else begin ic <= ic + 1; state <= READ; end
                        end else begin kr <= kr + 1; state <= READ; end
                    end else begin kc <= kc + 1; state <= READ; end
                  end

              WRITE: begin
                    logic signed [ACCW-1:0]       shifted;
                    logic signed [DATA_WIDTH-1:0] res;

                    shifted = acc >>> FRAC_BITS;
                    if      (shifted > S_MAXX) res <= S_MAX;
                    else if (shifted < S_MINX) res <= S_MIN;
                    else                       res <= shifted[DATA_WIDTH-1:0];

                    conv_addr <= of_addr_t'( lin3(oc, orow, ocol, HEIGHT, WIDTH) );
                    conv_d    <= res;
                    conv_en   <= 1'b1;
                    conv_we   <= 1'b1;

                    if (ocol == WIDTH-1) begin
                        ocol <= 0;
                        if (orow == HEIGHT-1) begin
                            orow <= 0;
                            if (oc == OUT_CHANNELS-1) begin
                                state <= FINISH;
                            end else begin
                                oc  <= oc + 1;
                                ic<=0; kr<=0; kc<=0;
                                acc <= bias_ext(B_rom[oc+1]);
                                state <= READ;
                            end
                        end else begin
                            orow <= orow + 1;
                            ic<=0; kr<=0; kc<=0;
                            acc <= bias_ext(B_rom[oc]);
                            state <= READ;
                        end
                    end else begin
                        ocol <= ocol + 1;
                        ic<=0; kr<=0; kc<=0;
                        acc <= bias_ext(B_rom[oc]);
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
