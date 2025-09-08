//======================================================================
// conv2d.sv — 2-D convolution layer (fixed-point, same output size)
//----------------------------------------------------------------------
// What this module does:
//   • Takes a multi-channel square image and a set of convolution kernels.
//   • Computes OUT_CHANNELS output feature maps, each the same size as the input.
//   • Uses fixed-point maths. Results are clipped to the output bit-width.
//   • You start it with `start`; it runs through the whole image once and
//     pulses `done` at the end.
//
// How it runs internally:
//   • A small state machine walks one output pixel at a time.
//   • For that pixel, it multiplies the relevant input pixels by the kernel
//     weights and adds them up, starting from the bias.
//   • It then scales the sum back to the output format, clips it if needed,
//     writes the pixel, and moves to the next location.
//   • Zero-padding is used at the image edges so the output size matches the input.
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

    input  logic signed [DATA_WIDTH-1:0] input_feature_flat [0:IN_CHANNELS*IMG_SIZE*IMG_SIZE-1],
    output logic signed [DATA_WIDTH-1:0] out_feature_flat   [0:OUT_CHANNELS*IMG_SIZE*IMG_SIZE-1],

    output logic done
);
    localparam int PAD    = (KERNEL-1)/2;
    localparam int HEIGHT = IMG_SIZE;
    localparam int WIDTH  = IMG_SIZE;
    localparam int IF_SZ  = IN_CHANNELS*HEIGHT*WIDTH;
    localparam int OF_SZ  = OUT_CHANNELS*HEIGHT*WIDTH;

    // Accumulator headroom
    localparam int ACCW = DATA_WIDTH*2 + $clog2(KERNEL*KERNEL*IN_CHANNELS) + 2;

    // FSM
    typedef enum logic [1:0] {IDLE, MAC, WRITE, FINISH} state_t;
    state_t state;

    integer oc, orow, ocol;
    integer ic, kr, kc;

    logic signed [ACCW-1:0]         acc;
    (* use_dsp = "yes" *) logic signed [2*DATA_WIDTH-1:0] prod;

    // Saturation bounds
    localparam logic signed [DATA_WIDTH-1:0] S_MAX = (1 <<< (DATA_WIDTH-1)) - 1;
    localparam logic signed [DATA_WIDTH-1:0] S_MIN = - (1 <<< (DATA_WIDTH-1));

    // Indexing helpers
    function automatic int idx3(input int ch, input int row, input int col, input int HH, input int WW);
        return (ch*HH + row)*WW + col;
    endfunction
    function automatic int w_idx(input int och, input int ich, input int r, input int c);
        return (((och*IN_CHANNELS + ich)*KERNEL + r)*KERNEL + c);
    endfunction

    // ---------------- Internal ROMs for weights/biases ----------------
    localparam int W_DEPTH = OUT_CHANNELS*IN_CHANNELS*KERNEL*KERNEL;
    (* rom_style = "block", ram_style = "block" *) logic signed [DATA_WIDTH-1:0] W_rom [0:W_DEPTH-1];
    (* rom_style = "block", ram_style = "block" *) logic signed [DATA_WIDTH-1:0] B_rom [0:OUT_CHANNELS-1];

    initial begin
        $readmemh(WEIGHTS_FILE, W_rom);
        $readmemh(BIASES_FILE,  B_rom);
    end

    // Safe input fetch with zero padding
    function automatic logic signed [DATA_WIDTH-1:0]
    in_at(input int ch, input int r, input int c);
        if (r < 0 || r >= HEIGHT || c < 0 || c >= WIDTH) return '0;
        else return input_feature_flat[idx3(ch, r, c, HEIGHT, WIDTH)];
    endfunction

    // Main FSM
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; done <= 1'b0;
            oc<=0; orow<=0; ocol<=0;
            ic<=0; kr<=0; kc<=0;
            acc <= '0; prod <= '0;
        end else begin
            done <= 1'b0;

            unique case (state)
              IDLE: if (start) begin
                        oc<=0; orow<=0; ocol<=0;
                        ic<=0; kr<=0; kc<=0;
                        acc <= $signed({{(ACCW-DATA_WIDTH){B_rom[0][DATA_WIDTH-1]}}, B_rom[0]}) <<< FRAC_BITS;
                        state <= MAC;
                    end

              MAC: begin
                    automatic int ir  = orow + kr - PAD;
                    automatic int icc = ocol + kc - PAD;

                    prod = in_at(ic, ir, icc) * W_rom[w_idx(oc, ic, kr, kc)];
                    acc  = acc + prod; // blocking add so WRITE sees full sum next

                    if (kc == KERNEL-1) begin
                        kc <= 0;
                        if (kr == KERNEL-1) begin
                            kr <= 0;
                            if (ic == IN_CHANNELS-1) state <= WRITE;
                            else                      ic    <= ic + 1;
                        end else kr <= kr + 1;
                    end else kc <= kc + 1;
                   end

              WRITE: begin
                    logic signed [ACCW-1:0]       shifted;
                    logic signed [DATA_WIDTH-1:0] res;

                    shifted = acc >>> FRAC_BITS;
                    if (shifted > S_MAX)      res <= S_MAX;
                    else if (shifted < S_MIN) res <= S_MIN;
                    else                      res <= shifted[DATA_WIDTH-1:0];

                    out_feature_flat[idx3(oc, orow, ocol, HEIGHT, WIDTH)] <= res;

                    if (ocol == WIDTH-1) begin
                        ocol <= 0;
                        if (orow == HEIGHT-1) begin
                            orow <= 0;
                            if (oc == OUT_CHANNELS-1) begin
                                state <= FINISH;
                            end else begin
                                oc <= oc + 1;
                                ic<=0; kr<=0; kc<=0;
                                acc <= $signed({{(ACCW-DATA_WIDTH){B_rom[oc+1][DATA_WIDTH-1]}}, B_rom[oc+1]}) <<< FRAC_BITS;
                                state <= MAC;
                            end
                        end else begin
                            orow <= orow + 1;
                            ic<=0; kr<=0; kc<=0;
                            acc <= $signed({{(ACCW-DATA_WIDTH){B_rom[oc][DATA_WIDTH-1]}}, B_rom[oc]}) <<< FRAC_BITS;
                            state <= MAC;
                        end
                    end else begin
                        ocol <= ocol + 1;
                        ic<=0; kr<=0; kc<=0;
                        acc <= $signed({{(ACCW-DATA_WIDTH){B_rom[oc][DATA_WIDTH-1]}}, B_rom[oc]}) <<< FRAC_BITS;
                        state <= MAC;
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
