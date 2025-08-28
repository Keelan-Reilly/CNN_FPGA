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

module conv2d #(
    // Fixed-point word format and geometry
    parameter int DATA_WIDTH   = 16,  // total bits for inputs/weights/biases/output (signed)
    parameter int FRAC_BITS    = 7,   // fractional bits in fixed-point Q format
    parameter int IN_CHANNELS  = 1,   // number of input feature maps
    parameter int OUT_CHANNELS = 8,   // number of output feature maps (kernels)
    parameter int KERNEL       = 3,   // kernel size (assumed odd; e.g. 3, 5, 7)
    parameter int IMG_SIZE     = 28   // square image size (HxW)
)(
    // Clock / control
    input  logic clk,                 // rising-edge clock
    input  logic reset,               // synchronous reset (active high)
    input  logic start,               // pulse to start full convolution

    // Tensor ports (unpacked 3D/4D arrays)
    input  logic signed [DATA_WIDTH-1:0]
           input_feature [0:IN_CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1],  // Q(FRAC_BITS) input features

    input  logic signed [DATA_WIDTH-1:0]
           weights [0:OUT_CHANNELS-1][0:IN_CHANNELS-1][0:KERNEL-1][0:KERNEL-1], // Q(FRAC_BITS) kernels

    input  logic signed [DATA_WIDTH-1:0]
           biases  [0:OUT_CHANNELS-1],                                   // Q(FRAC_BITS) per-output bias

    output logic signed [DATA_WIDTH-1:0]
           out_feature [0:OUT_CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1],   // Q(FRAC_BITS) output features

    output logic done                // pulses high for one cycle at completion
);

    // Padding and accumulator width
    localparam int PAD  = (KERNEL-1)/2; // zero-padding for same-size output
    // ACCW: enough headroom for sum of (IN_CHANNELS × KERNEL×KERNEL) products
    localparam int ACCW = DATA_WIDTH*2 + $clog2(KERNEL*KERNEL*IN_CHANNELS) + 2;

    // FSM declaration
    typedef enum logic [1:0] {IDLE, MAC, WRITE, FINISH} state_t;
    state_t state; // Current FSM state

    // Loop indices (time-multiplexed counters)
    integer oc, orow, ocol; // output channel / row / col
    integer ic, kr, kc;     // input channel / kernel row / kernel col

    // Datapath registers
    logic signed [ACCW-1:0]         acc;  // widened accumulator (bias + sum of products)
    logic signed [2*DATA_WIDTH-1:0] prod; // raw product (input × weight)

    // Saturation bounds after scaling back to Q(FRAC_BITS)
    localparam logic signed [DATA_WIDTH-1:0] S_MAX = (1 <<< (DATA_WIDTH-1)) - 1;
    localparam logic signed [DATA_WIDTH-1:0] S_MIN = - (1 <<< (DATA_WIDTH-1));

    // Function: promote a DATA_WIDTH bias (Q(FRAC_BITS)) into accumulator domain (Q(FRAC_BITS)<<FRAC_BITS)
    function automatic logic signed [ACCW-1:0]
    bias_to_accq2f(input logic signed [DATA_WIDTH-1:0] b);
        // Sign-extend to ACCW then left-shift by FRAC_BITS so later >>> FRAC_BITS restores scale.
        return ({{(ACCW-DATA_WIDTH){b[DATA_WIDTH-1]}}, b}) <<< FRAC_BITS;
    endfunction

    // Function: safe input fetch with implicit zero padding outside [0, IMG_SIZE)
    function automatic logic signed [DATA_WIDTH-1:0]
    in_at(input integer ch, input integer r, input integer c);
        if (r < 0 || r >= IMG_SIZE || c < 0 || c >= IMG_SIZE) return '0;            // zero-pad
        else                                                  return input_feature[ch][r][c];
    endfunction

    // debug: count how many writes clipped at +/- saturation
    logic [31:0] sat_pos_cnt, sat_neg_cnt;

    //==========================================================================
    // Main sequential process: FSM + counters + datapath
    //==========================================================================
    always_ff @(posedge clk) begin
        if (reset) begin
            // Reset state, control, datapath, and debug counters
            state <= IDLE; done <= 1'b0;
            oc <= 0; orow <= 0; ocol <= 0;
            ic <= 0; kr <= 0; kc <= 0;
            acc <= '0; prod <= '0;
            sat_pos_cnt <= 0; sat_neg_cnt <= 0;

        end else begin
            done <= 1'b0; // default

            unique case (state)

              //==============================================================
              // IDLE: Wait for start; initialise indices and seed acc with bias
              //==============================================================
              IDLE: if (start) begin
                        oc<=0; orow<=0; ocol<=0;
                        ic<=0; kr<=0; kc<=0;
                        acc <= bias_to_accq2f(biases[0]); // bias preload for oc=0
                        state <= MAC;
                    end

              //==============================================================
              // MAC: Multiply-Accumulate over all taps for current (oc,orow,ocol)
              //  Compute coordinates of input sample under kernel window.
              //  Blocking assign for prod so its value is used in the same cycle.
              //  Non-blocking add into acc so the sum commits on this clock edge.
              //  Advance kc→kr→ic; on final tap, proceed to WRITE next cycle.
              //==============================================================
              MAC: begin
                    integer ir  = orow + kr - PAD;    // input row for current tap
                    integer icc = ocol + kc - PAD;    // input col for current tap

                    // Product (Q(FRAC_BITS)*Q(FRAC_BITS) → ~Q(2*FRAC_BITS))
                    prod = in_at(ic, ir, icc) * weights[oc][ic][kr][kc]; // blocking
                    acc  <= acc + prod;                                   // non-blocking

                    // Tap counters
                    if (kc == KERNEL-1) begin
                        kc <= 0;
                        if (kr == KERNEL-1) begin
                            kr <= 0;
                            if (ic == IN_CHANNELS-1) state <= WRITE;  // all taps done
                            else                      ic    <= ic + 1;
                        end else kr <= kr + 1;
                    end else kc <= kc + 1;
                  end

              //==============================================================
              // WRITE: Scale, saturate, store; then advance spatial and (maybe) channel
              //  Right-shift to return from accumulator scale to Q(FRAC_BITS).
              //  Saturate to DATA_WIDTH and update clip counters.
              //  Advance ocol → orow → oc with wrap; preload acc with next bias.
              //==============================================================
              WRITE: begin
                    logic signed [ACCW-1:0]      shifted;
                    logic signed [DATA_WIDTH-1:0] res;

                    shifted = acc >>> FRAC_BITS; // back to Q(FRAC_BITS)

                    // Saturate to signed DATA_WIDTH
                    if (shifted > S_MAX) begin
                        res <= S_MAX; sat_pos_cnt <= sat_pos_cnt + 1;
                    end else if (shifted < S_MIN) begin
                        res <= S_MIN; sat_neg_cnt <= sat_neg_cnt + 1;
                    end else res <= shifted[DATA_WIDTH-1:0];

                    // Commit pixel
                    out_feature[oc][orow][ocol] <= res;

                    // Spatial/channel progression and bias preload for next MAC pass
                    if (ocol == IMG_SIZE-1) begin
                        ocol <= 0;
                        if (orow == IMG_SIZE-1) begin
                            orow <= 0;
                            if (oc == OUT_CHANNELS-1) begin
                                state <= FINISH; // all outputs done
                            end else begin
                                oc <= oc + 1;
                                ic<=0; kr<=0; kc<=0;
                                acc <= bias_to_accq2f(biases[oc+1]); // preload next channel bias
                                state <= MAC;
                            end
                        end else begin
                            orow <= orow + 1;
                            ic<=0; kr<=0; kc<=0;
                            acc <= bias_to_accq2f(biases[oc]);       // same oc, next row
                            state <= MAC;
                        end
                    end else begin
                        ocol <= ocol + 1;
                        ic<=0; kr<=0; kc<=0;
                        acc <= bias_to_accq2f(biases[oc]);           // same oc, next col
                        state <= MAC;
                    end
                  end

              //==============================================================
              // FINISH: Pulse done and return to IDLE
              //==============================================================
              FINISH: begin
                    done  <= 1'b1;
                    state <= IDLE;
                  end
            endcase
        end
    end
endmodule