// conv2d.v
// Parameterised fixed-point 2D convolution (stride=1, padding=1) for MNIST-style input.
// Supports multiple output channels and single input channel (can be extended).
// Input and weight formats are signed fixed-point (Q format with FRAC_BITS fractional bits).

module conv2d #(
    parameter DATA_WIDTH     = 16,   // bit-width of input / weight (signed)
    parameter FRAC_BITS      = 7,    // number of fractional bits in fixed-point
    parameter IN_CHANNELS    = 1,    // input depth
    parameter OUT_CHANNELS   = 8,    // number of filters
    parameter KERNEL         = 3,    // kernel size (assumed square, odd)
    parameter IMG_SIZE       = 28    // input spatial dimension (assumes square)
)(
    input  logic                     clk,
    input  logic                     reset,    // synchronous reset
    input  logic                     start,    // pulse to begin convolution
    // interface to input feature map (assumed preloaded into internal buffer or external BRAM)
    input  logic signed [DATA_WIDTH-1:0] input_feature [0:IN_CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1],
    // weight memory: [out_chan][in_chan][k][k] loaded externally via $readmemh into an array wrapper
    input  logic signed [DATA_WIDTH-1:0] weights [0:OUT_CHANNELS-1][0:IN_CHANNELS-1][0:KERNEL-1][0:KERNEL-1],
    input  logic signed [DATA_WIDTH-1:0] biases  [0:OUT_CHANNELS-1],
    output logic signed [DATA_WIDTH-1:0] out_feature [0:OUT_CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1],
    output logic                     done      // asserted when full output map is ready
);

    // Derived constants
    localparam signed_accum_width = DATA_WIDTH*2 + $clog2(KERNEL*KERNEL*IN_CHANNELS); // safe accumulation width
    localparam PAD = (KERNEL-1)/2;

    // FSM states
    typedef enum logic [1:0] {
        IDLE,
        COMPUTE,
        WRITEBACK,
        FINISH
    } state_t;

    state_t state;
    integer oc, ic, i, j, ki, kj;

    // Accumulator (wide to hold sum of products + bias)
    logic signed [signed_accum_width-1:0] accum;

    // Output coordinates
    integer out_row, out_col;

    // Temporary product
    logic signed [2*DATA_WIDTH-1:0] mult; // product of input * weight

    // Scaling helper: after accumulation we need to right-shift by FRAC_BITS to return to same fixed point
    // Clipping to avoid overflow when reducing width back to DATA_WIDTH
    function logic signed [DATA_WIDTH-1:0] scale_and_saturate(input logic signed [signed_accum_width-1:0] val);
        logic signed [signed_accum_width-1:0] shifted;
        logic signed [DATA_WIDTH-1:0] result;
        begin
            // apply bias already included; shift down fractional bits
            shifted = val >>> FRAC_BITS; // arithmetic shift
            // saturation: clamp to signed DATA_WIDTH range
            if (shifted > $signed({1'b0, {(DATA_WIDTH-1){1'b1}}})) begin
                result = {1'b0, {(DATA_WIDTH-1){1'b1}}}; // max positive
            end else if (shifted < $signed({1'b1, {(DATA_WIDTH-1){1'b0}}})) begin
                result = {1'b1, {(DATA_WIDTH-1){1'b0}}}; // max negative (two's complement)
            end else begin
                result = shifted[DATA_WIDTH-1:0];
            end
            scale_and_saturate = result;
        end
    endfunction

    // Control logic
    always_ff @(posedge clk) begin
        if (reset) begin
            state     <= IDLE;
            done      <= 0;
            out_row   <= 0;
            out_col   <= 0;
            oc        <= 0;
            ic        <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        // begin convolution over all output channels and spatial positions
                        oc       <= 0;
                        out_row  <= 0;
                        out_col  <= 0;
                        state    <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // compute output at [oc][out_row][out_col]
                    accum = 0;
                    // loop over input channels and kernel window
                    for (ic = 0; ic < IN_CHANNELS; ic = ic + 1) begin
                        for (ki = 0; ki < KERNEL; ki = ki + 1) begin
                            for (kj = 0; kj < KERNEL; kj = kj + 1) begin
                                integer in_r = out_row + ki - PAD;
                                integer in_c = out_col + kj - PAD;
                                logic signed [DATA_WIDTH-1:0] in_val;
                                if (in_r < 0 || in_r >= IMG_SIZE || in_c < 0 || in_c >= IMG_SIZE) begin
                                    in_val = 0; // zero padding
                                end else begin
                                    in_val = input_feature[ic][in_r][in_c];
                                end
                                mult = in_val * weights[oc][ic][ki][kj]; // produce 2*DATA_WIDTH bits
                                // Align product to accumulator: product has 2*DATA_WIDTH bits, we shift to match scaling
                                accum = accum + mult; // accumulation in wide domain
                            end
                        end
                    end
                    // add bias (bias is in same fixed-point Q format; extend to accumulator width)
                    accum = accum + ({{(signed_accum_width-DATA_WIDTH){biases[oc][DATA_WIDTH-1]}}, biases[oc]}); // sign-extend bias

                    // write scaled, saturated result to output
                    out_feature[oc][out_row][out_col] <= scale_and_saturate(accum);

                    // advance spatial counters
                    if (out_col == IMG_SIZE-1) begin
                        out_col <= 0;
                        if (out_row == IMG_SIZE-1) begin
                            out_row <= 0;
                            // finished this output channel, move to next
                            if (oc == OUT_CHANNELS-1) begin
                                state <= FINISH;
                            end else begin
                                oc <= oc + 1;
                            end
                        end else begin
                            out_row <= out_row + 1;
                        end
                    end else begin
                        out_col <= out_col + 1;
                    end
                end

                FINISH: begin
                    done <= 1;
                    state <= IDLE; // wait for next start
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule