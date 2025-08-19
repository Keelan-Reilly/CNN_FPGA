// top_level.sv
module top_level #(
    parameter DATA_WIDTH   = 16,
    parameter FRAC_BITS    = 7,
    parameter IMG_SIZE     = 28,
    parameter IN_CHANNELS  = 1,
    parameter OUT_CHANNELS = 8,
    parameter NUM_CLASSES  = 10,
    parameter CLKS_PER_BIT = 434  // set for your sim clock vs 115200 baud
)(
    input  logic clk,
    input  logic reset,
    input  logic uart_rx_i,
    output logic uart_tx_o,

    output logic [3:0] predicted_digit
);
    // ---------------- Memories / feature maps ----------------
    // Input (one channel)
    logic signed [DATA_WIDTH-1:0] ifmap   [0:IN_CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1];

    // After conv/relu (28x28 * OUT_CHANNELS)
    logic signed [DATA_WIDTH-1:0] conv_out[0:OUT_CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1];
    logic signed [DATA_WIDTH-1:0] relu_out[0:OUT_CHANNELS-1][0:IMG_SIZE-1][0:IMG_SIZE-1];

    // After pool (14x14 * OUT_CHANNELS)
    localparam POOLED = IMG_SIZE/2;
    logic signed [DATA_WIDTH-1:0] pool_out[0:OUT_CHANNELS-1][0:POOLED-1][0:POOLED-1];

    // Flattened vector for dense
    localparam FLAT = OUT_CHANNELS*POOLED*POOLED; // 8*14*14=1568
    logic signed [DATA_WIDTH-1:0] flat_vec[0:FLAT-1];

    // Dense output (logits)
    logic signed [DATA_WIDTH-1:0] logits[0:NUM_CLASSES-1];

    // ---------------- Weights / biases ----------------
    // conv1: [out][in][3][3]
    logic signed [DATA_WIDTH-1:0] conv_w [0:OUT_CHANNELS-1][0:IN_CHANNELS-1][0:2][0:2];
    logic signed [DATA_WIDTH-1:0] conv_b [0:OUT_CHANNELS-1];

    // dense: [out][in]
    logic signed [DATA_WIDTH-1:0] dense_w[0:NUM_CLASSES-1][0:FLAT-1];
    logic signed [DATA_WIDTH-1:0] dense_b[0:NUM_CLASSES-1];

`ifdef VERILATOR
    initial begin
        $readmemh("weights/conv1_weights.mem", conv_w);
        $readmemh("weights/conv1_biases.mem",  conv_b);
        $readmemh("weights/fc1_weights.mem",   dense_w);
        $readmemh("weights/fc1_biases.mem",    dense_b);
    end
`endif

    // ---------------- UART: receive 28x28=784 bytes ----------------
    logic       rx_dv;
    logic [7:0] rx_byte;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) RX (.clk, .reset, .rx(uart_rx_i), .rx_dv(rx_dv), .rx_byte(rx_byte));

    // buffer pixels into ifmap[0][r][c] (unsigned 0..255 -> fixed-point)
    integer r, c;
    logic frame_loaded;
    always_ff @(posedge clk) begin
        if (reset) begin r<=0; c<=0; frame_loaded<=0; end
        else begin
            frame_loaded <= 1'b0;
            if (rx_dv) begin
                ifmap[0][r][c] <= $signed({8'd0, rx_byte}) <<< FRAC_BITS; // Q format load
                if (c == IMG_SIZE-1) begin
                    c <= 0;
                    if (r == IMG_SIZE-1) begin r <= 0; frame_loaded <= 1'b1; end
                    else r <= r + 1;
                end else c <= c + 1;
            end
        end
    end

    // ---------------- Stage instances ----------------
    logic conv_start, relu_start, pool_start, dense_start, tx_start;
    logic conv_done,  relu_done,  pool_done,  dense_done;

    conv2d #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS), .OUT_CHANNELS(OUT_CHANNELS),
        .KERNEL(3), .IMG_SIZE(IMG_SIZE)
    ) u_conv (
        .clk, .reset, .start(conv_start),
        .input_feature(ifmap),
        .weights(conv_w), .biases(conv_b),
        .out_feature(conv_out),
        .done(conv_done)
    );

    relu #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .CHANNELS(OUT_CHANNELS), .IMG_SIZE(IMG_SIZE)
    ) u_relu (
        .clk, .reset, .start(relu_start),
        .in_feature(conv_out), .out_feature(relu_out),
        .done(relu_done)
    );

    maxpool #(
        .DATA_WIDTH(DATA_WIDTH), .CHANNELS(OUT_CHANNELS), .IN_SIZE(IMG_SIZE), .POOL(2)
    ) u_pool (
        .clk, .reset, .start(pool_start),
        .in_feature(relu_out), .out_feature(pool_out),
        .done(pool_done)
    );

    // declare flat_done pulse instead of 'flattened' lingering signal
    logic flattening;
    logic flat_done;

    // Flatten control counters
    integer fi_c, fi_r, fi_q, fi_idx;

    // in the flattening block, make flat_done a 1-cycle pulse when flatten finishes
    always_ff @(posedge clk) begin
        if (reset) begin
            flattening <= 1'b0; flat_done <= 1'b0;
            fi_c <= 0; fi_r <= 0; fi_q <= 0; fi_idx <= 0;
        end else begin
            flat_done <= 1'b0;  // default

            if (pool_done && !flattening) begin
                flattening <= 1'b1;
                fi_c <= 0; fi_r <= 0; fi_q <= 0; fi_idx <= 0;
            end

            if (flattening) begin
                flat_vec[fi_idx] <= pool_out[fi_c][fi_r][fi_q];
                if (fi_q == POOLED-1) begin
                    fi_q <= 0;
                    if (fi_r == POOLED-1) begin
                        fi_r <= 0;
                        if (fi_c == OUT_CHANNELS-1) begin
                            flattening <= 1'b0;
                            flat_done  <= 1'b1;  // <-- one-cycle pulse
                        end else fi_c <= fi_c + 1;
                    end else fi_r <= fi_r + 1;
                end else fi_q <= fi_q + 1;
                fi_idx <= fi_idx + 1;
            end
        end
    end

    dense #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS), .IN_DIM(FLAT), .OUT_DIM(NUM_CLASSES)
    ) u_dense (
        .clk, .reset, .start(dense_start),
        .in_vec(flat_vec), .weights(dense_w), .biases(dense_b),
        .out_vec(logits), .done(dense_done)
    );

    // Argmax
    logic argmax_done;
    argmax #(.DATA_WIDTH(DATA_WIDTH), .DIM(NUM_CLASSES)) u_argmax (
        .clk, .reset, .start(dense_done), // start when dense finishes
        .vec(logits), .idx(predicted_digit), .done(argmax_done)
    );

    // ---------------- FSM to run the chain ----------------
    // tx_ready = argmax_done (we send one byte)
    logic tx_ready = argmax_done;
    // controller instance: add flat_done input, remove unused .busy()
    fsm_controller ctrl (
        .clk, .reset,
        .frame_loaded(frame_loaded),
        .conv_done, .relu_done, .pool_done, .flat_done, .dense_done, .tx_ready,
        .conv_start, .relu_start, .pool_start, .dense_start, .tx_start,
        .busy()
    );

    // ---------------- UART TX: send ASCII '0'+digit ----------------
    logic        tx_busy;
    logic        tx_dv;
    logic [7:0]  tx_byte;

    assign tx_dv   = tx_start;
    localparam logic [7:0] ASCII_0 = 8'h30;
    assign tx_byte = ASCII_0 + {4'b0000, predicted_digit};

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) TX (
        .clk, .reset, .tx_dv(tx_dv), .tx_byte(tx_byte),
        .tx(uart_tx_o), .tx_busy(tx_busy)
    );
endmodule