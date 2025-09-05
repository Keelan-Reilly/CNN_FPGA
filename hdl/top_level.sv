//======================================================================
// top_level.sv — Complete CNN Inference Pipeline with UART I/O
//----------------------------------------------------------------------
// What this module does:
//   • Implements a full fixed-point CNN pipeline in hardware.
//   • Takes pixel data in over UART (one byte per pixel).
//   • Runs it through convolution → ReLU → pooling → flatten → dense → argmax.
//   • Outputs the predicted digit both as a binary index and as an ASCII
//     character over UART.
//
// Key points:
//   • Input:  UART byte stream of IMG_SIZE×IMG_SIZE grayscale pixels.
//   • Output: predicted digit (0–9) on `predicted_digit`, and ASCII digit
//             sent back over UART.
//   • Control: an FSM controller starts each stage once the previous stage is
//     finished.
//   • Performance timestamps are printed (sim only) to show cycle counts.
//======================================================================

module top_level #(
    parameter int DATA_WIDTH   = 16,
    parameter int FRAC_BITS    = 7,
    parameter int IMG_SIZE     = 28,
    parameter int IN_CHANNELS  = 1,
    parameter int OUT_CHANNELS = 8,
    parameter int NUM_CLASSES  = 10,
    parameter int CLK_FREQ_HZ  = 100_000_000,
    parameter int BAUD_RATE    = 115_200,
    localparam int CLKS_PER_BIT = CLK_FREQ_HZ / BAUD_RATE
)(
    input  logic clk,
    input  logic reset,
    input  logic uart_rx_i,
    output logic uart_tx_o,
    output logic [3:0] predicted_digit   // final CNN prediction (0–9)
);

    // ---------------- Flattened geometry helpers ----------------
    localparam int IF_SZ   = IN_CHANNELS  * IMG_SIZE * IMG_SIZE;
    localparam int OF_SZ   = OUT_CHANNELS * IMG_SIZE * IMG_SIZE;
    localparam int POOLED  = IMG_SIZE/2;
    localparam int PO_SZ   = OUT_CHANNELS * POOLED * POOLED;
    localparam int FLAT    = PO_SZ;

    // ---------------- Flattened feature buffers ----------------
    logic signed [DATA_WIDTH-1:0] ifmap_flat [0:IF_SZ-1];
    logic signed [DATA_WIDTH-1:0] conv_out_flat [0:OF_SZ-1];
    logic signed [DATA_WIDTH-1:0] relu_out_flat [0:OF_SZ-1];
    logic signed [DATA_WIDTH-1:0] pool_out_flat [0:PO_SZ-1];
    logic signed [DATA_WIDTH-1:0] flat_vec      [0:FLAT-1];
    logic signed [DATA_WIDTH-1:0] logits        [0:NUM_CLASSES-1];

    // ---------------- UART RX (load pixels) ----------------
    logic       rx_dv;
    logic [7:0] rx_byte;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) RX (
        .clk, .reset, .rx(uart_rx_i), .rx_dv(rx_dv), .rx_byte(rx_byte)
    );

    // Store received pixels into ifmap_flat (channel 0), row by row
    integer r, c;
    logic frame_loaded;
    function automatic int idx3(input int ch, input int row, input int col, input int H, input int W);
        return (ch*H + row)*W + col;
    endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
            r <= 0; c <= 0; frame_loaded <= 1'b0;
        end else begin
            frame_loaded <= 1'b0;
            if (rx_dv) begin
                ifmap_flat[idx3(0, r, c, IMG_SIZE, IMG_SIZE)] <= $signed({8'd0, rx_byte}) <<< FRAC_BITS;
                if (c == IMG_SIZE-1) begin
                    c <= 0;
                    if (r == IMG_SIZE-1) begin
                        r <= 0;
                        frame_loaded <= 1'b1; // whole frame loaded
                    end else r <= r + 1;
                end else c <= c + 1;
            end
        end
    end

    // ---------------- CNN stages ----------------
    logic conv_start, relu_start, pool_start, dense_start, tx_start;
    logic conv_done,  relu_done,  pool_done,  dense_done;

    // Convolution stage (loads weights/biases internally from .mem)
    conv2d #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS), .OUT_CHANNELS(OUT_CHANNELS),
        .KERNEL(3), .IMG_SIZE(IMG_SIZE),
        .WEIGHTS_FILE("weights/conv1_weights.mem"),  // remove directory in vivado
        .BIASES_FILE ("weights/conv1_biases.mem")    // remove directory in vivado
    ) u_conv (
        .clk, .reset, .start(conv_start),
        .input_feature_flat(ifmap_flat),
        .out_feature_flat(conv_out_flat),
        .done(conv_done)
    );

    // ReLU activation (flattened)
    relu #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .CHANNELS(OUT_CHANNELS), .IMG_SIZE(IMG_SIZE)
    ) u_relu (
        .clk, .reset, .start(relu_start),
        .in_feature_flat(conv_out_flat),
        .out_feature_flat(relu_out_flat),
        .done(relu_done)
    );

    // Max pooling (2×2), flattened
    maxpool #(
        .DATA_WIDTH(DATA_WIDTH), .CHANNELS(OUT_CHANNELS),
        .IN_SIZE(IMG_SIZE), .POOL(2)
    ) u_pool (
        .clk, .reset, .start(pool_start),
        .in_feature_flat(relu_out_flat),
        .out_feature_flat(pool_out_flat),
        .done(pool_done)
    );

    // Flatten stage: simple linear copy to preserve pipeline / FSM timing
    logic flattening, flat_done;
    integer fi_idx;
    always_ff @(posedge clk) begin
        if (reset) begin
            flattening <= 1'b0; flat_done <= 1'b0; fi_idx <= 0;
        end else begin
            flat_done <= 1'b0;
            if (pool_done && !flattening) begin
                flattening <= 1'b1;
                fi_idx <= 0;
            end
            if (flattening) begin
                flat_vec[fi_idx] <= pool_out_flat[fi_idx];
                if (fi_idx == FLAT-1) begin
                    flattening <= 1'b0;
                    flat_done  <= 1'b1;
                end else fi_idx <= fi_idx + 1;
            end
        end
    end

    // Dense (loads weights/biases internally)
    dense #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .IN_DIM(FLAT), .OUT_DIM(NUM_CLASSES),
        .WEIGHTS_FILE("weights/fc1_weights.mem"), // remove directory in vivado
        .BIASES_FILE ("weights/fc1_biases.mem")  // remove directory in vivado
    ) u_dense (
        .clk, .reset, .start(dense_start),
        .in_vec(flat_vec),
        .out_vec(logits), .done(dense_done)
    );

    // Argmax layer: pick index of max logit = predicted class
    logic argmax_done;
    localparam int IDXW = (NUM_CLASSES <= 1) ? 1 : $clog2(NUM_CLASSES);
    argmax #(.DATA_WIDTH(DATA_WIDTH), .DIM(NUM_CLASSES), .IDXW(IDXW)) u_argmax (
        .clk, .reset, .start(dense_done),
        .vec(logits), .idx(predicted_digit), .done(argmax_done)
    );

    // ---------------- UART TX (send prediction) ----------------
    logic tx_ready = argmax_done;
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

    // FSM controller unchanged
    logic pipeline_busy;
    fsm_controller ctrl (
        .clk, .reset,
        .frame_loaded(frame_loaded),
        .conv_done, .relu_done, .pool_done, .flat_done, .dense_done,
        .tx_ready, .tx_busy,
        .conv_start, .relu_start, .pool_start, .dense_start, .tx_start,
        .busy(pipeline_busy)
    );

    // ---------------- Performance (sim only) ----------------
    logic [63:0] cycle_ctr;
    logic [63:0] t_start, t_conv, t_relu, t_pool, t_flat, t_dense, t_argmax, t_tx;

    always_ff @(posedge clk) begin
        if (reset) cycle_ctr <= 64'd0;
        else       cycle_ctr <= cycle_ctr + 64'd1;
    end

    logic tx_start_q;
    always_ff @(posedge clk) begin
        if (reset) begin
            t_start<=0; t_conv<=0; t_relu<=0; t_pool<=0;
            t_flat<=0; t_dense<=0; t_argmax<=0; t_tx<=0;
            tx_start_q <= 1'b0;
        end else begin
            tx_start_q <= tx_start;
            if (frame_loaded) t_start  <= cycle_ctr;
            if (conv_done)    t_conv   <= cycle_ctr;
            if (relu_done)    t_relu   <= cycle_ctr;
            if (pool_done)    t_pool   <= cycle_ctr;
            if (flat_done)    t_flat   <= cycle_ctr;
            if (dense_done)   t_dense  <= cycle_ctr;
            if (argmax_done)  t_argmax <= cycle_ctr;
            if (tx_start)     t_tx     <= cycle_ctr;
            if (tx_start_q) begin
                $display("---- Performance Report ----");
                $display("Frame cycles: %0d", t_tx - t_start);
                $display(" conv  = %0d",      t_conv   - t_start);
                $display(" relu  = %0d",      t_relu   - t_conv);
                $display(" pool  = %0d",      t_pool   - t_relu);
                $display(" flat  = %0d",      t_flat   - t_pool);
                $display(" dense = %0d",      t_dense  - t_flat);
                $display(" argmx = %0d",      t_tx     - t_dense);
                $display("----------------------------");
            end
        end
    end
endmodule
