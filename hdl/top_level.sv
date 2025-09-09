//======================================================================
// top_level.sv — CNN pipeline with BRAMs, lint-clean wiring
//======================================================================
(* keep_hierarchy = "yes" *)
module top_level #(
    parameter int DATA_WIDTH   = 16,
    parameter int FRAC_BITS    = 7,
    parameter int IMG_SIZE     = 28,
    parameter int IN_CHANNELS  = 1,
    parameter int OUT_CHANNELS = 8,
    parameter int NUM_CLASSES  = 10,
    parameter int CLK_FREQ_HZ  = 50_000_000,
    parameter int BAUD_RATE    = 115_200
)(
    input  logic clk,
    input  logic reset,
    input  logic uart_rx_i,
    output logic uart_tx_o,
    output logic [3:0] predicted_digit
);
    localparam int CLKS_PER_BIT = CLK_FREQ_HZ / BAUD_RATE;

`ifdef SYNTHESIS
    localparam string CONV_W_FILE = "conv1_weights.mem";
    localparam string CONV_B_FILE = "conv1_biases.mem";
    localparam string FC_W_FILE   = "fc1_weights.mem";
    localparam string FC_B_FILE   = "fc1_biases.mem";
`else
    localparam string CONV_W_FILE = "weights/conv1_weights.mem";
    localparam string CONV_B_FILE = "weights/conv1_biases.mem";
    localparam string FC_W_FILE   = "weights/fc1_weights.mem";
    localparam string FC_B_FILE   = "weights/fc1_biases.mem";
`endif

    // Geometry
    localparam int IF_SZ  = IN_CHANNELS*IMG_SIZE*IMG_SIZE;         // 784
    localparam int OF_SZ  = OUT_CHANNELS*IMG_SIZE*IMG_SIZE;        // 6272
    localparam int POOLED = IMG_SIZE/2;                            // 14
    localparam int PO_SZ  = OUT_CHANNELS*POOLED*POOLED;            // 1568

    localparam int IF_AW = $clog2(IF_SZ);
    localparam int OF_AW = $clog2(OF_SZ);
    localparam int PO_AW = $clog2(PO_SZ);

    typedef logic [IF_AW-1:0] if_addr_t;
    typedef logic [OF_AW-1:0] of_addr_t;

    // UART RX: load pixels into IFMAP BRAM
    logic       rx_dv;  logic [7:0] rx_byte;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) RX (
        .clk, .reset, .rx(uart_rx_i), .rx_dv(rx_dv), .rx_byte(rx_byte)
    );

    // ---- 8-bit -> Q7 LUT: scales [0..255] -> Q7 fixed-point ----
    logic signed [DATA_WIDTH-1:0] pix_q7_lut [0:255];
    initial begin : init_pix_q7
        integer k; 
        integer v;
        for (k = 0; k < 256; k++) begin
            v = (k * (1 << FRAC_BITS) + 127) / 255;  // rounded
            pix_q7_lut[k] = $signed(v[DATA_WIDTH-1:0]);
        end
    end

    // Address helpers returned in the right width (avoid trunc warns)
    function automatic if_addr_t lin3_if(input int ch, input int row, input int col);
        return if_addr_t'(((ch*IMG_SIZE + row)*IMG_SIZE + col));
    endfunction

    function automatic of_addr_t lin3_of(input int ch, input int row, input int col);
        return of_addr_t'(((ch*IMG_SIZE + row)*IMG_SIZE + col));
    endfunction

        // Map a logical HWC (row, col, ch) linear index to the physical CHW BRAM index
    function automatic [PO_AW-1:0] hwc_to_chw(input [PO_AW-1:0] a);
        int pix, ch, row, col;
        pix = a / OUT_CHANNELS;            // which (row,col) pixel
        ch  = a % OUT_CHANNELS;            // channel within that pixel
        row = pix / POOLED;
        col = pix % POOLED;
        return PO_AW'(((ch*POOLED + row)*POOLED) + col); // CHW address
    endfunction

    // Frame loader
    integer r, c;
    logic frame_loaded;
    always_ff @(posedge clk) begin
        if (reset) begin
            r<=0; c<=0; frame_loaded<=1'b0;
        end else begin
            frame_loaded <= 1'b0;
            if (rx_dv) begin
                if (c == IMG_SIZE-1) begin
                    c <= 0;
                    if (r == IMG_SIZE-1) begin
                        r <= 0;
                        frame_loaded <= 1'b1;
                    end else r <= r + 1;
                end else c <= c + 1;
            end
        end
    end

    // ---------------- BRAMs ----------------

    // IFMAP: UART writes (A), conv2d reads (B)
    logic                 if_a_en, if_a_we;
    logic [IF_AW-1:0]     if_a_addr;
    logic signed [DATA_WIDTH-1:0] if_a_din;
    logic                 if_b_en;
    logic [IF_AW-1:0]     if_b_addr;
    logic signed [DATA_WIDTH-1:0] if_b_q;

    assign if_a_en   = rx_dv;
    assign if_a_we   = rx_dv;
    assign if_a_addr = lin3_if(0, r, c);
    assign if_a_din  = pix_q7_lut[rx_byte];

    bram_sdp #(.DW(DATA_WIDTH), .DEPTH(IF_SZ)) ifmap_mem (
      .clk   (clk),
      .a_en  (if_a_en), .a_we(if_a_we), .a_addr(if_a_addr), .a_din(if_a_din),
      .b_en  (if_b_en), .b_addr(if_b_addr), .b_dout(if_b_q)
    );

    // CONV buffer: Port A = reader (relu or pool), Port B = writer (conv or relu writeback)
    logic                 convA_en;
    logic [OF_AW-1:0]     convA_addr;
    logic signed [DATA_WIDTH-1:0] convA_q;

    logic                 convB_en, convB_we;
    logic [OF_AW-1:0]     convB_addr;
    logic signed [DATA_WIDTH-1:0] convB_din, convB_q_unused;

    bram_tdp #(.DW(DATA_WIDTH), .DEPTH(OF_SZ)) conv_buf (
      .clk(clk),
      // Port A (read)
      .a_en  (convA_en),
      .a_we  (1'b0),
      .a_addr(convA_addr),
      .a_din ('0),
      .a_dout(convA_q),
      // Port B (write)
      .b_en  (convB_en),
      .b_we  (convB_we),
      .b_addr(convB_addr),
      .b_din (convB_din),
      .b_dout(convB_q_unused) // tie off to silence PINCONNECTEMPTY
    );

    // POOL buffer: Port A = pool writer, Port B = dense reader
    logic                 poolA_en, poolA_we;
    logic [PO_AW-1:0]     poolA_addr;
    logic signed [DATA_WIDTH-1:0] poolA_din;

    logic                 poolB_en;
    logic [PO_AW-1:0]     poolB_addr;
    logic signed [DATA_WIDTH-1:0] poolB_q;

    bram_sdp #(.DW(DATA_WIDTH), .DEPTH(PO_SZ)) pool_mem (
      .clk   (clk),
      .a_en  (poolA_en), .a_we(poolA_we), .a_addr(poolA_addr), .a_din(poolA_din),
      .b_en  (poolB_en), .b_addr(poolB_addr), .b_dout(poolB_q)
    );

    // ---------------- CNN stages ----------------
    logic conv_start, relu_start, pool_start, dense_start, tx_start;
    logic conv_done,  relu_done,  pool_done,  dense_done;

    // conv2d <-> IFMAP/CONV BRAM signals
    logic [IF_AW-1:0] conv_if_addr;  logic conv_if_en;
    logic [OF_AW-1:0] conv_w_addr;   logic conv_w_en;
    logic             conv_w_we;
    logic signed [DATA_WIDTH-1:0] conv_w_d;

    conv2d #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS), .OUT_CHANNELS(OUT_CHANNELS),
        .KERNEL(3), .IMG_SIZE(IMG_SIZE),
        .WEIGHTS_FILE(CONV_W_FILE), .BIASES_FILE(CONV_B_FILE)
    ) u_conv (
        .clk, .reset, .start(conv_start),
        .if_addr (conv_if_addr),
        .if_en   (conv_if_en),
        .if_q    (if_b_q),
        .conv_addr (conv_w_addr),
        .conv_en   (conv_w_en),
        .conv_we   (conv_w_we),
        .conv_d    (conv_w_d),
        .done(conv_done)
    );

    // ReLU in-place over CONV buffer (read A, write B)
    logic [OF_AW-1:0] relu_r_addr; logic relu_r_en;
    logic [OF_AW-1:0] relu_w_addr; logic relu_w_en, relu_w_we;
    logic signed [DATA_WIDTH-1:0] relu_w_d;

    relu #(
        .DATA_WIDTH(DATA_WIDTH), .CHANNELS(OUT_CHANNELS), .IMG_SIZE(IMG_SIZE)
    ) u_relu (
        .clk, .reset, .start(relu_start),
        .conv_r_addr(relu_r_addr), .conv_r_en(relu_r_en), .conv_r_q(convA_q),
        .conv_w_addr(relu_w_addr), .conv_w_en(relu_w_en), .conv_w_we(relu_w_we), .conv_w_d(relu_w_d),
        .done(relu_done)
    );

    // MaxPool reads CONV buffer (A), writes POOL buffer (A)
    logic [OF_AW-1:0] pool_conv_r_addr; logic pool_conv_r_en;
    maxpool #(
        .DATA_WIDTH(DATA_WIDTH), .CHANNELS(OUT_CHANNELS), .IN_SIZE(IMG_SIZE), .POOL(2)
    ) u_pool (
        .clk, .reset, .start(pool_start),
        .conv_addr(pool_conv_r_addr), .conv_en(pool_conv_r_en), .conv_q(convA_q),
        .pool_addr(poolA_addr), .pool_en(poolA_en), .pool_we(poolA_we), .pool_d(poolA_din),
        .done(pool_done)
    );

    // Flat done = 1-cycle delayed pool_done
    logic flat_done_q;  wire flat_done = flat_done_q;
    always_ff @(posedge clk) begin
        if (reset) flat_done_q <= 1'b0;
        else       flat_done_q <= pool_done;
    end

    // Dense reads POOL buffer (B), outputs logits
    logic signed [DATA_WIDTH-1:0] logits [0:NUM_CLASSES-1];
    logic [PO_AW-1:0] dense_in_addr; logic dense_in_en;

    dense #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .IN_DIM(PO_SZ), .OUT_DIM(NUM_CLASSES),
        .WEIGHTS_FILE(FC_W_FILE), .BIASES_FILE(FC_B_FILE),
        .POST_SHIFT(4)
    ) u_dense (
        .clk, .reset, .start(dense_start),
        .in_addr(dense_in_addr), .in_en(dense_in_en), .in_q(poolB_q),
        .out_vec(logits), .done(dense_done)
    );

    // Argmax
    logic argmax_done;
    localparam int IDXW = (NUM_CLASSES <= 1) ? 1 : $clog2(NUM_CLASSES);
    argmax #(.DATA_WIDTH(DATA_WIDTH), .DIM(NUM_CLASSES), .IDXW(IDXW)) u_argmax (
        .clk, .reset, .start(dense_done),
        .vec(logits), .idx(predicted_digit), .done(argmax_done)
    );

    // UART TX
    logic tx_ready = argmax_done;
    logic tx_busy, tx_dv; logic [7:0] tx_byte;
    assign tx_dv = tx_start;
    localparam logic [7:0] ASCII_0 = 8'h30;
    assign tx_byte = ASCII_0 + {4'b0000, predicted_digit};

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) TX (
        .clk, .reset, .tx_dv(tx_dv), .tx_byte(tx_byte),
        .tx(uart_tx_o), .tx_busy(tx_busy)
    );

    // Controller
    logic pipeline_busy;
    fsm_controller ctrl (
        .clk, .reset,
        .frame_loaded(frame_loaded),
        .conv_done, .relu_done, .pool_done, .flat_done, .dense_done,
        .tx_ready, .tx_busy,
        .conv_start, .relu_start, .pool_start, .dense_start, .tx_start,
        .busy(pipeline_busy)
    );

    // Stage ownership flags for muxing
    logic conv_active, relu_active, pool_active;
    always_ff @(posedge clk) begin
        if (reset) begin
            conv_active<=0; relu_active<=0; pool_active<=0;
        end else begin
            if (conv_start) conv_active <= 1'b1;
            if (conv_done)  conv_active <= 1'b0;

            if (relu_start) relu_active <= 1'b1;
            if (relu_done)  relu_active <= 1'b0;

            if (pool_start) pool_active <= 1'b1;
            if (pool_done)  pool_active <= 1'b0;
        end
    end

    // IFMAP BRAM Port B hookup (conv2d reads)
    assign if_b_en   = conv_if_en;
    assign if_b_addr = conv_if_addr;

    // CONV buffer Port A (reader: relu OR pool)
    assign convA_addr = (relu_active) ? relu_r_addr      :
                        (pool_active) ? pool_conv_r_addr : {OF_AW{1'b0}};
    assign convA_en   = (relu_active) ? relu_r_en        :
                        (pool_active) ? pool_conv_r_en   : 1'b0;

    // CONV buffer Port B (writer: conv OR relu writeback)
    assign convB_en   = (conv_active) ? conv_w_en   :
                        (relu_active) ? relu_w_en   : 1'b0;
    assign convB_we   = (conv_active) ? conv_w_we   :
                        (relu_active) ? relu_w_we   : 1'b0;
    assign convB_addr = (conv_active) ? conv_w_addr :
                        (relu_active) ? relu_w_addr : {OF_AW{1'b0}};
    assign convB_din  = (conv_active) ? conv_w_d    :
                        (relu_active) ? relu_w_d    : '0;

    // POOL BRAM Port B (dense reader)
    assign poolB_en   = dense_in_en;
    assign poolB_addr = hwc_to_chw(dense_in_addr);

`ifndef SYNTHESIS
    // ---------------- Performance (sim only) ----------------
    logic [63:0] cycle_ctr;
    logic [63:0] t_start, t_conv, t_relu, t_pool, t_flat, t_dense, t_argmax, t_tx;
    logic        tx_start_q;

    always_ff @(posedge clk) begin
        if (reset) cycle_ctr <= 64'd0;
        else       cycle_ctr <= cycle_ctr + 64'd1;
    end
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

            if (dense_done) begin
                integer k;
                integer maxk;
                logic signed [DATA_WIDTH-1:0] maxv;

                $write("Logits:");
                for (k = 0; k < NUM_CLASSES; k = k + 1) $write(" %0d", logits[k]);
                $write("\n");

                maxk = 0;
                maxv = logits[0];
                for (k = 1; k < NUM_CLASSES; k = k + 1)
                    if (logits[k] > maxv) begin maxv = logits[k]; maxk = k; end
                $display("SW argmax = %0d", maxk);
            end
        end
    end
`endif

endmodule
