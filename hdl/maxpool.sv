//======================================================================
// Module: maxpool
//
// Overview
//   2×2 max-pooling layer for CNN feature maps.
//
//   For each channel `ch` and pooled output coordinate (r, c):
//
//     out[ch, r, c] = max( in[ch, 2r,   2c  ],
//                          in[ch, 2r,   2c+1],
//                          in[ch, 2r+1, 2c  ],
//                          in[ch, 2r+1, 2c+1] )
//
//   The input feature map is stored in BRAM in CHW-linear order. The
//   output is written back in the same CHW-linear order at half the
//   spatial resolution (POOL=2).
//
// Timing / BRAM latency
//   • The input BRAM has 1-cycle read latency.
//   • The FSM issues four reads per output pixel and captures the four
//     returned values into window registers (a0..a3), then writes the
//     maximum to the output BRAM.
//
// Addressing strategy
//   • Uses a linear base pointer `conv_base` that always points to the
//     top-left element of the current 2×2 window.
//   • Advances `conv_base` with fixed step sizes to avoid recomputing
//     row/col products in the datapath.
//
//     STEP_COL: move to next pooled column (right by 2 input pixels)
//     STEP_ROW: move to next pooled row start (down by 2 rows, back to col 0)
//     STEP_CH : move to the next channel’s first window
//
// FSM flow (per pooled output element)
//   IDLE      : initialise indices and issue first read
//   READ0     : capture a0, request a1
//   READ1     : capture a1, request a2
//   READ2     : capture a2, request a3
//   READ3_WAIT: one-cycle spacer for BRAM latency
//   READ3_CAP : capture a3
//   WRITE     : write max(a0..a3), advance pointers/indices
//   FINISH    : pulse done and return to IDLE
//======================================================================

(* keep_hierarchy = "yes" *)
module maxpool #(
    parameter int DATA_WIDTH = 16,      // Activation bit width (signed)
    parameter int CHANNELS   = 8,       // Number of channels
    parameter int IN_SIZE    = 28,      // Input H=W (must be even for POOL=2)
    parameter int POOL       = 2        // Pooling factor (assumed 2 here)
)(
    input  logic clk, reset, start,

    // Read from CONV buffer (1-cycle latency)
    output logic [$clog2(CHANNELS*IN_SIZE*IN_SIZE)-1:0] conv_addr,
    output logic                                         conv_en,
    input  logic  signed [DATA_WIDTH-1:0]                conv_q,

    // Write to POOL buffer
    output logic [$clog2(CHANNELS*(IN_SIZE/POOL)*(IN_SIZE/POOL))-1:0] pool_addr,
    output logic                                                     pool_en,
    output logic                                                     pool_we,
    output logic  signed [DATA_WIDTH-1:0]                            pool_d,

    output logic done
);

    localparam int OUT_SIZE = IN_SIZE/POOL;

    localparam int CONV_AW  = $clog2(CHANNELS*IN_SIZE*IN_SIZE);
    localparam int POOL_AW  = $clog2(CHANNELS*OUT_SIZE*OUT_SIZE);

    typedef logic [CONV_AW-1:0] conv_addr_t;
    typedef logic [POOL_AW-1:0] pool_addr_t;

    // Narrow counters (avoid 32-bit integer in control/datapath)
    typedef logic [$clog2(CHANNELS)-1:0] ch_t;
    typedef logic [$clog2(OUT_SIZE)-1:0] os_t;

    // Linear base pointers
    conv_addr_t conv_base;   // top-left of current 2×2 input window
    pool_addr_t pool_base;   // destination address for current pooled pixel

    // Indices in pooled space
    ch_t ch;
    os_t r, q;               // pooled row/col

    // Compile-time step sizes (for POOL=2)
    localparam int STEP_COL = 2;            // advance to next pooled column
    localparam int STEP_ROW = IN_SIZE + 2;  // advance to next pooled row start
    localparam int STEP_CH  = IN_SIZE + 2;  // advance to next channel start

    // Window registers (captured from BRAM output)
    logic signed [DATA_WIDTH-1:0] a0, a1, a2, a3;

    // Max-of-4 with shallow compare depth
    function automatic logic signed [DATA_WIDTH-1:0] max4(
        input logic signed [DATA_WIDTH-1:0] x0, x1, x2, x3
    );
        logic signed [DATA_WIDTH-1:0] m0 = (x0 > x1) ? x0 : x1;
        logic signed [DATA_WIDTH-1:0] m1 = (x2 > x3) ? x2 : x3;
        return (m0 > m1) ? m0 : m1;
    endfunction

    typedef enum logic [2:0] {IDLE, READ0, READ1, READ2, READ3_WAIT, READ3_CAP, WRITE, FINISH} state_t;
    state_t state;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            done  <= 1'b0;

            ch <= '0; r <= '0; q <= '0;

            conv_en   <= 1'b0;
            conv_addr <= '0;

            pool_en   <= 1'b0;
            pool_we   <= 1'b0;
            pool_addr <= '0;
            pool_d    <= '0;

            conv_base <= '0;
            pool_base <= '0;

            a0 <= '0; a1 <= '0; a2 <= '0; a3 <= '0;

        end else begin
            done    <= 1'b0;
            conv_en <= 1'b0;
            pool_en <= 1'b0;
            pool_we <= 1'b0;

            unique case (state)

              // ------------------------------ IDLE ---------------------
              IDLE: if (start) begin
                  ch        <= '0;
                  r         <= '0;
                  q         <= '0;

                  conv_base <= '0;
                  pool_base <= '0;

                  // Issue first read (a0)
                  conv_addr <= '0;
                  conv_en   <= 1'b1;

                  state     <= READ0;
              end

              // ------------------------------ READ0 --------------------
              // Capture a0; request a1 (top-right)
              READ0: begin
                  a0        <= conv_q;
                  conv_addr <= conv_base + conv_addr_t'(1);
                  conv_en   <= 1'b1;
                  state     <= READ1;
              end

              // ------------------------------ READ1 --------------------
              // Capture a1; request a2 (bottom-left)
              READ1: begin
                  a1        <= conv_q;
                  conv_addr <= conv_base + conv_addr_t'(IN_SIZE);
                  conv_en   <= 1'b1;
                  state     <= READ2;
              end

              // ------------------------------ READ2 --------------------
              // Capture a2; request a3 (bottom-right)
              READ2: begin
                  a2        <= conv_q;
                  conv_addr <= conv_base + conv_addr_t'(IN_SIZE + 1);
                  conv_en   <= 1'b1;
                  state     <= READ3_WAIT;
              end

              // --------------------------- READ3_WAIT ------------------
              // One-cycle spacer for BRAM to present a3 at conv_q
              READ3_WAIT: begin
                  state <= READ3_CAP;
              end

              // --------------------------- READ3_CAP -------------------
              // Capture a3; proceed to WRITE
              READ3_CAP: begin
                  a3    <= conv_q;
                  state <= WRITE;
              end

              // ------------------------------ WRITE --------------------
              // Write max(a0..a3) and advance indices/pointers
              WRITE: begin
                  pool_addr <= pool_base;
                  pool_d    <= max4(a0, a1, a2, a3);
                  pool_en   <= 1'b1;
                  pool_we   <= 1'b1;

                  pool_base <= pool_base + pool_addr_t'(1);

                  // Advance pooled coordinates and update conv_base accordingly
                  if (q == OUT_SIZE-1) begin
                      q <= '0;

                      if (r == OUT_SIZE-1) begin
                          r <= '0;

                          if (ch == CHANNELS-1) begin
                              state <= FINISH;
                          end else begin
                              ch        <= ch + ch_t'(1);
                              conv_base <= conv_base + conv_addr_t'(STEP_CH);
                              conv_addr <= conv_base + conv_addr_t'(STEP_CH);
                              conv_en   <= 1'b1;
                              state     <= READ0;
                          end

                      end else begin
                          r        <= r + os_t'(1);
                          conv_base <= conv_base + conv_addr_t'(STEP_ROW);
                          conv_addr <= conv_base + conv_addr_t'(STEP_ROW);
                          conv_en   <= 1'b1;
                          state     <= READ0;
                      end

                  end else begin
                      q        <= q + os_t'(1);
                      conv_base <= conv_base + conv_addr_t'(STEP_COL);
                      conv_addr <= conv_base + conv_addr_t'(STEP_COL);
                      conv_en   <= 1'b1;
                      state     <= READ0;
                  end
              end

              // ----------------------------- FINISH --------------------
              FINISH: begin
                  done  <= 1'b1;
                  state <= IDLE;
              end

            endcase
        end
    end
endmodule
