(* keep_hierarchy = "yes" *)
module maxpool #(
    parameter int DATA_WIDTH = 16,   // Bit-width of activations
    parameter int CHANNELS   = 8,    // Number of feature map channels
    parameter int IN_SIZE    = 28,   // Input feature map height/width
    parameter int POOL       = 2     // Pooling factor (2×2)
)(
    input  logic clk, reset, start, // Clock, synchronous reset, start pulse

    // Convolution feature map BRAM interface (read-only)
    output logic [$clog2(CHANNELS*IN_SIZE*IN_SIZE)-1:0] conv_addr,
    output logic                                         conv_en,
    input  logic  signed [DATA_WIDTH-1:0]                conv_q,

    // Pooled output BRAM interface (write)
    output logic [$clog2(CHANNELS*(IN_SIZE/POOL)*(IN_SIZE/POOL))-1:0] pool_addr,
    output logic                                                     pool_en,
    output logic                                                     pool_we,
    output logic  signed [DATA_WIDTH-1:0]                            pool_d,

    // Asserted for one cycle when entire max-pool pass is complete
    output logic done
);

    // ------------------------------------------------------------
    // Derived sizes
    // ------------------------------------------------------------
    localparam int OUT_SIZE = IN_SIZE / POOL; // Spatial size after pooling

    localparam int CONV_AW  = $clog2(CHANNELS * IN_SIZE * IN_SIZE);
    localparam int POOL_AW  = $clog2(CHANNELS * OUT_SIZE * OUT_SIZE);

    typedef logic [CONV_AW-1:0] conv_addr_t;
    typedef logic [POOL_AW-1:0] pool_addr_t;

    typedef logic [$clog2(CHANNELS)-1:0] ch_t;  // Channel index
    typedef logic [$clog2(OUT_SIZE)-1:0] os_t;  // Output spatial index

    // ------------------------------------------------------------
    // Addressing state
    // ------------------------------------------------------------

    // Base address of current 2×2 pooling window (top-left element)
    conv_addr_t conv_base;

    // Sequential write pointer for pooled output
    pool_addr_t pool_base;

    // Current pooled output coordinates (CHW layout)
    ch_t ch;   // Channel
    os_t r;    // Row
    os_t q;    // Column

    // ------------------------------------------------------------
    // Registers holding the 2×2 window values
    // ------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] a0, a1, a2, a3;

    // ------------------------------------------------------------
    // Max of four signed values (2×2 pooling kernel)
    // ------------------------------------------------------------
    function automatic logic signed [DATA_WIDTH-1:0] max4(
        input logic signed [DATA_WIDTH-1:0] x0,
        input logic signed [DATA_WIDTH-1:0] x1,
        input logic signed [DATA_WIDTH-1:0] x2,
        input logic signed [DATA_WIDTH-1:0] x3
    );
        logic signed [DATA_WIDTH-1:0] m0;
        logic signed [DATA_WIDTH-1:0] m1;
        begin
            m0   = (x0 > x1) ? x0 : x1;
            m1   = (x2 > x3) ? x2 : x3;
            max4 = (m0 > m1) ? m0 : m1;
        end
    endfunction

    // ------------------------------------------------------------
    // BRAM latency handling
    // ------------------------------------------------------------
    // Allows this block to tolerate up to MAX_BRAM_LAT cycles
    // between address issue and valid conv_q.
    localparam int MAX_BRAM_LAT = 2;
    localparam int WAITW = (MAX_BRAM_LAT <= 1) ? 1 : $clog2(MAX_BRAM_LAT+1);
    logic [WAITW-1:0] wait_cnt;

    // Phase selects which of the 2×2 window entries is being fetched:
    // 0 → a0, 1 → a1, 2 → a2, 3 → a3
    logic [1:0] phase;

    // ------------------------------------------------------------
    // FSM definition
    // ------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE,    // Waiting for start
        ISSUE,   // Present conv_addr and assert conv_en
        WAIT,    // Stall for BRAM latency
        CAP,     // Capture conv_q into a0–a3
        WRITE,   // Write pooled result and advance indices
        FINISH   // Signal completion
    } state_t;

    state_t state;

    // conv_en is asserted during address ISSUE cycle
    assign conv_en = (state == ISSUE);

    // ------------------------------------------------------------
    // Compute address offset within a 2×2 window
    // ------------------------------------------------------------
    function automatic conv_addr_t phase_addr(
        input conv_addr_t base,
        input logic [1:0] ph
    );
        begin
            case (ph)
                2'd0: phase_addr = base;                         // (0,0)
                2'd1: phase_addr = base + conv_addr_t'(1);       // (0,1)
                2'd2: phase_addr = base + conv_addr_t'(IN_SIZE); // (1,0)
                default:
                      phase_addr = base + conv_addr_t'(IN_SIZE + 1); // (1,1)
            endcase
        end
    endfunction

    // ------------------------------------------------------------
    // Compute top-left address of a pooled window
    // Layout: CHW (channel-major, row-major within channel)
    // ------------------------------------------------------------
    function automatic conv_addr_t base_addr(
        input ch_t ch_i,
        input os_t r_i,
        input os_t q_i
    );
        int tmp;
        begin
            tmp = int'(ch_i) * (IN_SIZE * IN_SIZE)
                + (int'(POOL) * int'(r_i)) * IN_SIZE
                + (int'(POOL) * int'(q_i));
            base_addr = conv_addr_t'(tmp);
        end
    endfunction

    // ------------------------------------------------------------
    // Main sequential control
    // ------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            // Full synchronous reset
            state <= IDLE;
            done  <= 1'b0;

            ch <= '0;
            r  <= '0;
            q  <= '0;

            conv_addr <= '0;
            conv_base <= '0;

            pool_en   <= 1'b0;
            pool_we   <= 1'b0;
            pool_addr <= '0;
            pool_d    <= '0;
            pool_base <= '0;

            a0 <= '0; a1 <= '0; a2 <= '0; a3 <= '0;

            phase    <= 2'd0;
            wait_cnt <= '0;

        end else begin
            // Default de-assertions
            done    <= 1'b0;
            pool_en <= 1'b0;
            pool_we <= 1'b0;

            unique case (state)

              // --------------------------------------------------
              // IDLE: wait for start pulse
              // --------------------------------------------------
              IDLE: begin
                  if (start) begin
                      ch        <= '0;
                      r         <= '0;
                      q         <= '0;
                      pool_base <= '0;

                      phase     <= 2'd0;

                      conv_base <= base_addr('0, '0, '0);
                      conv_addr <= phase_addr(base_addr('0, '0, '0), 2'd0);

                      state <= ISSUE;
                  end
              end

              // --------------------------------------------------
              // ISSUE: drive address, assert conv_en
              // --------------------------------------------------
              ISSUE: begin
                  wait_cnt <= MAX_BRAM_LAT[WAITW-1:0];
                  state    <= WAIT;
              end

              // --------------------------------------------------
              // WAIT: absorb BRAM read latency
              // --------------------------------------------------
              WAIT: begin
                  if (wait_cnt == 0)
                      state <= CAP;
                  else
                      wait_cnt <= wait_cnt - 1'b1;
              end

              // --------------------------------------------------
              // CAP: latch current conv_q into a0–a3
              // --------------------------------------------------
              CAP: begin
                  case (phase)
                      2'd0: a0 <= conv_q;
                      2'd1: a1 <= conv_q;
                      2'd2: a2 <= conv_q;
                      default: a3 <= conv_q;
                  endcase

                  if (phase == 2'd3) begin
                      state <= WRITE;
                  end else begin
                      phase     <= phase + 2'd1;
                      conv_addr <= phase_addr(conv_base, phase + 2'd1);
                      state     <= ISSUE;
                  end
              end

              // --------------------------------------------------
              // WRITE: emit pooled value, advance indices
              // --------------------------------------------------
              WRITE: begin : WRITE_BLK
                  ch_t  ch_n;
                  os_t  r_n;
                  os_t  q_n;
                  logic last_pixel;

                  ch_n = ch;
                  r_n  = r;
                  q_n  = q;

                  // Write pooled max
                  pool_addr <= pool_base;
                  pool_d    <= max4(a0, a1, a2, a3);
                  pool_en   <= 1'b1;
                  pool_we   <= 1'b1;
                  pool_base <= pool_base + pool_addr_t'(1);

                  // Check for final output element
                  last_pixel =
                      (ch == CHANNELS-1) &&
                      (r  == OUT_SIZE-1) &&
                      (q  == OUT_SIZE-1);

                  if (last_pixel) begin
                      state <= FINISH;
                  end else begin
                      // q fastest, then r, then ch
                      if (q == OUT_SIZE-1) begin
                          q_n = '0;
                          if (r == OUT_SIZE-1) begin
                              r_n  = '0;
                              ch_n = ch + ch_t'(1);
                          end else begin
                              r_n = r + os_t'(1);
                          end
                      end else begin
                          q_n = q + os_t'(1);
                      end

                      ch <= ch_n;
                      r  <= r_n;
                      q  <= q_n;

                      conv_base <= base_addr(ch_n, r_n, q_n);
                      phase     <= 2'd0;
                      conv_addr <= phase_addr(base_addr(ch_n, r_n, q_n), 2'd0);
                      state     <= ISSUE;
                  end
              end

              // --------------------------------------------------
              // FINISH: pulse done and return to IDLE
              // --------------------------------------------------
              FINISH: begin
                  done  <= 1'b1;
                  state <= IDLE;
              end

            endcase
        end
    end

endmodule
