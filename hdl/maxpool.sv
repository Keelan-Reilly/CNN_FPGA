(* keep_hierarchy = "yes" *)
module maxpool #(
    parameter int DATA_WIDTH = 16,
    parameter int CHANNELS   = 8,
    parameter int IN_SIZE    = 28,
    parameter int POOL       = 2
)(
    input  logic clk, reset, start,

    output logic [$clog2(CHANNELS*IN_SIZE*IN_SIZE)-1:0] conv_addr,
    output logic                                         conv_en,
    input  logic  signed [DATA_WIDTH-1:0]                conv_q,

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

    typedef logic [$clog2(CHANNELS)-1:0] ch_t;
    typedef logic [$clog2(OUT_SIZE)-1:0] os_t;

    // Linear base pointers
    conv_addr_t conv_base;   // top-left of current 2×2 window
    pool_addr_t pool_base;   // destination address for current pooled pixel

    // Indices in pooled space
    ch_t ch;
    os_t r, q;               // pooled row/col

    // Step sizes (POOL=2)
    localparam int STEP_COL = 2;
    localparam int STEP_ROW = IN_SIZE + 2;
    localparam int STEP_CH  = IN_SIZE + 2;

    // Window registers
    logic signed [DATA_WIDTH-1:0] a0, a1, a2, a3;

    function automatic logic signed [DATA_WIDTH-1:0] max4(
        input logic signed [DATA_WIDTH-1:0] x0, x1, x2, x3
    );
        logic signed [DATA_WIDTH-1:0] m0 = (x0 > x1) ? x0 : x1;
        logic signed [DATA_WIDTH-1:0] m1 = (x2 > x3) ? x2 : x3;
        return (m0 > m1) ? m0 : m1;
    endfunction

    // ============================================================
    // Fix: explicit ISSUE/WAIT/CAP so we don't capture stale conv_q
    // TB varies BRAM latency (0/1/2) and holds q when idle.
    // We therefore wait a conservative worst-case of 2 cycles.
    // ============================================================
    localparam int MAX_BRAM_LAT = 2;
    localparam int WAITW = (MAX_BRAM_LAT <= 1) ? 1 : $clog2(MAX_BRAM_LAT+1);
    logic [WAITW-1:0] wait_cnt;

    logic [1:0] phase; // 0..3 corresponds to a0..a3 and address offsets

    typedef enum logic [2:0] {IDLE, ISSUE, WAIT, CAP, WRITE, FINISH} state_t;
    state_t state;

    // conv_en is combinational: asserted for the whole ISSUE cycle
    assign conv_en = (state == ISSUE);

    // Address offset for current phase
    function automatic conv_addr_t phase_addr(input conv_addr_t base, input logic [1:0] ph);
        case (ph)
            2'd0: return base;                                 // a0
            2'd1: return base + conv_addr_t'(1);               // a1
            2'd2: return base + conv_addr_t'(IN_SIZE);         // a2
            default: return base + conv_addr_t'(IN_SIZE + 1);  // a3
        endcase
    endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            done  <= 1'b0;

            ch <= '0; r <= '0; q <= '0;

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
            done    <= 1'b0;
            pool_en <= 1'b0;
            pool_we <= 1'b0;

            unique case (state)

              IDLE: begin
                  if (start) begin
                      ch        <= '0;
                      r         <= '0;
                      q         <= '0;
                      conv_base <= '0;
                      pool_base <= '0;

                      phase     <= 2'd0;

                      // Prepare first request address; conv_en will be high in ISSUE
                      conv_addr <= phase_addr(conv_addr_t'(0), 2'd0);
                      state     <= ISSUE;
                  end
              end

              // ISSUE holds conv_en high for the full cycle; TB BRAM samples at *this* posedge
              // (i.e., one cycle after we entered ISSUE) because state is registered.
              // We then wait conservatively MAX_BRAM_LAT cycles before capturing.
              ISSUE: begin
                  wait_cnt <= MAX_BRAM_LAT[WAITW-1:0];
                  state    <= WAIT;
              end

              WAIT: begin
                  if (wait_cnt == 0) begin
                      state <= CAP;
                  end else begin
                      wait_cnt <= wait_cnt - 1'b1;
                  end
              end

              CAP: begin
                  // Capture the value corresponding to the most recently issued address.
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

              WRITE: begin
                  conv_addr_t cb_next;
                  os_t        q_next;
                  os_t        r_next;
                  ch_t        ch_next;

                  // Write pooled result
                  pool_addr <= pool_base;
                  pool_d    <= max4(a0, a1, a2, a3);
                  pool_en   <= 1'b1;
                  pool_we   <= 1'b1;
                  pool_base <= pool_base + pool_addr_t'(1);

                  // Advance pooled coordinates and conv_base
                  if (q == OUT_SIZE-1) begin
                      q_next = '0;

                      if (r == OUT_SIZE-1) begin
                          r_next = '0;

                          if (ch == CHANNELS-1) begin
                              state <= FINISH;
                              // no next read
                          end else begin
                              ch_next = ch + ch_t'(1);
                              ch <= ch_next;
                              r  <= r_next;
                              q  <= q_next;

                              cb_next  = conv_base + conv_addr_t'(STEP_CH);
                              conv_base <= cb_next;

                              phase     <= 2'd0;
                              conv_addr <= phase_addr(cb_next, 2'd0);
                              state     <= ISSUE;
                          end

                      end else begin
                          r_next = r + os_t'(1);
                          r <= r_next;
                          q <= q_next;

                          cb_next  = conv_base + conv_addr_t'(STEP_ROW);
                          conv_base <= cb_next;

                          phase     <= 2'd0;
                          conv_addr <= phase_addr(cb_next, 2'd0);
                          state     <= ISSUE;
                      end

                  end else begin
                      q_next = q + os_t'(1);
                      q <= q_next;

                      cb_next  = conv_base + conv_addr_t'(STEP_COL);
                      conv_base <= cb_next;

                      phase     <= 2'd0;
                      conv_addr <= phase_addr(cb_next, 2'd0);
                      state     <= ISSUE;
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
