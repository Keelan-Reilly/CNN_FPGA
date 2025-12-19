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

    // Base pointer for current 2×2 window (top-left)
    conv_addr_t conv_base;

    // Destination pointer for pooled output
    pool_addr_t pool_base;

    // Pooled indices
    ch_t ch;
    os_t r, q;

    // Window regs
    logic signed [DATA_WIDTH-1:0] a0, a1, a2, a3;

    // ---------------- max4 ----------------
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

    // ---------------- BRAM latency handling ----------------
    localparam int MAX_BRAM_LAT = 2;
    localparam int WAITW = (MAX_BRAM_LAT <= 1) ? 1 : $clog2(MAX_BRAM_LAT+1);
    logic [WAITW-1:0] wait_cnt;

    logic [1:0] phase; // 0..3 => a0..a3

    typedef enum logic [2:0] {IDLE, ISSUE, WAIT, CAP, WRITE, FINISH} state_t;
    state_t state;

    // conv_en asserted for the whole ISSUE cycle
    assign conv_en = (state == ISSUE);

    // Address within the 2×2 window
    function automatic conv_addr_t phase_addr(input conv_addr_t base, input logic [1:0] ph);
        begin
            case (ph)
                2'd0: phase_addr = base;
                2'd1: phase_addr = base + conv_addr_t'(1);
                2'd2: phase_addr = base + conv_addr_t'(IN_SIZE);
                default: phase_addr = base + conv_addr_t'(IN_SIZE + 1);
            endcase
        end
    endfunction

    // Compute top-left base for pooled coordinate (CHW layout)
    function automatic conv_addr_t base_addr(input ch_t ch_i, input os_t r_i, input os_t q_i);
        int tmp;
        begin
            tmp = int'(ch_i) * (IN_SIZE*IN_SIZE)
                + (int'(POOL) * int'(r_i)) * IN_SIZE
                + (int'(POOL) * int'(q_i));
            base_addr = conv_addr_t'(tmp);
        end
    endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
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
            done    <= 1'b0;
            pool_en <= 1'b0;
            pool_we <= 1'b0;

            unique case (state)

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

              WRITE: begin : WRITE_BLK
                  ch_t  ch_n;
                  os_t  r_n;
                  os_t  q_n;
                  logic last_pixel;

                  // init locals (NO declaration initialisers)
                  ch_n = ch;
                  r_n  = r;
                  q_n  = q;

                  // write pooled value
                  pool_addr <= pool_base;
                  pool_d    <= max4(a0, a1, a2, a3);
                  pool_en   <= 1'b1;
                  pool_we   <= 1'b1;
                  pool_base <= pool_base + pool_addr_t'(1);

                  // termination check (after this write)
                  last_pixel = (ch == CHANNELS-1) && (r == OUT_SIZE-1) && (q == OUT_SIZE-1);

                  if (last_pixel) begin
                      state <= FINISH;
                  end else begin
                      // advance pooled indices: q fastest, then r, then ch
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

              FINISH: begin
                  done  <= 1'b1;
                  state <= IDLE;
              end

            endcase
        end
    end

endmodule
