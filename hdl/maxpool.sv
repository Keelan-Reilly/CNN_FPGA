//======================================================================
// maxpool.sv — 2×2 Max-Pooling over BRAM, linear-pointer (fast) version
//   • BRAM has 1-cycle read latency
//   • We walk a single linear pointer conv_base through CHW layout:
//       - next pooled column  : +2
//       - next pooled row     : +(2*IN_SIZE - 2*(OUT_SIZE-1)) = IN_SIZE+2
//       - next channel (wrap) : +(IN_SIZE+2) from last window of last row
//======================================================================
(* keep_hierarchy = "yes" *)
module maxpool #(
    parameter int DATA_WIDTH = 16,
    parameter int CHANNELS   = 8,
    parameter int IN_SIZE    = 28,      // H=W, must be even for POOL=2
    parameter int POOL       = 2
)(
    input  logic clk, reset, start,

    // Read from CONV buffer (Port A)
    output logic [$clog2(CHANNELS*IN_SIZE*IN_SIZE)-1:0] conv_addr,
    output logic                                         conv_en,
    input  logic  signed [DATA_WIDTH-1:0]                conv_q,

    // Write to POOL buffer
    output logic [$clog2(CHANNELS*(IN_SIZE/POOL)*(IN_SIZE/POOL))-1:0] pool_addr,
    output logic                                         pool_en,
    output logic                                         pool_we,
    output logic  signed [DATA_WIDTH-1:0]                pool_d,

    output logic done
);
    localparam int OUT_SIZE = IN_SIZE/POOL;
    localparam int CONV_AW  = $clog2(CHANNELS*IN_SIZE*IN_SIZE);
    localparam int POOL_AW  = $clog2(CHANNELS*OUT_SIZE*OUT_SIZE);

    typedef logic [CONV_AW-1:0] conv_addr_t;
    typedef logic [POOL_AW-1:0] pool_addr_t;

    // Small, sized counters (avoid 32-bit 'integer' on critical path)
    typedef logic[$clog2(CHANNELS) -1:0] ch_t;
    typedef logic[$clog2(OUT_SIZE) -1:0] os_t;

    // Linear pointer through input and output
    conv_addr_t conv_base;
    pool_addr_t pool_base;

    // Indices in pooled space
    ch_t ch;
    os_t r, q; // r=row, q=col

    // Step constants (compile-time)
    localparam int STEP_COL = 2;
    localparam int STEP_ROW = IN_SIZE + 2; // 2*IN_SIZE - 2*(OUT_SIZE-1)
    localparam int STEP_CH  = IN_SIZE + 2; // from last window to next channel start

    // Window registers (account for 1-cycle BRAM read)
    logic signed [DATA_WIDTH-1:0] a0,a1,a2,a3;

    // Max-of-4 with shallow logic depth
    function automatic logic signed [DATA_WIDTH-1:0] max4(
        input logic signed [DATA_WIDTH-1:0] x0, x1, x2, x3
    );
        logic signed [DATA_WIDTH-1:0] m0 = (x0>x1)?x0:x1;
        logic signed [DATA_WIDTH-1:0] m1 = (x2>x3)?x2:x3;
        return (m0>m1)?m0:m1;
    endfunction

    typedef enum logic [2:0] {IDLE, READ0, READ1, READ2, READ3, WRITE, FINISH} state_t;
    state_t state;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; done <= 1'b0;
            ch <= '0; r <= '0; q <= '0;
            conv_en <= 1'b0; conv_addr <= '0;
            pool_en <= 1'b0; pool_we <= 1'b0; pool_addr <= '0; pool_d <= '0;
            conv_base <= '0; pool_base <= '0;
            a0 <= '0; a1 <= '0; a2 <= '0; a3 <= '0;
        end else begin
            done    <= 1'b0;
            conv_en <= 1'b0;
            pool_en <= 1'b0; pool_we <= 1'b0;

            unique case (state)
              // Prime first read
              IDLE: if (start) begin
                  ch <= '0; r <= '0; q <= '0;
                  conv_base <= '0;
                  pool_base <= '0;
                  conv_addr <= '0;     // top-left
                  conv_en   <= 1'b1;   // request a0
                  state     <= READ0;
              end

              // Capture a0; request a1 (top-right)
              READ0: begin
                  a0       <= conv_q;
                  conv_addr <= conv_base + conv_addr_t'(1);
                  conv_en   <= 1'b1;
                  state     <= READ1;
              end
              // Capture a1; request a2 (bottom-left)
              READ1: begin
                  a1       <= conv_q;
                  conv_addr <= conv_base + conv_addr_t'(IN_SIZE);
                  conv_en   <= 1'b1;
                  state     <= READ2;
              end
              // Capture a2; request a3 (bottom-right)
              READ2: begin
                  a2       <= conv_q;
                  conv_addr <= conv_base + conv_addr_t'(IN_SIZE + 1);
                  conv_en   <= 1'b1;
                  state     <= READ3;
              end
              // Capture a3; no new request this cycle
              READ3: begin
                  a3    <= conv_q;
                  state <= WRITE;
              end

              // Write max(a0..a3), then advance linear pointers
              WRITE: begin
                  pool_addr <= pool_base;
                  pool_d    <= max4(a0,a1,a2,a3);
                  pool_en   <= 1'b1; 
                  pool_we   <= 1'b1;
                  pool_base <= pool_base + pool_addr_t'(1);

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

              FINISH: begin
                  done  <= 1'b1;
                  state <= IDLE;
              end
            endcase
        end
    end
endmodule