//======================================================================
// maxpool.sv — 2×2 Max-Pooling over BRAM, lint-clean
//======================================================================
(* keep_hierarchy = "yes" *)
module maxpool #(
    parameter int DATA_WIDTH = 16,
    parameter int CHANNELS   = 8,
    parameter int IN_SIZE    = 28,
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
    localparam int CONV_AW = $clog2(CHANNELS*IN_SIZE*IN_SIZE);
    localparam int OUT_SIZE = IN_SIZE/POOL;
    localparam int POOL_AW = $clog2(CHANNELS*OUT_SIZE*OUT_SIZE);

    typedef logic [CONV_AW-1:0] conv_addr_t;
    typedef logic [POOL_AW-1:0] pool_addr_t;

    typedef enum logic [2:0] {IDLE, READ0, READ1, READ2, READ3, WRITE, FINISH} state_t;
    state_t state;
    integer ch, r, q;

    logic signed [DATA_WIDTH-1:0] a0,a1,a2,a3;

    conv_addr_t conv_base;           // top-left of current 2x2 window in CONV buffer
    pool_addr_t pool_base;           // linear index in POOL buffer (CHW order)

    function automatic int lin3(
        input int ch_i, input int row_i, input int col_i,
        input int H_i,  input int W_i
    ); return (ch_i*H_i + row_i)*W_i + col_i; endfunction

    function automatic logic signed [DATA_WIDTH-1:0] max2(
        input logic signed [DATA_WIDTH-1:0] x, y
    ); return (x>y)?x:y; endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
            state<=IDLE; done<=0; ch<=0; r<=0; q<=0;
            conv_en<=0; pool_en<=0; pool_we<=0;
            a0<='0; a1<='0; a2<='0; a3<='0;
            conv_base <= '0; 
            pool_base <= '0; 
            conv_addr <= '0;
        end else begin
            done<=0; conv_en<=0; pool_en<=0; pool_we<=0;

            unique case(state)
              IDLE: if (start) begin
                      ch<=0; r<=0; q<=0;
                      conv_base <= 0;
                      conv_addr <= conv_base;
                      conv_en   <= 1'b1;
                      state <= READ0;
                    end

              READ0: begin
                      a0 <= conv_q;
                      conv_addr <= conv_base + 1;
                      conv_en   <= 1'b1;
                      state <= READ1;
                    end

              READ1: begin
                      a1 <= conv_q;
                      conv_addr <= conv_base + IN_SIZE;
                      conv_en   <= 1'b1;
                      state <= READ2;
                    end

              READ2: begin
                      a2 <= conv_q;
                      conv_addr <= conv_base + IN_SIZE + 1;
                      conv_en   <= 1'b1;
                      state <= READ3;
                    end

              READ3: begin
                      a3 <= conv_q;
                      state <= WRITE;
                    end

              WRITE: begin
                      pool_addr <= pool_base;
                      pool_d    <= (a0>a1 ? a0:a1) > (a2>a3 ? a2:a3)
                                   ? (a0>a1 ? a0:a1) : (a2>a3 ? a2:a3); // max of 4                                   
                      pool_en   <= 1'b1; pool_we <= 1'b1;
                      pool_base <= pool_base + 1; // next POOL address

                      if (q==OUT_SIZE-1) begin
                        q<=0;
                        if (r==OUT_SIZE-1) begin
                          r<=0;
                          if (ch==CHANNELS-1) begin
                            state<=FINISH;
                          end else begin
                            ch<=ch+1;
                            conv_base <= conv_base + (IN_SIZE + 2); // next CH conv base
                            conv_addr <= conv_base + (IN_SIZE + 2); // next CH conv addr
                            conv_en   <= 1'b1;
                            state<=READ0;
                          end
                        end else begin
                          r<=r+1;
                          conv_base <= conv_base + (IN_SIZE + 2);
                          conv_addr <= conv_base + (IN_SIZE + 2);
                          conv_en   <= 1'b1;
                          state<=READ0;
                        end
                      end else begin
                        q<=q+1;
                        conv_base <= conv_base + 2;
                        conv_addr <= conv_base + 2;
                        conv_en   <= 1'b1;
                        state<=READ0;
                      end
                    end

              FINISH: begin done<=1; state<=IDLE; end
            endcase
        end
    end
endmodule
