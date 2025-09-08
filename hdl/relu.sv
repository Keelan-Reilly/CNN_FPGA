//======================================================================
// relu.sv — ReLU Activation over BRAM (in-place), lint-clean
//======================================================================
(* keep_hierarchy = "yes" *)
module relu #(
    parameter int DATA_WIDTH = 16,
    parameter int CHANNELS   = 8,
    parameter int IMG_SIZE   = 28
)(
    input  logic clk, reset, start,

    // CONV buffer BRAM Port A (read)
    output logic [$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_r_addr,
    output logic                                           conv_r_en,
    input  logic signed [DATA_WIDTH-1:0]                   conv_r_q,

    // CONV buffer BRAM Port B (write back)
    output logic [$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_w_addr,
    output logic                                           conv_w_en,
    output logic                                           conv_w_we,
    output logic signed [DATA_WIDTH-1:0]                   conv_w_d,

    output logic done
);
    localparam int H = IMG_SIZE, W = IMG_SIZE;

    typedef enum logic [1:0] {IDLE, READ, WRITE, FINISH} st_t;
    st_t st;

    integer ch, r, c;
    logic signed [DATA_WIDTH-1:0] v_reg;

    // Avoid VARHIDDEN by using _i names
    function automatic int lin3(input int ch_i,input int row_i,input int col_i);
        return (ch_i*H + row_i)*W + col_i;
    endfunction

    always_ff @(posedge clk) begin
      if (reset) begin
        st<=IDLE; done<=0; ch<=0; r<=0; c<=0;
        conv_r_en<=0; conv_w_en<=0; conv_w_we<=0;
      end else begin
        done<=0; conv_r_en<=0; conv_w_en<=0; conv_w_we<=0;

        unique case(st)
          IDLE: if (start) begin
                  ch<=0; r<=0; c<=0;
                  conv_r_addr <= lin3(0,0,0)[$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0];
                  conv_r_en   <= 1'b1;
                  st <= READ;
                end

          READ: begin
                  v_reg <= conv_r_q;
                  conv_w_addr <= conv_r_addr;
                  st <= WRITE;
                end

          WRITE: begin
                  conv_w_en <= 1'b1; conv_w_we <= 1'b1;
                  conv_w_d  <= v_reg[DATA_WIDTH-1] ? '0 : v_reg;

                  if (c==W-1) begin
                    c<=0;
                    if (r==H-1) begin
                      r<=0;
                      if (ch==CHANNELS-1) st<=FINISH;
                      else begin
                        ch<=ch+1;
                        conv_r_addr <= lin3(ch+1,0,0)[$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0];
                        conv_r_en<=1'b1; st<=READ;
                      end
                    end else begin
                      r<=r+1;
                      conv_r_addr <= lin3(ch,r+1,0)[$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0];
                      conv_r_en<=1'b1; st<=READ;
                    end
                  end else begin
                    c<=c+1;
                    conv_r_addr <= lin3(ch,r,c+1)[$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0];
                    conv_r_en<=1'b1; st<=READ;
                  end
                end

          FINISH: begin done<=1; st<=IDLE; end
        endcase
      end
    end
endmodule
