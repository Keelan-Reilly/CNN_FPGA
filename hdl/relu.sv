//======================================================================
// Module: relu
// Description:
//   Implements the Rectified Linear Unit (ReLU) activation function.
//
//   Functionality:
//   • Reads each activation value sequentially from BRAM.
//   • Applies ReLU: output = max(0, input).
//   • Writes the result back to the same BRAM address (in-place update).
//   • Operates over a full CHW (Channel–Height–Width) feature map.
//
//   Key points:
//   • Uses separate read and write ports of dual-port BRAM (read on A, write on B).
//   • State machine pipelines READ → WRITE for every element.
//   • Linearised addressing scheme (CHW → flat index).
//   • Designed to be lint-clean and synthesis-friendly.
//
//   Use case:
//   • Typically inserted after convolution or dense layers to introduce non-linearity
//     in FPGA-based CNN accelerators.
//======================================================================

(* keep_hierarchy = "yes" *)
module relu #(
    parameter int DATA_WIDTH = 16,
    parameter int CHANNELS   = 8,
    parameter int IMG_SIZE   = 28
)(
    input  logic clk, reset, start,

    // CONV buffer BRAM Port A (read)
    output logic [$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_r_addr,  // Address into input BRAM (CHW-linearised).
    output logic                                           conv_r_en,   // Read enable
    input  logic signed [DATA_WIDTH-1:0]                   conv_r_q,   // Data read

    // CONV buffer BRAM Port B (write back)
    output logic [$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_w_addr, // Address into output BRAM (CHW-linearised).
    output logic                                           conv_w_en,  // Write enable
    output logic                                           conv_w_we, // Write strobe. Check if this is necessary for BRAM
    output logic signed [DATA_WIDTH-1:0]                   conv_w_d,  // Data to write

    output logic done
);
    localparam int H = IMG_SIZE, W = IMG_SIZE;                       // Height/Width of feature map   
    localparam int AW = $clog2(CHANNELS*IMG_SIZE*IMG_SIZE);          // Address width for BRAM.  
    typedef logic [AW-1:0] addr_t;                                   // Type for BRAM addresses
    addr_t addr;                                                     // Linear address pointer through CHW space

    typedef enum logic [1:0] {IDLE, READ, WRITE, FINISH} st_t;
    st_t st;

    integer ch, r, c;                                               // Channel, row, col loop counters.
    logic signed [DATA_WIDTH-1:0] v_reg;                            // Register to hold read value.         

    always_ff @(posedge clk) begin
      if (reset) begin
        st<=IDLE; done<=0; ch<=0; r<=0; c<=0;
        conv_r_en<=0; conv_w_en<=0; conv_w_we<=0;
        addr <= '0;
      end else begin
        done<=0; conv_r_en<=0; conv_w_en<=0; conv_w_we<=0;

        unique case(st)
          IDLE: if (start) begin
                  ch<=0; r<=0; c<=0;      // Reset loop indices.
                  conv_r_addr <= addr;    // Issue read for address 0.
                  conv_r_en   <= 1'b1;    // Assert read enable.
                  st <= READ;
                end

          // Capture input value from BRAM.
          READ: begin
                  v_reg <= conv_r_q;           // Register the value from BRAM.
                  conv_w_addr <= conv_r_addr;  // Copy read address to write address (in-place).
                  st <= WRITE;
                end
          // Apply ReLU, write back, then increment indices.
          WRITE: begin
                  conv_w_en <= 1'b1; conv_w_we <= 1'b1;
                  conv_w_d  <= v_reg[DATA_WIDTH-1] ? '0 : v_reg;   // ReLU: if sign bit=1 (negative), write 0; else write value.

                  if (c==W-1) begin                     // End of row
                    c<=0;
                    if (r==H-1) begin                // End of image
                      r<=0;
                      if (ch==CHANNELS-1) st<=FINISH;  // End of all channels
                      else begin
                        ch<=ch+1;                     // Next channel
                        addr <= addr + 1;             // Move to next address
                        conv_r_addr <= addr + 1;      // Issue read for next address
                        conv_r_en<=1'b1; st<=READ;
                      end
                    end else begin                  // Next row
                      r<=r+1;
                      addr <= addr + 1; 
                      conv_r_addr <= addr + 1;
                      conv_r_en<=1'b1; st<=READ;
                    end
                  end else begin                    // Next column
                    c<=c+1;
                    addr <= addr + 1;
                    conv_r_addr <= addr + 1;
                    conv_r_en<=1'b1; st<=READ;
                  end
                end

          FINISH: begin done<=1; st<=IDLE; end
        endcase
      end
    end
endmodule
