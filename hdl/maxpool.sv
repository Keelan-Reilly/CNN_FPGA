//======================================================================
// Module: maxpool
// Description:
//   Implements a 2×2 max-pooling layer for CNNs.
//
//   Functionality:
//   • Reads feature map data sequentially from BRAM in CHW order.
//   • For each pooled output position, fetches a 2×2 block of input values
//     and computes their maximum.
//   • Handles BRAM’s 1-cycle read latency by pipelining four read states.
//   • Writes the pooled result into an output BRAM in CHW order.
//
//   Key points:
//   • Reduces spatial resolution by a factor of 2 in both height and width:
//       out[ch, r, c] = max( in[ch, 2r..2r+1, 2c..2c+1] )
//   • Linear pointer arithmetic avoids costly row/col index calculations.
//   • Step sizes (STEP_COL, STEP_ROW, STEP_CH) are precomputed to advance
//     through columns, rows, and channels efficiently.
//   • State machine reads 4 inputs → computes max → writes 1 output.
//======================================================================
(* keep_hierarchy = "yes" *)
module maxpool #(
    parameter int DATA_WIDTH = 16,      // Bit width of activations (signed).
    parameter int CHANNELS   = 8,       // Number of feature map channels.
    parameter int IN_SIZE    = 28,      // H=W, must be even for POOL=2
    parameter int POOL       = 2        // Pooling factor 
)(
    input  logic clk, reset, start,

    // Read from CONV buffer (Port A)
    output logic [$clog2(CHANNELS*IN_SIZE*IN_SIZE)-1:0] conv_addr,  // Address into input BRAM (CHW-linearised).
    output logic                                         conv_en,   // Read enable for input BRAM.    
    input  logic  signed [DATA_WIDTH-1:0]                conv_q,    // Data read from input BRAM (1-cycle latency).

    // Write to POOL buffer
    output logic [$clog2(CHANNELS*(IN_SIZE/POOL)*(IN_SIZE/POOL))-1:0] pool_addr,  // Address into output BRAM (CHW-linearised).
    output logic                                         pool_en,                 // Write enable for output BRAM.
    output logic                                         pool_we,                 // Write enable for output BRAM.
    output logic  signed [DATA_WIDTH-1:0]                pool_d,                  // Data to write to output BRAM.

    output logic done
);
    localparam int OUT_SIZE = IN_SIZE/POOL;                         // Output height/width after pooling
    localparam int CONV_AW  = $clog2(CHANNELS*IN_SIZE*IN_SIZE);     // Address width for input BRAM.
    localparam int POOL_AW  = $clog2(CHANNELS*OUT_SIZE*OUT_SIZE);   // Address width for output BRAM.

    typedef logic [CONV_AW-1:0] conv_addr_t;            // Type for input BRAM addresses
    typedef logic [POOL_AW-1:0] pool_addr_t;           // Type for output BRAM addresses

    // Small, sized counters (avoid 32-bit 'integer' on critical path)
    typedef logic[$clog2(CHANNELS) -1:0] ch_t;   // Narrow channel index type.
    typedef logic[$clog2(OUT_SIZE) -1:0] os_t;   // Narrow row/col index type in pooled space.

    // Linear pointer through input and output
    conv_addr_t conv_base;                  // Linear base address for the current 2×2 window (top-left).
    pool_addr_t pool_base;                  // Linear base address for the current pooled pixel.     

    // Indices in pooled space
    ch_t ch;                                 // Current channel (0..CHANNELS-1)
    os_t r, q;                               // r=row, q=col. Current pooled row/col (0..OUT_SIZE-1).

    // Step constants (compile-time)
    localparam int STEP_COL = 2;                // Move right by 2 input pixels (next pooled column).
    localparam int STEP_ROW = IN_SIZE + 2;      // Move down to next pooled row start from current window. (2*IN_SIZE - 2*(OUT_SIZE-1)) simplifies to IN_SIZE + 2 here.
    localparam int STEP_CH  = IN_SIZE + 2;      // From last pooled row/col to next channel’s first window.

    // Window registers (account for 1-cycle BRAM read)
    logic signed [DATA_WIDTH-1:0] a0,a1,a2,a3;   // Registers for 2×2 window: top-left, top-right, bottom-left, bottom-right.

    // Max-of-4 with shallow logic depth
    function automatic logic signed [DATA_WIDTH-1:0] max4(
        input logic signed [DATA_WIDTH-1:0] x0, x1, x2, x3   // Four signed inputs to compare.
    );
        logic signed [DATA_WIDTH-1:0] m0 = (x0>x1)?x0:x1;    // Max of top pair.
        logic signed [DATA_WIDTH-1:0] m1 = (x2>x3)?x2:x3;    // Max of bottom pair.
        return (m0>m1)?m0:m1;                                // Max of the two maxima.   
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
                  ch <= '0; r <= '0; q <= '0;    // Start at channel 0, pooled row 0, pooled col 0.
                  conv_base <= '0;               // Start at input address 0 (top-left of channel 0).
                  pool_base <= '0;               // Start at output address 0.
                  conv_addr <= '0;               // Issue address for top-left (a0).
                  conv_en   <= 1'b1;             // Enable read.
                  state     <= READ0;
              end

              // Capture a0; request a1 (top-right)
              READ0: begin
                  a0       <= conv_q;                          // Latch a0 from BRAM
                  conv_addr <= conv_base + conv_addr_t'(1);    // Address top-right (a1): base + 1.
                  conv_en   <= 1'b1;                           
                  state     <= READ1;
              end
              // Capture a1; request a2 (bottom-left)
              READ1: begin
                  a1       <= conv_q;                              // Latch a1 from BRAM
                  conv_addr <= conv_base + conv_addr_t'(IN_SIZE);  // Address bottom-left (a2): base + row stride.
                  conv_en   <= 1'b1;
                  state     <= READ2;
              end
              // Capture a2; request a3 (bottom-right)
              READ2: begin
                  a2       <= conv_q;                                    // Latch a2 from BRAM
                  conv_addr <= conv_base + conv_addr_t'(IN_SIZE + 1);   // Address bottom-right (a3): base + row stride + 1.
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
                  pool_addr <= pool_base;                           // Set destination address for this pooled pixel.
                  pool_d    <= max4(a0,a1,a2,a3);                   // Compute max of 2×2 window.
                  pool_en   <= 1'b1; 
                  pool_we   <= 1'b1;
                  pool_base <= pool_base + pool_addr_t'(1);         // Advance output address to next pooled pixel.


                  if (q == OUT_SIZE-1) begin                        // If last pooled column in this row:
                      q <= '0;                                      // Wrap column index to 0.
                      if (r == OUT_SIZE-1) begin                    // If last pooled row in this channel:
                          r <= '0;                                  // Wrap row index to 0.
                          if (ch == CHANNELS-1) begin
                              state <= FINISH;
                          end else begin
                              ch        <= ch + ch_t'(1);                       // Next channel
                              conv_base <= conv_base + conv_addr_t'(STEP_CH);   // Jump to next channel’s first window.
                              conv_addr <= conv_base + conv_addr_t'(STEP_CH);   // Issue address for its first a0.
                              conv_en   <= 1'b1;                                // Start next channel's first read.
                              state     <= READ0;                               // Capture a0 next cycle.
                          end
                      end else begin
                          r        <= r + os_t'(1);                             // Next pooled row     
                          conv_base <= conv_base + conv_addr_t'(STEP_ROW);      // Move base down to next row’s first window.
                          conv_addr <= conv_base + conv_addr_t'(STEP_ROW);      // Issue address for a0 of next row.
                          conv_en   <= 1'b1;
                          state     <= READ0;
                      end
                  end else begin
                      q        <= q + os_t'(1);                                // Next pooled column
                      conv_base <= conv_base + conv_addr_t'(STEP_COL);         // Move base right by 2 input pixels.   
                      conv_addr <= conv_base + conv_addr_t'(STEP_COL);         // Issue address for a0 of next pooled column.
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
