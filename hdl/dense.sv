//----------------------------------------------------------------------
// Module: dense
// Description:
//   Implements a fully connected (dense) neural network layer.
//
//   Functionality:
//   • Reads a full input vector sequentially from BRAM.
//   • For each output neuron, multiplies every input activation by its 
//     corresponding stored weight, accumulates the products, and adds a bias.
//   • After processing all inputs, the accumulated sum is scaled, saturated, 
//     and written into the output vector array.
//
//   Key points:
//   • This is a matrix–vector multiplication: 
//       out_vec[o] = bias[o] + Σ_i ( W[o,i] * in_vec[i] )
//   • Each output neuron has its own block of weights, covering *all* inputs.
//   • Unlike convolution, which uses small shared kernels, the dense layer 
//     forms a global all-to-all connection between inputs and outputs.
//   • The design streams inputs and weights in sync, using one MAC per cycle, 
//     making it efficient and synthesis-friendly for FPGA implementation.
//----------------------------------------------------------------------

(* keep_hierarchy = "yes" *)
module dense #(
  parameter int DATA_WIDTH = 16,    // bit width for inputs/weights/biases.
  parameter int FRAC_BITS  = 7,     // number of fractional bits (Q-format).
  parameter int IN_DIM     = 1568,  // input vector length.
  parameter int OUT_DIM    = 10,    // output vector length (# neurons).
  parameter int POST_SHIFT = 2,     // extra shift applied after accumulation (acts like scaling).
  parameter string WEIGHTS_FILE = "fc1_weights.mem",
  parameter string BIASES_FILE  = "fc1_biases.mem"
)(
  input  logic clk,
  input  logic reset,
  input  logic start,    

  // Input vector BRAM (read)
  output logic [$clog2(IN_DIM)-1:0]            in_addr,  // address to read input vector
  output logic                                 in_en,    // read enable for input vector
  input  logic  signed [DATA_WIDTH-1:0]        in_q,     // data read from input vector

  output logic signed [DATA_WIDTH-1:0] out_vec [0:OUT_DIM-1],   // output vector (logits written here)

  output logic done
);

  localparam int ACCW = DATA_WIDTH*2 + $clog2(IN_DIM);                   // Accumulator width to avoid overflow plus growth for summing IN_DIM products
  typedef enum logic [2:0] {IDLE, READ, MAC, WRITE, FINISH} state_t;                  
  state_t state;

  integer o, i;                                                // loop indices: o = output neuron index, i = input element index.
  (* use_dsp = "yes" *)         
  logic signed [ACCW-1:0]         acc;                         // Accumulator for MAC operation
  (* use_dsp = "yes" *) logic signed [2*DATA_WIDTH-1:0] prod;  // Product of input * weight

  // Saturation limits; Same as Conv2d Limits
  localparam logic signed [DATA_WIDTH-1:0] S_MAX = (1 <<< (DATA_WIDTH-1)) - 1;  
  localparam logic signed [DATA_WIDTH-1:0] S_MIN = - (1 <<< (DATA_WIDTH-1));
  localparam logic signed [ACCW-1:0] S_MAXX = {{(ACCW-DATA_WIDTH){S_MAX[DATA_WIDTH-1]}}, S_MAX};
  localparam logic signed [ACCW-1:0] S_MINX = {{(ACCW-DATA_WIDTH){S_MIN[DATA_WIDTH-1]}}, S_MIN};

  // Function to compute linear index into weight array
  function automatic int w_idx(input int oo, input int ii);
    return oo*IN_DIM + ii;
  endfunction

  localparam int W_DEPTH = OUT_DIM*IN_DIM;                      // Weight memory depth
  localparam int AW = (W_DEPTH<=1)?1:$clog2(W_DEPTH);           // Address width for weight memory

  localparam int IN_AW = (IN_DIM<=1)?1:$clog2(IN_DIM);           // Address width for input vector memory
  typedef logic [IN_AW-1:0] in_addr_t;                           // Typedef for input address
  typedef logic [AW-1:0]    w_addr_t;                           // Typedef for weight address
  logic [AW-1:0] w_base;                                        // Base address for current output neuron's weight block

  // Weight/Bias ROMs (sync read for W)
  (* rom_style="block", ram_style="block" *)
  logic signed [DATA_WIDTH-1:0] W [0:W_DEPTH-1]; // Weight memory
  (* rom_style="block", ram_style="block" *)
  logic signed [DATA_WIDTH-1:0] B [0:OUT_DIM-1]; // Bias memory

  logic [AW-1:0]                 w_addr;                      // Current weight address
  logic signed [DATA_WIDTH-1:0]  w_q;                        // Registered weight value

  always_ff @(posedge clk) begin
    w_q <= W[w_addr]; // weight read (1-cycle latency)
  end

  // File I/O for weights and biases
  integer fdw, fdb; 
  initial begin
  `ifndef SYNTHESIS
    integer i; integer sumW; integer sumB;

    fdw = $fopen(WEIGHTS_FILE, "r");
    if (fdw == 0) $fatal(1, "%m: cannot open weights file '%s'", WEIGHTS_FILE);
    else $fclose(fdw);
    fdb = $fopen(BIASES_FILE, "r");
    if (fdb == 0) $fatal(1, "%m: cannot open biases file '%s'", BIASES_FILE);
    else $fclose(fdb);
  `endif

    $readmemh(WEIGHTS_FILE, W);
    $readmemh(BIASES_FILE,  B);

  `ifndef SYNTHESIS
  // Tiny checksum to verify non-zero data
  sumW = 0; sumB = 0;
  for (i = 0; i < $size(W); i++) sumW = sumW + W[i];
  for (i = 0; i < $size(B); i++) sumB = sumB + B[i];
  $display("%m: loaded %0d weights, %0d biases; sums: W=%0d B=%0d",
           $size(W), $size(B), sumW, sumB);
  `endif
  end

  // This function sign-extends bias to ACCW bits and shifts left by FRAC_BITS to align
  // with the products (which are in Q(DATA_WIDTH*2-FRAC_BITS))
  function automatic logic signed [ACCW-1:0] bias_ext(input logic signed [DATA_WIDTH-1:0] b);
    return $signed({{(ACCW-DATA_WIDTH){b[DATA_WIDTH-1]}}, b}) <<< FRAC_BITS;
  endfunction

  always_ff @(posedge clk) begin
    if (reset) begin
      state <= IDLE; 
      done <= 1'b0;
      o <= 0; 
      i <= 0; 
      acc <= '0; 
      prod <= '0; 
      w_addr <= '0; w_base <= '0;
      in_en <= 1'b0; in_addr <= '0;
    end else begin
      done  <= 1'b0;
      in_en <= 1'b0;                 // default LOW; raised in READ

      unique case (state)
        // Prime addresses only; wait for `start`
        IDLE: if (start) begin
                 o <= 0; i <= 0;
                 acc     <= bias_ext(B[0]);
                 in_addr <= in_addr_t'(0);
                 w_base  <= w_addr_t'(0);
                 w_addr  <= w_addr_t'(0);
                 state   <= READ;
               end

        // Assert in_en so input BRAM outputs in_q next cycle. Queue up weight address.
        READ: begin
                 in_en <= 1'b1;      // enable happens in READ
                 state <= MAC;
               end

        // Multiply in_q (input activation) and w_q (weight), add to accumulator.
        MAC:  begin
                 automatic logic signed [2*DATA_WIDTH-1:0] p;
                 acc <= acc + ($signed(in_q) * $signed(w_q)); // multiply and accumulate.

                 if (i == IN_DIM-1) begin // if last input element reached:
                   state <= WRITE;        // go write this output neuron’s result.
                 end else begin
                   i       <= i + 1;                  // increment input index.
                   in_addr <= in_addr_t'(i + 1);      // queue address for next input.
                   w_addr  <= w_base + w_addr_t'(i + 1); // queue address for next weight.
                   state   <= READ;                      // in_en will be asserted next cycle
                 end
               end

        // Scale and saturate accumulator, store result in out_vec[o].
        WRITE: begin
                 logic signed [ACCW-1:0]       shifted;
                 logic signed [DATA_WIDTH-1:0] res;
                 shifted = acc >>> (FRAC_BITS + POST_SHIFT);  // Scale down accumulator (arithmetic right shift)
                 if      (shifted > S_MAXX) res <= S_MAX;     // Saturate to max
                 else if (shifted < S_MINX) res <= S_MIN;     // Saturate to min
                 else                        res <= shifted[DATA_WIDTH-1:0];   // Take LSBs because in range
                 out_vec[o] <= res;                           // store into output vector.

                 if (o == OUT_DIM-1) begin                   // if last output neuron:
                   state <= FINISH;                          // all outputs done.
                 end else begin
                   o <= o + 1;                             // increment output index. next output neuron.
                   i <= 0;                                 // reset input index.
                   acc     <= bias_ext(B[o+1]);            // preload next neuron’s bias.
                   in_addr <= in_addr_t'(0);               // restart from input[0].   

                   w_base  <= w_base + w_addr_t'(IN_DIM);  // advance base by IN_DIM (next block of weights).
                   w_addr  <= w_base + w_addr_t'(IN_DIM);  // start of next block.
                   state   <= READ;                        // back to READ to start next neuron.
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
