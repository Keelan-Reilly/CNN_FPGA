//======================================================================
// dense.sv — Fully-connected layer (input from BRAM), lint-clean
//======================================================================
(* keep_hierarchy = "yes" *)
module dense #(
  parameter int DATA_WIDTH = 16,
  parameter int FRAC_BITS  = 7,
  parameter int IN_DIM     = 1568,
  parameter int OUT_DIM    = 10,
  parameter int POST_SHIFT = 4, // try 1..3;
  parameter string WEIGHTS_FILE = "fc1_weights.mem",
  parameter string BIASES_FILE  = "fc1_biases.mem"
)(
  input  logic clk,
  input  logic reset,
  input  logic start,

  // Input vector BRAM (read)
  output logic [$clog2(IN_DIM)-1:0]            in_addr,
  output logic                                 in_en,
  input  logic  signed [DATA_WIDTH-1:0]        in_q,

  // Output logits
  output logic signed [DATA_WIDTH-1:0] out_vec [0:OUT_DIM-1],

  output logic done
);

  localparam int ACCW = DATA_WIDTH*2 + $clog2(IN_DIM);
  typedef enum logic [2:0] {IDLE, READ, MAC, WRITE, FINISH} state_t;  // <- 3 bits
  state_t state;

  integer o, i;
  logic signed [ACCW-1:0]         acc;
  (* use_dsp = "yes" *) logic signed [2*DATA_WIDTH-1:0] prod;

  localparam logic signed [DATA_WIDTH-1:0] S_MAX = (1 <<< (DATA_WIDTH-1)) - 1;
  localparam logic signed [DATA_WIDTH-1:0] S_MIN = - (1 <<< (DATA_WIDTH-1));
  localparam logic signed [ACCW-1:0] S_MAXX = {{(ACCW-DATA_WIDTH){S_MAX[DATA_WIDTH-1]}}, S_MAX};
  localparam logic signed [ACCW-1:0] S_MINX = {{(ACCW-DATA_WIDTH){S_MIN[DATA_WIDTH-1]}}, S_MIN};


  function automatic int w_idx(input int oo, input int ii);
    return oo*IN_DIM + ii;
  endfunction

  localparam int W_DEPTH = OUT_DIM*IN_DIM;
  localparam int AW = (W_DEPTH<=1)?1:$clog2(W_DEPTH);

  localparam int IN_AW = (IN_DIM<=1)?1:$clog2(IN_DIM);
  typedef logic [IN_AW-1:0] in_addr_t;
  typedef logic [AW-1:0]    w_addr_t;
  logic [AW-1:0] w_base;

  // Weight/Bias ROMs (sync read for W)
  (* rom_style="block", ram_style="block" *)
  logic signed [DATA_WIDTH-1:0] W [0:W_DEPTH-1];
  (* rom_style="block", ram_style="block" *)
  logic signed [DATA_WIDTH-1:0] B [0:OUT_DIM-1];

  logic [AW-1:0]                 w_addr;
  logic signed [DATA_WIDTH-1:0]  w_q;
  always_ff @(posedge clk) begin
    w_q <= W[w_addr];
  end

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

  function automatic logic signed [ACCW-1:0] bias_ext(input logic signed [DATA_WIDTH-1:0] b);
    return $signed({{(ACCW-DATA_WIDTH){b[DATA_WIDTH-1]}}, b}) <<< FRAC_BITS;
  endfunction

  always_ff @(posedge clk) begin
    if (reset) begin
      state <= IDLE; done <= 1'b0;
      o <= 0; i <= 0; acc <= '0; prod <= '0; w_addr <= '0; w_base <= '0;
      in_en <= 1'b0; in_addr <= '0;
    end else begin
      done  <= 1'b0;
      in_en <= 1'b0;                 // default LOW; we'll raise it in READ

      unique case (state)
        // Prime addresses only; don't assert in_en here
        IDLE: if (start) begin
                 o <= 0; i <= 0;
                 acc     <= bias_ext(B[0]);
                 in_addr <= in_addr_t'(0);
                 w_base  <= w_addr_t'(0);
                 w_addr  <= w_addr_t'(0);
                 state   <= READ;
               end

        // Assert in_en so BRAM outputs in_q for the address set previously
        READ: begin
                 in_en <= 1'b1;      // <-- important: enable happens in READ
                 state <= MAC;
               end

        // Consume in_q & w_q this cycle; queue up next addresses
        MAC:  begin
                 automatic logic signed [2*DATA_WIDTH-1:0] p;
                 acc <= acc + ($signed(in_q) * $signed(w_q));

                 if (i == IN_DIM-1) begin
                   state <= WRITE;
                 end else begin
                   i       <= i + 1;
                   in_addr <= in_addr_t'(i + 1);      // request next input
                   w_addr  <= w_base + w_addr_t'(i + 1); // request next weight
                   state   <= READ;                   // in_en will be asserted next cycle
                 end
               end

        WRITE: begin
                 logic signed [ACCW-1:0]       shifted;
                 logic signed [DATA_WIDTH-1:0] res;
                 shifted = acc >>> (FRAC_BITS + POST_SHIFT);
                 if      (shifted > S_MAXX) res <= S_MAX;
                 else if (shifted < S_MINX) res <= S_MIN;
                 else                        res <= shifted[DATA_WIDTH-1:0];
                 out_vec[o] <= res;

                 if (o == OUT_DIM-1) begin
                   state <= FINISH;
                 end else begin
                   o <= o + 1; 
                   i <= 0;
                   acc     <= bias_ext(B[o+1]);
                   in_addr <= in_addr_t'(0);

                   w_base  <= w_base + w_addr_t'(IN_DIM);
                   w_addr  <= w_base + w_addr_t'(IN_DIM);
                   state   <= READ;    // next cycle we'll assert in_en
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
