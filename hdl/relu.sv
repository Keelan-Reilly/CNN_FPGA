(* keep_hierarchy = "yes" *)
module relu #(
    parameter int DATA_WIDTH = 16, // Bit-width of activations
    parameter int CHANNELS   = 8,  // Number of feature map channels
    parameter int IMG_SIZE   = 28  // Height/width of each channel
)(
    input  logic clk,
    input  logic reset,
    input  logic start,

    // ------------------------------------------------------------
    // BRAM Port A (read-only)
    // ------------------------------------------------------------
    output logic [$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_r_addr,
    output logic                                           conv_r_en,
    input  logic signed [DATA_WIDTH-1:0]                   conv_r_q,

    // ------------------------------------------------------------
    // BRAM Port B (write-only)
    // ------------------------------------------------------------
    output logic [$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_w_addr,
    output logic                                           conv_w_en,
    output logic                                           conv_w_we,
    output logic signed [DATA_WIDTH-1:0]                   conv_w_d,

    // Pulsed for one cycle when the entire ReLU pass completes
    output logic done
);

    // ------------------------------------------------------------
    // Total number of elements and address width
    // ------------------------------------------------------------
    localparam int N  = CHANNELS * IMG_SIZE * IMG_SIZE;
    localparam int AW = (N <= 1) ? 1 : $clog2(N);
    typedef logic [AW-1:0] addr_t;

    // ------------------------------------------------------------
    // FSM states
    //
    // Two explicit wait states are used to be safe with:
    //  • synchronous-read BRAM
    //  • nonblocking assignment (NBA) ordering
    //  • Verilator timing semantics
    // ------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE,     // Waiting for start
        WAIT1,    // BRAM latency stage 1
        WAIT2,    // BRAM latency stage 2 (NBA-safe)
        CAPTURE,  // Capture read data
        WRITE,    // Write ReLU result back in-place
        FINISH    // Signal completion
    } st_t;

    st_t st;

    // ------------------------------------------------------------
    // Address and data registers
    // ------------------------------------------------------------
    addr_t addr;                         // Current element index
    logic signed [DATA_WIDTH-1:0] v_reg; // Latched BRAM read value

    // ------------------------------------------------------------
    // ReLU activation function
    // ------------------------------------------------------------
    function automatic logic signed [DATA_WIDTH-1:0] relu_fn(
        input logic signed [DATA_WIDTH-1:0] x
    );
        return (x < 0) ? '0 : x;
    endfunction

    // ------------------------------------------------------------
    // Read-issue control
    //
    // • First read is issued when start is seen in IDLE
    // • Subsequent reads are issued during WRITE (for addr+1)
    // • Read address is sampled on the issuing clock edge
    // ------------------------------------------------------------
    wire issue_first = (st == IDLE)  && start;
    wire issue_next  = (st == WRITE) && (addr != addr_t'(N-1));

    assign conv_r_en   = issue_first || issue_next;

    assign conv_r_addr =
        issue_first ? addr_t'(0) :
        issue_next  ? (addr + addr_t'(1)) :
                      addr;

    // ------------------------------------------------------------
    // Write port control
    //
    // In-place ReLU:
    //   read addr i → apply ReLU → write back to addr i
    // ------------------------------------------------------------
    assign conv_w_en   = (st == WRITE);
    assign conv_w_we   = (st == WRITE);
    assign conv_w_addr = addr;
    assign conv_w_d    = relu_fn(v_reg);

    // ------------------------------------------------------------
    // Sequential control logic
    // ------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            st    <= IDLE;
            done  <= 1'b0;
            addr  <= '0;
            v_reg <= '0;
        end else begin
            done <= 1'b0;

            unique case (st)

                // --------------------------------------------------
                // IDLE: wait for start pulse
                // --------------------------------------------------
                IDLE: begin
                    if (start) begin
                        addr <= '0;
                        // Read of addr=0 is sampled on THIS posedge
                        st   <= WAIT1;
                    end
                end

                // --------------------------------------------------
                // WAIT1 / WAIT2:
                //   BRAM read latency + NBA safety margin
                // --------------------------------------------------
                WAIT1: st <= WAIT2;
                WAIT2: st <= CAPTURE;

                // --------------------------------------------------
                // CAPTURE:
                //   Safely latch conv_r_q after BRAM + NBA delay
                // --------------------------------------------------
                CAPTURE: begin
                    v_reg <= conv_r_q;
                    st    <= WRITE;
                end

                // --------------------------------------------------
                // WRITE:
                //   Write ReLU result and advance address
                // --------------------------------------------------
                WRITE: begin
                    if (addr == addr_t'(N-1)) begin
                        st <= FINISH;
                    end else begin
                        addr <= addr + addr_t'(1);
                        // Next read is triggered combinationally via issue_next
                        st   <= WAIT1;
                    end
                end

                // --------------------------------------------------
                // FINISH: pulse done and return to IDLE
                // --------------------------------------------------
                FINISH: begin
                    done <= 1'b1;
                    st   <= IDLE;
                end

                default: st <= IDLE;
            endcase
        end
    end

endmodule
