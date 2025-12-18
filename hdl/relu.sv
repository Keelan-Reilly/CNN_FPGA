//======================================================================
// Module: relu
//
// Overview
//   In-place ReLU over a CHW-linearised feature map stored in a dual-port
//   BRAM. Each element is read from Port A, passed through ReLU, and
//   written back to the same address on Port B.
//
// Operation
//   • The feature map is treated as a single linear array of
//     N = CHANNELS * IMG_SIZE * IMG_SIZE elements.
//   • The module iterates addr = 0 .. N-1:
//       1) issue a synchronous BRAM read on Port A
//       2) wait for the 1-cycle BRAM latency
//       3) capture the read value
//       4) write max(x, 0) back in-place via Port B
//
// Timing model (1-cycle synchronous read BRAM)
//   • Cycle N:   drive conv_r_addr, assert conv_r_en
//   • Cycle N+1: conv_r_q becomes valid (captured in CAPTURE)
//   • Cycle N+2: write occurs in WRITE (Port B)
//
// FSM sequencing (explicit latency handling)
//   IDLE      : wait for `start`, prime addr=0 and issue first read
//   ISSUE     : assert read strobe for current addr
//   WAIT      : one-cycle spacer for BRAM read latency
//   CAPTURE   : latch conv_r_q and set up the matching write address
//   WRITE     : write ReLU result; advance addr and immediately issue next read
//   FINISH    : pulse `done` for one cycle, return to IDLE
//
// Notes
//   • Uses a narrow, sized address counter (addr_t) rather than `integer`.
//   • ReLU is pure combinational logic: (x < 0) ? 0 : x.
//   • In-place update assumes read and write ports are independent.
//======================================================================

(* keep_hierarchy = "yes" *)
module relu #(
    parameter int DATA_WIDTH = 16,
    parameter int CHANNELS   = 8,
    parameter int IMG_SIZE   = 28
)(
    input  logic clk,
    input  logic reset,
    input  logic start,

    // BRAM Port A (read)
    output logic [$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_r_addr,
    output logic                                           conv_r_en,
    input  logic signed [DATA_WIDTH-1:0]                   conv_r_q,

    // BRAM Port B (write)
    output logic [$clog2(CHANNELS*IMG_SIZE*IMG_SIZE)-1:0] conv_w_addr,
    output logic                                           conv_w_en,
    output logic                                           conv_w_we,
    output logic signed [DATA_WIDTH-1:0]                   conv_w_d,

    output logic done
);

    localparam int N  = CHANNELS*IMG_SIZE*IMG_SIZE;
    localparam int AW = (N <= 1) ? 1 : $clog2(N);

    typedef logic [AW-1:0] addr_t;

    addr_t addr;                         // current linear element address
    logic signed [DATA_WIDTH-1:0] v_reg; // captured read value (one element)

    typedef enum logic [2:0] {IDLE, ISSUE, WAIT, CAPTURE, WRITE, FINISH} st_t;
    st_t st;

    // Pure combinational ReLU
    function automatic logic signed [DATA_WIDTH-1:0] relu_fn(
        input logic signed [DATA_WIDTH-1:0] x
    );
        return (x < 0) ? '0 : x;
    endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
            st          <= IDLE;
            done        <= 1'b0;

            addr        <= '0;
            v_reg       <= '0;

            conv_r_addr <= '0;
            conv_r_en   <= 1'b0;

            conv_w_addr <= '0;
            conv_w_en   <= 1'b0;
            conv_w_we   <= 1'b0;
            conv_w_d    <= '0;

        end else begin
            // defaults
            done      <= 1'b0;
            conv_r_en <= 1'b0;
            conv_w_en <= 1'b0;
            conv_w_we <= 1'b0;

            unique case (st)

                // ------------------------------ IDLE ------------------------------
                IDLE: begin
                    if (start) begin
                        addr        <= '0;
                        conv_r_addr <= '0;
                        conv_r_en   <= 1'b1;  // issue first read immediately
                        st          <= WAIT;  // data available next cycle
                    end
                end

                // ----------------------------- ISSUE ------------------------------
                // Kept for clarity; in this implementation, reads are issued from
                // IDLE and WRITE, but ISSUE remains as a consistent hook point.
                ISSUE: begin
                    conv_r_addr <= addr;
                    conv_r_en   <= 1'b1;
                    st          <= WAIT;
                end

                // ------------------------------ WAIT ------------------------------
                // Explicit 1-cycle BRAM read latency.
                WAIT: begin
                    st <= CAPTURE;
                end

                // ---------------------------- CAPTURE -----------------------------
                // Capture the BRAM output and set up the in-place write address.
                CAPTURE: begin
                    v_reg       <= conv_r_q;
                    conv_w_addr <= conv_r_addr; // write back to the same address read
                    st          <= WRITE;
                end

                // ------------------------------ WRITE -----------------------------
                // Write ReLU result, then advance to the next element and issue
                // the next read immediately.
                WRITE: begin
                    conv_w_en <= 1'b1;
                    conv_w_we <= 1'b1;
                    conv_w_d  <= relu_fn(v_reg);

                    if (addr == addr_t'(N-1)) begin
                        st <= FINISH;
                    end else begin
                        addr        <= addr + addr_t'(1);
                        conv_r_addr <= addr + addr_t'(1);
                        conv_r_en   <= 1'b1;   // issue next read
                        st          <= WAIT;   // wait for next data
                    end
                end

                // ----------------------------- FINISH -----------------------------
                FINISH: begin
                    done <= 1'b1;  // one-cycle pulse
                    st   <= IDLE;
                end

                default: st <= IDLE;

            endcase
        end
    end

endmodule
