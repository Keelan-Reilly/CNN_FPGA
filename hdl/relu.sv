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

    // NOTE: two wait states to be Verilator-NBA safe with a sync-read BRAM model
    typedef enum logic [2:0] {IDLE, WAIT1, WAIT2, CAPTURE, WRITE, FINISH} st_t;
    st_t st;

    addr_t addr;                         // element being written in WRITE
    logic signed [DATA_WIDTH-1:0] v_reg; // captured read value

    function automatic logic signed [DATA_WIDTH-1:0] relu_fn(
        input logic signed [DATA_WIDTH-1:0] x
    );
        return (x < 0) ? '0 : x;
    endfunction

    // Read issue:
    // - On start: issue read for addr=0 (sampled on that posedge)
    // - Each WRITE (except last): issue read for addr+1
    wire issue_first = (st == IDLE)  && start;
    wire issue_next  = (st == WRITE) && (addr != addr_t'(N-1));

    assign conv_r_en   = issue_first || issue_next;
    assign conv_r_addr = issue_first ? addr_t'(0)
                   : issue_next  ? (addr + addr_t'(1))
                   : addr;

    // Write in WRITE state
    assign conv_w_en   = (st == WRITE);
    assign conv_w_we   = (st == WRITE);
    assign conv_w_addr = addr;
    assign conv_w_d    = relu_fn(v_reg);

    always_ff @(posedge clk) begin
        if (reset) begin
            st    <= IDLE;
            done  <= 1'b0;
            addr  <= '0;
            v_reg <= '0;
        end else begin
            done <= 1'b0;

            unique case (st)
                IDLE: begin
                    if (start) begin
                        addr <= '0;   // read of addr=0 is sampled on THIS posedge
                        st   <= WAIT1;
                    end
                end

                // BRAM updates r_q on the *next* posedge (LAT=1), but NBAs mean
                // it's only visible after that edge, so we wait one more cycle.
                WAIT1: st <= WAIT2;
                WAIT2: st <= CAPTURE;

                CAPTURE: begin
                    v_reg <= conv_r_q; // safe capture (r_q has settled from prior edge)
                    st    <= WRITE;
                end

                WRITE: begin
                    if (addr == addr_t'(N-1)) begin
                        st <= FINISH;
                    end else begin
                        addr <= addr + addr_t'(1); // also triggers next read via issue_next
                        st   <= WAIT1;
                    end
                end

                FINISH: begin
                    done <= 1'b1;
                    st   <= IDLE;
                end

                default: st <= IDLE;
            endcase
        end
    end

endmodule

