// uart.sv
module uart_rx #(
    parameter CLKS_PER_BIT = 434 // e.g., 50MHz/115200 ≈ 434
)(
    input  logic clk, reset,
    input  logic rx,                // serial in
    output logic rx_dv,             // data valid (1 clk)
    output logic [7:0] rx_byte
);
    typedef enum logic [2:0] {IDLE, START, DATA, STOP, CLEANUP} state_t;
    state_t state;
    logic [$clog2(CLKS_PER_BIT)-1:0] clk_cnt;
    logic [2:0] bit_idx;
    logic [7:0] data;

    always_ff @(posedge clk) begin
        if (reset) begin state<=IDLE; rx_dv<=0; clk_cnt<='0; bit_idx<=0; data<=0; end
        else begin
            rx_dv <= 1'b0;
            case (state)
                IDLE:   if (~rx) begin state<=START; clk_cnt<='0; end
                START:  if (clk_cnt == (CLKS_PER_BIT/2)) begin clk_cnt<='0; state<=DATA; bit_idx<=0; end
                        else clk_cnt <= clk_cnt + 1;
                DATA:   begin
                            if (clk_cnt == CLKS_PER_BIT-1) begin
                                clk_cnt <= '0;
                                data[bit_idx] <= rx;
                                if (bit_idx==3'd7) state<=STOP;
                                else bit_idx <= bit_idx + 1;
                            end else clk_cnt <= clk_cnt + 1;
                        end
                STOP:   if (clk_cnt == CLKS_PER_BIT-1) begin
                            rx_byte <= data; rx_dv <= 1'b1; state<=CLEANUP; clk_cnt<='0;
                        end else clk_cnt <= clk_cnt + 1;
                CLEANUP: state<=IDLE;
            endcase
        end
    end
endmodule

module uart_tx #(
    parameter CLKS_PER_BIT = 434
)(
    input  logic clk, reset,
    input  logic tx_dv,           // pulse with tx_byte valid
    input  logic [7:0] tx_byte,
    output logic tx,              // serial out
    output logic tx_busy
);
    typedef enum logic [2:0] {IDLE, START, DATA, STOP, CLEANUP} state_t;
    state_t state;
    logic [$clog2(CLKS_PER_BIT)-1:0] clk_cnt;
    logic [2:0] bit_idx;
    logic [7:0] data_reg;

    always_ff @(posedge clk) begin
        if (reset) begin state<=IDLE; tx<=1'b1; tx_busy<=0; clk_cnt<='0; bit_idx<=0; data_reg<=0; end
        else begin
            case (state)
                IDLE:   begin tx<=1'b1; tx_busy<=0; if (tx_dv) begin tx_busy<=1; data_reg<=tx_byte; state<=START; clk_cnt<='0; end end
                START:  begin tx<=1'b0; if (clk_cnt==CLKS_PER_BIT-1) begin clk_cnt<='0; bit_idx<=0; state<=DATA; end else clk_cnt<=clk_cnt+1; end
                DATA:   begin tx<=data_reg[bit_idx]; if (clk_cnt==CLKS_PER_BIT-1) begin clk_cnt<='0; if (bit_idx==3'd7) state<=STOP; else bit_idx<=bit_idx+1; end else clk_cnt<=clk_cnt+1; end
                STOP:   begin tx<=1'b1; if (clk_cnt==CLKS_PER_BIT-1) begin state<=CLEANUP; clk_cnt<='0; end else clk_cnt<=clk_cnt+1; end
                CLEANUP: state<=IDLE;
            endcase
        end
    end
endmodule