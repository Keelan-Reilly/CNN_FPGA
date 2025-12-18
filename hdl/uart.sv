//==============================================================================
// uart.sv — Minimal 8N1 UART RX/TX (LSB-first) for FPGA testbenches + simple SoCs
//
// Implements:
//   • uart_rx: samples an 8N1 frame and pulses rx_dv for 1 cycle per byte
//   • uart_tx: transmits an 8N1 frame when tx_dv is pulsed with tx_byte
//
// Format:
//   • 1 start bit (0), 8 data bits (LSB first), 1 stop bit (1), no parity
//
// Timing model:
//   • CLKS_PER_BIT = CLK_FREQ / BAUD (integer divider)
//   • RX samples each data bit near the centre of the bit period.
//   • Both blocks are fully synchronous to clk; reset is synchronous.
//
// Notes:
//   • For RX: after detecting start, we wait HALF a bit to confirm start,
//     then sample bit0 after one full bit period (≈1.5 bits from edge),
//     then each subsequent bit every 1 bit period.
//==============================================================================


//==============================================================================
// UART Receiver (uart_rx)
//------------------------------------------------------------------------------
// FSM:
//   IDLE    : wait for rx == 0 (start edge)
//   START   : wait half-bit, confirm still low; then begin data timing
//   DATA    : sample 8 bits, one per bit period
//   STOP    : wait 1 bit period, optionally check stop == 1, then pulse rx_dv
//   CLEANUP : return to IDLE
//==============================================================================

module uart_rx #(
    parameter int CLK_FREQ      = 100_000_000,
    parameter int BAUD          = 115200,
    parameter int CLKS_PER_BIT  = CLK_FREQ / BAUD
)(
    input  logic       clk,
    input  logic       reset,
    input  logic       rx,
    output logic       rx_dv,
    output logic [7:0] rx_byte
);

    typedef enum logic [2:0] {IDLE, START, DATA, STOP, CLEANUP} state_t;
    state_t state;

    localparam int CNTW = (CLKS_PER_BIT <= 1) ? 1 : $clog2(CLKS_PER_BIT);
    logic [CNTW-1:0] clk_cnt;
    logic [2:0]      bit_idx;
    logic [7:0]      data;

    // Handy constants for 0-based counters
    localparam int HALF_BIT = (CLKS_PER_BIT/2) - 1;  // mid start-bit sample
    localparam int FULL_BIT = CLKS_PER_BIT - 1;      // one whole bit time

    always_ff @(posedge clk) begin
        if (reset) begin
            state   <= IDLE;
            rx_dv   <= 1'b0;
            rx_byte <= '0;
            clk_cnt <= '0;
            bit_idx <= '0;
            data    <= '0;
        end else begin
            rx_dv <= 1'b0; // default

            unique case (state)

                IDLE: begin
                    clk_cnt <= '0;
                    bit_idx <= '0;
                    if (!rx) begin
                        // Start bit edge detected
                        state   <= START;
                        clk_cnt <= '0;
                    end
                end

                START: begin
                    // Wait to the middle of the start bit and confirm it's still low.
                    if (clk_cnt == HALF_BIT[CNTW-1:0]) begin
                        if (!rx) begin
                            clk_cnt <= '0;
                            bit_idx <= '0;
                            state   <= DATA;
                        end else begin
                            // False start/glitch
                            state <= IDLE;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                DATA: begin
                    // Sample once per full bit period (centre-aligned due to START half-bit wait).
                    if (clk_cnt == FULL_BIT[CNTW-1:0]) begin
                        clk_cnt <= '0;
                        data[bit_idx] <= rx;
                        if (bit_idx == 3'd7) state <= STOP;
                        else                  bit_idx <= bit_idx + 1'b1;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                STOP: begin
                    if (clk_cnt == FULL_BIT[CNTW-1:0]) begin
                        clk_cnt <= '0;
                        // Optional: could check rx==1 here for framing error detection.
                        rx_byte <= data;
                        rx_dv   <= 1'b1;
                        state   <= CLEANUP;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                CLEANUP: begin
                    state <= IDLE;
                end

                default: state <= IDLE;

            endcase
        end
    end
endmodule


//==============================================================================
// UART Transmitter (uart_tx)
//------------------------------------------------------------------------------
// FSM:
//   IDLE    : tx high; wait tx_dv
//   START   : drive 0 for 1 bit
//   DATA    : drive 8 bits (LSB first), 1 bit each
//   STOP    : drive 1 for 1 bit
//   CLEANUP : return to IDLE
//==============================================================================

module uart_tx #(
    parameter int CLK_FREQ      = 100_000_000,
    parameter int BAUD          = 115200,
    parameter int CLKS_PER_BIT  = CLK_FREQ / BAUD
)(
    input  logic       clk,
    input  logic       reset,
    input  logic       tx_dv,
    input  logic [7:0] tx_byte,
    output logic       tx,
    output logic       tx_busy
);

    typedef enum logic [2:0] {IDLE, START, DATA, STOP, CLEANUP} state_t;
    state_t state;

    localparam int CNTW = (CLKS_PER_BIT <= 1) ? 1 : $clog2(CLKS_PER_BIT);
    logic [CNTW-1:0] clk_cnt;
    logic [2:0]      bit_idx;
    logic [7:0]      data_reg;

    localparam int FULL_BIT = CLKS_PER_BIT - 1;

    always_ff @(posedge clk) begin
        if (reset) begin
            state    <= IDLE;
            tx       <= 1'b1;
            tx_busy  <= 1'b0;
            clk_cnt  <= '0;
            bit_idx  <= '0;
            data_reg <= '0;
        end else begin
            unique case (state)

                IDLE: begin
                    tx      <= 1'b1;
                    tx_busy <= 1'b0;
                    clk_cnt <= '0;
                    bit_idx <= '0;
                    if (tx_dv) begin
                        tx_busy  <= 1'b1;
                        data_reg <= tx_byte;
                        state    <= START;
                    end
                end

                START: begin
                    tx <= 1'b0;
                    if (clk_cnt == FULL_BIT[CNTW-1:0]) begin
                        clk_cnt <= '0;
                        bit_idx <= '0;
                        state   <= DATA;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                DATA: begin
                    tx <= data_reg[bit_idx];
                    if (clk_cnt == FULL_BIT[CNTW-1:0]) begin
                        clk_cnt <= '0;
                        if (bit_idx == 3'd7) state <= STOP;
                        else                 bit_idx <= bit_idx + 1'b1;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                STOP: begin
                    tx <= 1'b1;
                    if (clk_cnt == FULL_BIT[CNTW-1:0]) begin
                        clk_cnt <= '0;
                        state   <= CLEANUP;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                CLEANUP: begin
                    state <= IDLE;
                end

                default: state <= IDLE;

            endcase
        end
    end
endmodule
