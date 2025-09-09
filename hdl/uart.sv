//======================================================================
// uart.sv — Simple UART Receiver and Transmitter
//----------------------------------------------------------------------
// What this file provides:
//   • `uart_rx` : Receives serial data (8N1 format), outputs each byte
//                 with a one-cycle `rx_dv` pulse.
//   • `uart_tx` : Transmits bytes over a serial line (8N1 format) when
//                 `tx_dv` is pulsed with a valid `tx_byte`.
//
// Key points (applies to both):
//   • Parameter CLKS_PER_BIT sets baud rate relative to system clock.
//     For example: at 50 MHz and 115200 baud, CLKS_PER_BIT ≈ 434.
//   • Both use small FSMs to step through START, DATA, STOP, CLEANUP.
//   • Data is 8 bits, no parity, 1 stop bit.
//
//======================================================================

//======================================================================
// UART Receiver (uart_rx)
//----------------------------------------------------------------------
// How it works:
//   • IDLE    : Wait for start bit (rx goes low).
//   • START   : Wait half a bit period to align sample point.
//   • DATA    : Sample 8 data bits in sequence.
//   • STOP    : Wait one stop bit; then assert rx_dv and output rx_byte.
//   • CLEANUP : Return to IDLE for next frame.
//======================================================================
module uart_rx #(

    parameter int CLK_FREQ = 50_000_000,  // in Hz
    parameter int BAUD     = 115200,
    parameter CLKS_PER_BIT = CLK_FREQ / BAUD
)(
    input  logic clk, reset,
    input  logic rx,                // serial input line
    output logic rx_dv,             // pulses high for 1 clk when a byte is ready
    output logic [7:0] rx_byte      // received byte
);
    typedef enum logic [2:0] {IDLE, START, DATA, STOP, CLEANUP} state_t;
    state_t state;

    logic [$clog2(CLKS_PER_BIT)-1:0] clk_cnt; // clock counter per bit
    logic [2:0] bit_idx;                      // which data bit (0–7)
    logic [7:0] data;                         // shift register

    always_ff @(posedge clk) begin
        if (reset) begin
            state<=IDLE; rx_dv<=0; clk_cnt<='0; bit_idx<=0; data<=0;
        end else begin
            rx_dv <= 1'b0; // default: not valid
            case (state)
                IDLE:   if (~rx) begin                 // detect start bit (rx=0)
                            state<=START; clk_cnt<='0;
                        end
                START:  if (clk_cnt == (CLKS_PER_BIT/2)) begin
                            // mid-bit sample point
                            clk_cnt<='0; state<=DATA; bit_idx<=0;
                        end else clk_cnt <= clk_cnt + 1;
                DATA:   begin
                            if (clk_cnt == CLKS_PER_BIT-1) begin
                                clk_cnt <= '0;
                                data[bit_idx] <= rx;   // sample data bit
                                if (bit_idx==3'd7) state<=STOP;
                                else bit_idx <= bit_idx + 1;
                            end else clk_cnt <= clk_cnt + 1;
                        end
                STOP:   if (clk_cnt == CLKS_PER_BIT-1) begin
                            rx_byte <= data;           // output full byte
                            rx_dv   <= 1'b1;           // pulse valid
                            state   <= CLEANUP;
                            clk_cnt <= '0;
                        end else clk_cnt <= clk_cnt + 1;
                CLEANUP: state<=IDLE;                  // wait for next start bit
            endcase
        end
    end
endmodule

//======================================================================
// UART Transmitter (uart_tx)
//----------------------------------------------------------------------
// How it works:
//   • IDLE    : Line idle high. Wait for tx_dv pulse with tx_byte.
//   • START   : Drive line low for one bit (start bit).
//   • DATA    : Send 8 data bits, LSB first.
//   • STOP    : Drive line high for one stop bit.
//   • CLEANUP : Return to IDLE.
//
// Notes:
//   • tx_busy is high from START through STOP to block new requests.
//   • tx_dv must be a one-cycle pulse with valid tx_byte.
//======================================================================
module uart_tx #(
    parameter int CLK_FREQ = 50_000_000,  // in Hz
    parameter int BAUD     = 115200,
    parameter CLKS_PER_BIT = CLK_FREQ / BAUD
)(
    input  logic clk, reset,
    input  logic tx_dv,           // pulse high with valid tx_byte
    input  logic [7:0] tx_byte,   // byte to transmit
    output logic tx,              // serial output line
    output logic tx_busy          // high while transmission is in progress
);
    typedef enum logic [2:0] {IDLE, START, DATA, STOP, CLEANUP} state_t;
    state_t state;

    logic [$clog2(CLKS_PER_BIT)-1:0] clk_cnt; // clock counter per bit
    logic [2:0] bit_idx;                      // which data bit (0–7)
    logic [7:0] data_reg;                     // latched tx byte

    always_ff @(posedge clk) begin
        if (reset) begin
            state<=IDLE; tx<=1'b1; tx_busy<=0;
            clk_cnt<='0; bit_idx<=0; data_reg<=0;
        end else begin
            case (state)
                IDLE:   begin
                            tx<=1'b1; tx_busy<=0;     // idle line high
                            if (tx_dv) begin
                                tx_busy<=1;
                                data_reg<=tx_byte;    // latch data
                                state<=START; clk_cnt<='0;
                            end
                        end
                START:  begin
                            tx<=1'b0;                 // start bit
                            if (clk_cnt==CLKS_PER_BIT-1) begin
                                clk_cnt<='0; bit_idx<=0; state<=DATA;
                            end else clk_cnt<=clk_cnt+1;
                        end
                DATA:   begin
                            tx<=data_reg[bit_idx];    // send data bit
                            if (clk_cnt==CLKS_PER_BIT-1) begin
                                clk_cnt<='0;
                                if (bit_idx==3'd7) state<=STOP;
                                else bit_idx<=bit_idx+1;
                            end else clk_cnt<=clk_cnt+1;
                        end
                STOP:   begin
                            tx<=1'b1;                 // stop bit
                            if (clk_cnt==CLKS_PER_BIT-1) begin
                                state<=CLEANUP; clk_cnt<='0;
                            end else clk_cnt<=clk_cnt+1;
                        end
                CLEANUP: state<=IDLE;                 // back to idle
            endcase
        end
    end
endmodule
