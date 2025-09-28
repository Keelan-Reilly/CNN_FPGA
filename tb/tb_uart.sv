`timescale 1ns/1ps
`default_nettype none

module tb_uart;
  localparam int CLK_FREQ=100_000_000;
  localparam int BAUD=115200;
  localparam int CLKS_PER_BIT=CLK_FREQ/BAUD;

  logic clk=0, reset=1; always #5 clk=~clk;
  initial begin repeat(4) @(posedge clk); reset=0; end

  // TX side
  logic        tx_dv = 0;
  logic [7:0]  tx_byte = '0;
  logic        tx;
  logic        tx_busy;

  // RX side
  logic        rx_dv;
  logic [7:0]  rx_byte;

  uart_tx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD), .CLKS_PER_BIT(CLKS_PER_BIT)) UTX (
    .clk, .reset, .tx_dv, .tx_byte, .tx, .tx_busy
  );
  uart_rx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD), .CLKS_PER_BIT(CLKS_PER_BIT)) URX (
    .clk, .reset, .rx(tx), .rx_dv, .rx_byte
  );

  task automatic send_byte(input byte b);

    int guard;
    bit got;
    logic [7:0] rcv;
    // single-cycle tx_dv pulse
    @(posedge clk);
    tx_byte <= b; tx_dv <= 1;
    @(posedge clk);
    tx_dv <= 0;
    // wait until rx_dv arrives (bounded)
    guard=0; got=0;
    while (guard< (12*CLKS_PER_BIT*2)) begin // generous
      @(posedge clk); guard++;
      if (rx_dv) begin rcv=rx_byte; got=1; break; end
    end
    if (!got) $fatal(1,"UART: no rx_dv for 0x%0h", b);
    if (rcv!==b) $error("UART: rx 0x%0h != tx 0x%0h", rcv, b);
    else $display("PASS: UART 0x%0h", b);
    // small idle
    repeat(20) @(posedge clk);
  endtask

  initial begin
    @(negedge reset);
    send_byte(8'h55);
    send_byte(8'hAA);
    send_byte("0");
    send_byte("9");
    $display("UART loopback complete.");
    $finish;
  end
endmodule

