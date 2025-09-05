## =========================================================
## Basys3 (Artix-7 xc7a35tcpg236-1) constraints for CNN top
## Ports: clk, reset, uart_rx_i, uart_tx_o, predicted_digit[3:0]
## =========================================================

# --- Configuration bank voltage (clears CFGBVS/CONFIG_VOLTAGE DRC) ---
set_property CFGBVS VCCO            [current_design]
set_property CONFIG_VOLTAGE 3.3     [current_design]

# --- 100 MHz system clock (onboard oscillator) ---
# Basys3 sysclk is on pin W5, 3.3V LVCMOS
set_property PACKAGE_PIN W5         [get_ports clk]
set_property IOSTANDARD LVCMOS33    [get_ports clk]
create_clock -name sys_clk -period 10.000 [get_ports clk]

# --- Reset button (BTN_C / Center button) ---
# Active-high reset in your CNN code -> press BTN_C to assert reset
# BTN_C is U18 on Basys3
set_property PACKAGE_PIN U18        [get_ports reset]
set_property IOSTANDARD LVCMOS33    [get_ports reset]

# --- USB-UART bridge (onboard FTDI) ---
# FPGA receives on B18 (RsRx) and transmits on A18 (RsTx)
# Map to your UART ports: uart_rx_i <- B18, uart_tx_o -> A18
set_property PACKAGE_PIN B18        [get_ports uart_rx_i]
set_property IOSTANDARD LVCMOS33    [get_ports uart_rx_i]

set_property PACKAGE_PIN A18        [get_ports uart_tx_o]
set_property IOSTANDARD LVCMOS33    [get_ports uart_tx_o]

# --- LEDs for predicted_digit[3:0] ---
# LED0..LED3 are U16, E19, U19, V19 respectively
set_property PACKAGE_PIN U16        [get_ports {predicted_digit[0]}]
set_property IOSTANDARD LVCMOS33    [get_ports {predicted_digit[0]}]

set_property PACKAGE_PIN E19        [get_ports {predicted_digit[1]}]
set_property IOSTANDARD LVCMOS33    [get_ports {predicted_digit[1]}]

set_property PACKAGE_PIN U19        [get_ports {predicted_digit[2]}]
set_property IOSTANDARD LVCMOS33    [get_ports {predicted_digit[2]}]

set_property PACKAGE_PIN V19        [get_ports {predicted_digit[3]}]
set_property IOSTANDARD LVCMOS33    [get_ports {predicted_digit[3]}]
