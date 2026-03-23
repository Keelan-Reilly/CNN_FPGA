## Minimal Basys3 constraints for the direct MAC-array slice.

set_property CFGBVS VCCO            [current_design]
set_property CONFIG_VOLTAGE 3.3     [current_design]

set_property PACKAGE_PIN W5         [get_ports clk]
set_property IOSTANDARD LVCMOS33    [get_ports clk]
create_clock -name sys_clk -period 10.000 [get_ports clk]

set_property PACKAGE_PIN U18        [get_ports reset]
set_property IOSTANDARD LVCMOS33    [get_ports reset]

set_property PACKAGE_PIN B18        [get_ports start_i]
set_property IOSTANDARD LVCMOS33    [get_ports start_i]

set_property PACKAGE_PIN A18        [get_ports done_o]
set_property IOSTANDARD LVCMOS33    [get_ports done_o]

set_property PACKAGE_PIN U16        [get_ports {signature_o[0]}]
set_property IOSTANDARD LVCMOS33    [get_ports {signature_o[0]}]
set_property PACKAGE_PIN E19        [get_ports {signature_o[1]}]
set_property IOSTANDARD LVCMOS33    [get_ports {signature_o[1]}]
set_property PACKAGE_PIN U19        [get_ports {signature_o[2]}]
set_property IOSTANDARD LVCMOS33    [get_ports {signature_o[2]}]
set_property PACKAGE_PIN V19        [get_ports {signature_o[3]}]
set_property IOSTANDARD LVCMOS33    [get_ports {signature_o[3]}]
