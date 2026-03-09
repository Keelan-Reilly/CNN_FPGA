# Vivado batch flow for one run.
# Args:
#   0 repo_root
#   1 run_dir
#   2 part
#   3 top_module
#   4 xdc_path
#   5 clock_period_ns (optional)
#   6 jobs
#   7 generic_csv (e.g. DATA_WIDTH=16,FRAC_BITS=7)

if { $argc < 7 } {
  puts "ERROR: expected at least 7 args: repo_root run_dir part top xdc clock_period_ns jobs [generic_csv]"
  exit 1
}

set repo_root       [file normalize [lindex $argv 0]]
set run_dir         [file normalize [lindex $argv 1]]
set part            [lindex $argv 2]
set top_module      [lindex $argv 3]
set xdc_path        [file normalize [lindex $argv 4]]
set clock_period_ns [lindex $argv 5]
set jobs            [lindex $argv 6]
set generic_csv     ""
if { $argc >= 8 } {
  set generic_csv [lindex $argv 7]
}

set reports_dir [file join $run_dir reports]
set project_dir [file join $run_dir project]
file mkdir $run_dir
file mkdir $reports_dir
file mkdir $project_dir

if { ![file isdirectory [file join $repo_root hdl]] } {
  puts "ERROR: missing HDL directory under $repo_root"
  exit 1
}
if { ![file exists $xdc_path] } {
  puts "ERROR: XDC file not found: $xdc_path"
  exit 1
}

set hdl_files [lsort [glob -nocomplain [file join $repo_root hdl *.sv]]]
if { [llength $hdl_files] == 0 } {
  puts "ERROR: no HDL sources found in [file join $repo_root hdl]"
  exit 1
}

# top_level uses synthesis-time memory file names without "weights/" prefix.
# Make those files available in the project working directory deterministically.
set required_mem_files {
  conv1_weights.mem
  conv1_biases.mem
  fc1_weights.mem
  fc1_biases.mem
}
foreach memf $required_mem_files {
  set src [file join $repo_root weights $memf]
  if { ![file exists $src] } {
    puts "ERROR: missing required memory file: $src"
    exit 1
  }
  file copy -force $src [file join $project_dir $memf]
}

set generic_tokens {}
if { $generic_csv ne "" } {
  foreach item [split $generic_csv ","] {
    set trimmed [string trim $item]
    if { $trimmed ne "" } {
      lappend generic_tokens $trimmed
    }
  }
}

create_project -force run_project $project_dir -part $part
set_param general.maxThreads $jobs
set_property target_language Verilog [current_project]

foreach src $hdl_files {
  read_verilog -sv $src
}
read_xdc $xdc_path

if { $clock_period_ns ne "" } {
  set clk_list [get_clocks -quiet]
  if { [llength $clk_list] > 0 } {
    set_property PERIOD $clock_period_ns [lindex $clk_list 0]
  } else {
    if { [llength [get_ports -quiet clk]] > 0 } {
      create_clock -name sys_clk -period $clock_period_ns [get_ports clk]
    }
  }
}

set synth_cmd [list synth_design -top $top_module -part $part]
foreach generic_item $generic_tokens {
  lappend synth_cmd -generic $generic_item
}
puts "INFO: Running synth command: $synth_cmd"
cd $project_dir
{*}$synth_cmd

report_utilization -file [file join $reports_dir post_synth_utilization.rpt]
report_timing_summary -file [file join $reports_dir post_synth_timing_summary.rpt] -max_paths 20

opt_design
place_design
phys_opt_design
route_design

report_utilization -file [file join $reports_dir post_route_utilization.rpt]
report_timing_summary -file [file join $reports_dir post_route_timing_summary.rpt] -max_paths 20
report_timing -file [file join $reports_dir post_route_timing_paths.rpt] -max_paths 5 -nworst 1
report_drc -file [file join $reports_dir post_route_drc.rpt]

puts "INFO: Vivado batch run complete. Reports in $reports_dir"
exit 0
