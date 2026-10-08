# Read-only checks of the exact final implementation checkpoint.
if {$argc != 2} {
    error "Expected: final-checkpoint.dcp new-report-directory"
}
set checkpoint [file normalize [lindex $argv 0]]
set report_dir [file normalize [lindex $argv 1]]
if {![file isfile $checkpoint]} { error "Missing checkpoint: $checkpoint" }
if {[file exists $report_dir]} { error "Refusing to overwrite: $report_dir" }
file mkdir $report_dir
set hash_file [open [file join $report_dir checkpoint.sha256] w]
puts $hash_file [exec sha256sum $checkpoint]
close $hash_file
open_checkpoint $checkpoint
report_timing_summary -max_paths 10 -warn_on_violation \
    -file [file join $report_dir final-timing.rpt]
report_accelerator_utilization \
    -kernels {tile_conv:vitis_design_i/tile_conv_1/inst:tile_conv_1} \
    -file [file join $report_dir final-utilization.rpt] -json
report_bus_skew -warn_on_violation -file [file join $report_dir final-bus-skew.rpt]
close_design
exit
