# Normal timing analysis remains enabled. A linked image is not qualified when
# either routed setup or hold slack is negative.
report_timing_summary -delay_type min_max -report_unconstrained -file ternip_routed_timing.rpt
if {[llength [get_clocks -quiet]] == 0} {
    error "Ternip U280: no timing clocks found"
}
set checks [check_timing -override_defaults {no_clock unconstrained_internal_endpoints loops latch_loops} -return_string]
set channel [open ternip_routed_check_timing.rpt w]
puts $channel $checks
close $channel
foreach check {no_clock unconstrained_internal_endpoints loops latch_loops} {
    set pattern [format {checking %s \(([0-9]+)\)} $check]
    if {![regexp $pattern $checks ignored count] || $count != 0} {
        error "Ternip U280: missing or failing timing check $check"
    }
}
foreach delay {max min} {
    if {[llength [get_timing_paths -delay_type $delay -max_paths 1]] == 0} {
        error "Ternip U280: no $delay timing paths found"
    }
    set failing [get_timing_paths -delay_type $delay -slack_lesser_than 0.0 -max_paths 1]
    if {[llength $failing] != 0} {
        error "Ternip U280: routed $delay timing failed"
    }
}
report_utilization -file ternip_routed_utilization.rpt
