# Normal timing analysis remains enabled. A linked image is not qualified when
# either routed setup or hold slack is negative.
report_timing_summary -delay_type min_max -report_unconstrained -file ternip_routed_timing.rpt
foreach delay {max min} {
    set failing [get_timing_paths -delay_type $delay -slack_lesser_than 0.0 -max_paths 1]
    if {[llength $failing] != 0} {
        error "Ternip U280: routed $delay timing failed"
    }
}
report_utilization -file ternip_routed_utilization.rpt
