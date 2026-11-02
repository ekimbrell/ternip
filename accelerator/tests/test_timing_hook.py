from pathlib import Path
import pytest
tkinter = pytest.importorskip('tkinter', reason='Timing-hook unit tests require Tcl (python3-tk on Linux)')

HOOK = Path(__file__).parents[1] / 'scripts/check_timing.tcl'


def interpreter(failure='none'):
    tcl = tkinter.Tcl()
    tcl.eval('''
proc report_timing_summary {args} {}
proc report_utilization {args} {}
proc get_clocks {args} {return clk}
proc check_timing {args} {return "checking no_clock (0)\nchecking unconstrained_internal_endpoints (0)\nchecking loops (0)\nchecking latch_loops (0)"}
proc get_timing_paths {args} {
    if {[lsearch $args -slack_lesser_than] >= 0} {return {}}
    return path
}
''')
    if failure == 'clocks': tcl.eval('proc get_clocks {args} {return {}}')
    if failure == 'paths': tcl.eval('proc get_timing_paths {args} {return {}}')
    if failure == 'unconstrained':
        tcl.eval('proc check_timing {args} {return "checking no_clock (0)\nchecking unconstrained_internal_endpoints (3)"}')
    if failure == 'unknown_report': tcl.eval('proc check_timing {args} {return "unexpected schema"}')
    if failure in ('max', 'min'):
        tcl.eval('''proc get_timing_paths {args} {
            if {[lsearch $args -slack_lesser_than] < 0} {return path}
            if {[lindex $args 1] == "%s"} {return failing_path}
            return {}
        }''' % failure)
    return tcl


@pytest.mark.parametrize('failure', ['clocks', 'paths', 'unconstrained', 'unknown_report', 'max', 'min'])
def test_incomplete_or_failing_timing_is_rejected(monkeypatch, tmp_path, failure):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(tkinter.TclError, match='Ternip U280'):
        interpreter(failure).eval(HOOK.read_text())


def test_complete_timing_passes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    interpreter().eval(HOOK.read_text())
    assert (tmp_path/'ternip_routed_check_timing.rpt').exists()
