"""Standalone component synthesis. Never qualifies an XRT image or board timing.

An explicit diagnostic part can check tool compatibility when U280 support is
missing, but its resources/timing must not be attributed to a U280.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "accelerator"))
from ternip_accel.compiler import canonical, digest
from ternip_accel.target import validate
from package_inputs import generate, tcl_word


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--flow", choices=("rtl", "hls", "package"), required=True)
    p.add_argument("--bsg", type=Path)
    p.add_argument("--tool", help="Full path to vivado[.bat] or vitis_hls[.bat]")
    p.add_argument("--diagnostic-part", help="Explicit non-U280 compatibility probe; never changes the target descriptor")
    a = p.parse_args()
    target = validate(json.loads(a.target.read_text()))
    if target.get("backend") != "rtl-tiles":
        p.error("This check requires an rtl-tiles descriptor")
    part = a.diagnostic_part or target["part"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", part):
        p.error("Invalid part identifier")
    if a.flow != "hls" and not a.bsg:
        p.error("RTL synthesis/packaging requires --bsg")
    executable = "vitis_hls" if a.flow == "hls" else "vivado"
    tool = a.tool or shutil.which(executable) or shutil.which(executable + ".bat")
    if not tool:
        p.error(executable + " unavailable; provide --tool")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    period = 1000 / target["clock_mhz"]
    record = dict(status="running", target=target, actual_part=part,
                  diagnostic=bool(a.diagnostic_part), flow=a.flow,
                  hardware_qualified=False, tool=str(tool),
                  scope="component synthesis only; no platform link or routed timing")
    if a.flow == "package":
        manifest, files = generate(target, a.bsg, out / "inputs")
        packager = ROOT / "accelerator/scripts/package_rtl.tcl"
        files.append(packager)
        body = f"""puts "TERNIP_TOOL_VERSION=[version -short]"
if {{[llength [get_parts -quiet {part}]] != 1}} {{ error "Required part {part} is not installed" }}
set argv [list {tcl_word(manifest)} {part} {tcl_word(out)}]
set argc 3
source {tcl_word(packager)}
"""
        expected = out / "tiles.xo"
        record["scope"] = "RTL kernel packaging only; no synthesis, platform link or routed timing"
    elif a.flow == "rtl":
        manifest, files = generate(target, a.bsg, out / "inputs")
        clock = out / "clock.xdc"
        clock.write_text(f"create_clock -name ap_clk -period {period:.6f} [get_ports ap_clk]\n")
        files.append(clock)
        body = f"""puts "TERNIP_TOOL_VERSION=[version -short]"
if {{[llength [get_parts -quiet {part}]] != 1}} {{ error "Required part {part} is not installed" }}
source {tcl_word(manifest)}
create_project ternip_ooc {tcl_word(out / 'project')} -part {part}
set_property source_mgmt_mode None [current_project]
add_files -norecurse $rtl_sources
set_property file_type SystemVerilog [get_files $rtl_sources]
add_files -norecurse $rtl_headers
set_property file_type {{Verilog Header}} [get_files $rtl_headers]
set_property include_dirs $rtl_includes [current_fileset]
set_property verilog_define {{TERNIP_CFG=ternip_pkg::RegressionCfg}} [current_fileset]
set_property top ternip_rtl [current_fileset]
add_files -fileset constrs_1 {tcl_word(clock)}
synth_design -top ternip_rtl -part {part} -mode out_of_context
report_utilization -file {tcl_word(out / 'utilization.rpt')}
report_timing_summary -delay_type min_max -report_unconstrained -file {tcl_word(out / 'timing.rpt')}
check_timing -verbose -file {tcl_word(out / 'check_timing.rpt')}
write_checkpoint {tcl_word(out / 'synthesized.dcp')}
"""
        expected = out / "synthesized.dcp"
    else:
        source = ROOT / "accelerator/hls/ternip_group.cpp"
        files = [source, *source.parent.glob("*.hpp")]
        flags = (f"-std=c++14 -ffp-contract=off -DTERNIP_RTL_TILES -DTERNIP_LANES={target['lanes_per_group']} "
                 f"-DTERNIP_TERNARY_PARALLELISM={target['ternary_parallelism']} "
                 f"-DTERNIP_VECTOR_PARALLELISM={target['vector_parallelism']}")
        record["compiler_flags"] = flags
        body = f"""open_project {tcl_word(out / 'project')}
set_top ternip_group
add_files {tcl_word(source)} -cflags "{flags}"
open_solution solution -flow_target vitis
set_part {{{part}}}
create_clock -period {period:.6f} -name default
csynth_design
"""
        expected = out / "project/solution/syn/report/ternip_group_csynth.xml"
    record["source_sha256"] = {str(f): digest(f.read_bytes()) for f in files}
    script = out / "run.tcl"
    # Some HLS Windows launchers return zero even when Tcl failed. Require a
    # success marker AND the expected synthesis artifact in addition to exit 0.
    script.write_text('if {[catch {\n' + body + '\n} message]} {\nputs stderr "TERNIP_COMPONENT_FAILED: $message"\nexit 1\n}\nputs "TERNIP_COMPONENT_SYNTHESIS_PASSED"\nexit 0\n')
    record["script_sha256"] = digest(script.read_bytes())
    record["recipe_sha256"] = digest(Path(__file__).read_bytes())
    report = out / "result.json"
    report.write_bytes(canonical(record))
    command = ([str(tool), "-mode", "batch", "-nojournal", "-log", str(out / "tool.log"), "-source", str(script)]
               if a.flow != "hls" else [str(tool), "-f", str(script)])
    with (out / "console.log").open("wb") as log:
        result = subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT)
    text = (out / "console.log").read_text(errors="replace")
    passed = result.returncode == 0 and "TERNIP_COMPONENT_SYNTHESIS_PASSED" in text.splitlines() and expected.is_file()
    success = "component-packaged-not-qualified" if a.flow == "package" else "component-synthesized-not-qualified"
    record.update(status=success if passed else "failed",
                  returncode=result.returncode, artifact_present=expected.is_file())
    report.write_bytes(canonical(record))
    print(record["status"], report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
