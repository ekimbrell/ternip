"""Specialize the class-only type namespace for a fixed FPGA tile build.

Vivado 2023.2 fails resolving parameterized-class typedef dimensions. Generate
an equivalent package from the original declaration body, leaving arithmetic
sources untouched. Guards reject any instantiated module with a different Cfg.
The raw configurable RTL remains the simulation/regression source of truth.
"""
from pathlib import Path
import re
from rtl_sources import ROOT


def specialize(files, target, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = (ROOT / "rtl/ternip_types.sv").read_text(encoding="utf-8")
    match = re.search(r"class ternip_types #\(.*?\);(.*?)endclass", raw, flags=re.S)
    if not match or raw.count("endclass") != 1:
        raise ValueError("Unsupported type namespace; audit FPGA specialization")
    body = match.group(1)
    if re.search(r"^\s*(?:static\s+)?(?:function|task)\b", body, re.M):
        raise ValueError("Only declarative type namespaces can be specialized")
    package = output / "ternip_fixed_types.sv"
    config = f"""package ternip_fixed_types;
function automatic ternip_pkg::ternip_cfg_t configuration();
    ternip_pkg::ternip_cfg_t c = ternip_pkg::RegressionCfg;
    c.D = 256;
    c.TmatmulParallelism = {target['ternary_parallelism']};
    c.VectorParallelism = {target['vector_parallelism']};
    c.FixedPointExponent = 0;
    c.BatchSize = 1;
    return c;
endfunction
localparam ternip_pkg::ternip_cfg_t Cfg = configuration();
"""
    package.write_bytes((raw[:match.start()] + config + body + "endpackage\n").encode())
    converted = []
    for path in files:
        path = Path(path)
        if path == ROOT / "rtl/ternip_types.sv":
            converted.append(package)
            continue
        text = path.read_text(encoding="utf-8")
        if "ternip_types#(Cfg)::" not in text:
            if re.search(r"ternip_types\s*#\([^)]*\)\s*::", text):
                raise ValueError("Unexpected type configuration: " + str(path))
            if "`TERNIP_CFG" not in text:
                converted.append(path)
                continue
        # Packaged kernels must not depend on the parent project's macro flags.
        text = text.replace("`TERNIP_CFG", "ternip_pkg::RegressionCfg")
        text = text.replace("ternip_types#(Cfg)::", "ternip_fixed_types::")
        if re.search(r"ternip_types\s*#\([^)]*\)\s*::", text):
            raise ValueError("Mixed type configurations: " + str(path))
        def guard(match):
            block = match.group(0)
            if "ternip_fixed_types::" in block:
                block = block.rsplit("endmodule", 1)[0] + """if (Cfg != ternip_fixed_types::Cfg) begin : g_configuration_mismatch
    initial $error("Fixed build types require the exact generated configuration");
end
endmodule"""
            return block
        text = re.sub(r"^module\s.*?^endmodule", guard, text, flags=re.M | re.S)
        dest = output / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(text.encode())
        converted.append(dest)
    return converted
