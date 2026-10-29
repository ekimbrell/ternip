"""Generate a fixed-parameter RTL wrapper and Vivado's source manifest."""
from pathlib import Path
from rtl_sources import sources

def tcl_word(path):
    value=Path(path).resolve().as_posix()
    if any(c in value for c in "{}\n\r"):
        raise ValueError('Unsupported Tcl path characters')
    return '{'+value+'}'

def generate(target,bsg,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    files,includes=sources(bsg)
    headers=[Path(bsg).resolve()/'bsg_misc/bsg_defines.sv',Path(__file__).resolve().parents[2]/'rtl/ternip_readmem_path.svh']
    wrapper=out/'ternip_rtl.sv'
    wrapper.write_text(f"""// Generated from the hashed target descriptor.
module ternip_rtl (
    input wire ap_clk, input wire ap_rst_n,
    input wire [511:0] command_TDATA, input wire command_TVALID, output wire command_TREADY,
    output wire [511:0] response_TDATA, output wire response_TVALID, input wire response_TREADY
);
ternip_tile_group #(.Lanes({target['lanes_per_group']}),
    .Parallelism({target['ternary_parallelism']}),.VectorParallelism({target['vector_parallelism']})) group_i (.*);
endmodule
""")
    manifest=out/'inputs.tcl'
    manifest.write_text('set rtl_sources [list '+' '.join(tcl_word(p) for p in files+[wrapper])+']\n'+
                        'set rtl_headers [list '+' '.join(tcl_word(p) for p in headers)+']\n'+
                        'set rtl_includes [list '+' '.join(tcl_word(p) for p in includes)+']\n')
    return manifest,files+[wrapper]+headers
