"""Build/run the resident C++ controller against real RTL cores, with stalls.

Requires Verilator supporting parameterized SystemVerilog class constants and
C++20, plus the pinned BaseJump files. This is RTL/C++ co-simulation, not Vitis
hardware emulation: vector arithmetic remains the controller's C++ model.
"""
import argparse
import hashlib
import shutil
import json
from pathlib import Path
import subprocess
import sys
from rtl_sources import ROOT,sources
p=argparse.ArgumentParser()
p.add_argument('--bsg',type=Path,required=True)
p.add_argument('--verilator',default='verilator')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--case',type=Path,required=True,help='Device test directory containing config.bin')
p.add_argument('--parallelism',type=int,choices=(128,256),default=256)
p.add_argument('--vector-parallelism',type=int,choices=(4,8,16),default=8)
p.add_argument('--jobs',type=int,default=2)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
files,includes=sources(a.bsg)
# A rerun must never inherit a previous passing result after a failed build.
(out/'result.json').unlink(missing_ok=True)
case=a.case.resolve()
input_names=('weights.bin','program.bin','parameters.bin','state.in','control.in','scratch.in','config.bin')
input_hashes={name:hashlib.sha256((case/name).read_bytes()).hexdigest() for name in input_names}

command=[a.verilator,'--cc','--exe','--timing','--assert','--top-module','ternip_tile_group',
         '-GLanes=2',f'-GParallelism={a.parallelism}',f'-GVectorParallelism={a.vector_parallelism}',
         '-Wno-fatal','-DTERNIP_CFG=ternip_pkg::RegressionCfg','--Mdir',str(out),
         '-CFLAGS','-std=c++20 -O2 -ffp-contract=off -DTERNIP_CPU -DTERNIP_RTL_TILES -DTERNIP_LANES=2 -DTERNIP_DEBUG_LOGITS']
command += ['-I'+str(p) for p in includes]+list(map(str,files))
command += [str(ROOT/'accelerator/hls'/name) for name in ('ternip_group.cpp','cpu_runner.cpp','rtl_transport.cpp')]
version=subprocess.check_output([a.verilator,'--version'],text=True).strip()
with (out/'build.log').open('w') as log:
    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    # These are normal GCC settings. Explicit overrides also support binary
    # distributions whose verilated.mk retains build-machine paths/PCH defaults.
    subprocess.run(['make','-C',str(out),'-f','Vternip_tile_group.mk',f'-j{a.jobs}',
                    'CXX=c++','LINK=c++','AR=ar','CFG_CXXFLAGS_PCH_I=-include'],check=True,stdout=log,stderr=subprocess.STDOUT)
protocol=out/'protocol_test'
subprocess.run(['c++','-std=c++20','-O2','-DTERNIP_CPU','-DTERNIP_RTL_TILES','-DTERNIP_LANES=2',
    '-I'+str(ROOT/'accelerator/hls'),str(ROOT/'dv/accelerator/tile_group_test.cpp')]+
    [str(out/name) for name in ('rtl_transport.o','verilated.o','verilated_timing.o','verilated_threads.o','Vternip_tile_group__ALL.a')]+
    ['-pthread','-o',str(protocol)],check=True)
with (out/'protocol.log').open('w') as log:
    subprocess.run([str(protocol)],check=True,stdout=log,stderr=subprocess.STDOUT)
with (out/'run.log').open('w') as log:
    subprocess.run([str(out/'Vternip_tile_group'),str(a.case.resolve())],check=True,stdout=log,stderr=subprocess.STDOUT)
subprocess.run([sys.executable,str(ROOT/'accelerator/scripts/check_device_test.py'),str(a.case.resolve())],check=True)
# Check exact equivalence to the resident portable controller, independently
# of the looser graph-reference tolerance check above. Use fresh copies of the
# original inputs so prior RTL execution cannot contaminate this reference.
portable_case=out/'portable-case';portable_case.mkdir(exist_ok=True)
for name in input_names: shutil.copyfile(case/name,portable_case/name)
portable=out/'portable_runner'
subprocess.run(['c++','-std=c++17','-O2','-ffp-contract=off','-Wno-unknown-pragmas',
    '-DTERNIP_CPU','-DTERNIP_LANES=2','-DTERNIP_DEBUG_LOGITS',
    str(ROOT/'accelerator/hls/ternip_group.cpp'),str(ROOT/'accelerator/hls/cpu_runner.cpp'),
    '-o',str(portable)],check=True)
subprocess.run([str(portable),str(portable_case)],check=True)
outputs={}
for name in ('state.out','scratch.out','control.out'):
    actual=(case/name).read_bytes();expected=(portable_case/name).read_bytes()
    if actual!=expected: raise RuntimeError('RTL/portable byte mismatch: '+name)
    outputs[name]={'byte_identical':True,'bytes':len(actual),'sha256':hashlib.sha256(actual).hexdigest()}
source_paths=files+list((ROOT/'accelerator/hls').glob('*.hpp'))+[
    ROOT/'accelerator/hls'/name for name in ('ternip_group.cpp','cpu_runner.cpp','rtl_transport.cpp')]
source_paths += [ROOT/'rtl/ternip_readmem_path.svh',a.bsg.resolve()/'bsg_misc/bsg_defines.sv',
                 ROOT/'dv/accelerator/tile_group_test.cpp',Path(__file__).resolve()]
source_hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
               hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
(out/'result.json').write_text(json.dumps(dict(status='rtl-cpp-cosimulation-passed',simulator=version,
    lanes=2,tile=256,ternary_parallelism=a.parallelism,vector_parallelism=a.vector_parallelism,
    hardware_qualified=False,input_sha256=input_hashes,source_sha256=source_hashes,
    outputs=outputs),indent=2)+'\n')
