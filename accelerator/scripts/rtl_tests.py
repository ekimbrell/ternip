"""Run committed RTL simulations; propagates compile and simulation failures."""
import argparse
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--verilator',default='verilator');p.add_argument('--output',default='build/rtl-tests')
a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
cases={'broadcast_tb':['rtl/accelerator/ternip_weight_broadcast.sv'],
       'arithmetic_tb':['rtl/ternip_pkg.sv','rtl/math/ternip_add.sv','rtl/math/ternip_sub.sv']}
for name,sources in cases.items():
    command=[a.verilator,'--binary','--timing','--assert','--top-module',name,'-Wno-fatal',
             '-DTERNIP_CFG=ternip_pkg::RegressionCfg','--Mdir',str(out/name)]
    command += [str(root/s) for s in sources]+[str(root/'dv/accelerator'/f'{name}.sv')]
    with (out/f'{name}-build.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    result=subprocess.run([str(out/name/f'V{name}')],check=True,text=True,capture_output=True)
    (out/f'{name}-result.log').write_text(result.stdout+result.stderr);print(result.stdout)
