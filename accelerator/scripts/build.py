"""Build a model controller and optionally real RTL tiles on an installed U280 platform."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ternip_accel.target import validate,link_config
from ternip_accel.platform import inspect_platform
from ternip_accel.compiler import canonical,digest
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--target',required=True);p.add_argument('--output',required=True)
p.add_argument('--bsg',type=Path,help='Pinned BaseJump sources (required for rtl-tiles)')
p.add_argument('--mode',choices=['sw_emu','hw_emu','hw'],default='hw_emu')
a=p.parse_args();t=validate(json.loads(Path(a.target).read_text()))
rtl=t.get('backend')=='rtl-tiles'
if rtl and (not a.bsg or a.mode=='sw_emu'): raise SystemExit('RTL tiles require --bsg and hw_emu or hw')
for tool in (('v++','platforminfo','vivado') if rtl else ('v++','platforminfo')):
    if not shutil.which(tool): raise SystemExit(f'{tool} unavailable; no FPGA build performed')
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
version=subprocess.check_output(['v++','--version'],text=True)
if '2022.2' not in version: raise SystemExit('This build recipe is pinned to Vitis 2022.2')
platform=subprocess.check_output(['platforminfo','-p',t['platform'],'--json'],text=True)
(out/'platforminfo.json').write_text(platform)
# Fail closed if the installed platform lacks explicit required topology.
# Unsupported vendor schemas require an audited parser update, not a bypass.
contract=inspect_platform(json.loads(platform),t)
(out/'platform-contract.json').write_bytes(canonical(contract))
(out/'link.cfg').write_text(link_config(t))
source=root/'accelerator/hls/ternip_group.cpp'
record=dict(target=t,platform_contract=contract,toolchain=version,platform_sha256=digest(platform.encode()),mode=a.mode,
            source_sha256=digest(source.read_bytes()),header_sha256=digest(source.with_suffix('.hpp').read_bytes()))
record['hls_headers_sha256']={p.name:digest(p.read_bytes()) for p in source.parent.glob('*.hpp')}
record['recipe_sha256']={p.name:digest(p.read_bytes()) for p in
    (Path(__file__),root/'accelerator/scripts/check_timing.tcl',root/'accelerator/ternip_accel/platform.py',root/'accelerator/ternip_accel/target.py')}
if rtl:
    from package_inputs import generate
    manifest,rtl_sources=generate(t,a.bsg,out/'rtl')
    record['rtl_sources_sha256']={str(p.relative_to(root)) if p.is_relative_to(root) else p.name:digest(p.read_bytes()) for p in rtl_sources}
    for name in ('package_rtl.tcl','package_inputs.py','rtl_sources.py','fixed_types.py'):
        record['recipe_sha256'][name]=digest((root/'accelerator/scripts'/name).read_bytes())
    record['vivado_version']=subprocess.check_output(['vivado','-version'],text=True)
    if '2022.2' not in record['vivado_version']: raise SystemExit('Vivado 2022.2 is required')
record['build_hash']=digest(canonical(record))
(out/'build.json').write_bytes(canonical(record))
if rtl:
    subprocess.run(['vivado','-mode','batch','-source',str(root/'accelerator/scripts/package_rtl.tcl'),
                    '-tclargs',str(manifest),t['part'],str(out/'rtl')],cwd=out,check=True)
common=['v++','-t',a.mode,'--platform',t['platform'],'--save-temps']
subprocess.run(common+(['-D','TERNIP_RTL_TILES'] if rtl else [])+['-c','-k','ternip_group','--hls.clock',f"{t['clock_mhz']*1000000}:ternip_group",
    '-D',f"TERNIP_LANES={t['lanes_per_group']}",'-D',f"TERNIP_TERNARY_PARALLELISM={t['ternary_parallelism']}",
    '-D',f"TERNIP_VECTOR_PARALLELISM={t['vector_parallelism']}",'-I',str(source.parent),str(source),'-o',str(out/'kernel.xo')],cwd=out,check=True)
link=common+['-l','--config',str(out/'link.cfg')]
if a.mode=='hw': link += ['--vivado.prop','run.impl_1.STEPS.ROUTE_DESIGN.TCL.POST='+str(root/'accelerator/scripts/check_timing.tcl')]
subprocess.run(link+([str(out/'rtl/tiles.xo')] if rtl else [])+[str(out/'kernel.xo'),'-o',str(out/'ternip.xclbin')],cwd=out,check=True)
record.update(status='linked-not-qualified',xclbin_sha256=digest((out/'ternip.xclbin').read_bytes()))
(out/'build.json').write_bytes(canonical(record))
print('Linked. Routed timing, numerical quality, connectivity, and hardware acceptance remain separate gates.')
