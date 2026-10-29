"""One source manifest for RTL simulation and Vivado packaging."""
import hashlib
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]

def sources(bsg):
    bsg=Path(bsg).resolve()
    lock=json.loads((ROOT/'accelerator/locks/basejump.json').read_text())
    for name in lock['files']:
        data=(bsg/name).read_bytes().replace(b'\r\n',b'\n').rstrip()+b'\n'
        if hashlib.sha256(data).hexdigest()!=lock['sha256'][name]:
            raise ValueError(f'BaseJump source does not match lock: {name}')
    core=yaml.safe_load((ROOT/'ternip.core').read_text())
    files=[bsg/p for p in lock['files'] if not p.endswith('bsg_defines.sv')]
    files += [ROOT/p for p in core['filesets']['rtl']['files'] if isinstance(p,str)]
    files += [ROOT/'rtl/accelerator'/p for p in ('ternip_tile_lane.sv','ternip_weight_broadcast.sv','ternip_tile_group.sv')]
    return files,[ROOT/'rtl',bsg/'bsg_misc']
