"""Require bit-identical checkpoint logits/state to the portable controller fixture."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ternip_accel.compiler import Bundle
from ternip_accel.controller_emulator import ControllerEmulator
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--library',required=True)
p.add_argument('--device-case',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.device_case);layout=json.loads((root/'layout.json').read_text())
engine=ControllerEmulator(Bundle(a.bundle),Path(a.library).resolve())
def result_file(name):
    pinned=root/(name+'.portable')
    return pinned if pinned.exists() else root/name
control=np.fromfile(result_file('control.out'),dtype='<u4')
initial=np.fromfile(root/'control.in',dtype='<u4')
expected_state=np.fromfile(result_file('state.out'),dtype='<f4').reshape(2,layout['layers'],layout['hidden'])
scratch=np.fromfile(result_file('scratch.out'),dtype='<f4').reshape(2,layout['scratch_stride'])
start=layout['scratch_offsets']['logits'];cases=[]
for lane in range(2):
    r=control[16*lane:16*(lane+1)];original=initial[16*lane:16*(lane+1)]
    assert r[0]==2 and r[11]>0,'Expected completed nonempty request'
    prompt=initial[original[1]:original[1]+original[2]].tolist()
    output=control[r[9]:r[9]+r[11]].tolist()
    tokens=prompt+output[:-1]
    logits,state=engine.sequence(tokens)
    expected_logits=scratch[lane,start:start+layout['vocab']]
    assert logits[-1].tobytes()==expected_logits.tobytes(),'Controller logits are not bit-identical'
    assert state.tobytes()==expected_state[lane].tobytes(),'Controller state is not bit-identical'
    assert logits[len(prompt)-1:].argmax(-1).tolist()==output,'Controller tokens differ'
    cases.append(dict(lane=lane,input_tokens=len(tokens),generated_tokens=output))
result=dict(status='bit-identical-controller-smoke-passed',cases=cases,layers=layout['layers'],vocab=layout['vocab'],
    logits_max_abs_error=0,state_max_abs_error=0,scope='fixture sequences; vendor FPGA float operators remain unvalidated')
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(result)
