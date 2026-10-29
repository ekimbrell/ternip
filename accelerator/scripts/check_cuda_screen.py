"""Bound screening drift on a supplied bundle; does not certify bit accuracy."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ternip_accel.compiler import Bundle
from ternip_accel.emulator import Emulator
from ternip_accel.cuda_screen import CudaScreen
p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('--output',required=True)
a=p.parse_args();b=Bundle(a.bundle);gpu=CudaScreen(b);cpu=Emulator(b)
tokens=[1,3,5];state=cpu.initial_state(1);expected=[]
for token in tokens:
    logits,state=cpu.step(np.array([token]),state);expected.append(logits[0])
actual,gs=gpu.sequence(tokens);actual=actual.cpu().numpy();gs=gs.cpu().numpy()
expected=np.stack(expected)
np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=5e-4)
np.testing.assert_allclose(gs,state[0],atol=2e-5,rtol=5e-4)
np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))
result=dict(status='screening-smoke-passed-not-bit-accurate',tokens=tokens,
    logits_max_abs_error=float(np.max(np.abs(actual-expected))),
    state_max_abs_error=float(np.max(np.abs(gs-state[0]))),qualification_eligible=False)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(result)
