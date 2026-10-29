"""Prepare a two-lane, one-token real-checkpoint comparison; not a quality gate."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ternip_accel.compiler import Bundle
from ternip_accel.device import lower
from ternip_accel.emulator import Emulator
p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('--output',required=True);p.add_argument('--token',type=int,default=1)
a=p.parse_args();b=Bundle(a.bundle);root=Path(a.output);l=lower(b,root)
e=Emulator(b);c=np.zeros(256,dtype='<u4')
for lane in range(2):
    c[lane*16:(lane+1)*16]=[1,64+lane,1,0,0,1,b.cfg.eos,0,0,128+lane*16,16,0,1,0,0,0];c[64+lane]=a.token
c.tofile(root/'control.in')
np.zeros(2*l['state_stride'],dtype='<f4').tofile(root/'state.in')
np.zeros(2*l['scratch_stride'],dtype='<f4').tofile(root/'scratch.in')
np.array([l['ops'],l['hidden'],l['layers'],l['vocab'],l['scratch_stride'],l['state_stride'],l['scratch_offsets']['token'],1,len(c)],dtype='<u4').tofile(root/'config.bin')
logits,state=e.step(np.array([a.token,a.token]),e.initial_state(2))
state.astype('<f4').tofile(root/'state.expected');logits.astype('<f4').tofile(root/'logits.expected')
(root/'tokens.expected.json').write_text(json.dumps([[int(x)] for x in logits.argmax(axis=-1)]))
print('Prepared real-model smoke test; this does not establish perplexity or long-sequence equivalence')
