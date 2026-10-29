"""Write a non-square, multi-column-tile test for the actual device engine."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ternip_accel.compiler import compile_bundle,Bundle
from ternip_accel.device import lower
from ternip_accel.model import ModelConfig
from ternip_accel.emulator import Emulator
from ternip_accel.scheduler import Scheduler
root=Path(sys.argv[1])
root.mkdir(parents=True,exist_ok=True)
c=dict(model_type="hgrn_bit",hidden_size=263,intermediate_size=271,num_hidden_layers=2,
       vocab_size=17,eos_token_id=16,use_lower_bound=True,rms_norm_eps=1e-6)
rng=np.random.default_rng(184)
t={n:(rng.normal(0,.05,s) if len(s)>1 else rng.uniform(.8,1.2,s)).astype(np.float32)
   for n,s in ModelConfig.from_dict(c).tensor_shapes().items()}
compile_bundle(c,t,root/"bundle",tile=128)
b=Bundle(root/"bundle")
l=lower(b,root/"device")
e=Emulator(b);q=Scheduler(e,2)
control=np.zeros(1024,dtype="<u4")
for lane,prompt in enumerate(([1,4,7],[3])):
    q.submit(str(lane),list(prompt),4,16)
    r=control[lane*16:(lane+1)*16]
    r[:13]=[1,64+lane*32,len(prompt),0,0,4,16,0,0,256+lane*32,16,0,1]
    control[r[1]:r[1]+len(prompt)]=prompt
q.advance(7)
state=np.zeros((2,l["state_stride"]),dtype="<f4")
scratch=np.zeros((2,l["scratch_stride"]),dtype="<f4")
for name,value in [('state.in',state),('scratch.in',scratch),('control.in',control)]: value.tofile(root/"device"/name)
np.array([l['ops'],l['hidden'],l['layers'],l['vocab'],l['scratch_stride'],l['state_stride'],l['scratch_offsets']['token'],7,len(control)],dtype='<u4').tofile(root/"device/config.bin")
q.state.astype('<f4').tofile(root/"device/state.expected")
(root/"device/tokens.expected.json").write_text(json.dumps([q.collect(str(i))[0] for i in range(2)]))
print(root/"device")
