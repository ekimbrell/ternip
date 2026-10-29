"""CUDA numerical screening; output explicitly ineligible for hardware qualification."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ternip_accel.compiler import Bundle,canonical,digest
from ternip_accel.cuda_screen import CudaScreen
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--tokens',required=True)
p.add_argument('--provenance',required=True);p.add_argument('--output',required=True)
p.add_argument('--fraction-bits',type=int);p.add_argument('--max-windows',type=int)
a=p.parse_args();provenance=json.loads(Path(a.provenance).read_text())
if a.fraction_bits is not None and provenance['split']!='validation':
    raise SystemExit('Calibrate numerical formats only on validation data; fixed-point screening is not yet a qualified format')
bundle=Bundle(a.bundle)
if provenance['checkpoint_revision']!=bundle.manifest['provenance']['revision'] or provenance['tokenizer_sha256']!=bundle.manifest['files']['tokenizer.json']['sha256']:
    raise SystemExit('Token provenance does not match the exported checkpoint/tokenizer')
engine=CudaScreen(bundle,a.fraction_bits)
ids=np.load(a.tokens,allow_pickle=False).astype(np.int64);windows=len(ids)//2048
if a.max_windows is not None:
    if provenance['split']!='validation' or a.max_windows<1:raise SystemExit('Window limits are for calibration only')
    windows=min(windows,a.max_windows)
if not windows:raise SystemExit('No complete evaluation windows')
losses=[]
with torch.inference_mode():
    for index in range(windows):
        seq=ids[index*2048:(index+1)*2048]
        logits,_=engine.sequence(seq[:-1])
        targets=torch.as_tensor(seq[1:],device='cuda',dtype=torch.int64)
        loss=torch.nn.functional.cross_entropy(logits,targets,reduction='sum')
        losses.append(float(loss))
        if (index+1)%10==0 or index+1==windows:print(f'{index+1}/{windows} windows; perplexity {math.exp(math.fsum(losses)/((index+1)*2047)):.6f}',flush=True)
mean=math.fsum(losses)/(windows*2047)
result=dict(provenance,backend='cuda-numerical-screening',qualification_eligible=False,
    qualification_reason='CUDA floating-point reductions and transcendentals are not bit-accurate to the controller',
    window=2048,windows=windows,scored_tokens=windows*2047,token_sha256=digest(ids.astype('<i8').tobytes()),
    mean_nll=mean,perplexity=math.exp(mean),remainder_tokens=len(ids)%2048,
    model_build_hash=bundle.manifest['build_hash'],fraction_bits=a.fraction_bits,
    torch_version=torch.__version__,device=torch.cuda.get_device_name(0))
Path(a.output).write_bytes(canonical(result))
