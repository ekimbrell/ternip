"""Held-out evaluation of shared controller arithmetic with exact integer GPU dots."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ternip_accel.compiler import Bundle,canonical,digest
from ternip_accel.controller_emulator import ControllerEmulator
from ternip_accel.quality import nll
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--library',required=True)
p.add_argument('--tokens',required=True);p.add_argument('--provenance',required=True);p.add_argument('--output',required=True)
a=p.parse_args();b=Bundle(a.bundle);result=json.loads(Path(a.provenance).read_text())
if result['checkpoint_revision']!=b.manifest['provenance']['revision'] or result['tokenizer_sha256']!=b.manifest['files']['tokenizer.json']['sha256']:
    raise SystemExit('Token provenance mismatch')
engine=ControllerEmulator(b,Path(a.library).resolve());ids=np.load(a.tokens,allow_pickle=False).astype(np.int64)
windows=len(ids)//2048;losses=[]
if not windows:raise SystemExit('No complete evaluation windows')
for index in range(windows):
    seq=ids[index*2048:(index+1)*2048];logits,_=engine.sequence(seq[:-1])
    losses.append(math.fsum(nll(x,int(y)) for x,y in zip(logits,seq[1:])))
    if (index+1)%5==0 or index+1==windows:print(f'{index+1}/{windows} windows; perplexity {math.exp(math.fsum(losses)/((index+1)*2047)):.6f}',flush=True)
mean=math.fsum(losses)/(windows*2047)
root=Path(__file__).resolve().parents[2]
result.update(backend='shared-controller-fp32-int32',window=2048,windows=windows,scored_tokens=windows*2047,
    token_sha256=digest(ids.astype('<i8').tobytes()),mean_nll=mean,perplexity=math.exp(mean),remainder_tokens=len(ids)%2048,
    model_hash=b.manifest['build_hash'],qualification_scope='portable numerical contract; vendor FPGA FP32 operators remain unvalidated',
    numerical_sources={name:digest((root/'accelerator/hls'/name).read_bytes()) for name in ('scalar_math.hpp','numerical_cpu.cpp')},
    library_sha256=digest(Path(a.library).read_bytes()))
Path(a.output).write_bytes(canonical(result))
