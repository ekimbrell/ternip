"""Run the original pinned CUDA/Triton model, independently of candidate numerics."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ternip_accel.artifacts import verify
from ternip_accel.compiler import canonical,digest
p=argparse.ArgumentParser();p.add_argument('--reference-repo',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--lock',required=True);p.add_argument('--tokens',required=True);p.add_argument('--provenance',required=True);p.add_argument('--output',required=True)
p.add_argument('--max-windows',type=int)
a=p.parse_args();lock=json.loads(Path(a.lock).read_text());verify(a.checkpoint,lock)
revision=subprocess.check_output(['git','-C',a.reference_repo,'rev-parse','HEAD'],text=True).strip()
if revision!=lock['reference_revision']: raise SystemExit('Reference checkout is not the pinned revision')
if subprocess.check_output(['git','-C',a.reference_repo,'status','--porcelain'],text=True).strip(): raise SystemExit('Reference checkout must be clean')
if not torch.cuda.is_available(): raise SystemExit('Pinned Triton reference requires CUDA; no quality result produced')
sys.path.insert(0,str(Path(a.reference_repo).resolve()))
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
model=HGRNBitForCausalLM.from_pretrained(a.checkpoint,local_files_only=True,torch_dtype=torch.bfloat16).cuda().eval()
ids=np.load(a.tokens,allow_pickle=False).astype(np.int64);windows=len(ids)//2048;losses=[]
provenance=json.loads(Path(a.provenance).read_text())
if a.max_windows is not None:
    if provenance['split']!='validation' or a.max_windows<1:raise SystemExit('Window limits are for calibration only')
    windows=min(windows,a.max_windows)
if not windows: raise SystemExit('No full 2048-token windows')
with torch.inference_mode():
    for index in range(windows):
        seq=torch.tensor(ids[index*2048:(index+1)*2048],device='cuda').unsqueeze(0)
        logits=model(seq[:,:-1],use_cache=False).logits.float()
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),seq[:,1:].reshape(-1),reduction='sum')
        losses.append(float(loss))
        if (index+1)%10==0 or index+1==windows:print(f"{index+1}/{windows} upstream windows",flush=True)
mean=math.fsum(losses)/(windows*2047)
result=json.loads(Path(a.provenance).read_text())
if result['checkpoint_revision']!=lock['revision'] or result['tokenizer_sha256']!=lock['files']['tokenizer.json']['sha256']:
    raise SystemExit('Token artifact provenance does not match checkpoint/tokenizer')
result.update(torch_version=torch.__version__,device=torch.cuda.get_device_name(0),dtype='bfloat16',backend='pinned-upstream',reference_revision=revision,window=2048,windows=windows,scored_tokens=windows*2047,
              token_sha256=digest(ids.astype('<i8').tobytes()),mean_nll=mean,perplexity=math.exp(mean),remainder_tokens=len(ids)%2048)
Path(a.output).write_bytes(canonical(result))
