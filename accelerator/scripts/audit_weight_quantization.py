"""Audit every matrix against the pinned upstream BF16 weight_quant on CUDA."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ternip_accel.artifacts import verify
from ternip_accel.numerics import quantize_weight
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--lock',required=True)
p.add_argument('--reference-repo',required=True);p.add_argument('--output',required=True)
a=p.parse_args();lock=json.loads(Path(a.lock).read_text());verify(a.checkpoint,lock)
revision=subprocess.check_output(['git','-C',a.reference_repo,'rev-parse','HEAD'],text=True).strip()
if revision!=lock['reference_revision']:raise SystemExit('Unpinned upstream')
if subprocess.check_output(['git','-C',a.reference_repo,'status','--porcelain'],text=True).strip():raise SystemExit('Modified upstream')
sys.path.insert(0,str(Path(a.reference_repo).resolve()))
from mmfreelm.ops.fusedbitnet import weight_quant
rows=[]
with safe_open(str(Path(a.checkpoint)/'model.safetensors'),framework='pt',device='cpu') as checkpoint:
    for name in checkpoint.keys():
        tensor=checkpoint.get_tensor(name)
        if tensor.ndim!=2 or not name.endswith('.weight') or name=='model.embeddings.weight':continue
        reference=weight_quant(tensor.to(device='cuda',dtype=torch.bfloat16)).float().cpu().numpy()
        x=tensor.float().numpy();old,_=quantize_weight(x);q,scale=quantize_weight(x,'upstream-bf16')
        actual=np.sign(reference).astype(np.int8)
        mismatch=int(np.count_nonzero(q!=actual))
        scale_mismatch=bool(np.any(reference!=q.astype(np.float32)*scale))
        rows.append(dict(tensor=name,elements=x.size,fp32_weight_code_differences=int(np.count_nonzero(old!=actual)),
                         bf16_weight_code_differences=mismatch,dequantization_mismatch=scale_mismatch))
result=dict(reference_revision=revision,checkpoint_revision=lock['revision'],matrices=rows,
    passed=all(not r['bf16_weight_code_differences'] and not r['dequantization_mismatch'] for r in rows),
    fp32_weight_code_differences=sum(r['fp32_weight_code_differences'] for r in rows),
    elements=sum(r['elements'] for r in rows))
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='matrices'})
if not result['passed']:raise SystemExit('BF16 export differs from pinned weight quantization')
