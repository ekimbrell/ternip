import argparse
import json
from pathlib import Path
import numpy as np
from .compiler import compile_bundle,Bundle,canonical,digest
from .emulator import Emulator
from .device import lower
from .target import validate,link_config,assign_hbm,sweep,footprint
from .artifacts import fetch,load_checkpoint
from .scheduler import Scheduler

def read(path): return json.loads(Path(path).read_text())
def output(value): print(json.dumps(value,indent=2))

def main():
    p=argparse.ArgumentParser(description="Experimental U280 model/device tooling; no hardware qualification implied")
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('fetch');q.add_argument('--lock',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('export');q.add_argument('--lock',required=True);q.add_argument('--checkpoint',required=True);q.add_argument('--output',required=True);q.add_argument('--tile',type=int,default=256,choices=[128,256]);q.add_argument('--weight-format',choices=['fp32','upstream-bf16'],default='upstream-bf16')
    q=sub.add_parser('lower');q.add_argument('bundle');q.add_argument('--output',required=True)
    q=sub.add_parser('inspect');q.add_argument('bundle');q.add_argument('--lanes',type=int,default=1)
    q=sub.add_parser('target');q.add_argument('descriptor');q.add_argument('--bundle');q.add_argument('--hbm',action='store_true');q.add_argument('--output',required=True)
    q=sub.add_parser('sweep');q.add_argument('descriptor');q.add_argument('bundle');q.add_argument('--output',required=True)
    q=sub.add_parser('generate');q.add_argument('bundle');q.add_argument('--tokens',required=True,help='Comma-separated prompt token IDs');q.add_argument('--limit',type=int,default=16)
    q=sub.add_parser('quality');q.add_argument('bundle');q.add_argument('--tokens',required=True);q.add_argument('--provenance',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('compare-quality');q.add_argument('reference');q.add_argument('candidate');q.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='fetch': fetch(a.output,read(a.lock))
    elif a.command=='export':
        lock=read(a.lock);config,tensors=load_checkpoint(a.checkpoint,lock)
        m=compile_bundle(config,tensors,a.output,a.tile,lock,weight_format=a.weight_format)
        for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
            data=(Path(a.checkpoint)/name).read_bytes();(Path(a.output)/name).write_bytes(data)
            m['files'][name]=dict(bytes=len(data),sha256=digest(data))
        m.pop('build_hash');m['build_hash']=digest(canonical(m))
        (Path(a.output)/'manifest.json').write_bytes(canonical(m));output(m['allocation'])
    elif a.command=='lower': output(lower(Bundle(a.bundle),a.output))
    elif a.command=='inspect':
        b=Bundle(a.bundle);output(dict(manifest=b.manifest,group_footprint=footprint(b.manifest,a.lanes)))
    elif a.command=='target':
        t=validate(read(a.descriptor))
        if a.hbm:
            if not a.bundle: p.error('--hbm requires --bundle')
            t=assign_hbm(t,Bundle(a.bundle).manifest)
        root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
        (root/'target.json').write_bytes(canonical(t));(root/'link.cfg').write_text(link_config(t))
        output(t)
    elif a.command=='sweep':
        candidates=list(sweep(read(a.descriptor),Bundle(a.bundle).manifest))
        Path(a.output).write_bytes(canonical(dict(status='unqualified-candidates',candidates=candidates)))
        output(dict(candidates=len(candidates),qualified_winner=None))
    elif a.command=='generate':
        e=Emulator(Bundle(a.bundle));q=Scheduler(e)
        prompt=[int(x) for x in a.tokens.split(',')]
        q.submit('request',prompt,a.limit,max(1,a.limit));q.advance(len(prompt)+a.limit)
        tokens,status=q.collect('request');output(dict(tokens=tokens,status=status,backend='python-emulator'))
    elif a.command=='quality':
        from .quality import evaluate
        b=Bundle(a.bundle);result=evaluate(Emulator(b),np.load(a.tokens,allow_pickle=False))
        provenance=read(a.provenance)
        if provenance['checkpoint_revision']!=b.manifest['provenance']['revision']: raise ValueError('Checkpoint provenance mismatch')
        if provenance['tokenizer_sha256']!=b.manifest['files']['tokenizer.json']['sha256']: raise ValueError('Tokenizer provenance mismatch')
        for key in ('dataset','dataset_config','dataset_revision','split','tokenizer_sha256','checkpoint_revision','text_join','special_tokens'):
            result[key]=provenance[key]
        result.update(backend='fp32-int8-ir-emulator',model_hash=b.manifest['build_hash'])
        Path(a.output).write_bytes(canonical(result));output(result)
    elif a.command=='compare-quality':
        from .quality import compare
        report=compare(read(a.reference),read(a.candidate));Path(a.output).write_bytes(canonical(report));output(report)
        if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
