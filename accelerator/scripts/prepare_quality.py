"""Create pinned WikiText-2 raw test/validation token artifacts (network explicit)."""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ternip_accel.artifacts import sha256,verify
from ternip_accel.compiler import canonical
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--lock',required=True)
p.add_argument('--dataset-revision',default='b08601e04326c79dfdd32d625aee71d232d685c3');p.add_argument('--split',choices=['validation','test'],required=True);p.add_argument('--output',required=True)
a=p.parse_args()
if not re.fullmatch('[0-9a-f]{40}',a.dataset_revision): raise SystemExit('Pin an immutable dataset commit SHA')
lock=json.loads(Path(a.lock).read_text());verify(a.checkpoint,lock)
dataset=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split=a.split,revision=a.dataset_revision)
text='\n\n'.join(dataset['text'])
tokenizer=Tokenizer.from_file(str(Path(a.checkpoint)/'tokenizer.json'))
ids=tokenizer.encode(text,add_special_tokens=True).ids
root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
np.asarray(ids,dtype='<i8').tofile(root/'tokens.i64')
np.save(root/'tokens.npy',np.asarray(ids,dtype=np.int64),allow_pickle=False)
provenance=dict(dataset='Salesforce/wikitext',dataset_config='wikitext-2-raw-v1',dataset_revision=a.dataset_revision,
                split=a.split,tokenizer_sha256=sha256(Path(a.checkpoint)/'tokenizer.json'),checkpoint_revision=lock['revision'],
                text_join='double-newline',special_tokens=True)
(root/'provenance.json').write_bytes(canonical(provenance))
