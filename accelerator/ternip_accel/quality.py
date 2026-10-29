"""Identical nonoverlapping 2048-token windows with state reset per window."""
import math
import numpy as np
from .compiler import digest

def nll(logits, target):
    x=np.asarray(logits,dtype=np.float64)
    peak=float(np.max(x))
    return peak+math.log(float(np.exp(x-peak).sum()))-float(x[target])

def evaluate(engine, tokens, window=2048):
    ids=np.asarray(tokens,dtype=np.int64)
    if window!=2048: raise ValueError("The acceptance protocol requires 2048-token windows")
    windows=len(ids)//window
    if not windows: raise ValueError("At least one complete window is required")
    losses=[]
    for index in range(windows):
        seq=ids[index*window:(index+1)*window]
        state=engine.initial_state(1)
        for token,target in zip(seq[:-1],seq[1:]):
            logits,state=engine.step(np.array([token]),state)
            losses.append(nll(logits[0],target))
    mean=math.fsum(losses)/len(losses)
    return dict(window=window,windows=windows,scored_tokens=len(losses),
        token_sha256=digest(ids.astype('<i8').tobytes()),mean_nll=mean,perplexity=math.exp(mean),
        remainder_tokens=len(ids)%window)

def compare(reference,candidate):
    if candidate.get("qualification_eligible") is False: raise ValueError("Screening results cannot qualify hardware numerics")
    for key in ('window','windows','scored_tokens','token_sha256','dataset_revision','tokenizer_sha256','checkpoint_revision'):
        if reference.get(key) is None or reference.get(key)!=candidate.get(key):
            raise ValueError(f"Mismatched or missing quality provenance: {key}")
    if reference.get('backend')!='pinned-upstream': raise ValueError("Quality baseline must execute pinned upstream")
    if reference.get('split')!='test' or candidate.get('split')!='test': raise ValueError("Acceptance requires held-out test data")
    degradation=math.expm1(candidate['mean_nll']-reference['mean_nll'])
    if not math.isfinite(degradation): raise ValueError("Nonfinite quality result")
    return dict(passed=degradation<=0.01,relative_perplexity_degradation=degradation,
                reference=reference,candidate=candidate)
