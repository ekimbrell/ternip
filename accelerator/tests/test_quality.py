import math
import numpy as np
import pytest
from ternip_accel.quality import nll,compare

def test_stable_nll():
    assert nll([10000,10000],0)==pytest.approx(math.log(2))

def test_gate_requires_independent_matching_reference():
    ref=dict(window=2048,windows=2,scored_tokens=4094,token_sha256='tokens',
             dataset_revision='data',tokenizer_sha256='tok',checkpoint_revision='model',
             backend='pinned-upstream',split='test',mean_nll=2.0)
    candidate=dict(ref,backend='fp32-int8-ir-emulator',mean_nll=2+math.log(1.005))
    assert compare(ref,candidate)['passed']
    candidate['mean_nll']=2+math.log(1.02)
    assert not compare(ref,candidate)['passed']
    candidate['token_sha256']='different'
    with pytest.raises(ValueError): compare(ref,candidate)
    with pytest.raises(ValueError): compare(dict(ref,backend='self-reference'),ref)


def test_no_unqualified_performance_winner():
    from ternip_accel.benchmark import qualified_winner
    assert qualified_winner([{'steady_decode_tokens_per_second':[100,101,102]}]) is None


def test_screening_cannot_be_mistaken_for_hardware_quality_gate():
    with pytest.raises(ValueError,match='Screening'):
        compare({},dict(qualification_eligible=False,mean_nll=0))
