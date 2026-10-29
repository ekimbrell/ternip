import copy
import json
from pathlib import Path
import pytest
from ternip_accel.platform import inspect_platform

TARGET=json.loads((Path(__file__).parents[1]/'targets/u280-hbm-rtl.json').read_text())

def evidence():
    return dict(hardwarePlatform=dict(part=TARGET['part'],numSlrs=3,
        memories=[dict(tag=f'HBM[{i}]') for i in range(32)],
        systemClocks=[dict(name='kernel',frequencyMHz=300)]))

def test_explicit_platform_topology_and_clock():
    result=inspect_platform(evidence(),TARGET)
    assert result['slrs']==[0,1,2]
    assert result['requested_clock_hz']==300000000
    assert len(result['memory_tags'])==32
    t=copy.deepcopy(TARGET);t['groups'][0]['ports']['weights']='HBM[4:5]'
    assert inspect_platform(evidence(),t)['memory_tags']==result['memory_tags']

@pytest.mark.parametrize('field,value,match',[
    ('part','xcu250-figd2104-2L-e','part'),('numSlrs',4,'SLR'),
    ('systemClocks',[dict(frequency=300)],'clock'),
    ('systemClocks',[dict(frequencyMHz=200)],'clock'),
    ('memories',[dict(description='HBM is installed')],'memory')])
def test_missing_or_mismatched_evidence_is_rejected(field,value,match):
    info=evidence();info['hardwarePlatform'][field]=value
    with pytest.raises(ValueError,match=match):inspect_platform(info,TARGET)
