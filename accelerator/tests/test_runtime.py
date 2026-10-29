import copy
import json
from pathlib import Path
import numpy as np
import pytest
from ternip_accel.target import *
from ternip_accel.scheduler import Scheduler

BASE = json.loads((Path(__file__).parents[1]/"targets/u280-hbm-reference.json").read_text())

def test_mapping_hash_and_link():
    a = validate(BASE)
    b = copy.deepcopy(BASE)
    b["groups"][0]["ports"]["weights"] = "HBM[4]"
    assert validate(b)["build_hash"] != a["build_hash"]
    assert "sp=ternip_group_0.weights:HBM[0]" in link_config(a)
    assert "SLR3" not in link_config(a)

@pytest.mark.parametrize("field,value",[("slr",3),("slr",-1)])
def test_bad_slrs(field,value):
    b = copy.deepcopy(BASE)
    b["groups"][0][field] = value
    with pytest.raises(ValueError): validate(b)

def test_device_selection():
    devices = [dict(name="xilinx_u250_xdma",id="0"),dict(name="xilinx_u280_xdma",id="1")]
    assert select_device(devices)["id"] == "1"
    with pytest.raises(ValueError): select_device(devices+devices)
    assert select_device(devices,identifier="1")["id"] == "1"
    with pytest.raises(ValueError): select_device(devices,identifier="0")

def test_memory_resolution():
    metadata = [dict(tag="HBM[0:1]",group_id=7,size_bytes=2*PC_BYTES)]
    with pytest.raises(ValueError,match="verified"): resolve_memory(metadata,"HBM[0:1]",100)
    assert resolve_memory(metadata,"HBM[0:1]",100,True) == 7
    with pytest.raises(ValueError,match="capacity"): resolve_memory(metadata,"HBM[0:1]",3*PC_BYTES,True)
    with pytest.raises(ValueError): resolve_memory(metadata,"HBM[2]",100)

def test_allocation_and_exhaustion(engine):
    m = copy.deepcopy(engine.bundle.manifest)
    m["allocation"]["weights_bytes"] = PC_BYTES+1
    t = assign_hbm(BASE,m)
    assert t["groups"][0]["ports"]["weights"] == "HBM[0:1]"
    m["allocation"]["weights_bytes"] = 32*PC_BYTES
    with pytest.raises(ValueError): assign_hbm(BASE,m)

def test_backpressure_cancel_and_lane_reuse(engine):
    s = Scheduler(engine,2)
    s.submit("a",[1,3,4],5,output_capacity=1,eos=12)
    s.submit("b",[4],5,output_capacity=1,eos=12)
    s.advance(3)
    before = s.state.copy()
    s.advance(4)
    np.testing.assert_array_equal(before,s.state)
    first,_ = s.collect("a")
    assert len(first) == 1
    s.cancel("a")
    s.advance()
    assert s.collect("a")[1] in ("cancelled","complete")
    lane = s.submit("c",[2,3],1)
    assert not s.state[lane].any()
    s.advance(2)
    assert len(s.collect("c")[0]) == 1
    s.reset()
    assert not s.state.any()

def test_scheduler_matches_serial(engine):
    def run(lanes):
        s = Scheduler(engine,lanes)
        results = {}
        for i in range(lanes): s.submit(str(i),[1]*(i+1),4,16,eos=12)
        s.advance(8)
        for i in range(lanes): results[str(i)] = s.collect(str(i))[0]
        return results
    assert run(1)["0"] == run(3)["0"]

def test_tie_break_eos_reset(engine):
    class Zero:
        cfg=engine.cfg
        def initial_state(self,n): return np.zeros((n,2,7),np.float32)
        def step(self,tokens,state): return np.zeros((len(tokens),13),np.float32),state+1
    s=Scheduler(Zero(),1)
    s.submit("a",[1],5,eos=0)
    s.advance(4)
    assert s.collect("a") == ([0],"complete")
    s.submit("b",[1],5)
    s.reset()
    assert s.collect("b")[1] == "reset"


def test_rtl_controller_and_tiles_share_placement_and_clock():
    t=copy.deepcopy(BASE);t['backend']='rtl-tiles'
    t['groups'].append(dict(cu='group_b',slr=2,ports=dict(weights='HBM[4]',state='HBM[5]',control='HBM[5]')))
    cfg=link_config(t)
    for group in t['groups']:
        name=group['cu'];tile=name+'_tiles'
        assert f"slr={tile}:SLR{group['slr']}" in cfg
        assert f"stream_connect={name}.tile_command:{tile}.command:32" in cfg
        assert f"stream_connect={tile}.response:{name}.tile_response:32" in cfg
        for cu in (name,tile): assert f"freqHz=300000000:{cu}" in cfg
    other=copy.deepcopy(t);other['backend']='hls-reference'
    assert validate(other)['build_hash']!=validate(t)['build_hash']
    assert 'stream_connect=' not in link_config(other)


def test_unknown_backend_and_rtl_cu_collision_rejected():
    t=copy.deepcopy(BASE);t['backend']='unknown'
    with pytest.raises(ValueError,match='backend'): validate(t)
    t['backend']='rtl-tiles'
    t['groups'].append(dict(cu=t['groups'][0]['cu']+'_tiles',slr=0,
                            ports=dict(weights='HBM[4]',state='HBM[5]',control='HBM[5]')))
    with pytest.raises(ValueError,match='collide'): link_config(t)
