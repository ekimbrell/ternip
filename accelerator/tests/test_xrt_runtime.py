import json
import sys
import types
from pathlib import Path
import numpy as np
import pytest
from ternip_accel.compiler import canonical,digest
from ternip_accel.device import lower
from ternip_accel.target import validate,PC_BYTES
from ternip_accel.xrt_runtime import XrtRuntime

@pytest.fixture
def mocked_xrt(monkeypatch,tmp_path,engine):
    events=[]
    class Mem:
        def __init__(self,i): self.i=i
        def get_tag(self): return f'HBM[{self.i}]'
        def get_size_kb(self): return PC_BYTES//1024
        def get_index(self): return self.i
        def get_used(self): return True
    class Device:
        def __init__(self,i): pass
        def get_info(self,key): return 'xilinx_u280_xdma' if key=='name' else '0000:01:00.1'
        def load_xclbin(self,xb): return 'uuid'
    class Xclbin:
        def __init__(self,path): pass
        def get_xsa_name(self): return 'xilinx_u280_xdma'
        def get_mems(self): return [Mem(i) for i in range(6)]
    class BO:
        flags=types.SimpleNamespace(normal=0)
        def __init__(self,dev,size,flags,gid): self.data=bytearray(size);self.gid=gid
        def write(self,data,offset): self.data[offset:offset+len(data)]=data
        def read(self,size,offset): return np.frombuffer(self.data,dtype=np.int8)[offset:offset+size]
        def sync(self,*args): pass
    class Kernel:
        def __init__(self,dev,uuid,name): self.group=int(name.split('_')[-1].rstrip('}'))
        def group_id(self,index): return [0,1,2,2,2,1,2][index]+3*self.group
    class Run:
        def __init__(self,kernel): self.args={};self.k=kernel
        def set_arg(self,i,value): self.args[i]=value
        def start(self): events.append(('start',self.k.group))
        def wait(self):
            events.append(('wait',self.k.group))
            c=np.frombuffer(self.args[2].data,dtype='<u4')
            if c[0]==1:
                c[c[9]]=5;c[11]=1;c[0]=2
            return 4
    fake=types.SimpleNamespace(device=Device,xclbin=Xclbin,bo=BO,kernel=Kernel,run=Run,
          enumerate_devices=lambda:1,xrt_info_device=types.SimpleNamespace(name='name',bdf='bdf'),
          ert_cmd_state=types.SimpleNamespace(ERT_CMD_STATE_COMPLETED=4),
          xclBOSyncDirection=types.SimpleNamespace(XCL_BO_SYNC_BO_TO_DEVICE=0,XCL_BO_SYNC_BO_FROM_DEVICE=1))
    monkeypatch.setitem(sys.modules,'pyxrt',fake)
    lower(engine.bundle,tmp_path/'device')
    target=json.loads((Path(__file__).parents[1]/'targets/u280-hbm-reference.json').read_text())
    target['groups'].append(dict(cu='ternip_group_1',slr=1,ports=dict(weights='HBM[3]',state='HBM[4]',control='HBM[5]')))
    xb=tmp_path/'test.xclbin';xb.write_bytes(b'mock-only')
    report=tmp_path/'build.json';report.write_bytes(canonical(dict(target=validate(target),xclbin_sha256=digest(xb.read_bytes()))))
    return (xb,target,tmp_path/'device',report),events,fake

def test_xrt_queue_and_group_concurrency(mocked_xrt):
    args,events,_=mocked_xrt
    runtime=XrtRuntime(*args,prompt_capacity=8,output_capacity=4)
    runtime.submit('a',[1,3],2);runtime.submit('b',[4],2)
    runtime.advance(2)
    assert events==[('start',0),('start',1),('wait',0),('wait',1)]
    assert runtime.collect('a')==([5],'complete')
    runtime.submit('c',[3],2)
    assert runtime.groups[0]['control'][12]==1
    runtime.cancel('c')
    assert runtime.groups[0]['control'][7]==1
    runtime.reset()
    assert runtime.collect('c')[1]=='reset'

def test_xrt_rejects_wrong_reachability(mocked_xrt):
    args,_,fake=mocked_xrt
    fake.kernel.group_id=lambda self,index:31
    with pytest.raises(ValueError,match='does not reach'): XrtRuntime(*args)

def test_xrt_rejects_mismatched_image(mocked_xrt):
    args,_,_=mocked_xrt
    args[0].write_bytes(b'changed')
    with pytest.raises(ValueError,match='manifest'): XrtRuntime(*args)
