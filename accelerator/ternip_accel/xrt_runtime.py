"""Experimental XRT backend for the HLS reference engine; untested on hardware.

All methods run at command boundaries. advance() launches every group before
waiting, so independent groups execute concurrently. No host tensor arithmetic.
"""
from pathlib import Path
from collections import defaultdict,deque
import json
import numpy as np
from .compiler import align,canonical,digest
from .artifacts import sha256
from .target import validate,select_device,resolve_memory

class XrtRuntime:
    def __init__(self,xclbin,target,device_bundle,build_manifest,identifier=None,grouped_verified=False,
                 prompt_capacity=2048,output_capacity=64):
        import pyxrt as xrt
        self.xrt=xrt;self.target=validate(target);self.root=Path(device_bundle)
        record=json.loads(Path(build_manifest).read_text())
        if validate(record['target'])['build_hash']!=self.target['build_hash'] or record.get('xclbin_sha256')!=sha256(xclbin):
            raise ValueError('Build manifest does not match target/xclbin')
        self.layout=json.loads((self.root/'layout.json').read_text())
        layout_contract=self.layout.copy();layout_hash=layout_contract.pop('build_hash',None)
        if layout_contract.get('version')!=1 or digest(canonical(layout_contract))!=layout_hash:
            raise ValueError('Corrupted or unsupported device layout')
        for name,info in self.layout['files'].items():
            if name not in ('weights.bin','parameters.bin','program.bin'): raise ValueError('Unexpected device artifact')
            if sha256(self.root/name)!=info['sha256']: raise ValueError('Corrupted device artifact')
        self.lanes=target['lanes_per_group'];self.prompt_capacity=prompt_capacity;self.output_capacity=output_capacity
        if prompt_capacity<1 or output_capacity<1: raise ValueError('Invalid queue capacities')
        devices=[]
        for i in range(xrt.enumerate_devices()):
            dev=xrt.device(i)
            devices.append(dict(id=dev.get_info(xrt.xrt_info_device.bdf),name=dev.get_info(xrt.xrt_info_device.name),device=dev))
        selected=select_device(devices,identifier=identifier)
        self.device=selected['device']
        xb=xrt.xclbin(str(xclbin))
        if 'u280' not in xb.get_xsa_name().lower(): raise ValueError('xclbin is not a U280 image')
        # get_mems exposes physical memory on some XRT versions, not grouped
        # entries. Fail closed when the selected stack cannot resolve a group.
        metadata=[dict(tag=m.get_tag(),size_bytes=m.get_size_kb()*1024,
                       group_id=m.get_index(),used=m.get_used()) for m in xb.get_mems()]
        uuid=self.device.load_xclbin(xb)
        self.groups=[];self.requests={};self.pending={}
        committed=defaultdict(int)
        for group in self.target['groups']:
            kernel=xrt.kernel(self.device,uuid,'ternip_group:{'+group['cu']+'}')
            control=np.zeros(self.lanes*(16+prompt_capacity+output_capacity),dtype='<u4')
            arrays=[(self.root/'weights.bin').read_bytes(),bytes(4*self.lanes*self.layout['state_stride']),control.tobytes(),
                    (self.root/'program.bin').read_bytes(),(self.root/'parameters.bin').read_bytes(),
                    bytes(4*self.lanes*self.layout['scratch_stride']),bytes(64)]
            ports=['weights','state','control','control','control','state','control']
            bos=[]
            for index,(data,port) in enumerate(zip(arrays,ports)):
                tag=group['ports'][port];size=align(len(data))
                gid=resolve_memory(metadata,tag,committed[tag]+size,grouped_verified)
                if kernel.group_id(index)!=gid: raise ValueError(f'CU {group["cu"]} argument {index} does not reach {tag}')
                committed[tag]+=size
                bo=xrt.bo(self.device,size,xrt.bo.flags.normal,gid)
                bo.write(data,0);bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,size,0)
                bos.append(bo)
            self.groups.append(dict(kernel=kernel,bos=bos,control=control,owners=[None]*self.lanes))

    def submit(self,request_id,prompt,limit,eos=2):
        if request_id in self.requests: raise ValueError('Duplicate request')
        if not prompt or len(prompt)>self.prompt_capacity or not 0<=limit<2**32: raise ValueError('Invalid request size')
        if any(not isinstance(x,(int,np.integer)) or not 0<=x<self.layout['vocab'] for x in prompt): raise ValueError('Invalid token')
        if not 0<=eos<self.layout['vocab']: raise ValueError('Invalid EOS')
        for gi,g in enumerate(self.groups):
            for lane,owner in enumerate(g['owners']):
                if owner is not None: continue
                c=g['control'];po=16*self.lanes+lane*self.prompt_capacity
                oo=16*self.lanes+self.lanes*self.prompt_capacity+lane*self.output_capacity
                c[po:po+len(prompt)]=prompt
                c[lane*16:(lane+1)*16]=[1,po,len(prompt),0,0,limit,eos,0,0,oo,self.output_capacity,0,1,0,0,0]
                g['owners'][lane]=request_id
                self.requests[request_id]=dict(group=gi,lane=lane,status='active',output=deque())
                return gi,lane
        raise RuntimeError('No free lane')

    def cancel(self,request_id):
        r=self.requests[request_id]
        if r['status']=='active': self.groups[r['group']]['control'][r['lane']*16+7]=1

    def collect(self,request_id):
        r=self.requests[request_id];values=list(r['output']);r['output'].clear()
        return values,r['status']

    def advance(self,quantum=8):
        if not 1<=quantum<2**32: raise ValueError('Invalid quantum')
        xrt=self.xrt;runs=[];l=self.layout
        for g in self.groups:
            if not any(x is not None for x in g['owners']): continue
            c=g['control']
            for lane,owner in enumerate(g['owners']):
                if owner is not None: c[lane*16+10]=self.output_capacity-len(self.requests[owner]['output'])
            g['bos'][2].write(c.tobytes(),0)
            g['bos'][2].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,c.nbytes,0)
            run=xrt.run(g['kernel'])
            args=g['bos']+[l['ops'],l['hidden'],l['layers'],l['vocab'],l['scratch_stride'],l['state_stride'],l['scratch_offsets']['token'],quantum,len(c)]
            for index,value in enumerate(args): run.set_arg(index,value)
            run.start();runs.append((g,run))
        for g,run in runs:
            if run.wait()!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('Kernel command failed')
            c=g['control'];g['bos'][2].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,c.nbytes,0)
            c[:]=np.frombuffer(g['bos'][2].read(c.nbytes,0),dtype='<u4')
            for lane,owner in enumerate(g['owners']):
                if owner is None: continue
                r=c[lane*16:(lane+1)*16];req=self.requests[owner]
                req['output'].extend(int(x) for x in c[r[9]:r[9]+r[11]])
                r[11]=0
                req['status']={1:'active',2:'complete',3:'cancelled',4:'error'}.get(int(r[0]),'error')
                if req['status']!='active': g['owners'][lane]=None

    def reset(self):
        for g in self.groups:
            for owner in g['owners']:
                if owner is not None: self.requests[owner]['status']='reset'
            g['owners']=[None]*self.lanes;g['control'].fill(0)
            for index,size in ((1,4*self.lanes*self.layout['state_stride']),(6,64)):
                bo=g['bos'][index];bo.write(bytes(size),0)
                bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,size,0)
        # submit sets reset_state; each reused lane is cleared on the device.

    def counters(self):
        totals=np.zeros(8,dtype=np.uint64)
        for g in self.groups:
            bo=g['bos'][6]
            bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,64,0)
            totals+=np.frombuffer(bo.read(64,0),dtype='<u8')
        names=['weight_bytes','token_completions','active_lane_steps','available_lane_steps','output_stalls','instructions']
        result={name:int(totals[i]) for i,name in enumerate(names)}
        result['lane_occupancy']=float(totals[2]/totals[3]) if totals[3] else 0.0
        result['kind']='logical-events-not-cycle-counters'
        return result
