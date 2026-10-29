"""Optional fast numerical screening, NOT a bit-accurate hardware emulator.

Integer dot products are exact int8->int32. CUDA's FP32 reductions/transcendentals
can differ from the portable controller, so these reports cannot qualify a
hardware numerical format. Sequence batching accelerates calibration/evaluation;
recurrence remains causal and starts from zero at every evaluation window.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from .numerics import unpack_ternary

@triton.jit
def _recurrent(I,F,O,H,T:tl.constexpr,D:tl.constexpr,B:tl.constexpr,FRACTION:tl.constexpr):
    j=tl.program_id(0)*B+tl.arange(0,B)
    state=tl.full((B,),0,tl.float32)
    for t in range(T):
        i=tl.load(I+t*D+j,j<D,other=0).to(tl.float32)
        f=tl.load(F+t*D+j,j<D,other=0).to(tl.float32)
        state=f*state+(1.0-f)*i
        if FRACTION>=0:
            scale=2.0**FRACTION
            # Round ties to even and saturate signed 32-bit vector boundaries.
            scaled=state*scale
            floor=tl.floor(scaled)
            remainder=scaled-floor
            rounded=floor+((remainder>0.5)|((remainder==0.5)&((floor.to(tl.int64)&1)!=0))).to(tl.float32)
            state=tl.minimum(tl.maximum(rounded,-2147483648.0),2147483520.0)/scale
        tl.store(O+t*D+j,state,j<D)
    tl.store(H+j,state,j<D)

class CudaScreen:
    def __init__(self,bundle,fraction_bits=None):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required for numerical screening')
        if fraction_bits is not None and not 0<=fraction_bits<=24: raise ValueError('Invalid fixed-point fraction bits')
        self.bundle=bundle;self.cfg=bundle.cfg;self.fraction=fraction_bits
        self.p={k:torch.from_numpy(np.array(v,copy=True)).cuda() for k,v in bundle.parameters.items()}
        self.weights={}
        torch.backends.cuda.matmul.allow_tf32=False
        for name,info in bundle.manifest['tensors'].items():
            rows,cols=info['shape'];pr,pc=info['padded_shape'];start=info['offset']
            w=unpack_ternary(bundle.weights[start:start+info['bytes']],(pr,pc))[:rows,:cols]
            value=torch.from_numpy(w.copy()).cuda()
            value=torch.nn.functional.pad(value,(0,(-cols)%16,0,(-rows)%16)).t().contiguous()
            self.weights[name]=value

    def boundary(self,x):
        if self.fraction is None:return x
        scale=2.0**self.fraction
        # 2147483520 is the largest FP32 number below signed-int32 overflow.
        return torch.round(x*scale).clamp(-2147483648,2147483520)/scale

    def rms(self,x,weight,eps):
        return (x*(1.0/torch.sqrt((x*x).mean(-1,keepdim=True)+eps)))*weight

    def dense(self,name,norm,x):
        info=self.bundle.manifest['tensors'][name]
        rows,cols=info['shape']
        normalized=self.rms(x,self.p[norm],1e-6)
        multiplier=127.0/normalized.abs().amax(-1,keepdim=True).clamp_min(1e-5)
        scaled=normalized*multiplier
        q=torch.copysign(torch.floor(scaled.abs()+0.5),scaled).clamp(-128,127).to(torch.int8)
        padded_rows=max(32,((len(q)+15)//16)*16)
        q=torch.nn.functional.pad(q,(0,(-cols)%16,0,padded_rows-len(q)))
        dot=torch._int_mm(q.contiguous(),self.weights[name])[:len(x),:rows]
        return (dot.float()*(1.0/multiplier))*float(info['scale'])

    @staticmethod
    def sigmoid(x):
        e=torch.exp(-torch.abs(x))
        return torch.where(x>=0,1.0/(1.0+e),e/(1.0+e))

    @torch.inference_mode()
    def sequence(self,tokens,trace=None):
        ids=torch.as_tensor(tokens,dtype=torch.int64,device='cuda')
        if ids.ndim!=1 or not len(ids): raise ValueError('Expected one nonempty token sequence')
        if bool(((ids<0)|(ids>=self.cfg.vocab)).any()): raise ValueError('Invalid token')
        values={};states=[]
        for index,op in enumerate(self.bundle.program['ops']):
            kind=op['op'];dst=op['dst']
            if kind=='embedding':y=self.p[op['weight']][ids]
            elif kind=='rms':y=self.rms(values[op['src']],self.p[op['weight']],op['eps'])
            elif kind=='bitlinear':y=self.dense(op['weight'],op['norm'],values[op['src']])
            elif kind=='sigmoid':y=self.sigmoid(values[op['src']])
            elif kind=='silu':
                x=values[op['src']];y=x*self.sigmoid(x)
            elif kind=='lower_bound':
                bound=self.p['derived.lower_bounds'][op['layer']]
                y=bound+(1.0-bound)*values[op['src']]
            elif kind=='recurrent':
                i=values[op['src']].contiguous();f=values[op['gate']].contiguous()
                y=torch.empty_like(i);h=torch.empty((self.cfg.hidden,),device='cuda',dtype=torch.float32)
                _recurrent[(triton.cdiv(self.cfg.hidden,128),)](i,f,y,h,len(ids),self.cfg.hidden,128,
                    -1 if self.fraction is None else self.fraction,enable_fp_fusion=False)
                states.append(h)
            elif kind=='swiglu':
                gate,value=values[op['src']].chunk(2,-1);y=(gate*self.sigmoid(gate))*value
            elif kind=='add':y=values[op['a']]+values[op['b']]
            elif kind=='mul':y=values[op['a']]*values[op['b']]
            elif kind=='argmax':y=values[op['src']].argmax(-1)
            else:raise ValueError(kind)
            if kind!='argmax':y=self.boundary(y)
            values[dst]=y
            if trace is not None:trace(index,op,y)
        return values['logits'],torch.stack(states)
