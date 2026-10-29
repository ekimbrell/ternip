"""Fast numerical emulator using shared controller FP32 and exact CUDA int dots.

Only integer matmul runs on CUDA. All reductions, activation rounding, scaling,
transcendentals and recurrent state use the C++ controller arithmetic. This
qualifies the *portable numerical contract*, not synthesized vendor float IP.
"""
import ctypes as ct
import numpy as np
import torch
from .numerics import unpack_ternary

class ControllerEmulator:
    def __init__(self,bundle,library):
        self.bundle=bundle;self.cfg=bundle.cfg;self.p=bundle.parameters
        if max(self.cfg.hidden,self.cfg.intermediate)>8192:raise ValueError('Controller vector capacity exceeded')
        self.lib=ct.CDLL(str(library));v=ct.c_void_p;u=ct.c_uint;f=ct.c_float
        for name,args in {
            'norm_many':[v,v,v,u,u,f], 'quant_many':[v,v,v,v,u,u],
            'rescale_many':[v,v,v,u,u,f], 'recurrent_many':[v,v,v,v,u,u],
            'vector_many':[u,v,v,v,v,u,u]}.items():
            fn=getattr(self.lib,'ternip_'+name);fn.argtypes=args;fn.restype=None
        self.weights={}
        for name,info in bundle.manifest['tensors'].items():
            rows,cols=info['shape'];start=info['offset']
            w=unpack_ternary(bundle.weights[start:start+info['bytes']],tuple(info['padded_shape']))[:rows,:cols]
            t=torch.from_numpy(w.copy()).cuda()
            self.weights[name]=torch.nn.functional.pad(t,(0,(-cols)%16,0,(-rows)%16)).t().contiguous()

    @staticmethod
    def ptr(array):
        if array is None:return None
        if not array.flags.c_contiguous:raise ValueError('Numerical buffer is not contiguous')
        return array.ctypes.data

    def norm(self,x,w,eps):
        y=np.empty_like(x);self.lib.ternip_norm_many(self.ptr(x),self.ptr(w),self.ptr(y),*x.shape,eps);return y

    def dense(self,name,norm,x):
        rows=len(x);n=x.shape[1];info=self.bundle.manifest['tensors'][name];outputs=info['shape'][0]
        if n>8192:raise ValueError('Controller quantization capacity exceeded')
        q=np.empty(x.shape,np.int8);scales=np.empty(rows,np.float32)
        self.lib.ternip_quant_many(self.ptr(x),self.ptr(self.p[norm]),self.ptr(q),self.ptr(scales),rows,n)
        qt=torch.from_numpy(q).cuda()
        qt=torch.nn.functional.pad(qt,(0,(-n)%16,0,max(32,((rows+15)//16)*16)-rows))
        dot=torch._int_mm(qt.contiguous(),self.weights[name])[:rows,:outputs]
        dot=np.ascontiguousarray(dot.cpu().numpy())
        y=np.empty((rows,outputs),np.float32)
        self.lib.ternip_rescale_many(self.ptr(dot),self.ptr(scales),self.ptr(y),rows,outputs,float(info['scale']))
        return y

    @torch.inference_mode()
    def sequence(self,tokens,trace=None):
        ids=np.asarray(tokens,dtype=np.int64)
        if ids.ndim!=1 or not len(ids) or np.any(ids<0) or np.any(ids>=self.cfg.vocab):raise ValueError('Invalid token sequence')
        values={};states=[]
        for index,op in enumerate(self.bundle.program['ops']):
            kind=op['op'];dst=op['dst']
            if kind=='embedding':y=self.p[op['weight']][ids].copy()
            elif kind=='rms':y=self.norm(values[op['src']],self.p[op['weight']],op['eps'])
            elif kind=='bitlinear':y=self.dense(op['weight'],op['norm'],values[op['src']])
            elif kind=='recurrent':
                x=values[op['src']];f=values[op['gate']];y=np.empty_like(x);h=np.empty(x.shape[1],np.float32)
                self.lib.ternip_recurrent_many(self.ptr(x),self.ptr(f),self.ptr(y),self.ptr(h),*x.shape);states.append(h)
            elif kind=='argmax':y=values[op['src']].argmax(-1)
            else:
                opcode={'sigmoid':4,'silu':5,'lower_bound':6,'swiglu':8,'add':9,'mul':10}[kind]
                x=values[op['a']] if kind in ('add','mul') else values[op['src']]
                b=values[op['b']] if kind in ('add','mul') else None
                p=self.p['derived.lower_bounds'][op['layer']] if kind=='lower_bound' else None
                n=x.shape[1]//2 if kind=='swiglu' else x.shape[1]
                y=np.empty((len(ids),n),np.float32)
                self.lib.ternip_vector_many(opcode,self.ptr(x),self.ptr(b),self.ptr(p),self.ptr(y),len(ids),n)
            values[dst]=y
            if trace is not None:trace(index,op,y)
        return values['logits'],np.stack(states)
