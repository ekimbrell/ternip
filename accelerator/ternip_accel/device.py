"""Lower the model IR to the portable/HLS device engine ABI (version 1)."""
import json
import struct
from pathlib import Path
import numpy as np
from .compiler import align, canonical, digest

OPS = dict(embedding=1,rms=2,bitlinear=3,sigmoid=4,silu=5,lower_bound=6,
           recurrent=7,swiglu=8,add=9,mul=10,argmax=11)

def fbits(x): return struct.unpack("<I",struct.pack("<f",x))[0]

def lower(bundle, destination):
    root = Path(destination)
    root.mkdir(parents=True,exist_ok=False)
    c,b = bundle.cfg,bundle
    sizes = dict(x=c.hidden,a=max(c.hidden,c.intermediate),i=c.hidden,f=c.hidden,g=c.hidden,
                 h=c.hidden,o=c.hidden,uv=2*c.intermediate,logits=c.vocab,token=1)
    offsets, end = {}, 0
    for name,size in sizes.items():
        offsets[name] = end
        end += align(size,16)
    scratch_stride = end
    params, poff = [], {}
    end = 0
    for name,value in sorted(b.parameters.items()):
        poff[name] = end
        params.append(value.reshape(-1))
        end += value.size
    program = []
    for op in b.program["ops"]:
        # [opcode,dst,src,a,b,length,rows,cols,weights_byte_offset,param,norm,layer,scale,eps,padded_cols,reserved]
        r = [0]*16
        r[0],r[1] = OPS[op["op"]],offsets[op["dst"]]
        for field,index in (("src",2),("a",3),("b",4)):
            if field in op: r[index] = offsets[op[field]]
        kind = op["op"]
        r[5] = c.hidden
        if kind == "embedding": r[9] = poff[op["weight"]]
        elif kind == "rms": r[9],r[13] = poff[op["weight"]],fbits(op["eps"])
        elif kind == "bitlinear":
            w = b.manifest["tensors"][op["weight"]]
            r[6],r[7] = w["shape"]
            r[8],r[10],r[12],r[14] = w["offset"],poff[op["norm"]],fbits(w["scale"]),w["padded_shape"][1]
        elif kind == "lower_bound": r[9] = poff["derived.lower_bounds"]+op["layer"]*c.hidden
        elif kind == "recurrent": r[3],r[11] = offsets[op["gate"]],op["layer"]
        elif kind == "swiglu": r[5] = c.intermediate
        elif kind == "argmax": r[5] = c.vocab
        program.extend(r)
    np.asarray(program,dtype="<u4").tofile(root/"program.bin")
    np.concatenate(params).astype("<f4").tofile(root/"parameters.bin")
    (root/"weights.bin").write_bytes(b.weights)
    layout = dict(version=1,model_hash=b.manifest["build_hash"],hidden=c.hidden,layers=c.layers,
        vocab=c.vocab,ops=len(program)//16,scratch_stride=scratch_stride,
        state_stride=c.layers*c.hidden,scratch_offsets=offsets,
        request_words=16,request_fields=["status","prompt_offset","prompt_length","prompt_position",
        "generated","limit","eos","cancel","next_token","output_offset","output_capacity","output_count",
        "reset_state","error","reserved0","reserved1"],
        status=dict(free=0,active=1,complete=2,cancelled=3,error=4),
        numerics=b.manifest["numerics"],qualification="cpu-validation-required")
    layout["files"] = {n:dict(bytes=(root/n).stat().st_size,sha256=digest((root/n).read_bytes()))
                       for n in ("program.bin","parameters.bin","weights.bin")}
    layout['build_hash']=digest(canonical(layout))
    (root/"layout.json").write_bytes(canonical(layout))
    return layout
