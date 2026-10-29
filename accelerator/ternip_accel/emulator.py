"""Executable model IR; software FP32 candidate, not a cycle/RTL emulator."""
import numpy as np
from .numerics import f32, rms, sigmoid, silu, quantize_activation, unpack_ternary

class Emulator:
    def __init__(self, bundle):
        self.bundle, self.cfg = bundle, bundle.cfg
        self.counters = {"weight_bytes":0, "dot_products":0, "tokens":0}
        self.decoded = {}

    def initial_state(self, batch):
        return np.zeros((batch,self.cfg.layers,self.cfg.hidden), np.float32)

    def dense(self, name, norm, x):
        b = self.bundle
        info = b.manifest["tensors"][name]
        rows, cols = info["shape"]
        pr, pc = info["padded_shape"]
        if name not in self.decoded:
            start = info["offset"]
            self.decoded[name] = unpack_ternary(b.weights[start:start+info["bytes"]], (pr,pc))
        w = self.decoded[name]
        # The reduction domain is the full logical vector, excluding padded values.
        q, scale = quantize_activation(rms(x, b.parameters[norm], 1e-6))
        result = np.zeros((len(x),rows),np.float32)
        tile = b.manifest["tile"]
        for r in range(0,rows,tile):
            end = min(r+tile,rows)
            acc = np.zeros((len(x),end-r),np.int32)
            for c in range(0,cols,tile):
                ce = min(c+tile,cols)
                acc += q[:,c:ce].astype(np.int32) @ w[r:end,c:ce].astype(np.int32).T
            result[:,r:end] = acc.astype(np.float32)*scale*np.float32(info["scale"])
        # Logical broadcast accounting: one tile fetch services the active batch.
        self.counters["weight_bytes"] += info["bytes"]
        self.counters["dot_products"] += len(x)*rows*cols
        return result

    def step(self, tokens, state, trace=None):
        tokens = np.asarray(tokens)
        if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer) or np.any(tokens<0) or np.any(tokens>=self.cfg.vocab):
            raise ValueError("Invalid token IDs")
        if np.shape(state) != (len(tokens), self.cfg.layers,self.cfg.hidden):
            raise ValueError("State shape does not match lanes")
        state = f32(state).copy()
        v, p = {}, self.bundle.parameters
        for index, op in enumerate(self.bundle.program["ops"]):
            kind, dst = op["op"], op["dst"]
            if kind == "embedding": y = p[op["weight"]][tokens].copy()
            elif kind == "rms": y = rms(v[op["src"]],p[op["weight"]],op["eps"])
            elif kind == "bitlinear": y = self.dense(op["weight"],op["norm"],v[op["src"]])
            elif kind == "sigmoid": y = sigmoid(v[op["src"]])
            elif kind == "silu": y = silu(v[op["src"]])
            elif kind == "lower_bound":
                bound = p["derived.lower_bounds"][op["layer"]]
                y = bound + (1-bound)*v[op["src"]]
            elif kind == "recurrent":
                f = v[op["gate"]]
                l = op["layer"]
                state[:,l] = f32(f*state[:,l]+(1-f)*v[op["src"]])
                y = state[:,l].copy()
            elif kind == "swiglu":
                gate, value = np.split(v[op["src"]],2,axis=-1)
                y = silu(gate)*value
            elif kind == "add": y = v[op["a"]]+v[op["b"]]
            elif kind == "mul": y = v[op["a"]]*v[op["b"]]
            elif kind == "argmax": y = np.argmax(v[op["src"]],axis=-1)
            else: raise ValueError(f"Unsupported operation: {kind}")
            v[dst] = y
            if trace is not None: trace(index,op,y.copy(),state.copy())
        self.counters["tokens"] += len(tokens)
        return v["logits"], state
