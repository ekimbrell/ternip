"""Portable FP32 interpretation of the pinned model graph, independent of microcode.

This is a numerical candidate, NOT a claim of bit identity to upstream BF16/Triton.
The quality evaluator must compare it against the original checkpoint execution.
"""
from dataclasses import dataclass
import numpy as np
from .numerics import f32, rms, sigmoid, silu, bitlinear, lower_bounds

@dataclass(frozen=True)
class ModelConfig:
    hidden: int
    intermediate: int
    layers: int
    vocab: int
    eps: float
    eos: int
    use_lower_bound: bool

    @classmethod
    def from_dict(cls, cfg):
        if cfg.get("model_type") != "hgrn_bit" or cfg.get("use_short_conv", False):
            raise ValueError("Only HGRNBit without short convolution is supported")
        if cfg.get("expand_ratio", 1) != 1 or cfg.get("hidden_act", "swish") not in ("swish", "silu"):
            raise ValueError("Unsupported expansion or activation")
        if cfg.get("tie_word_embeddings", False):
            raise ValueError("Tied embeddings require a separately validated exporter")
        h = int(cfg["hidden_size"])
        inner = cfg.get("intermediate_size")
        if inner is None:
            inner = ((int(h * cfg.get("hidden_ratio", 4) * 2 / 3) + 255) // 256) * 256
        result = cls(h, int(inner), int(cfg["num_hidden_layers"]), int(cfg["vocab_size"]),
                     float(cfg.get("rms_norm_eps", 1e-6)), int(cfg.get("eos_token_id", 2)),
                     bool(cfg.get("use_lower_bound", False)))
        if min(result.hidden, result.intermediate, result.layers, result.vocab) < 1 or not 0 <= result.eos < result.vocab:
            raise ValueError("Invalid model dimensions or EOS")
        return result

    def tensor_shapes(self):
        h, i = self.hidden, self.intermediate
        shapes = {"model.embeddings.weight": (self.vocab, h), "model.norm.weight": (h,),
                  "lm_head.weight": (self.vocab, h), "lm_head.norm.weight": (h,)}
        if self.use_lower_bound:
            shapes["model.lower_bounds"] = (self.layers, h)
        for l in range(self.layers):
            p = f"model.layers.{l}."
            for name in ("attn_norm", "mlp_norm", "attn.g_norm"):
                shapes[p + name + ".weight"] = (h,)
            for name, rows, cols in [(f"attn.{n}_proj", h, h) for n in ("i", "f", "g", "o")] + [("mlp.gate_proj", 2*i, h), ("mlp.down_proj", h, i)]:
                shapes[p+name+".weight"] = (rows, cols)
                shapes[p+name+".norm.weight"] = (cols,)
        return shapes

    def validate_tensors(self, tensors):
        for name, shape in self.tensor_shapes().items():
            if name not in tensors or tuple(tensors[name].shape) != shape:
                raise ValueError(f"Missing or mismatched tensor {name}: expected {shape}")
            if not np.isfinite(tensors[name]).all():
                raise ValueError(f"Nonfinite tensor {name}")
        extras = set(tensors) - set(self.tensor_shapes())
        if extras:
            raise ValueError(f"Unsupported checkpoint tensors: {sorted(extras)}")

class ReferenceModel:
    def __init__(self, cfg, tensors, weight_format="fp32"):
        self.cfg = ModelConfig.from_dict(cfg)
        self.cfg.validate_tensors(tensors)
        self.t = tensors
        self.weight_format = weight_format
        self.bounds = lower_bounds(tensors["model.lower_bounds"]) if self.cfg.use_lower_bound else np.zeros((self.cfg.layers,self.cfg.hidden),np.float32)

    def initial_state(self, batch):
        return np.zeros((batch, self.cfg.layers, self.cfg.hidden), np.float32)

    def step(self, tokens, state):
        t, c = self.t, self.cfg
        tokens = np.asarray(tokens)
        if np.any(tokens < 0) or np.any(tokens >= c.vocab):
            raise ValueError("Invalid token id")
        x = f32(t["model.embeddings.weight"][tokens]).copy()
        state = f32(state).copy()
        def linear(name, value):
            return bitlinear(value, t[name+".weight"], t[name+".norm.weight"],self.weight_format)
        for l in range(c.layers):
            p = f"model.layers.{l}."
            a = rms(x, t[p+"attn_norm.weight"], c.eps)
            i, f, g = [linear(p+f"attn.{n}_proj", a) for n in ("i","f","g")]
            f = sigmoid(f)
            if c.use_lower_bound and l > 0:
                f = self.bounds[l] + (1-self.bounds[l]) * f
            state[:,l] = f32(f * state[:,l] + silu(i) * (1-f))
            # Upstream calls g_norm(g_proj(x), recurrent_output): normalize g,
            # then multiply by swish(state). Reversing these is a different model.
            o = rms(g, t[p+"attn.g_norm.weight"], c.eps) * silu(state[:,l])
            x = f32(x + linear(p+"attn.o_proj", o))
            a = rms(x, t[p+"mlp_norm.weight"], c.eps)
            gate, value = np.split(linear(p+"mlp.gate_proj", a), 2, axis=-1)
            x = f32(x + linear(p+"mlp.down_proj", silu(gate)*value))
        return linear("lm_head", rms(x, t["model.norm.weight"], c.eps)), state
