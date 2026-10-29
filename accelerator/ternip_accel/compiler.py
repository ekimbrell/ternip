"""Versioned model IR. This is not the legacy core's binary instruction set."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .model import ModelConfig
from .numerics import quantize_weight, pack_ternary, lower_bounds

VERSION = 1

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def align(n, alignment=4096):
    return (n + alignment - 1) // alignment * alignment

def compile_bundle(config, tensors, destination, tile=256, provenance=None, weight_format="fp32"):
    cfg = ModelConfig.from_dict(config)
    cfg.validate_tensors(tensors)
    if tile not in (128, 256):
        raise ValueError("Tile must be 128 or 256")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    arrays, records, relocations = {}, {}, []
    weights = bytearray()
    for name, shape in cfg.tensor_shapes().items():
        x = np.asarray(tensors[name], dtype=np.float32)
        if len(shape) == 2 and name.endswith(".weight") and name != "model.embeddings.weight":
            rows, cols = shape
            if cols * 128 > np.iinfo(np.int32).max:
                raise ValueError("Dot product exceeds int32 accumulator range")
            q, scale = quantize_weight(x,weight_format)
            pr, pc = align(rows, tile), align(cols, 256)
            padded = np.zeros((pr, pc), dtype=np.int8)
            padded[:rows, :cols] = q
            # Row-major padded matrices. Accumulate all column tiles BEFORE scaling.
            payload = pack_ternary(padded)
            offset = align(len(weights))
            weights.extend(bytes(offset-len(weights)))
            weights.extend(payload)
            records[name] = {"shape": list(shape), "padded_shape": [pr,pc], "offset": offset,
                             "bytes": len(payload), "scale": float(scale)}
            relocations.append({"tensor": name, "allocation": "weights", "offset": offset})
        else:
            arrays[name] = x
    arrays["derived.lower_bounds"] = lower_bounds(tensors["model.lower_bounds"]) if cfg.use_lower_bound else np.zeros((cfg.layers,cfg.hidden),np.float32)
    ops = []
    def emit(op, dst, **args):
        ops.append(dict(op=op, dst=dst, **args))
    def norm(dst, src, tensor, eps):
        emit("rms", dst, src=src, weight=tensor, eps=eps)
    def dense(dst, src, name):
        emit("bitlinear", dst, src=src, weight=name+".weight", norm=name+".norm.weight")
    emit("embedding", "x", weight="model.embeddings.weight")
    for l in range(cfg.layers):
        p = f"model.layers.{l}."
        norm("a", "x", p+"attn_norm.weight", cfg.eps)
        for n in ("i", "f", "g"):
            dense(n, "a", p+f"attn.{n}_proj")
        emit("sigmoid", "f", src="f")
        if cfg.use_lower_bound and l > 0:
            emit("lower_bound", "f", src="f", layer=l)
        emit("silu", "i", src="i")
        emit("recurrent", "h", gate="f", src="i", layer=l)
        norm("g", "g", p+"attn.g_norm.weight", cfg.eps)
        emit("silu", "h", src="h")
        emit("mul", "o", a="g", b="h")
        dense("o", "o", p+"attn.o_proj")
        emit("add", "x", a="x", b="o")
        norm("a", "x", p+"mlp_norm.weight", cfg.eps)
        dense("uv", "a", p+"mlp.gate_proj")
        emit("swiglu", "a", src="uv")
        dense("a", "a", p+"mlp.down_proj")
        emit("add", "x", a="x", b="a")
    norm("x", "x", "model.norm.weight", cfg.eps)
    dense("logits", "x", "lm_head")
    emit("argmax", "token", src="logits", tie_break="lowest_id")
    (root/"weights.bin").write_bytes(weights)
    np.savez(root/"parameters.npz", **arrays)
    program = {"version": VERSION, "ops": ops, "relocations": relocations}
    (root/"program.json").write_bytes(canonical(program))
    files = {n: {"sha256":digest((root/n).read_bytes()), "bytes":(root/n).stat().st_size}
             for n in ("weights.bin", "parameters.npz", "program.json")}
    # Explicit conservative storage for named scratch vectors, not an optimized allocator.
    scratch = (9*cfg.hidden + 3*cfg.intermediate + cfg.vocab)*4
    manifest = {"version": VERSION, "config": config, "tile":tile, "tensors":records,
        "files":files, "provenance":provenance or {}, "status":"software-candidate",
        "numerics":{"vector":"ieee754-float32", "activation":"int8", "dot":"int32",
                    "activation_rounding":"half-away-from-zero", "weight_rounding":"ties-even", "weight_quantization":weight_format,
                    "weight_encoding":"2bit-lsb-first:0=0,1=+1,3=-1,2=invalid",
                    "bitlinear_epsilon":1e-6, "upstream_bf16_equivalent":False},
        "allocation":{"weights_bytes":align(len(weights)),
          "parameters_bytes":align(sum(x.nbytes for x in arrays.values())),
          "program_bytes":align(len(canonical(program))),
          "scratch_bytes_per_lane":align(scratch),
          "state_bytes_per_lane":align(cfg.layers*cfg.hidden*4)},
        "quality":{"passed":False, "reason":"Pinned-checkpoint perplexity not evaluated"}}
    manifest["build_hash"] = digest(canonical(manifest))
    (root/"manifest.json").write_bytes(canonical(manifest))
    return manifest

class Bundle:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        m = self.manifest.copy()
        expected = m.pop("build_hash")
        if m.get("version") != VERSION or digest(canonical(m)) != expected:
            raise ValueError("Unsupported or corrupted bundle manifest")
        for name, info in m["files"].items():
            if name not in ("weights.bin", "parameters.npz", "program.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                raise ValueError("Unsupported bundle file")
            data = (self.root/name).read_bytes()
            if len(data) != info["bytes"] or digest(data) != info["sha256"]:
                raise ValueError(f"Corrupted bundle file: {name}")
        self.cfg = ModelConfig.from_dict(m["config"])
        self.weights = (self.root/"weights.bin").read_bytes()
        with np.load(self.root/"parameters.npz", allow_pickle=False) as z:
            self.parameters = {name:z[name] for name in z.files}
        self.program = json.loads((self.root/"program.json").read_text())
        if self.program["version"] != VERSION:
            raise ValueError("Unsupported program version")
