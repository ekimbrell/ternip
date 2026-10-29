"""Explicit arithmetic contracts shared by export and the software model."""
import numpy as np


def f32(x):
    return np.asarray(x, dtype=np.float32)


def rms(x, weight, eps):
    x = f32(x)
    return f32(x * (np.float32(1) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32) + np.float32(eps))) * weight)


def sigmoid(x):
    x = f32(x)
    e = np.exp(-np.abs(x))
    return f32(np.where(x >= 0, 1 / (1 + e), e / (1 + e)))


def silu(x):
    return f32(x * sigmoid(x))


def round_away(x):
    return np.copysign(np.floor(np.abs(x) + np.float32(0.5)), x)


def quantize_activation(x, rounding="away"):
    x = f32(x)
    multiplier = f32(np.float32(127) / np.maximum(np.max(np.abs(x), axis=-1, keepdims=True), np.float32(1e-5)))
    scale = f32(np.float32(1) / multiplier)
    # Reference fused path rounds half away from zero; torch.round is ties-to-even.
    z = x * multiplier
    if rounding not in ("away", "even"):
        raise ValueError("Unknown rounding rule")
    q = np.clip(round_away(z) if rounding == "away" else np.rint(z), -128, 127).astype(np.int8)
    return q, scale


def bf16_round(x):
    """Round finite FP32 to BF16 ties-to-even and return exact FP32 values."""
    x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite BF16 input')
    bits=x.view(np.uint32)
    rounded=(bits+np.uint32(0x7fff)+((bits>>np.uint32(16))&np.uint32(1))) & np.uint32(0xffff0000)
    return np.asarray(rounded,dtype=np.uint32).view(np.float32)


def quantize_weight(weight, weight_format="fp32"):
    if weight_format=="upstream-bf16":
        # The pinned reference invokes weight_quant on BF16 parameters. Mean,
        # reciprocal, multiplication and dequantization all materialize BF16.
        weight=bf16_round(weight)
        mean=bf16_round(np.mean(np.abs(weight),dtype=np.float32))
        mean=bf16_round(np.maximum(mean,np.float32(1e-5)))
        multiplier=bf16_round(np.float32(1)/mean)
        q=np.clip(np.rint(bf16_round(weight*multiplier)),-1,1).astype(np.int8)
        scale=np.float32(bf16_round(np.float32(1)/multiplier))
        return q,scale
    if weight_format!="fp32": raise ValueError('Unknown weight quantization format')
    weight = f32(weight)
    if not np.isfinite(weight).all():
        raise ValueError("Nonfinite checkpoint tensor")
    scale = np.float32(max(float(np.mean(np.abs(weight), dtype=np.float32)), 1e-5))
    q = np.clip(np.rint(weight * (np.float32(1) / scale)), -1, 1).astype(np.int8)
    return q, scale


def pack_ternary(q):
    q = np.asarray(q)
    if not np.isin(q, [-1, 0, 1]).all():
        raise ValueError("Weights must be ternary")
    flat = q.reshape(-1)
    if flat.size % 4:
        raise ValueError("Packed ternary payload must have a multiple of four elements")
    codes = (flat.astype(np.int8).view(np.uint8) & 3).reshape(-1, 4)
    return (codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)).tobytes()


def unpack_ternary(data, shape):
    codes = np.frombuffer(data, dtype=np.uint8)
    q = ((codes[:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1)
    if np.any(q == 2):
        raise ValueError("Reserved two-bit weight code")
    return np.where(q == 3, -1, q).astype(np.int8).reshape(shape)


def bitlinear(x, weight, norm, weight_format="fp32"):
    q, scale = quantize_activation(rms(x, norm, 1e-6))
    w, ws = quantize_weight(weight,weight_format)
    return f32((q.astype(np.int32) @ w.astype(np.int32).T).astype(np.float32) * scale * ws)


def lower_bounds(raw):
    e = np.exp(f32(raw) - np.max(raw, axis=0, keepdims=True))
    p = e / np.sum(e, axis=0, keepdims=True)
    return f32(np.cumsum(p, axis=0) - p[0])
