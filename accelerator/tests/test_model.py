import json
import numpy as np
import pytest
from ternip_accel.numerics import *
from ternip_accel.compiler import compile_bundle, Bundle
from ternip_accel.model import ModelConfig, ReferenceModel
from ternip_accel.emulator import Emulator

def test_encoding_and_reserved():
    q = np.array([-1,0,1,-1],np.int8)
    assert pack_ternary(q) == bytes([0xd3])
    np.testing.assert_array_equal(unpack_ternary(pack_ternary(q),(4,)),q)
    with pytest.raises(ValueError): unpack_ternary(bytes([2]),(4,))

def test_rounding_is_explicit():
    x = np.array([[127.,0.5,-0.5,2.5,-2.5]],np.float32)
    away,_ = quantize_activation(x,"away")
    even,_ = quantize_activation(x,"even")
    np.testing.assert_array_equal(away,[[127,1,-1,3,-3]])
    np.testing.assert_array_equal(even,[[127,0,0,2,-2]])
    q,s = quantize_activation(np.zeros((1,7),np.float32))
    assert not q.any() and np.isfinite(s).all()

def test_lower_bound_axes():
    bounds = lower_bounds(np.zeros((3,7),np.float32))
    np.testing.assert_allclose(bounds[:,0],[0,1/3,2/3],rtol=1e-6)

def test_full_model_recurrent_trace(tiny,engine):
    ref = ReferenceModel(*tiny)
    state_a = ref.initial_state(3)
    state_b = engine.initial_state(3)
    for step in range(20):
        tokens = np.array([step%13,(step+3)%13,(step+7)%13])
        expected,state_a = ref.step(tokens,state_a)
        actual,state_b = engine.step(tokens,state_b)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=1e-6)
        np.testing.assert_allclose(state_b,state_a,rtol=2e-5,atol=1e-6)
        np.testing.assert_array_equal(np.argmax(actual,axis=-1),np.argmax(expected,axis=-1))

def test_rectangular_accumulation_and_global_norm(tmp_path,tiny):
    config,tensors = tiny
    config.update(hidden_size=263,intermediate_size=271,num_hidden_layers=1,vocab_size=17)
    rng = np.random.default_rng(94)
    tensors = {n:rng.normal(size=s).astype(np.float32)*0.1 for n,s in ModelConfig.from_dict(config).tensor_shapes().items()}
    compile_bundle(config,tensors,tmp_path/"b",tile=128)
    e = Emulator(Bundle(tmp_path/"b"))
    x = rng.normal(size=(3,263)).astype(np.float32)
    name = "model.layers.0.attn.i_proj"
    actual = e.dense(name+".weight",name+".norm.weight",x)
    expected = bitlinear(x,tensors[name+".weight"],tensors[name+".norm.weight"])
    np.testing.assert_array_equal(actual,expected)
    info = e.bundle.manifest["tensors"][name+".weight"]
    assert info["padded_shape"] == [384,512]
    assert e.counters["weight_bytes"] == info["bytes"] # not multiplied by lanes

def test_corruption_rejected(tiny,tmp_path):
    compile_bundle(*tiny,tmp_path/"b")
    (tmp_path/"b"/"weights.bin").write_bytes(b"bad")
    with pytest.raises(ValueError,match="Corrupted"): Bundle(tmp_path/"b")

def test_unknown_graph_rejected(tiny):
    c,t = tiny
    c["use_short_conv"] = True
    with pytest.raises(ValueError): ModelConfig.from_dict(c)

def test_fused_epsilon_contract(engine):
    assert engine.bundle.manifest["numerics"]["bitlinear_epsilon"] == 1e-6
    # Upstream FusedBitLinear calls the helper's default epsilon; norm.eps=1e-8 is not passed.


def test_bf16_rounding_and_upstream_weight_scale():
    from ternip_accel.numerics import bf16_round,quantize_weight
    bits=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000],dtype=np.uint32)
    rounded=bf16_round(bits.view(np.float32)).view(np.uint32)
    np.testing.assert_array_equal(rounded,[0x3f800000,0x3f820000,0xbf800000,0xbf820000])
    q,scale=quantize_weight(np.array([[-1,0,1]],np.float32),'upstream-bf16')
    np.testing.assert_array_equal(q,[[-1,0,1]])
    assert scale==np.float32(0.66796875)
    with pytest.raises(ValueError):quantize_weight(np.array([1],np.float32),'unknown')


def test_bf16_export_matches_independent_model(tiny,tmp_path):
    compile_bundle(*tiny,tmp_path/'bf16',weight_format='upstream-bf16')
    e=Emulator(Bundle(tmp_path/'bf16'));r=ReferenceModel(*tiny,weight_format='upstream-bf16')
    es=e.initial_state(2);rs=r.initial_state(2)
    for token in range(4):
        expected,rs=r.step(np.array([token,token+1]),rs)
        actual,es=e.step(np.array([token,token+1]),es)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=1e-6)
        np.testing.assert_allclose(es,rs,rtol=2e-5,atol=1e-6)
    assert e.bundle.manifest['numerics']['weight_quantization']=='upstream-bf16'
