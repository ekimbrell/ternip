import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from ternip_accel.model import ModelConfig
from ternip_accel.compiler import compile_bundle, Bundle
from ternip_accel.emulator import Emulator

@pytest.fixture
def tiny():
    config = dict(model_type="hgrn_bit",hidden_size=7,intermediate_size=11,
                  num_hidden_layers=2,vocab_size=13,eos_token_id=2,use_lower_bound=True,
                  rms_norm_eps=1e-6,hidden_act="swish")
    rng = np.random.default_rng(381)
    tensors = {}
    for name,shape in ModelConfig.from_dict(config).tensor_shapes().items():
        tensors[name] = (rng.normal(0,0.3,shape) if len(shape)>1 else rng.uniform(0.7,1.3,shape)).astype(np.float32)
    return config,tensors

@pytest.fixture
def engine(tiny,tmp_path):
    config,tensors = tiny
    compile_bundle(config,tensors,tmp_path/"bundle",tile=128)
    return Emulator(Bundle(tmp_path/"bundle"))
