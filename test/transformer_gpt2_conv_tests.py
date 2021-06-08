import sys
sys.path.append('..')
import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass
from trphysx.transformer import PhysformerConvGPT2

@dataclass
class PhooConfig:
    n_ctx:int = 16
    n_positions:int = 16
    n_embd:int = 16
    n_layer:int = 2
    n_head:int = 2
    k_size:int = 9
    activation_function:str = "gelu_new"
    resid_pdrop:float = 0.0
    embd_pdrop:float = 0.0
    attn_pdrop:float = 0.0
    layer_norm_epsilon:float = 1e-5
    initializer_range:float = 0.1
    output_hidden_states:bool = False
    output_attentions:bool = True
    use_cache:bool = False
    model_type:str = "Phoo"

if __name__ == "__main__":
    # === GPT2 Tests ===
    config = PhooConfig()
    model = PhysformerConvGPT2(config)

    # === Forward test ===
    batch_size = np.random.randint(1, 10)
    n_steps = np.random.randint(1, config.n_ctx)
    x = torch.randn(batch_size, n_steps, config.n_embd) # Batch, time-steps, embed
    output = model(x, use_cache=False)

    # Test output tensor size is correct
    assert output[0].size() == torch.Size((batch_size, n_steps, config.n_ctx))