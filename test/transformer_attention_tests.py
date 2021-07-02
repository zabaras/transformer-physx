import sys
sys.path.append('..')
import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass
from trphysx.transformer.attention import MaskedAttention

@dataclass
class PhooConfig:
    n_ctx:int = 4
    n_head:int = 1
    attn_pdrop:float = 0.0
    resid_pdrop:float = 0.0

@dataclass
class PhooBlockConfig:
    n_patches:int = np.random.randint(1, 5)
    n_ctx:int = n_patches * np.random.randint(5, 10)
    n_head:int = 1
    attn_pdrop:float = 0.0
    resid_pdrop:float = 0.0

if __name__ == "__main__":
    # === Masked Attention Tests ===
    config = PhooConfig()
    n_state = np.random.randint(low=1, high=10)
    attention = MaskedAttention(n_state, config.n_ctx, config)

    q = torch.linspace(0, 1, n_state).unsqueeze(0).repeat(config.n_ctx, 1)
    k = torch.ones(n_state, config.n_ctx)
    v = torch.ones(config.n_ctx, config.n_ctx)

    outputs = attention._attn(q, k, v, output_attentions=True)

    w_target = nn.Softmax(dim=-1)(torch.tril(torch.ones(config.n_ctx, config.n_ctx), diagonal=0))
    target = torch.ones(config.n_ctx, config.n_ctx)
    assert (outputs[1].squeeze() == w_target).sum()
    assert (outputs[0].squeeze() == target).sum()

    # === Split/Merge Head Test ===
    config.n_head = np.random.randint(1, 8)
    n_state = config.n_head*np.random.randint(1, 10)
    n_batch = np.random.randint(1, 10)
    n_steps = np.random.randint(1, 10)
    attention = MaskedAttention(n_state, config.n_ctx, config)

    x = torch.randn(n_batch, n_steps, n_state) #(batch, sequence, dimension)
    # Test key split heads
    output = attention.split_heads(x, k=True)
    assert (output.size(1) == config.n_head)

    # Test query/value split heads
    output = attention.split_heads(x)
    assert (output.size(1) == config.n_head) # Dimension 1 is number of attention heads

    # Merge heads and check recovery
    output = attention.merge_heads(output)
    assert (output == x).sum()


    # === Blocked Masked Attention Tests ===
    config = PhooBlockConfig()
    n_state = np.random.randint(low=2, high=10)
    attention = MaskedAttention(n_state, config.n_ctx, config, mask='block')

    # Check if bias is block diag based on row sums which should increase
    # at an interval of config.n_patches
    bias = attention.bias.squeeze()
    bias_row_sum = bias.sum(dim=1) - config.n_patches
    target_row_sum = torch.arange(0, config.n_ctx, 1) - torch.arange(0, config.n_ctx, 1)%config.n_patches
    assert torch.all(torch.eq(bias_row_sum, target_row_sum))

    # config = PhooBlockConfig()
    # config.n_ctx = 4
    # config.n_patches = 2
    # n_state = np.random.randint(low=1, high=10)
    # attention = MaskedAttention(n_state, config.n_ctx, config, mask='block')

    # v = torch.ones(2,4,2)
    # v[:,0] = 0
    # v[:,1] = 1
    # v[:,2] = 2
    # v[:,3] = 3
    # q = torch.ones(2,4,2)
    # k = torch.ones(2,2,4)
    # out = attention._attn(q, k, v)
    # print(out[0].size())