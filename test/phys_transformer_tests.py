import sys
from dataclasses import dataclass
sys.path.append('../src/trphysx/transformer/') # Avoid loading the entire package
from attention import Attention
import torch
import torch.nn as nn

@dataclass
class PhooConfig:
    n_ctx:int = 4
    n_head:int = 1
    attn_pdrop:float = 0.0
    resid_pdrop:float = 0.0

# === Masked Attention Tests ===
config = PhooConfig()
n_state = 5
attention = Attention(n_state, config.n_ctx, config)

q = torch.linspace(0, 1, n_state).unsqueeze(0).repeat(config.n_ctx, 1)
k = torch.ones(n_state, config.n_ctx)
v = torch.ones(config.n_ctx, config.n_ctx)

outputs = attention._attn(q, k, v, output_attentions=True)

w_target = nn.Softmax(dim=-1)(torch.tril(torch.ones(config.n_ctx, config.n_ctx), diagonal=0))
target = torch.ones(config.n_ctx, config.n_ctx)
assert (outputs[1].squeeze() == w_target).sum()
assert (outputs[0].squeeze() == target).sum()