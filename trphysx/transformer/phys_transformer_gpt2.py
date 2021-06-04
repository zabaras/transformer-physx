import logging
import math
import os, sys
sys.path.append("../")
from typing import Optional
from abc import abstractmethod
import torch
from torch import nn
from torch.nn import MSELoss

from .attention import MaskedAttention
from .utils import Conv1D, ACT2FN
from .phys_transformer_base import PhysformerBase
from .generate_utils import GenerationMixin

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    '''
    Word specific FCNN implementation from:
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py
    '''
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = MaskedAttention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(
        self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False,
    ):
        # Evaluate attention heads
        output_attn = self.attn.forward(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)
        # Residual connection 1
        x = x + a
        # FCNN
        m = self.mlp(self.ln_2(x))
        # Residual connection 2
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)

class PhysformerGPT2(GenerationMixin, PhysformerBase): # Mixins come first before base to overload

    def __init__(self, config, model_name:Optional[str] = None):
        PhysformerBase.__init__(self, config)
        self.output_hidden_states = config.output_hidden_states

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp_f = nn.Linear(config.n_embd, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        self.apply(self._init_weights)

        self.n_embd = config.n_embd
        # If custom transformer model name is provided, update it.
        if not model_name is None:
            self.model_name = "transformer_"+model_name

        logger.info('Number of parameters: {}'.format(self._num_parameters()))

    def forward(
        self,
        inputs_embeds=None,
        past=None,
        attention_mask=None,
        position_ids=None,
        prop_embeds=None,
        head_mask=None,
        use_cache=True,
        output_attentions=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # Input embeddings
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if prop_embeds is not None:
            assert inputs_embeds.size(0) == prop_embeds.size(0), 'Property embeddings do not match the size of the input'
            prop_embeds = prop_embeds[:,:inputs_embeds.size(1)]
        else:
            prop_embeds = torch.zeros_like(inputs_embeds)

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.float, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1]).repeat(inputs_embeds.size(0),1)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Function embeddings proposed in original transformer paper
        # http://papers.nips.cc/paper/7181-attention-is-all-you-need
        position_embeds = torch.zeros_like(inputs_embeds)
        i = torch.arange(0, self.config.n_embd // 2, dtype=torch.float, device=inputs_embeds.device).unsqueeze(0).unsqueeze(0)
        position_embeds[:, :, ::2] = torch.sin(position_ids.unsqueeze(-1) / 10000 ** (2 * i / self.config.n_embd))
        i = i[:, :, self.config.n_embd % 2]
        position_embeds[:, :, 1::2] = torch.cos(position_ids.unsqueeze(-1) / 10000 ** (2 * i / self.config.n_embd))
        
        hidden_states = inputs_embeds + position_embeds + prop_embeds
        # hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                # head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.mlp_f(self.ln_f(hidden_states))
        # hidden_states = self.mlp_f(self.ln_f(hidden_states).view(-1, self.n_embd // 64, 64))

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
            
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)

