"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import logging
from typing import List
import torch
from torch import nn

from .attention import MaskedAttention
from .utils import Conv1D, ACT2FN
from .phys_transformer_base import PhysformerBase
from .generate_utils import GenerationMixin
from ..config.configuration_phys import PhysConfig

logger = logging.getLogger(__name__)

Tensor = torch.Tensor
LongTensor = torch.LongTensor

class MLP(nn.Module):
    """Simple fully connected neural network layer.
    Includes activations function and dropout.

    Args:
        n_state (int): dimensionality of input features
        config (PhysConfig): Phys-transformer config object
    """
    def __init__(self, n_state: int, config: PhysConfig) -> None:
        """Constructor 
        """
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): [B, T, n_state] input features

        Returns:
            Tensor: Output features
        """
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    """Transformer decoder block consisting of layer norm, masked self-attention,
    layer norm and fully connected layer.

    Args:
        n_ctx (int): contex length of block
        config (PhysConfig): Phys-transformer config object
        scale (bool, optional): Scaled self-attention calculation. Defaults to False.
    """
    def __init__(self, n_ctx: int, config: PhysConfig, scale: bool = False) -> None:
        """Constructor
        """
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = MaskedAttention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(
        self, 
        x: Tensor, 
        layer_past: List[Tensor] = None, 
        attention_mask: LongTensor = None, 
        head_mask: LongTensor = None, 
        use_cache: bool = False, 
        output_attentions: bool = False,
    ) -> List[Tensor]:
        """Forward pass

        Args:
            x (Tensor): [B, T, n_state] input features
            layer_past ([type], optional): Past self-attention calculation. Defaults to None.
            attention_mask (LongTensor, optional): Attention mask. Defaults to None.
            head_mask (LongTensor, optional): Attention value. Defaults to None.
            use_cache (bool, optional): Store attention state (key values). Defaults to False.
            output_attentions (bool, optional): Return attention values. Defaults to False.

        Returns:
            List[Tensor]: List of output tensors
        """
        # Evaluate attention heads
        output_attn = self.attn.forward(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        a = output_attn[0] 
        # Residual connection 1
        x = x + a
        # FCNN
        m = self.mlp(self.ln_2(x))
        # Residual connection 2
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)

class PhysformerGPT2(GenerationMixin, PhysformerBase): # Mixins come first before base to overload
    """Transformer decoder model for modeling physics

    Args:
            config (PhysConfig): Phys-transformer config object
            model_name (str, optional): Model name. Defaults to None.
    """
    def __init__(self, config: PhysConfig, model_name: str = None) -> None:
        """Constructor        
        """
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
        inputs_embeds: Tensor,
        position_ids: Tensor = None,
        prop_embeds: Tensor =None,
        past: List[List[Tensor]] = None,
        attention_mask: LongTensor = None,
        head_mask: LongTensor = None,
        use_cache: bool = True,
        output_attentions: bool = False
    ) -> List[Tensor]:
        """Forward pass

        Note: Attention masks are not properly implemented presently and will likely not work.

        Args:
            inputs_embeds (Tensor): [B, T, n_embed] Input features
            position_ids (Tensor, optional): [T, n_embed] Manually specify position ids. Defaults to None.
            prop_embeds (Tensor, optional): [B, T, n_embed] Optional property feature. Defaults to None.
            past (List[List[Tensor]], optional): Transformer past state. Defaults to None.
            attention_mask (LongTensor, optional): [B, T] Sequence attention mask. Defaults to None.
            head_mask (LongTensor, optional): Attention value mask. Defaults to None.
            use_cache (bool, optional): Return attention states (keys). Defaults to True.
            output_attentions (bool, optional): Return attention scores. Defaults to False.

        Returns:
            List[Tensor]:  Output features, attention state (if requested), 
            hidden states of all layers (if requested), attention tensor (if requested)
        """

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
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Set mask to 0 for positions we want to attend and -10000 for ones we do not
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Function embeddings proposed in original transformer paper
        # http://papers.nips.cc/paper/7181-attention-is-all-you-need
        position_embeds = torch.zeros_like(inputs_embeds)
        i = torch.arange(0, self.config.n_embd // 2, dtype=torch.float, device=inputs_embeds.device).unsqueeze(0).unsqueeze(0)
        position_embeds[:, :, ::2] = torch.sin(position_ids.unsqueeze(-1) / 10000 ** (2 * i / self.config.n_embd))
        i = i[:, :, self.config.n_embd % 2]
        position_embeds[:, :, 1::2] = torch.cos(position_ids.unsqueeze(-1) / 10000 ** (2 * i / self.config.n_embd))
        
        # Combine input embedding, position embeding and prop embeddings
        hidden_states = inputs_embeds + position_embeds + prop_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        # Loop through transformer self-attention layers
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
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.mlp_f(self.ln_f(hidden_states))

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

