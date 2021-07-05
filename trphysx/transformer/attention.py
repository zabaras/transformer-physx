"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import torch
import torch.nn as nn
from typing import List
from .utils import Conv1D
from ..config.configuration_phys import PhysConfig

Tensor = torch.Tensor

class MaskedAttention(nn.Module):
    '''
    Attention implementation used in:
    
    '''
    """Masked self-attention module based on the Hugging face implementation
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py

    Args:
        nx (int): Dimensionality of feature vector
        n_ctx (int): Context length of the attention (TODO: Not needed with config object?)
        config (PhysConfig): Transformer config object
        scale (bool, optional): Scale the attention scores. Defaults to False.
        mask (str, optional): Attention mask type. Defaults to 'tril'.

    Raises:
        ValueError: Invalid mask type
    """
    def __init__(
        self, 
        nx: int, 
        n_ctx: int, 
        config: PhysConfig, 
        scale: bool  = False, 
        mask: str = 'tril'
    ) -> None:
        """Constructor
        """
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % config.n_head == 0
        
        # Create attention mask
        if mask == 'tril': # Upper triangular mask
            self.register_buffer(
                "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
            )
        elif mask == 'block': # Block diagonal, tril mask
            tril = torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8))
            block = torch.ones((config.n_patches, config.n_patches), dtype=torch.uint8)
            block_diag = torch.block_diag(*[block for i in range(n_ctx//config.n_patches)])
            self.register_buffer(
                "bias", (tril + block_diag).clamp(0,1).view(1, 1, n_ctx, n_ctx) )
        else:
            raise ValueError('Specified mask type {} is not currently supported.'.format(mask))

        
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        # Conv1D are not PyTorch Conv1D
        # Conv1D(out_features, in_features)
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def _attn(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        attention_mask: Tensor = None,
        head_mask: Tensor = None, 
        output_attentions: bool =False
    ) -> List[Tensor]:
        """Dot product attention calculation

        Args:
            q (Tensor): [batch, head, seq_length, head_features] query
            k (Tensor): [batch, head, head_features, seq_length] key
            v (Tensor): [batch, head, seq_length, head_features] value
            attention_mask (Tensor, optional): Optional defined attention mask. Defaults to None.
            head_mask (Tensor, optional): Optional attention value mask. Defaults to None.
            output_attentions (bool, optional): Output attention matrix. Defaults to False.

        Returns:
            List[Tensor]: Output consisting of output feature, attention tensor (if requested)
        """
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)

        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x: Tensor) -> Tensor:
        """Merge attention heads

        Args:
            x (Tensor): [batch, head, seq_length, head_features] Input tensor

        Returns:
            Tensor: [batch, seq_length, head * head_features] Concatenated output tensor
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k: bool = False) ->  Tensor:
        """Splits key, query or value tensor into separate heads.
        Dimensionality of output depends if tensor is a key.

        Args:
            x (Tensor): [batch, seq_length, nx] Input tensor
            k (bool): If input tensor is a key tensor

        Returns:
            Tensor: [batch, head, seq_length, head_features] Split features for query
            and value, [batch, head, seq_length, head_features] split feature for key
        """
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self, 
        x: Tensor, 
        layer_past: List[Tensor] = None, 
        attention_mask: Tensor =None, 
        head_mask: Tensor = None,
        use_cache: bool = False, 
        output_attentions: bool = False
    ) -> List[Tensor]:
        """Masked attention forward pass

        Args:
            x (Tensor): [batch, seq_length, nx] Input feature.
            layer_past (List[Tensor], optional): Precomputed self-attention vectors. Defaults to None.
            attention_mask (Tensor, optional): Optional defined attention mask. Applied before soft mask.
                 Defaults to None.
            head_mask (Tensor, optional): Optional attention value mask. Applied after softmax Defaults to None.
            use_cache (bool, optional): Return calculated key values or faster generation. Defaults to False.
            output_attentions (bool, optional): Return attention matrix. Defaults to False.

        Returns:
            List[Tensor]: Output consisting of output feature, key values (if requested), attention tensor (if requested)
        """
        x = self.c_attn(x) # x -> q, k, v
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True) # k=True for keys which transposes the last two dims
        value = self.split_heads(value)
        # Concat previous key and value tensors 
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)