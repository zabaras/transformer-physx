import torch
import torch.nn as nn
from .utils import Conv1D

class MaskedAttention(nn.Module):
    '''
    Attention implementation used in:
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py
    '''
    def __init__(self, nx, n_ctx, config, scale=False, mask='tril'):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
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

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
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

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False
    ):
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

class MaskedAttentionConv(nn.Module):
    '''
    Attention implementation used in:
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py
    '''
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(1, config.k_size), padding=(0,config.k_size//2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(1, config.k_size), padding=(0,config.k_size//2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, kernel_size=(1, config.k_size), padding=(0,config.k_size//2)) )
            
        self.c_proj = nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(1, config.k_size), padding=(0,config.k_size//2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(1, config.k_size), padding=(0,config.k_size//2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=(1, config.k_size), padding=(0,config.k_size//2)) )
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
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

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False
    ):
        x = self.c_attn(x.unsqueeze(1)) # x -> q, k, v
        query = x[:,0]
        key = x[:,1]
        value = x[:,2]
        
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
        a = self.c_proj(a.unsqueeze(1)).squeeze(1)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)