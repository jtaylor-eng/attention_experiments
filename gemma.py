#Reference Implementation of Gemma Transformer
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Literal

# -- Implementation from Gemma team --
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_dim]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same result
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


#Modified to use custom softmax alternatives
class GemmaAttention(nn.Module):
    def __init__(
            self,
            hidden_size=2048,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=256,
            attention_bias=False,
            softmax_fn: Literal['traditional', 'adaptive_temperature', 'custom_fn'] = 'custom_fn' #'traditional'
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=attention_bias)

        self.softmax_fn = softmax_fn

    def forward(self, hidden_states: torch.Tensor, rotary_emb: GemmaRotaryEmbedding, seq_len: int, attention_mask: torch.Tensor = None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Grouped query attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
             attn_weights = attn_weights + attention_mask

        if self.softmax_fn == 'traditional':
            attn_weights = traditional_softmax(attn_weights)
        elif self.softmax_fn == 'adaptive_temperature':
            attn_weights = adaptive_temperature_softmax(attn_weights)
        elif self.softmax_fn == 'custom_fn':
            attn_weights = custom_fn(attn_weights)
        else:
            raise ValueError(f"Unknown softmax function: {self.softmax_fn}")

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

# -- Modified from above skeleton -- 
class GemmaMLP(nn.Module):
    def __init__(self, hidden_size=2048, intermediate_size=16384, hidden_act="gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU() # Using GELU as per the original skeleton, though Gemma config might specify otherwise

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class GemmaDecoderLayer(nn.Module):
    def __init__(self, hidden_size=2048, attention_bias=False, rms_norm_eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = GemmaAttention(hidden_size=hidden_size, attention_bias=attention_bias)
        self.mlp = GemmaMLP(hidden_size=hidden_size)
        self.input_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x, rotary_emb: GemmaRotaryEmbedding, seq_len: int, attention_mask: torch.Tensor = None):
        residual = x
        x = self.input_layernorm(x)
        # Self Attention
        attn_output, attn_weights = self.self_attn(
            hidden_states=x,
            rotary_emb=rotary_emb,
            seq_len=seq_len,
            attention_mask=attention_mask,
        )
        x = residual + attn_output

        residual = x
        x = self.post_attention_layernorm(x)
        # MLP
        x = self.mlp(x)
        x = residual + x

        return x

class GemmaModel(nn.Module):
    def __init__(self, vocab_size=256000, hidden_size=2048, num_hidden_layers=18, rms_norm_eps=1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary_emb = GemmaRotaryEmbedding(dim=hidden_size // 8) # Assuming head_dim is hidden_size / num_heads (2048/8=256), and RoPE dim is half of head_dim (256/2=128)
        self.layers = nn.ModuleList([GemmaDecoderLayer(hidden_size=hidden_size, rms_norm_eps=rms_norm_eps) for _ in range(num_hidden_layers)])
        self.norm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        seq_len = input_ids.size(-1)

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if attention_mask is None:
            attention_mask = torch.full((seq_len, seq_len), torch.finfo(hidden_states.dtype).min, device=hidden_states.device)
            attention_mask = torch.triu(attention_mask, 1)

        for layer in self.layers:
            hidden_states = layer(
                x=hidden_states,
                rotary_emb=self.rotary_emb,
                seq_len=seq_len,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

class GemmaForCausalLM(nn.Module):
    def __init__(self, vocab_size=256000, hidden_size=2048, num_hidden_layers=18):
        super().__init__()
        self.gemma = GemmaModel(vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        hidden_states = self.gemma(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits

    def _init_weights(self, module):pass
    def checkpoint(self, path): pass
    @classmethod
    def from_pretrained(cls, path): raise NotImplementedError('no implementation for from_pretrained')