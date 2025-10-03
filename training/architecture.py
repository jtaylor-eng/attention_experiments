#Language model architecture for experiments: Typical decoder only transformer using RoPE, layernorm.
import os
import math
import time
import inspect
import logging
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from typing import Literal
from utils import TransformerConfig


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.dim_emb, 4 * config.dim_emb)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.dim_emb, config.dim_emb)
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class MHA(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        self.config = config
        self.rope = rope
        self.num_heads = config.num_heads
        self.head_dim = config.dim_emb // config.num_heads
        self.c_attn = nn.Linear(config.dim_emb, 3 * config.dim_emb, bias=False)
        self.c_proj = nn.Linear(config.dim_emb, config.dim_emb, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, values for all heads in one go
        q, k, v = self.c_attn(x).split(C, dim=2)
        
        # Reshape to separate heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply RoPE to q and k
        q, k = self.rope(q, k, T)
        
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        att = self.config.softmax_implementation.translate_logits(att, dim=-1)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # Output projection
        y = self.c_proj(y)
        return y


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input): return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class TransformerBlock(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        self.ln_1 = LayerNorm(config.dim_emb, bias=False)
        self.attn = MHA(config, rope)
        self.ln_2 = LayerNorm(config.dim_emb, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.dim_emb // config.num_heads
        self.theta = config.rope_theta
        
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for embeddings
        self.register_buffer("cached_emb", None, persistent=False)
        self.cached_seq_len = 0

    def _update_cache(self, seq_len, device):
        if seq_len > self.cached_seq_len:
            self.cached_seq_len = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_emb = emb.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, T, hs)

    def forward(self, q, k, seq_len):
        B, nh, T, hs = q.shape
        self._update_cache(T, q.device)
        
        # Slice the cache to the current sequence length
        cos = self.cached_emb[:, :, :T, :].cos()
        sin = self.cached_emb[:, :, :T, :].sin()
        
        q_rot = self._apply_rotary_pos_emb(q, cos, sin)
        k_rot = self._apply_rotary_pos_emb(k, cos, sin)
        
        return q_rot, k_rot

    # _apply_rotary_pos_emb method remains the same
    def _apply_rotary_pos_emb(self, x, cos, sin):
        # ... (your existing implementation is correct)
        x1 = x[..., :x.size(-1)//2]
        x2 = x[..., x.size(-1)//2:]
        cos_half = cos[..., :cos.size(-1)//2]
        sin_half = sin[..., :sin.size(-1)//2]
        rotated = torch.cat([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half
        ], dim=-1)
        return rotated


class LM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim_emb)
        self.rope = RoPE(config)
        self.dropout = config.mlp_dropout

        self.transformer = nn.ModuleList([TransformerBlock(config, self.rope) for _ in range(config.num_layers)])
        self.ln_f = LayerNorm(config.dim_emb)

        self.unembedding = nn.Linear(config.dim_emb, config.vocab_size)
        self.unembedding.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for block in self.transformer: x = block(x)
        x = self.ln_f(x)
        return self.unembedding(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

