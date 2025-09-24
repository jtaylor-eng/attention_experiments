import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Literal


class DataLoader:
    def __init__(self):
        ...


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int
    num_layers: int
    num_heads: int
    dim_emb: int
    pos_embeddings: Literal['RoPE', 'Sinusoidal', 'Learnable'] = 'RoPE'
    mlp_dropout: float = 0.0


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
    def __init__(self, config):
        super().__init__()
        ...

    def forward(self):
        ...


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input): return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=False)
        self.attn = MHA(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim_emb)
        self.pos_embeddings = ...

        self.dropout = config.mlp_dropout

        self.transformer = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        self.unembedding = nn.Linear(config.dim_emb, config.vocab_size)
        
        self.unembedding.weight = self.tok_embeddings.weight

    def forward(self, idx):
        x = self.tok_embeddings(idx) + self.pos_embeddings(torch.arange(idx.size(1), device=idx.device))
        for block in self.transformer:
            x = block(x)
        # unembedding directly reuses tok_embeddings.weight
        logits = self.unembedding(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self,): ...


def get_lr(it, max_lr, min_lr, warmup): raise NotImplementedError()

if __name__ == '__main__':
    GPT_124M_CFG = TransformerConfig(
        vocab_size=50257,
        block_size=1024,
        num_layers=12,
        num_heads=12,
        dim_emb=768,
    )

    compile_model = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42) ; torch.cuda.manual_seed(42)

    encoder = ...

    total_batch_size = ...
    micro_batch_size = ...
    sequence_length  = ...

    train_loader = ...
    val_loader = ...

    model = LM(config=GPT_124M_CFG).to(device)
    if compile_model: model = torch.compile(model)

    max_lr = ...
    min_lr = ...
    warmup_steps = ...
    max_steps = 50

    optimizer = ...

    logger = ...

    for step in range(max_steps):
        #train
        ...
