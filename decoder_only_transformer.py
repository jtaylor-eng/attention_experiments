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

class DataLoader:
    def __init__(self, data_path, block_size, batch_size, train_split=0.9, rank=0, world_size=1):
        self.block_size = block_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        
        # Read and tokenize data
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create character-level tokenizer
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode text
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        
        # Split into train and validation
        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
    
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        
        # For DDP, each process gets different random samples
        if self.world_size > 1:
            # Use rank-specific random seed for different samples per process
            torch.manual_seed(42 + self.rank)
        
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int
    num_layers: int
    num_heads: int
    dim_emb: int
    mlp_dropout: float = 0.0
    rope_theta: float = 10000.0


class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.dim_emb // config.num_heads
        self.theta = config.rope_theta
        
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, q, k, seq_len):
        # q, k shape: (B, nh, T, hs)
        B, nh, T, hs = q.shape
        
        t = torch.arange(T, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Reshape to match input dimensions: (1, 1, T, hs)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        
        q_rot = self._apply_rotary_pos_emb(q, cos, sin)
        k_rot = self._apply_rotary_pos_emb(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotary_pos_emb(self, x, cos, sin):
        # x shape: (B, nh, T, hs)
        # cos, sin shape: (1, 1, T, hs)
        
        # Split x into two halves along the last dimension
        x1 = x[..., :x.size(-1)//2]  # First half
        x2 = x[..., x.size(-1)//2:]  # Second half
        
        # Split cos and sin similarly
        cos = cos[..., :cos.size(-1)//2]
        sin = sin[..., :sin.size(-1)//2]
        
        # Apply rotation
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated


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
        att = F.softmax(att, dim=-1)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # Output projection
        y = self.c_proj(y)
        return y


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
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


class LM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim_emb)
        self.rope = RoPE(config)

        self.dropout = config.mlp_dropout

        self.transformer = nn.ModuleList([TransformerBlock(config, self.rope) for _ in range(config.num_layers)])

        self.unembedding = nn.Linear(config.dim_emb, config.vocab_size)
        
        self.unembedding.weight = self.tok_embeddings.weight

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for block in self.transformer: x = block(x)
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


def get_lr(it, max_lr, min_lr, warmup_steps):
    if it < warmup_steps: return max_lr * it / warmup_steps

    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (it - warmup_steps) / (10000 - warmup_steps)))


def save_checkpoint(model, optimizer, step, loss, checkpoint_dir="checkpoints"):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    return checkpoint_path


def setup_logging(log_file="log.txt"):
    """Setup logging to file"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def setup_ddp():
    """Initialize DDP process group"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    return True, rank, world_size, local_rank


def cleanup_ddp():
    """Cleanup DDP process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    # Setup DDP
    is_ddp, rank, world_size, local_rank = setup_ddp()
    
    # Setup logging (only on rank 0 to avoid duplicate logs)
    if rank == 0:
        logger = setup_logging("log.txt")
        logger.info("Starting training session")
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
    
    SMALL_CFG = TransformerConfig(
        vocab_size=50257,  # Will be updated by DataLoader
        block_size=64,
        num_layers=3,
        num_heads=4,
        dim_emb=256,
    )

    compile_model = True
    device = f'cuda:{local_rank}' if is_ddp else ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    if rank == 0:
        logger.info(f"Using device: {device}")
        logger.info(f"DDP enabled: {is_ddp}, World size: {world_size}")

    # Data loading
    data_path = 'tiny_shakespeare.txt'
    batch_size = 4
    train_loader = DataLoader(data_path, SMALL_CFG.block_size, batch_size, rank=rank, world_size=world_size)
    
    # Update vocab size from actual data
    SMALL_CFG.vocab_size = train_loader.vocab_size
    if rank == 0:
        logger.info(f"Vocabulary size: {SMALL_CFG.vocab_size}")
    
    # Model
    model = LM(config=SMALL_CFG).to(device)
    
    # Wrap model with DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if rank == 0:
            logger.info('Model wrapped with DDP')
    
    if rank == 0:
        logger.info('Compiling model')
    if compile_model: 
        model = torch.compile(model)
    if rank == 0:
        logger.info('Model compiled')

    # Training hyperparameters
    max_lr = 3e-4
    min_lr = max_lr / 10
    warmup_steps = 10
    max_steps = 10000
    checkpoint_steps = 1000

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
    if rank == 0:
        logger.info(f"Training for {max_steps} steps with checkpointing every {checkpoint_steps} steps")

    # Training loop
    model.train()
    for step in range(max_steps):
        # Get batch
        xb, yb = train_loader.get_batch('train')
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward pass
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-1)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # DDP synchronization
        if is_ddp:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / world_size
        
        optimizer.step()
        
        # Learning rate scheduling
        lr = get_lr(step, max_lr, min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Logging (only on rank 0)
        if step % 100 == 0 and rank == 0:
            logger.info(f"Step {step}: loss = {loss.item():.4f}, lr = {lr:.6f}")
        
        # Checkpointing (only on rank 0)
        if step % checkpoint_steps == 0 and step > 0 and rank == 0:
            # Get the underlying model for checkpointing
            model_to_save = model.module if is_ddp else model
            checkpoint_path = save_checkpoint(model_to_save, optimizer, step, loss.item())
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Validation (only on rank 0)
        if step % 500 == 0 and step > 0 and rank == 0:
            model.eval()
            with torch.no_grad():
                val_xb, val_yb = train_loader.get_batch('val')
                val_xb, val_yb = val_xb.to(device), val_yb.to(device)
                val_logits = model(val_xb)
                val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), val_yb.view(-1), ignore_index=-1)
                logger.info(f"Validation loss: {val_loss.item():.4f}")
                
                # Generate sample text
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=100, temperature=1.0, top_k=5)
                generated_text = train_loader.decode(generated[0].tolist())
                logger.info(f"Generated text: {generated_text[:200]}")
            model.train()
    
    # Final checkpoint (only on rank 0)
    if rank == 0:
        model_to_save = model.module if is_ddp else model
        final_checkpoint_path = save_checkpoint(model_to_save, optimizer, max_steps, loss.item())
        logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
        logger.info("Training completed!")
    
    # Cleanup DDP
    cleanup_ddp()
