#DataLoaders, learning rate schedulers, model checkpointing, DDP setup/teardown
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
    def __init__(
        self,
        data_path,
        block_size,
        batch_size,
        rank=0, 
        world_size=1
    ):
        ...


"""
NOTE: trivial dataloader for a .txt document
Used in testing with tiny_shakespeare.txt
"""
class DataLoaderTxt:
    def __init__(
        self,
        data_path,
        block_size,
        batch_size,
        train_split=0.9,
        rank=0, 
        world_size=1
    ):
        """Data loader for one text file, uses character level tokenization."""
        self.block_size = block_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        
        with open(data_path, 'r', encoding='utf-8') as f: text = f.read()
        
        #character-level tokenizer
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        
        #train, val split
        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
    
    def encode(self, s): return [self.stoi[c] for c in s]
    
    def decode(self, l): return ''.join([self.itos[i] for i in l])
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        
        #for DDP, each process gets diff samples
        if self.world_size > 1: torch.manual_seed(42 + self.rank)
        
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y


#Settings needed in training transformer architecture
@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int
    num_layers: int
    num_heads: int
    dim_emb: int
    mlp_dropout: float = 0.0
    rope_theta: float = 10000.0


def get_lr(it, max_lr, min_lr, warmup_steps):
    if it < warmup_steps: return max_lr * it / warmup_steps
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (it - warmup_steps) / (10000 - warmup_steps)))

def save_checkpoint(model, optimizer, step, loss, checkpoint_dir="checkpoints"):
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
    if dist.is_initialized(): dist.destroy_process_group()
