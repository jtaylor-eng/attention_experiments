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
import numpy as np
from abc import ABC, abstractmethod
from transformers import LlamaTokenizer


# --- TOKENIZATION ---
#provides uniform handling for various tokenizers
class CustomTokenizer(ABC):
    @abstractmethod
    def encode(self, inp: str) -> torch.Tensor: pass

    @abstractmethod
    def encode(self, inp: torch.Tensor) -> str: pass

class LlamaTokenizerWrapper(CustomTokenizer):
    def __init__(self, path='meta-llama/Llama-2-7b-hf'):
        self.tokenizer = LlamaTokenizer.from_pretrained(path)

    def encode(self, inp): return torch.tensor(self.tokenizer.encode(inp), dtype=torch.long)
    def decode(self, inp): return self.tokenizer.decode(inp)

class CharacterLevelTokenizerWrapper(CustomTokenizer):
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    def encode(self, inp): return torch.tensor([self.stoi[c] for c in inp], dtype=torch.long)
    def decode(self, inp): return ''.join([self.itos[int(i)] for i in inp])


# --- DATALOADERS ---
class DataLoader:
    def __init__(
        self,
        data_path: str,
        block_size: int,
        batch_size: int,
        rank: int, 
        world_size: int,
        split: Literal['train', 'val'],
    ):
        self.block_size = block_size
        self.batch_size = batch_size
        self.data_path  = data_path

        shards = sorted(
            os.path.join(self.data_path, s)
            for s in os.listdir(self.data_path)
            if split in s
        )
        assert len(shards) > 0, f'no shards found for split {split}'
        print(f'found {len(shards)} shards for split {split}')

        self.shards = shards
        self.reset()

    def _load_tokens(filename):
        npt = np.load(filename, allow_pickle=True)
        npt = npt.astype(np.int32)
        return torch.tensor(npt, dtype=torch.long)

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def get_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y


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
        world_size=1,
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

        self._tokenizer = CharacterLevelTokenizerWrapper(stoi=self.stoi, itos=self.itos)

    @property
    def tokenizer(self): return self._tokenizer

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        
        #for DDP, each process gets diff samples
        if self.world_size > 1: torch.manual_seed(42 + self.rank)
        
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y


# -- TRANSFORMER PARAMS --
@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int
    num_layers: int
    num_heads: int
    dim_emb: int
    mlp_dropout: float = 0.0
    rope_theta: float = 10000.0


# -- LEARNING RATE -- 
def get_lr(it, max_lr, min_lr, warmup_steps):
    if it < warmup_steps: return max_lr * it / warmup_steps
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (it - warmup_steps) / (10000 - warmup_steps)))

# -- CHECKPOINTING --
def save_checkpoint(model, optimizer, step, loss, checkpoint_dir='../checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    return checkpoint_path

# -- LOGGING -- 
def setup_logging(log_file='log.txt'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# -- DDP --
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
