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

import utils
from architecture import LM


if __name__ == '__main__':
    is_ddp, rank, world_size, local_rank = utils.setup_ddp()
    
    # Setup logging (only on rank 0 to avoid duplicate logs)
    if rank == 0:
        logger = utils.setup_logging("log.txt")
        logger.info("Starting training session")
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
    
    SMALL_CFG = utils.TransformerConfig(
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
    train_loader = utils.DataLoader(data_path, SMALL_CFG.block_size, batch_size, rank=rank, world_size=world_size)
    
    # Update vocab size from actual data
    SMALL_CFG.vocab_size = train_loader.vocab_size
    if rank == 0: logger.info(f"Vocabulary size: {SMALL_CFG.vocab_size}")
    
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
        lr = utils.get_lr(step, max_lr, min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Logging (only on rank 0)
        if step % 100 == 0 and rank == 0:
            logger.info(f"Step {step}: loss = {loss.item():.4f}, lr = {lr:.6f}")
        
        # Checkpointing (only on rank 0)
        if step % checkpoint_steps == 0 and step > 0 and rank == 0:
            # Get the underlying model for checkpointing
            model_to_save = model.module if is_ddp else model
            checkpoint_path = utils.save_checkpoint(model_to_save, optimizer, step, loss.item())
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
        final_checkpoint_path = utils.save_checkpoint(model_to_save, optimizer, max_steps, loss.item())
        logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
        logger.info("Training completed!")
    
    utils.cleanup_ddp()