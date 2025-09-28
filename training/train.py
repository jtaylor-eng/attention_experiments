#Training loop
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

#TODO: probably best transferred to command line args at some point
#Set to True to run training loop with ../data/tiny_shakespeare.txt
TEST_RUN = False
COMPILE_MODEL = True

if TEST_RUN:
    #no ddp for test run
    is_ddp, rank, world_size, local_rank = False, 0, 1, 0
    TRANSFORMER_CONFIG = utils.TransformerConfig(
        vocab_size=...,#will be updated by DataLoaderTxt which builds character level tokenizer
        block_size=256,
        num_layers=4,
        num_heads=4,
        dim_emb=128,
    )

    DATA_LOADER = utils.DataLoaderTxt(
        data_path= '../data/tiny_shakespeare.txt',
        block_size=TRANSFORMER_CONFIG.block_size,
        batch_size=4,
        rank=rank,
        world_size=world_size,
    )
    TRANSFORMER_CONFIG.vocab_size = DATA_LOADER.vocab_size

    TOKENIZER = DATA_LOADER.tokenizer
else:
    #setup ddp
    is_ddp, rank, world_size, local_rank = utils.setup_ddp()
    TOKENIZER = utils.LlamaTokenizerWrapper('meta-llama/Llama-2-7b-hf')

    TRANSFORMER_CONFIG = utils.TransformerConfig(
        vocab_size=TOKENIZER.get_vocab_size(),
        block_size=1024,
        num_layers=12,
        num_heads=12,
        dim_emb=768,
    )
    
    DATA_LOADER = utils.DataLoader(
        data_path='../data/c4_en_llama2_tokens',
        block_size=TRANSFORMER_CONFIG.block_size,
        batch_size=16,
        rank=rank,
        world_size=world_size,
    )


if __name__ == '__main__':
    #setup logging
    if rank == 0: logger = utils.setup_logging('log.txt')
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    logger.info('Starting training session')

    device = f'cuda:{local_rank}' if is_ddp else ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42) ; torch.cuda.manual_seed(42)
    
    if rank == 0:
        logger.info(f'Using device: {device}')
        logger.info(f'DDP enabled: {is_ddp}, World size: {world_size}')

    model = LM(config=TRANSFORMER_CONFIG).to(device)
    logger.info(f'Model has: {sum(p.numel() for p in model.parameters())} parameters')
    # Wrap model with DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if rank == 0: logger.info('Model wrapped with DDP')
    
    if rank==0 and COMPILE_MODEL:
        logger.info('Compiling model') 
        model = torch.compile(model)
        logger.info('Model compiled')

    # Training hyperparameters
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 250
    max_steps = 5000
    checkpoint_steps = 1000
    
    # Gradient accumulation settings
    micro_batch_size = 16
    grad_accum_steps = 8
    effective_batch_size = micro_batch_size * grad_accum_steps
    if rank == 0: logger.info(f'Effective batch size: {effective_batch_size} (micro_batch={micro_batch_size} x grad_accum={grad_accum_steps})')

    # Optimizer with proper parameter grouping
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
    if rank == 0: logger.info(f'Training for {max_steps} steps with checkpointing every {checkpoint_steps} steps')

    # Training loop
    torch.set_float32_matmul_precision('high')
    model.train()
    for step in range(max_steps):
        optimizer.zero_grad()
        loss_accum = 0.0
        
        # Gradient accumulation over multiple micro-batches
        for micro_step in range(grad_accum_steps):
            # Get batch
            xb, yb = DATA_LOADER.get_batch('train')
            xb, yb = xb.to(device), yb.to(device)
            
            # Forward pass with mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-1)
            
            # Scale loss by accumulation steps
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            
            # Backward pass
            loss.backward()
        
        # DDP synchronization (loss only for logging)
        if is_ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            loss_accum = loss_accum / world_size
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Learning rate scheduling
        lr = utils.get_lr(step, max_lr, min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Logging (only on rank 0)
        if step % 100 == 0 and rank == 0:
            logger.info(f'Step {step}: loss = {loss_accum.item():.4f}, lr = {lr:.6f}')
        
        # Checkpointing (only on rank 0)
        if step % checkpoint_steps == 0 and step > 0 and rank == 0:
            # Get the underlying model for checkpointing
            model_to_save = model.module if is_ddp else model
            checkpoint_path = utils.save_checkpoint(model_to_save, optimizer, step, loss_accum.item())
            logger.info(f'Checkpoint saved: {checkpoint_path}')
        
        # Validation (only on rank 0)
        if step % 100 == 0 and step > 0 and rank == 0:
            model.eval()
            with torch.no_grad():
                val_xb, val_yb = DATA_LOADER.get_batch('val')
                val_xb, val_yb = val_xb.to(device), val_yb.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    val_logits = model(val_xb)
                    val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), val_yb.view(-1), ignore_index=-1)
                logger.info(f'Validation loss: {val_loss.item():.4f}')
                
                # Generate sample text
                context = TOKENIZER.encode('Why is ').to(device=device, dtype=torch.long).unsqueeze(0)
                generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=50)
                generated_text = TOKENIZER.decode(generated[0].tolist())
                logger.info(f'Generated text: {generated_text[:200]}')
            model.train()
    
    # Final checkpoint (only on rank 0)
    if rank == 0:
        model_to_save = model.module if is_ddp else model
        final_checkpoint_path = utils.save_checkpoint(model_to_save, optimizer, max_steps, loss_accum.item())
        logger.info(f'Final checkpoint saved: {final_checkpoint_path}')
        logger.info('Training completed!')
    
    utils.cleanup_ddp()