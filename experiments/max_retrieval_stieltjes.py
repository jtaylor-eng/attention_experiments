# Implementation of figure 2 in softmax is not enough paper
#  - Goal - show out of distribution sequence lengths do not have sharp attention
#  - Methodology - train single attention layer on max seq len of 16, plot highest attention weights for seqeunces lengths 16, 32, 64, ..., see if attention is sharp.
#  - The task is retrieving the max element in a random sequence (eg [5, 10, 3])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from typing import Literal, Optional, Callable, List
import time
import os

# from softmax_fns import TraditionalSoftmax, StieltjesTransform, AdaptiveSoftmax 

# ---- ALTERNATIVE SOFTMAX IMPLEMENTATIONS ---- #
"""Provide uniform interface to use alternate softmax fns"""
class CustomSoftmaxFn(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def translate_logits(
        self,
        logits: torch.Tensor,
        dim: int,
        **kwargs,
    ) -> torch.Tensor:
        pass


class TraditionalSoftmax(CustomSoftmaxFn):
    def translate_logits(self, logits, dim): return F.softmax(logits, dim=dim)


class StieltjesTransform(CustomSoftmaxFn):
    def _line_search_bs(self, num_iter, shifted_logits, eps, q, dim, lb, ub):
        for _ in range(num_iter):
            mid = (lb + ub) / 2.0
            
            prob_sum = torch.sum(
                torch.pow((mid - shifted_logits).clamp(min=eps), -q),
                dim=dim,
                keepdim=True
            ) - 1
            
            lb = torch.where(prob_sum > 0, mid, lb)
            ub = torch.where(prob_sum <= 0, mid, ub)

        return lb, ub

    def translate_logits(
        self,
        logits,
        dim,
        q: int = 1,
        num_iter: int = 32,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """Calculates 1 / (lambda_q - x_i)^q"""
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        x_max = torch.max(logits, dim=dim, keepdim=True).values
        x_i = logits - x_max

        # line search bounds, will be found from _line_search
        lb = torch.full_like(x_max, eps)
        ub = torch.full_like(x_max, logits.shape[dim] ** (1.0/q))

        lb, ub = self._line_search_bs(
            num_iter=num_iter,
            shifted_logits=x_i,
            eps=eps,
            q=q,
            dim=dim,
            lb=lb,
            ub=ub
        )
        lambda_1 = (lb + ub) / 2.0
        
        # 1 / (lambda_q - x_i)^q
        return torch.pow((lambda_1 - x_i).clamp(min=eps), -q)
        

class AdaptiveSoftmax(CustomSoftmaxFn):
    """Adapted from softmax is not enough paper (from JAX implementation)"""
    #I'd like to allow coeffs as a kwarg to translate_logits
    #but need to register_buffer for efficiency
    def __init__(self, coeffs=None):
        super().__init__()
        if coeffs is None: coeffs = [-0.037, 0.481, -2.3, 4.917, -1.791]
        self.register_buffer('poly_fit', torch.tensor(coeffs, dtype=torch.float32))
        self.register_buffer('one', torch.tensor(1.0, dtype=torch.float32))

    @staticmethod
    def _polyval_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.float32)
        for c in coeffs: out = out * x + c
        return out

    def translate_logits(self, logits: torch.Tensor, dim: int) -> torch.Tensor:
        with torch.no_grad():
            probs = F.softmax(logits, dim=dim).to(torch.float32)
            log_probs = F.log_softmax(logits, dim=dim).to(torch.float32)
            entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        
        poly_fit = self.poly_fit.to(logits.device)
        one = self.one.to(logits.device)
        
        poly_val = self._polyval_horner(poly_fit, entropy)
        greater_mask = entropy > 0.5
        
        poly_val = torch.clamp(poly_val, min=1.0, max=10.0)
        
        beta = torch.where(greater_mask, torch.maximum(poly_val, one), one)
        beta = beta.to(dtype=logits.dtype)
        
        logits_clamped = torch.clamp(logits, min=-50.0, max=50.0)
        logits_scaled = logits_clamped * beta
        
        return F.softmax(logits_scaled, dim=dim)


class SelfAttnLayer(nn.Module):
    def __init__(
        self,
        d_emb,
        softmax: Literal['traditional', 'stieltjes', 'adaptive'] = 'traditional',
    ):
        super().__init__()
        self._translation_name = softmax
        if softmax == 'traditional': self._translate_logits = TraditionalSoftmax()
        elif softmax == 'stieltjes': self._translate_logits = StieltjesTransform()
        elif softmax == 'adaptive': self._translate_logits = AdaptiveSoftmax()
        else: raise ValueError('Error: Invalid softmax option.')

        self.c_attn = nn.Linear(d_emb, 3 * d_emb)
        self.c_proj = nn.Linear(d_emb, d_emb)
        self.fc = nn.Linear(d_emb, 1)
        self.d_emb = d_emb

    def forward(self, x, return_attn=False):
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_emb, dim=2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) #(B, T, T)

        if self._translation_name == 'traditional': #scale for traditional softmax
            attn_scores *= q.size(-1) ** -0.5

        #no causal mask, global attention
        attn_weights = self._translate_logits.translate_logits(attn_scores, dim=-1) #(B, T, T)

        y = torch.matmul(attn_weights, v) #(B, T, C)
        y = self.c_proj(y)
        out = self.fc(y)

        return (out, attn_weights, attn_scores) if return_attn else out

# ---- PLOTTING ---- #
def plot_max_retrieval_attention(model, embedding, device, save_path, element_range, start_len=16, num_doubles=8, batch_size=32):
    """
    plots attention maps like Figure 2 from the paper.
    The columns are sorted by the input item's actual value in ascending order.
    (i.e. a well performing model will have high attention to rightmost value, the max, and low attention to others)
    """
    model.eval()
    fig, axes = plt.subplots(1, num_doubles, figsize=(15, 5), sharey=True)
    fig.subplots_adjust(wspace=0.1)

    current_len = start_len
    for i in range(num_doubles):
        ax = axes[i]
        xs = torch.randint(element_range[0], element_range[1], (batch_size, current_len)).to(device)

        with torch.no_grad():
            emb = embedding(xs)
            _, attn_weights, _ = model(emb, return_attn=True)

        final_token_weights = attn_weights[:, -1, :] #(batch_size, current_len)
        sorted_value_indices = torch.argsort(xs, dim=1)
        top_16_value_indices = sorted_value_indices[:, -16:]
        top_k_weights = torch.gather(final_token_weights, 1, top_16_value_indices)
        ax.imshow(top_k_weights.cpu().numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f"{current_len}", fontsize=8) ; ax.set_xticks([])
        current_len *= 2

    plt.savefig(f'{save_path}.png') ; plt.close()

def plot_learning_curve(
    train_losses: List[float],
    val_ID_losses: List[float],
    val_OOD_losses:List[float],
    len_train_dataset: int,
    len_val_dataset: int,
    train_time: float,
    name: str,
    save_path: str
):
    train_losses  = [l / len_train_dataset for l in train_losses]
    val_ID_losses = [l / len_val_dataset for l in val_ID_losses]
    val_OOD_losses= [l / len_val_dataset for l in val_OOD_losses]
    epochs_range  = list(range(1, len(train_losses) + 1))

    plt.plot(epochs_range, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs_range, val_ID_losses, 'g-o', label='In Distribution Validation Loss')
    plt.plot(epochs_range, val_OOD_losses, 'r-o', label='OOD Validation Loss')
    plt.title(f'Learning curves for {name} | trained for {train_time:.2f} seconds.')
    plt.legend()
    plt.xlabel('Epoch') ; plt.ylabel('Loss (normalized)')
    plt.savefig(f'{save_path}.png') ; plt.close()

# ---- DATASET SETUP ---- #
class MaxDataset(Dataset):
    def __init__(self, data):self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_max_data(batch):
    xs, ys = zip(*batch)
    return rnn_utils.pad_sequence(xs, batch_first=True, padding_value=0), torch.stack(ys)

def make_dataset(
    len_dataset: int,
    max_len_seq: int,
    data_range: tuple = (0, 64),
    batch_size: int = 1024,
    shuffle: bool = True,
    collate_max_data: Callable = collate_max_data,
):
    data = []
    for _ in tqdm(range(len_dataset), desc='creating dataset'):
        #pick random len within 1 to max_len_seq
        current_len = np.random.randint(1, max_len_seq + 1) 
        x = np.random.randint(low=data_range[0], high=data_range[1], size=current_len)
        y = max(x)
        data.append((torch.tensor(x), torch.tensor(y)))

    dataset = MaxDataset(data)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_max_data
    )

# ---- TRAINING & MAIN ---- #
def train_or_val(
    model: nn.Module,
    embedding: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    train_mode: bool = True,
    optim: Optional[Optimizer] = None,
) -> float:
    """run training or validation on model given other args."""
    total_loss = 0.0
    
    if train_mode:
        assert optim, 'need optimizer for train mode'
        model.train()
        embedding.train()
        
        for xs, ys in dataloader:
            xs = xs.to(device)
            ys = ys.to(device)
            
            optim.zero_grad()
            
            emb = embedding(xs)
            out = model(emb)
            pred = out[:, -1, 0]
            
            loss = loss_fn(pred, ys.float())
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
    
    else:
        model.eval()
        embedding.eval()
        
        with torch.no_grad(): 
            for xs, ys in dataloader:
                xs = xs.to(device)
                ys = ys.to(device)
                                
                emb = embedding(xs)
                out = model(emb)
                pred = out[:, -1, 0]
                
                loss = loss_fn(pred, ys.float())
                total_loss += loss.item()
                
    return total_loss

if __name__ == '__main__':
    results_folder = './results/'
    if not os.path.isdir(results_folder): os.mkdir(results_folder)

    #hyperparams
    d_emb = 128
    epochs = 50
    log_every_epochs = 5
    batch_size = 1024
    len_train_dataset = 102400
    len_val_dataset = 1024
    max_len_seq = 16
    ood_len_seq = 128
    element_range = (0, 256) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #dataloaders
    dataloader = make_dataset(len_train_dataset, max_len_seq, data_range=element_range)       #train dataloader
    dataloader_val_ID = make_dataset(len_val_dataset, max_len_seq, data_range=element_range)  #in distribution val dl
    dataloader_val_OOD = make_dataset(len_val_dataset, ood_len_seq, data_range=element_range) #ood val dl

    #train attn model with various logit translation methods
    for logits_translation in ['traditional', 'stieltjes', 'adaptive']:
        start = time.time()

        #model
        embedding = nn.Embedding(element_range[1] - element_range[0], d_emb).to(device)
        model = SelfAttnLayer(d_emb, softmax=logits_translation).to(device)
        optimizer = optim.AdamW(list(model.parameters()) + list(embedding.parameters()), lr=0.001)
        loss_fn = nn.MSELoss()

        #train
        train_losses, val_ID_losses, val_OOD_losses = [], [], []
        for i, epoch in enumerate(range(epochs)):
            train_loss   = train_or_val(model, embedding, dataloader, loss_fn, optim=optimizer)
            val_ID_loss  = train_or_val(model, embedding, dataloader_val_ID, loss_fn, train_mode=False)
            val_OOD_loss = train_or_val(model, embedding, dataloader_val_OOD, loss_fn, train_mode=False)

            train_losses.append(train_loss)
            val_ID_losses.append(val_ID_loss)
            val_OOD_losses.append(val_OOD_loss)

            if i % log_every_epochs == 0:
                print(f'Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss (in dist): {val_ID_loss:.4f} | Val Loss (OOD): {val_OOD_loss:.4f}')

        #plot 
        time_train = time.time() - start
        print(f'Time to train using {logits_translation}: {time_train}')
        
        plot_max_retrieval_attention(
            model=model,
            embedding=embedding,
            device=device,
            save_path=results_folder + logits_translation,
            element_range=element_range
        )

        plot_learning_curve(
            train_losses=train_losses,
            val_ID_losses=val_ID_losses,
            val_OOD_losses=val_OOD_losses,
            len_train_dataset=len_train_dataset,
            len_val_dataset=len_val_dataset,
            train_time=time_train,
            name=logits_translation,
            save_path=results_folder + f'{logits_translation}_learning_curves'
        )