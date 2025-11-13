# Implementation of figure 2 in softmax is not enough paper
# - Goal - show out of distribution sequence lengths do not have sharp attention
# - Methodology - train single attention layer on max seq len of 16, plot highest attention weights for seqeunces lengths 16, 32, 64, ..., see if attention is sharp.
# - The task is retrieving the class of the max element in a random sequence [cite: 536]

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


class CustomSoftmaxFn(nn.Module, ABC):
    """Provide uniform interface to use alternate softmax fns"""
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
    """Just torch.nn.functional.softmax"""
    def translate_logits(self, logits, dim, **kwargs): return F.softmax(logits, dim=dim)


class StieltjesTransform(CustomSoftmaxFn):
    """Stieltjes transform as introduced, using binary search"""
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
        q: float = 1.0,
        num_iter: int = 32,
        eps: float = 1e-9,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates 1 / (lambda_q - x_i)^q"""
        
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        x_max = torch.max(logits, dim=dim, keepdim=True).values
        x_i = logits - x_max

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
    def __init__(self, coeffs=None): #init included for registering
        super().__init__()
        if coeffs is None: coeffs = [-0.037, 0.481, -2.3, 4.917, -1.791]
        self.register_buffer("poly_fit", torch.tensor(coeffs, dtype=torch.float32))
        self.register_buffer("one", torch.tensor(1.0, dtype=torch.float32))

    @staticmethod
    def _polyval_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.float32)
        for c in coeffs: out = out * x + c
        return out

    def translate_logits(self, logits: torch.Tensor, dim: int, **kwargs) -> torch.Tensor:
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


class AdaptiveStieltjes(CustomSoftmaxFn):
    """
    This class combines the adaptive mechanism of AdaptiveSoftmax
    with the Stieltjes transform function. 'q' is no longer fixed,
    but is calculated based on entropy, just like 'beta' in the paper.
    """
    def __init__(self, coeffs=None, num_iter: int = 32, eps: float = 1e-9):
        super().__init__()
        # Use the same polynomial fit from AdaptiveSoftmax [cite: 220]
        if coeffs is None: coeffs = [-0.037, 0.481, -2.3, 4.917, -1.791]
        self.register_buffer("poly_fit", torch.tensor(coeffs, dtype=torch.float32))
        self.register_buffer("one", torch.tensor(1.0, dtype=torch.float32))
        self.num_iter = num_iter
        self.eps = eps

    @staticmethod
    def _polyval_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Helper for polynomial calculation."""
        out = torch.zeros_like(x, dtype=torch.float32)
        for c in coeffs: out = out * x + c
        return out
    
    def _line_search_bs(self, num_iter, shifted_logits, eps, q, dim, lb, ub):
        """Identical to the binary search in StieltjesTransform."""
        for _ in range(num_iter):
            mid = (lb + ub) / 2.0
            
            # Note: q can now be a tensor, but this will broadcast correctly
            prob_sum = torch.sum(
                torch.pow((mid - shifted_logits).clamp(min=eps), -q),
                dim=dim,
                keepdim=True
            ) - 1
            
            lb = torch.where(prob_sum > 0, mid, lb)
            ub = torch.where(prob_sum <= 0, mid, ub)

        return lb, ub

    def translate_logits(self, logits: torch.Tensor, dim: int, **kwargs) -> torch.Tensor:
        
        # --- 1. Calculate Adaptive 'q' (from AdaptiveSoftmax) ---
        with torch.no_grad():
            # We must use standard softmax to calculate the initial entropy
            probs = F.softmax(logits, dim=dim).to(torch.float32)
            log_probs = F.log_softmax(logits, dim=dim).to(torch.float32)
            # entropy shape will be (B, 1, 1) in our model
            entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        
        poly_fit = self.poly_fit.to(logits.device)
        one = self.one.to(logits.device)
        poly_val = self._polyval_horner(poly_fit, entropy)
        greater_mask = entropy > 0.5
        
        poly_val = torch.clamp(poly_val, min=1.0, max=10.0)
        
        # 'beta' from AdaptiveSoftmax is our new 'q'
        # q shape will be (B, 1, 1), same as entropy
        q = torch.where(greater_mask, torch.maximum(poly_val, one), one)
        q = q.to(dtype=logits.dtype) + self.eps

        # --- 2. Run Stieltjes Calculation (using adaptive 'q') ---
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        x_max = torch.max(logits, dim=dim, keepdim=True).values
        x_i = logits - x_max

        lb = torch.full_like(x_max, self.eps)
        # ub = n^(1/q). This handles q being a tensor.
        n = torch.tensor(logits.shape[dim], device=logits.device, dtype=logits.dtype)
        ub = torch.pow(n, 1.0 / q)

        # q is now a tensor, but the binary search handles this
        lb, ub = self._line_search_bs(
            num_iter=self.num_iter,
            shifted_logits=x_i,
            eps=self.eps,
            q=q, 
            dim=dim,
            lb=lb,
            ub=ub
        )
        lambda_1 = (lb + ub) / 2.0
        
        # 1 / (lambda_q - x_i)^q
        return torch.pow((lambda_1 - x_i).clamp(min=self.eps), -q)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class MaxRetrievalModel(nn.Module):
    """
    Implements the Deep Sets / Global Query model from Appendix A.3 [cite: 538-552].
    This REPLACES the SelfAttnLayer.
    """
    def __init__(
        self,
        d_emb: int,
        n_classes: int,
        item_input_dim: int, # e.g., 1 (priority) + 10 (classes) = 11
        query_input_dim: int = 1, # The single random value
        softmax: Literal['traditional', 'stieltjes', 'adaptive', 'adaptive_stieltjes'] = 'traditional',
        **softmax_kwargs # To pass 'q=2' etc. to stieltjes
    ):
        super().__init__()
        self._translation_name = softmax
        self.d_emb = d_emb
        
        self.psi_x = MLP(item_input_dim, d_emb, d_emb) # MLP for items
        self.psi_q = MLP(query_input_dim, d_emb, d_emb) # MLP for query
        
        self.q_proj = nn.Linear(d_emb, d_emb)
        self.k_proj = nn.Linear(d_emb, d_emb)
        self.v_proj = nn.Linear(d_emb, d_emb)
        
        self.phi = MLP(d_emb, d_emb, n_classes)
        
        if softmax == 'traditional': self._translate_logits = TraditionalSoftmax()
        elif softmax == 'stieltjes': self._translate_logits = StieltjesTransform()
        elif softmax == 'adaptive': self._translate_logits = AdaptiveSoftmax()
        elif softmax == 'adaptive_stieltjes': self._translate_logits = AdaptiveStieltjes()
        else: raise ValueError('Error: Invalid softmax option.')
            
        self.softmax_kwargs = softmax_kwargs

    def forward(self, x_items, x_query, return_attn=False):
        # x_items shape: (B, T, item_input_dim)
        # x_query shape: (B, 1) [cite: 535]
        
        # 1. Process items and query with MLPs (Eqs. 9-10) [cite: 540]
        x_query_unsqueezed = x_query.unsqueeze(-1) # (B, 1) -> (B, 1, 1)
        h_items = self.psi_x(x_items)  # (B, T, d_emb)
        h_query = self.psi_q(x_query_unsqueezed)  # (B, 1, d_emb)
        
        # 2. Get Q, K, V (Eq. 11) [cite: 541]
        q = self.q_proj(h_query) # (B, 1, d_emb)
        k = self.k_proj(h_items) # (B, T, d_emb)
        v = self.v_proj(h_items) # (B, T, d_emb)
        
        # 3. Calculate Attention (Eq. 11) [cite: 541]
        # (B, 1, d_emb) @ (B, d_emb, T) -> (B, 1, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if self._translation_name == 'traditional':
            attn_scores *= k.size(-1) ** -0.5 # Scale for traditional softmax

        # 4. Apply Softmax (Eq. 12) [cite: 542]
        attn_weights = self._translate_logits.translate_logits(
            attn_scores, 
            dim=-1,
            **self.softmax_kwargs
        ) # (B, 1, T)

        # 5. Get aggregated vector z (Eq. 13) [cite: 544]
        # (B, 1, T) @ (B, T, d_emb) -> (B, 1, d_emb)
        z = torch.matmul(attn_weights, v)
        z = z.squeeze(1) # (B, d_emb)
        
        # 6. Final prediction (Eq. 14) [cite: 547]
        out_logits = self.phi(z) # (B, n_classes)

        return (out_logits, attn_weights) if return_attn else out_logits


def plot_max_retrieval_attention(model, device, save_path, n_classes, item_input_dim, start_len=16, num_doubles=8, batch_size=32):
    """
    Plots attention maps like Figure 2 from the paper [cite: 108-110].
    """
    model.eval()
    fig, axes = plt.subplots(1, num_doubles, figsize=(15, 5), sharey=True)
    fig.subplots_adjust(wspace=0.1)

    current_len = start_len
    for i in range(num_doubles):
        ax = axes[i]
        
        # --- Generate plot data matching the new data format ---
        priorities = torch.rand(batch_size, current_len).to(device)
        classes = torch.randint(0, n_classes, (batch_size, current_len)).to(device)
        
        priorities_t = priorities.unsqueeze(-1)
        classes_t = F.one_hot(classes, n_classes).float()
        items = torch.cat([priorities_t, classes_t], dim=-1) # (B, T, 1+C)
        
        query_vec = torch.rand(batch_size, 1).to(device) # (B, 1)
        # --------------------------------------------------------

        with torch.no_grad():
            _, attn_weights = model(items, query_vec, return_attn=True)
            attn_weights = attn_weights.squeeze(1) # (B, 1, T) -> (B, T)

        # Sort by priority
        sorted_value_indices = torch.argsort(priorities, dim=1)
        top_16_value_indices = sorted_value_indices[:, -16:]
        top_k_weights = torch.gather(attn_weights, 1, top_16_value_indices)
        
        ax.imshow(top_k_weights.cpu().numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f"{current_len}", fontsize=8)
        ax.set_xticks([])
        
        current_len *= 2

    plt.savefig(f'{save_path}.png') ; plt.close()

def plot_learning_curve(
    train_losses: List[float],
    val_ID_losses: List[float],
    val_OOD_losses: List[float],
    train_time: float,
    name: str,
    save_path: str,
    log_every_steps: int
):
    steps_range = list(range(1, len(train_losses) + 1))
    steps_ticks = [s * log_every_steps for s in steps_range] # Show actual steps

    plt.figure() # Create a new figure
    plt.plot(steps_ticks, train_losses, 'b-o', label='Training Loss')
    plt.plot(steps_ticks, val_ID_losses, 'g-o', label='In-Distribution Val Loss')
    plt.plot(steps_ticks, val_OOD_losses, 'r-o', label='OOD Val Loss')
    plt.title(f'Learning curves for {name} | trained for {train_time:.2f} seconds.')
    plt.legend()
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (normalized)')
    plt.grid(True)
    plt.savefig(f'{save_path}.png') ; plt.close()


class MaxDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_max_data(batch):
    """Collates (items, query_vec, target_class)"""
    items, queries, targets = zip(*batch) 
    
    items_padded = rnn_utils.pad_sequence(items, batch_first=True, padding_value=0.0)
    queries_stacked = torch.stack(queries)
    targets_stacked = torch.stack(targets)
    
    return items_padded, queries_stacked, targets_stacked

def make_dataset(
    len_dataset: int,
    max_len_seq: int,
    n_classes: int = 10,
    min_len_seq: int = 5, # Paper uses 5-16 [cite: 583]
    batch_size: int = 128,
    shuffle: bool = True,
):
    data = []
    for _ in tqdm(range(len_dataset), desc='creating dataset'):
        current_len = np.random.randint(min_len_seq, max_len_seq + 1)
        
        # 1. Sample priorities ~ U(0,1) [cite: 529]
        priorities = np.random.uniform(0, 1, size=current_len)
        
        # 2. Sample classes ~ U{0..C-1} [cite: 532]
        classes = np.random.randint(0, n_classes, size=current_len)
        
        # 3. Find target class (class of max priority item) [cite: 536]
        target_class_idx = np.argmax(priorities)
        target_class = classes[target_class_idx]
        
        # 4. Create item features: [priority, one_hot_class] [cite: 533]
        priorities_t = torch.tensor(priorities).float().unsqueeze(-1)
        classes_t = F.one_hot(torch.tensor(classes), n_classes).float()
        items = torch.cat([priorities_t, classes_t], dim=-1) # (T, 1 + n_classes)
        
        # 5. Create query vector q ~ U(0,1) [cite: 535]
        query_vec = torch.tensor([np.random.uniform(0, 1)]).float() # (1,)
        
        data.append((items, query_vec, torch.tensor(target_class).long()))

    dataset = MaxDataset(data)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_max_data,
        num_workers=4
    )

# ---- 5. TRAINING & MAIN (Updated) ---- #

def train_or_val(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    train_mode: bool = True,
    optim: Optional[Optimizer] = None,
) -> float:
    """Run training or validation on model given other args."""
    total_loss = 0.0
    
    if train_mode:
        assert optim, 'need optimizer for train mode'
        model.train()
        
        for items, queries, targets in dataloader:
            items = items.to(device)
            queries = queries.to(device)
            targets = targets.to(device)
            
            optim.zero_grad()
            
            out_logits = model(items, queries)
            loss = loss_fn(out_logits, targets)
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
    
    else:
        model.eval()
        # --- FIX: Add torch.no_grad() for validation ---
        with torch.no_grad(): 
            for items, queries, targets in dataloader:
                items = items.to(device)
                queries = queries.to(device)
                targets = targets.to(device)
                
                out_logits = model(items, queries)
                loss = loss_fn(out_logits, targets)
                total_loss += loss.item()
                
    return total_loss / len(dataloader) # Return average batch loss

if __name__ == '__main__':
    results_folder = './results/'
    if not os.path.isdir(results_folder): os.mkdir(results_folder)

    # --- HYPERPARAMS (Updated to match Appendix A.4) --- [cite: 580-587]
    d_emb = 128
    n_classes = 10
    training_steps = 5000
    log_every_steps = 500
    batch_size = 128
    max_len_seq = 16
    min_len_seq = 5
    ood_len_seq = 128 # For OOD validation
    lr = 0.001
    weight_decay = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- DATALOADERS ---
    item_input_dim = 1 + n_classes
    # Make dataset large enough to not repeat (or just use iter)
    len_train_dataset = training_steps * batch_size 
    len_val_dataset = 1024
    
    dataloader = make_dataset(
        len_train_dataset, max_len_seq, n_classes, min_len_seq, batch_size=batch_size
    )
    dataloader_val_ID = make_dataset(
        len_val_dataset, max_len_seq, n_classes, min_len_seq, batch_size=batch_size
    )
    dataloader_val_OOD = make_dataset(
        len_val_dataset, ood_len_seq, n_classes, min_len_seq, batch_size=batch_size
    )
    
    train_iter = iter(dataloader) # For step-based training

    # --- Train for each softmax type ---
    for logits_translation in ['traditional', 'stieltjes', 'adaptive', 'adaptive_stieltjes']:
        start = time.time()

        # --- MODEL ---
        model = MaxRetrievalModel(
            d_emb=d_emb,
            n_classes=n_classes,
            item_input_dim=item_input_dim,
            query_input_dim=1,
            softmax=logits_translation,
            q=3
        ).to(device)
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss() # --- FIX: Use CrossEntropyLoss ---

        # --- TRAINING LOOP (Step-based) ---
        train_losses, val_ID_losses, val_OOD_losses = [], [], []
        
        for step in tqdm(range(training_steps), desc=f"Training {logits_translation}"):
            try:
                items, queries, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader)
                items, queries, targets = next(train_iter)
            
            items, queries, targets = items.to(device), queries.to(device), targets.to(device)
            
            model.train()
            optimizer.zero_grad()
            out_logits = model(items, queries)
            loss = loss_fn(out_logits, targets)
            loss.backward()
            optimizer.step()
            
            # --- Validation & Logging ---
            if step > 0 and (step % log_every_steps == 0 or step == training_steps - 1):
                val_ID_loss = train_or_val(model, dataloader_val_ID, loss_fn, device, train_mode=False)
                val_OOD_loss = train_or_val(model, dataloader_val_OOD, loss_fn, device, train_mode=False)
                
                train_losses.append(loss.item()) # Current batch loss
                val_ID_losses.append(val_ID_loss)
                val_OOD_losses.append(val_OOD_loss)
                
                print(f'\nStep {step} | Train Loss: {train_losses[-1]:.4f} | Val Loss (ID): {val_ID_losses[-1]:.4f} | Val Loss (OOD): {val_OOD_losses[-1]:.4f}')
        
        # --- PLOTTING ---
        time_train = time.time() - start
        print(f'Time to train using {logits_translation}: {time_train}')
        
        plot_max_retrieval_attention(
            model=model,
            device=device,
            save_path=results_folder + logits_translation,
            n_classes=n_classes,
            item_input_dim=item_input_dim
        )

        plot_learning_curve(
            train_losses=train_losses,
            val_ID_losses=val_ID_losses,
            val_OOD_losses=val_OOD_losses,
            train_time=time_train,
            name=logits_translation,
            save_path=results_folder + f'{logits_translation}_learning_curves',
            log_every_steps=log_every_steps
        )