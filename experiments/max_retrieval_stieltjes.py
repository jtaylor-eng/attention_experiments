# Implementation of figure 2 in softmax is not enough paper
#  - Goal - show out of distribution sequence lengths do not have sharp attention
#  - Methodology - train single attention layer on max seq len of 16, plot highest attention weights for seqeunces lengths 16, 32, 64, ..., see if attention is sharp.
#  - The task is retrieving the max element in a random sequence

"""
TODO:
 - fix sorting in plots
 - validation (both in and out of distribution)
 - plot training time (title) with various approaches
 - plot train, val learning curves
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from typing import Literal
import time

from softmax_fns import TraditionalSoftmax, StieltjesTransform, AdaptiveSoftmax 

# class MaxDataLoader:
#     def __init__(self, dataset, batch_size, shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#     def __iter__(self):
#         self.idx = 0
#         if self.shuffle: random.shuffle(self.dataset)

#         return self

#     def __next__(self):
#         if self.idx >= len(self.dataset): raise StopIteration
#         batch = self.dataset[self.idx : self.idx + self.batch_size]
#         self.idx += self.batch_size

#         xs, ys = zip(*batch)
#         return list(xs), torch.stack(ys)
class MaxDataset(Dataset): #for use of collator
    def __init__(self, data):self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class SelfAttnLayer(nn.Module):
    def __init__(
        self,
        d_emb,
        softmax: Literal['traditional', 'stieltjes', 'adaptive'] = 'traditional',
    ):
        super().__init__()
        self.logit_translation = softmax
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

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  #(B, T, T)

        if self._translate_logits == 'softmax': #scale for traditional softmax
            attn_scores *= q.size(-1) ** -0.5

        #causal mask removed for this task, allowing global attention.
        # mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = self._translate_logits.translate_logits(attn_scores, dim=-1)  #(B, T, T)

        y = torch.matmul(attn_weights, v)  #(B, T, C)
        y = self.c_proj(y)
        out = self.fc(y)

        return (out, attn_weights, attn_scores) if return_attn else out
    

# def make_dataset(len_dataset: int, max_len_seq: int):
#     data = []
#     seqs = [np.random.randint(low=0, high=30, size=max_len_seq) for _ in tqdm(range(len_dataset), desc='generating nums')]

#     for seq in tqdm(seqs, desc='creating dataset'):
#         for i, _ in enumerate(seq):
#             x = seq[:i+1]
#             y = max(x)
#             data.append((torch.tensor(x),torch.tensor(y)))

#     return data

def make_dataset(len_dataset: int, max_len_seq: int):
    data = []
    for _ in tqdm(range(len_dataset), desc='creating dataset'):
        #pick random len within 1 to max_len_seq
        current_len = np.random.randint(1, max_len_seq + 1) 
        x = np.random.randint(low=0, high=30, size=current_len)
        y = max(x)
        data.append((torch.tensor(x), torch.tensor(y)))
    return data

def collate_max_data(batch):
    xs, ys = zip(*batch)
    return rnn_utils.pad_sequence(xs, batch_first=True, padding_value=0), torch.stack(ys)

def plot_max_retrieval_attention(model, embedding, device, name, start_len=32, num_doubles=7, batch_size=32):
    """
    plots attention maps like Figure 2 from the paper.
    The columns are sorted by the input item's actual value in ascending order.
    """
    #NOTE: paper uses 11 different doubles for max_seq_len, but I am reaching OOM for large sequences (16 * 2^11)
    model.eval()
    fig, axes = plt.subplots(1, num_doubles, figsize=(15, 5), sharey=True)
    fig.subplots_adjust(wspace=0.1)

    current_len = start_len
    for i in range(num_doubles):
        ax = axes[i]
        xs = torch.randint(0, 30, (batch_size, current_len)).to(device)

        with torch.no_grad():
            emb = embedding(xs)
            _, attn_weights, _ = model(emb, return_attn=True)

        #TODO: not sorting
        final_token_weights = attn_weights[:, -1, :] #(batch_size, current_len)
        sorted_value_indices = torch.argsort(xs, dim=1)
        top_16_value_indices = sorted_value_indices[:, -16:]
        top_k_weights = torch.gather(final_token_weights, 1, top_16_value_indices)
        ax.imshow(top_k_weights.cpu().numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f"{current_len}", fontsize=8) ; ax.set_xticks([])
        current_len *= 2

    plt.savefig(f'{name}.png')

if __name__ == '__main__':
    results_folder = './results/'

    #hyperparameters
    d_emb = 16
    epochs = 50
    batch_size = 1028
    max_len_seq = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #dataset
    train_data = make_dataset(102800, max_len_seq)
    train_dataset = MaxDataset(train_data)
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_max_data
    )

    #train attn model with various logit translation methods
    for logits_translation in ['traditional', 'stieltjes', 'adaptive']:
        start = time.time()
        #model
        embedding = nn.Embedding(30, d_emb).to(device)
        model = SelfAttnLayer(d_emb, softmax=logits_translation).to(device)
        optimizer = optim.AdamW(list(model.parameters()) + list(embedding.parameters()), lr=1e-4)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for xs, ys in dataloader:
                xs = xs.to(device)
                ys = ys.to(device)
                
                optimizer.zero_grad()
                
                emb = embedding(xs)
                out = model(emb)
                pred = out[:, -1, 0]
                
                loss = loss_fn(pred, ys.float())
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                #TODO:
                #val
                #model.eval()
                #...
            
            print(f'Epoch {epoch+1} | Train Loss: {total_loss:.4f}')# | Val Loss: {val_loss:.4f}')

        print(f'Time to train using {logits_translation}: {time.time() - start}')
        plot_max_retrieval_attention(model, embedding, device, results_folder + logits_translation)