# Implementation of figure 2 in softmax is not enough paper
#  - Goal - show out of distribution sequence lengths do not have sharp attention
#  - Methodology - train single attention layer on max seq len of 16, plot highest attention weights for seqeunces lengths 16, 32, 64, ..., see if attention is sharp.
#  - The task is retrieving the max element in a random sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from typing import Literal

from softmax_fns import TraditionalSoftmax, StieltjesTransform, AdaptiveSoftmax, TopKSoftmax

class MaxDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.idx = 0
        if self.shuffle: random.shuffle(self.dataset)

        return self

    def __next__(self):
        if self.idx >= len(self.dataset): raise StopIteration
        batch = self.dataset[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size

        xs, ys = zip(*batch)
        return list(xs), torch.stack(ys)


class SelfAttnLayer(nn.Module):
    def __init__(
        self,
        d_emb,
        softmax: Literal['traditional', 'adaptive', 'stieltjes', 'top_k'] = 'traditional',
    ):
        super().__init__()
        if softmax == 'traditional': self._translate_logits = TraditionalSoftmax()
        elif softmax == 'stieltjes': self._translate_logits = StieltjesTransform()
        elif softmax == 'adaptive': self._translate_logits = AdaptiveSoftmax()
        elif softmax == 'top_k': self._translate_logits = TopKSoftmax()
        else: raise ValueError('Error: Invalid softmax option.')

        self.c_attn = nn.Linear(d_emb, 3 * d_emb)
        self.c_proj = nn.Linear(d_emb, d_emb)
        self.fc = nn.Linear(d_emb, 1)
        self.d_emb = d_emb

    def forward(self, x, return_attn=False):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_emb, dim=2)

        scale = q.size(-1) ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  #(B, T, T)

        # Causal mask removed for this task, allowing global attention.
        # mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = self._translate_logits.translate_logits(attn_scores, dim=-1)  #(B, T, T)

        y = torch.matmul(attn_weights, v)  #(B, T, C)

        y = self.c_proj(y)
        out = self.fc(y)

        return (out, attn_weights, attn_scores) if return_attn else out
    

def make_dataset(len_dataset: int, max_len_seq: int):
    data = []
    seqs = [np.random.randint(low=0, high=30, size=max_len_seq) for _ in tqdm(range(len_dataset), desc='generating nums')]

    for seq in tqdm(seqs, desc='creating dataset'):
        for i, _ in enumerate(seq):
            x = seq[:i+1]
            y = max(x)
            data.append((torch.tensor(x),torch.tensor(y)))

    return data

def plot_max_retrieval_attention(model, embedding, device, name, start_len=16, num_doubles=7, batch_size=32):
    """
    plots attention maps like Figure 2 from the paper.
    The columns are sorted by the input item's actual value in ascending order.
    """
    #NOTE: paper uses 11 different doubles for max_seq_len, but I am reaching OOM for L4 gpu for large sequences (16 * 2^11)
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

        final_token_weights = attn_weights[:, -1, :] #(batch_size, current_len)
        sorted_value_indices = torch.argsort(xs, dim=1)
        top_16_value_indices = sorted_value_indices[:, -16:]
        top_k_weights = torch.gather(final_token_weights, 1, top_16_value_indices)
        ax.imshow(top_k_weights.cpu().numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f"{current_len}", fontsize=8) ; ax.set_xticks([])
        current_len *= 2

    plt.savefig(f'{name}.png')

if __name__ == '__main__':
    for logits_translation in ['traditional', 'adaptive']:
        #hyperparameters
        d_emb = 16
        epochs = 10
        batch_size = 1000
        max_len_seq = 16
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #dataset
        train = make_dataset(1000, max_len_seq)
        dev = make_dataset(50, max_len_seq)
        dataloader = MaxDataLoader(train, batch_size)
        val_loader = MaxDataLoader(dev, batch_size)

        #model
        embedding = nn.Embedding(30, d_emb).to(device)
        model = SelfAttnLayer(d_emb, softmax=logits_translation).to(device)
        optimizer = optim.AdamW(list(model.parameters()) + list(embedding.parameters()), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            dataloader = MaxDataLoader(train, batch_size)
            total_loss = 0
            for xs, ys in dataloader:
                ys = ys.to(device)
                optimizer.zero_grad()
                batch_losses = []

                for x, y in zip(xs, ys):
                    x=x.to(device)

                    emb = embedding(x).unsqueeze(0)  #(1, T, d_emb)
                    out = model(emb)  #(1, T, 1)
                    pred = out[0, -1, 0]  #last token predicts max
                    loss = loss_fn(pred, y.float())
                    batch_losses.append(loss)

                batch_loss = torch.stack(batch_losses).mean()
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()

                #val
                #model.eval()
                #val_loss = 0
                # with torch.no_grad():
                #     for xs, ys in val_loader:
                #         ys = ys.to(device)
                #         batch_losses = []

                #         for x, y in zip(xs, ys):
                #             x = x.to(device)

                #             emb = embedding(x).unsqueeze(0)
                #             out = model(emb)
                #             pred = out[0, -1, 0]
                #             loss = loss_fn(pred, y.float())
                #             batch_losses.append(loss)

                #         batch_loss = torch.stack(batch_losses).mean()
                #         val_loss += batch_loss.item()
            print(f'Epoch {epoch+1} | Train Loss: {total_loss:.4f}')# | Val Loss: {val_loss:.4f}')

        plot_max_retrieval_attention(model, embedding, device, logits_translation)