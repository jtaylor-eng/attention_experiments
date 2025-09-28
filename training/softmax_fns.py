#Crux of the research: evaluating alternate softmax functions within attention blocks

#NOTE: utils.TransformerConfig requires a customsoftmaxfn object to use in transfomer training. 

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

"""Provide uniform interface to use alternate softmax fns / alternatives."""
class CustomSoftmaxFn(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def translate_logits(
        self,
        logits: torch.Tensor,
        dim: int
    ) -> torch.Tensor:
        pass


#Just torch.nn.functional.softmax
class TraditionalSoftmax(CustomSoftmaxFn):
    def translate_logits(self, logits, dim): return F.softmax(logits, dim=dim)


#Adapted from softmax is not enough paper (from JAX implementation)
class AdaptiveSoftmax(CustomSoftmaxFn):
    def __init__(self, coeffs=None):
        super().__init__()
        if coeffs is None: coeffs = [-0.037, 0.481, -2.3, 4.917, -1.791]
        self.register_buffer("poly_fit", torch.tensor(coeffs, dtype=torch.float32))
        self.register_buffer("one", torch.tensor(1.0, dtype=torch.float32))

    @staticmethod
    def _polyval_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.float32)
        for c in coeffs:
            out = out * x + c
        return out

    def translate_logits(self, logits: torch.Tensor, dim: int) -> torch.Tensor:
        probs = F.softmax(logits, dim=dim).to(torch.float32)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1, keepdim=True)
        
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


#Sample replacement for Softmax
class TopKSoftmax(CustomSoftmaxFn):
    def __init__(self, top_k_pct=0.1, max_sharp_tokens=10, smoothing_factor=0.1):
        super().__init__()
        self.top_k_pct = float(top_k_pct)
        self.max_sharp_tokens = int(max_sharp_tokens)
        self.smoothing_factor = float(smoothing_factor)

    def translate_logits(self, logits: torch.Tensor, dim: int) -> torch.Tensor:
        # Expect logits of shape [..., seq_len] along the softmax dim
        seq_len = logits.size(dim)
        top_k = max(1, int(seq_len * self.top_k_pct))
        top_k = min(self.max_sharp_tokens, top_k)

        # topk along the 'dim' dimension; torch.topk requires dim param
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=dim, largest=True, sorted=True)

        # mask for top-k positions
        top_k_mask = torch.zeros_like(logits, dtype=torch.bool)
        # scatter True into positions
        top_k_mask.scatter_(dim, top_k_indices, True)

        # softmax over the top-k values (along last axis of top_k_values which corresponds to dim)
        topk_soft = F.softmax(top_k_values, dim=-1).to(logits.dtype)

        # remaining weights: set top-k positions to -inf so softmax ignores them
        neg_inf = torch.finfo(logits.dtype).min
        remaining_logits = logits.masked_fill(top_k_mask, neg_inf)
        remaining_soft = F.softmax(remaining_logits, dim=dim)

        # assemble final weights
        final = torch.zeros_like(logits, dtype=logits.dtype)
        final.scatter_(dim, top_k_indices, topk_soft)

        remaining_masked = remaining_soft.masked_fill(top_k_mask, 0.0)
        final = final * (1.0 - self.smoothing_factor) + remaining_masked * self.smoothing_factor

        return final
