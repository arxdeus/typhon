# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn

class MaxNorm(nn.Module):
    max_norm_val: float
    eps: float
    def __init__(self, max_norm_val: float = 2.0, eps: float = 1e-05):
        super().__init__()
        self.max_norm_val = max_norm_val
        self.eps = eps
    def forward(self, X: Tensor) -> Tensor:
        norm = X.norm(2, dim=1, keepdim=True)
        denom = norm.clamp(min=self.max_norm_val / 2)
        number = denom.clamp(max=self.max_norm_val)
        return X * (number / (denom + self.eps))
    def right_inverse(self, X: Tensor) -> Tensor:
        norm = X.norm(2, dim=1, keepdim=True)
        denom = norm.clamp(min=self.max_norm_val / 2)
        number = denom.clamp(max=self.max_norm_val)
        scale = number / (denom + self.eps)
        return X / scale
    def _get_name(self) -> str:
        return "MaxNorm"

class MaxNormParametrize(nn.Module):
    max_norm: float
    def __init__(self, max_norm: float = 1.0):
        super().__init__()
        if max_norm < 0:
            raise ValueError(f"max_norm must be >= 0, got {max_norm}.")
        self.max_norm = max_norm
    def forward(self, X: Tensor) -> Tensor:
        if X.numel() == 0:
            return X
        rows = X.reshape(X.shape[0], -1)
        norm = rows.norm(2, dim=1, keepdim=True)
        mask = norm > self.max_norm
        ones = torch.ones_like(norm)
        safe_norm = torch.where(mask, norm, ones)
        scale = torch.where(mask, self.max_norm / (safe_norm + 1e-07), ones)
        return (rows * scale).reshape_as(X)
    def _get_name(self) -> str:
        return "MaxNormParametrize"
