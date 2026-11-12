# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import braindecode.functional as F

class Square(nn.Module):
    _bd: int
    def __init__(self):
        super().__init__()
        self._bd = 0
    def forward(self, x: Tensor) -> Tensor:
        return x * x
    def _get_name(self) -> str:
        return "Square"

class SafeLog(nn.Module):
    epsilon: float
    def __init__(self, epsilon: float = 1e-06):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x: Tensor) -> Tensor:
        return F.safe_log(x=x, eps=self.epsilon)
    def extra_repr(self) -> str:
        return f"eps={self.epsilon}"
    def _get_name(self) -> str:
        return "SafeLog"

class LogActivation(nn.Module):
    epsilon: float
    def __init__(self, epsilon: float = 1e-06):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x + self.epsilon)
    def _get_name(self) -> str:
        return "LogActivation"

class SmoothMaximumUnit(nn.Module):
    alpha: float
    mu: Tensor
    def __init__(self, alpha: float = 0.01, mu: float = 2.5):
        super().__init__()
        self.alpha = float(alpha)
        self.mu = nn.Parameter(torch.tensor(float(mu)))
        self.register_parameter("mu", self.mu)
    def forward(self, x: Tensor) -> Tensor:
        positive = (1.0 + self.alpha) * x
        smooth = (1.0 - self.alpha) * x * torch.erf(self.mu * (1.0 - self.alpha) * x)
        return 0.5 * (positive + smooth)
    def _get_name(self) -> str:
        return "SmoothMaximumUnit"
