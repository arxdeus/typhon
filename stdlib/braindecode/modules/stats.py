# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn

class StatLayer(nn.Module):
    # stat: "mean" | "max" | "var" | "std" | "power" (Codon replacement for
    # the stat_fn callable).
    stat: str
    dim: int
    keepdim: bool
    has_clamp: bool
    clamp_lo: float
    clamp_hi: float
    apply_log: bool

    def __init__(self, stat: str, dim: int, keepdim: bool = True, clamp_range: Optional[Tuple[float, float]] = None,
                 apply_log: bool = False):
        super().__init__()
        self.stat = stat
        self.dim = dim
        self.keepdim = keepdim
        self.has_clamp = clamp_range is not None
        self.clamp_lo = 0.0
        self.clamp_hi = 0.0
        if clamp_range is not None:
            self.clamp_lo, self.clamp_hi = clamp_range.__val__()
        self.apply_log = apply_log

    def forward(self, x: Tensor) -> Tensor:
        out = x
        if self.stat == "mean":
            out = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
        elif self.stat == "max":
            out = x.amax(dim=self.dim, keepdim=self.keepdim)
        elif self.stat == "var":
            out = torch.var(x, dim=self.dim, keepdim=self.keepdim)
        elif self.stat == "std":
            out = torch.std(x, dim=self.dim, keepdim=self.keepdim)
        elif self.stat == "power":
            out = torch.mean(x * x, dim=self.dim, keepdim=self.keepdim)
        else:
            raise ValueError(self.stat)
        if self.has_clamp:
            out = torch.clamp(out, min=self.clamp_lo, max=self.clamp_hi)
        if self.apply_log:
            out = torch.log(out)
        return out

    def _get_name(self) -> str:
        return "StatLayer"

def MeanLayer(dim: int, keepdim: bool = True) -> StatLayer:
    return StatLayer("mean", dim, keepdim)

def MaxLayer(dim: int, keepdim: bool = True) -> StatLayer:
    return StatLayer("max", dim, keepdim)

def VarLayer(dim: int, keepdim: bool = True) -> StatLayer:
    return StatLayer("var", dim, keepdim)

def StdLayer(dim: int, keepdim: bool = True) -> StatLayer:
    return StatLayer("std", dim, keepdim)

def LogVarLayer(dim: int, keepdim: bool = True) -> StatLayer:
    return StatLayer("var", dim, keepdim, (1e-06, 1000000.0), True)

def LogPowerLayer(dim: int, keepdim: bool = True) -> StatLayer:
    return StatLayer("power", dim, keepdim, (0.0001, 10000.0), True)
