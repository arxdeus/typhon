# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange
from braindecode.functional import drop_path

class Ensure4d(nn.Module):
    _bd: int
    def __init__(self):
        super().__init__()
        self._bd = 0
    def forward(self, x: Tensor) -> Tensor:
        while x.ndim < 4:
            x = x.unsqueeze(-1)
        return x
    def _get_name(self) -> str:
        return "Ensure4d"

class Chomp1d(nn.Module):
    chomp_size: int
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def extra_repr(self) -> str:
        return f"chomp_size={self.chomp_size}"
    def forward(self, x: Tensor) -> Tensor:
        if self.chomp_size == 0:
            return x.contiguous()
        return x[:, :, :-self.chomp_size].contiguous()
    def _get_name(self) -> str:
        return "Chomp1d"

class TimeDistributed(nn.Module):
    module: nn.Module
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.add_module("module", module)
    def forward(self, x: Tensor) -> Tensor:
        b, s, c, t = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        out = self.module(x.reshape(b * s, c, t))
        return out.reshape(b, s, -1)
    def _get_name(self) -> str:
        return "TimeDistributed"

class DropPath(nn.Module):
    drop_prob: float
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)
    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"
    def _get_name(self) -> str:
        return "DropPath"

class Residual(nn.Module):
    fn: nn.Module
    drop_path: nn.Module
    def __init__(self, fn: nn.Module, drop_path: float = 0.0):
        super().__init__()
        self.fn = fn
        self.add_module("fn", fn)
        dp: nn.Module = nn.Identity()
        if drop_path:
            dp = DropPath(drop_path)
        self.drop_path = dp
        self.add_module("drop_path", dp)
    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.fn(x))
    def _get_name(self) -> str:
        return "Residual"

class PreNormTransformerBlock(nn.Sequential):
    _pn: int
    def __init__(self, dim: int, attn: nn.Module, ffn: nn.Module, drop_prob: float):
        super().__init__()
        self._pn = 0
        self.add_module("0", Residual(nn.Sequential(nn.LayerNorm(dim), attn, nn.Dropout(drop_prob))))
        self.add_module("1", Residual(nn.Sequential(nn.LayerNorm(dim), ffn, nn.Dropout(drop_prob))))
    def _get_name(self) -> str:
        return "PreNormTransformerBlock"

class SqueezeFinalOutput(nn.Module):
    _bd: int
    def __init__(self):
        super().__init__()
        self._bd = 0
    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b c t 1 -> b c t")
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x
    def _get_name(self) -> str:
        return "SqueezeFinalOutput"

class PostNormResidual(nn.Module):
    module: nn.Module
    drop: nn.Dropout
    layernorm: nn.LayerNorm
    def __init__(self, module: nn.Module, dim: int, drop_prob: float):
        super().__init__()
        self.module = module
        self.add_module("module", module)
        self.drop = nn.Dropout(drop_prob)
        self.add_module("drop", self.drop)
        self.layernorm = nn.LayerNorm(dim)
        self.add_module("layernorm", self.layernorm)
    def forward(self, x: Tensor) -> Tensor:
        return self.layernorm(self.drop(self.module(x)) + x)
    def _get_name(self) -> str:
        return "PostNormResidual"

class PostNormTransformerBlock(nn.Module):
    attention: PostNormResidual
    feed_forward: PostNormResidual
    def __init__(self, dim: int, attn: nn.Module, ffn: nn.Module, drop_prob: float):
        super().__init__()
        self.attention = PostNormResidual(attn, dim, drop_prob)
        self.add_module("attention", self.attention)
        self.feed_forward = PostNormResidual(ffn, dim, drop_prob)
        self.add_module("feed_forward", self.feed_forward)
    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(self.attention(x))
    def _get_name(self) -> str:
        return "PostNormTransformerBlock"

class LearnedPositionalEncoding(nn.Module):
    dropout: nn.Dropout
    encoding: Tensor
    def __init__(self, length: int, dim: int, drop_prob: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, dim))
        self.register_parameter("encoding", self.encoding)
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.encoding[:, :x.size(1), :])
    def _get_name(self) -> str:
        return "LearnedPositionalEncoding"
