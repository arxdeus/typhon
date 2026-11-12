# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn

class _PreNorm(nn.Module):
    fn: nn.Module
    norm: nn.LayerNorm
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.add_module("fn", fn)
        self.norm = nn.LayerNorm(dim)
        self.add_module("norm", self.norm)
    def forward(self, x: Tensor) -> Tensor:
        return self.fn(self.norm(x))
    def _get_name(self) -> str:
        return "_PreNorm"

class _Chunk(nn.Module):
    fn: nn.Module
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.add_module("fn", fn)
    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)
    def _get_name(self) -> str:
        return "_Chunk"

class _LAFeedForward(nn.Module):
    w1: nn.Linear
    act: nn.GELU
    dropout: nn.Dropout
    w2: nn.Linear
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult)
        self.add_module("w1", self.w1)
        self.act = nn.GELU()
        self.add_module("act", self.act)
        self.dropout = nn.Dropout(0.0)
        self.add_module("dropout", self.dropout)
        self.w2 = nn.Linear(dim * mult, dim)
        self.add_module("w2", self.w2)
    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(self.act(self.w1(x))))
    def _get_name(self) -> str:
        return "_FeedForward"

class _LASelfAttention(nn.Module):
    d_heads: int
    to_q: nn.Linear
    to_k: nn.Linear
    to_v: nn.Linear
    to_out: nn.Linear
    dropout: nn.Dropout
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")
        self.d_heads = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.add_module("to_q", self.to_q)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.add_module("to_k", self.to_k)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.add_module("to_v", self.to_v)
        self.to_out = nn.Linear(dim, dim)
        self.add_module("to_out", self.to_out)
        self.dropout = nn.Dropout(dropout)
        self.add_module("dropout", self.dropout)
    def forward(self, x: Tensor) -> Tensor:
        b, t = x.shape[0], x.shape[1]
        q = self.to_q(x).reshape(b, t, -1, self.d_heads).transpose(1, 2)
        k = self.to_k(x).reshape(b, t, -1, self.d_heads).transpose(1, 2)
        v = self.to_v(x).reshape(b, t, -1, self.d_heads).transpose(1, 2)
        q = q.softmax(dim=-1) * (float(self.d_heads) ** -0.5)
        k = k.softmax(dim=-2)
        context = torch.matmul(k.transpose(-1, -2), v)
        attn = torch.matmul(q, context)
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        return self.dropout(self.to_out(attn))
    def _get_name(self) -> str:
        return "_SelfAttention"

class _Layers(nn.Module):
    layers: nn.ModuleList
    _pairs: List[Tuple[_PreNorm, _PreNorm]]
    def __init__(self, layers: nn.ModuleList, pairs: List[Tuple[_PreNorm, _PreNorm]]):
        super().__init__()
        self.layers = layers
        self.add_module("layers", layers)
        self._pairs = pairs
    def forward(self, x: Tensor) -> Tensor:
        for a, f in self._pairs:
            x = x + a(x)
            x = x + f(x)
        return x
    def _get_name(self) -> str:
        return "_Layers"

class LinearAttentionTransformer(nn.Module):
    layers: _Layers
    def __init__(self, dim: int, depth: int, heads: int = 8, attn_layer_dropout: float = 0.0):
        super().__init__()
        layers = nn.ModuleList()
        pairs = List[Tuple[_PreNorm, _PreNorm]]()
        for _ in range(depth):
            ff = _Chunk(_LAFeedForward(dim))
            att = _LASelfAttention(dim, heads, attn_layer_dropout)
            pa = _PreNorm(dim, att)
            pf = _PreNorm(dim, ff)
            ml = nn.ModuleList()
            ml.append(pa)
            ml.append(pf)
            layers.append(ml)
            pairs.append((pa, pf))
        self.layers = _Layers(layers, pairs)
        self.add_module("layers", self.layers)
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
    def _get_name(self) -> str:
        return "LinearAttentionTransformer"
