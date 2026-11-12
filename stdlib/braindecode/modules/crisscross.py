# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn

class CrissCrossTransformerEncoderLayer(nn.Module):
    self_attn_s: nn.MultiheadAttention
    self_attn_t: nn.MultiheadAttention
    linear1: nn.Linear
    dropout: nn.Dropout
    linear2: nn.Linear
    norm_first: bool
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    dropout1: nn.Dropout
    dropout2: nn.Dropout
    activation: nn.Module

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = nn.GELU, layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 norm_first: bool = False, bias: bool = True):
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, bias=bias, batch_first=True)
        self.add_module("self_attn_s", self.self_attn_s)
        self.self_attn_t = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, bias=bias, batch_first=True)
        self.add_module("self_attn_t", self.self_attn_t)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.add_module("linear1", self.linear1)
        self.dropout = nn.Dropout(dropout)
        self.add_module("dropout", self.dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.add_module("linear2", self.linear2)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.add_module("norm1", self.norm1)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.add_module("norm2", self.norm2)
        self.dropout1 = nn.Dropout(dropout)
        self.add_module("dropout1", self.dropout1)
        self.dropout2 = nn.Dropout(dropout)
        self.add_module("dropout2", self.dropout2)
        self.activation = activation()
        self.add_module("activation", self.activation)

    def _sa_block(self, x: Tensor) -> Tensor:
        bz, ch, pn, ps = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        xs = x[:, :, :, :ps // 2].permute(0, 2, 1, 3).reshape(bz * pn, ch, -1)
        xt = x[:, :, :, ps // 2:].reshape(bz * ch, pn, -1)
        ys, _ = self.self_attn_s.attend(xs, xs, xs, need_weights=False)
        ys = ys.reshape(bz, pn, ch, -1).permute(0, 2, 1, 3)
        yt, _ = self.self_attn_t.attend(xt, xt, xt, need_weights=False)
        yt = yt.reshape(bz, ch, pn, -1)
        return self.dropout1(torch.cat([ys, yt], dim=3))

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src: Tensor) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _get_name(self) -> str:
        return "CrissCrossTransformerEncoderLayer"
