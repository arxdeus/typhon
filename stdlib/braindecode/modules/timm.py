# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# TimmAttention, TimmMlp and PreNormBlock (timm-style ViT building blocks).

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.functional import scaled_dot_product_attention
from .layers import DropPath

class TimmAttention(nn.Module):
    num_heads: int
    head_dim: int
    scale: float
    exact: bool
    qkv: nn.Linear
    q_bias: Optional[Tensor]
    v_bias: Optional[Tensor]
    attn_drop: nn.Dropout
    proj: nn.Linear
    proj_drop: nn.Dropout

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, exact: bool = False):
        super().__init__()
        if not (1 <= num_heads <= dim):
            raise ValueError(f"num_heads ({num_heads}) must be in [1, dim={dim}] so head_dim >= 1.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        ahd = self.head_dim * num_heads
        self.scale = float(self.head_dim) ** -0.5
        self.exact = exact
        self.qkv = nn.Linear(dim, ahd * 3, bias=qkv_bias)
        self.add_module("qkv", self.qkv)
        self.q_bias = None
        self.v_bias = None
        if qv_bias:
            qb = nn.Parameter(torch.zeros(ahd))
            vb = nn.Parameter(torch.zeros(ahd))
            self.q_bias = qb
            self.v_bias = vb
            self.register_parameter("q_bias", qb)
            self.register_parameter("v_bias", vb)
        self.attn_drop = nn.Dropout(attn_drop)
        self.add_module("attn_drop", self.attn_drop)
        self.proj = nn.Linear(ahd, dim)
        self.add_module("proj", self.proj)
        self.proj_drop = nn.Dropout(proj_drop)
        self.add_module("proj_drop", self.proj_drop)

    def heads(self, x: Tensor) -> List[Tensor]:
        qkv = self.qkv(x)
        if self.q_bias is not None and self.v_bias is not None:
            vb = self.v_bias.__val__()
            qkv = qkv + torch.cat([self.q_bias.__val__(), torch.zeros_like(vb), vb])
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)

    def attention_weights(self, x: Tensor) -> Tensor:
        h = self.heads(x)
        return torch.softmax(torch.matmul(h[0], h[1].transpose(-2, -1)) / math.sqrt(float(h[0].size(-1))), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.heads(x)
        p = self.attn_drop.p if self.training else 0.0
        y = torch.zeros(1)
        if self.exact:
            y = scaled_dot_product_attention(h[0] * self.scale, h[1], h[2], dropout_p=p, scale=1.0, exact=True)
        else:
            y = F.scaled_dot_product_attention(h[0], h[1], h[2], dropout_p=p)
        return self.proj_drop(self.proj(y.transpose(1, 2).flatten(2)))

    def _get_name(self) -> str:
        return "TimmAttention"

class TimmMlp(nn.Module):
    fc1: nn.Linear
    act: nn.Module
    fc2: nn.Linear
    drop: nn.Dropout

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
                 act_layer = nn.GELU, drop: float = 0.0):
        super().__init__()
        o = out_features if out_features is not None else in_features
        h = hidden_features if hidden_features is not None else in_features
        self.fc1 = nn.Linear(in_features, h)
        self.add_module("fc1", self.fc1)
        self.act = act_layer()
        self.add_module("act", self.act)
        self.fc2 = nn.Linear(h, o)
        self.add_module("fc2", self.fc2)
        self.drop = nn.Dropout(drop)
        self.add_module("drop", self.drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

    def _get_name(self) -> str:
        return "TimmMlp"

class PreNormBlock[A, Mlp](nn.Module):
    norm1: nn.LayerNorm
    attn: A
    drop_path: nn.Module
    norm2: nn.LayerNorm
    mlp: Mlp
    gamma_1: Optional[Tensor]
    gamma_2: Optional[Tensor]

    def __init__(self, dim: int, attn: A, mlp: Mlp, norm_eps: float = 1e-5, drop_path: float = 0.0,
                 layer_scale: Optional[float] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.add_module("norm1", self.norm1)
        self.attn = attn
        self.add_module("attn", attn)
        dp: nn.Module = nn.Identity()
        if drop_path > 0.0:
            dp = DropPath(drop_path)
        self.drop_path = dp
        self.add_module("drop_path", dp)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.add_module("norm2", self.norm2)
        self.mlp = mlp
        self.add_module("mlp", mlp)
        self.gamma_1 = None
        self.gamma_2 = None
        if layer_scale is not None:
            g1 = nn.Parameter(layer_scale.__val__() * torch.ones(dim))
            g2 = nn.Parameter(layer_scale.__val__() * torch.ones(dim))
            self.gamma_1 = g1
            self.gamma_2 = g2
            self.register_parameter("gamma_1", g1)
            self.register_parameter("gamma_2", g2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.attn(self.norm1(x))
        if self.gamma_1 is not None:
            y = self.gamma_1.__val__() * y
        x = x + self.drop_path(y)
        y = self.mlp(self.norm2(x))
        if self.gamma_2 is not None:
            y = self.gamma_2.__val__() * y
        return x + self.drop_path(y)

    def _get_name(self) -> str:
        return "PreNormBlock"
