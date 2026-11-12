# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiHeadAttention(nn.Module):
    emb_size: int
    num_heads: int
    head_dim: int
    scale: Optional[float]
    keys: nn.Linear
    queries: nn.Linear
    values: nn.Linear
    q_norm: Optional[nn.LayerNorm]
    k_norm: Optional[nn.LayerNorm]
    att_drop: float
    projection: nn.Linear

    def __init__(self, emb_size: int, num_heads: int, dropout: float = 0.0, scale: Optional[float] = None,
                 head_dim: Optional[int] = None, qk_norm: bool = False):
        super().__init__()
        if num_heads < 1 or (head_dim is not None and head_dim < 1):
            raise ValueError(f"num_heads ({num_heads}) and head_dim ({head_dim}) must be positive.")
        hd = 0
        if head_dim is None:
            if emb_size % num_heads != 0:
                raise ValueError(f"emb_size ({emb_size}) must be divisible by num_heads ({num_heads}).")
            hd = emb_size // num_heads
        else:
            hd = head_dim
        inner = num_heads * hd
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = hd
        self.scale = scale
        self.keys = nn.Linear(emb_size, inner)
        self.add_module("keys", self.keys)
        self.queries = nn.Linear(emb_size, inner)
        self.add_module("queries", self.queries)
        self.values = nn.Linear(emb_size, inner)
        self.add_module("values", self.values)
        self.q_norm = None
        self.k_norm = None
        if qk_norm:
            qn = nn.LayerNorm(hd)
            kn = nn.LayerNorm(hd)
            self.q_norm = qn
            self.k_norm = kn
            self.add_module("q_norm", qn)
            self.add_module("k_norm", kn)
        self.att_drop = dropout
        self.projection = nn.Linear(inner, emb_size)
        self.add_module("projection", self.projection)

    def _stack(self, t: Tensor) -> Tensor:
        return rearrange(t, "batch seq (heads head_dim) -> batch heads seq head_dim", heads=self.num_heads)

    def forward(self, x: Tensor) -> Tensor:
        queries = self._stack(self.queries(x))
        keys = self._stack(self.keys(x))
        values = self._stack(self.values(x))
        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)
        dp = self.att_drop if self.training else 0.0
        if self.scale is not None:
            queries = queries * (self.scale * math.sqrt(float(self.head_dim)))
        out = F.scaled_dot_product_attention(queries, keys, values, dropout_p=dp)
        out = rearrange(out, "batch heads seq head_dim -> batch seq (heads head_dim)")
        return self.projection(out)

    def _get_name(self) -> str:
        return "MultiHeadAttention"
