# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor

def square(x: Tensor) -> Tensor:
    return x * x

def safe_log(x: Tensor, eps: float = 1e-06) -> Tensor:
    return torch.log(torch.clamp(x, min=eps))

def identity(x: Tensor) -> Tensor:
    return x

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = [x.shape[0]] + [1 for _ in range(x.ndim - 1)]
    probabilities = torch.full(shape, keep_prob, dtype=torch.float32)
    random_tensor = torch.bernoulli(probabilities).to(x.dtype)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor

def _next_fast_len(n: int) -> int:
    n -= 1
    m = 0
    while m != 1:
        n += 1
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m = m // p
    return n

def sinusoidal_positional_encoding(n_positions: int, dim: int, base: float = 10000.0) -> Tensor:
    dim_even = dim + dim % 2
    position = torch.arange(n_positions).unsqueeze(1).to(torch.float32)
    div_term = torch.exp(torch.arange(0, dim_even, 2).to(torch.float32) * (-math.log(base) / dim_even))
    pe = torch.zeros(n_positions, dim_even)
    ang = position * div_term
    s = torch.sin(ang)
    c = torch.cos(ang)
    pe = torch.stack([s, c], dim=-1).reshape(n_positions, dim_even)
    return pe[:, :dim].contiguous()

def rotate_pairs(x: Tensor) -> Tensor:
    pairs = x.unflatten(-1, [-1, 2])
    return torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).flatten(-2)

def rotate_half(x: Tensor) -> Tensor:
    ch = x.chunk(2, dim=-1)
    return torch.cat([-ch[1], ch[0]], dim=-1)

def dwt_max_level(n_times: int, filter_len: int) -> int:
    if n_times < filter_len:
        return 0
    ratio = n_times // (filter_len - 1)
    level = 0
    while ratio >= 2:
        ratio = ratio // 2
        level += 1
    return level

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None,
                                 dropout_p: float = 0.0, scale: Optional[float] = None, exact: bool = False) -> Tensor:
    # braindecode's attention helper. `exact=True` is the explicit
    # softmax(QK^T/sqrt(d))V path; otherwise torch's SDPA.
    import torch.nn.functional as F
    if exact:
        scores = torch.matmul(query, key.transpose(-2, -1))
        if scale is None:
            scores = scores / math.sqrt(float(query.size(-1)))
        else:
            scores = scores * scale.__val__()
        if attn_mask is not None:
            m = attn_mask.__val__()
            if m.dtype == torch.bool:
                scores = scores.masked_fill(m.logical_not(), -math.inf)
            else:
                scores = scores + m
        weights = scores.softmax(dim=-1)
        if dropout_p > 0.0:
            weights = F.dropout(weights, dropout_p, True)
        return torch.matmul(weights, value)
    if scale is None:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale)

def rotary_frequencies(head_dim: int, base: float = 10000.0) -> Tensor:
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)
    return 1.0 / torch.pow(torch.full([exponent.numel()], base), exponent)

def rotary_angles(positions: Tensor, inv_freq: Tensor, interleaved: bool = True) -> Tensor:
    angles = positions.unsqueeze(-1) * inv_freq
    if interleaved:
        return torch.repeat_interleave(angles, 2, dim=-1)
    return torch.cat([angles, angles], dim=-1)

def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor, interleaved: bool = True) -> Tensor:
    rotated = rotate_pairs(x) if interleaved else rotate_half(x)
    return x * cos + rotated * sin
