# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional.patches import pad_or_crop_to_patches

def _build_rotary_frequency_table(positions: Tensor, axis_dim: int, theta: float) -> Tensor:
    ed = max(axis_dim, 4)
    ex = torch.arange(0, ed, 2, dtype=torch.float32) / float(ed)
    inv = 1.0 / torch.pow(torch.full([ex.numel()], theta), ex)
    ang = positions.to(torch.float32).unsqueeze(-1) * inv
    return torch.repeat_interleave(ang, 2, dim=-1)[:, :axis_dim]

def _rotate(x: Tensor, c: Tensor, s: Tensor) -> Tensor:
    # pairs (a, b) -> (b, a); the sine table carries the (-1, +1) pair sign
    B, L, H, D = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    r = x.reshape(B, L, H, D // 2, 2).flip(-1).reshape(B, L, H, D)
    return x * c + r * s

class _ZAttention(nn.Module):
    n_heads: int
    head_dim: int
    wq: nn.Linear
    wk: nn.Linear
    wv: nn.Linear
    wo: nn.Linear
    q_norm: nn.Module
    k_norm: nn.Module
    def __init__(self, embedding_dim: int, n_heads: int, head_dim: int, norm_eps: float = 1e-5, qk_norm: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(embedding_dim, n_heads * head_dim, bias=False)
        self.add_module("wq", self.wq)
        self.wk = nn.Linear(embedding_dim, n_heads * head_dim, bias=False)
        self.add_module("wk", self.wk)
        self.wv = nn.Linear(embedding_dim, n_heads * head_dim, bias=False)
        self.add_module("wv", self.wv)
        self.wo = nn.Linear(n_heads * head_dim, embedding_dim, bias=False)
        self.add_module("wo", self.wo)
        qn: nn.Module = nn.Identity()
        kn: nn.Module = nn.Identity()
        if qk_norm:
            qn = nn.RMSNorm(head_dim, eps=norm_eps)
            kn = nn.RMSNorm(head_dim, eps=norm_eps)
        self.q_norm = qn
        self.add_module("q_norm", qn)
        self.k_norm = kn
        self.add_module("k_norm", kn)
        self.add_module("rotary_embedding", nn.Identity())
    def attn(self, x: Tensor, c: Tensor, s: Tensor) -> Tensor:
        B, L = x.shape[0], x.shape[1]
        q = self.q_norm(self.wq(x).reshape(B, L, self.n_heads, self.head_dim))
        k = self.k_norm(self.wk(x).reshape(B, L, self.n_heads, self.head_dim))
        v = self.wv(x).reshape(B, L, self.n_heads, self.head_dim)
        cc = c.unsqueeze(0).unsqueeze(2)
        ss = s.unsqueeze(0).unsqueeze(2)
        q = _rotate(q, cc, ss)
        k = _rotate(k, cc, ss)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.wo(o.transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim))
    def _get_name(self) -> str:
        return "_Attention"

class _ZFeedForward(nn.Module):
    w1: nn.Linear
    w2: nn.Linear
    w3: nn.Linear
    activation: nn.Module
    def __init__(self, embedding_dim: int, multiple_of: int = 256, ffn_dim_multiplier: Optional[float] = None, activation = nn.SiLU):
        super().__init__()
        h = int(8.0 * float(embedding_dim) / 3.0)
        if ffn_dim_multiplier is not None:
            h = int(ffn_dim_multiplier.__val__() * float(h))
        h = multiple_of * int(math.ceil(float(h) / float(multiple_of)))
        self.w1 = nn.Linear(embedding_dim, h, bias=False)
        self.add_module("w1", self.w1)
        self.w2 = nn.Linear(h, embedding_dim, bias=False)
        self.add_module("w2", self.w2)
        self.w3 = nn.Linear(embedding_dim, h, bias=False)
        self.add_module("w3", self.w3)
        self.activation = activation()
        self.add_module("activation", self.activation)
    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))
    def _get_name(self) -> str:
        return "_FeedForward"

class _ZBlock(nn.Module):
    attention: _ZAttention
    feed_forward: _ZFeedForward
    attention_norm: nn.RMSNorm
    ffn_norm: nn.RMSNorm
    attention_norm_post: nn.Module
    ffn_norm_post: nn.Module
    def __init__(self, embedding_dim: int, n_heads: int, head_dim: int, norm_eps: float, multiple_of: int,
                 ffn_dim_multiplier: Optional[float], sandwich_norm: bool, qk_norm: bool, activation):
        super().__init__()
        self.attention = _ZAttention(embedding_dim, n_heads, head_dim, norm_eps, qk_norm)
        self.add_module("attention", self.attention)
        self.feed_forward = _ZFeedForward(embedding_dim, multiple_of, ffn_dim_multiplier, activation)
        self.add_module("feed_forward", self.feed_forward)
        self.attention_norm = nn.RMSNorm(embedding_dim, eps=norm_eps)
        self.add_module("attention_norm", self.attention_norm)
        self.ffn_norm = nn.RMSNorm(embedding_dim, eps=norm_eps)
        self.add_module("ffn_norm", self.ffn_norm)
        a: nn.Module = nn.Identity()
        f: nn.Module = nn.Identity()
        if sandwich_norm:
            a = nn.RMSNorm(embedding_dim, eps=norm_eps)
            f = nn.RMSNorm(embedding_dim, eps=norm_eps)
        self.attention_norm_post = a
        self.add_module("attention_norm_post", a)
        self.ffn_norm_post = f
        self.add_module("ffn_norm_post", f)
    def block(self, x: Tensor, c: Tensor, s: Tensor) -> Tensor:
        h = x + self.attention_norm_post(self.attention.attn(self.attention_norm(x), c, s))
        return h + self.ffn_norm_post(self.feed_forward(self.ffn_norm(h)))
    def _get_name(self) -> str:
        return "_TransformerBlock"

class _ZUNAEncoder(nn.Module):
    use_registers: bool
    tok_embeddings: nn.Linear
    registers: Tensor
    layers: nn.ModuleList
    _layers: List[_ZBlock]
    norm: nn.RMSNorm
    output: nn.Linear
    rotary_cosine: Tensor
    rotary_sine: Tensor
    def __init__(self, dim: int, n_layers: int, n_heads: int, head_dim: int, input_dim: int, output_dim: int,
                 rotary_frequencies: Tensor, norm_eps: float, multiple_of: int, ffn_dim_multiplier: Optional[float],
                 sandwich_norm: bool, qk_norm: bool, activation, registers: bool):
        super().__init__()
        self.use_registers = registers
        self.tok_embeddings = nn.Linear(input_dim, dim)
        self.add_module("tok_embeddings", self.tok_embeddings)
        self.registers = nn.Parameter(torch.zeros(1, input_dim), requires_grad=registers)
        self.register_parameter("registers", self.registers)
        self._layers = [_ZBlock(dim, n_heads, head_dim, norm_eps, multiple_of, ffn_dim_multiplier, sandwich_norm, qk_norm,
                                activation) for _ in range(n_layers)]
        self.layers = nn.ModuleList()
        for l in self._layers:
            self.layers.append(l)
        self.add_module("layers", self.layers)
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.add_module("norm", self.norm)
        self.output = nn.Linear(dim, output_dim, bias=False)
        self.add_module("output", self.output)
        self.rotary_cosine = rotary_frequencies.cos()
        self.register_buffer("rotary_cosine", self.rotary_cosine, persistent=False)
        sign = torch.tensor([-1.0, 1.0]).repeat(rotary_frequencies.shape[-1] // 2)
        self.rotary_sine = rotary_frequencies.sin() * sign
        self.register_buffer("rotary_sine", self.rotary_sine, persistent=False)
    def forward(self, p: Tensor) -> Tensor:
        if not self.use_registers:
            h = self.tok_embeddings(p)
            for l in self._layers:
                h = l.block(h, self.rotary_cosine, self.rotary_sine)
            return self.output(self.norm(h))
        B, L, P = p.shape[0], p.shape[1], p.shape[2]
        r = self.registers.expand(B, L, -1)
        h = self.tok_embeddings(torch.stack([r, p], dim=2).reshape(B, 2 * L, P))
        for l in self._layers:
            h = l.block(h, self.rotary_cosine, self.rotary_sine)
        return self.output(self.norm(h.reshape(B, L, 2, -1)[:, :, 0]))
    def _get_name(self) -> str:
        return "_ZUNAEncoder"

class ZUNA(EEGModuleMixin, nn.Module):
    """ZUNA encoder (Zyphra, 2026). Architecture only (no Hub weights).

    braindecode reads electrode positions (metres, MNE head frame) from
    chs_info; here they are passed as ``channel_positions`` (n_chans, 3).
    Name-based standard-montage fallback is not ported."""
    num_channels: int
    latent_dim: int
    fine_time_pts: int
    on_non_divisible: str
    encoder: _ZUNAEncoder
    final_layer: nn.Sequential

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, dim: int = 1024,
                 n_layers: int = 16, n_heads: int = 8, head_dim: int = 64, fine_time_pts: int = 32,
                 latent_dim: int = 32, max_seqlen: int = 256, rope_theta: float = 10000.0, pos_bins: int = 50,
                 pos_half_range: float = 0.12, norm_eps: float = 1e-5, multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[float] = None, sandwich_norm: bool = True, qk_norm: bool = True,
                 activation = nn.SiLU, on_non_divisible: str = "pad", registers: bool = True,
                 channel_positions: Optional[Tensor] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.num_channels = self.n_chans
        self.latent_dim = latent_dim
        self.fine_time_pts = fine_time_pts
        self.on_non_divisible = on_non_divisible
        ctp = -(-self.n_times // fine_time_pts) if on_non_divisible == "pad" else self.n_times // fine_time_pts
        if channel_positions is None:
            raise ValueError("ZUNA requires channel locations: pass channel_positions (n_chans, 3) in metres.")
        pos = channel_positions.__val__().to(torch.float32)
        if pos.ndim != 2 or pos.shape[0] != self.n_chans or pos.shape[1] != 3:
            raise ValueError(f"channel_positions must have shape ({self.n_chans}, 3).")
        if head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by eight for 4D RoPE.")
        np_ = (pos + pos_half_range) / (2.0 * pos_half_range)
        ci = torch.clamp((np_ * float(pos_bins)).long(), 0, pos_bins - 1)
        ci = torch.repeat_interleave(ci, ctp, dim=0)
        ti = torch.arange(ctp).repeat(self.num_channels)
        tpi = torch.cat([ci, ti.unsqueeze(1)], dim=1)
        tab = _build_rotary_frequency_table(torch.arange(max(max_seqlen, pos_bins, ctp), dtype=torch.float32),
                                            head_dim // 4, rope_theta)
        rf = tab.index_select(0, tpi.reshape(-1)).reshape(tpi.shape[0], -1)
        if registers:
            rf = torch.repeat_interleave(rf, 2, dim=0)
        self.add_module("patch_embedding", nn.Sequential(nn.Identity(), nn.Identity()))
        self.encoder = _ZUNAEncoder(dim, n_layers, n_heads, head_dim, fine_time_pts, latent_dim, rf, norm_eps, multiple_of,
                                    ffn_dim_multiplier, sandwich_norm, qk_norm, activation, registers)
        self.add_module("encoder", self.encoder)
        self.final_layer = nn.Sequential(nn.Flatten(1), nn.Linear(self.num_channels * latent_dim, self.n_outputs))
        self.add_module("final_layer", self.final_layer)

    def forward_features(self, x: Tensor) -> Tensor:
        x = pad_or_crop_to_patches(x, self.fine_time_pts, self.on_non_divisible)
        p = x.reshape(x.shape[0], -1, self.fine_time_pts)
        z = self.encoder(p)
        return z.reshape(z.shape[0], self.num_channels, -1, self.latent_dim).mean(dim=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x))

    def _get_name(self) -> str:
        return "ZUNA"
