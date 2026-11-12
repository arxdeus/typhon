# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import sinusoidal_positional_encoding

class _TokenGate(nn.Module):
    mlp: nn.Sequential
    _last: nn.Linear
    def __init__(self, embed_dim: int, reduction: int, activation, init_bias: List[float]):
        super().__init__()
        h = max(16, embed_dim // reduction)
        first = nn.Linear(embed_dim, h)
        act = activation()
        self._last = nn.Linear(h, len(init_bias))
        self.mlp = nn.Sequential(first, act, self._last)
        self.add_module("mlp", self.mlp)
        nn.init.zeros_(self._last.weight)
        with torch.no_grad():
            self._last.bias.__val__().copy_(torch.tensor(init_bias, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        return torch.softmax(self.mlp(x.detach()), dim=-1)
    def _get_name(self) -> str:
        return "_TokenGate"

class _GatedFeedForward(nn.Module):
    in_proj: nn.Linear
    out_proj: nn.Linear
    def __init__(self, embed_dim: int, expansion: int):
        super().__init__()
        h = expansion * embed_dim
        self.in_proj = nn.Linear(embed_dim, 2 * h)
        self.add_module("in_proj", self.in_proj)
        self.out_proj = nn.Linear(h, embed_dim)
        self.add_module("out_proj", self.out_proj)
    def forward(self, x: Tensor) -> Tensor:
        c = self.in_proj(x).chunk(2, dim=-1)
        g = c[1] * torch.sigmoid(c[1]) + F.gelu(c[1], approximate="tanh")
        return self.out_proj(c[0] * g)
    def _get_name(self) -> str:
        return "_GatedFeedForward"

class _AxisAttention(nn.Module):
    num_heads: int
    head_dim: int
    att_drop_prob: float
    temporal_qkv: nn.Linear
    temporal_out: nn.Linear
    spatial_qkv: nn.Linear
    spatial_out: nn.Linear
    axis_gate: _TokenGate
    scale_gate: _TokenGate
    def __init__(self, embed_dim: int, num_heads: int, gate_reduction: int, activation, att_drop_prob: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.att_drop_prob = att_drop_prob
        self.temporal_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.add_module("temporal_qkv", self.temporal_qkv)
        self.temporal_out = nn.Linear(embed_dim, embed_dim)
        self.add_module("temporal_out", self.temporal_out)
        self.spatial_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.add_module("spatial_qkv", self.spatial_qkv)
        self.spatial_out = nn.Linear(embed_dim, embed_dim)
        self.add_module("spatial_out", self.spatial_out)
        self.axis_gate = _TokenGate(embed_dim, gate_reduction, activation, [0.5, 0.0])
        self.add_module("axis_gate", self.axis_gate)
        self.scale_gate = _TokenGate(embed_dim, gate_reduction, activation, [0.0, 1.0])
        self.add_module("scale_gate", self.scale_gate)
    def _qkv(self, p: Tensor) -> List[Tensor]:
        return p.reshape(p.shape[0], p.shape[1], 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    def _attend(self, p: Tensor, mask: Optional[Tensor]) -> Tensor:
        h = self._qkv(p)
        d = self.att_drop_prob if self.training else 0.0
        a = F.scaled_dot_product_attention(h[0], h[1], h[2], attn_mask=mask, dropout_p=d)
        return a.transpose(1, 2).reshape(p.shape[0], p.shape[1], -1)
    def _two_scales(self, p: Tensor, band: Tensor, w: Tensor) -> Tensor:
        n, L = p.shape[0], p.shape[1]
        h = self._qkv(p)
        s = torch.matmul(h[0], h[1].transpose(-2, -1)) * (float(self.head_dim) ** -0.5)
        w = w.reshape(n, 1, L, 2)
        pr = w[..., 0:1] * s.masked_fill(band.logical_not(), -math.inf).softmax(-1) + w[..., 1:2] * s.softmax(-1)
        return torch.matmul(pr, h[2]).transpose(1, 2).reshape(n, L, -1)
    def attend(self, x: Tensor, band: Optional[Tensor]) -> Tensor:
        B, C, P, D = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        aw = self.axis_gate(x)
        sw = self.scale_gate(x)
        pc = self.temporal_qkv(x.reshape(B * C, P, D))
        drop = self.training and self.att_drop_prob > 0
        temporal: Tensor
        if band is not None and not drop:
            temporal = self.temporal_out(self._two_scales(pc, band.__val__(), sw)).reshape(B, C, P, D)
        else:
            r = self.temporal_out(self._attend(pc, band)).reshape(B, C, P, D)
            f = r
            if drop:
                f = self.temporal_out(self._attend(pc, None)).reshape(B, C, P, D)
            temporal = sw[..., 0:1] * r + sw[..., 1:2] * f
        ps = self.spatial_qkv(x.transpose(1, 2).reshape(B * P, C, D))
        spatial = self.spatial_out(self._attend(ps, None)).reshape(B, P, C, D).transpose(1, 2)
        return aw[..., 0:1] * temporal + aw[..., 1:2] * spatial
    def _get_name(self) -> str:
        return "_AxisAttention"

class _AXONBlock(nn.Module):
    norm_attn: nn.RMSNorm
    attn: _AxisAttention
    norm_ffn: nn.RMSNorm
    ffn: _GatedFeedForward
    def __init__(self, embed_dim: int, num_heads: int, ffn_expansion: int, gate_reduction: int, activation,
                 att_drop_prob: float):
        super().__init__()
        self.norm_attn = nn.RMSNorm(embed_dim, eps=1e-8)
        self.add_module("norm_attn", self.norm_attn)
        self.attn = _AxisAttention(embed_dim, num_heads, gate_reduction, activation, att_drop_prob)
        self.add_module("attn", self.attn)
        self.norm_ffn = nn.RMSNorm(embed_dim, eps=1e-8)
        self.add_module("norm_ffn", self.norm_ffn)
        self.ffn = _GatedFeedForward(embed_dim, ffn_expansion)
        self.add_module("ffn", self.ffn)
    def block(self, x: Tensor, band: Optional[Tensor]) -> Tensor:
        x = x + self.attn.attend(self.norm_attn(x), band)
        return x + self.ffn(self.norm_ffn(x))
    def _get_name(self) -> str:
        return "_AXONBlock"

class _AXONEncoder(nn.Module):
    patch_size: int
    patch_stride: int
    embed_dim: int
    temporal_window: int
    normalize_input: bool
    channel_positions: Tensor
    patch_embed: nn.Linear
    spatial_pos_embed: nn.Sequential
    blocks: nn.ModuleList
    _blocks: List[_AXONBlock]
    final_norm: nn.RMSNorm
    def __init__(self, channel_positions: Tensor, patch_size: int, patch_stride: int, embed_dim: int, depth: int,
                 num_heads: int, ffn_expansion: int, temporal_window: int, gate_reduction: int, normalize_input: bool,
                 activation, att_drop_prob: float):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.temporal_window = temporal_window
        self.normalize_input = normalize_input
        self.channel_positions = channel_positions
        self.register_buffer("channel_positions", channel_positions, persistent=False)
        self.patch_embed = nn.Linear(patch_size, embed_dim, bias=False)
        self.add_module("patch_embed", self.patch_embed)
        self.spatial_pos_embed = nn.Sequential(nn.Linear(3, embed_dim), activation(), nn.LayerNorm(embed_dim))
        self.add_module("spatial_pos_embed", self.spatial_pos_embed)
        self._blocks = [_AXONBlock(embed_dim, num_heads, ffn_expansion, gate_reduction, activation, att_drop_prob)
                        for _ in range(depth)]
        self.blocks = nn.ModuleList()
        for b in self._blocks:
            self.blocks.append(b)
        self.add_module("blocks", self.blocks)
        self.final_norm = nn.RMSNorm(embed_dim, eps=1e-8)
        self.add_module("final_norm", self.final_norm)
    def forward(self, x: Tensor) -> Tensor:
        if self.normalize_input:
            m = x.mean(dim=-1, keepdim=True)
            s = x.std(dim=-1, keepdim=True)
            x = (x - m) / torch.clamp(s, min=1e-12)
        tok = self.patch_embed(x.unfold(-1, self.patch_size, self.patch_stride))
        P = tok.shape[2]
        sp = self.spatial_pos_embed(self.channel_positions)
        tm = sinusoidal_positional_encoding(P, self.embed_dim).to(sp.dtype)
        tok = tok + (sp.unsqueeze(1) + tm.unsqueeze(0)).unsqueeze(0).to(tok.dtype)
        band: Optional[Tensor] = None
        if P > self.temporal_window + 1:
            vals = [abs(i - j) <= self.temporal_window for i in range(P) for j in range(P)]
            band = torch.tensor(vals).reshape(P, P)
        for b in self._blocks:
            tok = b.block(tok, band)
        return self.final_norm(tok)
    def _get_name(self) -> str:
        return "_AXONEncoder"

class AXON(EEGModuleMixin, nn.Module):
    """AXON from Ahmed et al (2026). Architecture only (no Hub weights).

    braindecode reads 3D electrode positions from ``chs_info[i]["loc"]``;
    here they are passed as ``channel_positions`` (n_chans, 3) in metres
    (scaled by 100 internally as in braindecode)."""
    embed_dim: int
    head_hidden_dim: int
    encoder: _AXONEncoder
    final_layer: nn.Sequential

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, patch_size: int = 200,
                 patch_stride: int = 180, embed_dim: int = 512, depth: int = 22, num_heads: int = 8,
                 ffn_expansion: int = 4, temporal_window: int = 5, gate_reduction: int = 4, head_hidden_dim: int = 128,
                 normalize_input: bool = True, activation = nn.GELU, head_activation = nn.ELU, drop_prob: float = 0.3,
                 att_drop_prob: float = 0.0, channel_positions: Optional[Tensor] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be even.")
        if self._n_times is not None and self.n_times < patch_size:
            raise ValueError(f"n_times ({self.n_times}) must be >= patch_size ({patch_size}).")
        if channel_positions is None:
            raise ValueError("AXON needs a 3D position for every channel: pass channel_positions (n_chans, 3).")
        pos = channel_positions.__val__().to(torch.float32)
        if pos.ndim != 2 or pos.shape[0] != self.n_chans or pos.shape[1] != 3:
            raise ValueError(f"channel_positions must have shape ({self.n_chans}, 3).")
        self.embed_dim = embed_dim
        self.head_hidden_dim = head_hidden_dim
        self.encoder = _AXONEncoder(100.0 * pos, patch_size, patch_stride, embed_dim, depth, num_heads, ffn_expansion,
                                    temporal_window, gate_reduction, normalize_input, activation, att_drop_prob)
        self.add_module("encoder", self.encoder)
        self.final_layer = nn.Sequential(nn.Linear(embed_dim, head_hidden_dim), head_activation(), nn.Dropout(drop_prob),
                                         nn.Linear(head_hidden_dim, self.n_outputs))
        self.add_module("final_layer", self.final_layer)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.encoder(x).mean(dim=[1, 2]))

    def _get_name(self) -> str:
        return "AXON"
