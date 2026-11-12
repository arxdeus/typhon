# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional.patches import _check_on_non_divisible, patch_count, pad_or_crop_to_patches

class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        c = x.chunk(2, dim=-1)
        return F.gelu(c[1]) * c[0]
    def _get_name(self) -> str:
        return "GEGLU"

class FeedForward(nn.Module):
    geglu: bool
    net: nn.Sequential
    _norm: nn.RMSNorm
    _l1: nn.Linear
    _l2: nn.Linear
    def __init__(self, dim: int, hidden_dim: int, geglu: bool):
        super().__init__()
        self.geglu = geglu
        self._norm = nn.RMSNorm(dim, eps=1e-6)
        self._l1 = nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False)
        self._l2 = nn.Linear(hidden_dim, dim, bias=False)
        act: nn.Module = nn.GELU()
        if geglu:
            act = GEGLU()
        self.net = nn.Sequential(self._norm, self._l1, act, self._l2)
        self.add_module("net", self.net)
    def forward(self, x: Tensor) -> Tensor:
        if not self.geglu:
            return self.net(x)
        x = self._norm(x)
        w = self._l1.weight
        h = w.shape[0] // 2
        return self._l2(F.gelu(F.linear(x, w[h:])) * F.linear(x, w[:h]))
    def _get_name(self) -> str:
        return "FeedForward"

class Attention(nn.Module):
    heads: int
    norm: nn.RMSNorm
    to_qkv: nn.Linear
    to_out: nn.Linear
    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner = head_dim * heads
        self.heads = heads
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.add_module("norm", self.norm)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.add_module("to_qkv", self.to_qkv)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.add_module("to_out", self.to_out)
    def forward(self, x: Tensor) -> Tensor:
        c = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q = c[0].unflatten(-1, [self.heads, -1]).transpose(1, 2)
        k = c[1].unflatten(-1, [self.heads, -1]).transpose(1, 2)
        v = c[2].unflatten(-1, [self.heads, -1]).transpose(1, 2)
        return self.to_out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))
    def _get_name(self) -> str:
        return "Attention"

class TransformerBackbone(nn.Module):
    dim: int
    layers: nn.ModuleList
    _pairs: List[Tuple[Attention, FeedForward]]
    def __init__(self, dim: int, depth: int, heads: int, head_dim: int, mlp_dim: int, geglu: bool):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self._pairs = []
        for _ in range(depth):
            a = Attention(dim, heads, head_dim)
            f = FeedForward(dim, mlp_dim, geglu)
            ml = nn.ModuleList()
            ml.append(a)
            ml.append(f)
            self.layers.append(ml)
            self._pairs.append((a, f))
        self.add_module("layers", self.layers)
    def forward(self, x: Tensor) -> Tensor:
        for a, f in self._pairs:
            x = a(x) + x
            x = f(x) + x
        return x
    def forward_all(self, x: Tensor) -> List[Tensor]:
        out = [x]
        for a, f in self._pairs:
            x = a(x) + x
            x = f(x) + x
            out.append(x)
        return out
    def _get_name(self) -> str:
        return "TransformerBackbone"

class FourierEmb4D(nn.Module):
    dimension: int
    freqs: int
    increment_time: float
    margin: float
    def __init__(self, dimension: int, freqs: int, increment_time: float = 0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin
    def forward(self, positions_: Tensor) -> Tensor:
        p = positions_.float()
        p = torch.cat([p[..., :3], p[..., 3:] * self.increment_time], dim=-1) + self.margin
        f = self.freqs
        w = 1.0 + 2.0 * self.margin
        fw = torch.arange(f).to(torch.float32) * (2.0 * math.pi / w)
        lead = list(p.shape[:-1])
        pe = p.reshape(lead + [1, 1, 1, 1, 4])
        loc = pe[..., 0] * fw.view(f, 1, 1, 1) + pe[..., 1] * fw.view(f, 1, 1) + pe[..., 2] * fw.view(f, 1) + pe[..., 3] * fw
        loc = loc.reshape(lead + [-1])
        h = self.dimension // 2
        if loc.shape[-1] != h:
            if loc.shape[-1] > h:
                loc = loc[..., :h]
            else:
                raise ValueError(f"freqs={f} gives {loc.shape[-1]} features, fewer than dimension // 2 = {h}.")
        return torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1).to(positions_.dtype)
    @staticmethod
    def add_time_patch(pos: Tensor, num_patches: int) -> Tensor:
        b, c = pos.shape[0], pos.shape[1]
        pr = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)
        tv = torch.arange(0, num_patches, 1).float().view(1, 1, num_patches, 1).expand(b, c, num_patches, 1)
        return torch.cat([pr, tv], dim=-1).view(b, c * num_patches, 4)
    def _get_name(self) -> str:
        return "FourierEmb4D"

class REVE(EEGModuleMixin, nn.Module):
    """REVE from El Ouahidi et al (2025). Architecture only (no Hub weights).

    The REVE position bank is downloaded from the Hugging Face Hub in
    braindecode, so name -> 3D position lookup from ``chs_info`` is not
    ported: pass electrode positions explicitly, either per call with
    ``forward_pos(x, pos)`` (pos: (batch, n_chans, 3)) or once with the
    extra ``default_pos`` (n_chans, 3) constructor argument."""
    embed_dim: int
    freqs: int
    patch_size: int
    patch_overlap: int
    on_non_divisible: str
    depth: int
    heads: int
    head_dim: int
    use_attention_pooling: bool
    default_pos: Optional[Tensor]
    to_patch_embedding: nn.Sequential
    fourier4d: FourierEmb4D
    mlp4d: nn.Sequential
    ln: nn.LayerNorm
    transformer: TransformerBackbone
    final_layer: nn.Sequential
    cls_query_token: Optional[Tensor]

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, embed_dim: int = 512,
                 depth: int = 22, heads: int = 8, head_dim: int = 64, mlp_dim_ratio: float = 2.66,
                 use_geglu: bool = True, freqs: int = 4, patch_size: int = 200, patch_overlap: int = 20,
                 attention_pooling: bool = False, on_non_divisible: Optional[str] = None,
                 default_pos: Optional[Tensor] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.freqs = freqs
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        if patch_overlap >= patch_size:
            raise ValueError(f"patch_overlap ({patch_overlap}) must be smaller than patch_size ({patch_size}).")
        mode = "crop" if on_non_divisible is None else on_non_divisible.__val__()
        _check_on_non_divisible(mode)
        self.on_non_divisible = mode
        patch_count(self.n_times, patch_size, mode, patch_size - patch_overlap)
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.use_attention_pooling = attention_pooling
        self.default_pos = default_pos
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_size, embed_dim))
        self.add_module("to_patch_embedding", self.to_patch_embedding)
        self.fourier4d = FourierEmb4D(embed_dim, freqs)
        self.add_module("fourier4d", self.fourier4d)
        self.mlp4d = nn.Sequential(nn.Linear(4, embed_dim, bias=False), nn.GELU(), nn.LayerNorm(embed_dim))
        self.add_module("mlp4d", self.mlp4d)
        self.ln = nn.LayerNorm(embed_dim)
        self.add_module("ln", self.ln)
        self.transformer = TransformerBackbone(embed_dim, depth, heads, head_dim, int(float(embed_dim) * mlp_dim_ratio), use_geglu)
        self.add_module("transformer", self.transformer)
        if attention_pooling:
            self.final_layer = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, self.n_outputs))
        else:
            np_ = patch_count(self.n_times, patch_size, mode, patch_size - patch_overlap)
            fd = self.n_chans * np_ * embed_dim
            self.final_layer = nn.Sequential(nn.Flatten(), nn.LayerNorm(fd), nn.Linear(fd, self.n_outputs))
        self.add_module("final_layer", self.final_layer)
        self.cls_query_token = None
        if attention_pooling:
            t = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.cls_query_token = t
            self.register_parameter("cls_query_token", t)

    def _encode(self, eeg: Tensor, pos: Tensor) -> Tensor:
        step = self.patch_size - self.patch_overlap
        eeg = pad_or_crop_to_patches(eeg, self.patch_size, self.on_non_divisible, step)
        patches = eeg.unfold(2, self.patch_size, step)
        B, C, NP = patches.shape[0], patches.shape[1], patches.shape[2]
        p4 = FourierEmb4D.add_time_patch(pos, NP)
        pe = self.ln(self.fourier4d(p4).to(eeg.dtype) + self.mlp4d(p4.to(eeg.dtype)))
        x = self.to_patch_embedding(patches).flatten(1, 2) + pe
        return self.transformer(x).reshape(B, C, NP, self.embed_dim)

    def forward_pos(self, eeg: Tensor, pos: Tensor) -> Tensor:
        x = self._encode(eeg, pos)
        if self.use_attention_pooling:
            B = x.shape[0]
            x = x.flatten(1, 2)
            q = self.cls_query_token.__val__().expand(B, -1, -1)
            s = torch.matmul(q, x.transpose(-1, -2)) / (float(self.embed_dim) ** 0.5)
            x = torch.matmul(torch.softmax(s, dim=-1), x).squeeze(1)
        return self.final_layer(x)

    def forward(self, eeg: Tensor) -> Tensor:
        if self.default_pos is None:
            raise ValueError("REVE needs electrode positions: pass default_pos or call forward_pos(x, pos).")
        return self.forward_pos(eeg, self.default_pos.__val__().unsqueeze(0).expand(eeg.shape[0], -1, -1))

    def _get_name(self) -> str:
        return "REVE"
