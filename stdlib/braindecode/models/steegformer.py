# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import sinusoidal_positional_encoding
from braindecode.functional.initialization import _param_by_name
from braindecode.functional.patches import _check_on_non_divisible, pad_or_crop_to_patches
from braindecode.modules import FeedForwardBlock, MultiHeadAttention, Residual

def _steeg_init(m):
    n = m._get_name()
    if n == "Linear":
        nn.init.xavier_uniform_(_param_by_name(m, "weight").__val__())
        b = _param_by_name(m, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())
    elif n == "LayerNorm":
        w = _param_by_name(m, "weight")
        if w is not None:
            nn.init.ones_(w.__val__())
        b = _param_by_name(m, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())

class _TemporalPositionalEncoding(nn.Module):
    max_len: int
    pe: Tensor
    def __init__(self, embed_dim: int, max_len: int = 2048):
        super().__init__()
        self.max_len = max_len
        self.pe = sinusoidal_positional_encoding(max_len, embed_dim)
        self.register_buffer("pe", self.pe, persistent=False)
    def cls_token_encoding(self) -> Tensor:
        return self.pe[0]
    def encode(self, seq: int) -> Tensor:
        if seq + 1 > self.max_len:
            raise ValueError(f"Sequence of {seq} patches exceeds the positional table ({self.max_len - 1}).")
        return self.pe[1:seq + 1].unsqueeze(0).unsqueeze(2)
    def _get_name(self) -> str:
        return "_TemporalPositionalEncoding"

class _ChannelPositionalEmbed(nn.Module):
    embedding: nn.Embedding
    def __init__(self, n_vocab: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.add_module("embedding", self.embedding)
        nn.init.zeros_(self.embedding.weight)
    def forward(self, idx: Tensor) -> Tensor:
        return self.embedding(idx).unsqueeze(0).unsqueeze(0)
    def _get_name(self) -> str:
        return "_ChannelPositionalEmbed"

def _encoder_block(embed_dim: int, num_heads: int, drop_prob: float, mlp_ratio: int, drop_path: float, activation) -> nn.Sequential:
    return nn.Sequential(
        Residual(nn.Sequential(nn.LayerNorm(embed_dim, eps=1e-6), MultiHeadAttention(embed_dim, num_heads, drop_prob),
                               nn.Dropout(drop_prob)), drop_path),
        Residual(nn.Sequential(nn.LayerNorm(embed_dim, eps=1e-6),
                               FeedForwardBlock(embed_dim, mlp_ratio, drop_prob, activation), nn.Dropout(drop_prob)),
                 drop_path))

class STEEGFormer(EEGModuleMixin, nn.Module):
    """ST-EEGFormer from Gao et al (2025). Architecture only (no Hub weights).

    The channel vocabulary is downloaded from the Hub in braindecode, so
    name-based channel resolution is not ported: pass ``chan_pos_idx``
    explicitly, otherwise the identity mapping is used (as braindecode does
    when the vocabulary is unavailable)."""
    patch_size: int
    on_non_divisible: str
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: int
    global_pool: str
    n_chans_pos: int
    channel_indices: Tensor
    cls_token: Tensor
    patch_embed: nn.Module
    _proj: nn.Linear
    temporal_pos: _TemporalPositionalEncoding
    channel_pos: _ChannelPositionalEmbed
    pos_drop: nn.Dropout
    encoder: nn.Sequential
    norm: nn.LayerNorm
    final_layer: nn.Linear
    head_drop: nn.Dropout

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, patch_size: int = 16,
                 embed_dim: int = 512, depth: int = 8, num_heads: int = 8, mlp_ratio: int = 4, drop_prob: float = 0.0,
                 drop_path: float = 0.0, activation = nn.GELU, global_pool: str = "avg", n_chans_pos: int = 145,
                 chan_pos_idx: Optional[List[int]] = None, on_non_divisible: str = "pad"):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if global_pool not in ("avg", "cls"):
            raise ValueError(f"global_pool must be 'avg' or 'cls', got {global_pool!r}.")
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}.")
        _check_on_non_divisible(on_non_divisible)
        self.patch_size = patch_size
        self.on_non_divisible = on_non_divisible
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.global_pool = global_pool
        self.n_chans_pos = n_chans_pos
        idx: Tensor
        if chan_pos_idx is not None:
            idx = torch.tensor(chan_pos_idx.__val__(), dtype=torch.long)
        else:
            if self.n_chans > n_chans_pos:
                raise ValueError(f"STEEGFormer: n_chans={self.n_chans} exceeds n_chans_pos={n_chans_pos}; pass chan_pos_idx.")
            idx = torch.arange(self.n_chans)
        if idx.numel() != self.n_chans:
            raise ValueError(f"chan_pos_idx must have {self.n_chans} entries, got {idx.numel()}.")
        if idx.numel() > 0 and (int(idx.max().item()) >= n_chans_pos or int(idx.min().item()) < 0):
            raise ValueError(f"chan_pos_idx entries must be in [0, {n_chans_pos}).")
        self.channel_indices = idx
        self.register_buffer("channel_indices", idx, persistent=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_parameter("cls_token", self.cls_token)
        self._proj = nn.Linear(patch_size, embed_dim)
        pe = nn.Module()
        pe.add_module("proj", self._proj)
        self.patch_embed = pe
        self.add_module("patch_embed", pe)
        self.temporal_pos = _TemporalPositionalEncoding(embed_dim)
        self.add_module("temporal_pos", self.temporal_pos)
        self.channel_pos = _ChannelPositionalEmbed(n_chans_pos, embed_dim)
        self.add_module("channel_pos", self.channel_pos)
        self.pos_drop = nn.Dropout(drop_prob)
        self.add_module("pos_drop", self.pos_drop)
        dpr = torch.linspace(0.0, drop_path, depth).tolist()
        self.encoder = nn.Sequential()
        for i in range(depth):
            self.encoder.add_module(str(i), _encoder_block(embed_dim, num_heads, drop_prob, mlp_ratio, float(dpr[i]), activation))
        self.add_module("encoder", self.encoder)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.add_module("norm", self.norm)
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        self.head_drop = nn.Dropout(drop_prob)
        self.add_module("head_drop", self.head_drop)
        self.apply(_steeg_init)
        nn.init.normal_(self.cls_token, std=0.02)

    def _tokens(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.n_chans:
            raise ValueError(f"Expected {self.n_chans} channels, got {x.shape[1]}.")
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        t = self._proj(x.reshape(x.shape[0], x.shape[1], -1, self.patch_size)).transpose(1, 2)
        seq = t.shape[1]
        t = t + self.temporal_pos.encode(seq) + self.channel_pos(self.channel_indices)
        t = t.flatten(1, 2)
        cls = (self.cls_token + self.temporal_pos.cls_token_encoding()).expand(t.shape[0], -1, -1)
        return self.encoder(self.pos_drop(torch.cat([cls, t], dim=1)))

    def forward_features(self, x: Tensor) -> Tensor:
        return self.norm(self._tokens(x))

    def forward(self, x: Tensor) -> Tensor:
        h = self._tokens(x)
        if self.global_pool == "avg":
            h = h[:, 1:].mean(dim=1)
        else:
            h = self.norm(h)[:, 0]
        return self.final_layer(self.head_drop(h))

    def _get_name(self) -> str:
        return "STEEGFormer"
