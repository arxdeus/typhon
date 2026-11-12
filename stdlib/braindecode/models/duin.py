# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import sinusoidal_positional_encoding
from braindecode.functional.initialization import _param_by_name
from braindecode.functional.patches import _check_on_non_divisible, patch_count, pad_or_crop_to_patches
from braindecode.modules import FeedForwardBlock, MultiHeadAttention

def _trunc_linear(scale: float):
    def f(m):
        if m._get_name() == "Linear":
            w = _param_by_name(m, "weight").__val__()
            nn.init.trunc_normal_(w, std=0.02)
            if scale != 1.0:
                with torch.no_grad():
                    w.div_(scale)
            b = _param_by_name(m, "bias")
            if b is not None:
                nn.init.zeros_(b.__val__())
    return f

class _DuINTransformerLayer(nn.Module):
    attn: MultiHeadAttention
    norm_attn: nn.LayerNorm
    ffn: FeedForwardBlock
    norm_ffn: nn.LayerNorm
    def __init__(self, embed_dim: int, n_heads: int, head_dim: int, ffn_dim: int, activation, attn_drop_prob: float,
                 drop_prob: float):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, n_heads, attn_drop_prob, head_dim=head_dim, qk_norm=True)
        self.add_module("attn", self.attn)
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.add_module("norm_attn", self.norm_attn)
        self.ffn = FeedForwardBlock(embed_dim, 1, drop_prob, activation, hidden_features=ffn_dim)
        self.add_module("ffn", self.ffn)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.add_module("norm_ffn", self.norm_ffn)
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_attn(x + self.attn(x))
        return self.norm_ffn(x + self.ffn(x))
    def _get_name(self) -> str:
        return "_DuINTransformerLayer"

class DuIN(EEGModuleMixin, nn.Module):
    """Du-IN from Zheng et al (2024). Architecture only (no Hub weights)."""
    n_patches: int
    patch_size: int
    on_non_divisible: str
    embed_dim: int
    spatial_projection: nn.Linear
    spatial_encoder: nn.Sequential
    time_embedding: Tensor
    encoder: nn.Sequential
    head_hidden: nn.Module
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, patch_size: int = 100,
                 on_non_divisible: str = "pad", n_neural: int = 16, n_filters: List[int] = [128, 128, 16],
                 kernel_sizes: List[int] = [19, 3, 3], strides: List[int] = [10, 1, 1], n_layers: int = 8,
                 n_heads: int = 8, head_dim: int = 64, ffn_dim: int = 320, head_hidden_dim: int = 128,
                 activation = nn.ReLU, attn_drop_prob: float = 0.2, drop_prob: float = 0.2):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        prod = 1
        for s in strides:
            if s < 1:
                raise ValueError("strides must be positive.")
            prod *= s
        if patch_size < 1:
            raise ValueError("patch_size must be positive.")
        if patch_size % prod:
            raise ValueError(f"patch_size ({patch_size}) must be divisible by the product of strides ({prod}).")
        if not (len(n_filters) == len(kernel_sizes) == len(strides)):
            raise ValueError("n_filters, kernel_sizes and strides must have the same length.")
        _check_on_non_divisible(on_non_divisible)
        self.patch_size = patch_size
        self.n_patches = patch_count(self.n_times, patch_size, on_non_divisible)
        self.on_non_divisible = on_non_divisible
        self.embed_dim = n_filters[-1] * (patch_size // prod)
        self.spatial_projection = nn.Linear(self.n_chans, n_neural)
        self.add_module("spatial_projection", self.spatial_projection)
        self.spatial_encoder = nn.Sequential()
        ic = n_neural
        for i in range(len(n_filters)):
            oc, k, s = n_filters[i], kernel_sizes[i], strides[i]
            conv: nn.Conv1d
            if s == 1:
                conv = nn.Conv1d(ic, oc, k, s, padding="same")
            else:
                conv = nn.Conv1d(ic, oc, k, s, (k - 1) // 2)
            self.spatial_encoder.add_module(str(i), nn.Sequential(conv, nn.BatchNorm1d(oc)))
            ic = oc
        self.add_module("spatial_encoder", self.spatial_encoder)
        self.time_embedding = sinusoidal_positional_encoding(self.n_patches, self.embed_dim)
        self.register_buffer("time_embedding", self.time_embedding, persistent=False)
        self.encoder = nn.Sequential()
        layers = List[_DuINTransformerLayer]()
        for i in range(n_layers):
            l = _DuINTransformerLayer(self.embed_dim, n_heads, head_dim, ffn_dim, activation, attn_drop_prob, drop_prob)
            layers.append(l)
            self.encoder.add_module(str(i), l)
        self.add_module("encoder", self.encoder)
        self.add_module("flatten", nn.Flatten(1))
        nf = self.n_patches * self.embed_dim
        hh: nn.Module = nn.Identity()
        if head_hidden_dim:
            hh = nn.Sequential(nn.Linear(nf, head_hidden_dim), activation())
            nf = head_hidden_dim
        self.head_hidden = hh
        self.add_module("head_hidden", hh)
        self.final_layer = nn.Linear(nf, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        self.spatial_projection.apply(_trunc_linear(1.0))
        for i in range(len(layers)):
            layers[i].apply(_trunc_linear(math.sqrt(2.0 * float(i + 1))))
        self.head_hidden.apply(_trunc_linear(1.0))
        self.final_layer.apply(_trunc_linear(1.0))

    def forward_features(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.n_times:
            raise ValueError(f"DuIN expects {self.n_times} samples, got {x.shape[-1]}.")
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        h = self.spatial_projection(x.transpose(1, 2)).transpose(1, 2)
        B, Nn = h.shape[0], h.shape[1]
        p = h.reshape(B, Nn, -1, self.patch_size).permute(0, 2, 1, 3).reshape(-1, Nn, self.patch_size)
        e = self.spatial_encoder(p)
        tok = e.permute(0, 2, 1).reshape(B, self.n_patches, -1)
        return self.encoder(tok + self.time_embedding.to(tok.dtype)).flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.head_hidden(self.forward_features(x)))

    def _get_name(self) -> str:
        return "DuIN"
