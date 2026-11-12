# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import apply_rotary, rotary_angles, rotary_frequencies
from braindecode.functional.initialization import _param_by_name
from braindecode.functional.patches import _check_on_non_divisible, patch_count, pad_or_crop_to_patches

def _barista_init(m):
    if m._get_name() == "Conv2d":
        nn.init.kaiming_normal_(_param_by_name(m, "weight").__val__())
        b = _param_by_name(m, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())

class GatedLinearUnit(nn.Module):
    activation: nn.Module
    def __init__(self, activation = nn.GELU):
        super().__init__()
        self.activation = activation()
        self.add_module("activation", self.activation)
    def forward(self, x: Tensor) -> Tensor:
        c = x.chunk(2, dim=-1)
        return c[0] * self.activation(c[1])
    def _get_name(self) -> str:
        return "GatedLinearUnit"

class _SpatialEmbedding(nn.Module):
    n_dims: int
    default_indices: Optional[Tensor]
    tables: nn.ModuleList
    _tables: List[nn.Embedding]
    def __init__(self, d_model: int, n_slots: int, n_dims: int, padding_idx: Optional[int], default_indices: Optional[Tensor]):
        super().__init__()
        self.n_dims = n_dims
        self.default_indices = default_indices
        self.register_buffer("default_indices", default_indices, persistent=False)
        self.tables = nn.ModuleList()
        self._tables = []
        for _ in range(n_dims):
            e = nn.Embedding(n_slots, d_model, padding_idx=padding_idx)
            self._tables.append(e)
            self.tables.append(e)
        self.add_module("tables", self.tables)
    def embed(self, indices: Optional[Tensor]) -> Tensor:
        idx: Tensor
        if indices is None:
            if self.default_indices is None:
                raise ValueError("No spatial indices available: pass spatial_indices to forward_indices or the constructor.")
            idx = self.default_indices.__val__()
        else:
            idx = indices.__val__()
        g = idx.unsqueeze(-1) if self.n_dims == 1 else idx
        out = self._tables[0](g[..., 0])
        for d in range(1, self.n_dims):
            out = out + self._tables[d](g[..., d])
        return out
    def _get_name(self) -> str:
        return "_SpatialEmbedding"

class _SamePadConv(nn.Module):
    conv: nn.Conv2d
    remove: int
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        rf = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, rf // 2), dilation=(1, dilation))
        self.add_module("conv", self.conv)
        self.remove = 1 - rf % 2
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.remove > 0:
            x = x[..., :x.shape[-1] - self.remove]
        return x
    def _get_name(self) -> str:
        return "_SamePadConv"

class _DilatedConvBlock(nn.Module):
    conv1: _SamePadConv
    conv2: _SamePadConv
    norm: nn.LayerNorm
    activation: nn.GELU
    projector: nn.Module
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, norm_size: int, final: bool):
        super().__init__()
        self.conv1 = _SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self.add_module("conv1", self.conv1)
        self.conv2 = _SamePadConv(out_channels, out_channels, kernel_size, dilation)
        self.add_module("conv2", self.conv2)
        self.norm = nn.LayerNorm(norm_size, elementwise_affine=False)
        self.add_module("norm", self.norm)
        self.activation = nn.GELU()
        self.add_module("activation", self.activation)
        p: nn.Module = nn.Identity()
        if in_channels != out_channels or final:
            p = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.projector = p
        self.add_module("projector", p)
    def forward(self, x: Tensor) -> Tensor:
        r = self.projector(x)
        x = self.activation(self.norm(self.conv1(x)))
        return self.activation(self.norm(self.conv2(x))) + r
    def _get_name(self) -> str:
        return "_DilatedConvBlock"

class _RotarySelfAttention(nn.Module):
    num_heads: int
    qkv_proj: nn.Linear
    o_proj: nn.Linear
    dropout: nn.Dropout
    def __init__(self, d_model: int, num_heads: int, drop_prob: float):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.add_module("qkv_proj", self.qkv_proj)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)
        self.add_module("o_proj", self.o_proj)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
    def attn(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        c = self.qkv_proj(x).chunk(3, dim=-1)
        q = apply_rotary(c[0].unflatten(-1, [self.num_heads, -1]).transpose(1, 2), cos, sin, False)
        k = apply_rotary(c[1].unflatten(-1, [self.num_heads, -1]).transpose(1, 2), cos, sin, False)
        v = c[2].unflatten(-1, [self.num_heads, -1]).transpose(1, 2)
        a = self.dropout(F.scaled_dot_product_attention(q, k, v))
        return self.o_proj(a.transpose(1, 2).flatten(2))
    def _get_name(self) -> str:
        return "_RotarySelfAttention"

class _BaristaLayer(nn.Module):
    norm1: nn.RMSNorm
    self_attn: _RotarySelfAttention
    dropout: nn.Dropout
    norm2: nn.RMSNorm
    mlp: nn.Sequential
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int, drop_prob: float, activation):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-8)
        self.add_module("norm1", self.norm1)
        self.self_attn = _RotarySelfAttention(d_model, num_heads, drop_prob)
        self.add_module("self_attn", self.self_attn)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-8)
        self.add_module("norm2", self.norm2)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * mlp_ratio * d_model), GatedLinearUnit(activation),
                                 nn.Dropout(drop_prob), nn.Linear(mlp_ratio * d_model, d_model), nn.Dropout(drop_prob))
        self.add_module("mlp", self.mlp)
    def layer(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.dropout(self.self_attn.attn(self.norm1(x), cos, sin))
        return x + self.mlp(self.norm2(x))
    def _get_name(self) -> str:
        return "_TransformerEncoderLayer"

class _BaristaTransformer(nn.Module):
    inv_freq: Tensor
    layers: nn.ModuleList
    _layers: List[_BaristaLayer]
    norm: nn.RMSNorm
    def __init__(self, d_model: int, n_layers: int, num_heads: int, mlp_ratio: int, drop_prob: float, activation):
        super().__init__()
        self.inv_freq = rotary_frequencies(d_model // num_heads)
        self.register_buffer("inv_freq", self.inv_freq, persistent=False)
        self._layers = [_BaristaLayer(d_model, num_heads, mlp_ratio, drop_prob, activation) for _ in range(n_layers)]
        self.layers = nn.ModuleList()
        for l in self._layers:
            self.layers.append(l)
        self.add_module("layers", self.layers)
        self.norm = nn.RMSNorm(d_model, eps=1e-8)
        self.add_module("norm", self.norm)
    def encode(self, x: Tensor, n_chans: int) -> Tensor:
        np_ = x.shape[1] // n_chans
        ang = torch.repeat_interleave(rotary_angles(torch.arange(np_).to(torch.float32), self.inv_freq, False), n_chans, dim=0)
        cos = ang.cos().unsqueeze(0).unsqueeze(0)
        sin = ang.sin().unsqueeze(0).unsqueeze(0)
        for l in self._layers:
            x = l.layer(x, cos, sin)
        return self.norm(x)
    def _get_name(self) -> str:
        return "_Transformer"

class BaRISTA(EEGModuleMixin, nn.Module):
    """BaRISTA from Oganesian et al (2025). Architecture only (no Hub weights).

    Spatial indices must be given explicitly (constructor ``spatial_indices``
    or ``forward_indices``): deriving coordinate bins from ``chs_info``
    positions is not ported. ``ch_mask`` is not ported."""
    spatial_scale: str
    coord_bins: int
    patch_size: int
    on_non_divisible: str
    d_model: int
    pooling: str
    n_patches: int
    n_tokens: int
    temporal_encoder: nn.Sequential
    temporal_pooler: nn.Linear
    spatial_emb: Optional[_SpatialEmbedding]
    backbone: _BaristaTransformer
    token_pooling: Optional[nn.Linear]
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 spatial_scale: str = "coords", spatial_indices: Optional[Tensor] = None, coord_bins: int = 200,
                 patch_size: int = 512, on_non_divisible: str = "pad", d_model: int = 64, n_layers: int = 12,
                 num_heads: int = 4, mlp_ratio: int = 4, cnn_depth: int = 4, cnn_channels: int = 5,
                 cnn_kernel_size: int = 3, pooling: str = "learned", drop_prob: float = 0.1, activation = nn.GELU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if spatial_scale not in ("coords", "parcels", "lobes", "none"):
            raise ValueError(f"spatial_scale must be 'coords', 'parcels', 'lobes' or 'none', got {spatial_scale!r}.")
        if pooling not in ("learned", "mean"):
            raise ValueError(f"pooling must be 'learned' or 'mean', got {pooling!r}.")
        if cnn_depth < 0:
            raise ValueError(f"cnn_depth must be non-negative, got {cnn_depth}.")
        _check_on_non_divisible(on_non_divisible)
        if d_model % num_heads or (d_model // num_heads) % 2:
            raise ValueError(f"d_model ({d_model}) must split into num_heads ({num_heads}) heads of even width.")
        self.spatial_scale = spatial_scale
        self.coord_bins = coord_bins
        self.patch_size = patch_size
        self.on_non_divisible = on_non_divisible
        self.d_model = d_model
        self.pooling = pooling
        self.n_patches = patch_count(self.n_times, patch_size, on_non_divisible)
        chs = [1] + [cnn_channels] * cnn_depth + [1]
        self.temporal_encoder = nn.Sequential()
        for i in range(len(chs) - 1):
            self.temporal_encoder.add_module(str(i), _DilatedConvBlock(chs[i], chs[i + 1], cnn_kernel_size, 1 << i,
                                                                       patch_size, i == cnn_depth))
        self.add_module("temporal_encoder", self.temporal_encoder)
        self.temporal_pooler = nn.Linear(patch_size, d_model, bias=False)
        self.add_module("temporal_pooler", self.temporal_pooler)
        self.spatial_emb = None
        if spatial_scale != "none":
            co = spatial_scale == "coords"
            ns = coord_bins if co else (121 if spatial_scale == "parcels" else 21)
            idx: Optional[Tensor] = None
            if spatial_indices is not None:
                t = spatial_indices.__val__().long()
                if co and (t.ndim != 2 or t.shape[0] != self.n_chans or t.shape[1] != 3):
                    raise ValueError(f"spatial_indices must have shape ({self.n_chans}, 3).")
                if not co and (t.ndim != 1 or t.shape[0] != self.n_chans):
                    raise ValueError(f"spatial_indices must have shape ({self.n_chans},).")
                if t.numel() > 0 and (int(t.min().item()) < 0 or int(t.max().item()) >= ns):
                    raise ValueError(f"spatial_indices must be integers in [0, {ns}).")
                idx = t
            pi: Optional[int] = None
            if not co:
                pi = 0
            se = _SpatialEmbedding(d_model, ns, 3 if co else 1, pi, idx)
            self.spatial_emb = se
            self.add_module("spatial_emb", se)
        self.backbone = _BaristaTransformer(d_model, n_layers, num_heads, mlp_ratio, drop_prob, activation)
        self.add_module("backbone", self.backbone)
        self.n_tokens = self.n_patches * self.n_chans
        self.token_pooling = None
        if pooling == "learned":
            tp = nn.Linear(self.n_tokens, 1, bias=False)
            self.token_pooling = tp
            self.add_module("token_pooling", tp)
        self.final_layer = nn.Linear(d_model, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        self.apply(_barista_init)

    def forward_indices(self, x: Tensor, spatial_indices: Optional[Tensor]) -> Tensor:
        B, C = x.shape[0], x.shape[1]
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        p = x.reshape(B, C, -1, self.patch_size).transpose(1, 2)
        NP = p.shape[1]
        p = p.reshape(B, 1, NP * C, self.patch_size)
        tok = self.temporal_pooler(self.temporal_encoder(p).squeeze(1))
        if self.spatial_emb is not None:
            sp = self.spatial_emb.__val__().embed(spatial_indices)
            if sp.shape[-2] != C:
                raise ValueError(f"Spatial embedding covers {sp.shape[-2]} channels, input has {C}.")
            if sp.dim() == 2:
                tok = tok + sp.repeat(NP, 1).unsqueeze(0)
            else:
                tok = tok + sp.repeat(1, NP, 1)
        lat = self.backbone.encode(tok, C)
        f: Tensor
        if self.token_pooling is not None:
            f = self.token_pooling.__val__()(lat.transpose(1, 2)).squeeze(-1)
        else:
            f = lat.mean(dim=1)
        return self.final_layer(f)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_indices(x, None)

    def _get_name(self) -> str:
        return "BaRISTA"
