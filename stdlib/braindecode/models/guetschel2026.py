# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# Guetschel2026 (EEG foundation model with masked pretexts). Typhon notes:
# - chs_info entries are Dict[str, str]; "loc" holds whitespace separated
#   floats in metres (MNE head frame); "coord_frame" if given must be "head"
#   or "4" (FIFFV_COORD_HEAD).
# - channel_strategy="native" only; return_features is not supported.

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import sinusoidal_positional_encoding

def _pos_encode_time(n_times: int, n_dim: int, max_n_times: int) -> Tensor:
    assert n_dim % 2 == 0
    return sinusoidal_positional_encoding(n_times, n_dim, base=float(max_n_times))

def _xyz_div_term(n_dim: int) -> Tensor:
    freqs = torch.arange(0, n_dim, 2)
    return torch.exp((1 - freqs / n_dim) * 2 * math.pi)

def _pos_encode_xyz(ch_pos: Tensor, x_min: float, x_max: float, div_term: Tensor) -> Tensor:
    xx = ((ch_pos - x_min) / (x_max - x_min)).unsqueeze(-1)
    s = torch.sin(xx * div_term)
    c = torch.cos(xx * div_term)
    shp = ch_pos.shape + [2 * div_term.shape[0]]
    return torch.stack([s, c], dim=-1).reshape(shp)

class _GaussianRandomProjection(nn.Module):
    n_features: int
    n_components: int
    projection: Tensor
    def __init__(self, n_features: int, n_components: int, seed: int = 0):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        g = torch.Generator()
        g.manual_seed(seed)
        p = torch.empty(n_components, n_features)
        p.normal_(0.0, 1.0 / math.sqrt(n_components), generator=g)
        self.projection = p
        self.register_buffer("projection", p)
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.projection)
    def _get_name(self) -> str:
        return "_GaussianRandomProjection"

class _LinearPatchEmbedding(nn.Module):
    patch_size: int
    patch_step: int
    linear: nn.Linear
    def __init__(self, embed_dim: int, patch_size: int, patch_overlap: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_step = patch_size - patch_overlap
        self.linear = nn.Linear(patch_size, embed_dim)
        self.add_module("linear", self.linear)
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.unfold(2, self.patch_size, self.patch_step))
    def _get_name(self) -> str:
        return "_LinearPatchEmbedding"

class _PositionalEncoder(nn.Module):
    coord_dim: int
    time_dim: int
    pos_half_range: float
    max_n_times: int
    ch_pos: Tensor
    div_term: Tensor
    encoding_time: Tensor
    def __init__(self, ch_pos: Tensor, embed_dim: int, pos_half_range: float, max_n_times: int):
        super().__init__()
        self.coord_dim = embed_dim // 4
        self.time_dim = embed_dim // 4
        self.pos_half_range = pos_half_range
        self.max_n_times = max_n_times
        self.ch_pos = ch_pos
        self.register_buffer("ch_pos", ch_pos, persistent=False)
        self.div_term = _xyz_div_term(self.coord_dim)
        self.register_buffer("div_term", self.div_term, persistent=False)
        self.encoding_time = _pos_encode_time(33, self.time_dim, max_n_times)
        self.register_buffer("encoding_time", self.encoding_time)
    def encode(self, n_patches: int, batch_size: int) -> Tensor:
        h = self.pos_half_range
        cp = self.ch_pos[None].expand(batch_size, -1, -1)
        spat = _pos_encode_xyz(cp, -h, h, self.div_term).flatten(-2)
        enc_t = self.encoding_time[:n_patches] if n_patches <= self.encoding_time.shape[0] else \
            _pos_encode_time(n_patches, self.time_dim, self.max_n_times).to(self.encoding_time.dtype)
        c = spat.shape[1]
        return torch.cat([spat[:, :, None].expand(-1, -1, n_patches, -1),
                          enc_t[None, None].expand(batch_size, c, -1, -1)], -1)
    def _get_name(self) -> str:
        return "_PositionalEncoder"

class _MultiHeadAttention(nn.Module):
    num_heads: int
    drop_prob: float
    packed_proj: nn.Linear
    out_proj: nn.Linear
    def __init__(self, embed_dim: int, num_heads: int, drop_prob: float):
        super().__init__()
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.packed_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.add_module("packed_proj", self.packed_proj)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.add_module("out_proj", self.out_proj)
    def forward(self, x: Tensor) -> Tensor:
        qkv = self.packed_proj(x).chunk(3, dim=-1)
        q = qkv[0].unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        k = qkv[1].unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        v = qkv[2].unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_prob if self.training else 0.0)
        return self.out_proj(a.transpose(1, 2).flatten(-2))
    def _get_name(self) -> str:
        return "_MultiHeadAttention"

class _TransformerEncoderLayer(nn.Module):
    self_attn: _MultiHeadAttention
    linear1: nn.Linear
    dropout: nn.Dropout
    linear2: nn.Linear
    norm1: nn.RMSNorm
    norm2: nn.RMSNorm
    dropout1: nn.Dropout
    dropout2: nn.Dropout
    activation: nn.Module
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int, drop_prob: float, activation):
        super().__init__()
        self.self_attn = _MultiHeadAttention(embed_dim, num_heads, drop_prob)
        self.add_module("self_attn", self.self_attn)
        self.linear1 = nn.Linear(embed_dim, 2 * dim_feedforward, bias=False)
        self.add_module("linear1", self.linear1)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim, bias=False)
        self.add_module("linear2", self.linear2)
        self.norm1 = nn.RMSNorm(embed_dim)
        self.add_module("norm1", self.norm1)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.add_module("norm2", self.norm2)
        self.dropout1 = nn.Dropout(drop_prob)
        self.add_module("dropout1", self.dropout1)
        self.dropout2 = nn.Dropout(drop_prob)
        self.add_module("dropout2", self.dropout2)
        self.activation = activation()
        self.add_module("activation", self.activation)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout1(self.self_attn(self.norm1(x)))
        gu = self.linear1(self.norm2(x)).chunk(2, dim=-1)
        return x + self.dropout2(self.linear2(self.dropout(self.activation(gu[0]) * gu[1])))
    def _get_name(self) -> str:
        return "_TransformerEncoderLayer"

class _TransformerEncoder(nn.Module):
    layers: nn.ModuleList
    _ls: List[_TransformerEncoderLayer]
    def __init__(self, depth: int, embed_dim: int, num_heads: int, dim_feedforward: int, drop_prob: float, activation):
        super().__init__()
        self._ls = [_TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, drop_prob, activation)
                    for _ in range(depth)]
        self.layers = nn.ModuleList()
        for l in self._ls:
            self.layers.append(l)
        self.add_module("layers", self.layers)
    def forward(self, z: Tensor) -> Tensor:
        c, t = z.shape[1], z.shape[2]
        z = z.flatten(1, 2)
        for l in self._ls:
            z = l(z)
        return z.unflatten(1, (c, t))
    def _get_name(self) -> str:
        return "_TransformerEncoder"

class _ContextualEncoder(nn.Module):
    pos_encoder: _PositionalEncoder
    transformer: _TransformerEncoder
    def __init__(self, pos_encoder: _PositionalEncoder, transformer: _TransformerEncoder):
        super().__init__()
        self.pos_encoder = pos_encoder
        self.add_module("pos_encoder", self.pos_encoder)
        self.transformer = transformer
        self.add_module("transformer", self.transformer)
    def forward(self, lf: Tensor) -> Tensor:
        pos = self.pos_encoder.encode(lf.shape[2], lf.shape[0])
        return self.transformer(lf + pos.to(lf.dtype))
    def _get_name(self) -> str:
        return "_ContextualEncoder"

class Guetschel2026(EEGModuleMixin, nn.Module):
    """EEG foundation model from Guetschel et al (2026)."""
    normalization: str
    input_scale: float
    clip_sigma: float
    patch_size: int
    patch_step: int
    embed_dim: int
    random_projection: Optional[int]
    random_projection_seed: int
    n_patches: int
    feature_encoder: _LinearPatchEmbedding
    model: _ContextualEncoder
    final_layer: nn.Sequential

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, embed_dim: int = 512,
                 depth: int = 4, num_heads: int = 8, dim_feedforward: int = 1365, patch_size: int = 200,
                 patch_overlap: int = 20, pos_half_range: float = 0.15, activation = nn.GELU, drop_prob: float = 0.0,
                 normalization: str = "median_std_clip", input_scale: float = 1000000.0, clip_sigma: float = 15.0,
                 random_projection: Optional[int] = None, random_projection_seed: int = 0,
                 channel_strategy: str = "native"):
        if channel_strategy != "native":
            raise NotImplementedError("Guetschel2026: only channel_strategy='native' is supported in Typhon")
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if normalization != "median_std_clip" and normalization != "none":
            raise ValueError(f"normalization must be one of ('median_std_clip', 'none'), got {normalization!r}.")
        if embed_dim % 8 != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 8: each of the four positional encodings needs an even width.")
        if num_heads < 1:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads!r}.")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        if not (math.isfinite(pos_half_range) and pos_half_range > 0):
            raise ValueError(f"pos_half_range must be a positive finite number, got {pos_half_range!r}.")
        if not (0 <= patch_overlap < patch_size):
            raise ValueError(f"patch_overlap ({patch_overlap}) must satisfy 0 <= patch_overlap < patch_size ({patch_size}).")
        if random_projection is not None and random_projection < 1:
            raise ValueError(f"random_projection must be None or an integer >= 1, got {random_projection!r}.")
        if self.n_times < patch_size:
            raise ValueError(f"n_times ({self.n_times}) must be at least patch_size ({patch_size}).")
        if self._sfreq is not None or self._input_window_seconds is not None:
            if abs(self.sfreq - 200.0) > 1e-6:
                print(f"UserWarning: Guetschel2026 was pretrained at 200 Hz, got sfreq={self.sfreq}. Resample the data to 200 Hz to use the pretrained weights.")
        if self._chs_info is None:
            raise ValueError("Guetschel2026 requires channel locations: a finite loc[:3] in metres for every channel.")
        pos = List[float]()
        allzero = True
        for ch in self.chs_info:
            if "loc" not in ch:
                raise ValueError("Guetschel2026 requires channel locations: a finite loc[:3] in metres for every channel.")
            parts = ch["loc"].split()
            if len(parts) < 3:
                raise ValueError("Guetschel2026 requires channel locations: a finite loc[:3] in metres for every channel.")
            for k in range(3):
                v = float(parts[k])
                if not math.isfinite(v):
                    raise ValueError("Guetschel2026 requires channel locations: a finite loc[:3] in metres for every channel.")
                if abs(v) > 1e-8:
                    allzero = False
                pos.append(v)
            if "coord_frame" in ch and ch["coord_frame"] not in ("head", "4"):
                raise ValueError("Guetschel2026 requires channel positions in the MNE head frame.")
        if allzero:
            raise ValueError("Guetschel2026 requires channel locations: a finite loc[:3] in metres for every channel.")
        nc = len(self.chs_info)
        for i in range(nc):
            d = math.sqrt(pos[3 * i] ** 2 + pos[3 * i + 1] ** 2 + pos[3 * i + 2] ** 2)
            if d < 0.05 or d > 0.2:
                raise ValueError("Guetschel2026 requires channel positions in METRES (MNE head frame), 5 to 20 cm from the origin.")
        ch_pos = torch.tensor(pos, dtype=torch.float32).reshape(nc, 3)
        self.normalization = normalization
        self.input_scale = input_scale
        self.clip_sigma = clip_sigma
        self.patch_size = patch_size
        self.patch_step = patch_size - patch_overlap
        self.embed_dim = embed_dim
        self.random_projection = random_projection
        self.random_projection_seed = random_projection_seed
        max_n_times = int(600.0 * (200.0 / self.patch_step))
        self.n_patches = (self.n_times - patch_size) // self.patch_step + 1
        self.feature_encoder = _LinearPatchEmbedding(embed_dim, patch_size, patch_overlap)
        self.add_module("feature_encoder", self.feature_encoder)
        self.model = _ContextualEncoder(_PositionalEncoder(ch_pos, embed_dim, pos_half_range, max_n_times),
                                        _TransformerEncoder(depth, embed_dim, num_heads, dim_feedforward, drop_prob,
                                                            activation))
        self.add_module("model", self.model)
        nf = self.n_chans * self.n_patches * embed_dim
        self.final_layer = nn.Sequential()
        self.final_layer.append(nn.Flatten())
        if random_projection is None:
            self.final_layer.append(nn.Linear(nf, self.n_outputs))
        else:
            rp = random_projection.__val__()
            if rp > nf:
                print(f"UserWarning: random_projection={rp} exceeds the {nf} flattened features.")
            self.final_layer.append(_GaussianRandomProjection(nf, rp, random_projection_seed))
            self.final_layer.append(nn.Linear(rp, self.n_outputs))
        self.add_module("final_layer", self.final_layer)

    def _scale(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float() * self.input_scale if not x.is_floating_point() or x.dtype != torch.float64 else x * self.input_scale
        if self.normalization == "median_std_clip":
            std = x.std(dim=-1, keepdim=True, correction=0)
            median = torch.median(std, -2, True).values
            x = x / (median + 1e-06)
            x = x.clamp(-self.clip_sigma, self.clip_sigma)
        return x.to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] < self.patch_size:
            raise ValueError(f"Input has {x.shape[-1]} samples, fewer than patch_size ({self.patch_size}).")
        if x.shape[-2] != self.n_chans:
            raise ValueError(f"The model was built for {self.n_chans} channels, but the input has {x.shape[-2]}.")
        n_patches = (x.shape[-1] - self.patch_size) // self.patch_step + 1
        if n_patches != self.n_patches:
            raise ValueError(f"The head expects {self.n_patches} patches (n_times={self.n_times}), but the input has {x.shape[-1]} samples ({n_patches} patches).")
        z = self.model(self.feature_encoder(self._scale(x)))
        return self.final_layer(z)

    def _get_name(self) -> str:
        return "Guetschel2026"
