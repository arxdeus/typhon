# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# SignalJEPA and its three downstream architectures. Codon notes:
# - chs_info entries are Dict[str, str]; the channel position is read from
#   "loc" as whitespace separated floats (first 3 used).
# - channel_embedding="scratch" and channel_strategy="native" only;
#   from_pretrained / reset_head / return_features are not available.

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import sinusoidal_positional_encoding

_DEFAULT_CONV_LAYER_SPEC = [(8, 32, 8), (16, 2, 2), (32, 2, 2), (64, 2, 2), (64, 2, 2)]

def _spec_list(spec) -> List[Tuple[int, int, int]]:
    return [(int(s[0]), int(s[1]), int(s[2])) for s in spec]

class _GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: int, affine: bool = True):
        super().__init__(num_groups, num_channels, affine=affine)
    def _get_name(self) -> str:
        return "_GroupNorm"

def _conv_feature_encoder(spec: List[Tuple[int, int, int]], channels: int, drop_prob: float, mode: str,
                          conv_bias: bool, activation) -> nn.Sequential:
    if mode != "default" and mode != "layer_norm":
        raise ValueError(f"feature_encoder__mode must be 'default' or 'layer_norm', got {mode}")
    enc = nn.Sequential()
    enc.append(Rearrange("b channels time -> (b channels) 1 time", channels=channels))
    cin = 1
    for i in range(len(spec)):
        cout, k, s = spec[i]
        conv = nn.Conv1d(cin, cout, k, stride=s, bias=conv_bias)
        init.kaiming_normal_(conv.weight)
        blk = nn.Sequential()
        blk.append(conv)
        blk.append(nn.Dropout(p=drop_prob))
        if mode == "layer_norm":
            blk.append(nn.Sequential(Rearrange("... channels time -> ... time channels"),
                                     nn.LayerNorm(cout, elementwise_affine=True),
                                     Rearrange("... time channels -> ... channels time")))
        elif i == 0:
            blk.append(_GroupNorm(cout, cout, affine=True))
        blk.append(activation())
        enc.append(blk)
        cin = cout
    enc.append(Rearrange("(b channels) emb_dim time_out -> b (channels time_out) emb_dim", channels=channels))
    return enc

def _n_times_out(spec: List[Tuple[int, int, int]], n_times: int) -> int:
    n = n_times
    for _, w, s in spec:
        n = int((n - w) / s) + 1
    return n

def _get_out_emb_dim(spec: List[Tuple[int, int, int]], n_times: int, n_spat_filters: int) -> int:
    return n_spat_filters * _n_times_out(spec, n_times) * spec[-1][0]

def _get_separable_clf_layer(spec: List[Tuple[int, int, int]], n_chans: int, n_times: int, n_classes: int,
                             n_spat_filters: int) -> nn.Sequential:
    d = _get_out_emb_dim(spec, n_times, n_spat_filters)
    clf = nn.Sequential()
    clf.add_module("unflatten_tokens", Rearrange("b (n_chans tokens) d -> b 1 n_chans tokens d", n_chans=n_chans))
    clf.add_module("spat_conv", nn.Conv3d(1, n_spat_filters, (n_chans, 1, 1)))
    clf.add_module("flatten", nn.Flatten(start_dim=1))
    clf.add_module("linear", nn.Linear(d, n_classes))
    return clf

def _pos_encode_contineous(x: float, x_min: float, x_max: float, n_dim: int) -> Tensor:
    div_term = torch.exp((1 - torch.arange(0, n_dim, 2) / n_dim) * 2 * math.pi)
    span = x_max - x_min
    xx = (x - x_min) / span if span != 0 else 0.0
    s = torch.sin(xx * div_term)
    c = torch.cos(xx * div_term)
    return torch.stack([s, c], dim=-1).reshape(n_dim)

def _parse_loc(ch: Dict[str, str]) -> Optional[List[float]]:
    if "loc" not in ch:
        return None
    parts = ch["loc"].split()
    if len(parts) < 3:
        return None
    v = [float(parts[k]) for k in range(3)]
    for t in v:
        if math.isnan(t) or math.isinf(t):
            return None
    return v

class _ChannelEmbedding(nn.Module):
    """nn.Embedding whose table is a deterministic positional code of the
    channel locations (no RNG draws, like braindecode's reset_parameters)."""
    weight: Tensor
    def __init__(self, channel_locations: List[Optional[List[float]]], embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(len(channel_locations), embedding_dim))
        self.register_parameter("weight", self.weight)
        valid = [l.__val__() for l in channel_locations if l is not None]
        ranges = List[Tuple[float, float]]()
        if len(valid) == 0:
            ranges = [(0.0, 0.0)] * 3
        else:
            for j in range(3):
                ranges.append((min(l[j] for l in valid), max(l[j] for l in valid)))
        gmin = min(r[0] for r in ranges)
        gmax = max(r[1] for r in ranges)
        mabs = max(abs(gmin), abs(gmax))
        per = embedding_dim // len(ranges)
        assert embedding_dim % len(ranges) == 0
        with torch.no_grad():
            for i in range(len(channel_locations)):
                self.weight[i].fill_(0.0)
                loc = channel_locations[i]
                if loc is not None:
                    lv = loc.__val__()
                    for j in range(3):
                        self.weight[i, j * per:(j + 1) * per].copy_(_pos_encode_contineous(lv[j], 0, 10 * mabs, per))
    def forward(self, idx: Tensor) -> Tensor:
        return F.embedding(idx, self.weight)
    def _get_name(self) -> str:
        return "_ChannelEmbedding"

class _PosEncoder(nn.Module):
    spat_dim: int
    time_dim: int
    max_n_times: int
    pos_encoder_spat: _ChannelEmbedding
    default_ch_idxs: Tensor
    encoding_time: Tensor
    def __init__(self, spat_dim: int, time_dim: int, channel_locations: List[Optional[List[float]]], ch_idxs: Tensor,
                 sfreq_features: float, max_seconds: float = 600.0):
        super().__init__()
        self.spat_dim = spat_dim
        self.time_dim = time_dim
        self.max_n_times = int(max_seconds * sfreq_features)
        self.pos_encoder_spat = _ChannelEmbedding(channel_locations, spat_dim)
        self.add_module("pos_encoder_spat", self.pos_encoder_spat)
        self.default_ch_idxs = ch_idxs
        self.register_buffer("default_ch_idxs", ch_idxs, persistent=False)
        self.encoding_time = torch.zeros(0, time_dim)
        self.register_buffer("encoding_time", self.encoding_time, persistent=False)
    def forward(self, local_features: Tensor) -> Tensor:
        b, nct, emb = local_features.shape[0], local_features.shape[1], local_features.shape[2]
        n_chans = self.default_ch_idxs.shape[0]
        assert emb >= self.spat_dim + self.time_dim
        assert nct % n_chans == 0
        n_times = nct // n_chans
        spat = self.pos_encoder_spat(self.default_ch_idxs)[None, :, None, :].expand(b, n_chans, n_times, self.spat_dim)
        if self.encoding_time.size(0) < n_times:
            assert self.time_dim % 2 == 0
            self.encoding_time = sinusoidal_positional_encoding(n_times, self.time_dim, base=float(self.max_n_times))
            self.register_buffer("encoding_time", self.encoding_time, persistent=False)
        te = self.encoding_time[:n_times].to(local_features.dtype)[None, None].expand(b, n_chans, n_times, self.time_dim)
        parts = [spat.to(local_features.dtype), te]
        rest = emb - self.spat_dim - self.time_dim
        if rest > 0:
            parts.append(torch.zeros(b, n_chans, n_times, rest, dtype=local_features.dtype))
        return torch.cat(parts, dim=-1).reshape(b, nct, emb)
    def _get_name(self) -> str:
        return "_PosEncoder"

def _scratch_locations(chs_info: Optional[List[Dict[str, str]]], channel_embedding: str) -> List[Optional[List[float]]]:
    if channel_embedding != "scratch":
        raise NotImplementedError("SignalJEPA: only channel_embedding='scratch' is supported in Codon")
    if chs_info is None:
        raise ValueError("chs_info is required when channel_embedding='scratch'")
    return [_parse_loc(c) for c in chs_info.__val__()]

def _check_native(channel_strategy: str):
    if channel_strategy != "native":
        raise NotImplementedError("SignalJEPA: only channel_strategy='native' is supported in Codon")

class SignalJEPA(EEGModuleMixin, nn.Module):
    """SignalJEPA from Guetschel et al (2024): contextual features
    (batch, n_chans * n_tokens, d_model)."""
    feature_encoder: nn.Sequential
    pos_encoder: _PosEncoder
    transformer: nn.Transformer
    final_layer: nn.Identity

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 feature_encoder__conv_layers_spec = _DEFAULT_CONV_LAYER_SPEC, drop_prob: float = 0.0,
                 feature_encoder__mode: str = "default", feature_encoder__conv_bias: bool = False,
                 activation = nn.GELU, pos_encoder__spat_dim: int = 30, pos_encoder__time_dim: int = 34,
                 pos_encoder__sfreq_features: float = 1.0, transformer__d_model: int = 64,
                 transformer__num_encoder_layers: int = 8, transformer__num_decoder_layers: int = 4,
                 transformer__nhead: int = 8, channel_embedding: str = "scratch", channel_strategy: str = "native"):
        _check_native(channel_strategy)
        locs = _scratch_locations(chs_info, channel_embedding)
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        spec = _spec_list(feature_encoder__conv_layers_spec)
        self.feature_encoder = _conv_feature_encoder(spec, self.n_chans, drop_prob, feature_encoder__mode,
                                                     feature_encoder__conv_bias, activation)
        self.add_module("feature_encoder", self.feature_encoder)
        self.pos_encoder = _PosEncoder(pos_encoder__spat_dim, pos_encoder__time_dim, locs,
                                       torch.arange(len(locs)), pos_encoder__sfreq_features)
        self.add_module("pos_encoder", self.pos_encoder)
        self.transformer = nn.Transformer(d_model=transformer__d_model, nhead=transformer__nhead,
                                          num_encoder_layers=transformer__num_encoder_layers,
                                          num_decoder_layers=transformer__num_decoder_layers, batch_first=True)
        self.add_module("transformer", self.transformer)
        self.final_layer = nn.Identity()
        self.add_module("final_layer", self.final_layer)

    def features(self, X: Tensor) -> Tensor:
        lf = self.feature_encoder(X)
        lf = lf + self.pos_encoder(lf)
        return self.transformer.encoder.encode(lf)

    def forward(self, X: Tensor) -> Tensor:
        return self.final_layer(self.features(X))

    def _get_name(self) -> str:
        return "SignalJEPA"

class SignalJEPA_Contextual(EEGModuleMixin, nn.Module):
    """SignalJEPA with a separable-convolution classification head on the
    contextual features."""
    feature_encoder: nn.Sequential
    pos_encoder: _PosEncoder
    transformer: nn.Transformer
    final_layer: nn.Sequential

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 n_spat_filters: int = 4, feature_encoder__conv_layers_spec = _DEFAULT_CONV_LAYER_SPEC,
                 drop_prob: float = 0.0, feature_encoder__mode: str = "default",
                 feature_encoder__conv_bias: bool = False, activation = nn.GELU, pos_encoder__spat_dim: int = 30,
                 pos_encoder__time_dim: int = 34, pos_encoder__sfreq_features: float = 1.0,
                 transformer__d_model: int = 64, transformer__num_encoder_layers: int = 8,
                 transformer__num_decoder_layers: int = 4, transformer__nhead: int = 8,
                 channel_embedding: str = "scratch", channel_strategy: str = "native"):
        _check_native(channel_strategy)
        locs = _scratch_locations(chs_info, channel_embedding)
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        spec = _spec_list(feature_encoder__conv_layers_spec)
        self.feature_encoder = _conv_feature_encoder(spec, self.n_chans, drop_prob, feature_encoder__mode,
                                                     feature_encoder__conv_bias, activation)
        self.add_module("feature_encoder", self.feature_encoder)
        self.pos_encoder = _PosEncoder(pos_encoder__spat_dim, pos_encoder__time_dim, locs,
                                       torch.arange(len(locs)), pos_encoder__sfreq_features)
        self.add_module("pos_encoder", self.pos_encoder)
        self.transformer = nn.Transformer(d_model=transformer__d_model, nhead=transformer__nhead,
                                          num_encoder_layers=transformer__num_encoder_layers,
                                          num_decoder_layers=transformer__num_decoder_layers, batch_first=True)
        self.add_module("transformer", self.transformer)
        self.final_layer = _get_separable_clf_layer(spec, self.n_chans, self.n_times, self.n_outputs, n_spat_filters)
        self.add_module("final_layer", self.final_layer)

    def forward(self, X: Tensor) -> Tensor:
        lf = self.feature_encoder(X)
        lf = lf + self.pos_encoder(lf)
        return self.final_layer(self.transformer.encoder.encode(lf))

    def _get_name(self) -> str:
        return "SignalJEPA_Contextual"

class SignalJEPA_PostLocal(EEGModuleMixin, nn.Module):
    """SignalJEPA local encoder followed by a separable classification head."""
    feature_encoder: nn.Sequential
    final_layer: nn.Sequential

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 n_spat_filters: int = 4, feature_encoder__conv_layers_spec = _DEFAULT_CONV_LAYER_SPEC,
                 drop_prob: float = 0.0, feature_encoder__mode: str = "default",
                 feature_encoder__conv_bias: bool = False, activation = nn.GELU, channel_strategy: str = "native"):
        _check_native(channel_strategy)
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        spec = _spec_list(feature_encoder__conv_layers_spec)
        self.feature_encoder = _conv_feature_encoder(spec, self.n_chans, drop_prob, feature_encoder__mode,
                                                     feature_encoder__conv_bias, activation)
        self.add_module("feature_encoder", self.feature_encoder)
        self.final_layer = _get_separable_clf_layer(spec, self.n_chans, self.n_times, self.n_outputs, n_spat_filters)
        self.add_module("final_layer", self.final_layer)

    def forward(self, X: Tensor) -> Tensor:
        return self.final_layer(self.feature_encoder(X))

    def _get_name(self) -> str:
        return "SignalJEPA_PostLocal"

class SignalJEPA_PreLocal(EEGModuleMixin, nn.Module):
    """Spatial convolution, then the SignalJEPA local encoder and a linear head."""
    n_spat_filters: int
    feature_encoder: nn.Sequential
    spatial_conv: nn.Sequential
    final_layer: nn.Sequential

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 n_spat_filters: int = 4, feature_encoder__conv_layers_spec = _DEFAULT_CONV_LAYER_SPEC,
                 drop_prob: float = 0.0, feature_encoder__mode: str = "default",
                 feature_encoder__conv_bias: bool = False, activation = nn.GELU, channel_strategy: str = "native"):
        _check_native(channel_strategy)
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.n_spat_filters = n_spat_filters
        spec = _spec_list(feature_encoder__conv_layers_spec)
        self.feature_encoder = _conv_feature_encoder(spec, n_spat_filters, drop_prob, feature_encoder__mode,
                                                     feature_encoder__conv_bias, activation)
        self.add_module("feature_encoder", self.feature_encoder)
        self.spatial_conv = nn.Sequential(Rearrange("b channels time -> b 1 channels time"),
                                          nn.Conv2d(1, n_spat_filters, (self.n_chans, 1)),
                                          Rearrange("b spat_filters 1 time -> b spat_filters time"))
        self.add_module("spatial_conv", self.spatial_conv)
        self.final_layer = nn.Sequential(nn.Flatten(start_dim=1),
                                         nn.Linear(_get_out_emb_dim(spec, self.n_times, n_spat_filters), self.n_outputs))
        self.add_module("final_layer", self.final_layer)

    def forward(self, X: Tensor) -> Tensor:
        return self.final_layer(self.feature_encoder(self.spatial_conv(X)))

    def _get_name(self) -> str:
        return "SignalJEPA_PreLocal"
