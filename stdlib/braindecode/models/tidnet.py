# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Ensure4d

class _BatchNormZG(nn.BatchNorm2d):
    # BatchNorm2d with zero-initialised affine weight and bias.
    _zg: int
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self._zg = 0
        with torch.no_grad():
            self.weight.__val__().zero_()
            self.bias.__val__().zero_()
    def _get_name(self) -> str:
        return "_BatchNormZG"

class _ConvBlock2D(nn.Module):
    activation: nn.Module
    residual: bool
    conv: nn.Conv2d
    dropout: nn.Dropout2d
    batch_norm: nn.Module

    def __init__(self, in_filters: int, out_filters: int, kernel, stride = (1, 1), padding = 0, dilation = 1,
                 groups: int = 1, drop_prob: float = 0.5, batch_norm: bool = True, activation = nn.LeakyReLU,
                 residual: bool = False):
        super().__init__()
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.residual = residual
        self.conv = nn.Conv2d(in_filters, out_filters, kernel, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=not batch_norm)
        self.add_module("conv", self.conv)
        self.dropout = nn.Dropout2d(p=float(drop_prob))
        self.add_module("dropout", self.dropout)
        bn: nn.Module = nn.Identity()
        if residual:
            bn = _BatchNormZG(out_filters)
        elif batch_norm:
            bn = nn.BatchNorm2d(out_filters)
        self.batch_norm = bn
        self.add_module("batch_norm", bn)

    def forward(self, input: Tensor) -> Tensor:
        x = self.batch_norm(self.activation(self.dropout(self.conv(input))))
        return x + input if self.residual else x

    def _get_name(self) -> str:
        return "_ConvBlock2D"

class _DenseFilter(nn.Module):
    net: nn.Sequential

    def __init__(self, in_features: int, growth_rate: int, filter_len: int = 5, drop_prob: float = 0.5,
                 bottleneck: int = 2, activation = nn.LeakyReLU, dim: int = -2):
        super().__init__()
        d = dim if dim > 0 else dim + 4
        if d < 2 or d > 3:
            raise ValueError("Only last two dimensions supported")
        k = (filter_len, 1) if d == 2 else (1, filter_len)
        self.net = nn.Sequential(nn.BatchNorm2d(in_features), activation(), nn.Conv2d(in_features, bottleneck * growth_rate, 1),
                                 nn.BatchNorm2d(bottleneck * growth_rate), activation(),
                                 nn.Conv2d(bottleneck * growth_rate, growth_rate, k, padding=(k[0] // 2, k[1] // 2)),
                                 nn.Dropout2d(p=float(drop_prob)))
        self.add_module("net", self.net)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x, self.net(x)], dim=1)

    def _get_name(self) -> str:
        return "_DenseFilter"

class _DenseSpatialFilter(nn.Module):
    net: nn.Sequential
    collapse: bool
    channel_collapse: Optional[_ConvBlock2D]

    def __init__(self, n_chans: int, growth: int, depth: int, in_ch: int = 1, bottleneck: int = 4,
                 drop_prob: float = 0.0, activation = nn.LeakyReLU, collapse: bool = True):
        super().__init__()
        self.net = nn.Sequential()
        for d in range(depth):
            self.net.add_module(str(d), _DenseFilter(in_ch + growth * d, growth, bottleneck=bottleneck, drop_prob=drop_prob,
                                                     activation=activation))
        self.add_module("net", self.net)
        nf = in_ch + growth * depth
        self.collapse = collapse
        self.channel_collapse = None
        if collapse:
            cc = _ConvBlock2D(nf, nf, (n_chans, 1), drop_prob=0.0, activation=activation)
            self.channel_collapse = cc
            self.add_module("channel_collapse", cc)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 4:
            x = x.unsqueeze(1).permute(0, 1, 3, 2)
        x = self.net(x)
        if self.collapse:
            return self.channel_collapse.__val__()(x).squeeze(-2)
        return x

    def _get_name(self) -> str:
        return "_DenseSpatialFilter"

class _TemporalConv(nn.Conv2d):
    # braindecode may evaluate this conv via FFT on CPU; mathematically equal.
    _tc: int
    def __init__(self, in_channels: int, out_channels: int, kernel_size, dilation = 1, padding = 0):
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self._tc = 0
    def _get_name(self) -> str:
        return "_TemporalConv"

class _TemporalFilter(nn.Module):
    residual_style: str
    net: nn.Sequential
    residual: nn.Conv2d

    def __init__(self, n_chans: int, filters: int, depth: int, temp_len: int, drop_prob: float = 0.0,
                 activation = nn.LeakyReLU, residual: str = "netwise"):
        super().__init__()
        tl = temp_len + 1 - temp_len % 2
        self.residual_style = residual
        if residual.lower() != "netwise":
            raise NotImplementedError("TIDNet _TemporalFilter: only residual='netwise' is supported in Codon")
        self.net = nn.Sequential()
        for i in range(depth):
            dil = depth - i
            conv = _TemporalConv(n_chans if i == 0 else filters, filters, kernel_size=(1, tl), dilation=dil,
                                 padding=(0, dil * (tl - 1) // 2))
            weight_norm(conv)
            self.net.add_module(str(i), nn.Sequential(conv, activation(), nn.Dropout2d(p=float(drop_prob))))
        self.add_module("net", self.net)
        self.residual = nn.Conv2d(n_chans, filters, (1, 1))
        self.add_module("residual", self.residual)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + self.residual(x)

    def _get_name(self) -> str:
        return "_TemporalFilter"

class _TIDNetFeatures(nn.Module):
    n_chans: int
    temp_len: int
    temporal: nn.Sequential
    spatial: _DenseSpatialFilter
    extract_features: nn.Sequential
    num_features: int

    def __init__(self, s_growth: int, t_filters: int, n_chans: int, n_times: int, drop_prob: float, pooling: int,
                 temp_layers: int, spat_layers: int, temp_span: float, bottleneck: int, summary: int,
                 activation = nn.LeakyReLU):
        super().__init__()
        self.n_chans = n_chans
        self.temp_len = int(math.ceil(temp_span * n_times))
        self.temporal = nn.Sequential(Ensure4d(), Rearrange("batch C T 1 -> batch 1 C T"),
                                      _TemporalFilter(1, t_filters, depth=temp_layers, temp_len=self.temp_len, activation=activation),
                                      nn.MaxPool2d((1, pooling)), nn.Dropout2d(p=float(drop_prob)))
        self.add_module("temporal", self.temporal)
        sm = n_times // pooling if summary == -1 else summary
        self.spatial = _DenseSpatialFilter(n_chans, s_growth, spat_layers, in_ch=t_filters, drop_prob=drop_prob,
                                           bottleneck=bottleneck, activation=activation)
        self.add_module("spatial", self.spatial)
        self.extract_features = nn.Sequential(nn.AdaptiveAvgPool1d(sm), nn.Flatten(start_dim=1))
        self.add_module("extract_features", self.extract_features)
        self.num_features = (t_filters + s_growth * spat_layers) * sm

    def forward(self, x: Tensor) -> Tensor:
        return self.extract_features(self.spatial(self.temporal(x)))

    def _get_name(self) -> str:
        return "_TIDNetFeatures"

class TIDNet(EEGModuleMixin, nn.Module):
    """Thinker Invariance DenseNet from Kostas & Rudzicz (2020)."""
    temp_len: int
    dscnn: _TIDNetFeatures
    _num_features: int
    flatten: nn.Flatten
    final_layer: nn.Sequential

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, s_growth: int = 24, t_filters: int = 32,
                 drop_prob: float = 0.4, pooling: int = 15, temp_layers: int = 2, spat_layers: int = 2,
                 temp_span: float = 0.05, bottleneck: int = 3, summary: int = -1, activation = nn.LeakyReLU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.temp_len = int(math.ceil(temp_span * self.n_times))
        self.dscnn = _TIDNetFeatures(s_growth, t_filters, self.n_chans, self.n_times, drop_prob, pooling, temp_layers,
                                     spat_layers, temp_span, bottleneck, summary, activation)
        self.add_module("dscnn", self.dscnn)
        self._num_features = self.dscnn.num_features
        self.flatten = nn.Flatten(start_dim=1)
        self.add_module("flatten", self.flatten)
        clf = nn.Linear(self._num_features, self.n_outputs)
        nn.init.xavier_normal_(clf.weight)
        nn.init.zeros_(clf.bias.__val__())
        self.final_layer = nn.Sequential(clf, nn.Identity())
        self.add_module("final_layer", self.final_layer)

    @property
    def num_features(self) -> int:
        return self._num_features

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.flatten(self.dscnn(x)))

    def _get_name(self) -> str:
        return "TIDNet"
