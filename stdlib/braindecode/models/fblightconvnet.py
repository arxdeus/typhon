# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FilterBankLayer, LogVarLayer, StatLayer

class _LightweightConv1d(nn.Module):
    input_size: int
    kernel_size: int
    heads: int
    padding: int
    weight_softmax: bool
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self, input_size: int, kernel_size: int = 1, padding: int = 0, heads: int = 1,
                 weight_softmax: bool = False, bias: bool = False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.heads = heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.empty(heads, 1, kernel_size))
        self.register_parameter("weight", self.weight)
        self.bias = None
        if bias:
            b = nn.Parameter(torch.empty(input_size))
            self.bias = b
            self.register_parameter("bias", b)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias.__val__(), 0.0)

    def forward(self, input: Tensor) -> Tensor:
        B, C, T = input.shape[0], input.shape[1], input.shape[2]
        w = self.weight
        if self.weight_softmax:
            w = F.softmax(w, dim=-1)
        x = input.view(-1, self.heads, T)
        out = F.conv1d(x, w, None, 1, self.padding, 1, self.heads).view(B, C, -1)
        if self.bias is not None:
            out = out + self.bias.__val__().view(1, -1, 1)
        return out

    def _get_name(self) -> str:
        return "_LightweightConv1d"

class FBLightConvNet(EEGModuleMixin, nn.Module):
    """FBLightConvNet from Ma et al (2023). filter_parameters is not supported."""
    n_bands: int
    n_filters_spat: int
    n_dim: int
    win_len: int
    heads: int
    n_times_truncated: int
    spectral_filtering: FilterBankLayer
    spatial_conv: nn.Sequential
    temporal_layer: StatLayer
    flatten_layer: Rearrange
    attn_conv: _LightweightConv1d
    final_layer: nn.Linear

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, n_bands: int = 9,
                 n_filters_spat: int = 32, n_dim: int = 3, stride_factor: Optional[int] = None, win_len: int = 250,
                 heads: int = 8, weight_softmax: bool = True, bias: bool = False, activation = nn.ELU,
                 verbose: bool = False):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.n_filters_spat = n_filters_spat
        self.n_dim = n_dim
        self.win_len = win_len
        self.heads = heads
        if self.n_times < win_len:
            raise ValueError(f"Time dimension ({self.n_times}) is shorter than win_len ({win_len}), so the model cannot build a single temporal window. Pass a longer input or lower `win_len`.")
        self.n_times_truncated = self.n_times - self.n_times % win_len
        self.spectral_filtering = FilterBankLayer(n_chans=self.n_chans, sfreq=self.sfreq, band_filters=n_bands, verbose=verbose)
        self.add_module("spectral_filtering", self.spectral_filtering)
        self.n_bands = self.spectral_filtering.n_bands
        self.spatial_conv = nn.Sequential(nn.Conv2d(self.n_bands, n_filters_spat, kernel_size=(self.n_chans, 1)),
                                          nn.BatchNorm2d(n_filters_spat), activation())
        self.add_module("spatial_conv", self.spatial_conv)
        self.temporal_layer = LogVarLayer(n_dim, False)
        self.add_module("temporal_layer", self.temporal_layer)
        self.flatten_layer = Rearrange("batch ... -> batch (...)")
        self.add_module("flatten_layer", self.flatten_layer)
        self.attn_conv = _LightweightConv1d(n_filters_spat, self.n_times // win_len, heads=heads,
                                            weight_softmax=weight_softmax, bias=bias)
        self.add_module("attn_conv", self.attn_conv)
        self.final_layer = nn.Linear(n_filters_spat, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        x = self.spatial_conv(self.spectral_filtering(x))
        x = x[:, :, :, :self.n_times_truncated]
        x = x.reshape(b, self.n_filters_spat, -1, self.win_len)
        x = self.temporal_layer(x)
        x = self.attn_conv(x)
        return self.final_layer(self.flatten_layer(x))

    def _get_name(self) -> str:
        return "FBLightConvNet"
