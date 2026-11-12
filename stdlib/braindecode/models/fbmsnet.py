# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.models.fbcnet import _temporal_layer
from braindecode.modules import Conv2dWithConstraint, FilterBankLayer, LinearWithConstraint, StatLayer

class _MixedConv2d(nn.Module):
    splits: List[int]
    convs: nn.ModuleList

    def __init__(self, in_channels: int, out_channels: int, kernels_weights = (15, 31, 63, 125), stride: int = 1,
                 dilation: int = 1, depthwise: bool = False):
        super().__init__()
        ks = [int(k) for k in kernels_weights]
        ng = len(ks)
        ins = _split(in_channels, ng)
        outs = _split(out_channels, ng)
        self.splits = ins
        self.convs = nn.ModuleList()
        self.add_module("convs", self.convs)
        for i in range(ng):
            g = outs[i] if depthwise else 1
            self.convs.append(nn.Conv2d(ins[i], outs[i], kernel_size=(1, ks[i]), stride=stride, padding="same",
                                        dilation=dilation, groups=g, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        xs = torch.split(x, self.splits, 1)
        cs = self.convs.children()
        return torch.cat([cs[i](xs[i]) for i in range(len(cs))], 1)

    def _get_name(self) -> str:
        return "_MixedConv2d"

def _split(n: int, g: int) -> List[int]:
    s = [n // g for _ in range(g)]
    s[0] += n - sum(s)
    return s

class FBMSNet(EEGModuleMixin, nn.Module):
    """FBMSNet from Liu et al (2022). filter_parameters and
    return_features are not supported."""
    n_bands: int
    n_filters_spat: int
    n_dim: int
    stride_factor: int
    dilatability: int
    out_channels_spatial: int
    n_times_padded: int
    spectral_filtering: FilterBankLayer
    mix_conv: nn.Sequential
    spatial_conv: nn.Sequential
    padding_layer: nn.Module
    temporal_layer: StatLayer
    flatten_layer: Rearrange
    final_layer: LinearWithConstraint

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, n_bands: int = 9,
                 n_filters_spat: int = 36, temporal_layer: str = "LogVarLayer", n_dim: int = 3,
                 stride_factor: int = 4, dilatability: int = 8, activation = nn.SiLU,
                 kernels_weights = (15, 31, 63, 125), cnn_max_norm: float = 2, linear_max_norm: float = 0.5,
                 verbose: bool = False, return_features: bool = False):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if return_features:
            raise NotImplementedError("FBMSNet(return_features=True) is not supported in Codon")
        self.n_filters_spat = n_filters_spat
        self.n_dim = n_dim
        self.stride_factor = stride_factor
        self.dilatability = dilatability
        self.out_channels_spatial = n_filters_spat * dilatability
        tl = _temporal_layer(temporal_layer, n_dim)
        self.spectral_filtering = FilterBankLayer(n_chans=self.n_chans, sfreq=self.sfreq, band_filters=n_bands, verbose=verbose)
        self.add_module("spectral_filtering", self.spectral_filtering)
        self.n_bands = self.spectral_filtering.n_bands
        self.mix_conv = nn.Sequential(_MixedConv2d(self.n_bands, n_filters_spat, kernels_weights, 1, 1, False),
                                      nn.BatchNorm2d(n_filters_spat))
        self.add_module("mix_conv", self.mix_conv)
        ocs = self.out_channels_spatial
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(n_filters_spat, ocs, kernel_size=(self.n_chans, 1), groups=n_filters_spat,
                                 max_norm=float(cnn_max_norm), padding=0),
            nn.BatchNorm2d(ocs), activation())
        self.add_module("spatial_conv", self.spatial_conv)
        if self.n_times % stride_factor != 0:
            ps = stride_factor - self.n_times % stride_factor
            self.n_times_padded = self.n_times + ps
            self.padding_layer = nn.ConstantPad1d((0, ps), 0.0)
        else:
            self.padding_layer = nn.Identity()
            self.n_times_padded = self.n_times
        self.add_module("padding_layer", self.padding_layer)
        self.temporal_layer = tl
        self.add_module("temporal_layer", tl)
        self.flatten_layer = Rearrange("batch ... -> batch (...)")
        self.add_module("flatten_layer", self.flatten_layer)
        self.final_layer = LinearWithConstraint(ocs * stride_factor, self.n_outputs, max_norm=linear_max_norm)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        x = self.spatial_conv(self.mix_conv(self.spectral_filtering(x)))
        x = self.padding_layer(x)
        x = x.view(b, self.out_channels_spatial, self.stride_factor, -1)
        return self.final_layer(self.flatten_layer(self.temporal_layer(x)))

    def _get_name(self) -> str:
        return "FBMSNet"
