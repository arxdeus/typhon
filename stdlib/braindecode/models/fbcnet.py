# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Conv2dWithConstraint, FilterBankLayer, LinearWithConstraint, StatLayer, LogVarLayer, VarLayer, StdLayer, MeanLayer, MaxLayer

def _temporal_layer(name: str, dim: int) -> StatLayer:
    if name == "VarLayer":
        return VarLayer(dim=dim)
    if name == "StdLayer":
        return StdLayer(dim=dim)
    if name == "LogVarLayer":
        return LogVarLayer(dim=dim)
    if name == "MeanLayer":
        return MeanLayer(dim=dim)
    if name == "MaxLayer":
        return MaxLayer(dim=dim)
    raise NotImplementedError(f"Temporal layer '{name}' is not implemented.")

class FBCNet(EEGModuleMixin, nn.Module):
    """FBCNet from Mane et al (2021). filter_parameters is not supported."""
    n_bands: int
    n_filters_spat: int
    n_dim: int
    stride_factor: int
    n_times_padded: int
    spectral_filtering: FilterBankLayer
    spatial_conv: nn.Sequential
    padding_layer: nn.Module
    temporal_layer: StatLayer
    flatten_layer: Rearrange
    final_layer: LinearWithConstraint

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, n_bands: int = 9,
                 n_filters_spat: int = 32, temporal_layer: str = "LogVarLayer", n_dim: int = 3,
                 stride_factor: int = 4, activation = nn.SiLU, linear_max_norm: float = 0.5,
                 cnn_max_norm: float = 2.0):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.n_filters_spat = n_filters_spat
        self.n_dim = n_dim
        self.stride_factor = stride_factor
        tl = _temporal_layer(temporal_layer, n_dim)
        self.spectral_filtering = FilterBankLayer(n_chans=self.n_chans, sfreq=self.sfreq, band_filters=n_bands, verbose=False)
        self.add_module("spectral_filtering", self.spectral_filtering)
        nb = self.spectral_filtering.n_bands
        self.n_bands = nb
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(in_channels=nb, out_channels=n_filters_spat * nb, kernel_size=(self.n_chans, 1),
                                 groups=nb, max_norm=cnn_max_norm, padding=0),
            nn.BatchNorm2d(n_filters_spat * nb), activation())
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
        self.final_layer = LinearWithConstraint(in_features=n_filters_spat * nb * stride_factor, out_features=self.n_outputs,
                                                max_norm=linear_max_norm)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.spectral_filtering(x)
        x = self.spatial_conv(x)
        b, c = x.shape[0], x.shape[1]
        x = self.padding_layer(x)
        x = x.view(b, c, self.stride_factor, self.n_times_padded // self.stride_factor)
        x = self.temporal_layer(x)
        x = self.flatten_layer(x)
        return self.final_layer(x)

    def _get_name(self) -> str:
        return "FBCNet"
