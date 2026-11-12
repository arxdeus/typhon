# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.functional import init_weights_trunc_normal_
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FilterBankLayer, LinearWithConstraint, LogPowerLayer, StatLayer

class _SpatioTemporalFeatureBlock(nn.Module):
    in_channels: int
    out_channels: int
    n_bands: int
    stride_factor: int
    n_times: int
    n_times_padded: int
    spatial_conv: nn.Conv1d
    spatial_bn: nn.BatchNorm1d
    temporal_convs: nn.ModuleList
    act: nn.Module
    padding_layer: nn.Module
    log_power: StatLayer
    dropout_layer: nn.Dropout

    def __init__(self, n_times: int, in_channels: int, out_channels: int = 64, kernel_sizes: List[int] = [63, 31],
                 stride_factor: int = 8, n_bands: int = 2, drop_prob: float = 0.5, activation = nn.GELU, dim: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_bands = n_bands
        self.stride_factor = stride_factor
        self.n_times = n_times
        ks = kernel_sizes
        if self.n_bands > len(ks):
            self.n_bands = len(ks)
        elif self.n_bands < len(ks):
            ks = ks[:self.n_bands]
        ocs = out_channels * self.n_bands
        self.spatial_conv = nn.Conv1d(in_channels, ocs, kernel_size=1, groups=self.n_bands, bias=False)
        self.add_module("spatial_conv", self.spatial_conv)
        self.spatial_bn = nn.BatchNorm1d(ocs)
        self.add_module("spatial_bn", self.spatial_bn)
        self.add_module("unpack_bands", nn.Unflatten(dim=1, unflattened_size=(self.n_bands, out_channels)))
        self.temporal_convs = nn.ModuleList()
        self.add_module("temporal_convs", self.temporal_convs)
        for k in ks:
            self.temporal_convs.append(nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size=k, padding=k // 2, groups=out_channels, bias=False),
                nn.BatchNorm1d(out_channels)))
        inter = nn.Sequential()
        self.act = activation()
        inter.add_module("activation", self.act)
        self.add_module("inter_frequency", inter)
        if n_times % stride_factor != 0:
            ps = stride_factor - n_times % stride_factor
            self.n_times_padded = n_times + ps
            self.padding_layer = nn.ConstantPad1d((0, ps), 0.0)
        else:
            self.padding_layer = nn.Identity()
            self.n_times_padded = n_times
        self.add_module("padding_layer", self.padding_layer)
        self.log_power = LogPowerLayer(dim=dim)
        self.add_module("log_power", self.log_power)
        self.dropout_layer = nn.Dropout(drop_prob)
        self.add_module("dropout_layer", self.dropout_layer)

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        x = self.spatial_bn(self.spatial_conv(x))
        xs = x.unflatten(1, [self.n_bands, self.out_channels])
        xt = List[Tensor]()
        convs = self.temporal_convs.children()
        for i in range(len(convs)):
            xt.append(convs[i](xs[:, i]))
        x = self.act(torch.stack(xt, dim=0).sum(dim=0))
        x = self.padding_layer(x)
        x = x.view(b, self.out_channels, self.stride_factor, self.n_times_padded // self.stride_factor)
        return self.dropout_layer(self.log_power(x))

    def _get_name(self) -> str:
        return "_SpatioTemporalFeatureBlock"

class IFNet(EEGModuleMixin, nn.Module):
    """IFNet from Wang et al (2023). filter_parameters is not supported."""
    n_filters_spat: int
    stride_factor: int
    n_bands: int
    spectral_filtering: FilterBankLayer
    ensuredim: Rearrange
    feature_block: _SpatioTemporalFeatureBlock
    final_layer: LinearWithConstraint
    flatten: Rearrange

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, bands = [(4.0, 16.0), (16.0, 40.0)], n_filters_spat: int = 64,
                 kernel_sizes = (63, 31), stride_factor: int = 8, drop_prob: float = 0.5,
                 linear_max_norm: float = 0.5, activation = nn.GELU, verbose: bool = False):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.n_filters_spat = n_filters_spat
        self.stride_factor = stride_factor
        self.spectral_filtering = FilterBankLayer(n_chans=self.n_chans, sfreq=self.sfreq, band_filters=bands, verbose=verbose)
        self.add_module("spectral_filtering", self.spectral_filtering)
        self.n_bands = self.spectral_filtering.n_bands
        self.ensuredim = Rearrange("batch nbands chans time -> batch (nbands chans) time")
        self.add_module("ensuredim", self.ensuredim)
        self.feature_block = _SpatioTemporalFeatureBlock(self.n_times, self.n_chans * self.n_bands, n_filters_spat,
                                                         [int(k) for k in kernel_sizes], stride_factor, self.n_bands,
                                                         drop_prob, activation)
        self.add_module("feature_block", self.feature_block)
        self.final_layer = LinearWithConstraint(in_features=n_filters_spat * stride_factor, out_features=self.n_outputs,
                                                max_norm=linear_max_norm)
        self.add_module("final_layer", self.final_layer)
        self.flatten = Rearrange("batch ... -> batch (...)")
        self.add_module("flatten", self.flatten)
        self.apply(lambda m: init_weights_trunc_normal_(m, ["Linear", "LinearWithConstraint"], ["Conv1d", "Conv2d"],
                                                        0.01, 0.01, ["LayerNorm", "BatchNorm1d", "BatchNorm2d"], True))

    def forward(self, x: Tensor) -> Tensor:
        x = self.spectral_filtering(x)
        x = self.ensuredim(x)
        x = self.feature_block(x)
        x = self.flatten(x)
        return self.final_layer(x)

    def _get_name(self) -> str:
        return "IFNet"
