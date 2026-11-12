# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import LogActivation

class SCCNet(EEGModuleMixin, nn.Module):
    """SCCNet from Wei et al (2019)."""
    n_spatial_filters: int
    n_spatial_filters_smooth: int
    drop_prob: float
    samples_100ms: int
    kernel_size_pool: int
    ensure_dim: Rearrange
    activation: nn.Module
    spatial_conv: nn.Conv2d
    spatial_batch_norm: nn.BatchNorm2d
    permute: Rearrange
    spatial_filt_conv: nn.Conv2d
    batch_norm: nn.BatchNorm2d
    dropout: nn.Dropout
    temporal_smoothing: nn.AvgPool2d
    final_layer: nn.Linear

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, n_spatial_filters: int = 22, n_spatial_filters_smooth: int = 20,
                 drop_prob: float = 0.5, activation = LogActivation, batch_norm_momentum: float = 0.1):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.n_spatial_filters = n_spatial_filters
        self.n_spatial_filters_smooth = n_spatial_filters_smooth
        self.drop_prob = drop_prob
        cks = int(math.floor(self.sfreq * 0.1))
        pks = int(math.floor(self.sfreq * 0.5))
        tot = cks + pks
        if self.n_times < tot:
            sf = self.n_times / tot
            cks = int(math.floor(cks * sf))
            pks = int(math.floor(pks * sf))
        self.samples_100ms = max(1, cks)
        self.kernel_size_pool = max(1, pks)
        w2 = self.n_times - self.samples_100ms + 1
        wp = (w2 - self.kernel_size_pool) // self.samples_100ms + 1
        num_features = n_spatial_filters_smooth * wp
        self.ensure_dim = Rearrange("batch nchan times -> batch 1 nchan times")
        self.add_module("ensure_dim", self.ensure_dim)
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.spatial_conv = nn.Conv2d(in_channels=1, out_channels=n_spatial_filters, kernel_size=(self.n_chans, 1))
        self.add_module("spatial_conv", self.spatial_conv)
        self.spatial_batch_norm = nn.BatchNorm2d(n_spatial_filters, momentum=batch_norm_momentum)
        self.add_module("spatial_batch_norm", self.spatial_batch_norm)
        self.permute = Rearrange("batch filspat nchans time -> batch nchans filspat time")
        self.add_module("permute", self.permute)
        self.spatial_filt_conv = nn.Conv2d(in_channels=1, out_channels=n_spatial_filters_smooth,
                                           kernel_size=(n_spatial_filters, self.samples_100ms), bias=False)
        self.add_module("spatial_filt_conv", self.spatial_filt_conv)
        self.batch_norm = nn.BatchNorm2d(n_spatial_filters_smooth, momentum=batch_norm_momentum)
        self.add_module("batch_norm", self.batch_norm)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.temporal_smoothing = nn.AvgPool2d(kernel_size=(1, self.kernel_size_pool), stride=(1, self.samples_100ms))
        self.add_module("temporal_smoothing", self.temporal_smoothing)
        self.final_layer = nn.Linear(num_features, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ensure_dim(x)
        x = self.spatial_conv(x)
        x = self.spatial_batch_norm(x)
        x = self.permute(x)
        x = self.spatial_filt_conv(x)
        x = self.batch_norm(x)
        x = torch.pow(x, 2)
        x = self.dropout(x)
        x = self.temporal_smoothing(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        return self.final_layer(x)

    def _get_name(self) -> str:
        return "SCCNet"
