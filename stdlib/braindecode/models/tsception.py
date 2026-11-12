# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin

def _tsc_conv_block(in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: Tuple[int, int],
                    pool_size: int, activation) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0),
        activation(),
        nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)))

class TSception(EEGModuleMixin, nn.Module):
    """TSception from Ding et al (2022)."""
    pool_size: int
    number_filter_spat: int
    number_filter_temp: int
    drop_prob: float
    inception_windows: Tuple[float, float, float]
    ensuredim: Rearrange
    temporal_blocks: nn.ModuleList
    _tb: List[nn.Sequential]
    batch_temporal_lay: nn.BatchNorm2d
    spatial_block_1: nn.Sequential
    spatial_block_2: nn.Sequential
    batch_spatial_lay: nn.BatchNorm2d
    feature_size: int
    dense_layer: nn.Sequential
    final_layer: nn.Linear

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, chs_info: Optional[List[Dict[str, str]]] = None,
                 n_times: Optional[int] = None, sfreq: Optional[float] = None, number_filter_temp: int = 9,
                 number_filter_spat: int = 6, hidden_size: int = 128, drop_prob: float = 0.5,
                 activation = nn.LeakyReLU, pool_size: int = 8,
                 inception_windows: Tuple[float, float, float] = (0.5, 0.25, 0.125)):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.pool_size = pool_size
        self.inception_windows = inception_windows
        self.number_filter_spat = number_filter_spat
        self.number_filter_temp = number_filter_temp
        self.drop_prob = drop_prob
        self.ensuredim = Rearrange("batch nchans time -> batch 1 nchans time")
        self.add_module("ensuredim", self.ensuredim)
        iws = self.input_window_seconds
        if iws < max(inception_windows[0], max(inception_windows[1], inception_windows[2])):
            self.inception_windows = (iws, iws / 2, iws / 4)
        self._tb = []
        self.temporal_blocks = nn.ModuleList()
        for window in self.inception_windows:
            kt = int(window * self.sfreq)
            conv_out_len = self.n_times - kt + 1
            dps = max(1, min(self.pool_size, conv_out_len))
            b = _tsc_conv_block(1, number_filter_temp, (1, kt), (1, 1), dps, activation)
            self._tb.append(b)
            self.temporal_blocks.append(b)
        self.add_module("temporal_blocks", self.temporal_blocks)
        self.batch_temporal_lay = nn.BatchNorm2d(number_filter_temp)
        self.add_module("batch_temporal_lay", self.batch_temporal_lay)
        pss = pool_size // 4
        self.spatial_block_1 = _tsc_conv_block(number_filter_temp, number_filter_spat, (self.n_chans, 1), (1, 1),
                                               pss, activation)
        self.add_module("spatial_block_1", self.spatial_block_1)
        k2 = max(1, self.n_chans // 2)
        self.spatial_block_2 = _tsc_conv_block(number_filter_temp, number_filter_spat, (k2, 1), (k2, 1),
                                               pss, activation)
        self.add_module("spatial_block_2", self.spatial_block_2)
        self.batch_spatial_lay = nn.BatchNorm2d(number_filter_spat)
        self.add_module("batch_spatial_lay", self.batch_spatial_lay)
        # braindecode infers the feature size with a dummy forward in train
        # mode, which also updates the BatchNorm running statistics.
        fs = 0
        with torch.no_grad():
            s_out = self._features(torch.ones(1, 1, self.n_chans, self.n_times))
            fs = s_out.view(1, -1).size(1)
        self.feature_size = fs
        self.dense_layer = nn.Sequential(nn.Linear(fs, hidden_size), activation(), nn.Dropout(drop_prob))
        self.add_module("dense_layer", self.dense_layer)
        self.final_layer = nn.Linear(hidden_size, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def _features(self, x: Tensor) -> Tensor:
        t_features = [layer(x) for layer in self._tb]
        t_out = self.batch_temporal_lay(torch.cat(t_features, dim=-1))
        s_out = torch.cat([self.spatial_block_1(t_out), self.spatial_block_2(t_out)], dim=2)
        return self.batch_spatial_lay(s_out)

    def forward(self, x: Tensor) -> Tensor:
        s_out = self._features(self.ensuredim(x))
        s_out = s_out.view(s_out.size(0), -1)
        return self.final_layer(self.dense_layer(s_out))

    def _get_name(self) -> str:
        return "TSception"
