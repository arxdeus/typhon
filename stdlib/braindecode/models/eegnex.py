# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Conv2dWithConstraint, LinearWithConstraint

class _TemporalConv(nn.Conv2d):
    # braindecode switches to an FFT convolution on large CPU inputs; it is
    # mathematically identical, so the direct convolution is used here.
    _tc: int
    def __init__(self, in_channels: int, out_channels: int, kernel_size, padding = 0, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self._tc = 0
    def _get_name(self) -> str:
        return "_TemporalConv"

class EEGNeX(EEGModuleMixin, nn.Module):
    """EEGNeX from Chen et al (2024)."""
    depth_multiplier: int
    filter_1: int
    filter_2: int
    filter_3: int
    drop_prob: float
    in_features: int
    block_1: nn.Sequential
    block_2: nn.Sequential
    block_3: nn.Sequential
    block_4: nn.Sequential
    block_5: nn.Sequential
    final_layer: LinearWithConstraint

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, activation = nn.ELU, depth_multiplier: int = 2, filter_1: int = 8,
                 filter_2: int = 32, drop_prob: float = 0.5, kernel_block_1_2: int = 64, kernel_block_4: int = 16,
                 dilation_block_4: int = 2, avg_pool_block4: int = 4, kernel_block_5: int = 16,
                 dilation_block_5: int = 4, avg_pool_block5: int = 8, max_norm_conv: float = 1.0,
                 max_norm_linear: float = 0.25):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.depth_multiplier = depth_multiplier
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = filter_2 * depth_multiplier
        self.drop_prob = drop_prob
        T3 = (self.n_times + 2 - avg_pool_block4) // avg_pool_block4 + 1
        T5 = (T3 + 2 - avg_pool_block5) // avg_pool_block5 + 1
        self.in_features = filter_1 * T5
        self.block_1 = nn.Sequential(Rearrange("batch ch time -> batch 1 ch time"),
            nn.Conv2d(in_channels=1, out_channels=filter_1, kernel_size=(1, kernel_block_1_2), padding="same", bias=False),
            nn.BatchNorm2d(num_features=filter_1))
        self.add_module("block_1", self.block_1)
        self.block_2 = nn.Sequential(_TemporalConv(filter_1, filter_2, (1, kernel_block_1_2), padding="same", bias=False),
            nn.BatchNorm2d(num_features=filter_2))
        self.add_module("block_2", self.block_2)
        self.block_3 = nn.Sequential(
            Conv2dWithConstraint(in_channels=filter_2, out_channels=self.filter_3, max_norm=max_norm_conv,
                                 kernel_size=(self.n_chans, 1), groups=filter_2, bias=False),
            nn.BatchNorm2d(num_features=self.filter_3), activation(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block4), padding=(0, 1)), nn.Dropout(p=drop_prob))
        self.add_module("block_3", self.block_3)
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.filter_3, out_channels=filter_2, kernel_size=(1, kernel_block_4),
                      dilation=(1, dilation_block_4), padding="same", bias=False),
            nn.BatchNorm2d(num_features=filter_2))
        self.add_module("block_4", self.block_4)
        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=filter_2, out_channels=filter_1, kernel_size=(1, kernel_block_5),
                      dilation=(1, dilation_block_5), padding="same", bias=False),
            nn.BatchNorm2d(num_features=filter_1), activation(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block5), padding=(0, 1)), nn.Dropout(p=drop_prob), nn.Flatten())
        self.add_module("block_5", self.block_5)
        self.final_layer = LinearWithConstraint(in_features=self.in_features, out_features=self.n_outputs, max_norm=max_norm_linear)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return self.final_layer(x)

    def _get_name(self) -> str:
        return "EEGNeX"
