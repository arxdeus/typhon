# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _bn_off_if_batch_one, _bn_restore
from braindecode.modules import Chomp1d, MaxNormLinear

class _EEGNetTC(nn.Module):
    activation: nn.Module
    drop_prob: float
    n_chans: int
    filter_1: int
    filter_2: int
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    depthwise_conv: nn.Conv2d
    bn2: nn.BatchNorm2d
    pool1: nn.AvgPool2d
    drop1: nn.Dropout
    separable_conv_depthwise: nn.Conv2d
    separable_conv_pointwise: nn.Conv2d
    bn3: nn.BatchNorm2d
    pool2: nn.AvgPool2d
    drop2: nn.Dropout

    def __init__(self, n_chans: int, filter_1: int = 8, kern_length: int = 64, depth_multiplier: int = 2,
                 drop_prob: float = 0.5, activation = nn.ELU):
        super().__init__()
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.drop_prob = drop_prob
        self.n_chans = n_chans
        self.filter_1 = filter_1
        f2 = filter_1 * depth_multiplier
        self.filter_2 = f2
        self.conv1 = nn.Conv2d(1, filter_1, kernel_size=(kern_length, 1), padding=(kern_length // 2, 0), bias=False)
        self.add_module("conv1", self.conv1)
        self.bn1 = nn.BatchNorm2d(filter_1)
        self.add_module("bn1", self.bn1)
        self.depthwise_conv = nn.Conv2d(filter_1, f2, kernel_size=(1, n_chans), groups=filter_1, bias=False)
        self.add_module("depthwise_conv", self.depthwise_conv)
        self.bn2 = nn.BatchNorm2d(f2)
        self.add_module("bn2", self.bn2)
        self.pool1 = nn.AvgPool2d(kernel_size=(8, 1))
        self.add_module("pool1", self.pool1)
        self.drop1 = nn.Dropout(p=drop_prob)
        self.add_module("drop1", self.drop1)
        self.separable_conv_depthwise = nn.Conv2d(f2, f2, kernel_size=(f2, 1), groups=f2, padding=(f2 // 2, 0), bias=False)
        self.add_module("separable_conv_depthwise", self.separable_conv_depthwise)
        self.separable_conv_pointwise = nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False)
        self.add_module("separable_conv_pointwise", self.separable_conv_pointwise)
        self.bn3 = nn.BatchNorm2d(f2)
        self.add_module("bn3", self.bn3)
        self.pool2 = nn.AvgPool2d(kernel_size=(filter_1, 1))
        self.add_module("pool2", self.pool2)
        self.drop2 = nn.Dropout(p=drop_prob)
        self.add_module("drop2", self.drop2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.depthwise_conv(x)))
        x = self.drop1(self.pool1(x))
        x = self.separable_conv_pointwise(self.separable_conv_depthwise(x))
        x = self.activation(self.bn3(x))
        return self.drop2(self.pool2(x))

    def _get_name(self) -> str:
        return "_EEGNetTC"

class _TCNBlock(nn.Module):
    activation: nn.Module
    drop_prob: float
    depth: int
    filters: int
    kernel_size: int
    batch_norm: bool
    layers: nn.ModuleList
    downsample: Optional[nn.Conv1d]

    def __init__(self, input_dimension: int, depth: int, kernel_size: int, filters: int, drop_prob: float,
                 activation = nn.ELU, batch_norm: bool = True):
        super().__init__()
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.drop_prob = drop_prob
        self.depth = depth
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.layers = nn.ModuleList()
        self.add_module("layers", self.layers)
        self.downsample = None
        if input_dimension != filters:
            ds = nn.Conv1d(input_dimension, filters, kernel_size=1, bias=batch_norm)
            self.downsample = ds
            self.add_module("downsample", ds)
        for i in range(depth):
            d = 1 << i
            pad = (kernel_size - 1) * d
            in_dim = input_dimension if i == 0 else filters
            blk = nn.Sequential()
            units = self._conv_unit(in_dim, d, pad) + self._conv_unit(filters, d, pad)
            for j in range(len(units)):
                blk.add_module(str(j), units[j])
            self.layers.append(blk)

    def _conv_unit(self, in_channels: int, dilation: int, padding: int) -> List[nn.Module]:
        unit = List[nn.Module]()
        unit.append(nn.Conv1d(in_channels, self.filters, kernel_size=self.kernel_size, dilation=dilation,
                              padding=padding, bias=False))
        unit.append(Chomp1d(padding))
        if self.batch_norm:
            unit.append(nn.BatchNorm1d(self.filters, momentum=0.01, eps=0.001))
        unit.append(self.activation)
        unit.append(nn.Dropout(self.drop_prob))
        return unit

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        res = x
        if self.downsample is not None:
            res = self.downsample.__val__()(x)
        out = x
        for layer in self.layers.children():
            out = layer(x)
            out = out + res
            out = self.activation(out)
            res = out
            x = out
        return out.permute(0, 2, 1)

    def _get_name(self) -> str:
        return "_TCNBlock"

class EEGTCNet(EEGModuleMixin, nn.Module):
    """EEG-TCNet from Ingolfsson et al (2020)."""
    drop_prob: float
    drop_prob_eeg: float
    drop_prob_tcn: float
    tcn_batch_norm: bool
    depth_multiplier: int
    filter_1: int
    kern_length: int
    depth: int
    kernel_size: int
    filters: int
    max_norm_const: float
    filter_2: int
    arrange_dim_input: Rearrange
    eegnet_tc: _EEGNetTC
    arrange_dim_eegnet: Rearrange
    tcn_block: _TCNBlock
    final_layer: MaxNormLinear

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, activation = nn.ELU, depth_multiplier: int = 2, filter_1: int = 8,
                 kern_length: int = 64, drop_prob: float = 0.5, depth: int = 2, kernel_size: int = 4,
                 filters: int = 12, max_norm_const: float = 0.25, drop_prob_eeg: Optional[float] = None,
                 drop_prob_tcn: Optional[float] = None, tcn_batch_norm: bool = True):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.drop_prob = drop_prob
        self.drop_prob_eeg = drop_prob if drop_prob_eeg is None else drop_prob_eeg.__val__()
        self.drop_prob_tcn = drop_prob if drop_prob_tcn is None else drop_prob_tcn.__val__()
        self.tcn_batch_norm = tcn_batch_norm
        self.depth_multiplier = depth_multiplier
        self.filter_1 = filter_1
        self.kern_length = kern_length
        self.depth = depth
        self.kernel_size = kernel_size
        self.filters = filters
        self.max_norm_const = max_norm_const
        self.filter_2 = filter_1 * depth_multiplier
        self.arrange_dim_input = Rearrange("batch nchans ntimes -> batch 1 ntimes nchans")
        self.add_module("arrange_dim_input", self.arrange_dim_input)
        self.eegnet_tc = _EEGNetTC(self.n_chans, filter_1, kern_length, depth_multiplier, self.drop_prob_eeg, activation)
        self.add_module("eegnet_tc", self.eegnet_tc)
        self.arrange_dim_eegnet = Rearrange("batch filter2 rtimes 1 -> batch rtimes filter2")
        self.add_module("arrange_dim_eegnet", self.arrange_dim_eegnet)
        self.tcn_block = _TCNBlock(self.filter_2, depth, kernel_size, filters, self.drop_prob_tcn, activation, tcn_batch_norm)
        self.add_module("tcn_block", self.tcn_block)
        self.final_layer = MaxNormLinear(in_features=filters, out_features=self.n_outputs, max_norm_val=max_norm_const)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        off = _bn_off_if_batch_one(self, x.shape[0])
        x = self.arrange_dim_input(x)
        x = self.eegnet_tc(x)
        x = self.arrange_dim_eegnet(x)
        x = self.tcn_block(x)
        x = x[:, -1, :]
        x = self.final_layer(x)
        _bn_restore(off)
        return x

    def _get_name(self) -> str:
        return "EEGTCNet"
