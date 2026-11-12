# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from .parametrization import MaxNormParametrize

class AvgPool2dWithConv(nn.Module):
    kernel_size: List[int]
    stride: List[int]
    dilation: List[int]
    padding: List[int]
    def __init__(self, kernel_size, stride, dilation = 1, padding = 0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
    def forward(self, x: Tensor) -> Tensor:
        in_channels = x.shape[1]
        n_pool = self.kernel_size[0] * self.kernel_size[1]
        w = torch.ones(in_channels, 1, self.kernel_size[0], self.kernel_size[1], dtype=x.dtype) / float(n_pool)
        return F.conv2d(x, w, None, (self.stride[0], self.stride[1]), (self.padding[0], self.padding[1]),
                        (self.dilation[0], self.dilation[1]), in_channels)
    def _get_name(self) -> str:
        return "AvgPool2dWithConv"

def _pair(v) -> List[int]:
    if isinstance(v, int):
        return [v, v]
    else:
        return [int(a) for a in v]

class Conv2dWithConstraint(nn.Conv2d):
    max_norm: float
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1,
                 groups: int = 1, bias: bool = True, padding_mode: str = "zeros", max_norm: float = 1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.max_norm = max_norm
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        register_parametrization(self, "weight", MaxNormParametrize(self.max_norm))
    def _get_name(self) -> str:
        return "Conv2dWithConstraint"

class Conv1dWithConstraint(nn.Conv1d):
    max_norm: float
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1,
                 groups: int = 1, bias: bool = True, padding_mode: str = "zeros", max_norm: float = 1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.max_norm = max_norm
        register_parametrization(self, "weight", MaxNormParametrize(self.max_norm))
    def _get_name(self) -> str:
        return "Conv1dWithConstraint"

class CombinedConv(nn.Module):
    bias_time: bool
    bias_spat: bool
    conv_time: nn.Conv2d
    conv_spat: nn.Conv2d
    def __init__(self, in_chans: int, n_filters_time: int = 40, n_filters_spat: int = 40, filter_time_length: int = 25,
                 bias_time: bool = True, bias_spat: bool = True, stride = 1):
        super().__init__()
        self.bias_time = bias_time
        self.bias_spat = bias_spat
        self.conv_time = nn.Conv2d(1, n_filters_time, (filter_time_length, 1), bias=bias_time, stride=1)
        self.add_module("conv_time", self.conv_time)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_spat, (1, in_chans), bias=bias_spat, stride=stride)
        self.add_module("conv_spat", self.conv_spat)
    def forward(self, x: Tensor) -> Tensor:
        combined_weight = (self.conv_time.weight * self.conv_spat.weight.permute(1, 0, 2, 3)).sum(0).unsqueeze(1)
        calculated_bias: Optional[Tensor] = None
        if self.bias_time:
            tb = self.conv_time.bias
            if tb is None:
                raise RuntimeError("conv_time.bias is None despite bias_time=True")
            calculated_bias = self.conv_spat.weight.squeeze(2).sum(-1).mm(tb.unsqueeze(-1)).squeeze(-1)
        if self.bias_spat:
            sb = self.conv_spat.bias
            if sb is None:
                raise RuntimeError("conv_spat.bias is None despite bias_spat=True")
            if calculated_bias is None:
                calculated_bias = sb
            else:
                calculated_bias = calculated_bias + sb
        st = self.conv_spat.stride
        return F.conv2d(x, combined_weight, calculated_bias, (st[0], st[1]))
    def _get_name(self) -> str:
        return "CombinedConv"

class CausalConv1d(nn.Conv1d):
    _causal: int
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, stride: int = 1,
                 groups: int = 1, bias: bool = True):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         dilation=dilation, padding=(kernel_size - 1) * dilation, groups=groups, bias=bias)
        self._causal = 0
    def forward(self, X: Tensor) -> Tensor:
        X = F.pad(X, [self.padding[0], 0])
        return F.conv1d(X, self.weight, self.bias, self.stride[0], 0, self.dilation[0], self.groups)
    def _get_name(self) -> str:
        return "CausalConv1d"

class DepthwiseConv2d(nn.Conv2d):
    _dw: int
    def __init__(self, in_channels: int, depth_multiplier: int = 2, kernel_size = 3, stride = 1, padding = 0,
                 dilation = 1, bias: bool = True, padding_mode: str = "zeros"):
        super().__init__(in_channels, in_channels * depth_multiplier, kernel_size, stride, padding, dilation,
                         in_channels, bias, padding_mode)
        self._dw = 0
    def _get_name(self) -> str:
        return "DepthwiseConv2d"
