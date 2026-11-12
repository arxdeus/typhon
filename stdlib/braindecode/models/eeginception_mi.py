# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Ensure4d

class _InceptionModuleMI(nn.Module):
    in_channels: int
    n_filters: int
    n_convs: int
    bottleneck: nn.Conv2d
    pooling: nn.MaxPool2d
    pooling_conv: nn.Conv2d
    conv_list: nn.ModuleList
    bn: nn.BatchNorm2d
    activation: nn.Module

    def __init__(self, in_channels: int, n_filters: int, n_convs: int, kernel_unit_s: float = 0.1, sfreq: float = 250.0,
                 activation = nn.ReLU):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.n_convs = n_convs
        self.bottleneck = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=1, bias=True)
        self.add_module("bottleneck", self.bottleneck)
        ku = int(kernel_unit_s * sfreq)
        self.pooling = nn.MaxPool2d(kernel_size=(1, ku), stride=1, padding=(0, ku // 2))
        self.add_module("pooling", self.pooling)
        self.pooling_conv = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=1, bias=True)
        self.add_module("pooling_conv", self.pooling_conv)
        convs = List[nn.Module]()
        for n in range(n_convs):
            convs.append(nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=(1, (n * 2 + 1) * ku),
                                   padding="same", bias=True))
        self.conv_list = nn.ModuleList(convs)
        self.add_module("conv_list", self.conv_list)
        self.bn = nn.BatchNorm2d(n_filters * (n_convs + 1))
        self.add_module("bn", self.bn)
        self.activation = activation()
        self.add_module("activation", self.activation)

    def forward(self, X: Tensor) -> Tensor:
        X1 = self.bottleneck(X)
        X1 = torch.cat([c(X1) for c in self.conv_list.children()], 1)
        X2 = self.pooling(X)
        X2 = self.pooling_conv(X2.contiguous())[..., :X.shape[-1]]
        out = torch.cat([X1, X2], 1)
        return self.activation(self.bn(out))

    def _get_name(self) -> str:
        return "_InceptionModuleMI"

class _ResidualModuleMI(nn.Module):
    in_channels: int
    n_filters: int
    activation: nn.Module
    bn: nn.BatchNorm2d
    conv: nn.Conv2d

    def __init__(self, in_channels: int, n_filters: int, activation = nn.ReLU):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.bn = nn.BatchNorm2d(n_filters)
        self.add_module("bn", self.bn)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=1, bias=True)
        self.add_module("conv", self.conv)

    def forward(self, X: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(X)))

    def _get_name(self) -> str:
        return "_ResidualModuleMI"

class EEGInceptionMI(EEGModuleMixin, nn.Module):
    """EEG-Inception for motor imagery from Zhang et al (2021). fft_conv is
    accepted for compatibility; the direct convolution is always used."""
    n_convs: int
    n_filters: int
    kernel_unit_s: float
    ensuredims: Ensure4d
    dimshuffle: Rearrange
    initial_inception_module: _InceptionModuleMI
    intermediate_inception_modules_1: nn.ModuleList
    residual_block_1: _ResidualModuleMI
    intermediate_inception_modules_2: nn.ModuleList
    residual_block_2: _ResidualModuleMI
    ave_pooling: nn.AvgPool2d
    flat: nn.Flatten
    final_layer: nn.Sequential

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, n_convs: int = 5,
                 n_filters: int = 48, kernel_unit_s: float = 0.1, activation = nn.ReLU,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 fft_conv: Optional[bool] = None):
        sf = sfreq
        if sf is None and (n_times is None or input_window_seconds is None):
            sf = 250.0
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sf)
        nn.Module.__init__(self)
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.kernel_unit_s = kernel_unit_s
        self.ensuredims = Ensure4d()
        self.add_module("ensuredims", self.ensuredims)
        self.dimshuffle = Rearrange("batch C T 1 -> batch C 1 T")
        self.add_module("dimshuffle", self.dimshuffle)
        self.mapping["fc.weight"] = "final_layer.fc.weight"
        self.mapping["fc.bias"] = "final_layer.fc.bias"
        self.initial_inception_module = _InceptionModuleMI(self.n_chans, n_filters, n_convs, kernel_unit_s, self.sfreq, activation)
        self.add_module("initial_inception_module", self.initial_inception_module)
        ic = (n_convs + 1) * n_filters
        l1 = List[nn.Module]()
        for _ in range(2):
            l1.append(_InceptionModuleMI(ic, n_filters, n_convs, kernel_unit_s, self.sfreq, activation))
        self.intermediate_inception_modules_1 = nn.ModuleList(l1)
        self.add_module("intermediate_inception_modules_1", self.intermediate_inception_modules_1)
        self.residual_block_1 = _ResidualModuleMI(self.n_chans, ic, activation)
        self.add_module("residual_block_1", self.residual_block_1)
        l2 = List[nn.Module]()
        for _ in range(3):
            l2.append(_InceptionModuleMI(ic, n_filters, n_convs, kernel_unit_s, self.sfreq, activation))
        self.intermediate_inception_modules_2 = nn.ModuleList(l2)
        self.add_module("intermediate_inception_modules_2", self.intermediate_inception_modules_2)
        self.residual_block_2 = _ResidualModuleMI(ic, ic, activation)
        self.add_module("residual_block_2", self.residual_block_2)
        self.ave_pooling = nn.AvgPool2d(kernel_size=(1, self.n_times))
        self.add_module("ave_pooling", self.ave_pooling)
        self.flat = nn.Flatten()
        self.add_module("flat", self.flat)
        module = nn.Sequential()
        module.add_module("fc", nn.Linear(in_features=ic, out_features=self.n_outputs, bias=True))
        module.add_module("out_fun", nn.Identity())
        self.final_layer = module
        self.add_module("final_layer", module)

    def forward(self, X: Tensor) -> Tensor:
        X = self.ensuredims(X)
        X = self.dimshuffle(X)
        res1 = self.residual_block_1(X)
        out = self.initial_inception_module(X)
        for layer in self.intermediate_inception_modules_1.children():
            out = layer(out)
        out = out + res1
        res2 = self.residual_block_2(out)
        for layer in self.intermediate_inception_modules_2.children():
            out = layer(out)
        out = res2 + out
        out = self.ave_pooling(out)
        out = self.flat(out)
        return self.final_layer(out)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "EEGInceptionMI"
