# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.parametrizations import weight_norm
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Chomp1d, Ensure4d, SqueezeFinalOutput

class _TemporalBlock(nn.Module):
    conv1: nn.Conv1d
    chomp1: Chomp1d
    relu1: nn.Module
    dropout1: nn.Dropout1d
    conv2: nn.Conv1d
    chomp2: Chomp1d
    relu2: nn.Module
    dropout2: nn.Dropout1d
    downsample: Optional[nn.Conv1d]
    relu: nn.Module

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int,
                 drop_prob: float, activation = nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        weight_norm(self.conv1)
        self.add_module("conv1", self.conv1)
        self.chomp1 = Chomp1d(padding)
        self.add_module("chomp1", self.chomp1)
        self.relu1 = activation()
        self.add_module("relu1", self.relu1)
        self.dropout1 = nn.Dropout1d(drop_prob)
        self.add_module("dropout1", self.dropout1)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        weight_norm(self.conv2)
        self.add_module("conv2", self.conv2)
        self.chomp2 = Chomp1d(padding)
        self.add_module("chomp2", self.chomp2)
        self.relu2 = activation()
        self.add_module("relu2", self.relu2)
        self.dropout2 = nn.Dropout1d(drop_prob)
        self.add_module("dropout2", self.dropout2)
        self.downsample = None
        if n_inputs != n_outputs:
            ds = nn.Conv1d(n_inputs, n_outputs, 1)
            self.downsample = ds
            self.add_module("downsample", ds)
        self.relu = activation()
        self.add_module("relu", self.relu)
        # As in torch, these write into the recomputed weights (RNG is consumed,
        # the parametrized originals are unchanged).
        init.normal_(self.conv1.weight, 0.0, 0.01)
        init.normal_(self.conv2.weight, 0.0, 0.01)
        if self.downsample is not None:
            init.normal_(self.downsample.__val__().weight, 0.0, 0.01)

    def forward(self, x: Tensor) -> Tensor:
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))
        res = x
        if self.downsample is not None:
            res = self.downsample.__val__()(x)
        return self.relu(out + res)

    def _get_name(self) -> str:
        return "_TemporalBlock"

class _FinalLayer(nn.Module):
    fc: nn.Linear
    out_fun: nn.Identity
    squeeze: SqueezeFinalOutput

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.add_module("fc", self.fc)
        self.out_fun = nn.Identity()
        self.add_module("out_fun", self.out_fun)
        self.squeeze = SqueezeFinalOutput()
        self.add_module("squeeze", self.squeeze)

    def _fwd(self, x: Tensor, batch_size: int, time_size: int, min_len: int) -> Tensor:
        fc_out = self.fc(x.view(batch_size * time_size, x.size(2)))
        fc_out = self.out_fun(fc_out)
        fc_out = fc_out.view(batch_size, time_size, fc_out.size(1))
        out_size = 1 + max(0, time_size - min_len)
        out = fc_out[:, -out_size:, :].transpose(1, 2)
        return self.squeeze(out.unsqueeze(3))

    def _get_name(self) -> str:
        return "_FinalLayer"

class TCN(nn.Module):
    """Temporal Convolutional Network (Bai et al 2018), used by BDTCN."""
    mapping: Dict[str, str]
    ensuredims: Ensure4d
    temporal_blocks: nn.Sequential
    final_layer: _FinalLayer
    min_len: int

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_blocks: int = 3,
                 n_filters: int = 30, kernel_size: int = 5, drop_prob: float = 0.5, activation = nn.ReLU):
        super().__init__()
        self.mapping = {"fc.weight": "final_layer.fc.weight", "fc.bias": "final_layer.fc.bias"}
        self.ensuredims = Ensure4d()
        self.add_module("ensuredims", self.ensuredims)
        tb = nn.Sequential()
        for i in range(n_blocks):
            n_in = n_chans.__val__() if i == 0 else n_filters
            d = 1 << i
            tb.add_module(f"temporal_block_{i}", _TemporalBlock(n_in, n_filters, kernel_size, 1, d,
                                                                (kernel_size - 1) * d, drop_prob, activation))
        self.temporal_blocks = tb
        self.add_module("temporal_blocks", tb)
        self.final_layer = _FinalLayer(n_filters, n_outputs.__val__())
        self.add_module("final_layer", self.final_layer)
        self.min_len = 1
        for i in range(n_blocks):
            self.min_len += 2 * (kernel_size - 1) * (1 << i)
        self.train()

    def forward(self, x: Tensor) -> Tensor:
        x = self.ensuredims(x)
        batch_size = x.shape[0]
        time_size = x.shape[2]
        assert time_size >= self.min_len
        x = x.squeeze(3)
        x = self.temporal_blocks(x)
        x = x.transpose(1, 2).contiguous()
        return self.final_layer._fwd(x, batch_size, time_size, self.min_len)

    def _get_name(self) -> str:
        return "TCN"

class BDTCN(EEGModuleMixin, nn.Module):
    """Braindecode TCN from Gemein et al (2020)."""
    base_tcn: TCN
    final_layer: nn.Sequential

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 sfreq: Optional[float] = None, input_window_seconds: Optional[float] = None, n_blocks: int = 3,
                 n_filters: int = 30, kernel_size: int = 5, drop_prob: float = 0.5, activation = nn.ReLU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.base_tcn = TCN(n_chans=self.n_chans, n_outputs=self.n_outputs, n_blocks=n_blocks, n_filters=n_filters,
                            kernel_size=kernel_size, drop_prob=drop_prob, activation=activation)
        self.add_module("base_tcn", self.base_tcn)
        if self._n_times is not None and self._n_times.__val__() < self.base_tcn.min_len:
            raise ValueError(f"n_times ({self._n_times}) must be at least {self.base_tcn.min_len}, the receptive field of the temporal blocks.")
        self.final_layer = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.base_tcn(x))

    def _get_name(self) -> str:
        return "BDTCN"
