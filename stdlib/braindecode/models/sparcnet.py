# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin

class _DenseLayer(nn.Sequential):
    drop_prob: float

    def __init__(self, in_channels: int, growth_rate: int, bottleneck_size: int, drop_prob: float = 0.5,
                 conv_bias: bool = True, batch_norm: bool = True, activation = nn.ELU, kernel_size_conv1: int = 1,
                 kernel_size_conv2: int = 3, stride_conv1: int = 1, stride_conv2: int = 1, padding_conv2: int = 1):
        super().__init__()
        if batch_norm:
            self.add_module("norm1", nn.BatchNorm1d(in_channels))
        self.add_module("elu1", activation())
        self.add_module("conv1", nn.Conv1d(in_channels, bottleneck_size * growth_rate, kernel_size=kernel_size_conv1,
                                           stride=stride_conv1, bias=conv_bias))
        if batch_norm:
            self.add_module("norm2", nn.BatchNorm1d(bottleneck_size * growth_rate))
        self.add_module("elu2", activation())
        self.add_module("conv2", nn.Conv1d(bottleneck_size * growth_rate, growth_rate, kernel_size=kernel_size_conv2,
                                           stride=stride_conv2, padding=padding_conv2, bias=conv_bias))
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for l in self.children():
            out = l(out)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        return torch.cat([x, out], dim=1)

    def _get_name(self) -> str:
        return "_DenseLayer"

def _kaiming_init(m):
    n = m._get_name()
    if n == "Conv1d":
        for i in range(len(m._pnames)):
            if m._pnames[i] == "weight" and m._params[i] is not None:
                nn.init.kaiming_normal_(m._params[i].__val__())
    elif n == "Linear":
        for i in range(len(m._pnames)):
            if m._pnames[i] == "bias" and m._params[i] is not None:
                nn.init.zeros_(m._params[i].__val__())
    elif n == "BatchNorm1d":
        for i in range(len(m._pnames)):
            if m._params[i] is not None:
                if m._pnames[i] == "weight":
                    nn.init.ones_(m._params[i].__val__())
                elif m._pnames[i] == "bias":
                    nn.init.zeros_(m._params[i].__val__())

class SPARCNet(EEGModuleMixin, nn.Module):
    """SPaRCNet from Jing et al (2023)."""
    encoder: nn.Sequential
    adaptative_pool: nn.AdaptiveAvgPool1d
    activation_layer: nn.Module
    flatten_layer: nn.Flatten
    final_layer: nn.Linear

    def __init__(self, n_chans: Optional[int] = None, n_times: Optional[int] = None, n_outputs: Optional[int] = None,
                 block_layers: int = 4, growth_rate: int = 16, bottleneck_size: int = 16, drop_prob: float = 0.5,
                 conv_bias: bool = True, batch_norm: bool = True, activation = nn.ELU, kernel_size_conv0: int = 7,
                 kernel_size_conv1: int = 1, kernel_size_conv2: int = 3, kernel_size_pool: int = 3,
                 stride_pool: int = 2, stride_conv0: int = 2, stride_conv1: int = 1, stride_conv2: int = 1,
                 padding_pool: int = 1, padding_conv0: int = 3, padding_conv2: int = 1, kernel_size_trans: int = 2,
                 stride_trans: int = 2, chs_info: Optional[List[Dict[str, str]]] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        oc = 1 << (int(math.floor(math.log2(self.n_chans))) + 1)
        enc = nn.Sequential()
        enc.add_module("conv0", nn.Conv1d(self.n_chans, oc, kernel_size=kernel_size_conv0, stride=stride_conv0,
                                          padding=padding_conv0, bias=conv_bias))
        enc.add_module("norm0", nn.BatchNorm1d(oc))
        enc.add_module("act_layer", activation())
        enc.add_module("pool0", nn.MaxPool1d(kernel_size=kernel_size_pool, stride=stride_pool, padding=padding_pool))
        nc = oc
        for nl in range(int(math.floor(math.log2(self.n_times // 4)))):
            blk = nn.Sequential()
            for il in range(block_layers):
                blk.add_module(f"denselayer{il + 1}", _DenseLayer(nc + il * growth_rate, growth_rate, bottleneck_size,
                               drop_prob, conv_bias, batch_norm, activation, kernel_size_conv1, kernel_size_conv2,
                               stride_conv1, stride_conv2, padding_conv2))
            enc.add_module(f"denseblock{nl + 1}", blk)
            nc = nc + block_layers * growth_rate
            tr = nn.Sequential()
            if batch_norm:
                tr.add_module("norm", nn.BatchNorm1d(nc))
            tr.add_module("elu", activation())
            tr.add_module("conv", nn.Conv1d(nc, nc // 2, kernel_size=1, stride=1, bias=conv_bias))
            tr.add_module("pool", nn.AvgPool1d(kernel_size=kernel_size_trans, stride=stride_trans))
            enc.add_module(f"transition{nl + 1}", tr)
            nc = nc // 2
        self.encoder = enc
        self.add_module("encoder", enc)
        self.adaptative_pool = nn.AdaptiveAvgPool1d(1)
        self.add_module("adaptative_pool", self.adaptative_pool)
        self.activation_layer = activation()
        self.add_module("activation_layer", self.activation_layer)
        self.flatten_layer = nn.Flatten()
        self.add_module("flatten_layer", self.flatten_layer)
        self.final_layer = nn.Linear(nc, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        self.apply(_kaiming_init)

    def forward(self, X: Tensor) -> Tensor:
        emb = self.activation_layer(self.adaptative_pool(self.encoder(X)))
        return self.final_layer(self.flatten_layer(emb))

    def _get_name(self) -> str:
        return "SPARCNet"
