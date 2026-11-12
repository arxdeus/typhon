# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin

class SyncNet(EEGModuleMixin, nn.Module):
    """SyncNet from Li et al (2017)."""
    num_filters: int
    filter_width: int
    pool_size: int
    activation: nn.Module
    amplitude: Tensor
    omega: Tensor
    bias: Tensor
    classifier_input_size: int
    t: Tensor
    phi_ini: Tensor
    beta: Tensor
    pad_input: nn.ConstantPad1d
    pad_res: nn.ConstantPad1d
    pool: nn.MaxPool2d
    ensuredim: Rearrange
    final_layer: nn.Linear

    def __init__(self, n_chans: Optional[int] = None, n_times: Optional[int] = None, n_outputs: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, num_filters: int = 1, filter_width: int = 40, pool_size: int = 40,
                 activation = nn.ReLU, ampli_init_values = (-0.05, 0.05), omega_init_values = (0.0, 1.0),
                 beta_init_values = (0.0, 0.05), phase_init_values = (0.0, 0.05)):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.num_filters = num_filters
        self.filter_width = filter_width
        self.pool_size = pool_size
        self.activation = activation()
        self.add_module("activation", self.activation)
        nc = self.n_chans
        self.amplitude = nn.Parameter(torch.empty(1, 1, nc, num_filters).uniform_(float(ampli_init_values[0]), float(ampli_init_values[1])))
        self.register_parameter("amplitude", self.amplitude)
        self.omega = nn.Parameter(torch.empty(1, 1, 1, num_filters).uniform_(float(omega_init_values[0]), float(omega_init_values[1])))
        self.register_parameter("omega", self.omega)
        self.bias = nn.Parameter(torch.zeros(num_filters))
        self.register_parameter("bias", self.bias)
        self.classifier_input_size = int(math.ceil(float(self.n_times) / float(pool_size))) * num_filters
        lo = 0
        if filter_width % 2 == 0:
            lo = -(filter_width // 2)
        else:
            lo = -((filter_width - 1) // 2)
        self.t = nn.Parameter(torch.tensor([float(lo + i) for i in range(filter_width)], dtype=torch.float32).reshape(1, filter_width, 1, 1))
        self.register_parameter("t", self.t)
        self.phi_ini = nn.Parameter(torch.empty(1, 1, nc, num_filters).normal_(float(phase_init_values[0]), float(phase_init_values[1])))
        self.register_parameter("phi_ini", self.phi_ini)
        self.beta = nn.Parameter(torch.empty(1, 1, 1, num_filters).uniform_(float(beta_init_values[0]), float(beta_init_values[1])))
        self.register_parameter("beta", self.beta)
        P = filter_width - 2
        pad = (P // 2, P // 2 + 1) if P % 2 == 0 else (P // 2, P // 2)
        self.pad_input = nn.ConstantPad1d(pad, 0.0)
        self.add_module("pad_input", self.pad_input)
        self.pad_res = nn.ConstantPad1d(pad, 0.0)
        self.add_module("pad_res", self.pad_res)
        self.pool = nn.MaxPool2d((1, pool_size), stride=(1, pool_size))
        self.add_module("pool", self.pool)
        self.ensuredim = Rearrange("batch ch time -> batch ch 1 time")
        self.add_module("ensuredim", self.ensuredim)
        self.final_layer = nn.Linear(self.classifier_input_size, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ensuredim(x)
        W_osc = self.amplitude * torch.cos(self.t * self.omega + self.phi_ini)
        W = W_osc * torch.exp(-(torch.pow(self.t, 2) * self.beta))
        W = W.permute(3, 2, 0, 1).contiguous()
        res = F.conv2d(self.pad_input(x), W.to(x.dtype), self.bias, 1)
        res = self.pool(self.pad_res(res))
        flat = res.view(-1, self.classifier_input_size)
        with torch.no_grad():
            self.beta.clamp_(min=0.0)
        return self.final_layer(self.activation(flat))

    def _get_name(self) -> str:
        return "SyncNet"
