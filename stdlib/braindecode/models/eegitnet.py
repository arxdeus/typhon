# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin, _get_output_shape
from braindecode.modules import DepthwiseConv2d, Ensure4d, InceptionBlock, LinearWithConstraint, MaxNormParametrize

class _TCBlock(nn.Module):
    pad: int
    tc1: nn.Sequential
    tc2: nn.Sequential

    def __init__(self, in_ch: int, kernel_length: int, dilatation: int, padding: int, drop_prob: float = 0.4,
                 activation = nn.ELU):
        super().__init__()
        self.pad = padding
        self.tc1 = nn.Sequential(DepthwiseConv2d(in_ch, kernel_size=(1, kernel_length), depth_multiplier=1,
                                 dilation=(1, dilatation), bias=False, padding="valid"),
                                 nn.BatchNorm2d(in_ch), activation(), nn.Dropout(drop_prob))
        self.add_module("tc1", self.tc1)
        self.tc2 = nn.Sequential(DepthwiseConv2d(in_ch, kernel_size=(1, kernel_length), depth_multiplier=1,
                                 dilation=(1, dilatation), bias=False, padding="valid"),
                                 nn.BatchNorm2d(in_ch), activation(), nn.Dropout(drop_prob))
        self.add_module("tc2", self.tc2)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        p = [self.pad, 0, 0, 0, 0, 0, 0, 0]
        x = F.pad(x, p)
        x = self.tc1(x)
        x = F.pad(x, p)
        return self.tc2(x) + residual

    def _get_name(self) -> str:
        return "_TCBlock"

def _inception_branch(in_channels: int, out_channels: int, kernel_length: int, activation) -> nn.Module:
    sd = DepthwiseConv2d(out_channels, kernel_size=(in_channels, 1), depth_multiplier=1, bias=False, padding="valid")
    register_parametrization(sd, "weight", MaxNormParametrize(max_norm=1.0))
    return nn.Sequential(nn.Conv2d(1, out_channels, kernel_size=(1, kernel_length), padding="same", bias=False),
                         nn.BatchNorm2d(out_channels), sd, nn.BatchNorm2d(out_channels), activation())

class EEGITNet(EEGModuleMixin, nn.Sequential):
    """EEG-ITNet from Salami et al (2022)."""
    _itn: int

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, n_filters_time: int = 2, kernel_length: int = 16,
                 pool_kernel: int = 4, tcn_in_channel: int = 14, tcn_kernel_size: int = 4, tcn_padding: int = 3,
                 drop_prob: float = 0.4, tcn_dilatation: int = 1, activation = nn.ELU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Sequential.__init__(self)
        self._itn = 0
        self.mapping["classification.1.weight"] = "final_layer.parametrizations.weight.original"
        self.mapping["classification.1.bias"] = "final_layer.bias"
        self.add_module("input_preprocess", nn.Sequential(Ensure4d(), Rearrange("ba ch t 1 -> ba 1 ch t")))
        b = [_inception_branch(self.n_chans, n_filters_time, kernel_length, activation),
             _inception_branch(self.n_chans, n_filters_time * 2, kernel_length * 2, activation),
             _inception_branch(self.n_chans, n_filters_time * 4, kernel_length * 4, activation)]
        self.add_module("inception_block", InceptionBlock(b))
        self.add_module("pooling", nn.Sequential(nn.AvgPool2d(kernel_size=(1, pool_kernel)), nn.Dropout(drop_prob)))
        for i in range(4):
            m = 1 << i
            self.add_module(f"TC_block{i + 1}", _TCBlock(tcn_in_channel, tcn_kernel_size, tcn_dilatation * m,
                                                         tcn_padding * m, drop_prob, activation))
        self.add_module("dim_reduction", nn.Sequential(nn.Conv2d(tcn_in_channel, tcn_in_channel, kernel_size=(1, 1)),
                        nn.BatchNorm2d(tcn_in_channel), activation(), nn.AvgPool2d((1, tcn_kernel_size)), nn.Dropout(drop_prob)))
        self.add_module("flatten", nn.Flatten())
        nf = _get_output_shape(self, self.n_chans, self.n_times)
        self.add_module("final_layer", LinearWithConstraint(nf[len(nf) - 1], self.n_outputs, max_norm=0.25))

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "EEGITNet"
