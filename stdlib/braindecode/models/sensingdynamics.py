# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# SensingDynamics (16-channel emg2pose adaptation). forward(x) only: the
# optional precomputed x_lowpass argument is not supported in Codon.

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import SmoothMaximumUnit
from scipy.signal import butter

class _ButterworthLowpass(nn.Module):
    """Zero-phase Butterworth low-pass applied in the frequency domain
    (|H(e^{jw})|^2 times the rFFT spectrum)."""
    a_coeffs: Tensor
    b_coeffs: Tensor
    _b: List[float]
    _a: List[float]
    def __init__(self, sfreq: float, cutoff_hz: float, order: int):
        super().__init__()
        b, a = butter(order, cutoff_hz, btype="low", fs=sfreq)
        self._b = [float(b[i]) for i in range(b.size)]
        self._a = [float(a[i]) for i in range(a.size)]
        self.a_coeffs = torch.tensor(self._a, dtype=torch.float64)
        self.register_buffer("a_coeffs", self.a_coeffs)
        self.b_coeffs = torch.tensor(self._b, dtype=torch.float64)
        self.register_buffer("b_coeffs", self.b_coeffs)
    def forward(self, x: Tensor) -> Tensor:
        n_times = x.shape[-1]
        nf = n_times // 2 + 1
        b = self.b_coeffs.tolist()
        a = self.a_coeffs.tolist()
        resp = List[float](capacity=nf)
        for k in range(nf):
            w = 2 * math.pi * float(k) / n_times
            nr = 0.0
            ni = 0.0
            dr = 0.0
            di = 0.0
            for d in range(min(len(b), len(a))):
                c = math.cos(w * d)
                s = -math.sin(w * d)
                nr += b[d] * c
                ni += b[d] * s
                dr += a[d] * c
                di += a[d] * s
            den = dr * dr + di * di
            hr = (nr * dr + ni * di) / den
            hi = (ni * dr - nr * di) / den
            m = math.sqrt(hr * hr + hi * hi)
            resp.append(m * m)
        xin = x.float() if x.dtype != torch.float64 else x
        spec = torch.view_as_real(torch.fft.rfft(xin, dim=-1))
        r = torch.tensor(resp, dtype=xin.dtype).unsqueeze(-1)
        out = torch.fft.irfft(torch.view_as_complex(spec * r), n=n_times, dim=-1)
        return out.to(x.dtype)
    def _get_name(self) -> str:
        return "_ButterworthLowpass"

class _ConvBlock(nn.Module):
    conv: nn.Conv2d
    batch_norm: nn.BatchNorm2d
    activation: nn.Module
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1), dilation: Tuple[int, int] = (1, 1), activation = SmoothMaximumUnit):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, dilation=dilation)
        self.add_module("conv", self.conv)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.add_module("batch_norm", self.batch_norm)
        self.activation = activation()
        self.add_module("activation", self.activation)
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.batch_norm(self.conv(x)))
    def _get_name(self) -> str:
        return "_ConvBlock"

def _conv_output_size(n: int, kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    return (n - dilation * (kernel_size - 1) - 1) // stride + 1

def _sd_init(m: nn.Module):
    n = m._get_name()
    if n == "Conv1d" or n == "Conv2d" or n == "Linear":
        for i in range(len(m._pnames)):
            p = m._params[i]
            if p is not None:
                if m._pnames[i] == "weight":
                    init.trunc_normal_(p.__val__(), std=0.02)
                elif m._pnames[i] == "bias":
                    init.zeros_(p.__val__())
    elif n in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm"):
        for i in range(len(m._pnames)):
            p = m._params[i]
            if p is not None:
                if m._pnames[i] == "weight":
                    init.ones_(p.__val__())
                elif m._pnames[i] == "bias":
                    init.zeros_(p.__val__())

class SensingDynamics(EEGModuleMixin, nn.Module):
    """SensingDynamics (emg2pose) from Salter et al (2024)."""
    receptive_field_samples: int
    lowpass: _ButterworthLowpass
    to_feature_plane: Rearrange
    circular_pad: nn.CircularPad2d
    conv1: _ConvBlock
    conv_dropout: nn.Dropout2d
    conv2: _ConvBlock
    conv3: _ConvBlock
    to_sequence: Rearrange
    sequence_to_channels: Rearrange
    channels_to_sequence: Rearrange
    mlp: nn.Sequential
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 temporal_channels: int = 256, mid_channels: int = 32, spatial_channels: int = 64,
                 mlp_hidden: int = 512, drop_prob: float = 0.25, mlp_drop_prob: float = 0.4, lowpass_hz: float = 20.0,
                 lowpass_order: int = 4, temporal_kernel: Tuple[int, int] = (1, 31),
                 temporal_stride: Tuple[int, int] = (1, 8), mid_kernel: Tuple[int, int] = (8, 18),
                 mid_dilation: Tuple[int, int] = (2, 1), spatial_kernel: Tuple[int, int] = (3, 1),
                 electrode_wrap: int = 4, activation = SmoothMaximumUnit):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        for name, v in (("temporal_kernel", temporal_kernel), ("temporal_stride", temporal_stride),
                        ("mid_kernel", mid_kernel), ("mid_dilation", mid_dilation), ("spatial_kernel", spatial_kernel)):
            if v[0] <= 0 or v[1] <= 0:
                raise ValueError(f"{name} must contain two positive integers.")
        if electrode_wrap < 0:
            raise ValueError(f"electrode_wrap must be non-negative; got {electrode_wrap}.")
        if lowpass_order <= 0:
            raise ValueError(f"lowpass_order must be positive; got {lowpass_order}.")
        if self.n_chans != 16:
            raise ValueError(f"SensingDynamics implements the 16-channel emg2pose adaptation; got n_chans={self.n_chans}.")
        rf = 1
        sf = 1
        for k, d, s in ((temporal_kernel[1], 1, temporal_stride[1]), (mid_kernel[1], mid_dilation[1], 1),
                        (spatial_kernel[1], 1, 1)):
            rf += (k - 1) * d * sf
            sf *= s
        self.receptive_field_samples = rf
        if self.n_times < rf:
            raise ValueError(f"SensingDynamics requires at least {rf} input samples; got {self.n_times}.")
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive; got {self.sfreq}.")
        if not (0 < lowpass_hz < self.sfreq / 2):
            raise ValueError("lowpass_hz must lie between 0 and the Nyquist frequency")
        self.lowpass = _ButterworthLowpass(self.sfreq, lowpass_hz, lowpass_order)
        self.add_module("lowpass", self.lowpass)
        self.to_feature_plane = Rearrange("batch nchans ntimes -> batch 1 nchans ntimes")
        self.add_module("to_feature_plane", self.to_feature_plane)
        self.circular_pad = nn.CircularPad2d(padding=(0, 0, electrode_wrap, electrode_wrap))
        self.add_module("circular_pad", self.circular_pad)
        self.conv1 = _ConvBlock(2, temporal_channels, temporal_kernel, stride=temporal_stride, activation=activation)
        self.add_module("conv1", self.conv1)
        self.conv_dropout = nn.Dropout2d(p=drop_prob)
        self.add_module("conv_dropout", self.conv_dropout)
        self.conv2 = _ConvBlock(temporal_channels, mid_channels, mid_kernel, dilation=mid_dilation, activation=activation)
        self.add_module("conv2", self.conv2)
        self.conv3 = _ConvBlock(mid_channels, spatial_channels, spatial_kernel, activation=activation)
        self.add_module("conv3", self.conv3)
        ef = _conv_output_size(self.n_chans, temporal_kernel[0], temporal_stride[0])
        ef += 2 * electrode_wrap
        ef = _conv_output_size(ef, mid_kernel[0], 1, mid_dilation[0])
        ef = _conv_output_size(ef, spatial_kernel[0])
        if ef < 1:
            raise ValueError(f"The configured kernels consume the electrode axis: n_chans={self.n_chans} leaves {ef} positions after conv3. Widen electrode_wrap or shrink the kernels.")
        self.to_sequence = Rearrange("batch features nelectrodes ntimes -> batch ntimes (features nelectrodes)")
        self.add_module("to_sequence", self.to_sequence)
        self.sequence_to_channels = Rearrange("batch ntimes njoints -> batch njoints ntimes")
        self.add_module("sequence_to_channels", self.sequence_to_channels)
        self.channels_to_sequence = Rearrange("batch njoints ntimes -> batch ntimes njoints")
        self.add_module("channels_to_sequence", self.channels_to_sequence)
        self.mlp = nn.Sequential(nn.Dropout(p=mlp_drop_prob), nn.Linear(spatial_channels * ef, mlp_hidden), activation(),
                                 nn.Linear(mlp_hidden, mlp_hidden), activation())
        self.add_module("mlp", self.mlp)
        self.final_layer = nn.Linear(mlp_hidden, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        self.apply(_sd_init)

    def _forward(self, x: Tensor) -> Tensor:
        n_times = x.shape[-1]
        xl = self.lowpass(x)
        stacked = torch.cat([self.to_feature_plane(x), self.to_feature_plane(xl)], dim=1)
        t = self.conv_dropout(self.conv1(stacked))
        s = self.conv3(self.conv2(self.circular_pad(t)))
        pred = self.final_layer(self.mlp(self.to_sequence(s)))
        pc = F.interpolate(self.sequence_to_channels(pred), size=n_times, mode="linear", align_corners=False)
        return self.channels_to_sequence(pc)

    def forward(self, x: Tensor) -> Tensor:
        if self.training and x.shape[0] == 1:
            bns = [m for m in self.modules() if "BatchNorm" in m._get_name() and m.training]
            for m in bns:
                m.train(False)
            y = self._forward(x)
            for m in bns:
                m.train(True)
            return y
        return self._forward(x)

    def _get_name(self) -> str:
        return "SensingDynamics"
