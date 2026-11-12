# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# GeneralizedGaussianFilter (braindecode.modules.filter) and the Hilbert /
# phase-locking-value helpers from braindecode.functional used by EEGMiner.

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def _repeat_tuple(t: List[float], k: int) -> List[float]:
    out = List[float]()
    for _ in range(k):
        for v in t:
            out.append(float(v))
    return out

class GeneralizedGaussianFilter(nn.Module):
    """Generalized Gaussian filter in the frequency domain (Ludwig et al 2024)."""
    in_channels: int
    out_channels: int
    sequence_length: int
    sample_rate: float
    inverse_fourier: bool
    affine_group_delay: bool
    clamp_f_mean: Tuple[float, float]
    n_range: Tensor
    f_mean: Tensor
    bandwidth: Tensor
    shape: Tensor
    group_delay: Tensor

    def __init__(self, in_channels: int, out_channels: int, sequence_length: int, sample_rate: float,
                 inverse_fourier: bool = True, affine_group_delay: bool = False, group_delay = (20.0,),
                 f_mean = (23.0,), bandwidth = (44.0,), shape = (2.0,), clamp_f_mean = (1.0, 45.0)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.inverse_fourier = inverse_fourier
        self.affine_group_delay = affine_group_delay
        self.clamp_f_mean = (float(clamp_f_mean[0]), float(clamp_f_mean[1]))
        fm = [float(v) for v in f_mean]
        bw = [float(v) for v in bandwidth]
        sh = [float(v) for v in shape]
        gd = [float(v) for v in group_delay]
        if out_channels % in_channels != 0:
            raise ValueError("out_channels has to be multiple of in_channels")
        if len(fm) * in_channels != out_channels:
            raise ValueError("len(f_mean) * in_channels must equal out_channels")
        if len(bw) * in_channels != out_channels:
            raise ValueError("len(bandwidth) * in_channels must equal out_channels")
        if len(sh) * in_channels != out_channels:
            raise ValueError("len(shape) * in_channels must equal out_channels")
        half = sample_rate / 2
        # scipy.fft.fftfreq(n, d=1/sr)[: n // 2] + [sr / 2] (float64 values)
        nr = List[float]()
        val = 1.0 / (sequence_length * (1.0 / sample_rate))
        for k in range(sequence_length // 2):
            nr.append(k * val)
        nr.append(half)
        self.n_range = nn.Parameter(torch.tensor(nr) / half, requires_grad=False)
        self.register_parameter("n_range", self.n_range)
        self.f_mean = nn.Parameter(torch.tensor(_repeat_tuple(fm, in_channels)) / half, requires_grad=True)
        self.register_parameter("f_mean", self.f_mean)
        self.bandwidth = nn.Parameter(torch.tensor(_repeat_tuple(bw, in_channels)) / half, requires_grad=True)
        self.register_parameter("bandwidth", self.bandwidth)
        self.shape = nn.Parameter(torch.tensor(_repeat_tuple(sh, in_channels)), requires_grad=True)
        self.register_parameter("shape", self.shape)
        self.group_delay = nn.Parameter(torch.tensor(_repeat_tuple(gd, in_channels)) / 1000,
                                        requires_grad=affine_group_delay)
        self.register_parameter("group_delay", self.group_delay)

    @staticmethod
    def exponential_power(x: Tensor, mean: Tensor, fwhm: Tensor, shape: Tensor) -> Tensor:
        mean = mean.unsqueeze(1)
        fwhm = fwhm.unsqueeze(1)
        shape = shape.unsqueeze(1)
        log2 = torch.tensor(2.0, dtype=x.dtype).log()
        scale = fwhm / (2 * log2 ** (1 / shape))
        return torch.exp(-((((x - mean).abs() + 1e-8) / scale) ** shape))

    def construct_filters(self) -> Tensor:
        half = self.sample_rate / 2
        f_mean = torch.clamp(self.f_mean.clone(), min=self.clamp_f_mean[0] / half, max=self.clamp_f_mean[1] / half)
        bandwidth = torch.clamp(self.bandwidth.clone(), min=1.0 / half, max=1.0)
        shape = torch.clamp(self.shape.clone(), min=2.0, max=3.0)
        with torch.no_grad():
            self.f_mean.copy_(f_mean)
            self.bandwidth.copy_(bandwidth)
            self.shape.copy_(shape)
        mag = GeneralizedGaussianFilter.exponential_power(self.n_range, f_mean, bandwidth, shape * 8 - 14)
        mag = mag / mag.amax(dim=-1, keepdim=True)
        phase = torch.linspace(0, self.sample_rate, self.sequence_length // 2 + 1, dtype=mag.dtype)
        phase = phase.expand(mag.shape[0], -1)
        pha = -self.group_delay.unsqueeze(-1) * phase * math.pi
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        return torch.stack([real, imag], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        filters = self.construct_filters()
        dtype = x.dtype
        y = torch.view_as_real(torch.fft.rfft(x.float(), dim=-1))
        y = torch.repeat_interleave(y, self.out_channels // self.in_channels, dim=-3)
        y = y * filters
        if self.inverse_fourier:
            y = torch.fft.irfft(torch.view_as_complex(y), n=self.sequence_length, dim=-1)
        return y.to(dtype)

    def _get_name(self) -> str:
        return "GeneralizedGaussianFilter"

def hilbert_freq(x: Tensor, forward_fourier: bool = True) -> Tensor:
    """Analytic signal as (..., seq_len, 2) (braindecode.functional.hilbert_freq)."""
    input_dtype = x.dtype
    seq_len = 0
    if forward_fourier:
        seq_len = x.shape[-1]
        x = torch.view_as_real(torch.fft.rfft(x.float(), dim=-1))
    else:
        x = x.float()
        seq_len = 2 * (x.shape[-2] - 1)
    nf = x.shape[-2]
    sc = [2.0] * nf
    sc[0] = 1.0
    if forward_fourier and seq_len % 2 == 0:
        sc[nf - 1] = 1.0
    x = x * torch.tensor(sc, dtype=x.dtype).unsqueeze(-1)
    x = F.pad(x, [0, 0, 0, seq_len - nf])
    x = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(x), dim=-1))
    return x.to(input_dtype)

def plv_time(x: Tensor, forward_fourier: bool = True, epsilon: float = 1e-6) -> Tensor:
    """Phase locking value matrix (..., channels, channels)."""
    a = hilbert_freq(x, forward_fourier)
    amplitude = torch.sqrt(a[..., 0] ** 2 + a[..., 1] ** 2 + 1e-6)
    u = a / amplitude.unsqueeze(-1)
    real = u[..., 0].contiguous()
    imag = u[..., 1].contiguous()
    real_t = real.transpose(-2, -1).contiguous()
    imag_t = imag.transpose(-2, -1).contiguous()
    cr = torch.matmul(real, real_t) + torch.matmul(imag, imag_t)
    ci = torch.matmul(real, imag_t) - torch.matmul(imag, real_t)
    time = amplitude.shape[-1]
    return 1 / time * torch.sqrt(cr ** 2 + ci ** 2 + epsilon)
