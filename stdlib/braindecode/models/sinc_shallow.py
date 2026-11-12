# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin

class _SincFilter(nn.Module):
    num_filters: int
    kernel_size: int
    sfreq: float
    min_freq: float
    padding: str
    window: Tensor
    n_pi: Tensor
    bandwidths: Tensor
    low_freqs: Tensor
    ones: Tensor

    def __init__(self, low_freqs: Tensor, kernel_size: int, sfreq: float, bandwidth: float = 4.0,
                 min_freq: float = 1.0, padding: str = "same"):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        self.num_filters = low_freqs.numel()
        self.kernel_size = kernel_size
        self.sfreq = sfreq
        self.min_freq = min_freq
        self.padding = padding.lower()
        # torch.hamming_window(kernel_size, periodic=False), first half
        h = kernel_size // 2
        wv = [0.54 - 0.46 * math.cos(2.0 * math.pi * i / (kernel_size - 1)) for i in range(h)]
        self.window = torch.tensor(wv, dtype=torch.float32).unsqueeze(-1)
        self.register_buffer("window", self.window)
        n_pi = torch.arange(-h, 0, dtype=torch.float32) / sfreq * 2 * math.pi
        self.n_pi = n_pi.unsqueeze(-1)
        self.register_buffer("n_pi", self.n_pi)
        self.bandwidths = nn.Parameter(torch.full([1, self.num_filters], bandwidth))
        self.register_parameter("bandwidths", self.bandwidths)
        self.low_freqs = nn.Parameter(low_freqs.unsqueeze(0))
        self.register_parameter("low_freqs", self.low_freqs)
        self.ones = torch.ones(1, 1, 1, self.num_filters)
        self.register_buffer("ones", self.ones)

    def build_sinc_filters(self) -> Tensor:
        low = self.min_freq + torch.abs(self.low_freqs)
        high = torch.clamp(low + torch.abs(self.bandwidths), min=self.min_freq, max=self.sfreq / 2.0)
        bw = high - low
        fl = (torch.sin(self.n_pi * high) - torch.sin(self.n_pi * low)) / (self.n_pi / 2.0)
        fl = fl * self.window
        fl = fl / (2.0 * bw)
        fl = fl.unsqueeze(0).unsqueeze(2)
        fr = torch.flip(fl, dims=[1])
        f = torch.cat([fl, self.ones, fr], dim=1)
        return f / torch.std(f)

    def forward(self, inputs: Tensor) -> Tensor:
        filters = self.build_sinc_filters()
        inputs = inputs.permute(0, 3, 1, 2)
        filters = filters.permute(3, 2, 0, 1)
        out = F.conv2d(inputs, filters, None, 1, self.padding)
        return out.permute(0, 2, 3, 1)

    def _get_name(self) -> str:
        return "_SincFilter"

class SincShallowNet(EEGModuleMixin, nn.Module):
    """Sinc-ShallowNet from Borra et al (2020)."""
    n_filters: int
    ensuredims: Rearrange
    sinc_filter_layer: _SincFilter
    depthwiseconv: nn.Sequential
    temporal_aggregation: nn.Sequential
    final_layer: nn.Linear

    def __init__(self, num_time_filters: int = 32, time_filter_len: int = 33, depth_multiplier: int = 2,
                 activation = nn.ELU, drop_prob: float = 0.5, first_freq: float = 5.0, min_freq: float = 1.0,
                 freq_stride: float = 1.0, padding: str = "same", bandwidth: float = 4.0, pool_size: int = 55,
                 pool_stride: int = 12, n_chans: Optional[int] = None, n_outputs: Optional[int] = None,
                 n_times: Optional[int] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, chs_info: Optional[List[Dict[str, str]]] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        low_freqs = torch.arange(first_freq, first_freq + num_time_filters * freq_stride, freq_stride, dtype=torch.float32)
        self.n_filters = low_freqs.numel()
        nt = 0
        if padding.lower() == "valid":
            nt = self.n_times - time_filter_len + 1
        elif padding.lower() == "same":
            nt = self.n_times
        else:
            raise ValueError("Padding must be 'valid' or 'same'.")
        if nt < pool_size:
            raise ValueError(f"n_times ({self.n_times}) leaves only {nt} samples after the temporal sinc filter, fewer than pool_size ({pool_size}).")
        sap = (nt - pool_size) // pool_stride + 1
        flat = num_time_filters * depth_multiplier * sap
        self.ensuredims = Rearrange("batch chans times -> batch chans times 1")
        self.add_module("ensuredims", self.ensuredims)
        self.sinc_filter_layer = _SincFilter(low_freqs, time_filter_len, self.sfreq, bandwidth, min_freq, padding)
        self.add_module("sinc_filter_layer", self.sinc_filter_layer)
        nf = self.n_filters
        self.depthwiseconv = nn.Sequential(nn.BatchNorm2d(nf, momentum=0.99),
            nn.Conv2d(nf, depth_multiplier * nf, kernel_size=(self.n_chans, 1), groups=nf, bias=False))
        self.add_module("depthwiseconv", self.depthwiseconv)
        self.temporal_aggregation = nn.Sequential(nn.BatchNorm2d(depth_multiplier * nf, momentum=0.99), activation(),
            nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)), nn.Dropout(p=drop_prob), nn.Flatten())
        self.add_module("temporal_aggregation", self.temporal_aggregation)
        self.final_layer = nn.Linear(flat, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ensuredims(x)
        x = self.sinc_filter_layer(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.depthwiseconv(x)
        x = self.temporal_aggregation(x)
        return self.final_layer(x)

    def _get_name(self) -> str:
        return "SincShallowNet"
