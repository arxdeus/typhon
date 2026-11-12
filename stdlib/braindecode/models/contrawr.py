# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
import torch.fft
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _bn_off_if_batch_one, _bn_restore

class _ResBlock(nn.Module):
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    relu: nn.Module
    conv2: nn.Conv2d
    bn2: nn.BatchNorm2d
    maxpool: nn.MaxPool2d
    downsample: nn.Sequential
    use_downsampling: bool
    pooling: bool
    dropout: nn.Dropout

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_downsampling: bool = True,
                 pooling: bool = True, kernel_size: int = 3, padding: int = 1, drop_prob: float = 0.5, activation = nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.add_module("conv1", self.conv1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.add_module("bn1", self.bn1)
        self.relu = activation()
        self.add_module("relu", self.relu)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.add_module("conv2", self.conv2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.add_module("bn2", self.bn2)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.add_module("maxpool", self.maxpool)
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                                  padding=padding), nn.BatchNorm2d(out_channels))
        self.add_module("downsample", self.downsample)
        self.use_downsampling = use_downsampling
        self.pooling = pooling
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_downsampling:
            out = out + self.downsample(x)
        if self.pooling:
            out = self.maxpool(out)
        return self.dropout(out)

    def _get_name(self) -> str:
        return "_ResBlock"

class _STFTModule(nn.Module):
    # |STFT| with a rectangular window, center=True (reflect padding),
    # onesided, normalized: computed from frames with torch.fft.rfft.
    n_fft: int
    hop_length: int
    center: bool
    one_sided: bool
    normalized: bool

    def __init__(self, n_fft: int, hop_length: int, center: bool = True, onesided: bool = True,
                 return_complex: bool = True, normalized: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.one_sided = onesided
        self.normalized = normalized

    def forward(self, x: Tensor) -> Tensor:
        if not (self.center and self.one_sided):
            raise NotImplementedError("_STFTModule: only center=True, onesided=True")
        pad = self.n_fft // 2
        x = F.pad(x, [pad, pad], mode="reflect")
        n_frames = 1 + (x.shape[-1] - self.n_fft) // self.hop_length
        starts = torch.arange(n_frames) * self.hop_length
        index = (starts.unsqueeze(1) + torch.arange(self.n_fft).unsqueeze(0)).flatten()
        frames = x.index_select(-1, index).unflatten(-1, [n_frames, self.n_fft])
        spec = torch.abs(torch.fft.rfft(frames))
        if self.normalized:
            spec = spec / math.sqrt(float(self.n_fft))
        return spec.transpose(-1, -2).to(x.dtype)

    def _get_name(self) -> str:
        return "_STFTModule"

class ContraWR(EEGModuleMixin, nn.Module):
    """ContraWR encoder from Yang et al (2021)."""
    n_fft: int
    steps: int
    torch_stft: _STFTModule
    convs: nn.ModuleList
    adaptative_pool: nn.AdaptiveAvgPool2d
    flatten_layer: nn.Flatten
    activation_layer: nn.Module
    final_layer: nn.Linear

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, sfreq: Optional[float] = None,
                 emb_size: int = 256, res_channels: List[int] = [32, 64, 128], steps: int = 20, activation = nn.ELU,
                 drop_prob: float = 0.5, stride_res: int = 2, kernel_size_res: int = 3, padding_res: int = 1,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if self.input_window_seconds < 1.0:
            self.n_fft = self.n_times
        else:
            self.n_fft = int(self.sfreq)
        self.steps = steps
        rc = [self.n_chans] + res_channels + [emb_size]
        self.torch_stft = _STFTModule(n_fft=self.n_fft, hop_length=self.n_fft // steps)
        self.add_module("torch_stft", self.torch_stft)
        convs = List[nn.Module]()
        for i in range(len(rc) - 1):
            convs.append(_ResBlock(rc[i], rc[i + 1], stride_res, True, True, kernel_size_res, padding_res, drop_prob, activation))
        self.convs = nn.ModuleList(convs)
        self.add_module("convs", self.convs)
        self.adaptative_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.add_module("adaptative_pool", self.adaptative_pool)
        self.flatten_layer = nn.Flatten()
        self.add_module("flatten_layer", self.flatten_layer)
        self.activation_layer = activation()
        self.add_module("activation_layer", self.activation_layer)
        self.final_layer = nn.Linear(emb_size, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, X: Tensor) -> Tensor:
        off = _bn_off_if_batch_one(self, X.shape[0])
        X = self.torch_stft(X)
        for c in self.convs.children():
            X = c(X)
        emb = self.activation_layer(self.flatten_layer(self.adaptative_pool(X)))
        out = self.final_layer(emb)
        _bn_restore(off)
        return out

    def _get_name(self) -> str:
        return "ContraWR"
