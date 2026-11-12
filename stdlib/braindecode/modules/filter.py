# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class FilterBankLayer(nn.Module):
    """FIR filter bank (mne.filter.create_filter designs). Only method='fir'
    with phase='zero' (the default) is supported in Typhon."""
    band_filters: List[Tuple[float, float]]
    n_bands: int
    phase: str
    method: str
    n_chans: int
    fir_list: nn.ParameterList
    fir_a_list: nn.ParameterList
    b_list: nn.ParameterList
    a_list: nn.ParameterList

    def __init__(self, n_chans: int, sfreq: float, band_filters = None, method: str = "fir",
                 filter_length = "auto", l_trans_bandwidth = "auto", h_trans_bandwidth = "auto",
                 phase: str = "zero", fir_window: str = "hamming", fir_design: str = "firwin", verbose: bool = True):
        super().__init__()
        from mne.filter import create_filter
        bands = List[Tuple[float, float]]()
        if band_filters is None:
            for low in range(4, 37, 4):
                bands.append((float(low), float(low + 4)))
        elif isinstance(band_filters, int):
            bw = (40.0 - 4.0) / band_filters
            for i in range(band_filters):
                bands.append((float(4.0 + i * bw), float(4.0 + (i + 1) * bw)))
        else:
            for lo, hi in band_filters:
                bands.append((float(lo), float(hi)))
        if method != "fir" or phase not in ("zero", "forward", "causal"):
            raise NotImplementedError("FilterBankLayer: only method='fir' is supported in Typhon")
        if phase != "zero":
            raise NotImplementedError("FilterBankLayer: only phase='zero' is supported in Typhon")
        self.band_filters = bands
        self.n_bands = len(bands)
        self.phase = phase
        self.method = method
        self.n_chans = n_chans
        self.fir_list = nn.ParameterList()
        self.add_module("fir_list", self.fir_list)
        self.fir_a_list = nn.ParameterList()
        self.add_module("fir_a_list", self.fir_a_list)
        self.b_list = nn.ParameterList()
        self.add_module("b_list", self.b_list)
        self.a_list = nn.ParameterList()
        self.add_module("a_list", self.a_list)
        for lo, hi in bands:
            h = create_filter(None, sfreq, lo, hi, filter_length=filter_length, l_trans_bandwidth=l_trans_bandwidth,
                              h_trans_bandwidth=h_trans_bandwidth, method=method, phase=phase, fir_window=fir_window,
                              fir_design=fir_design, verbose=verbose)
            vals = [float(h[i]) for i in range(h.size)]
            filt = torch.tensor(vals, dtype=torch.float32)
            self.fir_list.append(nn.Parameter(filt, requires_grad=False))
            fa = torch.zeros(len(vals))
            fa[0] = 1.0
            self.fir_a_list.append(nn.Parameter(fa, requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        # zero-phase FIR bank as a single conv1d (braindecode's direct path).
        start = 0
        length = 0
        firs = [f for f in self.fir_list]
        for f in firs:
            start = max(start, (f.shape[-1] - 1) // 2)
        for f in firs:
            length = max(length, start - (f.shape[-1] - 1) // 2 + f.shape[-1])
        kernels = List[Tensor]()
        for f in firs:
            left = start - (f.shape[-1] - 1) // 2
            k = F.pad(f.to(x.dtype), [left, length - left - f.shape[-1]])
            kernels.append(k.flip(-1))
        b, c, t = x.shape[0], x.shape[1], x.shape[2]
        padded = F.pad(x.reshape(b * c, 1, t), [length - 1 - start, start])
        out = F.conv1d(padded, torch.stack(kernels).unsqueeze(1))
        return out.reshape(b, c, -1, t).transpose(1, 2)

    def _get_name(self) -> str:
        return "FilterBankLayer"
