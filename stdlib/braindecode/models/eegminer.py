# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.modules.gaussian_filter import GeneralizedGaussianFilter, plv_time

class _EEGMinerFeatures(nn.Module):
    method: str
    n_chans: int
    n_filters: int
    n_times: int
    triu_row: Tensor
    triu_col: Tensor
    _flat_idx: Tensor

    def __init__(self, method: str, n_chans: int, n_filters: int, n_times: int):
        super().__init__()
        self.method = method
        self.n_chans = n_chans
        self.n_filters = n_filters
        self.n_times = n_times
        rows = List[int]()
        cols = List[int]()
        flat = List[int]()
        for i in range(n_chans):
            for j in range(i + 1, n_chans):
                rows.append(i)
                cols.append(j)
                flat.append(i * n_chans + j)
        self.triu_row = torch.tensor(rows, dtype=torch.int64)
        self.triu_col = torch.tensor(cols, dtype=torch.int64)
        self._flat_idx = torch.tensor(flat, dtype=torch.int64)
        self.register_buffer("triu_row", self.triu_row, persistent=False)
        self.register_buffer("triu_col", self.triu_col, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.method == "mag":
            return torch.sqrt((x * x).mean(dim=-1))
        if self.method == "corr":
            x = x.reshape(x.shape[0], self.n_chans, self.n_filters, self.n_times).transpose(-3, -2)
            x = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-6)
            x = torch.matmul(x, x.transpose(-2, -1)) / x.shape[-1]
            x = x.permute(0, 2, 3, 1).abs()
        else:
            x = x.transpose(-4, -3)
            x = plv_time(x, forward_fourier=False)
            x = x.permute(0, 2, 3, 1)
        # x[:, triu_row, triu_col, :]
        b, f = x.shape[0], x.shape[3]
        x = x.reshape(b, self.n_chans * self.n_chans, f)
        return x.index_select(1, self._flat_idx)

    def _get_name(self) -> str:
        return "_EEGMinerFeatures"

class EEGMiner(EEGModuleMixin, nn.Module):
    """EEGMiner from Ludwig et al (2024)."""
    filter_f_mean: List[float]
    n_filters: int
    method: str
    n_features: int
    filter: GeneralizedGaussianFilter
    ensure_dim: nn.Module
    feature_layer: _EEGMinerFeatures
    flatten_layer: nn.Flatten
    batch_layer: nn.BatchNorm1d
    final_layer: nn.Linear

    def __init__(self, method: str = "plv", n_chans: Optional[int] = None, n_outputs: Optional[int] = None,
                 n_times: Optional[int] = None, chs_info: Optional[List[Dict[str, str]]] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 filter_f_mean = (23.0, 23.0), filter_bandwidth = (44.0, 44.0), filter_shape = (2.0, 2.0),
                 group_delay = (20.0, 20.0), clamp_f_mean = (1.0, 45.0)):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.filter_f_mean = [float(v) for v in filter_f_mean]
        self.n_filters = len(self.filter_f_mean)
        self.method = method.lower()
        if self.method not in ("mag", "corr", "plv"):
            raise ValueError(f"The method {self.method} is not one of the valid options ['mag', 'corr', 'plv']")
        inverse_fourier = True
        in_channels = self.n_chans
        out_channels = self.n_chans * self.n_filters
        if self.method == "plv":
            inverse_fourier = False
            in_channels = 1
            out_channels = self.n_filters
        self.filter = GeneralizedGaussianFilter(in_channels=in_channels, out_channels=out_channels,
                                                sequence_length=self.n_times, sample_rate=self.sfreq,
                                                f_mean=filter_f_mean, bandwidth=filter_bandwidth, shape=filter_shape,
                                                affine_group_delay=False, inverse_fourier=inverse_fourier,
                                                group_delay=group_delay, clamp_f_mean=clamp_f_mean)
        self.add_module("filter", self.filter)
        self.ensure_dim = nn.Identity()
        if self.method == "mag":
            self.n_features = self.n_chans * self.n_filters
        elif self.method == "corr":
            self.n_features = self.n_filters * self.n_chans * (self.n_chans - 1) // 2
        else:
            self.ensure_dim = nn.Unflatten(-1, (1, self.n_times))
            self.n_features = (self.n_filters * self.n_chans * (self.n_chans - 1)) // 2
        self.add_module("ensure_dim", self.ensure_dim)
        self.feature_layer = _EEGMinerFeatures(self.method, self.n_chans, self.n_filters, self.n_times)
        self.add_module("feature_layer", self.feature_layer)
        self.flatten_layer = nn.Flatten()
        self.add_module("flatten_layer", self.flatten_layer)
        self.batch_layer = nn.BatchNorm1d(self.n_features, affine=False)
        self.add_module("batch_layer", self.batch_layer)
        self.final_layer = nn.Linear(self.n_features, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        with torch.no_grad():
            self.final_layer.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = self.ensure_dim(x)
        x = self.filter(x)
        x = self.feature_layer(x)
        x = x.reshape(x.shape[0], self.n_features)
        if self.training and x.shape[0] == 1:
            self.batch_layer.eval()
            x = self.batch_layer(x)
            self.batch_layer.train()
        else:
            x = self.batch_layer(x)
        return self.final_layer(x)

    def _get_name(self) -> str:
        return "EEGMiner"
