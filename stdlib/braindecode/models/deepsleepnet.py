# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _bn_off_if_batch_one, _bn_restore

def _conv_out(w: int, k: int, s: int, p: int) -> int:
    return (w + 2 * p - k) // s + 1

class _CNNPath(nn.Module):
    conv1: nn.Sequential
    pool1: nn.MaxPool2d
    dropout: nn.Dropout
    conv2: nn.Sequential
    conv3: nn.Sequential
    conv4: nn.Sequential
    pool2: nn.MaxPool2d

    def __init__(self, n_filters_1: int = 64, n_filters_2: int = 128, first_kernel_size: int = 50, first_stride: int = 6,
                 first_padding: int = 22, pool1_kernel_size: int = 8, pool1_stride: int = 8, pool1_padding: int = 2,
                 deep_kernel_size: int = 8, pool2_kernel_size: int = 4, pool2_stride: int = 4, pool2_padding: int = 1,
                 activation = nn.ReLU, drop_prob: float = 0.5):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, n_filters_1, kernel_size=(1, first_kernel_size), stride=(1, first_stride),
                                             padding=(0, first_padding), bias=False), nn.BatchNorm2d(num_features=n_filters_1), activation())
        self.add_module("conv1", self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, pool1_kernel_size), stride=(1, pool1_stride), padding=(0, pool1_padding))
        self.add_module("pool1", self.pool1)
        self.dropout = nn.Dropout(p=drop_prob)
        self.add_module("dropout", self.dropout)
        self.conv2 = nn.Sequential(nn.Conv2d(n_filters_1, n_filters_2, kernel_size=(1, deep_kernel_size), stride=1, padding="same", bias=False),
                                   nn.BatchNorm2d(num_features=n_filters_2), activation())
        self.add_module("conv2", self.conv2)
        self.conv3 = nn.Sequential(nn.Conv2d(n_filters_2, n_filters_2, kernel_size=(1, deep_kernel_size), stride=1, padding="same", bias=False),
                                   nn.BatchNorm2d(num_features=n_filters_2), activation())
        self.add_module("conv3", self.conv3)
        self.conv4 = nn.Sequential(nn.Conv2d(n_filters_2, n_filters_2, kernel_size=(1, deep_kernel_size), stride=1, padding="same", bias=False),
                                   nn.BatchNorm2d(num_features=n_filters_2), activation())
        self.add_module("conv4", self.conv4)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, pool2_kernel_size), stride=(1, pool2_stride), padding=(0, pool2_padding))
        self.add_module("pool2", self.pool2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(self.pool1(self.conv1(x)))
        return self.pool2(self.conv4(self.conv3(self.conv2(x))))

    def _get_name(self) -> str:
        return "_CNNPath"

class DeepSleepNet(EEGModuleMixin, nn.Module):
    """DeepSleepNet from Supratak et al (2017)."""
    return_feats: bool
    cnn1: _CNNPath
    cnn2: _CNNPath
    dropout: nn.Dropout
    bilstm: nn.LSTM
    residual_shortcut: nn.Sequential
    features_extractor: nn.Identity
    len_last_layer: int
    final_layer: nn.Module

    def __init__(self, n_outputs: Optional[int] = 5, return_feats: bool = False, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 activation_large = nn.ELU, activation_small = nn.ReLU, drop_prob: float = 0.5,
                 bilstm_hidden_size: int = 512, bilstm_num_layers: int = 2, small_n_filters_1: int = 64,
                 small_n_filters_2: int = 128, small_first_kernel_size: int = 50, small_first_stride: int = 6,
                 small_first_padding: int = 22, small_pool1_kernel_size: int = 8, small_pool1_stride: int = 8,
                 small_pool1_padding: int = 2, small_deep_kernel_size: int = 8, small_pool2_kernel_size: int = 4,
                 small_pool2_stride: int = 4, small_pool2_padding: int = 1, large_n_filters_1: int = 64,
                 large_n_filters_2: int = 128, large_first_kernel_size: int = 400, large_first_stride: int = 50,
                 large_first_padding: int = 175, large_pool1_kernel_size: int = 4, large_pool1_stride: int = 4,
                 large_pool1_padding: int = 0, large_deep_kernel_size: int = 6, large_pool2_kernel_size: int = 2,
                 large_pool2_stride: int = 2, large_pool2_padding: int = 1):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.return_feats = return_feats
        self.cnn1 = _CNNPath(small_n_filters_1, small_n_filters_2, small_first_kernel_size, small_first_stride, small_first_padding,
                             small_pool1_kernel_size, small_pool1_stride, small_pool1_padding, small_deep_kernel_size,
                             small_pool2_kernel_size, small_pool2_stride, small_pool2_padding, activation_small, drop_prob)
        self.add_module("cnn1", self.cnn1)
        self.cnn2 = _CNNPath(large_n_filters_1, large_n_filters_2, large_first_kernel_size, large_first_stride, large_first_padding,
                             large_pool1_kernel_size, large_pool1_stride, large_pool1_padding, large_deep_kernel_size,
                             large_pool2_kernel_size, large_pool2_stride, large_pool2_padding, activation_large, drop_prob)
        self.add_module("cnn2", self.cnn2)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        nt = self.n_times
        sw = _conv_out(_conv_out(_conv_out(nt, small_first_kernel_size, small_first_stride, small_first_padding),
                                 small_pool1_kernel_size, small_pool1_stride, small_pool1_padding),
                       small_pool2_kernel_size, small_pool2_stride, small_pool2_padding)
        lw = _conv_out(_conv_out(_conv_out(nt, large_first_kernel_size, large_first_stride, large_first_padding),
                                 large_pool1_kernel_size, large_pool1_stride, large_pool1_padding),
                       large_pool2_kernel_size, large_pool2_stride, large_pool2_padding)
        if sw <= 0 or lw <= 0:
            raise ValueError(f"n_times={nt} is too small for the configured conv/pool parameters (small path width={sw}, large path width={lw}). Increase n_times or adjust kernel sizes.")
        fs = self.n_chans * (small_n_filters_2 * sw + large_n_filters_2 * lw)
        fo = bilstm_hidden_size * 2
        self.bilstm = nn.LSTM(fs, bilstm_hidden_size, num_layers=bilstm_num_layers, batch_first=True,
                              dropout=drop_prob if bilstm_num_layers > 1 else 0.0, bidirectional=True)
        self.add_module("bilstm", self.bilstm)
        self.residual_shortcut = nn.Sequential(nn.Linear(fs, fo, bias=False), nn.BatchNorm1d(num_features=fo), nn.ReLU())
        self.add_module("residual_shortcut", self.residual_shortcut)
        self.features_extractor = nn.Identity()
        self.add_module("features_extractor", self.features_extractor)
        self.len_last_layer = fo
        fl: nn.Module = nn.Identity()
        if not return_feats:
            fl = nn.Linear(fo, self.n_outputs)
        self.final_layer = fl
        self.add_module("final_layer", fl)

    def forward(self, x: Tensor) -> Tensor:
        off = _bn_off_if_batch_one(self, x.shape[0])
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x1 = self.cnn1(x).flatten(1)
        x2 = self.cnn2(x).flatten(1)
        x = self.dropout(torch.cat([x1, x2], dim=1))
        res = self.residual_shortcut(x)
        out, _ = self.bilstm(x.unsqueeze(1), None)
        x = self.dropout(out.squeeze(1) + res)
        y = self.final_layer(self.features_extractor(x))
        _bn_restore(off)
        return y

    def _get_name(self) -> str:
        return "DeepSleepNet"
