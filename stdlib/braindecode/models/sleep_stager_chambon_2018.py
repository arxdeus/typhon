# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin

class SleepStagerChambon2018(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Chambon et al (2018)."""
    spatial_conv: nn.Module
    feature_extractor: nn.Sequential
    return_feats: bool
    len_last_layer: int
    final_layer: Optional[nn.Sequential]

    def __init__(self, n_chans: Optional[int] = None, sfreq: Optional[float] = None, n_conv_chs: int = 8,
                 time_conv_size_s: float = 0.5, max_pool_size_s: float = 0.125, pad_size_s: float = 0.25,
                 activation = nn.ReLU, input_window_seconds: Optional[float] = None, n_outputs: Optional[int] = 5,
                 drop_prob: float = 0.25, apply_batch_norm: bool = False, return_feats: bool = False,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.mapping["fc.1.weight"] = "final_layer.1.weight"
        self.mapping["fc.1.bias"] = "final_layer.1.bias"
        tcs = int(math.ceil(time_conv_size_s * self.sfreq))
        mps = int(math.ceil(max_pool_size_s * self.sfreq))
        pad = int(math.ceil(pad_size_s * self.sfreq))
        sc: nn.Module = nn.Identity()
        if self.n_chans > 1:
            sc = nn.Conv2d(1, self.n_chans, (self.n_chans, 1))
        self.spatial_conv = sc
        self.add_module("spatial_conv", sc)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, n_conv_chs, (1, tcs), padding=(0, pad)), _bn(apply_batch_norm, n_conv_chs), activation(),
            nn.MaxPool2d((1, mps)),
            nn.Conv2d(n_conv_chs, n_conv_chs, (1, tcs), padding=(0, pad)), _bn(apply_batch_norm, n_conv_chs), activation(),
            nn.MaxPool2d((1, mps)))
        self.add_module("feature_extractor", self.feature_extractor)
        self.return_feats = return_feats
        d1 = (self.n_times + 2 * pad - (tcs - 1)) // mps
        d2 = (d1 + 2 * pad - (tcs - 1)) // mps
        self.len_last_layer = n_conv_chs * self.n_chans * d2
        self.final_layer = None
        if not return_feats:
            fl = nn.Sequential(nn.Dropout(p=drop_prob), nn.Linear(in_features=self.len_last_layer, out_features=self.n_outputs))
            self.final_layer = fl
            self.add_module("final_layer", fl)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if self.n_chans > 1:
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)
        feats = self.feature_extractor(x).flatten(start_dim=1)
        if self.return_feats:
            return feats
        fl = self.final_layer
        return fl(feats)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "SleepStagerChambon2018"

def _bn(apply: bool, n: int) -> nn.Module:
    if apply:
        return nn.BatchNorm2d(n)
    return nn.Identity(n)
