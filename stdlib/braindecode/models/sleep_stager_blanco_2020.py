# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.models.sleep_stager_chambon_2018 import _bn

class SleepStagerBlanco2020(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Blanco et al (2020)."""
    feature_extractor: nn.Sequential
    len_last_layer: int
    return_feats: bool
    final_layer: Optional[nn.Sequential]

    def __init__(self, n_chans: Optional[int] = None, sfreq: Optional[float] = None, n_conv_chans: int = 20,
                 input_window_seconds: Optional[float] = None, n_outputs: Optional[int] = 5, n_groups: int = 2,
                 max_pool_size: int = 2, drop_prob: float = 0.5, apply_batch_norm: bool = False,
                 return_feats: bool = False, activation = nn.ReLU, chs_info: Optional[List[Dict[str, str]]] = None,
                 n_times: Optional[int] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.mapping["fc.1.weight"] = "final_layer.1.weight"
        self.mapping["fc.1.bias"] = "final_layer.1.bias"
        ng = math.gcd(self.n_chans, n_groups)
        c = n_conv_chans
        fe = nn.Sequential()
        specs = [(self.n_chans, 7, ng), (c, 7, c), (c, 5, c), (c, 5, c), (c, 5, c), (c, 3, c), (c, 3, c)]
        idx = 0
        for cin, k, g in specs:
            fe.add_module(str(idx), nn.Conv2d(cin, c, (1, k), groups=g, padding=0))
            fe.add_module(str(idx + 1), _bn(apply_batch_norm, c))
            fe.add_module(str(idx + 2), activation())
            fe.add_module(str(idx + 3), nn.MaxPool2d((1, max_pool_size)))
            idx += 4
        self.feature_extractor = fe
        self.add_module("feature_extractor", fe)
        fe.eval()
        n = 0
        with torch.no_grad():
            n = fe(torch.zeros(1, self.n_chans, 1, self.n_times)).numel()
        fe.train()
        self.len_last_layer = n
        self.return_feats = return_feats
        self.final_layer = None
        if not return_feats:
            fl = nn.Sequential(nn.Dropout(drop_prob), nn.Linear(self.len_last_layer, self.n_outputs), nn.Identity())
            self.final_layer = fl
            self.add_module("final_layer", fl)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(2)
        feats = self.feature_extractor(x).flatten(start_dim=1)
        if self.return_feats:
            return feats
        fl = self.final_layer
        return fl(feats)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "SleepStagerBlanco2020"
