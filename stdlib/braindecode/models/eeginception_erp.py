# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.functional import glorot_weight_zero_bias
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import DepthwiseConv2d, Ensure4d, InceptionBlock

class EEGInceptionERP(EEGModuleMixin, nn.Sequential):
    """EEG-Inception for ERP from Santamaria-Vazquez et al (2020)."""
    drop_prob: float
    n_filters: int
    scales_samples: List[int]
    alpha_momentum: float
    depth_multiplier: int
    pooling_sizes: List[int]

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 sfreq: Optional[float] = None, drop_prob: float = 0.5, scales_samples_s = (0.5, 0.25, 0.125),
                 n_filters: int = 8, activation = nn.ELU, batch_norm_alpha: float = 0.01, depth_multiplier: int = 2,
                 pooling_sizes = (4, 2, 2, 2), chs_info: Optional[List[Dict[str, str]]] = None,
                 input_window_seconds: Optional[float] = None):
        sf = sfreq
        nt = n_times
        if sf is None and (nt is None or input_window_seconds is None):
            sf = 128.0
        if nt is None and input_window_seconds is None:
            nt = 1000
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=nt,
                                input_window_seconds=input_window_seconds, sfreq=sf)
        nn.Sequential.__init__(self)
        self.drop_prob = drop_prob
        self.n_filters = n_filters
        self.scales_samples = [int(float(s) * self.sfreq) for s in scales_samples_s]
        self.alpha_momentum = batch_norm_alpha
        self.depth_multiplier = depth_multiplier
        self.pooling_sizes = [int(p) for p in pooling_sizes]
        self.mapping["classification.1.weight"] = "final_layer.fc.weight"
        self.mapping["classification.1.bias"] = "final_layer.fc.bias"
        self.add_module("ensuredims", Ensure4d())
        self.add_module("dimshuffle", Rearrange("batch C T 1 -> batch 1 C T"))
        b1 = List[nn.Module]()
        for s in self.scales_samples:
            b1.append(nn.Sequential(
                nn.Conv2d(1, n_filters, kernel_size=(1, s), padding="same", bias=True),
                nn.BatchNorm2d(n_filters, momentum=batch_norm_alpha), activation(), nn.Dropout(drop_prob),
                DepthwiseConv2d(n_filters, kernel_size=(self.n_chans, 1), depth_multiplier=depth_multiplier, bias=False, padding="valid"),
                nn.BatchNorm2d(depth_multiplier * n_filters, momentum=batch_norm_alpha), activation(), nn.Dropout(drop_prob)))
        self.add_module("inception_block_1", InceptionBlock(b1))
        self.add_module("avg_pool_1", nn.AvgPool2d((1, self.pooling_sizes[0])))
        ncf = len(self.scales_samples) * n_filters
        ncdw = ncf * depth_multiplier
        b2 = List[nn.Module]()
        for s in self.scales_samples:
            b2.append(nn.Sequential(
                nn.Conv2d(ncdw, n_filters, kernel_size=(1, s // 4), padding="same", bias=False),
                nn.BatchNorm2d(n_filters, momentum=batch_norm_alpha), activation(), nn.Dropout(drop_prob)))
        self.add_module("inception_block_2", InceptionBlock(b2))
        self.add_module("avg_pool_2", nn.AvgPool2d((1, self.pooling_sizes[1])))
        self.add_module("final_block", nn.Sequential(
            nn.Conv2d(ncf, ncf // 2, (1, 8), padding="same", bias=False),
            nn.BatchNorm2d(ncf // 2, momentum=batch_norm_alpha), activation(), nn.Dropout(drop_prob),
            nn.AvgPool2d((1, self.pooling_sizes[2])),
            nn.Conv2d(ncf // 2, ncf // 4, (1, 4), padding="same", bias=False),
            nn.BatchNorm2d(ncf // 4, momentum=batch_norm_alpha), activation(), nn.Dropout(drop_prob),
            nn.AvgPool2d((1, self.pooling_sizes[3]))))
        prod = 1
        for p in self.pooling_sizes:
            prod *= p
        spatial = self.n_times // prod
        nch = n_filters * len(self.scales_samples) // 4
        self.add_module("flat", nn.Flatten())
        module = nn.Sequential()
        module.add_module("fc", nn.Linear(spatial * nch, self.n_outputs))
        module.add_module("identity", nn.Identity())
        self.add_module("final_layer", module)
        glorot_weight_zero_bias(self)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "EEGInceptionERP"
