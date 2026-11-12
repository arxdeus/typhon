# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin

class EEGSimpleConv(EEGModuleMixin, nn.Module):
    """EEG-SimpleConv from Ouahidi et al (2023). Resampling (torchaudio
    Resample) is not supported: sfreq must equal resampling_freq."""
    return_feature: bool
    resample: nn.Module
    conv: nn.Conv1d
    bn: nn.BatchNorm1d
    blocks: nn.ModuleList
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None, sfreq: Optional[float] = None,
                 feature_maps: int = 128, n_convs: int = 2, resampling_freq: float = 80, kernel_size: int = 8,
                 return_feature: bool = False, activation = nn.ReLU, chs_info: Optional[List[Dict[str, str]]] = None,
                 n_times: Optional[int] = None, input_window_seconds: Optional[float] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.return_feature = return_feature
        if self.sfreq != float(resampling_freq):
            raise NotImplementedError("EEGSimpleConv: resampling (sfreq != resampling_freq) is not supported in Codon")
        self.resample = nn.Identity()
        self.add_module("resample", self.resample)
        self.conv = nn.Conv1d(self.n_chans, feature_maps, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.add_module("conv", self.conv)
        self.bn = nn.BatchNorm1d(feature_maps)
        self.add_module("bn", self.bn)
        blocks = List[nn.Module]()
        new_fm = feature_maps
        old_fm = feature_maps
        for i in range(n_convs):
            if i > 0:
                new_fm = int(1.414 * new_fm)
            blocks.append(nn.Sequential(
                nn.Conv1d(old_fm, new_fm, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(new_fm), nn.MaxPool1d(2), activation(),
                nn.Conv1d(new_fm, new_fm, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(new_fm), activation()))
            old_fm = new_fm
        self.blocks = nn.ModuleList(blocks)
        self.add_module("blocks", self.blocks)
        self.final_layer = nn.Linear(old_fm, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x_rs = self.resample(x.contiguous())
        feat = torch.relu(self.bn(self.conv(x_rs)))
        for seq in self.blocks.children():
            feat = seq(feat)
        feat = feat.mean(dim=2)
        if self.return_feature:
            return feat
        return self.final_layer(feat)

    def _get_name(self) -> str:
        return "EEGSimpleConv"
