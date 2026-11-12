# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn._conv3d import Conv3d
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _bn_off_if_batch_one, _bn_restore

def _first_number(s: str) -> Tuple[int, int, int]:
    # (start, end, value) of the first run of digits, or (-1, -1, 0)
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    if i == len(s):
        return (-1, -1, 0)
    j = i
    while j < len(s) and s[j].isdigit():
        j += 1
    return (i, j, int(s[i:j]))

def division_channels_idx(ch_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    left = List[str]()
    right = List[str]()
    middle = List[str]()
    for ch in ch_names:
        a, b, v = _first_number(ch)
        if a < 0:
            middle.append(ch)
        elif v % 2:
            left.append(ch)
        else:
            right.append(ch)
    return (left, right, middle)

def match_hemisphere_chans(left_chs: List[str], right_chs: List[str]) -> List[Tuple[str, str]]:
    if len(left_chs) != len(right_chs):
        raise ValueError("Left and right channels do not match.")
    rights = list(right_chs)
    out = List[Tuple[str, str]]()
    for left in left_chs:
        a, b, v = _first_number(left)
        if a < 0:
            raise ValueError(f"Channel '{left}' does not contain a number.")
        # re.sub replaces every digit run; braindecode channel names have one
        target = left[:a] + str(v + 1) + left[b:]
        found = False
        for k in range(len(rights)):
            if rights[k] == target:
                out.append((left, rights[k]))
                rights.pop(k)
                found = True
                break
        if not found:
            raise ValueError(f"Found no right hemisphere matching channel for '{left}'.")
    return out

def _avg_pool_time(x: Tensor, k: int) -> Tensor:
    b, c, d, t, s = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
    y = F.avg_pool2d(x.reshape(b, c * d, t, s), (k, 1))
    return y.reshape(b, c, d, y.shape[2], s)

class _AvgPoolTime(nn.Module):
    k: int
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    def forward(self, x: Tensor) -> Tensor:
        return _avg_pool_time(x, self.k)
    def _get_name(self) -> str:
        return "_AvgPoolTime"

class _SpatialConv3d(Conv3d):
    _sc: int
    def __init__(self, in_channels: int, out_channels: int, kernel_size, padding = 0, groups: int = 1, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=bias)
        self._sc = 0
    def forward(self, x: Tensor) -> Tensor:
        b, c, z, t, s = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        y = F.conv2d(x.reshape(b, c, z * t, s), self.weight.squeeze(2), self.bias, 1, 0, 1, self.groups)
        return y.reshape(b, -1, z, t, y.shape[-1])
    def _get_name(self) -> str:
        return "_SpatialConv3d"

def _bn3(n: int) -> nn.Module:
    return nn.BatchNorm3d(n, eps=0.001, momentum=0.01)

def _tconv(cin: int, cout: int, k: int, bias: bool = True, groups: int = 1, same: bool = True) -> nn.Module:
    if same:
        return Conv3d(cin, cout, kernel_size=(1, k, 1), padding="same", bias=bias, groups=groups)
    return Conv3d(cin, cout, kernel_size=(1, k, 1), padding=(0, 0, 0), bias=bias, groups=groups)

class _InceptionBlock(nn.Module):
    temporal_convs: nn.ModuleList
    spatial_convs: Optional[nn.ModuleList]
    pool: nn.Module

    def __init__(self, in_channels: int, scales_samples: List[int], filters_per_branch: int, ncha: int,
                 activation: nn.Module, drop_prob: float, average_pool: int, spatial_resnet_repetitions: int):
        super().__init__()
        self.temporal_convs = nn.ModuleList()
        self.add_module("temporal_convs", self.temporal_convs)
        for s in scales_samples:
            self.temporal_convs.append(nn.Sequential(_tconv(in_channels, filters_per_branch, s), _bn3(filters_per_branch),
                                                     activation, nn.Dropout(drop_prob)))
        nf = filters_per_branch * len(scales_samples)
        self.spatial_convs = None
        if ncha != 1:
            sc = nn.ModuleList()
            for _ in range(spatial_resnet_repetitions):
                sc.append(nn.Sequential(_SpatialConv3d(nf, nf, (1, 1, ncha), padding=(0, 0, 0), groups=nf, bias=False),
                                        _bn3(nf), activation, nn.Dropout(drop_prob)))
            self.spatial_convs = sc
            self.add_module("spatial_convs", sc)
        p: nn.Module = nn.Identity()
        if average_pool != 1:
            p = _AvgPoolTime(average_pool)
        self.pool = p
        self.add_module("pool", p)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([c(x) for c in self.temporal_convs.children()], dim=1)
        out = self.pool(out + x)
        if self.spatial_convs is not None:
            for sc in self.spatial_convs.__val__().children():
                out = out + sc(out)
        return out

    def _get_name(self) -> str:
        return "_InceptionBlock"

class _ResidualBlock(nn.Module):
    temporal_conv: nn.Sequential
    projection: nn.Sequential
    avg_pool: _AvgPoolTime
    spatial_convs: Optional[nn.ModuleList]

    def __init__(self, in_channels: int, filters: int, kernel_size: int, ncha: int, activation: nn.Module,
                 drop_prob: float, average_pool: int, spatial_resnet_repetitions: int = 1):
        super().__init__()
        self.temporal_conv = nn.Sequential(_tconv(in_channels, filters, kernel_size), _bn3(filters), activation, nn.Dropout(drop_prob))
        self.add_module("temporal_conv", self.temporal_conv)
        self.projection = nn.Sequential(Conv3d(in_channels, filters, kernel_size=(1, 1, 1), bias=False), _bn3(filters),
                                        activation, nn.Dropout(drop_prob))
        self.add_module("projection", self.projection)
        self.avg_pool = _AvgPoolTime(average_pool)
        self.add_module("avg_pool", self.avg_pool)
        self.spatial_convs = None
        if ncha != 1:
            sc = nn.ModuleList()
            for _ in range(spatial_resnet_repetitions):
                sc.append(nn.Sequential(Conv3d(filters, filters, kernel_size=(1, 1, ncha), padding=(0, 0, 0), groups=1, bias=False),
                                        _bn3(filters), activation, nn.Dropout(drop_prob)))
            self.spatial_convs = sc
            self.add_module("spatial_convs", sc)

    def forward(self, x: Tensor) -> Tensor:
        out = self.avg_pool(self.temporal_conv(x) + self.projection(x))
        if self.spatial_convs is not None:
            for sc in self.spatial_convs.__val__().children():
                out = out + sc(out)
        return out

    def _get_name(self) -> str:
        return "_ResidualBlock"

class _TemporalBlock(nn.Module):
    conv: nn.Sequential
    def __init__(self, in_channels: int, filters: int, kernel_size: int, activation: nn.Module, drop_prob: float):
        super().__init__()
        self.conv = nn.Sequential(_tconv(in_channels, filters, kernel_size, bias=False), _bn3(filters), activation, nn.Dropout(drop_prob))
        self.add_module("conv", self.conv)
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x) + x
    def _get_name(self) -> str:
        return "_TemporalBlock"

class _ChannelMergingBlock(nn.Module):
    residual_convs: nn.ModuleList
    grouped_conv: nn.Sequential
    def __init__(self, in_channels: int, filters: int, groups: int, ncha: int, division: int, activation: nn.Module,
                 drop_prob: float):
        super().__init__()
        self.residual_convs = nn.ModuleList()
        self.add_module("residual_convs", self.residual_convs)
        for _ in range(2):
            self.residual_convs.append(nn.Sequential(Conv3d(in_channels, filters, kernel_size=(division, 1, ncha), padding=(0, 0, 0), bias=False),
                                                     _bn3(filters), activation, nn.Dropout(drop_prob)))
        self.grouped_conv = nn.Sequential(Conv3d(in_channels, filters, kernel_size=(division, 1, ncha), groups=groups, padding=(0, 0, 0), bias=False),
                                          _bn3(filters), activation, nn.Dropout(drop_prob))
        self.add_module("grouped_conv", self.grouped_conv)
    def forward(self, x: Tensor) -> Tensor:
        for r in self.residual_convs.children():
            x = x + r(x)
        return self.grouped_conv(x)
    def _get_name(self) -> str:
        return "_ChannelMergingBlock"

class _TemporalMergingBlock(nn.Module):
    temporal_kernel: int
    residual_conv: nn.Sequential
    grouped_conv: nn.Sequential
    def __init__(self, in_channels: int, filters: int, groups: int, n_times: int, activation: nn.Module, drop_prob: float):
        super().__init__()
        self.temporal_kernel = n_times
        self.residual_conv = nn.Sequential(_tconv(in_channels, in_channels, n_times, bias=False, same=False), _bn3(in_channels),
                                           activation, nn.Dropout(drop_prob))
        self.add_module("residual_conv", self.residual_conv)
        self.grouped_conv = nn.Sequential(_tconv(in_channels, filters, n_times, bias=False, groups=groups, same=False), _bn3(filters),
                                          activation, nn.Dropout(drop_prob))
        self.add_module("grouped_conv", self.grouped_conv)
    def forward(self, x: Tensor) -> Tensor:
        return self.grouped_conv(x + self.residual_conv(x))
    def _get_name(self) -> str:
        return "_TemporalMergingBlock"

class _OutputBlock(nn.Module):
    conv_blocks: nn.ModuleList
    def __init__(self, in_channels: int, activation: nn.Module, drop_prob: float, n_residual: int = 4):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.add_module("conv_blocks", self.conv_blocks)
        for _ in range(n_residual):
            self.conv_blocks.append(nn.Sequential(Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False),
                                                  _bn3(in_channels), activation, nn.Dropout(drop_prob)))
    def forward(self, x: Tensor) -> Tensor:
        for c in self.conv_blocks.children():
            x = x + c(x)
        return x
    def _get_name(self) -> str:
        return "_OutputBlock"

class EEGSym(EEGModuleMixin, nn.Module):
    """EEGSym from Perez-Velasco et al (2022). chs_info entries need a
    "ch_name" key."""
    filters_per_branch: int
    drop_prob: float
    activation: nn.Module
    scales_samples: List[int]
    left_idx: Tensor
    right_idx: Tensor
    middle_idx: Tensor
    n_channels_per_hemi: int
    inception_block1: _InceptionBlock
    inception_block2: _InceptionBlock
    residual_blocks: nn.Sequential
    temporal_reduction: nn.Sequential
    channel_merging: _ChannelMergingBlock
    temporal_merging: _TemporalMergingBlock
    output_blocks: nn.Sequential
    final_layer: nn.Linear

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, filters_per_branch: int = 8, scales_time = (500, 250, 125),
                 drop_prob: float = 0.25, activation = nn.ELU, spatial_resnet_repetitions: int = 1,
                 left_right_chs: Optional[List[Tuple[str, str]]] = None, middle_chs: Optional[List[str]] = None):
        if (left_right_chs is None) != (middle_chs is None):
            raise ValueError("Either both or none of left_right_chs and middle_chs must be provided.")
        if filters_per_branch <= 0 or filters_per_branch % 8 != 0:
            raise ValueError("filters_per_branch must be a positive multiple of 8.")
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.filters_per_branch = filters_per_branch
        self.drop_prob = drop_prob
        act = activation()
        self.activation = act
        self.add_module("activation", act)
        self.scales_samples = sorted([int(float(s) * self.sfreq / 1000) for s in scales_time])
        tif = filters_per_branch * len(self.scales_samples)
        rf = tif // 2
        redf = tif // 4
        names = [c["ch_name"] for c in self.chs_info]
        lch = List[str]()
        rch = List[str]()
        mch = List[str]()
        if left_right_chs is None:
            l0, r0, m0 = division_channels_idx(names)
            ok = True
            pairs = List[Tuple[str, str]]()
            try:
                pairs = match_hemisphere_chans(l0, r0)
            except ValueError:
                ok = False
            if ok:
                lch = [p[0] for p in pairs]
                rch = [p[1] for p in pairs]
                mch = m0
            else:
                lch = names
                rch = names
        else:
            for a, b in left_right_chs.__val__():
                lch.append(a)
                rch.append(b)
            mch = middle_chs.__val__()
        self.left_idx = torch.tensor([names.index(c) for c in lch], dtype=torch.long)
        self.register_buffer("left_idx", self.left_idx)
        self.right_idx = torch.tensor([names.index(c) for c in rch], dtype=torch.long)
        self.register_buffer("right_idx", self.right_idx)
        self.middle_idx = torch.tensor([names.index(c) for c in mch], dtype=torch.long)
        self.register_buffer("middle_idx", self.middle_idx)
        nh = len(lch) + len(mch)
        self.n_channels_per_hemi = nh
        self.add_module("include_extra_dim", Rearrange("batch channel time -> batch 1 channel time"))
        self.add_module("permute_layer", Rearrange("batch features z time space -> batch features z space time"))
        self.inception_block1 = _InceptionBlock(1, self.scales_samples, filters_per_branch, nh, act, drop_prob, 2, spatial_resnet_repetitions)
        self.add_module("inception_block1", self.inception_block1)
        self.inception_block2 = _InceptionBlock(tif, [max(1, s // 4) for s in self.scales_samples], filters_per_branch, nh, act,
                                                drop_prob, 2, spatial_resnet_repetitions)
        self.add_module("inception_block2", self.inception_block2)
        self.residual_blocks = nn.Sequential(
            _ResidualBlock(tif, rf, 16, nh, act, drop_prob, 2, spatial_resnet_repetitions),
            _ResidualBlock(rf, rf, 8, nh, act, drop_prob, 2, spatial_resnet_repetitions),
            _ResidualBlock(rf, redf, 4, nh, act, drop_prob, 2, spatial_resnet_repetitions))
        self.add_module("residual_blocks", self.residual_blocks)
        self.temporal_reduction = nn.Sequential(_TemporalBlock(redf, redf, 4, act, drop_prob), _AvgPoolTime(2))
        self.add_module("temporal_reduction", self.temporal_reduction)
        self.channel_merging = _ChannelMergingBlock(redf, redf, tif // 8, nh, 2, act, drop_prob)
        self.add_module("channel_merging", self.channel_merging)
        self.temporal_merging = _TemporalMergingBlock(redf, rf, redf, self.n_times // 64, act, drop_prob)
        self.add_module("temporal_merging", self.temporal_merging)
        self.output_blocks = nn.Sequential(_OutputBlock(rf, act, drop_prob), nn.Flatten())
        self.add_module("output_blocks", self.output_blocks)
        self.final_layer = nn.Linear(rf, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        off = _bn_off_if_batch_one(self, x.shape[0])
        x = x.unsqueeze(1)
        l = torch.index_select(x, 2, self.left_idx)
        r = torch.index_select(x, 2, self.right_idx)
        m = torch.index_select(x, 2, self.middle_idx)
        x = torch.stack([torch.cat([l, m], dim=2), torch.cat([r, m], dim=2)], dim=2)
        x = x.permute(0, 1, 2, 4, 3)
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.residual_blocks(x)
        x = self.temporal_reduction(x)
        x = self.channel_merging(x)
        x = self.temporal_merging(x)
        x = self.output_blocks(x)
        x = self.final_layer(x)
        _bn_restore(off)
        return x

    def _get_name(self) -> str:
        return "EEGSym"
