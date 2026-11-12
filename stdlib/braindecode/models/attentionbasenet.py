# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Ensure4d

class _FeatureExtractor(nn.Module):
    ensure4d: Ensure4d
    rearrange_input: Rearrange
    temporal_conv: nn.Conv2d
    intermediate_bn: nn.BatchNorm2d
    spatial_conv: nn.Conv2d
    bn: nn.BatchNorm2d
    nonlinearity: nn.Module
    pool: nn.AvgPool2d
    dropout: nn.Dropout

    def __init__(self, n_chans: int, n_temporal_filters: int = 40, temporal_filter_length: int = 25,
                 spatial_expansion: int = 1, pool_length: int = 75, pool_stride: int = 15, drop_prob: float = 0.5,
                 activation = nn.ELU):
        super().__init__()
        self.ensure4d = Ensure4d()
        self.add_module("ensure4d", self.ensure4d)
        self.rearrange_input = Rearrange("b c t 1 -> b 1 c t")
        self.add_module("rearrange_input", self.rearrange_input)
        self.temporal_conv = nn.Conv2d(1, n_temporal_filters, kernel_size=(1, temporal_filter_length),
                                       padding=(0, temporal_filter_length // 2), bias=False)
        self.add_module("temporal_conv", self.temporal_conv)
        self.intermediate_bn = nn.BatchNorm2d(n_temporal_filters)
        self.add_module("intermediate_bn", self.intermediate_bn)
        self.spatial_conv = nn.Conv2d(n_temporal_filters, n_temporal_filters * spatial_expansion, kernel_size=(n_chans, 1),
                                      groups=n_temporal_filters, bias=False)
        self.add_module("spatial_conv", self.spatial_conv)
        self.bn = nn.BatchNorm2d(n_temporal_filters * spatial_expansion)
        self.add_module("bn", self.bn)
        self.nonlinearity = activation()
        self.add_module("nonlinearity", self.nonlinearity)
        self.pool = nn.AvgPool2d((1, pool_length), stride=(1, pool_stride))
        self.add_module("pool", self.pool)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.rearrange_input(self.ensure4d(x))
        x = self.intermediate_bn(self.temporal_conv(x))
        x = self.nonlinearity(self.bn(self.spatial_conv(x)))
        return self.dropout(self.pool(x))

    def _get_name(self) -> str:
        return "_FeatureExtractor"

class _ChannelAttentionBlock(nn.Module):
    conv: nn.Sequential
    pool: nn.AvgPool2d
    dropout: nn.Dropout

    def __init__(self, attention_mode: Optional[str] = None, in_channels: int = 16, temp_filter_length: int = 15,
                 pool_length: int = 8, pool_stride: int = 8, drop_prob: float = 0.5, activation = nn.ELU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, temp_filter_length), padding=(0, temp_filter_length // 2),
                      bias=False, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, (1, 1), bias=False), nn.BatchNorm2d(in_channels), activation())
        self.add_module("conv", self.conv)
        self.pool = nn.AvgPool2d((1, pool_length), stride=(1, pool_stride))
        self.add_module("pool", self.pool)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        if attention_mode is not None:
            raise NotImplementedError(f"AttentionBaseNet: attention_mode={attention_mode!r} is not ported to Typhon yet")

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.pool(self.conv(x)))

    def _get_name(self) -> str:
        return "_ChannelAttentionBlock"

def _seq_lengths(n: int, ks: List[int], pls: List[int], pss: List[int]) -> List[int]:
    out = List[int]()
    o = n
    for i in range(len(ks)):
        k = ks[i]
        o = o + 2 * (k // 2) - k + 1
        o = int(math.floor((o - pls[i]) / pss[i] + 1))
        o = max(1, o)
        out.append(o)
    return out

class AttentionBaseNet(EEGModuleMixin, nn.Module):
    """AttentionBaseNet from Wimpff et al (2023). Only attention_mode=None
    (the default) is supported in Typhon."""
    input_block: _FeatureExtractor
    channel_expansion: nn.Sequential
    channel_attention_block: _ChannelAttentionBlock
    final_layer: nn.Sequential

    def __init__(self, n_times: Optional[int] = None, n_chans: Optional[int] = None, n_outputs: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, sfreq: Optional[float] = None,
                 input_window_seconds: Optional[float] = None, n_temporal_filters: int = 40,
                 temp_filter_length_inp: int = 25, spatial_expansion: int = 1, pool_length_inp: int = 75,
                 pool_stride_inp: int = 15, drop_prob_inp: float = 0.5, ch_dim: int = 16,
                 temp_filter_length: int = 15, pool_length: int = 8, pool_stride: int = 8,
                 drop_prob_attn: float = 0.5, attention_mode: Optional[str] = None, reduction_rate: int = 4,
                 use_mlp: bool = False, freq_idx: int = 0, n_codewords: int = 4, kernel_size: int = 9,
                 activation = nn.ELU, extra_params: bool = False):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        min_req = pool_stride_inp * (pool_length - 1) + pool_length_inp
        tfl_inp = temp_filter_length_inp
        pl_inp = pool_length_inp
        ps_inp = pool_stride_inp
        tfl = temp_filter_length
        pl = pool_length
        ps = pool_stride
        if self.n_times < min_req:
            sf = self.n_times / min_req
            tfl_inp = max(1, int(tfl_inp * sf))
            pl_inp = max(1, int(pl_inp * sf))
            ps_inp = max(1, int(ps_inp * sf))
            tfl = max(1, int(tfl * sf))
            pl = max(1, int(pl * sf))
            ps = max(1, int(ps * sf))
        self.input_block = _FeatureExtractor(self.n_chans, n_temporal_filters, tfl_inp, spatial_expansion, pl_inp, ps_inp,
                                             drop_prob_inp, activation)
        self.add_module("input_block", self.input_block)
        self.channel_expansion = nn.Sequential(nn.Conv2d(n_temporal_filters * spatial_expansion, ch_dim, (1, 1), bias=False),
                                               nn.BatchNorm2d(ch_dim), activation())
        self.add_module("channel_expansion", self.channel_expansion)
        sl = _seq_lengths(self.n_times, [tfl_inp, tfl], [pl_inp, pl], [ps_inp, ps])
        self.channel_attention_block = _ChannelAttentionBlock(attention_mode, ch_dim, tfl, pl, ps, drop_prob_attn, activation)
        self.add_module("channel_attention_block", self.channel_attention_block)
        self.final_layer = nn.Sequential(nn.Flatten(), nn.Linear(sl[len(sl) - 1] * ch_dim, self.n_outputs))
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_block(x)
        x = self.channel_expansion(x)
        x = self.channel_attention_block(x)
        return self.final_layer(x)

    def _get_name(self) -> str:
        return "AttentionBaseNet"
