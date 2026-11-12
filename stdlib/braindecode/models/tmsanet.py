# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from braindecode.functional import scaled_dot_product_attention
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock

class _TMSAAttention(nn.Module):
    head_dim: int
    num_heads: int
    local_convs: nn.ModuleList
    local_bn: nn.BatchNorm1d
    local_drop: nn.Dropout
    w_q: nn.Linear
    w_k_local: nn.Linear
    w_k_global: nn.Linear
    w_v: nn.Linear
    w_o: nn.Linear
    att_drop: nn.Dropout
    att_p: float

    def __init__(self, embed_dim: int, num_heads: int, drop_prob: float, att_drop_prob: float):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        inner = self.head_dim * num_heads
        self.add_module("to_channels", Rearrange("b n d -> b d n"))
        self.local_convs = nn.ModuleList()
        self.add_module("local_convs", self.local_convs)
        for k in (3, 5):
            self.local_convs.append(nn.Conv1d(embed_dim, embed_dim, k, padding=k // 2))
        self.local_bn = nn.BatchNorm1d(2 * embed_dim)
        self.add_module("local_bn", self.local_bn)
        self.local_drop = nn.Dropout(drop_prob)
        self.add_module("local_drop", self.local_drop)
        self.add_module("to_tokens", Rearrange("b d n -> b n d"))
        self.w_q = nn.Linear(embed_dim, inner)
        self.add_module("w_q", self.w_q)
        self.w_k_local = nn.Linear(2 * embed_dim, inner)
        self.add_module("w_k_local", self.w_k_local)
        self.w_k_global = nn.Linear(embed_dim, inner)
        self.add_module("w_k_global", self.w_k_global)
        self.w_v = nn.Linear(embed_dim, inner)
        self.add_module("w_v", self.w_v)
        self.w_o = nn.Linear(inner, embed_dim)
        self.add_module("w_o", self.w_o)
        self.add_module("split_heads", Rearrange("b n (h d) -> b h n d", h=num_heads))
        self.add_module("merge_heads", Rearrange("b h n d -> b n (h d)"))
        self.att_drop = nn.Dropout(att_drop_prob)
        self.add_module("att_drop", self.att_drop)
        self.att_p = att_drop_prob

    def _split(self, t: Tensor) -> Tensor:
        return rearrange(t, "b n (h d) -> b h n d", h=self.num_heads)

    def forward(self, x: Tensor) -> Tensor:
        local = x.transpose(1, 2)
        local = torch.cat([c(local) for c in self.local_convs.children()], dim=1)
        local = self.local_drop(self.local_bn(local)).transpose(1, 2)
        q = self._split(self.w_q(x))
        kl = self._split(self.w_k_local(local))
        kg = self._split(self.w_k_global(x))
        v = self._split(self.w_v(x))
        dp = self.att_p if self.training else 0.0
        xl = scaled_dot_product_attention(q, kl, v, dropout_p=dp, exact=True)
        xg = scaled_dot_product_attention(q, kg, v, dropout_p=dp, exact=True)
        return self.w_o(rearrange(xl + xg, "b h n d -> b n (h d)"))

    def _get_name(self) -> str:
        return "_TMSAAttention"

class _TMSABlock(nn.Module):
    norm1: nn.LayerNorm
    attention: _TMSAAttention
    norm2: nn.LayerNorm
    feed_forward: FeedForwardBlock

    def __init__(self, embed_dim: int, num_heads: int, fc_ratio: int, drop_prob: float, att_drop_prob: float,
                 fc_drop_prob: float, activation):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.add_module("norm1", self.norm1)
        self.attention = _TMSAAttention(embed_dim, num_heads, drop_prob, att_drop_prob)
        self.add_module("attention", self.attention)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.add_module("norm2", self.norm2)
        self.feed_forward = FeedForwardBlock(embed_dim, fc_ratio, fc_drop_prob, activation, output_drop_p=fc_drop_prob)
        self.add_module("feed_forward", self.feed_forward)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))

    def _get_name(self) -> str:
        return "_TMSABlock"

class TMSANet(EEGModuleMixin, nn.Module):
    """TMSA-Net from Zhao et al (2025)."""
    add_channel_dim: Rearrange
    temp_conv1: nn.Conv2d
    temp_conv2: nn.Conv2d
    bn1: nn.BatchNorm2d
    spatial_conv: nn.Conv2d
    activation: nn.Module
    bn2: nn.BatchNorm2d
    to_sequence: Rearrange
    avg_pool: nn.AvgPool1d
    dropout: nn.Dropout
    transformer: nn.Sequential
    flatten: Rearrange
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, embed_dim: int = 19, pool_size: int = 50, pool_stride: int = 15,
                 num_heads: int = 4, fc_ratio: int = 2, depth: int = 1, drop_prob: float = 0.5,
                 att_drop_prob: float = 0.5, fc_drop_prob: float = 0.5, activation = nn.GELU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if embed_dim < num_heads:
            raise ValueError(f"embed_dim ({embed_dim}) must be at least num_heads ({num_heads}).")
        self.add_channel_dim = Rearrange("b c t -> b 1 c t")
        self.add_module("add_channel_dim", self.add_channel_dim)
        self.temp_conv1 = nn.Conv2d(1, embed_dim, (1, 31), padding=(0, 15))
        self.add_module("temp_conv1", self.temp_conv1)
        self.temp_conv2 = nn.Conv2d(1, embed_dim, (1, 15), padding=(0, 7))
        self.add_module("temp_conv2", self.temp_conv2)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.add_module("bn1", self.bn1)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (self.n_chans, 1))
        self.add_module("spatial_conv", self.spatial_conv)
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.add_module("bn2", self.bn2)
        self.to_sequence = Rearrange("b d 1 t -> b d t")
        self.add_module("to_sequence", self.to_sequence)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        self.add_module("avg_pool", self.avg_pool)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.transformer = nn.Sequential()
        self.transformer.add_module("0", Rearrange("b d n -> b n d"))
        for i in range(depth):
            self.transformer.add_module(str(i + 1), _TMSABlock(embed_dim, num_heads, fc_ratio, drop_prob, att_drop_prob,
                                                               fc_drop_prob, activation))
        self.add_module("transformer", self.transformer)
        self.flatten = Rearrange("b n d -> b (d n)")
        self.add_module("flatten", self.flatten)
        nt = (self.n_times - pool_size) // pool_stride + 1
        if nt < 1:
            raise ValueError(f"n_times ({self.n_times}) must be at least pool_size ({pool_size}) so the average pooling keeps at least one temporal token (would keep {nt}).")
        self.final_layer = nn.Linear(embed_dim * nt, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.add_channel_dim(x)
        x = self.bn1(self.temp_conv1(x) + self.temp_conv2(x))
        x = self.bn2(self.activation(self.spatial_conv(x)))
        x = self.dropout(self.avg_pool(self.to_sequence(x)))
        x = self.transformer(x)
        return self.final_layer(self.flatten(x))

    def _get_name(self) -> str:
        return "TMSANet"
