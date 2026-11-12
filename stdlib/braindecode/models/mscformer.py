# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock, LearnedPositionalEncoding, MultiHeadAttention, PostNormTransformerBlock

class _MultiScalePatchEmbeddingCNN(nn.Module):
    branches: nn.ModuleList
    projection: nn.Sequential
    _t: List[nn.Conv2d]
    _s: List[nn.Conv2d]

    def __init__(self, kernel_sizes: List[int], n_filters_time: int, pooling_size: int, drop_prob: float,
                 n_chans: int, activation = nn.ELU):
        super().__init__()
        self.branches = nn.ModuleList()
        self.add_module("branches", self.branches)
        self._t = []
        self._s = []
        for k in kernel_sizes:
            t = nn.Conv2d(1, n_filters_time, (1, k), (1, 1), padding="same")
            s = nn.Conv2d(n_filters_time, n_filters_time, (n_chans, 1), (1, 1), groups=n_filters_time)
            self._t.append(t)
            self._s.append(s)
            self.branches.append(nn.Sequential(t, s, nn.BatchNorm2d(n_filters_time), activation(),
                                               nn.AvgPool2d((1, pooling_size)), nn.Dropout(drop_prob)))
        self.projection = nn.Sequential(Rearrange("b e h w -> b (h w) e"))
        self.add_module("projection", self.projection)

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze(1)
        outs = List[Tensor]()
        brs = self.branches.children()
        for bi in range(len(brs)):
            t = self._t[bi]
            s = self._s[bi]
            bt = t.bias.__val__()
            bs = s.bias.__val__()
            w = s.weight.squeeze(1) * t.weight.squeeze(1)
            b = torch.addcmul(bs, s.weight.sum([1, 2, 3]), bt)
            z = F.conv1d(x, w, b, 1, "same").unsqueeze(2)
            layers = brs[bi].children()
            for i in range(2, len(layers)):
                z = layers[i](z)
            outs.append(z)
        return self.projection(torch.cat(outs, dim=1))

    def _get_name(self) -> str:
        return "_MultiScalePatchEmbeddingCNN"

class MSCFormer(EEGModuleMixin, nn.Module):
    """MSCFormer from Zhao et al (2025)."""
    embed_dim: int
    ensuredim: Rearrange
    cnn: _MultiScalePatchEmbeddingCNN
    position: LearnedPositionalEncoding
    trans: nn.Sequential
    final_layer: nn.Sequential

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None, sfreq: Optional[float] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, kernel_sizes = (85, 65, 45), n_filters_time: int = 16,
                 pooling_size: int = 44, cnn_drop_prob: float = 0.5, activation_cnn = nn.ELU,
                 activation_ffn = nn.GELU, num_heads: int = 8, num_layers: int = 5, forward_expansion: int = 4,
                 att_drop_prob: float = 0.5, att_positional_drop_prob: float = 0.1, final_drop_prob: float = 0.25,
                 attention_scale: Optional[float] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if attention_scale is not None and attention_scale.__val__() <= 0:
            raise ValueError("attention_scale must be positive or None.")
        ks = [int(k) for k in kernel_sizes]
        self.embed_dim = n_filters_time * len(ks)
        self.ensuredim = Rearrange("batch nchans time -> batch 1 nchans time")
        self.add_module("ensuredim", self.ensuredim)
        self.cnn = _MultiScalePatchEmbeddingCNN(ks, n_filters_time, pooling_size, cnn_drop_prob, self.n_chans, activation_cnn)
        self.add_module("cnn", self.cnn)
        nt = int(math.floor((self.n_times - pooling_size) / pooling_size + 1)) + 1
        length = max(100, nt)
        self.position = LearnedPositionalEncoding(length, self.embed_dim, att_positional_drop_prob)
        self.add_module("position", self.position)
        sc = self.embed_dim ** (-0.5) if attention_scale is None else attention_scale.__val__()
        layers = nn.Sequential()
        for i in range(num_layers):
            layers.add_module(str(i), PostNormTransformerBlock(self.embed_dim,
                MultiHeadAttention(self.embed_dim, num_heads, att_drop_prob, scale=sc),
                FeedForwardBlock(self.embed_dim, expansion=forward_expansion, drop_p=att_drop_prob, activation=activation_ffn),
                att_drop_prob))
        self.trans = nn.Sequential()
        self.trans.add_module("layers", layers)
        self.add_module("trans", self.trans)
        self.final_layer = nn.Sequential(nn.Dropout(final_drop_prob), nn.Linear(self.embed_dim, self.n_outputs))
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(self.ensuredim(x))
        cls = torch.zeros(x.shape[0], 1, x.shape[2], dtype=x.dtype)
        x = torch.cat([cls, x], dim=1) * math.sqrt(float(self.embed_dim))
        x = self.trans(self.position(x))
        return self.final_layer(x[:, 0, :])

    def _get_name(self) -> str:
        return "MSCFormer"
