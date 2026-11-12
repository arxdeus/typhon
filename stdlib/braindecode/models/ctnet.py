# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock, LearnedPositionalEncoding, MultiHeadAttention, PostNormTransformerBlock

class _PatchEmbeddingEEGNet(nn.Module):
    eegnet_module: nn.Sequential
    projection: nn.Sequential
    _dw: nn.Conv2d

    def __init__(self, n_filters_time: int = 16, kernel_size: int = 64, depth_multiplier: int = 2, pool_size_1: int = 8,
                 pool_size_2: int = 8, drop_prob: float = 0.3, n_chans: int = 22, activation = nn.ELU):
        super().__init__()
        nfo = depth_multiplier * n_filters_time
        c0 = nn.Conv2d(1, n_filters_time, kernel_size=(1, kernel_size), stride=(1, 1), padding="same", bias=False)
        self._dw = nn.Conv2d(n_filters_time, nfo, kernel_size=(n_chans, 1), stride=(1, 1), groups=n_filters_time,
                             padding="valid", bias=False)
        self.eegnet_module = nn.Sequential(
            c0, nn.BatchNorm2d(n_filters_time), self._dw, nn.BatchNorm2d(nfo), activation(),
            nn.AvgPool2d(kernel_size=(1, pool_size_1)), nn.Dropout(drop_prob),
            nn.Conv2d(nfo, nfo, kernel_size=(1, 16), padding="same", bias=False),
            nn.BatchNorm2d(nfo), activation(), nn.AvgPool2d(kernel_size=(1, pool_size_2)), nn.Dropout(drop_prob))
        self.add_module("eegnet_module", self.eegnet_module)
        self.projection = nn.Sequential(Rearrange("b e h w -> b (h w) e"))
        self.add_module("projection", self.projection)

    def forward(self, x: Tensor) -> Tensor:
        layers = self.eegnet_module.children()
        x = layers[1](layers[0](x))
        b, f, c, t = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = torch.matmul(self._dw.weight.view(f, -1, c), x).view(b, -1, 1, t)
        for i in range(3, len(layers)):
            x = layers[i](x)
        return self.projection(x)

    def _get_name(self) -> str:
        return "_PatchEmbeddingEEGNet"

def _resolve_dims(depth_multiplier: Optional[int], n_filters_time: Optional[int], emb_size: Optional[int]) -> Tuple[int, int, int]:
    for name, val in (("depth_multiplier", depth_multiplier), ("n_filters_time", n_filters_time), ("emb_size", emb_size)):
        if val is not None and val.__val__() <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")
    missing = 0
    for v in (depth_multiplier, n_filters_time, emb_size):
        if v is None:
            missing += 1
    if missing >= 2:
        raise ValueError("Specify exactly two of {depth_multiplier, n_filters_time, emb_size}; the third will be inferred.")
    if emb_size is None:
        return (depth_multiplier.__val__(), n_filters_time.__val__(), depth_multiplier.__val__() * n_filters_time.__val__())
    e = emb_size.__val__()
    if n_filters_time is None:
        d = depth_multiplier.__val__()
        if e % d != 0:
            raise ValueError(f"emb_size={e} must be divisible by depth_multiplier={d}")
        return (d, e // d, e)
    n = n_filters_time.__val__()
    if depth_multiplier is None:
        if e % n != 0:
            raise ValueError(f"emb_size={e} must be divisible by n_filters_time={n}")
        return (e // n, n, e)
    d = depth_multiplier.__val__()
    if d * n != e:
        raise ValueError(f"`depth_multiplier * n_filters_time` must equal `emb_size`, but got {d} * {n} = {d * n} != {e}.")
    return (d, n, e)

class CTNet(EEGModuleMixin, nn.Module):
    """CTNet from Zhao et al (2024)."""
    cnn_drop_prob: float
    pool_size_1: int
    pool_size_2: int
    kernel_size: int
    att_positional_drop_prob: float
    final_drop_prob: float
    num_heads: int
    num_layers: int
    sequence_length: int
    depth_multiplier: int
    n_filters_time: int
    embed_dim: int
    ensuredim: Rearrange
    flatten: nn.Flatten
    cnn: _PatchEmbeddingEEGNet
    position: LearnedPositionalEncoding
    trans: nn.Sequential
    flatten_drop_layer: nn.Sequential
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None, sfreq: Optional[float] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, activation_patch = nn.ELU,
                 activation_transformer = nn.GELU, cnn_drop_prob: float = 0.3, att_positional_drop_prob: float = 0.1,
                 final_drop_prob: float = 0.5, num_heads: int = 4, embed_dim: Optional[int] = 40,
                 num_layers: int = 6, n_filters_time: Optional[int] = None, kernel_size: int = 64,
                 depth_multiplier: Optional[int] = 2, pool_size_1: int = 8, pool_size_2: int = 8):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.cnn_drop_prob = cnn_drop_prob
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.kernel_size = kernel_size
        self.att_positional_drop_prob = att_positional_drop_prob
        self.final_drop_prob = final_drop_prob
        self.num_heads = num_heads
        self.num_layers = num_layers
        s1 = int(math.floor((self.n_times - pool_size_1) / pool_size_1 + 1))
        self.sequence_length = int(math.floor((s1 - pool_size_2) / pool_size_2 + 1))
        d, n, e = _resolve_dims(depth_multiplier, n_filters_time, embed_dim)
        self.depth_multiplier = d
        self.n_filters_time = n
        self.embed_dim = e
        self.ensuredim = Rearrange("batch nchans time -> batch 1 nchans time")
        self.add_module("ensuredim", self.ensuredim)
        self.flatten = nn.Flatten()
        self.add_module("flatten", self.flatten)
        self.cnn = _PatchEmbeddingEEGNet(n, kernel_size, d, pool_size_1, pool_size_2, cnn_drop_prob, self.n_chans, activation_patch)
        self.add_module("cnn", self.cnn)
        length = 100
        needed = max(int(self.n_times / (pool_size_1 * pool_size_1)), self.sequence_length)
        if needed > length:
            length = needed
        self.position = LearnedPositionalEncoding(length, e, att_positional_drop_prob)
        self.add_module("position", self.position)
        layers = nn.Sequential()
        for i in range(num_layers):
            layers.add_module(str(i), PostNormTransformerBlock(e, nn.Sequential(MultiHeadAttention(e, num_heads, 0.5)),
                              nn.Sequential(FeedForwardBlock(e, expansion=4, drop_p=0.5, activation=activation_transformer)), 0.5))
        self.trans = nn.Sequential()
        self.trans.add_module("layers", layers)
        self.add_module("trans", self.trans)
        self.flatten_drop_layer = nn.Sequential(nn.Flatten(), nn.Dropout(p=final_drop_prob))
        self.add_module("flatten_drop_layer", self.flatten_drop_layer)
        self.final_layer = nn.Linear(in_features=e * self.sequence_length, out_features=self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ensuredim(x)
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(float(self.embed_dim))
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        features = cnn + trans
        return self.final_layer(self.flatten_drop_layer(features))

    def _get_name(self) -> str:
        return "CTNet"
