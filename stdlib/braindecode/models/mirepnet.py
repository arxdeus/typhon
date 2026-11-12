# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# MIRepNet (channel_strategy="native" only; return_features not supported).

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock, MultiHeadAttention, PreNormTransformerBlock

class _PatchEmbedding(nn.Module):
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    bn: nn.BatchNorm2d
    elu: nn.Module
    pool: nn.AvgPool2d
    dropout: nn.Dropout
    projection: nn.Conv2d
    def __init__(self, n_chans: int, embed_dim: int, n_filters_time: int, n_filters_spat: int, filter_time_length: int,
                 pool_time_length: int, pool_time_stride: int, activation, drop_prob: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filters_time, kernel_size=(1, filter_time_length))
        self.add_module("conv1", self.conv1)
        self.conv2 = nn.Conv2d(n_filters_time, n_filters_spat, kernel_size=(n_chans, 1))
        self.add_module("conv2", self.conv2)
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.add_module("bn", self.bn)
        self.elu = activation()
        self.add_module("elu", self.elu)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride))
        self.add_module("pool", self.pool)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.projection = nn.Conv2d(n_filters_spat, embed_dim, kernel_size=(1, 1))
        self.add_module("projection", self.projection)
    def forward(self, x: Tensor) -> Tensor:
        bt = self.conv1.bias.__val__()
        bs = self.conv2.bias.__val__()
        w_spat = self.conv2.weight.squeeze(-1).transpose(1, 2)
        weight = torch.matmul(w_spat, self.conv1.weight.flatten(1))
        bias = torch.matmul(self.conv2.weight.sum((2, 3)), bt) + bs
        x = F.conv1d(x, weight, bias).unsqueeze(2)
        x = self.dropout(self.pool(self.elu(self.bn(x))))
        x = self.projection(x)
        return x.flatten(start_dim=2).transpose(1, 2).contiguous()
    def _get_name(self) -> str:
        return "_PatchEmbedding"

class MIRepNet(EEGModuleMixin, nn.Module):
    """MIRepNet from Liu et al (2025)."""
    embed_dim: int
    embedding: _PatchEmbedding
    transformer: nn.Sequential
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, embed_dim: int = 256,
                 n_filters_time: int = 64, n_filters_spat: int = 128, filter_time_length: int = 25,
                 pool_time_length: int = 75, pool_time_stride: int = 15, num_layers: int = 6, num_heads: int = 8,
                 feedforward_expansion: int = 4, activation = nn.ELU, activation_trans = nn.GELU,
                 drop_prob: float = 0.5, att_drop_prob: float = 0.5, feedforward_drop_prob: float = 0.5,
                 attention_scale: Optional[float] = None, return_features: bool = False,
                 channel_strategy: str = "native"):
        if channel_strategy != "native":
            raise NotImplementedError("MIRepNet: only channel_strategy='native' is supported in Codon")
        if return_features:
            raise NotImplementedError("MIRepNet: return_features=True is not supported in Codon")
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        for name, v in (("embed_dim", embed_dim), ("n_filters_time", n_filters_time), ("n_filters_spat", n_filters_spat),
                        ("filter_time_length", filter_time_length), ("pool_time_length", pool_time_length),
                        ("pool_time_stride", pool_time_stride), ("num_layers", num_layers), ("num_heads", num_heads),
                        ("feedforward_expansion", feedforward_expansion)):
            if v <= 0:
                raise ValueError(f"{name} must be positive.")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        for name, p in (("drop_prob", drop_prob), ("att_drop_prob", att_drop_prob),
                        ("feedforward_drop_prob", feedforward_drop_prob)):
            if not (0 <= p <= 1):
                raise ValueError(f"{name} must be between 0 and 1.")
        if attention_scale is not None and attention_scale <= 0:
            raise ValueError("attention_scale must be positive or None.")
        self.embed_dim = embed_dim
        self.mapping = {"embedding.projection.0.weight": "embedding.projection.weight",
                        "embedding.projection.0.bias": "embedding.projection.bias",
                        "clshead.weight": "final_layer.weight", "clshead.bias": "final_layer.bias"}
        self.embedding = _PatchEmbedding(self.n_chans, embed_dim, n_filters_time, n_filters_spat, filter_time_length,
                                         pool_time_length, pool_time_stride, activation, drop_prob)
        self.add_module("embedding", self.embedding)
        sc = attention_scale if attention_scale is not None else embed_dim ** (-0.5)
        self.transformer = nn.Sequential()
        for _ in range(num_layers):
            self.transformer.append(PreNormTransformerBlock(
                embed_dim, MultiHeadAttention(embed_dim, num_heads, att_drop_prob, scale=sc),
                FeedForwardBlock(emb_size=embed_dim, expansion=feedforward_expansion, drop_p=feedforward_drop_prob,
                                 activation=activation_trans), att_drop_prob))
        self.add_module("transformer", self.transformer)
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def _forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.transformer(self.embedding(x)).mean(dim=1))

    def forward(self, x: Tensor) -> Tensor:
        if self.training and x.shape[0] == 1:
            bns = [m for m in self.modules() if "BatchNorm" in m._get_name() and m.training]
            for m in bns:
                m.train(False)
            y = self._forward(x)
            for m in bns:
                m.train(True)
            return y
        return self._forward(x)

    def _get_name(self) -> str:
        return "MIRepNet"
