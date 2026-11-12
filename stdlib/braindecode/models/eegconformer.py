# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock, MultiHeadAttention, PreNormTransformerBlock

class _PatchEmbedding(nn.Module):
    shallownet: nn.Sequential
    projection: nn.Sequential
    conv_time: nn.Conv2d
    conv_spat: nn.Conv2d

    def __init__(self, n_filters_time: int, filter_time_length: int, n_channels: int, pool_time_length: int,
                 stride_avg_pool: int, drop_prob: float, activation = nn.ELU):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1))
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1))
        self.shallownet = nn.Sequential(self.conv_time, self.conv_spat, nn.BatchNorm2d(num_features=n_filters_time),
                                        activation(), nn.AvgPool2d(kernel_size=(1, pool_time_length), stride=(1, stride_avg_pool)),
                                        nn.Dropout(p=drop_prob))
        self.add_module("shallownet", self.shallownet)
        self.projection = nn.Sequential(nn.Conv2d(n_filters_time, n_filters_time, (1, 1), stride=(1, 1)),
                                        Rearrange("b d_model 1 seq -> b seq d_model"))
        self.add_module("projection", self.projection)

    def forward(self, x: Tensor) -> Tensor:
        bto = self.conv_time.bias
        bso = self.conv_spat.bias
        assert bto is not None and bso is not None
        bt: Tensor = bto
        bs: Tensor = bso
        weight = (self.conv_time.weight * self.conv_spat.weight.permute(1, 0, 2, 3)).sum(0)
        bias = self.conv_spat.weight.sum((2, 3)) @ bt + bs
        x = F.conv1d(x.squeeze(1), weight, bias).unsqueeze(2)
        ch = self.shallownet.children()
        for i in range(2, len(ch)):
            x = ch[i](x)
        return self.projection(x)

    def _get_name(self) -> str:
        return "_PatchEmbedding"

class _FullyConnected(nn.Module):
    hidden_channels: int
    fc: nn.Sequential

    def __init__(self, final_fc_length: int, drop_prob_1: float = 0.5, drop_prob_2: float = 0.3, out_channels: int = 256,
                 hidden_channels: int = 32, activation = nn.ELU):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.fc = nn.Sequential(nn.Linear(final_fc_length, out_channels), activation(), nn.Dropout(drop_prob_1),
                                nn.Linear(out_channels, hidden_channels), activation(), nn.Dropout(drop_prob_2))
        self.add_module("fc", self.fc)

    def forward(self, x: Tensor) -> Tensor:
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)

    def _get_name(self) -> str:
        return "_FullyConnected"

class EEGConformer(EEGModuleMixin, nn.Module):
    """EEG Conformer from Song et al (2022). activation_checkpointing is
    accepted but has no effect (no recomputation in Codon)."""
    return_features: bool
    activation_checkpointing: bool
    final_fc_length: int
    patch_embedding: _PatchEmbedding
    transformer: nn.Sequential
    fc: _FullyConnected
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None, n_filters_time: int = 40,
                 filter_time_length: int = 25, pool_time_length: int = 75, pool_time_stride: int = 15,
                 drop_prob: float = 0.5, num_layers: int = 6, num_heads: int = 10, att_drop_prob: float = 0.5,
                 final_fc_length = "auto", return_features: bool = False, activation = nn.ELU,
                 activation_transfor = nn.GELU, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, activation_checkpointing: bool = False):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.mapping["classification_head.fc.6.weight"] = "final_layer.final_layer.0.weight"
        self.mapping["classification_head.fc.6.bias"] = "final_layer.final_layer.0.bias"
        if not self.n_chans <= 64:
            print("UserWarning: This model has only been tested on no more than 64 channels. no guarantee to work with more channels.")
        self.return_features = return_features
        self.activation_checkpointing = activation_checkpointing
        self.patch_embedding = _PatchEmbedding(n_filters_time=n_filters_time, filter_time_length=filter_time_length,
                                               n_channels=self.n_chans, pool_time_length=pool_time_length,
                                               stride_avg_pool=pool_time_stride, drop_prob=drop_prob, activation=activation)
        self.add_module("patch_embedding", self.patch_embedding)
        if isinstance(final_fc_length, str):
            if final_fc_length != "auto":
                raise ValueError("final_fc_length must be 'auto' or an int")
            self.final_fc_length = self.get_fc_size()
        else:
            self.final_fc_length = int(final_fc_length)
        self.transformer = nn.Sequential()
        for i in range(num_layers):
            blk = PreNormTransformerBlock(n_filters_time, MultiHeadAttention(n_filters_time, num_heads, att_drop_prob),
                                          FeedForwardBlock(n_filters_time, expansion=4, drop_p=att_drop_prob,
                                                           activation=activation_transfor), att_drop_prob)
            self.transformer.add_module(str(i), blk)
        self.add_module("transformer", self.transformer)
        self.fc = _FullyConnected(final_fc_length=self.final_fc_length, activation=activation)
        self.add_module("fc", self.fc)
        self.final_layer = nn.Linear(self.fc.hidden_channels, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def get_fc_size(self) -> int:
        out = torch.zeros(1)
        with torch.no_grad():
            out = self.patch_embedding(torch.ones(1, 1, self.n_chans, self.n_times))
        return out.shape[1] * out.shape[2]

    def forward(self, x: Tensor) -> Tensor:
        x = torch.unsqueeze(x, dim=1)
        x = self.patch_embedding(x)
        feature = self.transformer(x)
        if self.return_features:
            return feature
        x = self.fc(feature)
        return self.final_layer(x)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "EEGConformer"
