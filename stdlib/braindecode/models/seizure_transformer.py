# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import sinusoidal_positional_encoding

def _same_padding(k: int) -> nn.ConstantPad1d:
    return nn.ConstantPad1d(((k - 1) // 2, k // 2), 0.0)

class _ConvBlock(nn.Module):
    pad: nn.ConstantPad1d
    conv: nn.Conv1d
    activation: nn.Module
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation):
        super().__init__()
        self.pad = _same_padding(kernel_size)
        self.add_module("pad", self.pad)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.add_module("conv", self.conv)
        self.activation = activation()
        self.add_module("activation", self.activation)
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv(self.pad(x)))
    def _get_name(self) -> str:
        return "_ConvBlock"

class _ResidualConvBlock(nn.Module):
    pad: nn.ConstantPad1d
    norm1: nn.BatchNorm1d
    activation1: nn.Module
    dropout1: nn.Dropout1d
    conv1: nn.Conv1d
    norm2: nn.BatchNorm1d
    activation2: nn.Module
    dropout2: nn.Dropout1d
    conv2: nn.Conv1d
    def __init__(self, n: int, kernel_size: int, drop_prob: float, activation):
        super().__init__()
        self.pad = _same_padding(kernel_size)
        self.add_module("pad", self.pad)
        self.norm1 = nn.BatchNorm1d(n, eps=1e-3)
        self.add_module("norm1", self.norm1)
        self.activation1 = activation()
        self.add_module("activation1", self.activation1)
        self.dropout1 = nn.Dropout1d(drop_prob)
        self.add_module("dropout1", self.dropout1)
        self.conv1 = nn.Conv1d(n, n, kernel_size)
        self.add_module("conv1", self.conv1)
        self.norm2 = nn.BatchNorm1d(n, eps=1e-3)
        self.add_module("norm2", self.norm2)
        self.activation2 = activation()
        self.add_module("activation2", self.activation2)
        self.dropout2 = nn.Dropout1d(drop_prob)
        self.add_module("dropout2", self.dropout2)
        self.conv2 = nn.Conv1d(n, n, kernel_size)
        self.add_module("conv2", self.conv2)
    def forward(self, x: Tensor) -> Tensor:
        o = self.conv1(self.pad(self.dropout1(self.activation1(self.norm1(x)))))
        o = self.conv2(self.pad(self.dropout2(self.activation2(self.norm2(o)))))
        return x + o
    def _get_name(self) -> str:
        return "_ResidualConvBlock"

class SeizureTransformer(EEGModuleMixin, nn.Module):
    """SeizureTransformer from Wu et al (2025). Architecture only.

    Batch-norm-in-training with batch size 1 is not special-cased."""
    max_n_times: int
    encoder: nn.ModuleList
    pool: nn.MaxPool1d
    res_blocks: nn.Sequential
    positional_encoding: Tensor
    positional_dropout: nn.Dropout
    transformer: nn.TransformerEncoder
    decoder: nn.ModuleList
    final_layer: nn.Conv1d

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 n_filters: List[int] = [32, 64, 128, 256, 512], encoder_kernel_sizes: List[int] = [11, 9, 7, 7, 5],
                 decoder_kernel_sizes: List[int] = [3, 5, 5, 7, 7], res_kernel_sizes: List[int] = [3, 3, 3, 3, 2, 3, 2],
                 final_kernel_size: int = 11, num_layers: int = 8, num_heads: int = 4, dim_feedforward: int = 2048,
                 drop_prob: float = 0.1, activation = nn.ELU, activation_res = nn.ReLU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if not (len(n_filters) == len(encoder_kernel_sizes) == len(decoder_kernel_sizes)):
            raise ValueError("n_filters, encoder_kernel_sizes and decoder_kernel_sizes must have the same length.")
        E = n_filters[-1]
        if E % num_heads != 0:
            raise ValueError(f"n_filters[-1] ({E}) must be divisible by num_heads ({num_heads}).")
        self.max_n_times = self.n_times
        ns = self.n_times
        for _ in n_filters:
            ns = (ns + 1) // 2
        self.encoder = nn.ModuleList()
        ins = [self.n_chans] + n_filters[:-1]
        for i in range(len(n_filters)):
            self.encoder.append(_ConvBlock(ins[i], n_filters[i], encoder_kernel_sizes[i], activation))
        self.add_module("encoder", self.encoder)
        self.pool = nn.MaxPool1d(2, 2, ceil_mode=True)
        self.add_module("pool", self.pool)
        self.res_blocks = nn.Sequential()
        for i in range(len(res_kernel_sizes)):
            self.res_blocks.add_module(str(i), _ResidualConvBlock(E, res_kernel_sizes[i], drop_prob, activation_res))
        self.add_module("res_blocks", self.res_blocks)
        self.positional_encoding = sinusoidal_positional_encoding(ns, E)
        self.register_buffer("positional_encoding", self.positional_encoding, persistent=False)
        self.positional_dropout = nn.Dropout(drop_prob)
        self.add_module("positional_dropout", self.positional_dropout)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=E, nhead=num_heads, dim_feedforward=dim_feedforward,
                                                                            dropout=drop_prob, batch_first=True),
                                                 num_layers=num_layers, enable_nested_tensor=False)
        self.add_module("transformer", self.transformer)
        self.decoder = nn.ModuleList()
        df = n_filters[::-1]
        dins = [E] + df[:-1]
        for i in range(len(df)):
            self.decoder.append(_ConvBlock(dins[i], df[i], decoder_kernel_sizes[i], activation))
        self.add_module("decoder", self.decoder)
        self.final_layer = nn.Conv1d(n_filters[0], self.n_outputs, final_kernel_size, padding="same")
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] > self.max_n_times:
            raise ValueError(f"Input has {x.shape[-1]} samples, more than n_times={self.max_n_times}.")
        skips = List[Tensor]()
        for b in self.encoder.children():
            x = b(x)
            skips.append(x)
            x = self.pool(x)
        f = self.res_blocks(x)
        st = f.transpose(1, 2)
        st = st + self.positional_encoding[:st.shape[1]]
        st = self.transformer(self.positional_dropout(st))
        x = st.transpose(1, 2) + f
        dec = self.decoder.children()
        for lv in range(len(dec)):
            sk = skips[len(skips) - 1 - lv]
            x = torch.repeat_interleave(x, 2, dim=-1)[..., :sk.shape[-1]]
            x = dec[lv](x) + sk
        return self.final_layer(x)

    def _get_name(self) -> str:
        return "SeizureTransformer"
