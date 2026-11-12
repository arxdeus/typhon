# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin

class _TSConv(nn.Sequential):
    _c0: nn.Conv2d
    _bn1: nn.BatchNorm2d
    _c2: nn.Conv2d

    def __init__(self, n_channels: int, n_filters: int, conv1_kernel_size: int, conv2_kernel_size: int,
                 depth_multiplier: int, pool1_size: int, pool2_size: int, drop_prob: float, activation = nn.ELU):
        super().__init__()
        nd = n_filters * depth_multiplier
        self._c0 = nn.Conv2d(1, n_filters, kernel_size=(1, conv1_kernel_size), padding="same", bias=False)
        self._bn1 = nn.BatchNorm2d(n_filters)
        self._c2 = nn.Conv2d(n_filters, nd, kernel_size=(n_channels, 1), groups=n_filters, bias=False)
        mods = [self._c0, self._bn1, self._c2]
        for i in range(3):
            self.add_module(str(i), mods[i])
        rest = List[nn.Module]()
        rest.append(nn.BatchNorm2d(nd))
        rest.append(activation())
        rest.append(nn.AvgPool2d(kernel_size=(1, pool1_size)))
        rest.append(nn.Dropout(drop_prob))
        rest.append(nn.Conv2d(nd, nd, kernel_size=(1, conv2_kernel_size), padding="same", groups=nd, bias=False))
        rest.append(nn.BatchNorm2d(nd))
        rest.append(activation())
        rest.append(nn.AvgPool2d(kernel_size=(1, pool2_size)))
        rest.append(nn.Dropout(drop_prob))
        for i in range(len(rest)):
            self.add_module(str(i + 3), rest[i])

    def forward(self, x: Tensor) -> Tensor:
        # braindecode switches to an algebraically equal spatial-first path
        # for very large inputs; the direct path is used here.
        x = self._bn1(self._c0(x))
        b, f, c, t = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = torch.matmul(self._c2.weight.view(f, -1, c), x).view(b, -1, 1, t)
        ch = self.children()
        for i in range(3, len(ch)):
            x = ch[i](x)
        return x

    def _get_name(self) -> str:
        return "_TSConv"

class _PositionalEncoding(nn.Module):
    seq_length: int
    d_model: int
    pe: Tensor

    def __init__(self, seq_length: int, d_model: int):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.pe = nn.Parameter(torch.zeros(1, seq_length, d_model))
        self.register_parameter("pe", self.pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe

    def _get_name(self) -> str:
        return "_PositionalEncoding"

class _Transformer(nn.Module):
    cls_embedding: Tensor
    pos_embedding: _PositionalEncoding
    dropout: nn.Dropout
    trans: nn.TransformerEncoder

    def __init__(self, seq_length: int, d_model: int, num_heads: int, ffn_expansion_factor: float,
                 drop_prob: float = 0.5, num_layers: int = 4):
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_parameter("cls_embedding", self.cls_embedding)
        self.pos_embedding = _PositionalEncoding(seq_length + 1, d_model)
        self.add_module("pos_embedding", self.pos_embedding)
        dim_ff = int(d_model * ffn_expansion_factor)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, drop_prob,
                                                                      batch_first=True, norm_first=True),
                                           num_layers, norm=nn.LayerNorm(d_model))
        self.add_module("trans", self.trans)

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        x = self.dropout(self.pos_embedding(x))
        return self.trans(x)[:, 0]

    def _get_name(self) -> str:
        return "_Transformer"

class MSVTNet(EEGModuleMixin, nn.Module):
    """MSVTNet from Liu et al (2024). return_features is not supported
    (forward has a single Tensor return type in Typhon)."""
    ensure_dim: Rearrange
    mstsconv: nn.ModuleList
    branch_head: nn.ModuleList
    transformer: _Transformer
    flatten_layer: nn.Flatten
    final_layer: nn.Linear

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_filters_list = (9, 9, 9, 9),
                 conv1_kernels_size = (15, 31, 63, 125), conv2_kernel_size: int = 15, depth_multiplier: int = 2,
                 pool1_size: int = 8, pool2_size: int = 7, drop_prob: float = 0.3, num_heads: int = 8,
                 ffn_expansion_factor: float = 1, att_drop_prob: float = 0.5, num_layers: int = 2,
                 activation = nn.ELU, return_features: bool = False):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if return_features:
            raise NotImplementedError("MSVTNet(return_features=True) is not supported in Typhon")
        nfl = [int(v) for v in n_filters_list]
        cks = [int(v) for v in conv1_kernels_size]
        assert len(nfl) == len(cks), "The length of n_filters_list and conv1_kernel_sizes should be equal."
        self.ensure_dim = Rearrange("batch chans time -> batch 1 chans time")
        self.add_module("ensure_dim", self.ensure_dim)
        self.mstsconv = nn.ModuleList()
        self.add_module("mstsconv", self.mstsconv)
        for b in range(len(nfl)):
            self.mstsconv.append(nn.Sequential(_TSConv(self.n_chans, nfl[b], cks[b], conv2_kernel_size, depth_multiplier,
                                                       pool1_size, pool2_size, drop_prob, activation),
                                               Rearrange("batch channels 1 time -> batch time channels")))
        # torch probes three times in train mode (randn input, BatchNorm
        # running stats updated, dropout RNG drawn); mirrored exactly.
        outs = self._probe()
        self.branch_head = nn.ModuleList()
        self.add_module("branch_head", self.branch_head)
        for b in range(len(nfl)):
            self.branch_head.append(nn.Sequential(nn.Flatten(), nn.Linear(outs[b].flatten(1).shape[1], self.n_outputs)))
        cat = torch.cat(self._probe(), dim=2)
        seq_len, d_model = cat.shape[1], cat.shape[2]
        self.transformer = _Transformer(seq_len, d_model, num_heads, float(ffn_expansion_factor), att_drop_prob, num_layers)
        self.add_module("transformer", self.transformer)
        linear_in = self.transformer(torch.cat(self._probe(), dim=2)).flatten(1).shape[1]
        self.flatten_layer = nn.Flatten()
        self.add_module("flatten_layer", self.flatten_layer)
        self.final_layer = nn.Linear(linear_in, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def _probe(self) -> List[Tensor]:
        p = torch.randn(1, 1, self.n_chans, self.n_times)
        return [m(p) for m in self.mstsconv.children()]

    def forward(self, x: Tensor) -> Tensor:
        x = self.ensure_dim(x)
        xl = [m(x) for m in self.mstsconv.children()]
        x = torch.stack(xl, dim=2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.transformer(x)
        return self.final_layer(x)

    def _get_name(self) -> str:
        return "MSVTNet"
