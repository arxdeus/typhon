# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization
from einops.layers.torch import Rearrange
from braindecode.functional import scaled_dot_product_attention
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import CausalConv1d, Ensure4d, MaxNormLinear, MaxNormParametrize

def _register_max_norm(max_norm: Optional[float], *convs):
    if max_norm is None:
        return
    for i in static.range(static.len(convs)):
        c: nn.Module = convs[i]
        n = c._get_name()
        if n == "Conv1d" or n == "Conv2d" or n == "CausalConv1d":
            register_parametrization(c, "weight", MaxNormParametrize(max_norm.__val__()))

class _ConvBlock(nn.Module):
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    conv2: nn.Conv2d
    bn2: nn.BatchNorm2d
    activation2: nn.ELU
    pool2: nn.AvgPool2d
    drop2: nn.Dropout2d
    conv3: nn.Conv2d
    bn3: nn.BatchNorm2d
    activation3: nn.ELU
    pool3: nn.AvgPool2d
    drop3: nn.Dropout2d

    def __init__(self, n_channels: int, n_filters: int = 16, kernel_length_1: int = 64, kernel_length_2: int = 16,
                 pool_size_1: int = 8, pool_size_2: int = 7, depth_mult: int = 2, dropout: float = 0.3,
                 conv_max_norm: Optional[float] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=(kernel_length_1, 1), padding="same", bias=False)
        self.add_module("conv1", self.conv1)
        self.bn1 = nn.BatchNorm2d(num_features=n_filters, eps=0.0001)
        self.add_module("bn1", self.bn1)
        nd = n_filters * depth_mult
        self.conv2 = nn.Conv2d(n_filters, nd, groups=n_filters, kernel_size=(1, n_channels), padding="valid", bias=False)
        self.add_module("conv2", self.conv2)
        self.bn2 = nn.BatchNorm2d(num_features=nd, eps=0.0001)
        self.add_module("bn2", self.bn2)
        self.activation2 = nn.ELU()
        self.add_module("activation2", self.activation2)
        self.pool2 = nn.AvgPool2d(kernel_size=(pool_size_1, 1))
        self.add_module("pool2", self.pool2)
        self.drop2 = nn.Dropout2d(dropout)
        self.add_module("drop2", self.drop2)
        self.conv3 = nn.Conv2d(nd, nd, kernel_size=(kernel_length_2, 1), padding="same", bias=False)
        self.add_module("conv3", self.conv3)
        self.bn3 = nn.BatchNorm2d(num_features=nd, eps=0.0001)
        self.add_module("bn3", self.bn3)
        self.activation3 = nn.ELU()
        self.add_module("activation3", self.activation3)
        self.pool3 = nn.AvgPool2d(kernel_size=(pool_size_2, 1))
        self.add_module("pool3", self.pool3)
        self.drop3 = nn.Dropout2d(dropout)
        self.add_module("drop3", self.drop3)
        _register_max_norm(conv_max_norm, self.conv1, self.conv2, self.conv3)

    def forward(self, X: Tensor) -> Tensor:
        X = self.bn1(self.conv1(X))
        X = self.drop2(self.pool2(self.activation2(self.bn2(self.conv2(X)))))
        X = self.drop3(self.pool3(self.activation3(self.bn3(self.conv3(X)))))
        return X

    def _get_name(self) -> str:
        return "_ConvBlock"

class _MHA(nn.Module):
    input_dim: int
    head_dim: int
    embed_dim: int
    fc_q: nn.Linear
    fc_k: nn.Linear
    fc_v: nn.Linear
    fc_o: nn.Linear
    dropout: nn.Dropout

    def __init__(self, input_dim: int, head_dim: int, output_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.embed_dim = head_dim * num_heads
        self.fc_q = nn.Linear(input_dim, self.embed_dim)
        self.add_module("fc_q", self.fc_q)
        self.fc_k = nn.Linear(input_dim, self.embed_dim)
        self.add_module("fc_k", self.fc_k)
        self.fc_v = nn.Linear(input_dim, self.embed_dim)
        self.add_module("fc_v", self.fc_v)
        self.fc_o = nn.Linear(self.embed_dim, output_dim)
        self.add_module("fc_o", self.fc_o)
        self.dropout = nn.Dropout(dropout)
        self.add_module("dropout", self.dropout)

    def forward(self, X: Tensor) -> Tensor:
        return self._qkv(X, X, X)

    def _qkv(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        bs = Q.shape[0]
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)
        Q_ = torch.cat(Q.split(self.head_dim, -1), 0)
        K_ = torch.cat(K.split(self.head_dim, -1), 0)
        V_ = torch.cat(V.split(self.head_dim, -1), 0)
        H = torch.cat(scaled_dot_product_attention(Q_, K_, V_, exact=True).split(bs, 0), -1)
        return self.dropout(self.fc_o(H))

    def _get_name(self) -> str:
        return "_MHA"

class _AttentionBlock(nn.Module):
    in_shape: int
    head_dim: int
    num_heads: int
    dimshuffle: Rearrange
    ln: nn.LayerNorm
    mha: _MHA
    drop: nn.Dropout

    def __init__(self, in_shape: int = 32, head_dim: int = 8, num_heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.in_shape = in_shape
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dimshuffle = Rearrange("batch C T -> batch T C")
        self.add_module("dimshuffle", self.dimshuffle)
        self.ln = nn.LayerNorm(normalized_shape=in_shape, eps=1e-06)
        self.add_module("ln", self.ln)
        self.mha = _MHA(in_shape, head_dim, in_shape, num_heads, dropout)
        self.add_module("mha", self.mha)
        self.drop = nn.Dropout(0.3)
        self.add_module("drop", self.drop)

    def forward(self, X: Tensor) -> Tensor:
        X = self.dimshuffle(X)
        out = self.ln(X)
        out = self.mha._qkv(out, out, out)
        out = X + self.drop(out)
        return self.dimshuffle(out)

    def _get_name(self) -> str:
        return "_AttentionBlock"

class _TCNResidualBlock(nn.Module):
    activation: nn.Module
    dilation: int
    dropout: float
    n_filters: int
    kernel_size: int
    in_channels: int
    conv1: CausalConv1d
    bn1: nn.BatchNorm1d
    drop1: nn.Dropout
    conv2: CausalConv1d
    bn2: nn.BatchNorm1d
    drop2: nn.Dropout
    reshaping_conv: nn.Module

    def __init__(self, in_channels: int, kernel_size: int = 4, n_filters: int = 32, dropout: float = 0.3,
                 activation = nn.ELU, dilation: int = 1, conv_max_norm: Optional[float] = None):
        super().__init__()
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.dilation = dilation
        self.dropout = dropout
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.conv1 = CausalConv1d(in_channels, n_filters, kernel_size, dilation=dilation)
        self.add_module("conv1", self.conv1)
        nn.init.kaiming_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.add_module("bn1", self.bn1)
        self.drop1 = nn.Dropout(dropout)
        self.add_module("drop1", self.drop1)
        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_size, dilation=dilation)
        self.add_module("conv2", self.conv2)
        nn.init.kaiming_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.add_module("bn2", self.bn2)
        self.drop2 = nn.Dropout(dropout)
        self.add_module("drop2", self.drop2)
        rc: nn.Module = nn.Identity()
        if in_channels != n_filters:
            rc = nn.Conv1d(in_channels, n_filters, kernel_size=1, padding="same")
        self.reshaping_conv = rc
        self.add_module("reshaping_conv", rc)
        _register_max_norm(conv_max_norm, self.conv1, self.conv2, self.reshaping_conv)

    def forward(self, X: Tensor) -> Tensor:
        out = self.drop1(self.activation(self.bn1(self.conv1(X))))
        out = self.drop2(self.activation(self.bn2(self.conv2(out))))
        X = self.reshaping_conv(X)
        return self.activation(X + out)

    def _get_name(self) -> str:
        return "_TCNResidualBlock"

class ATCNet(EEGModuleMixin, nn.Module):
    """ATCNet from Altaheri et al (2022)."""
    conv_block_n_filters: int
    n_windows: int
    head_dim: int
    num_heads: int
    tcn_depth: int
    tcn_kernel_size: int
    concat: bool
    tcn_n_filters: int
    F2: int
    Tc: int
    Tw: int
    ensuredims: Ensure4d
    dimshuffle: Rearrange
    conv_block: _ConvBlock
    attention_blocks: nn.ModuleList
    temporal_conv_nets: nn.ModuleList
    final_layer: nn.ModuleList
    out_fun: nn.Identity

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 conv_block_n_filters: int = 16, conv_block_kernel_length_1: int = 64,
                 conv_block_kernel_length_2: int = 16, conv_block_pool_size_1: int = 8,
                 conv_block_pool_size_2: int = 7, conv_block_depth_mult: int = 2, conv_block_dropout: float = 0.3,
                 n_windows: int = 5, head_dim: int = 8, num_heads: int = 2, att_drop_prob: float = 0.5,
                 tcn_depth: int = 2, tcn_kernel_size: int = 4, tcn_drop_prob: float = 0.3, tcn_activation = nn.ELU,
                 concat: bool = False, max_norm_const: float = 0.25, conv_max_norm_const: Optional[float] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None):
        sf = sfreq
        if sf is None and (n_times is None or input_window_seconds is None):
            sf = 250.0
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sf)
        nn.Module.__init__(self)
        k1 = conv_block_kernel_length_1
        k2 = conv_block_kernel_length_2
        p1 = conv_block_pool_size_1
        p2 = conv_block_pool_size_2
        nw = n_windows
        tks = tcn_kernel_size
        min_len_tcn = (tks - 1) * (1 << (tcn_depth - 1)) + 1
        min_n = (nw + min_len_tcn - 1) * p1 * p2
        if self.n_times < min_n:
            s = self.n_times / min_n
            k1 = max(1, int(k1 * s))
            k2 = max(1, int(k2 * s))
            p1 = max(1, int(p1 * s))
            p2 = max(1, int(p2 * s))
            nw = max(1, int(nw * s))
            tks = max(2, int(tks * s))
        self.conv_block_n_filters = conv_block_n_filters
        self.n_windows = nw
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.tcn_depth = tcn_depth
        self.tcn_kernel_size = tks
        self.concat = concat
        self.tcn_n_filters = conv_block_depth_mult * conv_block_n_filters
        for w in range(nw):
            self.mapping[f"max_norm_linears.[{w}].weight"] = f"final_layer.[{w}].weight"
            self.mapping[f"max_norm_linears.[{w}].bias"] = f"final_layer.[{w}].bias"
        self.ensuredims = Ensure4d()
        self.add_module("ensuredims", self.ensuredims)
        self.dimshuffle = Rearrange("batch C T 1 -> batch 1 T C")
        self.add_module("dimshuffle", self.dimshuffle)
        self.conv_block = _ConvBlock(self.n_chans, conv_block_n_filters, k1, k2, p1, p2, conv_block_depth_mult,
                                     conv_block_dropout, conv_max_norm_const)
        self.add_module("conv_block", self.conv_block)
        self.F2 = conv_block_depth_mult * conv_block_n_filters
        self.Tc = int(self.n_times / (p1 * p2))
        self.Tw = self.Tc - nw + 1
        ab = List[nn.Module]()
        for _ in range(nw):
            ab.append(_AttentionBlock(self.F2, head_dim, num_heads, att_drop_prob))
        self.attention_blocks = nn.ModuleList(ab)
        self.add_module("attention_blocks", self.attention_blocks)
        tc = List[nn.Module]()
        for _ in range(nw):
            sq = nn.Sequential()
            for i in range(tcn_depth):
                sq.add_module(str(i), _TCNResidualBlock(self.F2 if i == 0 else self.tcn_n_filters, tks, self.tcn_n_filters,
                                                        tcn_drop_prob, tcn_activation, 1 << i, conv_max_norm_const))
            tc.append(sq)
        self.temporal_conv_nets = nn.ModuleList(tc)
        self.add_module("temporal_conv_nets", self.temporal_conv_nets)
        fl = List[nn.Module]()
        if concat:
            fl.append(MaxNormLinear(self.tcn_n_filters * nw, self.n_outputs, max_norm_val=max_norm_const))
        else:
            for _ in range(nw):
                fl.append(MaxNormLinear(self.tcn_n_filters, self.n_outputs, max_norm_val=max_norm_const))
        self.final_layer = nn.ModuleList(fl)
        self.add_module("final_layer", self.final_layer)
        self.out_fun = nn.Identity()
        self.add_module("out_fun", self.out_fun)

    def forward(self, X: Tensor) -> Tensor:
        X = self.dimshuffle(self.ensuredims(X))
        conv_feat = self.conv_block(X).view(-1, self.F2, self.Tc)
        att = self.attention_blocks.children()
        tcn = self.temporal_conv_nets.children()
        sw = List[Tensor]()
        for idx in range(len(att)):
            w = conv_feat[..., idx:idx + self.Tw]
            sw.append(tcn[idx](att[idx](w))[..., -1])
        fls = self.final_layer.children()
        if not self.concat:
            for idx in range(len(fls)):
                sw[idx] = fls[idx](sw[idx])
        agg = sw[0]
        if self.concat:
            agg = fls[0](torch.cat(sw, dim=1))
        elif len(sw) > 1:
            agg = torch.mean(torch.stack(sw, dim=0), dim=0)
        return self.out_fun(agg)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "ATCNet"
