# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from braindecode.functional import scaled_dot_product_attention
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import CausalConv1d

# braindecode builds one module and deepcopies it; here each copy is built
# from the same saved RNG state, which yields identical initial weights and
# leaves the global RNG where a single construction would.

class _SELayer(nn.Module):
    avg_pool: nn.AdaptiveAvgPool1d
    fc: nn.Sequential
    def __init__(self, channel: int, reduction: int = 16, activation = nn.ReLU):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.add_module("avg_pool", self.avg_pool)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), activation(),
                                nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
        self.add_module("fc", self.fc)
    def forward(self, x: Tensor) -> Tensor:
        b, c = x.shape[0], x.shape[1]
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)
    def _get_name(self) -> str:
        return "_SELayer"

class _SEBasicBlock(nn.Module):
    relu: nn.Module
    downsample: Optional[nn.Sequential]
    features: nn.Sequential
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Sequential] = None,
                 activation = nn.ReLU, reduction: int = 16):
        super().__init__()
        conv1 = nn.Conv1d(inplanes, planes, stride)
        bn1 = nn.BatchNorm1d(planes)
        self.relu = activation()
        conv2 = nn.Conv1d(planes, planes, 1)
        bn2 = nn.BatchNorm1d(planes)
        se = _SELayer(planes, reduction, activation=activation)
        self.add_module("conv1", conv1)
        self.add_module("bn1", bn1)
        self.add_module("relu", self.relu)
        self.add_module("conv2", conv2)
        self.add_module("bn2", bn2)
        self.add_module("se", se)
        self.downsample = downsample
        if downsample is not None:
            self.add_module("downsample", downsample.__val__())
        self.features = nn.Sequential(conv1, bn1, self.relu, conv2, bn2, se)
        self.add_module("features", self.features)
    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        res = x
        if self.downsample is not None:
            res = self.downsample.__val__()(x)
        return self.relu(out + res)
    def _get_name(self) -> str:
        return "_SEBasicBlock"

class _MRCNN(nn.Module):
    GELU: nn.Module
    features1: nn.Sequential
    features2: nn.Sequential
    dropout: nn.Dropout
    AFR: nn.Sequential
    def __init__(self, after_reduced_cnn_size: int, kernel_size: int = 7, activation = nn.GELU, activation_se = nn.ReLU):
        super().__init__()
        dr = 0.5
        g = activation()
        self.GELU = g
        self.add_module("GELU", g)
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24), nn.BatchNorm1d(64), g,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4), nn.Dropout(dr),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4), nn.BatchNorm1d(128), g,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4), nn.BatchNorm1d(128), g,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2))
        self.add_module("features1", self.features1)
        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200), nn.BatchNorm1d(64), g,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2), nn.Dropout(dr),
            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=1, bias=False, padding=3), nn.BatchNorm1d(128), g,
            nn.Conv1d(128, 128, kernel_size=kernel_size, stride=1, bias=False, padding=3), nn.BatchNorm1d(128), g,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.add_module("features2", self.features2)
        self.dropout = nn.Dropout(dr)
        self.add_module("dropout", self.dropout)
        ds: Optional[nn.Sequential] = None
        if 128 != after_reduced_cnn_size:
            ds = nn.Sequential(nn.Conv1d(128, after_reduced_cnn_size, kernel_size=1, stride=1, bias=False),
                               nn.BatchNorm1d(after_reduced_cnn_size))
        self.AFR = nn.Sequential(_SEBasicBlock(128, after_reduced_cnn_size, 1, ds, activation=activation_se))
        self.add_module("AFR", self.AFR)
    def forward(self, x: Tensor) -> Tensor:
        xc = torch.cat([self.features1(x), self.features2(x)], dim=2)
        return self.AFR(self.dropout(xc))
    def _get_name(self) -> str:
        return "_MRCNN"

class _MultiHeadedAttention(nn.Module):
    d_per_head: int
    h: int
    convs: nn.ModuleList
    _c: List[CausalConv1d]
    linear: nn.Linear
    dropout: nn.Dropout
    p: float
    def __init__(self, h: int, d_model: int, after_reduced_cnn_size: int, dropout: float = 0.1):
        super().__init__()
        if h <= 0 or d_model % h != 0:
            raise ValueError(f"n_attn_heads must be a positive integer that divides d_model, got n_attn_heads={h} and d_model={d_model}.")
        self.d_per_head = d_model // h
        self.h = h
        s0 = torch.get_rng_state()
        self._c = []
        s1 = s0
        for i in range(3):
            torch.set_rng_state(s0)
            self._c.append(CausalConv1d(after_reduced_cnn_size, after_reduced_cnn_size, kernel_size=7, stride=1))
            s1 = torch.get_rng_state()
        torch.set_rng_state(s1)
        self.convs = nn.ModuleList()
        for c in self._c:
            self.convs.append(c)
        self.add_module("convs", self.convs)
        self.linear = nn.Linear(d_model, d_model)
        self.add_module("linear", self.linear)
        self.dropout = nn.Dropout(p=dropout)
        self.add_module("dropout", self.dropout)
        self.p = dropout
    def _attend(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        nb = query.size(0)
        q = query.view(nb, -1, self.h, self.d_per_head).transpose(1, 2)
        k = self._c[1](key).view(nb, -1, self.h, self.d_per_head).transpose(1, 2)
        v = self._c[2](value).view(nb, -1, self.h, self.d_per_head).transpose(1, 2)
        x = scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0, exact=True)
        x = x.transpose(1, 2).contiguous().view(nb, -1, self.h * self.d_per_head)
        return self.linear(x)
    def forward(self, x: Tensor) -> Tensor:
        return self._attend(x, x, x)
    def _get_name(self) -> str:
        return "_MultiHeadedAttention"

class _PositionwiseFeedForward(nn.Module):
    w_1: nn.Linear
    w_2: nn.Linear
    dropout: nn.Dropout
    activate: nn.Module
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation = nn.ReLU):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.add_module("w_1", self.w_1)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.add_module("w_2", self.w_2)
        self.dropout = nn.Dropout(dropout)
        self.add_module("dropout", self.dropout)
        self.activate = activation()
        self.add_module("activate", self.activate)
    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.activate(self.w_1(x))))
    def _get_name(self) -> str:
        return "_PositionwiseFeedForward"

class _ResidualLayerNormAttn(nn.Module):
    norm: nn.LayerNorm
    dropout: nn.Dropout
    fn_attn: _MultiHeadedAttention
    def __init__(self, size: int, dropout: float, fn_attn: _MultiHeadedAttention):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-06)
        self.add_module("norm", self.norm)
        self.dropout = nn.Dropout(dropout)
        self.add_module("dropout", self.dropout)
        self.fn_attn = fn_attn
        self.add_module("fn_attn", fn_attn)
    def _fwd(self, x: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return x + self.dropout(self.fn_attn._attend(self.norm(x), key, value))
    def forward(self, x: Tensor) -> Tensor:
        return self._fwd(x, x, x)
    def _get_name(self) -> str:
        return "_ResidualLayerNormAttn"

class _ResidualLayerNormFF(nn.Module):
    norm: nn.LayerNorm
    dropout: nn.Dropout
    fn_ff: _PositionwiseFeedForward
    def __init__(self, size: int, dropout: float, fn_ff: _PositionwiseFeedForward):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-06)
        self.add_module("norm", self.norm)
        self.dropout = nn.Dropout(dropout)
        self.add_module("dropout", self.dropout)
        self.fn_ff = fn_ff
        self.add_module("fn_ff", fn_ff)
    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.fn_ff(self.norm(x)))
    def _get_name(self) -> str:
        return "_ResidualLayerNormFF"

class _EncoderLayer(nn.Module):
    size: int
    residual_self_attn: _ResidualLayerNormAttn
    residual_ff: _ResidualLayerNormFF
    conv: CausalConv1d
    def __init__(self, size: int, self_attn: _MultiHeadedAttention, feed_forward: _PositionwiseFeedForward,
                 after_reduced_cnn_size: int, dropout: float):
        super().__init__()
        self.size = size
        self.add_module("self_attn", self_attn)
        self.add_module("feed_forward", feed_forward)
        self.residual_self_attn = _ResidualLayerNormAttn(size, dropout, self_attn)
        self.add_module("residual_self_attn", self.residual_self_attn)
        self.residual_ff = _ResidualLayerNormFF(size, dropout, feed_forward)
        self.add_module("residual_ff", self.residual_ff)
        self.conv = CausalConv1d(after_reduced_cnn_size, after_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)
        self.add_module("conv", self.conv)
    def forward(self, x_in: Tensor) -> Tensor:
        q = self.conv(x_in)
        return self.residual_ff(self.residual_self_attn._fwd(q, x_in, x_in))
    def _get_name(self) -> str:
        return "_EncoderLayer"

class _TCE(nn.Module):
    layers: nn.ModuleList
    norm: nn.LayerNorm
    def __init__(self, layers: List[nn.Module], size: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.add_module("layers", self.layers)
        self.norm = nn.LayerNorm(size, eps=1e-06)
        self.add_module("norm", self.norm)
    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers.children():
            x = l(x)
        return self.norm(x)
    def _get_name(self) -> str:
        return "_TCE"

class AttnSleep(EEGModuleMixin, nn.Module):
    """AttnSleep from Eldele et al (2021)."""
    len_last_layer: int
    return_feats: bool
    feature_extractor: nn.Sequential
    final_layer: Optional[nn.Linear]

    def __init__(self, sfreq: Optional[float] = None, n_tce: int = 2, d_model: int = 80, d_ff: int = 120,
                 n_attn_heads: int = 5, drop_prob: float = 0.1, activation_mrcnn = nn.GELU, activation = nn.ReLU,
                 input_window_seconds: Optional[float] = None, n_outputs: Optional[int] = None,
                 after_reduced_cnn_size: int = 30, return_feats: bool = False,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_chans: Optional[int] = None,
                 n_times: Optional[int] = None):
        cnt = 0
        if n_times is not None:
            cnt += 1
        if sfreq is not None:
            cnt += 1
        if input_window_seconds is not None:
            cnt += 1
        if cnt < 2:
            raise ValueError("AttnSleep requires at least two of n_times, sfreq, and input_window_seconds.")
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        nc: Optional[int] = self._n_chans
        if nc is None and self._chs_info is not None:
            nc = len(self._chs_info.__val__())
        if nc is not None and nc.__val__() != 1:
            raise ValueError(f"AttnSleep takes single-channel input; got n_chans={nc}.")
        self.mapping["fc.weight"] = "final_layer.weight"
        self.mapping["fc.bias"] = "final_layer.bias"
        ks = 6 if self.sfreq == 125 else 7
        mrcnn = _MRCNN(after_reduced_cnn_size, ks, activation_mrcnn, activation)
        mods = mrcnn.modules()
        modes = [m.training for m in mods]
        mrcnn.eval()
        fl = 0
        with torch.no_grad():
            fl = mrcnn(torch.zeros(1, 1, self.n_times)).shape[-1]
        for i in range(len(mods)):
            mods[i].training = modes[i]
        if fl != d_model:
            raise ValueError(f"d_model is {d_model} but the feature extractor returns {fl} time steps for an input of {self.n_times} samples at {self.sfreq} Hz. Set d_model={fl}, with an n_attn_heads that divides it.")
        s0 = torch.get_rng_state()
        s1 = s0
        layers = List[nn.Module]()
        for i in range(n_tce):
            torch.set_rng_state(s0)
            attn = _MultiHeadedAttention(n_attn_heads, d_model, after_reduced_cnn_size)
            ff = _PositionwiseFeedForward(d_model, d_ff, drop_prob, activation=activation)
            layers.append(_EncoderLayer(d_model, attn, ff, after_reduced_cnn_size, drop_prob))
            s1 = torch.get_rng_state()
        torch.set_rng_state(s1)
        tce = _TCE(layers, d_model)
        self.feature_extractor = nn.Sequential(mrcnn, tce)
        self.add_module("feature_extractor", self.feature_extractor)
        self.len_last_layer = fl * after_reduced_cnn_size
        self.return_feats = return_feats
        self.final_layer = None
        if not return_feats:
            f = nn.Linear(self.len_last_layer, self.n_outputs)
            self.final_layer = f
            self.add_module("final_layer", f)

    def forward(self, x: Tensor) -> Tensor:
        e = self.feature_extractor(x)
        e = e.contiguous().view(e.shape[0], -1)
        if self.return_feats:
            return e
        return self.final_layer.__val__()(e)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "AttnSleep"
