# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from braindecode.models.base import EEGModuleMixin

def _renorm_rows(w: Tensor, maxnorm: float) -> Tensor:
    """torch.renorm(w, p=2, dim=0, maxnorm)."""
    norms = w.detach().double().pow(2).sum(dim=1, keepdim=True).sqrt()
    scale = torch.where(norms > maxnorm, maxnorm / (norms + 1e-7), torch.ones_like(norms))
    return w * scale.float()

class _DepthwiseTemporalConv1d(nn.Module):
    in_channels: int
    kernel_size: int
    stride: int
    num_heads: int
    padding: str
    weight_softmax: bool
    weight: Tensor
    bias: Optional[Tensor]
    def __init__(self, in_channels: int, num_heads: int = 1, n_spectral_filters_temporal: int = 1,
                 kernel_size: int = 1, stride: int = 1, padding: str = "same", bias: bool = True,
                 weight_softmax: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.empty(num_heads * n_spectral_filters_temporal, 1, kernel_size))
        self.register_parameter("weight", self.weight)
        if bias:
            b = nn.Parameter(torch.empty(num_heads * n_spectral_filters_temporal))
            self.bias = b
            self.register_parameter("bias", b)
        else:
            self.bias = None
            self.register_parameter("bias", None)
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.constant_(self.bias.__val__(), 0.0)
    def forward(self, inp: Tensor) -> Tensor:
        B, C, T = inp.shape[0], inp.shape[1], inp.shape[2]
        H = self.num_heads
        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)
        inp = inp.reshape(B, H, C // H, T).permute(0, 2, 1, 3).reshape(-1, H, T)
        output = F.conv1d(inp, weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.num_heads)
        h, t = output.shape[1], output.shape[2]
        return output.reshape(B, -1, h, t).permute(0, 2, 1, 3).reshape(B, -1, t)
    def _get_name(self) -> str:
        return "_DepthwiseTemporalConv1d"

def _var_pool(x: Tensor, k: int, s: int, p: int) -> Tensor:
    ms = F.avg_pool1d(x ** 2, kernel_size=k, stride=s, padding=p)
    sm = F.avg_pool1d(x, kernel_size=k, stride=s, padding=p) ** 2
    return ms - sm

class _GlobalContextVarPool1D(nn.Module):
    kernel_size: int
    stride: int
    padding: int
    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
    def forward(self, x: Tensor) -> Tensor:
        v = _var_pool(x, self.kernel_size, self.stride, self.padding)
        return F.avg_pool1d(v, v.shape[-1])
    def _get_name(self) -> str:
        return "_GlobalContextVarPool1D"

class _VariancePool1D(nn.Module):
    kernel_size: int
    stride: int
    padding: int
    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
    def forward(self, x: Tensor) -> Tensor:
        return _var_pool(x, self.kernel_size, self.stride, self.padding)
    def _get_name(self) -> str:
        return "_VariancePool1D"

class _SpatSpectralAttn(nn.Module):
    alpha: Tensor
    gamma: Tensor
    beta: Tensor
    epsilon: float
    mode: str
    after_relu: bool
    global_ctx: _GlobalContextVarPool1D
    def __init__(self, T: int, num_channels: int, epsilon: float = 1e-05, mode: str = "var",
                 after_relu: bool = False, global_context_kernel: int = 250):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.register_parameter("alpha", self.alpha)
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.register_parameter("beta", self.beta)
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu
        if mode not in ("var", "l2", "l1"):
            raise ValueError(f"Unsupported Spatial-Spectral Attention mode: {mode}")
        self.global_ctx = _GlobalContextVarPool1D(global_context_kernel)
        self.add_module("global_ctx", self.global_ctx)
    def gate(self, x: Tensor) -> Tensor:
        embedding = x
        norm = x
        if self.mode == "l2":
            embedding = (x.pow(2).sum(2, keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == "l1":
            xx = x if self.after_relu else torch.abs(x)
            embedding = xx.sum(2, keepdim=True)
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            embedding = (self.global_ctx(x) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        return 1 + torch.tanh(embedding * norm + self.beta)
    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x)
    def _get_name(self) -> str:
        return "_SpatSpectralAttn"

class _MultiScaleVarPooler(nn.Module):
    var_layers: nn.ModuleList
    _vl: List[nn.Sequential]
    num_scales: int
    def __init__(self, kernel_sizes: List[int]):
        super().__init__()
        self.var_layers = nn.ModuleList()
        self._vl = []
        self.num_scales = len(kernel_sizes)
        for k in kernel_sizes:
            s = nn.Sequential(_VariancePool1D(kernel_size=k, stride=int(k / 2)), nn.Flatten(start_dim=1))
            self._vl.append(s)
            self.var_layers.append(s)
        self.add_module("var_layers", self.var_layers)
    def forward(self, x: Tensor) -> Tensor:
        c = x.shape[1]
        assert c % self.num_scales == 0
        xs = torch.split(x, c // self.num_scales, dim=1)
        return torch.cat([self._vl[i](xs[i]) for i in range(self.num_scales)], dim=1)
    def _get_name(self) -> str:
        return "_MultiScaleVarPooler"

class _SSTEncoder(nn.Module):
    temporal_conv: _DepthwiseTemporalConv1d
    spt_attn: _SpatSpectralAttn
    chan_conv: nn.Sequential
    mvp: _MultiScaleVarPooler
    def __init__(self, n_times: int, n_chans: int, n_spectral_filters_temporal: int, n_fused_filters: int,
                 temporal_conv_kernel_size: int, mvp_kernel_sizes: List[int], spt_attn_global_context_kernel: int,
                 spt_attn_epsilon: float, spt_attn_mode: str, activation):
        super().__init__()
        self.temporal_conv = _DepthwiseTemporalConv1d(n_chans, 1, n_spectral_filters_temporal,
                                                      temporal_conv_kernel_size, 1, "same", True, False)
        self.add_module("temporal_conv", self.temporal_conv)
        self.spt_attn = _SpatSpectralAttn(n_times, n_chans * n_spectral_filters_temporal, spt_attn_epsilon,
                                          spt_attn_mode, False, spt_attn_global_context_kernel)
        self.add_module("spt_attn", self.spt_attn)
        self.chan_conv = nn.Sequential(
            nn.Conv1d(n_chans * n_spectral_filters_temporal, n_fused_filters, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(n_fused_filters), activation())
        self.add_module("chan_conv", self.chan_conv)
        self.mvp = _MultiScaleVarPooler(mvp_kernel_sizes)
        self.add_module("mvp", self.mvp)
    def forward(self, x: Tensor) -> Tensor:
        x = self.temporal_conv(x)
        x = self.spt_attn(x)
        return self.mvp(self.chan_conv(x))
    def _get_name(self) -> str:
        return "_SSTEncoder"

class SSTDPN(EEGModuleMixin, nn.Module):
    """SSTDPN from Han et al (2025). forward returns logits (return_features
    is not supported in Typhon; call `encoder` for the features)."""
    n_spectral_filters_temporal: int
    n_fused_filters: int
    mvp_kernel_sizes: List[int]
    proto_sep_maxnorm: float
    proto_cpt_std: float
    encoder: _SSTEncoder
    proto_sep: Tensor
    proto_cpt: Tensor
    final_layer: nn.Identity

    def __init__(self, n_chans: Optional[int] = None, n_times: Optional[int] = None, n_outputs: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_spectral_filters_temporal: int = 9,
                 n_fused_filters: int = 48, temporal_conv_kernel_size: int = 75,
                 mvp_kernel_sizes: Optional[List[int]] = None, return_features: bool = False,
                 proto_sep_maxnorm: float = 1.0, proto_cpt_std: float = 1.0,
                 spt_attn_global_context_kernel: int = 250, spt_attn_epsilon: float = 1e-05,
                 spt_attn_mode: str = "var", activation = nn.ELU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if return_features:
            raise NotImplementedError("SSTDPN: return_features=True is not supported in Typhon")
        self.n_spectral_filters_temporal = n_spectral_filters_temporal
        self.n_fused_filters = n_fused_filters
        self.mvp_kernel_sizes = mvp_kernel_sizes if mvp_kernel_sizes is not None else [50, 100, 200]
        self.proto_sep_maxnorm = proto_sep_maxnorm
        self.proto_cpt_std = proto_cpt_std
        self.encoder = _SSTEncoder(self.n_times, self.n_chans, n_spectral_filters_temporal, n_fused_filters,
                                   temporal_conv_kernel_size, self.mvp_kernel_sizes, spt_attn_global_context_kernel,
                                   spt_attn_epsilon, spt_attn_mode, activation)
        self.add_module("encoder", self.encoder)
        ns = len(self.mvp_kernel_sizes)
        if ns == 0:
            raise ValueError("`mvp_kernel_sizes` must contain at least one kernel size.")
        if n_fused_filters % ns != 0:
            raise ValueError("Number of fused filters must be divisible by the number of MVP scales.")
        pooled = 0
        for k in self.mvp_kernel_sizes:
            if k // 2 == 0:
                raise ValueError(f"MVP kernel sizes too small to derive a valid stride (k//2 == 0): {k}")
            pooled += max(0, (self.n_times - k) // (k // 2) + 1)
        feat_dim = (n_fused_filters // ns) * pooled
        self.proto_sep = nn.Parameter(torch.empty(self.n_outputs, feat_dim))
        self.register_parameter("proto_sep", self.proto_sep)
        self.proto_cpt = nn.Parameter(torch.empty(self.n_outputs, feat_dim))
        self.register_parameter("proto_cpt", self.proto_cpt)
        self.final_layer = nn.Identity()
        self.add_module("final_layer", self.final_layer)
        init.kaiming_normal_(self.proto_sep)
        init.normal_(self.proto_cpt, mean=0.0, std=proto_cpt_std)

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        with torch.no_grad():
            self.proto_sep.copy_(_renorm_rows(self.proto_sep, self.proto_sep_maxnorm))
        logits = torch.matmul(features, self.proto_sep.t())
        return self.final_layer(logits)

    def _get_name(self) -> str:
        return "SSTDPN"
