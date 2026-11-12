# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import _next_fast_len

class _GConv(nn.Module):
    # SGConv. The FFT long convolution of the reference is computed as an
    # exact (Toeplitz) circular convolution with the same rolled kernel,
    # which is mathematically identical and avoids complex tensors.
    d_model: int
    channels: int
    bidirectional: bool
    transposed: bool
    linear: bool
    mode: str
    l_max: int
    init_scale: int
    kernel_dim: int
    num_scales: int
    multiplier: Tensor
    kernel_norm: Tensor
    kernel_norm_initialized: Tensor
    skip_weight: Tensor
    kernel_list: nn.ParameterList
    activation: Optional[nn.Module]
    dropout: Optional[nn.Module]
    norm: Optional[nn.Module]
    output_linear: Optional[nn.Linear]

    def __init__(self, d_model: int, d_state: int = 64, l_max: int = 1, channels: int = 1, bidirectional: bool = False,
                 activation = nn.GELU, dropout: float = 0.0, transposed: bool = True, linear: bool = False,
                 mode: str = "cat_randn", layer_norm: bool = False, init_scale: int = 0, kernel_dim: int = 2,
                 n_scales: Optional[int] = None, decay_min: float = 2.0, decay_max: float = 2.0):
        super().__init__()
        if "randn" not in mode:
            raise NotImplementedError(f"_GConv mode {mode} (only *randn modes are ported)")
        self.d_model = d_model
        self.channels = channels
        self.bidirectional = bidirectional
        self.transposed = transposed
        self.linear = linear
        self.mode = mode
        self.l_max = l_max
        self.init_scale = init_scale
        self.kernel_dim = kernel_dim
        ncc = channels * 2 if bidirectional else channels
        self.num_scales = n_scales.__val__() if n_scales is not None else \
            1 + int(math.ceil(math.log2(float(l_max) / float(kernel_dim)))) - init_scale
        self.multiplier = torch.linspace(decay_min, decay_max, d_model).view(1, -1, 1)
        self.register_buffer("multiplier", self.multiplier)
        self.kernel_norm = torch.ones(ncc, d_model, 1)
        self.register_buffer("kernel_norm", self.kernel_norm)
        self.kernel_norm_initialized = torch.tensor(False)
        self.register_buffer("kernel_norm_initialized", self.kernel_norm_initialized)
        self.skip_weight = nn.Parameter(torch.randn(channels, d_model))
        self.register_parameter("skip_weight", self.skip_weight)
        self.kernel_list = nn.ParameterList()
        for _ in range(self.num_scales):
            self.kernel_list.append(nn.Parameter(torch.randn(ncc, d_model, kernel_dim)))
        self.add_module("kernel_list", self.kernel_list)
        self.activation = None
        self.dropout = None
        self.norm = None
        self.output_linear = None
        if not linear:
            a = activation()
            self.activation = a
            self.add_module("activation", a)
            d: nn.Module = nn.Identity()
            if dropout > 0.0:
                d = nn.Dropout1d(dropout)
            self.dropout = d
            self.add_module("dropout", d)
            n: nn.Module = nn.Identity()
            if layer_norm:
                n = nn.LayerNorm(d_model * channels)
            self.norm = n
            self.add_module("norm", n)
            ol = nn.Linear(d_model * channels, d_model)
            self.output_linear = ol
            self.add_module("output_linear", ol)

    def forward(self, x: Tensor) -> Tensor:
        if not self.transposed:
            x = x.transpose(-1, -2)
        L = x.size(-1)
        im = "nearest" if "nearest" in self.mode else "linear"
        ks = List[Tensor]()
        for i in range(self.num_scales):
            sk = self.kernel_list[i]
            k = F.interpolate(sk, scale_factor=float(2 ** (max(0, i - 1) + self.init_scale)), mode=im)
            ks.append(k * self.multiplier ** float(self.num_scales - i - 1))
        kernel = torch.cat(ks, dim=-1)
        if not bool(self.kernel_norm_initialized.item()):
            with torch.no_grad():
                self.kernel_norm.copy_(torch.linalg.vector_norm(kernel.detach(), 2, dim=-1, keepdim=True))
                self.kernel_norm_initialized.fill_(True)
        if kernel.size(-1) > L:
            kernel = kernel[..., :L]
        elif kernel.size(-1) < L:
            kernel = F.pad(kernel, [0, L - kernel.size(-1)])
        kernel = kernel / self.kernel_norm
        if self.bidirectional:
            kk = kernel.unflatten(0, [2, -1]).unbind(0)
            kernel = F.pad(kk[0], [0, L]) + F.pad(kk[1].flip(-1), [L, 0])
        start = L if self.bidirectional else 0
        N = 2 * _next_fast_len(L)
        kr = kernel.roll(start, -1)
        if kr.size(-1) < N:
            kr = F.pad(kr, [0, N - kr.size(-1)])
        elif kr.size(-1) > N:
            kr = kr[..., :N]
        idx = List[int](capacity=L * L)
        for t in range(L):
            for s in range(L):
                idx.append((t + start - s) % N)
        K = kr.index_select(-1, torch.tensor(idx, dtype=torch.long)).unflatten(-1, [L, L])
        out = torch.matmul(K.unsqueeze(0), x.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
        out = out + x.unsqueeze(1) * self.skip_weight.unsqueeze(0).unsqueeze(-1)
        out = out.flatten(-3, -2)
        if not self.linear:
            out = self.dropout.__val__()(self.activation.__val__()(out)).transpose(1, 2)
            out = self.output_linear.__val__()(self.norm.__val__()(out)).transpose(1, 2)
        if not self.transposed:
            out = out.transpose(-1, -2)
        return out

    def _get_name(self) -> str:
        return "_GConv"

class _Conv(nn.Module):
    conv: nn.Conv1d
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        c = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=pad)
        weight_norm(c)
        nn.init.kaiming_normal_(c.get_parameter("parametrizations.weight.original1"))
        self.conv = c
        self.add_module("conv", c)
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
    def _get_name(self) -> str:
        return "_Conv"

class _ZeroConv1d(nn.Module):
    conv: nn.Conv1d
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.add_module("conv", self.conv)
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
    def _get_name(self) -> str:
        return "_ZeroConv1d"

class _RMSNorm(nn.Module):
    eps: float
    scale: Tensor
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, dim, 1))
        self.register_parameter("scale", self.scale)
    def forward(self, x: Tensor) -> Tensor:
        rms = torch.linalg.vector_norm(x, 2, dim=1, keepdim=True) / (float(x.shape[1]) ** 0.5)
        return self.scale * (x / (rms + self.eps))
    def _get_name(self) -> str:
        return "_RMSNorm"

def _wn_conv1x1(i: int, o: int) -> nn.Conv1d:
    c = nn.Conv1d(i, o, kernel_size=1)
    weight_norm(c)
    nn.init.kaiming_normal_(c.get_parameter("parametrizations.weight.original1"))
    return c

class _ResidualBlock(nn.Module):
    res_channels: int
    swa_window_size: int
    rms_norm: _RMSNorm
    conv_layer: _Conv
    gelu: nn.GELU
    sgconv: _GConv
    attention: nn.MultiheadAttention
    res_conv: nn.Conv1d
    skip_conv: nn.Conv1d

    def __init__(self, res_channels: int, skip_channels: int, s4_lmax: int, s4_d_state: int, s4_dropout: float,
                 s4_bidirectional: bool, s4_layernorm: bool, swa_window_size: int = 1):
        super().__init__()
        self.res_channels = res_channels
        self.swa_window_size = swa_window_size
        self.rms_norm = _RMSNorm(res_channels)
        self.add_module("rms_norm", self.rms_norm)
        self.conv_layer = _Conv(res_channels, 2 * res_channels, kernel_size=3)
        self.add_module("conv_layer", self.conv_layer)
        self.gelu = nn.GELU()
        self.add_module("gelu", self.gelu)
        self.sgconv = _GConv(2 * res_channels, s4_d_state, s4_lmax, 4, s4_bidirectional, dropout=s4_dropout,
                             layer_norm=s4_layernorm)
        self.add_module("sgconv", self.sgconv)
        self.attention = nn.MultiheadAttention(2 * res_channels, 4, dropout=s4_dropout, bias=True, batch_first=True)
        self.add_module("attention", self.attention)
        self.res_conv = _wn_conv1x1(res_channels, res_channels)
        self.add_module("res_conv", self.res_conv)
        self.skip_conv = _wn_conv1x1(res_channels, skip_channels)
        self.add_module("skip_conv", self.skip_conv)

    def _swa_mask(self, L: int, dtype) -> Tensor:
        if self.swa_window_size % 2 != 1:
            raise ValueError(f"swa_window_size must be odd, got {self.swa_window_size}.")
        h = self.swa_window_size // 2
        vals = [0.0 if abs(i - j) <= h else -math.inf for i in range(L) for j in range(L)]
        return torch.tensor(vals, dtype=dtype).reshape(L, L)

    def block(self, x: Tensor, original: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = x
        L = x.shape[2]
        x = self.rms_norm(x)
        hidden = self.gelu(self.conv_layer(hidden + original))
        h_ssm = self.sgconv(hidden)
        ha = h_ssm.transpose(1, 2)
        if self.swa_window_size == 1:
            # window 1: each token attends only to itself, so attention = V
            E = self.attention.embed_dim
            sf = ha.transpose(0, 1)
            w = self.attention.in_proj_weight.__val__()
            b = self.attention.in_proj_bias.__val__()
            value = F.linear(sf, w[2 * E:], b[2 * E:])
            p = self.attention.dropout
            if self.training and p > 0.0:
                nr = sf.shape[1] * self.attention.num_heads
                keep = (torch.rand(L, nr) >= p).to(value.dtype) / (1.0 - p)
                value = value.reshape(L, nr, -1) * keep.unsqueeze(-1)
            ha = self.attention.out_proj(value.reshape(-1, E)).view(sf.shape).transpose(0, 1)
        else:
            ha, _ = self.attention.attend(ha, ha, ha, attn_mask=self._swa_mask(L, x.dtype), need_weights=False)
        comb = h_ssm + ha.transpose(1, 2)
        R = self.res_channels
        out = torch.tanh(comb[:, :R, :]) * torch.sigmoid(comb[:, R:, :])
        return (x + self.res_conv(out)) * math.sqrt(0.5), self.skip_conv(out)

    def _get_name(self) -> str:
        return "_ResidualBlock"

class _ResidualGroup(nn.Module):
    num_res_layers: int
    residual_blocks: nn.ModuleList
    _blocks: List[_ResidualBlock]
    def __init__(self, res_channels: int, skip_channels: int, num_res_layers: int, s4_lmax: int, s4_d_state: int,
                 s4_dropout: float, s4_bidirectional: bool, s4_layernorm: bool, swa_window_size: int = 1):
        super().__init__()
        self.num_res_layers = num_res_layers
        self.residual_blocks = nn.ModuleList()
        self._blocks = []
        for _ in range(num_res_layers):
            b = _ResidualBlock(res_channels, skip_channels, s4_lmax, s4_d_state, s4_dropout, s4_bidirectional,
                               s4_layernorm, swa_window_size)
            self._blocks.append(b)
            self.residual_blocks.append(b)
        self.add_module("residual_blocks", self.residual_blocks)
    def forward(self, x: Tensor) -> Tensor:
        h = x
        skip = torch.zeros(1, dtype=x.dtype).squeeze(0)
        for b in self._blocks:
            h, s = b.block(h, x)
            skip = s + skip
        return skip * math.sqrt(1.0 / float(self.num_res_layers))
    def _get_name(self) -> str:
        return "_ResidualGroup"

class _CodeBrainPatchEmbedding(nn.Module):
    emb_dim: int
    d_model: int
    mask_encoding: Tensor
    proj_in: nn.Sequential
    spectral_proj: nn.Sequential
    positional_encoding: nn.Sequential

    def __init__(self, conv_out_chans: int, patch_size: int, conv_groups: int, proj_kernel_size: int = 49,
                 proj_padding: int = 24, proj_refine_kernel: int = 3, pos_kernel: Tuple[int, int] = (19, 7),
                 spectral_dropout: float = 0.1):
        super().__init__()
        rp = proj_refine_kernel // 2
        tl = (patch_size + 2 * proj_padding - proj_kernel_size) // conv_out_chans + 1
        self.emb_dim = conv_out_chans * tl
        self.d_model = self.emb_dim
        E = self.emb_dim
        self.mask_encoding = nn.Parameter(torch.zeros(E), requires_grad=False)
        self.register_parameter("mask_encoding", self.mask_encoding)
        c = conv_out_chans
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=(1, proj_kernel_size), stride=(1, c), padding=(0, proj_padding)),
            nn.GroupNorm(conv_groups, c), nn.GELU(),
            nn.Conv2d(c, c, kernel_size=(1, proj_refine_kernel), stride=(1, 1), padding=(0, rp)),
            nn.GroupNorm(conv_groups, c), nn.GELU(),
            nn.Conv2d(c, c, kernel_size=(1, proj_refine_kernel), stride=(1, 1), padding=(0, rp)),
            nn.GroupNorm(conv_groups, c), nn.GELU())
        self.add_module("proj_in", self.proj_in)
        self.spectral_proj = nn.Sequential(nn.Linear(patch_size // 2 + 1, E), nn.Dropout(spectral_dropout))
        self.add_module("spectral_proj", self.spectral_proj)
        self.positional_encoding = nn.Sequential(nn.Conv2d(E, E, kernel_size=pos_kernel, stride=(1, 1),
                                                           padding=(pos_kernel[0] // 2, pos_kernel[1] // 2), groups=E))
        self.add_module("positional_encoding", self.positional_encoding)

    def forward(self, x: Tensor) -> Tensor:
        b, c, s, p = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        pe = self.proj_in(x.contiguous().view(b, 1, c * s, p))
        pe = pe.unflatten(2, [c, s]).permute(0, 2, 3, 1, 4).flatten(3)
        flat = x.flatten(0, 2)
        spec = torch.abs(torch.fft.rfft(flat, dim=-1, norm="forward")).to(flat.dtype).unflatten(0, [b, c, s])
        pe = pe + self.spectral_proj(spec)
        return pe + self.positional_encoding(pe.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def _get_name(self) -> str:
        return "_PatchEmbedding"

class CodeBrain(EEGModuleMixin, nn.Module):
    """CodeBrain from Ma et al (2025). Architecture only (no Hub weights).

    The SGConv FFT convolution is computed as an equivalent dense circular
    convolution (O(L^2) per layer, exact up to float rounding). Pre-training
    heads (lm_head_t/f) exist for state-dict parity; ``forward_pretrain``
    returns their logits."""
    patch_size: int
    out_channels: int
    pretrain_mode: bool
    patch_embedding: _CodeBrainPatchEmbedding
    init_conv: nn.Sequential
    residual_layer: _ResidualGroup
    final_conv: nn.Sequential
    norm: nn.LayerNorm
    lm_head_t: nn.Linear
    lm_head_f: nn.Linear
    final_layer: nn.Sequential

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, patch_size: int = 200,
                 res_channels: int = 200, skip_channels: int = 200, out_channels: int = 200, num_res_layers: int = 8,
                 drop_prob: float = 0.1, s4_bidirectional: bool = True, s4_layernorm: bool = False,
                 s4_lmax: int = 570, s4_d_state: int = 64, conv_out_chans: int = 25, conv_groups: int = 5,
                 proj_kernel_size: int = 49, proj_padding: int = 24, proj_refine_kernel: int = 3,
                 pos_kernel: Tuple[int, int] = (19, 7), spectral_dropout: float = 0.1, mlp_hidden_multiplier: int = 4,
                 swa_window_size: int = 1, codebook_size_t: int = 4096, codebook_size_f: int = 4096,
                 pretrain_mode: bool = False, activation = nn.ReLU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.pretrain_mode = pretrain_mode
        self.patch_embedding = _CodeBrainPatchEmbedding(conv_out_chans, patch_size, conv_groups, proj_kernel_size,
                                                        proj_padding, proj_refine_kernel, pos_kernel, spectral_dropout)
        self.add_module("patch_embedding", self.patch_embedding)
        E = self.patch_embedding.emb_dim
        self.init_conv = nn.Sequential(_Conv(E, res_channels, kernel_size=1), activation())
        self.add_module("init_conv", self.init_conv)
        self.residual_layer = _ResidualGroup(res_channels, skip_channels, num_res_layers, s4_lmax, s4_d_state,
                                             drop_prob, s4_bidirectional, s4_layernorm, swa_window_size)
        self.add_module("residual_layer", self.residual_layer)
        self.final_conv = nn.Sequential(_Conv(skip_channels, skip_channels, kernel_size=1), activation(),
                                        _ZeroConv1d(skip_channels, out_channels))
        self.add_module("final_conv", self.final_conv)
        self.norm = nn.LayerNorm(out_channels)
        self.add_module("norm", self.norm)
        self.lm_head_t = nn.Linear(out_channels, codebook_size_t, bias=False)
        self.add_module("lm_head_t", self.lm_head_t)
        self.lm_head_f = nn.Linear(out_channels, codebook_size_f, bias=False)
        self.add_module("lm_head_f", self.lm_head_f)
        flat = self.n_chans * (self.n_times // patch_size) * out_channels
        h = mlp_hidden_multiplier * out_channels
        self.final_layer = nn.Sequential(nn.Flatten(), nn.Linear(flat, h), nn.ELU(), nn.Dropout(drop_prob),
                                         nn.Linear(h, out_channels), nn.ELU(), nn.Dropout(drop_prob),
                                         nn.Linear(out_channels, self.n_outputs))
        self.add_module("final_layer", self.final_layer)

    def forward_features(self, x: Tensor) -> Tensor:
        B, C, T = x.shape[0], x.shape[1], x.shape[2]
        S = T // self.patch_size
        x = x[:, :, :S * self.patch_size].reshape(B, C, S, self.patch_size)
        x = self.patch_embedding(x).permute(0, 3, 1, 2).flatten(2)
        x = self.final_conv(self.residual_layer(self.init_conv(x)))
        return self.norm(x.unflatten(2, [C, S]).permute(0, 2, 3, 1))

    def forward_pretrain(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        f = self.forward_features(x)
        return self.lm_head_t(f), self.lm_head_f(f)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x))

    def _get_name(self) -> str:
        return "CodeBrain"
