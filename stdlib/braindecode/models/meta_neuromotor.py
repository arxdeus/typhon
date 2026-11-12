# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# MetaNeuromotorHand (generic neuromotor interface, handwriting model).
# Typhon notes: the multivariate power-frequency featurizer (STFT, cross
# spectral density, band averaging, SPD matrix log) has no trainable
# parameters and is computed in float64 without autograd, using the
# featurizer buffers (window, freq_masks); torch computes it in complex64 /
# float32, so tiny eigenvalues may differ in the last bits.

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin

# ------------------------------------------------------------ featurizer

class _ChannelwiseSTFT(nn.Module):
    n_fft: int
    hop_length: int
    window: Tensor
    window_norm: Tensor
    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft, periodic=False)
        self.register_buffer("window", self.window)
        self.window_norm = self.window.double().pow(2).sum().sqrt().float()
        self.register_buffer("window_norm", self.window_norm)
    def stft(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Real and imaginary parts (batch, chans, freqs, frames), float64,
        normalized by the window L2 norm (torchaudio normalized=True)."""
        n = self.n_fft
        nf = n // 2 + 1
        w = self.window.detach().double()
        wn = w.pow(2).sum().sqrt()
        cv = List[float](capacity=nf * n)
        sv = List[float](capacity=nf * n)
        for f in range(nf):
            for k in range(n):
                a = 2 * math.pi * float((f * k) % n) / n
                cv.append(math.cos(a))
                sv.append(-math.sin(a))
        cb = torch.tensor(cv, dtype=torch.float64).reshape(nf, n) * w
        sb = torch.tensor(sv, dtype=torch.float64).reshape(nf, n) * w
        frames = x.detach().double().unfold(-1, n, self.hop_length)
        re = torch.matmul(frames, cb.t()) / wn
        im = torch.matmul(frames, sb.t()) / wn
        return re.transpose(-1, -2), im.transpose(-1, -2)
    def _get_name(self) -> str:
        return "_ChannelwiseSTFT"

class _FrequencyBandAverager(nn.Module):
    has_bins: bool
    n_output_freqs: int
    freq_masks: Tensor
    def __init__(self, n_fft: int, fs: float, frequency_bins: Optional[List[Tuple[float, float]]]):
        super().__init__()
        self.n_output_freqs = n_fft // 2 + 1
        self.has_bins = frequency_bins is not None
        self.freq_masks = torch.zeros(0)
        if frequency_bins is not None:
            bins = frequency_bins.__val__()
            freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs)
            ms = [((freqs > lo) & (freqs <= hi)).float() for lo, hi in bins]
            fm = torch.stack(ms)
            if (fm.sum(dim=1) == 0).any().item():
                raise ValueError("Each frequency band must contain at least one FFT bin")
            self.freq_masks = fm.reshape(1, 1, len(bins), fm.shape[1], 1, 1)
            self.register_buffer("freq_masks", self.freq_masks)
            self.n_output_freqs = len(bins)
    def average(self, x: Tensor) -> Tensor:
        if not self.has_bins:
            return x
        fm = self.freq_masks.detach().double()
        m = fm[0, 0, :, :, 0, 0]  # n f
        b, t, f, i, j = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        s = torch.matmul(m, x.permute(0, 1, 3, 4, 2).unsqueeze(-1)).squeeze(-1)  # b t i j n
        s = s.permute(0, 1, 4, 2, 3)
        return s / fm.sum(dim=3)
    def _get_name(self) -> str:
        return "_FrequencyBandAverager"

class _MultivariatePowerFrequencyFeatures(nn.Module):
    window_length: int
    stride: int
    n_fft: int
    fft_stride: int
    fs: float
    left_context: int
    stft: _ChannelwiseSTFT
    band_averager: _FrequencyBandAverager
    def __init__(self, window_length: int, stride: int, n_fft: int, fft_stride: int, fs: float,
                 frequency_bins: Optional[List[Tuple[float, float]]]):
        super().__init__()
        if window_length < n_fft:
            raise ValueError("window_length must be greater than n_fft")
        if fft_stride > n_fft:
            raise ValueError("fft_stride must be lower than n_fft")
        if fft_stride > stride:
            raise ValueError("stride must be greater than fft_stride")
        if stride % fft_stride != 0:
            raise ValueError("stride must be a multiple of fft_stride")
        self.window_length = window_length
        self.stride = stride
        self.n_fft = n_fft
        self.fft_stride = fft_stride
        self.fs = fs
        self.stft = _ChannelwiseSTFT(n_fft, fft_stride)
        self.add_module("stft", self.stft)
        self.band_averager = _FrequencyBandAverager(n_fft, fs, frequency_bins)
        self.add_module("band_averager", self.band_averager)
        self.left_context = window_length - fft_stride + n_fft - 1
    def forward(self, inputs: Tensor) -> Tensor:
        re, im = self.stft.stft(inputs)  # b c f T
        W = self.window_length // self.fft_stride
        S = self.stride // self.fft_stride
        re = re.unfold(3, W, S).permute(0, 3, 2, 1, 4)  # b t f c W
        im = im.unfold(3, W, S).permute(0, 3, 2, 1, 4)
        ret = re.transpose(-1, -2)
        imt = im.transpose(-1, -2)
        cr = (torch.matmul(re, ret) + torch.matmul(im, imt)) / W
        ci = (torch.matmul(im, ret) - torch.matmul(re, imt)) / W
        x = cr * cr + ci * ci
        x = self.band_averager.average(x)
        L, V = torch.linalg.eigh(x)
        L = L.log().nan_to_num(nan=0.0, neginf=0.0)
        x = torch.matmul(V * L.unsqueeze(-2), V.transpose(-1, -2))
        return x.permute(0, 2, 3, 4, 1).to(inputs.dtype)
    def compute_time_downsampling(self, input_lengths: Tensor) -> Tensor:
        cl = 1 + torch.div(input_lengths - self.n_fft, self.fft_stride, rounding_mode="floor")
        return torch.div(cl - self.window_length // self.fft_stride, self.stride // self.fft_stride,
                         rounding_mode="floor") + 1
    def _get_name(self) -> str:
        return "_MultivariatePowerFrequencyFeatures"

def _kept_indices(num_channels: int, num_adjacent_cov: int) -> List[int]:
    if num_adjacent_cov > num_channels // 2:
        raise ValueError(f"num_adjacent_cov={num_adjacent_cov} must be <= {num_channels // 2}")
    out = List[int]()
    for r in range(num_channels):
        for c in range(r, num_channels):
            d = (r - c) % num_channels
            if d <= num_adjacent_cov or d >= num_channels - num_adjacent_cov:
                out.append(num_channels * r + c)
    return out

class _RotationInvariantMPFMLP(nn.Module):
    offsets: List[int]
    activation: nn.Module
    rotation_index: Tensor
    fully_connected_layers: nn.ModuleList
    _fc: List[nn.Linear]
    def __init__(self, num_channels: int, num_freqs: int, hidden_dims: List[int], offsets: List[int],
                 num_adjacent_cov: int, activation):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be non-empty")
        self.offsets = offsets
        self.activation = activation()
        self.add_module("activation", self.activation)
        kept = _kept_indices(num_channels, num_adjacent_cov)
        idx = List[int]()
        C = num_channels
        for off in offsets:
            for f in range(num_freqs):
                for k in kept:
                    r, c = k // C, k % C
                    idx.append(f * C * C + ((r - off) % C) * C + (c - off) % C)
        self.rotation_index = torch.tensor(idx, dtype=torch.int64)
        self.register_buffer("rotation_index", self.rotation_index, persistent=False)
        self._fc = []
        self.fully_connected_layers = nn.ModuleList()
        d = num_freqs * min(C * (num_adjacent_cov + 1), C * (C + 1) // 2)
        for h in hidden_dims:
            l = nn.Linear(d, h)
            self._fc.append(l)
            self.fully_connected_layers.append(l)
            d = h
        self.add_module("fully_connected_layers", self.fully_connected_layers)
    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.permute(0, 4, 1, 2, 3).flatten(2)
        x = x.index_select(-1, self.rotation_index)
        x = x.unflatten(-1, (len(self.offsets), -1))
        for l in self._fc:
            x = self.activation(l(x))
        return x.mean(dim=2).transpose(1, 2)
    def _get_name(self) -> str:
        return "_RotationInvariantMPFMLP"

class _MaskAug(nn.Module):
    """SpecAugment-style masking on (batch, freqs, chans, chans, time); training only."""
    _plans: List[Tuple[int, int, List[int]]]
    mask_value: float
    def __init__(self, max_num_masks: List[int], max_mask_lengths: List[int], dims: str = "TF", mask_value: float = 0.0):
        super().__init__()
        if len(max_num_masks) != len(dims):
            raise ValueError("max_num_masks length must match dims length")
        if len(max_mask_lengths) != len(dims):
            raise ValueError("max_mask_lengths length must match dims length")
        self._plans = []
        for d in "CFT":
            if d not in dims:
                continue
            axes = [2, 3] if d == "C" else ([1] if d == "F" else [4])
            i = dims.index(d)
            self._plans.append((max_num_masks[i], max_mask_lengths[i], axes))
        self.mask_value = mask_value
    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training:
            return inputs
        x = inputs
        b = x.size(0)
        for mx, ml, axes in self._plans:
            nm = int(torch.randint(0, mx + 1, List[int]()).item())
            for _ in range(nm):
                L = x.size(axes[0])
                eff = min(ml, L)
                mlen = torch.randint(0, eff + 1, [b])
                sr = (L - mlen).float()
                ms = (torch.rand(b) * sr).long()
                me = ms + mlen
                for _ in range(x.ndim - 1):
                    ms = ms.unsqueeze(-1)
                    me = me.unsqueeze(-1)
                idx = torch.arange(0, L)
                mask = (idx >= ms) & (idx < me)
                for ax in axes:
                    if x.size(ax) != L:
                        raise ValueError("All axes to mask must have the same length")
                    x = x.transpose(ax, -1).masked_fill(mask, self.mask_value).transpose(-1, ax)
        return x
    def _get_name(self) -> str:
        return "_MaskAug"

# ------------------------------------------------------------ conformer

class _Window(nn.Module):
    receptive_field: int
    stride: int
    state_size: int
    lpad: int
    extra_left_context: int
    dilation: int
    kernel_size: int
    def __init__(self, kernel_size: int, stride: int = 1, dilation: int = 1, lpad: str = "none"):
        super().__init__()
        self.receptive_field = 1 + dilation * (kernel_size - 1)
        self.stride = stride
        self.state_size = self.receptive_field - stride
        if self.receptive_field < stride:
            raise ValueError(f"receptive_field={self.receptive_field} < stride={stride} is not supported")
        if lpad == "none":
            self.lpad = 0
        elif lpad == "steady":
            self.lpad = self.state_size
        elif lpad == "full":
            self.lpad = self.receptive_field - 1
        else:
            raise ValueError(f"Invalid lpad={lpad!r}")
        self.extra_left_context = self.receptive_field - 1 - self.lpad
        self.dilation = dilation
        self.kernel_size = kernel_size
    def forward(self, inputs: Tensor) -> Tensor:
        if self.state_size > 0:
            inputs = F.pad(inputs, [0, 0, self.state_size, 0], "constant", 0.0)
        w = inputs.unfold(1, self.receptive_field, self.stride)
        if self.dilation > 1:
            w = w[..., ::self.dilation]
        return w
    def _get_name(self) -> str:
        return "_Window"

class _SlicedSequential(nn.Sequential):
    extra_left_context: int
    stride: int
    def __init__(self):
        super().__init__()
        self.extra_left_context = 0
        self.stride = 1
    def push(self, name: str, m: nn.Module, elc: int = 0, stride: int = 1, sliced: bool = False):
        self.add_module(name, m)
        if sliced:
            self.extra_left_context += elc * self.stride
            self.stride *= stride
    def _get_name(self) -> str:
        return "_SlicedSequential"

class _Residual(nn.Module):
    child: _SlicedSequential
    extra_left_context: int
    stride: int
    dropout: nn.Dropout
    weight: float
    def __init__(self, child: _SlicedSequential, dropout: float = 0.0, weight: float = 1.0):
        super().__init__()
        self.child = child
        self.add_module("child", self.child)
        self.extra_left_context = child.extra_left_context
        self.stride = child.stride
        self.dropout = nn.Dropout(dropout)
        self.add_module("dropout", self.dropout)
        self.weight = weight
    def forward(self, inputs: Tensor) -> Tensor:
        r = inputs[:, self.extra_left_context::self.stride]
        return self.child(inputs) + self.weight * self.dropout(r)
    def _get_name(self) -> str:
        return "_Residual"

class _MultiHeadAttention(nn.Module):
    input_dim: int
    num_heads: int
    window_size: int
    op: nn.MultiheadAttention
    attention_window: _Window
    lpad: int
    extra_left_context: int
    stride: int
    attn_mask: Tensor
    def __init__(self, input_dim: int, num_heads: int, window_size: int, stride: int = 1, lpad: str = "steady",
                 dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.op = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attention_window = _Window(window_size, stride, 1, lpad)
        self.lpad = self.attention_window.lpad
        if self.lpad > 0 and stride > 1:
            raise NotImplementedError("MultiHeadAttention only supports unit stride when lpad > 0")
        self.extra_left_context = self.attention_window.extra_left_context
        self.stride = self.attention_window.stride
        self.attn_mask = torch.ones(window_size, window_size, dtype=torch.bool).triu(diagonal=1).flip([1])
        self.register_buffer("attn_mask", self.attn_mask)
        self.add_module("op", self.op)
        self.add_module("attention_window", self.attention_window)
    def forward(self, inputs: Tensor) -> Tensor:
        b, T = inputs.shape[0], inputs.shape[1]
        x = inputs.reshape(b, T, -1)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected flattened input dim {self.input_dim}, got {x.shape[-1]}")
        win = self.attention_window(x).transpose(-1, -2)  # b To W C
        To, W, C = win.shape[1], win.shape[2], win.shape[3]
        wm = win.reshape(b * To, W, C)
        query = wm[:, W - 1]
        kv = wm.transpose(1, 2).reshape(b * To, W * C).reshape(b * To, W, C)
        am = self.attn_mask
        lo = max(0, self.extra_left_context)
        hi = min(T, self.window_size - 1)
        warm = am[lo:hi:self.stride] if hi > lo else am[0:0]
        nw = warm.shape[0]
        steady = am[am.shape[0] - 1:, :].expand(To - nw, -1)
        mask = torch.cat([warm, steady], dim=0)
        full = mask[None, :, None].expand(b, -1, self.num_heads, -1).reshape(-1, mask.shape[-1])
        out, _ = self.op.attend(query.unsqueeze(1), kv, kv, None, True, full.unsqueeze(1))
        return out.unflatten(0, (b, -1))
    def _get_name(self) -> str:
        return "_MultiHeadAttention"

class _Conv1d(nn.Module):
    receptive_field: int
    state_size: int
    stride: int
    extra_left_context: int
    to_channels_first: Rearrange
    to_time_first: Rearrange
    net: nn.Conv1d
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1,
                 bias: bool = True, groups: int = 1):
        super().__init__()
        self.to_channels_first = Rearrange("batch time channels -> batch channels time")
        self.add_module("to_channels_first", self.to_channels_first)
        self.to_time_first = Rearrange("batch channels time -> batch time channels")
        self.add_module("to_time_first", self.to_time_first)
        self.receptive_field = 1 + dilation * (kernel_size - 1)
        self.state_size = self.receptive_field - stride
        self.stride = stride
        self.extra_left_context = self.receptive_field - 1
        self.net = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias,
                             groups=groups)
        self.add_module("net", self.net)
    def forward(self, x: Tensor) -> Tensor:
        return self.to_time_first(self.net(self.to_channels_first(x)))
    def _get_name(self) -> str:
        return "_Conv1d"

def _ff_block(d: int, ffn: int, drop: float, activation) -> _Residual:
    s = _SlicedSequential()
    s.push("0", nn.LayerNorm(d))
    s.push("1", nn.Linear(d, ffn))
    s.push("2", activation())
    s.push("3", nn.Dropout(drop))
    s.push("4", nn.Linear(ffn, d))
    s.push("5", nn.Dropout(drop))
    return _Residual(s, weight=0.5)

def _conformer_block(d: int, ffn: int, kernel_size: int, stride: int, num_heads: int, win: int, drop: float,
                     activation) -> _SlicedSequential:
    ff1 = _ff_block(d, ffn, drop, activation)
    ff2 = _ff_block(d, ffn, drop, activation)
    blk = _SlicedSequential()
    blk.push("ff_block1", ff1, ff1.extra_left_context, ff1.stride, True)
    if win > 0:
        s = _SlicedSequential()
        s.push("0", nn.LayerNorm(d))
        s.push("1", Rearrange("batch time channels -> batch time 1 channels"))
        mha = _MultiHeadAttention(d, num_heads, win, 1, "steady", drop)
        s.push("2", mha, mha.extra_left_context, mha.stride, True)
        s.push("3", Rearrange("batch time 1 channels -> batch time channels"))
        s.push("4", nn.Dropout(drop))
        a = _Residual(s)
        blk.push("attn_block", a, a.extra_left_context, a.stride, True)
    else:
        blk.push("attn_block", nn.Identity())
    if kernel_size > 0:
        s = _SlicedSequential()
        s.push("0", nn.LayerNorm(d))
        s.push("1", nn.Linear(d, 2 * d))
        s.push("2", nn.GLU(dim=-1))
        cv = _Conv1d(d, d, kernel_size, stride=stride, groups=d)
        s.push("3", cv, cv.extra_left_context, cv.stride, True)
        s.push("4", nn.LayerNorm(d))
        s.push("5", activation())
        s.push("6", nn.Linear(d, d))
        s.push("7", nn.Dropout(drop))
        c = _Residual(s)
        blk.push("conv_block", c, c.extra_left_context, c.stride, True)
    else:
        blk.push("conv_block", nn.Identity())
    blk.push("ff_block2", ff2, ff2.extra_left_context, ff2.stride, True)
    blk.push("layer_norm", nn.LayerNorm(d))
    return blk

def _int_list(v, n: int, name: str) -> List[int]:
    if isinstance(v, int):
        return [v] * n
    out = [int(x) for x in v]
    if len(out) != n:
        raise ValueError(f"{name} length {len(out)} != num_layers {n}")
    return out

class MetaNeuromotorHand(EEGModuleMixin, nn.Module):
    """Handwriting model of the generic neuromotor interface (Meta Reality Labs 2025).
    Output (batch, time, n_outputs)."""
    log_softmax: bool
    featurizer: _MultivariatePowerFrequencyFeatures
    specaug: _MaskAug
    rotation_invariant_mlp: _RotationInvariantMPFMLP
    features_to_sequence: Rearrange
    conformer: _SlicedSequential
    final_layer: nn.Linear

    def __init__(self, n_outputs: int = 100, n_chans: int = 16, sfreq: Optional[float] = None,
                 mpf_window_length: int = 160, mpf_stride: int = 40, mpf_n_fft: int = 64, mpf_fft_stride: int = 10,
                 mpf_frequency_bins = ((0.0, 50.0), (30.0, 100.0), (100.0, 225.0), (225.0, 375.0), (375.0, 700.0),
                                       (700.0, 1000.0)),
                 mask_max_num_masks = (3, 2), mask_max_lengths = (5, 1), mask_dims: str = "TF", mask_value: float = 0.0,
                 invariance_hidden_dims = (64,), invariance_offsets = (-1, 0, 1), num_adjacent_cov: int = 8,
                 conformer_input_dim: int = 64, conformer_ffn_dim: int = 128, conformer_kernel_size = 8,
                 conformer_stride = (1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1), conformer_num_heads: int = 4,
                 conformer_attn_window_size = (16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8),
                 conformer_num_layers: int = 15, drop_prob: float = 0.1, time_reduction_stride: int = 2,
                 log_softmax: bool = False, activation = nn.SiLU, invariance_activation = nn.LeakyReLU,
                 n_times: Optional[int] = None, input_window_seconds: Optional[float] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None):
        sf = sfreq
        if sf is None and (n_times is None or input_window_seconds is None):
            sf = 2000.0
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sf)
        nn.Module.__init__(self)
        self.mapping = {"featurizer.window": "featurizer.stft.window",
                        "featurizer.window_normalization_factor": "featurizer.stft.window_norm",
                        "featurizer.freq_masks": "featurizer.band_averager.freq_masks",
                        "conformer.3.weight": "final_layer.weight", "conformer.3.bias": "final_layer.bias"}
        self.log_softmax = log_softmax
        bins: Optional[List[Tuple[float, float]]] = None
        n_mpf = mpf_n_fft // 2 + 1
        if mpf_frequency_bins is not None:
            bl = List[Tuple[float, float]]()
            for band in mpf_frequency_bins:
                if len(band) != 2:
                    raise ValueError("Each frequency bin must contain exactly two values")
                lo, hi = float(band[0]), float(band[1])
                if hi <= lo:
                    raise ValueError("Frequency bin end must be greater than start")
                bl.append((lo, hi))
            bins = bl
            n_mpf = len(bl)
        self.featurizer = _MultivariatePowerFrequencyFeatures(mpf_window_length, mpf_stride, mpf_n_fft, mpf_fft_stride,
                                                              self.sfreq, bins)
        self.add_module("featurizer", self.featurizer)
        self.specaug = _MaskAug([int(v) for v in mask_max_num_masks], [int(v) for v in mask_max_lengths], mask_dims,
                                mask_value)
        self.add_module("specaug", self.specaug)
        hd = [int(v) for v in invariance_hidden_dims]
        self.rotation_invariant_mlp = _RotationInvariantMPFMLP(self.n_chans, n_mpf, hd,
                                                               [int(v) for v in invariance_offsets], num_adjacent_cov,
                                                               invariance_activation)
        self.add_module("rotation_invariant_mlp", self.rotation_invariant_mlp)
        self.features_to_sequence = Rearrange("batch features time -> batch time features")
        self.add_module("features_to_sequence", self.features_to_sequence)
        nl = conformer_num_layers
        ks = _int_list(conformer_kernel_size, nl, "kernel_size")
        aw = _int_list(conformer_attn_window_size, nl, "attn_window_size")
        st = List[int]()
        if isinstance(conformer_stride, int):
            st = [1] * (nl - 1) + [conformer_stride]
        else:
            st = [int(v) for v in conformer_stride]
            if len(st) != nl:
                raise ValueError("stride length must match num_layers")
        self.conformer = _SlicedSequential()
        dim = hd[-1]
        idx = 0
        if time_reduction_stride > 1:
            w = _Window(time_reduction_stride, time_reduction_stride, 1, "none")
            ws = _SlicedSequential()
            ws.push("0", w, w.extra_left_context, w.stride, True)
            ws.push("1", Rearrange("batch time channels stride -> batch time (stride channels)"))
            self.conformer.push(str(idx), ws, ws.extra_left_context, ws.stride, True)
            idx += 1
            self.conformer.push(str(idx), nn.Linear(dim * time_reduction_stride, conformer_input_dim))
            idx += 1
            dim = conformer_input_dim
        if dim != conformer_input_dim:
            self.conformer.push(str(idx), nn.Linear(dim, conformer_input_dim))
            idx += 1
        enc = _SlicedSequential()
        for i in range(nl):
            b = _conformer_block(conformer_input_dim, conformer_ffn_dim, ks[i], st[i], conformer_num_heads, aw[i],
                                 drop_prob, activation)
            enc.push(str(i), b, b.extra_left_context, b.stride, True)
        self.conformer.push(str(idx), enc, enc.extra_left_context, enc.stride, True)
        self.add_module("conformer", self.conformer)
        self.final_layer = nn.Linear(conformer_input_dim, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.featurizer(x)
        x = self.specaug(x)
        x = self.rotation_invariant_mlp(x)
        x = self.conformer(self.features_to_sequence(x))
        x = self.final_layer(x)
        if self.log_softmax:
            x = F.log_softmax(x, dim=-1)
        return x

    def compute_output_lengths(self, input_lengths: Tensor) -> Tensor:
        l = self.featurizer.compute_time_downsampling(input_lengths)
        l = torch.div(l - self.conformer.extra_left_context - 1, self.conformer.stride, rounding_mode="trunc") + 1
        return l.clamp_min(0)

    def _get_name(self) -> str:
        return "MetaNeuromotorHand"
