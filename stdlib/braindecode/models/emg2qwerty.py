# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# EMG2QwertyNet. Codon notes: the torchaudio log-spectrogram (periodic Hann,
# center=False, normalized="window", power=2) is computed with an explicit
# float64 DFT; spec_augment=True, return_feature=True and
# forward(return_features=True) are not supported.

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin

class _LogSpectrogram(nn.Module):
    n_fft: int
    hop_length: int
    num_bands: int
    electrodes_per_band: int
    log_eps: float
    _cos: Tensor
    _sin: Tensor
    def __init__(self, n_fft: int, hop_length: int, num_bands: int, electrodes_per_band: int, log_eps: float = 1e-06):
        super().__init__()
        if log_eps <= 0:
            raise ValueError(f"log_eps must be > 0; got {log_eps}.")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_bands = num_bands
        self.electrodes_per_band = electrodes_per_band
        self.log_eps = log_eps
        nf = n_fft // 2 + 1
        win = [0.5 - 0.5 * math.cos(2 * math.pi * k / n_fft) for k in range(n_fft)]
        wn = math.sqrt(sum(w * w for w in win))
        cv = List[float](capacity=nf * n_fft)
        sv = List[float](capacity=nf * n_fft)
        for f in range(nf):
            for k in range(n_fft):
                a = 2 * math.pi * float((f * k) % n_fft) / n_fft
                cv.append(win[k] * math.cos(a) / wn)
                sv.append(-win[k] * math.sin(a) / wn)
        self._cos = torch.tensor(cv, dtype=torch.float64).reshape(nf, n_fft)
        self._sin = torch.tensor(sv, dtype=torch.float64).reshape(nf, n_fft)
    def forward(self, x: Tensor) -> Tensor:
        b, c = x.shape[0], x.shape[1]
        if c != self.num_bands * self.electrodes_per_band:
            raise ValueError(f"_LogSpectrogram expected {self.num_bands * self.electrodes_per_band} channels; got {c}.")
        frames = x.detach().double().unfold(-1, self.n_fft, self.hop_length)  # b c t n_fft
        re = torch.matmul(frames, self._cos.t())
        im = torch.matmul(frames, self._sin.t())
        p = (re * re + im * im).float().transpose(-1, -2)  # b c f t
        lp = torch.log10(p + self.log_eps).to(x.dtype)
        nfb, nfr = lp.shape[-2], lp.shape[-1]
        return lp.reshape(b, self.num_bands, self.electrodes_per_band, nfb, nfr).movedim(-1, 0)
    def _get_name(self) -> str:
        return "_LogSpectrogram"

class _SpectrogramNorm(nn.Module):
    channels: int
    batch_norm: nn.BatchNorm2d
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.batch_norm = nn.BatchNorm2d(channels)
        self.add_module("batch_norm", self.batch_norm)
    def forward(self, x: Tensor) -> Tensor:
        t, b, nb, ne, nf = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        if self.channels != nb * ne:
            raise ValueError(f"_SpectrogramNorm expected {self.channels} flat channels; got {nb * ne}.")
        fl = x.movedim(0, -1).reshape(b, nb * ne, nf, t)
        return self.batch_norm(fl).reshape(b, nb, ne, nf, t).movedim(-1, 0)
    def _get_name(self) -> str:
        return "_SpectrogramNorm"

class _RotationInvariantMLP(nn.Module):
    mlp: nn.Sequential
    pooling: str
    offsets: List[int]
    def __init__(self, in_features: int, mlp_features: List[int], pooling: str, offsets: List[int], activation):
        super().__init__()
        self.mlp = nn.Sequential()
        d = in_features
        for o in mlp_features:
            self.mlp.append(nn.Linear(d, o))
            self.mlp.append(activation())
            d = o
        self.add_module("mlp", self.mlp)
        if pooling != "max" and pooling != "mean":
            raise ValueError(f"pooling must be 'max' or 'mean'; got {pooling}.")
        self.pooling = pooling
        self.offsets = offsets if len(offsets) > 0 else [0]
    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs.contiguous()
        rolled = torch.stack([inputs.roll(o, dims=2) for o in self.offsets], dim=2)
        f = self.mlp(rolled.flatten(start_dim=3))
        if self.pooling == "max":
            return f.amax(dim=2)
        return f.mean(dim=2)
    def _get_name(self) -> str:
        return "_RotationInvariantMLP"

class _MultiBandRotationInvariantMLP(nn.Module):
    num_bands: int
    stack_dim: int
    mlps: nn.ModuleList
    _m: List[_RotationInvariantMLP]
    def __init__(self, in_features: int, mlp_features: List[int], pooling: str, offsets: List[int], num_bands: int,
                 activation):
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = 2
        self._m = [_RotationInvariantMLP(in_features, mlp_features, pooling, offsets, activation) for _ in range(num_bands)]
        self.mlps = nn.ModuleList()
        for m in self._m:
            self.mlps.append(m)
        self.add_module("mlps", self.mlps)
    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[2] != self.num_bands:
            raise ValueError(f"_MultiBandRotationInvariantMLP expected dim 2 == {self.num_bands}; got input shape {inputs.shape}.")
        per = inputs.unbind(2)
        return torch.stack([self._m[i](per[i]) for i in range(self.num_bands)], dim=2)
    def _get_name(self) -> str:
        return "_MultiBandRotationInvariantMLP"

class _TDSConv2dBlock(nn.Module):
    channels: int
    width: int
    conv2d: nn.Conv2d
    activation: nn.Module
    layer_norm: nn.LayerNorm
    fold_features: Rearrange
    unfold_features: Rearrange
    features_to_sequence: Rearrange
    sequence_to_features: Rearrange
    time_first: bool
    def __init__(self, channels: int, width: int, kernel_width: int, activation, time_first: bool = True):
        super().__init__()
        self.time_first = time_first
        self.channels = channels
        self.width = width
        self.conv2d = nn.Conv2d(channels, channels, kernel_size=(1, kernel_width))
        self.add_module("conv2d", self.conv2d)
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.layer_norm = nn.LayerNorm(channels * width)
        self.add_module("layer_norm", self.layer_norm)
        if time_first:
            self.fold_features = Rearrange("time batch (channels width) -> batch channels width time", channels=channels, width=width)
            self.unfold_features = Rearrange("batch channels width time -> time batch (channels width)")
        else:
            self.fold_features = Rearrange("batch (channels width) time -> batch channels width time", channels=channels, width=width)
            self.unfold_features = Rearrange("batch channels width time -> batch (channels width) time")
        self.add_module("fold_features", self.fold_features)
        self.add_module("unfold_features", self.unfold_features)
        self.features_to_sequence = Rearrange("batch features time -> batch time features")
        self.add_module("features_to_sequence", self.features_to_sequence)
        self.sequence_to_features = Rearrange("batch time features -> batch features time")
        self.add_module("sequence_to_features", self.sequence_to_features)
    def forward(self, inputs: Tensor) -> Tensor:
        a = self.activation(self.conv2d(self.fold_features(inputs)))
        n = a.shape[-1]
        if self.time_first:
            return self.layer_norm(self.unfold_features(a) + inputs[inputs.shape[0] - n:])
        r = self.unfold_features(a) + inputs[..., inputs.shape[-1] - n:]
        return self.sequence_to_features(self.layer_norm(self.features_to_sequence(r)))
    def _get_name(self) -> str:
        return "_TDSConv2dBlock"

class _TDSFullyConnectedBlock(nn.Module):
    fc_block: nn.Sequential
    layer_norm: nn.LayerNorm
    features_to_sequence: Rearrange
    sequence_to_features: Rearrange
    time_first: bool
    def __init__(self, num_features: int, activation, drop_prob: float, time_first: bool = True):
        super().__init__()
        self.time_first = time_first
        self.fc_block = nn.Sequential()
        self.fc_block.append(nn.Linear(num_features, num_features))
        self.fc_block.append(activation())
        if drop_prob > 0:
            self.fc_block.append(nn.Dropout(drop_prob))
        self.fc_block.append(nn.Linear(num_features, num_features))
        if drop_prob > 0:
            self.fc_block.append(nn.Dropout(drop_prob))
        self.add_module("fc_block", self.fc_block)
        self.layer_norm = nn.LayerNorm(num_features)
        self.add_module("layer_norm", self.layer_norm)
        self.features_to_sequence = Rearrange("batch features time -> batch time features")
        self.add_module("features_to_sequence", self.features_to_sequence)
        self.sequence_to_features = Rearrange("batch time features -> batch features time")
        self.add_module("sequence_to_features", self.sequence_to_features)
    def forward(self, inputs: Tensor) -> Tensor:
        if self.time_first:
            return self.layer_norm(self.fc_block(inputs) + inputs)
        sq = self.features_to_sequence(inputs)
        return self.sequence_to_features(self.layer_norm(self.fc_block(sq) + sq))
    def _get_name(self) -> str:
        return "_TDSFullyConnectedBlock"

class TDSConvEncoder(nn.Module):
    """Time-depth-separable convolutional encoder (braindecode.modules.TDSConvEncoder)."""
    tds_conv_blocks: nn.Sequential
    def __init__(self, num_features: int, block_channels: List[int], kernel_width: int = 32, activation = nn.ReLU,
                 drop_prob: float = 0.0, time_first: bool = True):
        super().__init__()
        if len(block_channels) == 0:
            raise ValueError("block_channels must be non-empty.")
        self.tds_conv_blocks = nn.Sequential()
        for c in block_channels:
            if num_features % c != 0:
                raise ValueError(f"block_channels entry {c} does not evenly divide num_features={num_features}.")
            self.tds_conv_blocks.append(_TDSConv2dBlock(c, num_features // c, kernel_width, activation, time_first))
            self.tds_conv_blocks.append(_TDSFullyConnectedBlock(num_features, activation, drop_prob, time_first))
        self.add_module("tds_conv_blocks", self.tds_conv_blocks)
    def forward(self, x: Tensor) -> Tensor:
        return self.tds_conv_blocks(x)
    def _get_name(self) -> str:
        return "TDSConvEncoder"

class EMG2QwertyNet(EEGModuleMixin, nn.Module):
    """emg2qwerty baseline (Sivakumar et al 2024). Output (T, batch, n_outputs)."""
    num_bands: int
    electrodes_per_band: int
    n_fft: int
    hop_length: int
    kernel_width: int
    log_softmax: bool
    _n_conv_blocks: int
    spectrogram: _LogSpectrogram
    spec_augment: nn.Identity
    model: nn.Sequential
    final_layer: nn.Linear

    def __init__(self, n_outputs: int = 99, n_chans: int = 32, sfreq: Optional[float] = None, num_bands: int = 2,
                 electrodes_per_band: int = 16, n_fft: int = 64, hop_length: int = 16, log_eps: float = 1e-06,
                 mlp_features = (384,), rotation_offsets = (-1, 0, 1), pooling: str = "mean",
                 block_channels = (24, 24, 24, 24), kernel_width: int = 32, log_softmax: bool = False,
                 activation = nn.ReLU, drop_prob: float = 0.0, spec_augment: bool = False, n_time_masks: int = 3,
                 time_mask_param: int = 25, n_freq_masks: int = 2, freq_mask_param: int = 4,
                 spec_augment_prob: float = 1.0, return_feature: bool = False, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, chs_info: Optional[List[Dict[str, str]]] = None):
        sf = sfreq
        if sf is None and (n_times is None or input_window_seconds is None):
            sf = 2000.0
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sf)
        nn.Module.__init__(self)
        if spec_augment:
            raise NotImplementedError("EMG2QwertyNet: spec_augment=True is not supported in Codon")
        if return_feature:
            raise NotImplementedError("EMG2QwertyNet: return_feature=True is not supported in Codon")
        if num_bands < 1 or electrodes_per_band < 1:
            raise ValueError("num_bands and electrodes_per_band must be >= 1.")
        if self.n_chans != num_bands * electrodes_per_band:
            raise ValueError(f"EMG2QwertyNet expects n_chans == num_bands * electrodes_per_band; got n_chans={self.n_chans}.")
        mf = [int(v) for v in mlp_features]
        bc = [int(v) for v in block_channels]
        if len(mf) == 0:
            raise ValueError("mlp_features must contain at least one entry.")
        if len(bc) == 0:
            raise ValueError("block_channels must contain at least one entry.")
        if pooling != "max" and pooling != "mean":
            raise ValueError(f"pooling must be 'max' or 'mean'; got {pooling!r}.")
        self.num_bands = num_bands
        self.electrodes_per_band = electrodes_per_band
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.kernel_width = kernel_width
        self.log_softmax = log_softmax
        nfb = n_fft // 2 + 1
        nfeat = num_bands * mf[-1]
        self.spectrogram = _LogSpectrogram(n_fft, hop_length, num_bands, electrodes_per_band, log_eps)
        self.add_module("spectrogram", self.spectrogram)
        self.spec_augment = nn.Identity()
        self.add_module("spec_augment", self.spec_augment)
        self.model = nn.Sequential(_SpectrogramNorm(num_bands * electrodes_per_band),
                                   _MultiBandRotationInvariantMLP(electrodes_per_band * nfb, mf, pooling,
                                                                  [int(o) for o in rotation_offsets], num_bands,
                                                                  activation),
                                   nn.Flatten(start_dim=2),
                                   TDSConvEncoder(nfeat, bc, kernel_width, activation, drop_prob))
        self.add_module("model", self.model)
        self.final_layer = nn.Linear(nfeat, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        self._n_conv_blocks = len(bc)
        self.mapping = {"model.4.weight": "final_layer.weight", "model.4.bias": "final_layer.bias"}

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-2] != self.n_chans:
            raise ValueError(f"expected (batch, {self.n_chans}, T) input; got shape {x.shape}.")
        mn = self.n_fft + self._n_conv_blocks * (self.kernel_width - 1) * self.hop_length
        if x.shape[-1] < mn:
            raise ValueError(f"n_times={x.shape[-1]} is shorter than the encoder's receptive field ({mn} samples).")
        enc = self.model(self.spec_augment(self.spectrogram(x)))
        em = self.final_layer(enc)
        if self.log_softmax:
            em = F.log_softmax(em, dim=-1)
        return em.transpose(0, 1).contiguous()

    def compute_output_lengths(self, input_lengths: Tensor) -> Tensor:
        sl = torch.div(input_lengths - self.n_fft, self.hop_length, rounding_mode="floor") + 1
        return (sl - self._n_conv_blocks * (self.kernel_width - 1)).clamp_min(0)

    def _get_name(self) -> str:
        return "EMG2QwertyNet"
