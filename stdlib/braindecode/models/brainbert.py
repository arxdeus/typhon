# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import sinusoidal_positional_encoding
from braindecode.functional.patches import _check_on_non_divisible, patch_count, pad_or_crop_to_patches

def _act_name(activation) -> str:
    a = activation()
    n = a._get_name()
    if n == "GELU":
        return "gelu"
    if n == "ReLU":
        return "relu"
    raise NotImplementedError(f"activation {n} is not supported by nn.TransformerEncoderLayer here (GELU/ReLU only)")

class _STFTSpectrogram(nn.Module):
    nperseg: int
    idx_freq_cutoff: int
    clip: int
    zscore_before_clip: bool
    min_frames: int
    step: int
    boundary_pad: int
    window: Tensor

    def __init__(self, nperseg: int = 400, noverlap: int = 350, idx_freq_cutoff: int = 40, clip: int = 10,
                 zscore_before_clip: bool = True, min_frames: int = 0):
        super().__init__()
        if not (0 <= noverlap < nperseg):
            raise ValueError(f"noverlap must be in [0, nperseg); got noverlap={noverlap}, nperseg={nperseg}.")
        if not (1 <= idx_freq_cutoff <= nperseg // 2 + 1):
            raise ValueError(f"idx_freq_cutoff must be in [1, nperseg // 2 + 1]; got {idx_freq_cutoff}.")
        if clip < 0:
            raise ValueError(f"clip must be >= 0; got {clip}.")
        self.nperseg = nperseg
        self.idx_freq_cutoff = idx_freq_cutoff
        self.clip = clip
        self.zscore_before_clip = zscore_before_clip
        self.min_frames = min_frames
        self.step = nperseg - noverlap
        self.boundary_pad = nperseg // 2
        self.window = torch.hann_window(nperseg, periodic=True)
        self.register_buffer("window", self.window, persistent=False)

    def n_frames(self, n_times: int) -> int:
        return patch_count(n_times + 2 * self.boundary_pad, self.nperseg, "pad", self.step) - 2 * self.clip

    def min_n_times(self, n_frames: int) -> int:
        n_seg = n_frames + 2 * self.clip
        return self.nperseg + (n_seg - 2) * self.step + 1 - 2 * self.boundary_pad

    @staticmethod
    def _zscore(mag: Tensor) -> Tensor:
        mean = mag.mean(dim=-2, keepdim=True)
        std = mag.std(dim=-2, unbiased=False, keepdim=True)
        std = std.masked_fill(std == 0.0, 1.0)
        return (mag - mean) / std

    def forward(self, x: Tensor) -> Tensor:
        xp = F.pad(x, [self.boundary_pad, self.boundary_pad])
        ms = self.min_frames + 2 * self.clip if self.min_frames else 1
        xp = pad_or_crop_to_patches(xp, self.nperseg, "pad", self.step, ms, center=True)
        frames = xp.unfold(-1, self.nperseg, self.step)
        win = self.window.to(frames.dtype)
        scale = 1.0 / win.sum().item()
        spec = torch.fft.rfft(frames * win, dim=-1)
        mag = torch.abs(spec[..., :self.idx_freq_cutoff]) * scale
        if self.zscore_before_clip:
            mag = self._zscore(mag)
            deg = mag.flatten(-2).std(dim=-1) == 0.0
            mag = mag.masked_fill(deg.unsqueeze(-1).unsqueeze(-1), 1.0)
        if self.clip:
            mag = mag[..., self.clip:mag.shape[-2] - self.clip, :]
        if not self.zscore_before_clip:
            mag = self._zscore(mag)
        return torch.nan_to_num(mag, nan=0.0).to(x.dtype)

    def _get_name(self) -> str:
        return "_STFTSpectrogram"

class _BrainBERTInputEmbedding(nn.Module):
    in_proj: nn.Linear
    layer_norm: nn.LayerNorm
    dropout: nn.Dropout
    pe: Tensor

    def __init__(self, input_dim: int, hidden_dim: int, drop_prob: float, max_len: int):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.add_module("in_proj", self.in_proj)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.add_module("layer_norm", self.layer_norm)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.pe = sinusoidal_positional_encoding(max_len, hidden_dim)
        self.register_buffer("pe", self.pe, persistent=False)

    def forward(self, spec: Tensor) -> Tensor:
        h = self.in_proj(spec)
        if h.size(1) > self.pe.size(0):
            raise ValueError(f"Sequence of {h.size(1)} frames exceeds the positional table ({self.pe.size(0)}).")
        return self.dropout(self.layer_norm(h + self.pe[:h.size(1)]))

    def _get_name(self) -> str:
        return "_BrainBERTInputEmbedding"

class BrainBERT(EEGModuleMixin, nn.Module):
    """BrainBERT from Wang et al (2023). Architecture only (no Hub weights).

    The optional ``ch_mask`` of braindecode's forward is not ported (all
    channels are treated as real); activation must be GELU or ReLU."""
    pool_n_frames: Optional[int]
    min_frames: int
    spectrogram: _STFTSpectrogram
    input_embedding: _BrainBERTInputEmbedding
    transformer: nn.TransformerEncoder
    final_layer: nn.Linear

    def __init__(self, hidden_dim: int = 192, ffn_dim: int = 384, n_layers: int = 2, n_heads: int = 4,
                 nperseg: int = 400, noverlap: int = 350, idx_freq_cutoff: int = 40, stft_clip: int = 10,
                 stft_zscore_before_clip: bool = True, pool_n_frames: Optional[int] = 10, activation = nn.GELU,
                 drop_prob: float = 0.1, on_non_divisible: str = "pad", n_outputs: Optional[int] = None,
                 n_chans: Optional[int] = None, chs_info: Optional[List[Dict[str, str]]] = None,
                 n_times: Optional[int] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if pool_n_frames is not None and pool_n_frames.__val__() <= 0:
            raise ValueError(f"pool_n_frames must be positive or None; got {pool_n_frames}.")
        _check_on_non_divisible(on_non_divisible)
        self.pool_n_frames = pool_n_frames
        self.min_frames = 1 if pool_n_frames is None else pool_n_frames.__val__()
        self.spectrogram = _STFTSpectrogram(nperseg, noverlap, idx_freq_cutoff, stft_clip, stft_zscore_before_clip,
                                            self.min_frames if on_non_divisible == "pad" else 0)
        self.add_module("spectrogram", self.spectrogram)
        nf = self.spectrogram.n_frames(self.n_times)
        if nf < self.min_frames:
            if on_non_divisible != "pad":
                raise ValueError(f"n_times={self.n_times} gives {nf} frames, fewer than {self.min_frames}.")
            nf = self.min_frames
        for name in ("in_proj.weight", "in_proj.bias", "layer_norm.weight", "layer_norm.bias"):
            self.mapping["input_encoding." + name] = "input_embedding." + name
        self.input_embedding = _BrainBERTInputEmbedding(idx_freq_cutoff, hidden_dim, drop_prob, max(5000, nf))
        self.add_module("input_embedding", self.input_embedding)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=ffn_dim,
                                           dropout=drop_prob, activation=_act_name(activation), batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.add_module("transformer", self.transformer)
        self.final_layer = nn.Linear(hidden_dim, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward_features(self, x: Tensor) -> Tensor:
        nc = x.shape[1]
        spec = self.spectrogram(x.flatten(0, 1))
        nf = spec.shape[1]
        if nf < self.min_frames:
            raise ValueError(f"Input gives {nf} STFT frames, fewer than the {self.min_frames} required.")
        z = self.transformer(self.input_embedding(spec))
        if self.pool_n_frames is not None:
            p = self.pool_n_frames.__val__()
            s = nf // 2 - p // 2
            z = z[:, s:s + p]
        return z.unflatten(0, [-1, nc]).mean(dim=[1, 2])

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x))

    def _get_name(self) -> str:
        return "BrainBERT"
