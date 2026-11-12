# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin

class _GLU(nn.Module):
    dim: int
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    def forward(self, x: Tensor) -> Tensor:
        return F.glu(x, self.dim)
    def _get_name(self) -> str:
        return "_GLU"

class _ConvSequence(nn.Module):
    sequence: nn.ModuleList
    glus: nn.ModuleList
    skip_projections: nn.ModuleList
    def __init__(self, channels: List[int], kernel_size: int = 3, dilation_growth: float = 2.0, dilation_period: int = 5,
                 dropout: float = 0.0, dropout_input: float = 0.0, batch_norm: bool = True, glu: int = 2,
                 glu_context: int = 1, activation = nn.GELU):
        super().__init__()
        if dilation_growth > 1.0 and kernel_size % 2 == 0:
            raise ValueError("Supports only odd kernel with dilation for now")
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        dil = 1.0
        for k in range(len(channels) - 1):
            ci, co = channels[k], channels[k + 1]
            seq = nn.Sequential()
            j = 0
            if k == 0 and dropout_input > 0:
                seq.add_module(str(j), nn.Dropout(dropout_input))
                j += 1
            if dilation_period and k % dilation_period == 0:
                dil = 1.0
            d = int(dil)
            seq.add_module(str(j), nn.Conv1d(ci, co, kernel_size=kernel_size, stride=1, padding=kernel_size // 2 * d, dilation=d))
            j += 1
            if batch_norm:
                seq.add_module(str(j), nn.BatchNorm1d(co))
                j += 1
            seq.add_module(str(j), activation())
            j += 1
            if dropout > 0:
                seq.add_module(str(j), nn.Dropout(dropout))
            dil *= dilation_growth
            self.sequence.append(seq)
            sp: nn.Module = nn.Identity()
            if ci != co:
                sp = nn.Conv1d(ci, co, 1)
            self.skip_projections.append(sp)
            g: nn.Module = nn.Identity()
            if glu > 0 and (k + 1) % glu == 0:
                g = nn.Sequential(nn.Conv1d(co, 2 * co, 1 + 2 * glu_context, padding=glu_context), _GLU(1))
            self.glus.append(g)
        self.add_module("sequence", self.sequence)
        self.add_module("glus", self.glus)
        self.add_module("skip_projections", self.skip_projections)
    def forward(self, x: Tensor) -> Tensor:
        s = self.sequence.children()
        g = self.glus.children()
        p = self.skip_projections.children()
        for i in range(len(s)):
            x = g[i](p[i](x) + s[i](x))
        return x
    def _get_name(self) -> str:
        return "_ConvSequence"

class BrainModule(EEGModuleMixin, nn.Module):
    """BrainModule (simple conv) from Defossez et al (2023). Architecture only.

    Ported: the dilated residual GLU conv encoder with the input projection
    and pooled linear head. Not ported (raise if requested): the spatial
    ChannelMerger (use_merger), STFT input (n_fft), subject layers /
    subject embeddings, and channel dropout."""
    hidden_dim: int
    input_projection: nn.Conv1d
    encoder: _ConvSequence
    final_layer: nn.Sequential

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 sfreq: Optional[float] = None, chs_info: Optional[List[Dict[str, str]]] = None,
                 input_window_seconds: Optional[float] = None, hidden_dim: int = 320, depth: int = 10,
                 kernel_size: int = 3, growth: float = 1.0, dilation_growth: float = 2.0, dilation_period: int = 5,
                 use_merger: bool = False, conv_drop_prob: float = 0.0, dropout_input: float = 0.0,
                 batch_norm: bool = True, activation = nn.GELU, subject_dim: int = 0, subject_layers: bool = False,
                 n_fft: Optional[int] = None, channel_dropout_prob: float = 0.0, glu: int = 2, glu_context: int = 1):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if use_merger or n_fft is not None or subject_dim > 0 or subject_layers or channel_dropout_prob > 0:
            raise NotImplementedError("BrainModule: merger, STFT, subject layers and channel dropout are not ported.")
        if depth < 1 or kernel_size < 1 or growth <= 0 or glu < 0 or glu_context < 0:
            raise ValueError("BrainModule: invalid depth / kernel_size / growth / glu parameters.")
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Conv1d(self.n_chans, hidden_dim, 1)
        self.add_module("input_projection", self.input_projection)
        dims = [hidden_dim] + [int(round(float(hidden_dim) * growth ** float(k))) for k in range(depth)]
        self.encoder = _ConvSequence(dims, kernel_size, dilation_growth, dilation_period, conv_drop_prob, dropout_input,
                                     batch_norm, glu, glu_context, activation)
        self.add_module("encoder", self.encoder)
        self.final_layer = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(1), nn.Linear(dims[-1], self.n_outputs))
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.n_chans:
            raise ValueError(f"Expected {self.n_chans} channels, got {x.shape[1]}")
        return self.final_layer(self.encoder(self.input_projection(x)))

    def _get_name(self) -> str:
        return "BrainModule"
