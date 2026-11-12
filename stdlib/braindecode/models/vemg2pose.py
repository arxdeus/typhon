# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# VEMG2Pose (emg2pose vemg2pose baseline). forward(x) starts from a zero pose;
# tracking_forward(x, y0) starts from y0. reset_head is not available.

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.models.emg2qwerty import TDSConvEncoder

def _vp_init(m: nn.Module):
    n = m._get_name()
    if n == "Conv1d" or n == "Conv2d" or n == "Linear" or n == "LSTM":
        for i in range(len(m._pnames)):
            p = m._params[i]
            if p is not None:
                nm = m._pnames[i]
                if "weight" in nm:
                    if n == "LSTM":
                        init.normal_(p.__val__(), std=0.02)
                    else:
                        init.trunc_normal_(p.__val__(), std=0.02)
                elif "bias" in nm:
                    init.zeros_(p.__val__())

class _Conv1dBlock(nn.Module):
    kernel_size: int
    stride: int
    conv: nn.Sequential
    norm: nn.LayerNorm
    channels_to_sequence: Rearrange
    sequence_to_channels: Rearrange
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, drop_prob: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0),
                                  nn.ReLU(), nn.Dropout(p=drop_prob))
        self.add_module("conv", self.conv)
        self.norm = nn.LayerNorm(out_channels)
        self.add_module("norm", self.norm)
        self.channels_to_sequence = Rearrange("batch channels time -> batch time channels")
        self.add_module("channels_to_sequence", self.channels_to_sequence)
        self.sequence_to_channels = Rearrange("batch time channels -> batch channels time")
        self.add_module("sequence_to_channels", self.sequence_to_channels)
    def forward(self, x: Tensor) -> Tensor:
        return self.sequence_to_channels(self.norm(self.channels_to_sequence(self.conv(x))))
    def _get_name(self) -> str:
        return "_Conv1dBlock"

class _TdsStage(nn.Module):
    out_channels: int
    layers: nn.Sequential
    linear_layer: nn.Module
    features_to_sequence: Rearrange
    sequence_to_features: Rearrange
    def __init__(self, in_channels: int, subsample_kernel: int, subsample_stride: int, n_blocks: int, channels: int,
                 feature_width: int, kernel_width: int, out_channels: Optional[int], drop_prob: float):
        super().__init__()
        nf = channels * feature_width
        self.out_channels = out_channels if out_channels is not None else nf
        self.layers = nn.Sequential()
        self.layers.add_module("conv1dblock", _Conv1dBlock(in_channels, nf, subsample_kernel, subsample_stride, drop_prob))
        self.layers.add_module("tds_block", TDSConvEncoder(nf, [channels] * n_blocks, kernel_width, time_first=False))
        self.add_module("layers", self.layers)
        if out_channels is not None:
            self.linear_layer = nn.Linear(nf, out_channels.__val__())
        else:
            self.linear_layer = nn.Identity()
        self.add_module("linear_layer", self.linear_layer)
        self.features_to_sequence = Rearrange("batch features time -> batch time features")
        self.add_module("features_to_sequence", self.features_to_sequence)
        self.sequence_to_features = Rearrange("batch time features -> batch features time")
        self.add_module("sequence_to_features", self.sequence_to_features)
    def forward(self, x: Tensor) -> Tensor:
        return self.sequence_to_features(self.linear_layer(self.features_to_sequence(self.layers(x))))
    def _get_name(self) -> str:
        return "_TdsStage"

class _PoseDecoder(nn.Module):
    """_LstmPoseDecoder (lstm) or _MlpPoseDecoder (layers)."""
    kind: str
    lstm: Optional[nn.LSTM]
    layers: Optional[nn.Sequential]
    activation: Optional[nn.Module]
    add_time_axis: Rearrange
    remove_time_axis: Rearrange
    def __init__(self, kind: str, in_features: int, hidden_size: int, num_layers: int, hidden_sizes: List[int], activation):
        super().__init__()
        self.kind = kind
        self.add_time_axis = Rearrange("batch features -> batch 1 features")
        self.remove_time_axis = Rearrange("batch 1 features -> batch features")
        if kind == "lstm":
            l = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.lstm = l
            self.add_module("lstm", l)
            a = activation()
            self.activation = a
            self.add_module("activation", a)
            self.layers = None
            self.add_module("add_time_axis", self.add_time_axis)
            self.add_module("remove_time_axis", self.remove_time_axis)
        else:
            s = nn.Sequential()
            w = in_features
            for h in hidden_sizes:
                s.append(nn.Linear(w, h))
                s.append(nn.LayerNorm(h))
                s.append(activation())
                w = h
            self.layers = s
            self.add_module("layers", s)
            self.lstm = None
            self.activation = None
    def step(self, x: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.kind == "lstm":
            out, nh, nc = self.lstm.__val__().run(self.add_time_axis(x), h, c)
            return self.activation.__val__()(self.remove_time_axis(out)), nh, nc
        return self.layers.__val__()(x), h, c
    def _get_name(self) -> str:
        return "_LstmPoseDecoder" if self.kind == "lstm" else "_MlpPoseDecoder"

class VEMG2Pose(EEGModuleMixin, nn.Module):
    """vemg2pose from Salter et al (2024). Output (batch, n_times - left_context, n_outputs)."""
    input_sfreq: float
    rollout_rate: float
    num_position_steps: int
    parameterization: str
    output_scalar: float
    left_context: int
    _state_layers: int
    _state_size: int
    _head_multiplier: int
    encoder: nn.Sequential
    decoder: _PoseDecoder
    trajectory_to_sequence: Rearrange
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 encoder_channels: int = 256, feature_dim: int = 64, hidden_size: int = 512, lstm_layers: int = 2,
                 decoder: str = "lstm", decoder_hidden_sizes = (512, 512), stem_kernel_sizes = (11, 5),
                 stem_strides = (5, 2), tds_subsample_kernels = (17, 9), tds_subsample_strides = (4, 2),
                 tds_kernel_widths = (9, 5), tds_blocks: int = 2, tds_channels: int = 16, rollout_rate: float = 50.0,
                 num_position_steps: int = 500, parameterization: str = "hybrid", output_scalar: float = 0.01,
                 activation = nn.LeakyReLU, drop_prob: float = 0.0):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        sk = [int(v) for v in stem_kernel_sizes]
        ss = [int(v) for v in stem_strides]
        tk = [int(v) for v in tds_subsample_kernels]
        ts = [int(v) for v in tds_subsample_strides]
        tw = [int(v) for v in tds_kernel_widths]
        dh = [int(v) for v in decoder_hidden_sizes]
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive; got {self.sfreq}.")
        if rollout_rate <= 0:
            raise ValueError(f"rollout_rate must be positive; got {rollout_rate}.")
        if len(sk) == 0:
            raise ValueError("stem_kernel_sizes and stem_strides must be non-empty.")
        if len(sk) != len(ss):
            raise ValueError("stem_kernel_sizes and stem_strides must have the same length.")
        if any(k <= 0 for k in sk) or any(s <= 0 for s in ss):
            raise ValueError("stem kernels and strides must all be positive.")
        if len(tk) != len(ts) or len(tk) != len(tw):
            raise ValueError("tds_subsample_kernels, tds_subsample_strides and tds_kernel_widths must have the same length.")
        if len(tk) == 0:
            raise ValueError("At least one TDS stage must be configured.")
        if any(k <= 0 for k in tk) or any(s <= 0 for s in ts):
            raise ValueError("TDS subsampling kernels and strides must be positive.")
        if any(k <= 0 or k % 2 == 0 for k in tw):
            raise ValueError("TDS kernel widths must be positive odd integers.")
        if decoder != "lstm" and decoder != "mlp":
            raise ValueError(f"decoder must be 'lstm' or 'mlp'; got {decoder!r}.")
        if decoder == "mlp" and len(dh) == 0:
            raise ValueError("decoder_hidden_sizes must be non-empty when decoder='mlp'.")
        if parameterization not in ("hybrid", "velocity", "position"):
            raise ValueError(f"parameterization must be 'hybrid', 'velocity' or 'position'; got {parameterization!r}.")
        if tds_channels <= 0:
            raise ValueError(f"tds_channels must be positive; got {tds_channels}.")
        if encoder_channels % tds_channels:
            raise ValueError(f"tds_channels ({tds_channels}) must divide encoder_channels ({encoder_channels}).")
        self.input_sfreq = float(self.sfreq)
        self.rollout_rate = float(rollout_rate)
        self.num_position_steps = int(num_position_steps)
        self.parameterization = parameterization
        self.output_scalar = float(output_scalar)
        self.encoder = nn.Sequential()
        for i in range(len(sk)):
            self.encoder.append(_Conv1dBlock(self.n_chans if i == 0 else encoder_channels, encoder_channels, sk[i], ss[i],
                                             drop_prob))
        ns = len(tk)
        for i in range(ns):
            oc: Optional[int] = None
            if i == ns - 1:
                oc = feature_dim
            self.encoder.append(_TdsStage(encoder_channels, tk[i], ts[i], tds_blocks, tds_channels,
                                          encoder_channels // tds_channels, tw[i], oc, drop_prob))
        self.add_module("encoder", self.encoder)
        lc = 0
        st = 1
        for i in range(len(sk)):
            lc += (sk[i] - 1) * st
            st *= ss[i]
        for i in range(ns):
            lc += (tk[i] - 1) * st
            st *= ts[i]
            lc += tds_blocks * (tw[i] - 1) * st
        self.left_context = lc
        if self.n_times <= lc:
            raise ValueError(f"VEMG2Pose consumes a left context of {lc} samples before its first prediction; got n_times={self.n_times}. Use a longer window.")
        din = feature_dim + self.n_outputs
        self.decoder = _PoseDecoder(decoder, din, hidden_size, lstm_layers, dh, activation)
        self.add_module("decoder", self.decoder)
        head_in = hidden_size if decoder == "lstm" else dh[-1]
        self._state_layers = lstm_layers if decoder == "lstm" else 1
        self._state_size = hidden_size if decoder == "lstm" else 1
        self._head_multiplier = 2 if parameterization == "hybrid" else 1
        self.trajectory_to_sequence = Rearrange("batch outputs time -> batch time outputs")
        self.add_module("trajectory_to_sequence", self.trajectory_to_sequence)
        self.final_layer = nn.Linear(head_in, self._head_multiplier * self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        self.apply(_vp_init)

    def _run(self, x: Tensor, y0: Optional[Tensor]) -> Tensor:
        b, n_times = x.shape[0], x.shape[2]
        if n_times <= self.left_context:
            raise ValueError(f"VEMG2Pose consumes a left context of {self.left_context} samples before its first prediction; got n_times={n_times}. Use a longer window.")
        features = self.encoder(x)
        vs = (n_times - self.left_context) / self.input_sfreq
        n_steps = max(1, int(round(vs * self.rollout_rate)))
        features = F.interpolate(features, size=n_steps, mode="linear", align_corners=False)
        ps = int(round(self.num_position_steps * (self.rollout_rate / self.input_sfreq)))
        pose = y0.__val__() if y0 is not None else torch.zeros(b, self.n_outputs, dtype=x.dtype)
        h = torch.zeros(self._state_layers, b, self._state_size, dtype=x.dtype)
        c = torch.zeros_like(h)
        steps = List[Tensor]()
        for step in range(n_steps):
            di = torch.cat([features[:, :, step], pose], dim=-1)
            df, h, c = self.decoder.step(di, h, c)
            out = self.final_layer(df) * self.output_scalar
            if self.parameterization == "hybrid":
                pv = torch.split(out, self.n_outputs, dim=1)
                pose = pv[0] if step < ps else pose + pv[1]
            elif self.parameterization == "velocity":
                pose = pose + out
            else:
                pose = out
            steps.append(pose)
        traj = torch.stack(steps, dim=-1)
        traj = F.interpolate(traj, size=n_times - self.left_context, mode="linear", align_corners=False)
        return self.trajectory_to_sequence(traj)

    def forward(self, x: Tensor) -> Tensor:
        return self._run(x, None)

    def tracking_forward(self, x: Tensor, y0: Tensor) -> Tensor:
        return self._run(x, y0)

    def _get_name(self) -> str:
        return "VEMG2Pose"
