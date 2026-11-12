# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# NeuroPose. The 2-D nearest nn.Upsample of the decoder (integer factors) is
# implemented as repeat_interleave along time and electrodes (identical
# result); state_dict and module indices match braindecode.

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin

class _NearestUpsample2d(nn.Module):
    sh: int
    sw: int
    def __init__(self, sh: int, sw: int):
        super().__init__()
        self.sh = sh
        self.sw = sw
    def forward(self, x: Tensor) -> Tensor:
        return torch.repeat_interleave(torch.repeat_interleave(x, self.sh, dim=2), self.sw, dim=3)
    def _get_name(self) -> str:
        return "Upsample"

def _np_init(m: nn.Module):
    n = m._get_name()
    if n == "Conv1d" or n == "Conv2d" or n == "Linear":
        for i in range(len(m._pnames)):
            p = m._params[i]
            if p is not None:
                if m._pnames[i] == "weight":
                    init.trunc_normal_(p.__val__(), std=0.02)
                elif m._pnames[i] == "bias":
                    init.zeros_(p.__val__())
    elif n in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm"):
        for i in range(len(m._pnames)):
            p = m._params[i]
            if p is not None:
                if m._pnames[i] == "weight":
                    init.ones_(p.__val__())
                elif m._pnames[i] == "bias":
                    init.zeros_(p.__val__())

class _EncoderBlock(nn.Module):
    network: nn.Sequential
    def __init__(self, cin: int, cout: int, kernel_size: Tuple[int, int], pool_size: Tuple[int, int], activation,
                 drop_prob: float):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(cin, cout, kernel_size=kernel_size, padding="same"),
                                     nn.BatchNorm2d(cout), activation(), nn.Dropout(p=drop_prob),
                                     nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
        self.add_module("network", self.network)
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
    def _get_name(self) -> str:
        return "_EncoderBlock"

class _ResBlock(nn.Module):
    network: nn.Sequential
    def __init__(self, channels: int, kernel_size: Tuple[int, int], n_convs: int, activation, drop_prob: float):
        super().__init__()
        self.network = nn.Sequential()
        for _ in range(n_convs):
            self.network.append(nn.Conv2d(channels, channels, kernel_size=kernel_size, padding="same"))
            self.network.append(nn.BatchNorm2d(channels))
            self.network.append(activation())
            self.network.append(nn.Dropout(p=drop_prob))
        self.add_module("network", self.network)
    def forward(self, x: Tensor) -> Tensor:
        return x + self.network(x)
    def _get_name(self) -> str:
        return "_ResBlock"

class _DecoderBlock(nn.Module):
    network: nn.Sequential
    def __init__(self, cin: int, cout: int, kernel_size: Tuple[int, int], upsample_size: Tuple[int, int], activation,
                 drop_prob: float):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(cin, cout, kernel_size=kernel_size, padding="same"),
                                     nn.BatchNorm2d(cout), activation(), nn.Dropout(p=drop_prob),
                                     _NearestUpsample2d(upsample_size[0], upsample_size[1]))
        self.add_module("network", self.network)
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
    def _get_name(self) -> str:
        return "_DecoderBlock"

def _pairs(v) -> List[Tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in v]

class NeuroPose(EEGModuleMixin, nn.Module):
    """NeuroPose from Salter et al (2024)."""
    encoder: nn.Sequential
    resnet: nn.Sequential
    decoder: nn.Sequential
    input_to_encoder: Rearrange
    decoder_to_sequence: Rearrange
    sequence_to_channels: Rearrange
    channels_to_sequence: Rearrange
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 encoder_channels = (32, 128, 256), encoder_pool_sizes = ((10, 2), (8, 2), (4, 4)),
                 decoder_channels = (128, 32, 1), decoder_upsample_sizes = ((10, 4), (8, 4), (4, 2)),
                 n_res_blocks: int = 5, n_convs_per_block: int = 3, kernel_size: Tuple[int, int] = (3, 2),
                 activation = nn.ReLU, drop_prob: float = 0.05):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        ec = [int(c) for c in encoder_channels]
        ep = _pairs(encoder_pool_sizes)
        dc = [int(c) for c in decoder_channels]
        du = _pairs(decoder_upsample_sizes)
        if len(ec) != len(ep):
            raise ValueError(f"encoder_channels and encoder_pool_sizes must have the same length; got {len(ec)} and {len(ep)}.")
        if len(dc) != len(du):
            raise ValueError(f"decoder_channels and decoder_upsample_sizes must have the same length; got {len(dc)} and {len(du)}.")
        if n_res_blocks < 0:
            raise ValueError(f"n_res_blocks must be >= 0; got {n_res_blocks}.")
        if n_convs_per_block < 1:
            raise ValueError(f"n_convs_per_block must be >= 1; got {n_convs_per_block}.")
        self.encoder = nn.Sequential()
        cin = 1
        for i in range(len(ec)):
            self.encoder.append(_EncoderBlock(cin, ec[i], kernel_size, ep[i], activation, drop_prob))
            cin = ec[i]
        self.add_module("encoder", self.encoder)
        self.resnet = nn.Sequential()
        for _ in range(n_res_blocks):
            self.resnet.append(_ResBlock(ec[-1], kernel_size, n_convs_per_block, activation, drop_prob))
        self.add_module("resnet", self.resnet)
        self.decoder = nn.Sequential()
        cin = ec[-1]
        for i in range(len(dc)):
            self.decoder.append(_DecoderBlock(cin, dc[i], kernel_size, du[i], activation, drop_prob))
            cin = dc[i]
        self.add_module("decoder", self.decoder)
        ne = self.n_chans
        for _, pe in ep:
            ne //= pe
        if ne < 1:
            raise ValueError(f"n_chans={self.n_chans} is too small for the encoder's electrode pooling.")
        for _, ue in du:
            ne *= ue
        self.input_to_encoder = Rearrange("batch nchans ntimes -> batch 1 ntimes nchans")
        self.add_module("input_to_encoder", self.input_to_encoder)
        self.decoder_to_sequence = Rearrange("batch features ntimes nchans -> batch ntimes (features nchans)")
        self.add_module("decoder_to_sequence", self.decoder_to_sequence)
        self.sequence_to_channels = Rearrange("batch ntimes njoints -> batch njoints ntimes")
        self.add_module("sequence_to_channels", self.sequence_to_channels)
        self.channels_to_sequence = Rearrange("batch njoints ntimes -> batch ntimes njoints")
        self.add_module("channels_to_sequence", self.channels_to_sequence)
        self.final_layer = nn.Linear(dc[-1] * ne, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        self.apply(_np_init)

    def _forward(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        d = self.decoder(self.resnet(self.encoder(self.input_to_encoder(x))))
        pose = self.final_layer(self.decoder_to_sequence(d))
        pc = F.interpolate(self.sequence_to_channels(pose), size=n, mode="linear", align_corners=False)
        return self.channels_to_sequence(pc)

    def forward(self, x: Tensor) -> Tensor:
        if self.training and x.shape[0] == 1:
            bns = [m for m in self.modules() if "BatchNorm" in m._get_name() and m.training]
            for m in bns:
                m.train(False)
            y = self._forward(x)
            for m in bns:
                m.train(True)
            return y
        return self._forward(x)

    def _get_name(self) -> str:
        return "NeuroPose"
