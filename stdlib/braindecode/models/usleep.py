# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _bn_off_if_batch_one, _bn_restore

class _EncoderBlock(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    downsample: int
    block_prepool: nn.Sequential
    pad: nn.ConstantPad1d
    maxpool: nn.MaxPool1d

    def __init__(self, in_channels: int = 2, out_channels: int = 2, kernel_size: int = 9, downsample: int = 2,
                 activation = nn.ELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.block_prepool = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same"),
                                           activation(), nn.BatchNorm1d(num_features=out_channels))
        self.add_module("block_prepool", self.block_prepool)
        self.pad = nn.ConstantPad1d(padding=1, value=0.0)
        self.add_module("pad", self.pad)
        self.maxpool = nn.MaxPool1d(kernel_size=downsample, stride=downsample)
        self.add_module("maxpool", self.maxpool)

    def _fwd(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.block_prepool(x)
        residual = x
        if x.shape[-1] % 2:
            x = self.pad(x)
        return (self.maxpool(x), residual)

    def _get_name(self) -> str:
        return "_EncoderBlock"

class _DecoderBlock(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    upsample: int
    with_skip_connection: bool
    block_preskip: nn.Sequential
    block_postskip: nn.Sequential

    def __init__(self, in_channels: int = 2, out_channels: int = 2, kernel_size: int = 9, upsample: int = 2,
                 with_skip_connection: bool = True, activation = nn.ELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.with_skip_connection = with_skip_connection
        self.block_preskip = nn.Sequential(nn.Upsample(scale_factor=float(upsample)),
                                           nn.Conv1d(in_channels, out_channels, kernel_size=2, padding="same"),
                                           activation(), nn.BatchNorm1d(num_features=out_channels))
        self.add_module("block_preskip", self.block_preskip)
        self.block_postskip = nn.Sequential(
            nn.Conv1d(2 * out_channels if with_skip_connection else out_channels, out_channels, kernel_size=kernel_size, padding="same"),
            activation(), nn.BatchNorm1d(num_features=out_channels))
        self.add_module("block_postskip", self.block_postskip)

    def _fwd(self, x: Tensor, residual: Tensor) -> Tensor:
        x = self.block_preskip(x)
        if self.with_skip_connection:
            d = min(x.shape[-1], residual.shape[-1])
            x = torch.cat([x.narrow(-1, 0, d), residual.narrow(-1, 0, d)], dim=1)
        return self.block_postskip(x)

    def _get_name(self) -> str:
        return "_DecoderBlock"

class USleep(EEGModuleMixin, nn.Module):
    """U-Sleep from Perslev et al (2021)."""
    channels: List[int]
    encoder_blocks: nn.ModuleList
    _enc: List[_EncoderBlock]
    _dec: List[_DecoderBlock]
    bottom: nn.Sequential
    decoder_blocks: nn.ModuleList
    clf: nn.Sequential
    final_layer: nn.Sequential

    def __init__(self, n_chans: Optional[int] = None, sfreq: Optional[float] = None, depth: int = 12,
                 n_time_filters: int = 5, complexity_factor: float = 1.67, with_skip_connection: bool = True,
                 n_outputs: Optional[int] = 5, input_window_seconds: Optional[float] = None,
                 time_conv_size_s: float = 9 / 128, ensure_odd_conv_size: bool = False, activation = nn.ELU,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.mapping["clf.3.weight"] = "final_layer.0.weight"
        self.mapping["clf.3.bias"] = "final_layer.0.bias"
        self.mapping["clf.5.weight"] = "final_layer.2.weight"
        self.mapping["clf.5.bias"] = "final_layer.2.bias"
        mps = 2
        # np.round: round half to even
        tcs = int(_round_half_even(time_conv_size_s * self.sfreq))
        if tcs % 2 == 0:
            if ensure_odd_conv_size:
                tcs += 1
            else:
                raise ValueError(f"time_conv_size = round(time_conv_size_s * sfreq) = {tcs} must be odd for the upsampling step in the decoder blocks; pass ensure_odd_conv_size=True.")
        ch = [self.n_chans]
        nf = float(n_time_filters)
        for _ in range(depth + 1):
            ch.append(int(nf * math.sqrt(complexity_factor)))
            nf = float(int(nf * math.sqrt(2.0)))
        self.channels = ch
        self._enc = [_EncoderBlock(ch[i], ch[i + 1], tcs, mps, activation) for i in range(depth)]
        self.encoder_blocks = nn.ModuleList()
        for e in self._enc:
            self.encoder_blocks.append(e)
        self.add_module("encoder_blocks", self.encoder_blocks)
        self.bottom = nn.Sequential(nn.Conv1d(ch[-2], ch[-1], kernel_size=tcs, padding=(tcs - 1) // 2), activation(),
                                    nn.BatchNorm1d(num_features=ch[-1]))
        self.add_module("bottom", self.bottom)
        rev = ch[::-1]
        self._dec = [_DecoderBlock(rev[i], rev[i + 1], tcs, mps, with_skip_connection, activation) for i in range(depth)]
        self.decoder_blocks = nn.ModuleList()
        for d in self._dec:
            self.decoder_blocks.append(d)
        self.add_module("decoder_blocks", self.decoder_blocks)
        self.clf = nn.Sequential(nn.Conv1d(ch[1], ch[1], kernel_size=1, stride=1, padding=0), nn.Tanh(),
                                 nn.AvgPool1d(self.n_times))
        self.add_module("clf", self.clf)
        self.final_layer = nn.Sequential(nn.Conv1d(ch[1], self.n_outputs, kernel_size=1, stride=1, padding=0), activation(),
                                         nn.Conv1d(self.n_outputs, self.n_outputs, kernel_size=1, stride=1, padding=0),
                                         nn.Identity())
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        off = _bn_off_if_batch_one(self, x.shape[0])
        if x.ndim == 4:
            x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        res = List[Tensor]()
        for e in self._enc:
            x, r = e._fwd(x)
            res.append(r)
        x = self.bottom(x)
        nb = len(self._dec)
        for i in range(nb):
            x = self._dec[i]._fwd(x, res[nb - 1 - i])
        x = self.clf(x)
        y = self.final_layer(x)
        if y.shape[-1] == 1:
            y = y[:, :, 0]
        _bn_restore(off)
        return y

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "USleep"

def _round_half_even(v: float) -> float:
    f = math.floor(v)
    d = v - f
    if d > 0.5:
        return f + 1.0
    if d < 0.5:
        return f
    return f + 1.0 if int(f) % 2 != 0 else f
