# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional.initialization import _param_by_name
from braindecode.modules.crisscross import CrissCrossTransformerEncoderLayer

def _cbramod_init(m):
    n = m._get_name()
    if n in ("Linear", "Conv1d"):
        w = _param_by_name(m, "weight")
        if w is not None:
            nn.init.kaiming_normal_(w.__val__(), mode="fan_out", nonlinearity="relu")
    elif n == "BatchNorm1d":
        w = _param_by_name(m, "weight")
        if w is not None:
            nn.init.ones_(w.__val__())
        b = _param_by_name(m, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())

def _cbramod_d_model(patch_size: int, cksp: List[Tuple[int, int, int, int, Tuple[int, int]]]) -> int:
    ps = patch_size
    for _, k, s, p, _ in cksp:
        ps = int(float(ps + 2 * p - k) / float(s) + 1.0)
    return cksp[-1][0] * ps

class _CBraModPatchEmbedding(nn.Module):
    patch_size: int
    d_model: int
    positional_encoding: nn.Sequential
    mask_encoding: Tensor
    proj_in: nn.Sequential
    spectral_proj: nn.Sequential

    def __init__(self, patch_size: int, cksp: List[Tuple[int, int, int, int, Tuple[int, int]]], drop_prob: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = _cbramod_d_model(patch_size, cksp)
        d = self.d_model
        self.positional_encoding = nn.Sequential(nn.Conv2d(d, d, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3), groups=d))
        self.add_module("positional_encoding", self.positional_encoding)
        self.mask_encoding = nn.Parameter(torch.zeros(patch_size), requires_grad=False)
        self.register_parameter("mask_encoding", self.mask_encoding)
        self.proj_in = nn.Sequential()
        last = 1
        i = 0
        for c, k, s, p, g in cksp:
            self.proj_in.add_module(str(i), nn.Conv2d(last, c, kernel_size=(1, k), stride=(1, s), padding=(0, p)))
            self.proj_in.add_module(str(i + 1), nn.GroupNorm(g[0], g[1]))
            self.proj_in.add_module(str(i + 2), nn.GELU())
            i += 3
            last = c
        self.add_module("proj_in", self.proj_in)
        self.spectral_proj = nn.Sequential(nn.Linear(patch_size // 2 + 1, d), nn.Dropout(drop_prob))
        self.add_module("spectral_proj", self.spectral_proj)

    def forward(self, x: Tensor) -> Tensor:
        bz, ch, pn, ps = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        mx = x.reshape(bz, 1, ch * pn, ps)
        pe = self.proj_in(mx)
        d, p2 = pe.shape[1], pe.shape[3]
        pe = pe.reshape(bz, d, ch, pn, p2).permute(0, 2, 3, 1, 4).reshape(bz, ch, pn, d * p2)
        flat = mx.reshape(-1, ps)
        spec = torch.abs(torch.fft.rfft(flat, dim=-1, norm="forward")).to(flat.dtype).reshape(bz, ch, pn, -1)
        pe = pe + self.spectral_proj(spec)
        pos = self.positional_encoding(pe.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return pe + pos

    def _get_name(self) -> str:
        return "_PatchEmbedding"

class _CBraModEncoder(nn.Module):
    layers: nn.ModuleList
    num_layers: int
    def __init__(self, layers: List[CrissCrossTransformerEncoderLayer]):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in layers:
            self.layers.append(l)
        self.add_module("layers", self.layers)
        self.num_layers = len(layers)
    def forward(self, x: Tensor) -> Tensor:
        for m in self.layers.children():
            x = m(x)
        return x
    def _get_name(self) -> str:
        return "_TransformerEncoder"

class CBraMod(EEGModuleMixin, nn.Module):
    """CBraMod from Wang et al (2025). Architecture only (no Hub weights).

    The reference deep-copies one encoder layer ``n_layer`` times, so every
    layer starts from identical weights; this is reproduced by replaying the
    RNG state for each copy."""
    patch_size: int
    emb_dim: int
    on_non_divisible: str
    patch_embedding: _CBraModPatchEmbedding
    encoder: _CBraModEncoder
    proj_out: nn.Sequential
    final_layer: nn.Module

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 patch_size: int = 200, dim_feedforward: int = 800, n_layer: int = 12, nhead: int = 8,
                 activation = nn.GELU, emb_dim: int = 200,
                 channels_kernel_stride_padding_norm: List[Tuple[int, int, int, int, Tuple[int, int]]] = [
                     (25, 49, 25, 24, (5, 25)), (25, 3, 1, 1, (5, 25)), (25, 3, 1, 1, (5, 25))],
                 drop_prob: float = 0.1, return_encoder_output: bool = False, on_non_divisible: str = "pad"):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if on_non_divisible not in ("pad", "crop", "error"):
            raise ValueError(f"on_non_divisible must be 'pad', 'crop' or 'error', got {on_non_divisible!r}.")
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.on_non_divisible = on_non_divisible
        if on_non_divisible == "error" and self.n_times % patch_size:
            raise ValueError(f"Time dimension ({self.n_times}) is not divisible by patch_size ({patch_size}).")
        self.patch_embedding = _CBraModPatchEmbedding(patch_size, channels_kernel_stride_padding_norm, drop_prob)
        self.add_module("patch_embedding", self.patch_embedding)
        d_model = self.patch_embedding.d_model
        s0 = torch.get_rng_state()
        layers = [CrissCrossTransformerEncoderLayer(d_model, nhead, dim_feedforward, drop_prob, activation,
                                                    batch_first=True, norm_first=True)]
        s1 = torch.get_rng_state()
        for _ in range(1, n_layer):
            torch.set_rng_state(s0)
            layers.append(CrissCrossTransformerEncoderLayer(d_model, nhead, dim_feedforward, drop_prob, activation,
                                                            batch_first=True, norm_first=True))
        torch.set_rng_state(s1)
        self.encoder = _CBraModEncoder(layers)
        self.add_module("encoder", self.encoder)
        self.proj_out = nn.Sequential(nn.Linear(d_model, emb_dim))
        self.add_module("proj_out", self.proj_out)
        self.apply(_cbramod_init)
        fl: nn.Module = nn.Identity()
        if not return_encoder_output:
            fl = nn.Sequential(nn.Flatten(), nn.Linear(self.n_chans * self._n_patch() * emb_dim, self.n_outputs))
        self.final_layer = fl
        self.add_module("final_layer", fl)

    def _n_patch(self) -> int:
        if self.on_non_divisible == "pad":
            return -(-self.n_times // self.patch_size)
        return self.n_times // self.patch_size

    def _tokenize(self, x: Tensor) -> Tensor:
        t = x.shape[-1]
        r = t % self.patch_size
        if r != 0:
            if self.on_non_divisible == "pad":
                x = F.pad(x, [0, self.patch_size - r])
            elif self.on_non_divisible == "crop":
                x = x[:, :, :t - r]
            else:
                raise ValueError(f"Time dimension ({t}) is not divisible by patch_size ({self.patch_size}).")
        return x.reshape(x.shape[0], x.shape[1], -1, self.patch_size)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.proj_out(self.encoder(self.patch_embedding(self._tokenize(x))))

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x))

    def _get_name(self) -> str:
        return "CBraMod"
