# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.functional.patches import _check_on_non_divisible, patch_count, pad_or_crop_to_patches

BRANT_FREQ_BANDS = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0), (50.0, 70.0), (70.0, 90.0), (90.0, 110.0),
                    (110.0, 128.0)]

class _BandPowerFeatures(nn.Module):
    sfreq: float
    patch_size: int
    band_matrix: Tensor
    _scale: Tensor
    def __init__(self, sfreq: float, bands: List[Tuple[float, float]], patch_size: int):
        super().__init__()
        self.sfreq = sfreq
        self.patch_size = patch_size
        nf = patch_size // 2 + 1
        vals = List[float]()
        for i in range(nf):
            f = float(i) * sfreq / float(patch_size)
            for lo, hi in bands:
                vals.append(1.0 if f > lo and f <= hi else 0.0)
        self.band_matrix = torch.tensor(vals, dtype=torch.float32).reshape(nf, len(bands))
        self.register_buffer("band_matrix", self.band_matrix, persistent=False)
        sc = [2.0] * nf
        sc[0] = 1.0
        if patch_size % 2 == 0:
            sc[nf - 1] = 1.0
        self._scale = torch.tensor(sc, dtype=torch.float32) / (sfreq * float(patch_size))
    def forward(self, p: Tensor) -> Tensor:
        if p.shape[-1] != self.patch_size:
            raise ValueError(f"Expected patches of {self.patch_size} samples, got {p.shape[-1]}.")
        x = p - p.mean(dim=-1, keepdim=True)
        psd = torch.abs(torch.fft.rfft(x, dim=-1)).pow(2.0) * self._scale
        band = (psd.unsqueeze(-1) * self.band_matrix).sum(dim=-2)
        return torch.log10(band + 1.0).to(p.dtype)
    def _get_name(self) -> str:
        return "_BandPowerFeatures"

class _BrantInputEmbedding(nn.Module):
    band_encoding: Tensor
    positional_encoding: Tensor
    proj: nn.Sequential
    def __init__(self, patch_size: int, d_model: int, seq_len: int, n_bands: int):
        super().__init__()
        self.band_encoding = nn.Parameter(torch.randn(n_bands, d_model))
        self.register_parameter("band_encoding", self.band_encoding)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        self.register_parameter("positional_encoding", self.positional_encoding)
        self.proj = nn.Sequential(nn.Linear(patch_size, d_model))
        self.add_module("proj", self.proj)
        self.add_module("softmax", nn.Softmax(dim=-1))
    def embed(self, data: Tensor, power: Tensor) -> Tensor:
        w = torch.softmax(power, dim=-1)
        return self.proj(data) + torch.matmul(w, self.band_encoding) + self.positional_encoding
    def _get_name(self) -> str:
        return "_BrantInputEmbedding"

def _encoder(d_model: int, n_heads: int, ffn: int, drop: float, n_layers: int) -> nn.TransformerEncoder:
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
                                                            dropout=drop, batch_first=True), num_layers=n_layers)

class _BrantTemporalEncoder(nn.Module):
    input_embedding: _BrantInputEmbedding
    trans_enc: nn.TransformerEncoder
    def __init__(self, patch_size: int, d_model: int, seq_len: int, n_bands: int, dim_feedforward: int, n_layers: int,
                 n_heads: int, drop_prob: float):
        super().__init__()
        self.input_embedding = _BrantInputEmbedding(patch_size, d_model, seq_len, n_bands)
        self.add_module("input_embedding", self.input_embedding)
        self.trans_enc = _encoder(d_model, n_heads, dim_feedforward, drop_prob, n_layers)
        self.add_module("trans_enc", self.trans_enc)
    def encode(self, data: Tensor, power: Tensor) -> Tensor:
        return self.trans_enc(self.input_embedding.embed(data, power))
    def _get_name(self) -> str:
        return "_BrantTemporalEncoder"

class _BrantSpatialEncoder(nn.Module):
    proj_out: nn.Sequential
    trans_enc: nn.TransformerEncoder
    def __init__(self, d_model: int, out_dim: int, dim_feedforward: int, n_layers: int, n_heads: int, drop_prob: float):
        super().__init__()
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        self.add_module("proj_out", self.proj_out)
        self.trans_enc = _encoder(d_model, n_heads, dim_feedforward, drop_prob, n_layers)
        self.add_module("trans_enc", self.trans_enc)
    def forward(self, x: Tensor) -> Tensor:
        return self.trans_enc(x)
    def _get_name(self) -> str:
        return "_BrantSpatialEncoder"

class Brant(EEGModuleMixin, nn.Module):
    """Brant from Zhang et al (2023). Architecture only (no Hub weights).

    The optional ``ch_mask`` of braindecode's forward is not ported."""
    patch_size: int
    on_non_divisible: str
    embed_dim: int
    seq_len: int
    band_power: _BandPowerFeatures
    temporal_encoder: _BrantTemporalEncoder
    spatial_encoder: _BrantSpatialEncoder
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, patch_size: int = 250,
                 on_non_divisible: str = "pad", embed_dim: int = 256, ffn_dim: int = 384, temporal_n_layers: int = 4,
                 spatial_n_layers: int = 2, n_heads: int = 8, n_freq_bands: int = 8, band_power_sfreq: float = 256.0,
                 drop_prob: float = 0.1):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if n_freq_bands != len(BRANT_FREQ_BANDS):
            raise ValueError(f"n_freq_bands must be {len(BRANT_FREQ_BANDS)}, got {n_freq_bands}.")
        _check_on_non_divisible(on_non_divisible)
        self.patch_size = patch_size
        self.on_non_divisible = on_non_divisible
        self.embed_dim = embed_dim
        self.seq_len = patch_count(self.n_times, patch_size, on_non_divisible)
        self.band_power = _BandPowerFeatures(band_power_sfreq, BRANT_FREQ_BANDS, patch_size)
        self.add_module("band_power", self.band_power)
        self.temporal_encoder = _BrantTemporalEncoder(patch_size, embed_dim, self.seq_len, n_freq_bands, ffn_dim,
                                                      temporal_n_layers, n_heads, drop_prob)
        self.add_module("temporal_encoder", self.temporal_encoder)
        self.spatial_encoder = _BrantSpatialEncoder(embed_dim, patch_size, ffn_dim, spatial_n_layers, n_heads, drop_prob)
        self.add_module("spatial_encoder", self.spatial_encoder)
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward_features(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.n_times:
            raise ValueError(f"Brant expects {self.n_times} samples, got {x.shape[-1]}.")
        B, C = x.shape[0], x.shape[1]
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        tok = x.reshape(B * C, -1, self.patch_size)
        tz = self.temporal_encoder.encode(tok, self.band_power(tok))
        tz = tz.unflatten(0, [B, C]).permute(0, 2, 1, 3).flatten(0, 1)
        emb = self.spatial_encoder(tz).unflatten(0, [B, self.seq_len])
        return emb.mean(dim=[1, 2])

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x))

    def _get_name(self) -> str:
        return "Brant"
