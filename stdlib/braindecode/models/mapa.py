# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import rotate_pairs
from braindecode.functional.initialization import _param_by_name
from braindecode.modules import FeedForwardBlock

MAPA_DKT_REGIONS = ['ctx-lh-caudalanteriorcingulate', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-lh-entorhinal', 'ctx-lh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-lh-inferiortemporal', 'ctx-lh-insula', 'ctx-lh-isthmuscingulate', 'ctx-lh-lateraloccipital', 'ctx-lh-lateralorbitofrontal', 'ctx-lh-lingual', 'ctx-lh-medialorbitofrontal', 'ctx-lh-middletemporal', 'ctx-lh-paracentral', 'ctx-lh-parahippocampal', 'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-parstriangularis', 'ctx-lh-pericalcarine', 'ctx-lh-postcentral', 'ctx-lh-posteriorcingulate', 'ctx-lh-precentral', 'ctx-lh-precuneus', 'ctx-lh-rostralanteriorcingulate', 'ctx-lh-rostralmiddlefrontal', 'ctx-lh-superiorfrontal', 'ctx-lh-superiorparietal', 'ctx-lh-superiortemporal', 'ctx-lh-supramarginal', 'ctx-lh-transversetemporal', 'ctx-rh-caudalanteriorcingulate', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-cuneus', 'ctx-rh-entorhinal', 'ctx-rh-fusiform', 'ctx-rh-inferiorparietal', 'ctx-rh-inferiortemporal', 'ctx-rh-insula', 'ctx-rh-isthmuscingulate', 'ctx-rh-lateraloccipital', 'ctx-rh-lateralorbitofrontal', 'ctx-rh-lingual', 'ctx-rh-medialorbitofrontal', 'ctx-rh-middletemporal', 'ctx-rh-paracentral', 'ctx-rh-parahippocampal', 'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-parstriangularis', 'ctx-rh-pericalcarine', 'ctx-rh-postcentral', 'ctx-rh-posteriorcingulate', 'ctx-rh-precentral', 'ctx-rh-precuneus', 'ctx-rh-rostralanteriorcingulate', 'ctx-rh-rostralmiddlefrontal', 'ctx-rh-superiorfrontal', 'ctx-rh-superiorparietal', 'ctx-rh-superiortemporal', 'ctx-rh-supramarginal', 'ctx-rh-transversetemporal', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Left-Thalamus-Proper', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Thalamus-Proper']
_SUP_TAPS = [3, 6, 9, 12]
# (n_fft, k0, k1, stride) for the slow, mid and fast bands
_BANDS = [(1024, 1, 7, 8), (256, 2, 7, 2), (128, 4, 10, 1)]
_BAND_BINS = [7, 6, 7]
_BAND_STRIDES = [8, 2, 1]
_INPUT_CLIP_Z = [15.0, 15.0, 20.0]

def _frame_count(n_times: int, spectrogram: bool = False) -> int:
    q = 8
    fr = n_times if spectrogram else 1 + n_times // 64
    n = (fr // q) * q
    if n < q:
        raise ValueError(f"MAPA needs at least {q} frames, got {fr}.")
    return n

def _vit_init(m):
    n = m._get_name()
    if n == "Linear":
        nn.init.trunc_normal_(_param_by_name(m, "weight").__val__(), std=0.02)
        b = _param_by_name(m, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())
    elif n == "LayerNorm":
        w = _param_by_name(m, "weight")
        if w is not None:
            nn.init.ones_(w.__val__())
        b = _param_by_name(m, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())

class _SpectrogramFrontend(nn.Module):
    normalization: str
    def __init__(self, normalization: str):
        super().__init__()
        self.normalization = normalization
    def bands(self, x: Tensor) -> List[Tensor]:
        bs = List[Tensor]()
        if self.normalization == "session":
            nf = _frame_count(x.shape[-1], True)
            bs = x[..., :nf].split(_BAND_BINS, dim=2)
        else:
            B, C = x.shape[0], x.shape[1]
            nf = _frame_count(x.shape[-1])
            w = x.reshape(B * C, x.shape[-1])
            for nfft, k0, k1, _ in _BANDS:
                p = w
                if p.shape[-1] < nfft:
                    p = F.pad(p, [0, nfft - p.shape[-1]])
                sp = torch.stft(p, n_fft=nfft, hop_length=64, win_length=nfft, window=torch.hann_window(nfft, dtype=w.dtype),
                                center=True, normalized=False, return_complex=True)
                band = torch.abs(sp)[:, k0:k1 + 1, :nf]
                if self.normalization == "window":
                    med = torch.median(band, -1, True).values
                    sig = 1.4826 * torch.median(torch.abs(band - med), -1, True).values
                    z = (band - med) / torch.clamp(sig, min=1e-6)
                    band = torch.where(sig >= 1e-6, z, torch.zeros_like(z))
                bs.append(band.reshape(B, C, k1 - k0 + 1, nf).to(x.dtype))
        out = List[Tensor]()
        for i in range(3):
            c = _INPUT_CLIP_Z[i]
            s = _BAND_STRIDES[i]
            b = torch.clamp(bs[i], -c, c)
            idx = torch.arange(0, b.shape[-1], s)
            out.append(b.index_select(-1, idx))
        return out
    def _get_name(self) -> str:
        return "_SpectrogramFrontend"

class _PerBandStem(nn.Module):
    projs: nn.ModuleList
    _projs: List[nn.Linear]
    band_type_emb: Tensor
    def __init__(self, d_model: int, band_bins: List[int]):
        super().__init__()
        self._projs = [nn.Linear(n, d_model) for n in band_bins]
        self.projs = nn.ModuleList()
        for p in self._projs:
            self.projs.append(p)
        self.add_module("projs", self.projs)
        self.band_type_emb = nn.Parameter(torch.empty(len(band_bins), d_model))
        self.register_parameter("band_type_emb", self.band_type_emb)
        self.projs.apply(_vit_init)
        nn.init.trunc_normal_(self.band_type_emb, std=1e-6)
    def stem(self, bands: List[Tensor]) -> Tensor:
        toks = [self._projs[i](bands[i].transpose(-1, -2)) + self.band_type_emb[i] for i in range(len(self._projs))]
        return torch.cat(toks, dim=-2)
    def _get_name(self) -> str:
        return "_PerBandStem"

class _WithinArrayBlock(nn.Module):
    n_heads: int
    norm1: nn.LayerNorm
    qkv: nn.Linear
    out: nn.Linear
    norm2: nn.LayerNorm
    mlp: FeedForwardBlock
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, activation):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.add_module("norm1", self.norm1)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.add_module("qkv", self.qkv)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.add_module("out", self.out)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.add_module("norm2", self.norm2)
        self.mlp = FeedForwardBlock(d_model, mlp_ratio, 0.0, activation)
        self.add_module("mlp", self.mlp)
    def block(self, x: Tensor, cos: Tensor, sin: Tensor, key_mask: Tensor) -> Tensor:
        c = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = c[0].unflatten(-1, [self.n_heads, -1]).transpose(2, 3)
        k = c[1].unflatten(-1, [self.n_heads, -1]).transpose(2, 3)
        v = c[2].unflatten(-1, [self.n_heads, -1]).transpose(2, 3)
        q = q * cos + rotate_pairs(q) * sin
        k = k * cos + rotate_pairs(k) * sin
        sh = q.shape
        m = key_mask.expand(sh[0], -1, -1, -1, -1).flatten(0, 1)
        a = F.scaled_dot_product_attention(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), attn_mask=m).view(sh)
        x = x + self.out(a.transpose(2, 3).flatten(3))
        return x + self.mlp(self.norm2(x))
    def _get_name(self) -> str:
        return "_WithinArrayBlock"

class _MapaEncoder(nn.Module):
    region_embed: nn.ModuleDict
    _region: Optional[nn.Embedding]
    blocks: nn.ModuleList
    _blocks: List[_WithinArrayBlock]
    norms_block: Optional[nn.ModuleList]
    _norms: List[nn.LayerNorm]
    norm_out: Optional[nn.LayerNorm]
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, region_embed: bool, deep_sup: bool, activation):
        super().__init__()
        self.region_embed = nn.ModuleDict()
        self._region = None
        if region_embed:
            e = nn.Embedding(len(MAPA_DKT_REGIONS) + 1, d_model)
            self.region_embed["embed"] = e
            self._region = e
        self.add_module("region_embed", self.region_embed)
        if region_embed:
            nn.init.trunc_normal_(self._region.__val__().weight, std=1e-6)
        self._blocks = [_WithinArrayBlock(d_model, n_heads, mlp_ratio, activation) for _ in range(12)]
        self.blocks = nn.ModuleList()
        for b in self._blocks:
            self.blocks.append(b)
        self.add_module("blocks", self.blocks)
        self._norms = []
        self.norms_block = None
        self.norm_out = None
        if deep_sup:
            nb = nn.ModuleList()
            for _ in _SUP_TAPS:
                n = nn.LayerNorm(d_model, eps=1e-6)
                self._norms.append(n)
                nb.append(n)
            self.norms_block = nb
            self.add_module("norms_block", nb)
        else:
            no = nn.LayerNorm(d_model, eps=1e-6)
            self.norm_out = no
            self.add_module("norm_out", no)
        self.apply(_vit_init)
        with torch.no_grad():
            for i in range(len(self._blocks)):
                s = math.sqrt(2.0 * float(i + 1))
                self._blocks[i].out.weight.div_(s)
                _param_by_name(self._blocks[i].mlp.children()[3], "weight").__val__().div_(s)
    def encode(self, x: Tensor, rid: Tensor, cos: Tensor, sin: Tensor, km: Tensor) -> Tensor:
        if self._region is not None:
            x = x + self._region.__val__()(rid).to(x.dtype)
        lv = List[Tensor]()
        for i in range(len(self._blocks)):
            x = self._blocks[i].block(x, cos, sin, km)
            for t in range(len(_SUP_TAPS)):
                if len(self._norms) > 0 and i + 1 == _SUP_TAPS[t]:
                    lv.append(self._norms[t](x))
        if self.norm_out is not None:
            return self.norm_out.__val__()(x)
        return torch.cat(lv, dim=-1)
    def _get_name(self) -> str:
        return "_Encoder"

def sensor_indices(labels: List[str], regions: Optional[List[int]] = None) -> Tensor:
    seen = Dict[str, int]()
    arrays = List[int]()
    contacts = List[int]()
    for l in labels:
        j = len(l)
        while j > 0 and l[j - 1].isdigit():
            j -= 1
        if j == len(l):
            raise ValueError(f"Contact label {l!r} must end with a contact number.")
        pre = l[:j]
        if pre not in seen:
            seen[pre] = len(seen)
        arrays.append(seen[pre])
        contacts.append(int(l[j:]))
    un = len(MAPA_DKT_REGIONS)
    slots = [un] * len(labels)
    if regions is not None:
        r = regions.__val__()
        if len(r) != len(labels):
            raise ValueError("regions must have one entry per contact.")
        for i in range(len(r)):
            if not (0 <= r[i] <= un):
                raise ValueError(f"region slot {r[i]} out of range.")
            slots[i] = r[i]
    return torch.stack([torch.tensor(arrays, dtype=torch.long), torch.tensor(contacts, dtype=torch.long),
                        torch.tensor(slots, dtype=torch.long)], dim=1)

class MAPA(EEGModuleMixin, nn.Module):
    """MAPA from Tang, Spalding and Cogan (2026). Architecture only.

    Not ported: in-model resampling (input must be at 2048 Hz, or 32 Hz for
    normalization="session"); regions takes integer DKT slots
    (index into MAPA_DKT_REGIONS, or len(...) for unassigned), not names."""
    d_model: int
    pooling: str
    normalization: str
    n_frames: int
    k_full: int
    space_rope: bool
    frontend: _SpectrogramFrontend
    stem: _PerBandStem
    encoder: _MapaEncoder
    default_sensor_indices: Tensor
    gather_idx: Tensor
    key_mask: Tensor
    token_region: Tensor
    scatter_idx: Tensor
    rope_cos: Tensor
    rope_sin: Tensor
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 contact_labels: Optional[List[str]] = None, regions: Optional[List[int]] = None, d_model: int = 384,
                 mlp_ratio: int = 4, region_embed: bool = True, space_rope: bool = True, deep_sup: bool = True,
                 pooling: str = "mean", normalization: str = "window", activation = nn.GELU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if d_model <= 0 or d_model % 64:
            raise ValueError(f"d_model must be a positive multiple of 64, got {d_model}.")
        if pooling not in ("mean", "flatten"):
            raise ValueError(f"pooling must be 'mean' or 'flatten', got {pooling!r}.")
        if normalization not in ("window", "session", "none"):
            raise ValueError(f"normalization must be 'window', 'session' or 'none', got {normalization!r}.")
        exp = 32.0 if normalization == "session" else 2048.0
        known = self._sfreq is not None or (self._input_window_seconds is not None and self._n_times is not None)
        if known and self.sfreq != exp:
            raise NotImplementedError(f"MAPA: in-model resampling to {exp} Hz is not ported.")
        self.d_model = d_model
        self.pooling = pooling
        self.normalization = normalization
        self.space_rope = space_rope
        self.n_frames = _frame_count(self.n_times, normalization == "session")
        self.k_full = sum([self.n_frames // s for s in _BAND_STRIDES])
        self.frontend = _SpectrogramFrontend(normalization)
        self.add_module("frontend", self.frontend)
        self.stem = _PerBandStem(d_model, _BAND_BINS)
        self.add_module("stem", self.stem)
        self.encoder = _MapaEncoder(d_model, d_model // 64, mlp_ratio, region_embed, deep_sup, activation)
        self.add_module("encoder", self.encoder)
        labels = List[str]()
        if contact_labels is not None:
            labels = contact_labels.__val__()
            if len(labels) != self.n_chans:
                raise ValueError("contact_labels must have one entry per channel.")
        elif self._chs_info is not None and len(self.chs_info) > 0:
            labels = [ch["ch_name"] for ch in self.chs_info]
        else:
            labels = [f"A{c + 1}" for c in range(self.n_chans)]
        self.default_sensor_indices = sensor_indices(labels, regions)
        self.register_buffer("default_sensor_indices", self.default_sensor_indices, persistent=False)
        g, km, tr, sc, co, si = self._token_layout(self.default_sensor_indices, self.n_frames)
        self.gather_idx = g
        self.register_buffer("gather_idx", g, persistent=False)
        self.key_mask = km
        self.register_buffer("key_mask", km, persistent=False)
        self.token_region = tr
        self.register_buffer("token_region", tr, persistent=False)
        self.scatter_idx = sc
        self.register_buffer("scatter_idx", sc, persistent=False)
        self.rope_cos = co
        self.register_buffer("rope_cos", co, persistent=False)
        self.rope_sin = si
        self.register_buffer("rope_sin", si, persistent=False)
        fd = d_model * (len(_SUP_TAPS) if deep_sup else 1)
        nf = fd if pooling == "mean" else fd * self.n_chans * self.k_full
        self.final_layer = nn.Linear(nf, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def _token_layout(self, ind: Tensor, n_frames: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        C = ind.shape[0]
        arr = [int(ind[i, 0].item()) for i in range(C)]
        con = [int(ind[i, 1].item()) for i in range(C)]
        reg = [int(ind[i, 2].item()) for i in range(C)]
        ua = sorted(set(arr))
        aoc = [ua.index(a) for a in arr]
        na = len(ua)
        counts = [aoc.count(a) for a in range(na)]
        mc = max(counts)
        gi = [0] * (na * mc)
        va = [False] * (na * mc)
        sc = [0] * C
        fill = [0] * na
        for c in range(C):
            r = aoc[c]
            s = fill[r]
            fill[r] += 1
            gi[r * mc + s] = c
            va[r * mc + s] = True
            sc[c] = r * mc + s
        lat = List[int]()
        for st in _BAND_STRIDES:
            for i in range(n_frames // st):
                lat.append(i * st)
        K = len(lat)
        pairs = 16
        ex = [float(i) / float(pairs) for i in range(pairs)]
        cf = [0.0 if not self.space_rope else 1.0 / (8.0 ** e) for e in ex]
        tf = [1.0 / (64.0 ** e) for e in ex]
        cos = List[float](capacity=na * mc * K * 64)
        sin = List[float](capacity=na * mc * K * 64)
        for a in range(na):
            for s in range(mc):
                cn = float(con[gi[a * mc + s]])
                for t in lat:
                    for i in range(pairs):
                        an32 = torch.tensor([cn], dtype=torch.float32) * torch.tensor([cf[i]], dtype=torch.float32)
                        v = an32.item()
                        cos.append(math.cos(v)); cos.append(math.cos(v))
                        sin.append(math.sin(v)); sin.append(math.sin(v))
                    for i in range(pairs):
                        v = (torch.tensor([float(t)], dtype=torch.float32) * torch.tensor([tf[i]], dtype=torch.float32)).item()
                        cos.append(math.cos(v)); cos.append(math.cos(v))
                        sin.append(math.sin(v)); sin.append(math.sin(v))
        gt = torch.tensor(gi, dtype=torch.long).reshape(na, mc)
        km = torch.tensor(va).reshape(na, mc)
        km = torch.repeat_interleave(km, K, dim=1).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        rg = torch.tensor([reg[gi[j]] for j in range(na * mc)], dtype=torch.long).reshape(na, mc)
        rg = torch.repeat_interleave(rg, K, dim=1)
        ct = torch.tensor(cos, dtype=torch.float32).reshape(na, mc * K, 64).unsqueeze(0).unsqueeze(2)
        stt = torch.tensor(sin, dtype=torch.float32).reshape(na, mc * K, 64).unsqueeze(0).unsqueeze(2)
        return gt, km, rg, torch.tensor(sc, dtype=torch.long), ct, stt

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.n_chans:
            raise ValueError(f"MAPA expects {self.n_chans} channels, got {x.shape[1]}.")
        tok = self.stem.stem(self.frontend.bands(x))
        na, mc = self.gather_idx.shape[0], self.gather_idx.shape[1]
        packed = tok.index_select(1, self.gather_idx.reshape(-1)).reshape(tok.shape[0], na, mc, tok.shape[2], -1).flatten(2, 3)
        enc = self.encoder.encode(packed, self.token_region, self.rope_cos.to(x.dtype), self.rope_sin.to(x.dtype), self.key_mask)
        enc = enc.unflatten(2, [-1, self.k_full]).flatten(1, 2).index_select(1, self.scatter_idx)
        f = enc.mean(dim=[1, 2]) if self.pooling == "mean" else enc.flatten(1)
        return self.final_layer(f)

    def _get_name(self) -> str:
        return "MAPA"
