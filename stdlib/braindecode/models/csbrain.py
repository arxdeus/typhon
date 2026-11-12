# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional.initialization import _param_by_name
from braindecode.functional.patches import _check_on_non_divisible, patch_count
from braindecode.modules import FeedForwardBlock
from braindecode.models.cbramod import _CBraModPatchEmbedding

REGION_FRONTAL = 0
REGION_PARIETAL = 1
REGION_TEMPORAL = 2
REGION_OCCIPITAL = 3
REGION_CENTRAL = 4
_REGION_PREFIXES = [("FP", 0), ("AF", 0), ("FC", 0), ("FT", 2), ("TP", 2), ("CP", 4), ("PO", 3), ("F", 0), ("C", 4),
                    ("T", 2), ("P", 1), ("O", 3), ("I", 3)]

def region_of_electrode(ch_name: str, default: int = REGION_CENTRAL) -> int:
    letters = "".join([c for c in ch_name.upper() if c.isalpha()])
    for prefix, region in _REGION_PREFIXES:
        if letters.startswith(prefix):
            return region
    return default

def group_regions(regions: List[int]) -> Tuple[List[int], List[int]]:
    idx = List[int]()
    for r in sorted(set(regions)):
        for i in range(len(regions)):
            if regions[i] == r:
                idx.append(i)
    return [regions[i] for i in idx], idx

def order_regions(regions: List[int], channel_order: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
    if channel_order is None:
        return group_regions(regions)
    idx = channel_order.__val__()
    if sorted(idx) != list(range(len(regions))):
        raise ValueError(f"channel_order must be a permutation of the {len(regions)} input channels, got {idx}.")
    ordered = [regions[i] for i in idx]
    if ordered != sorted(regions):
        raise ValueError(f"channel_order must group the channels by ascending region id, got regions {ordered} in that order.")
    return ordered, idx

def make_area_config(brain_regions: List[int]) -> List[Tuple[str, int, int]]:
    # (key, start, stop) per contiguous region run, in order
    out = List[Tuple[str, int, int]]()
    start = 0
    for i in range(1, len(brain_regions) + 1):
        if i == len(brain_regions) or brain_regions[i] != brain_regions[i - 1]:
            out.append((f"region_{brain_regions[i - 1]}", start, i))
            start = i
    return out

def build_region_attention_mask(area: List[Tuple[str, int, int]], n_channels: int) -> Tensor:
    vals = [-math.inf] * (n_channels * n_channels)
    groups = [list(range(s, e)) for _, s, e in area]
    ng = max([len(g) for g in groups])
    for g in range(ng):
        grp = [ind[g % len(ind)] for ind in groups if len(ind) > 0]
        for i in grp:
            for j in grp:
                vals[i * n_channels + j] = 0.0
    return torch.tensor(vals, dtype=torch.float32).reshape(n_channels, n_channels)

class _CrossScaleTemporalEmbedding(nn.Module):
    convs: nn.ModuleList
    def __init__(self, dim_in: int, dim_out: int, kernel_sizes: List[int] = [1, 3, 5]):
        super().__init__()
        ks = sorted(kernel_sizes)
        ds = [int(float(dim_out) / float(2 ** i)) for i in range(1, len(ks))]
        ds.append(dim_out - sum(ds))
        self.convs = nn.ModuleList()
        for i in range(len(ks)):
            self.convs.append(nn.Conv2d(dim_in, ds[i], kernel_size=(ks[i], 1), stride=(1, 1), padding=((ks[i] - 1) // 2, 0)))
        self.add_module("convs", self.convs)
    def forward(self, x: Tensor) -> Tensor:
        b, c, t, d = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b * c, d, t, 1)
        return torch.cat([m(x) for m in self.convs.children()], dim=1).view(b, c, t, -1)
    def _get_name(self) -> str:
        return "_CrossScaleTemporalEmbedding"

class _RegionEmbedding(nn.Module):
    dim_out: int
    region_bounds: List[Tuple[int, int]]
    tap_starts: List[int]
    region_blocks: nn.ModuleDict
    _convs: List[List[nn.Conv2d]]
    chan_region: Tensor
    tap_select: Tensor

    def __init__(self, dim_in: int, dim_out: int, area: List[Tuple[str, int, int]], kernel_sizes: List[int] = [1, 3, 5]):
        super().__init__()
        self.dim_out = dim_out
        self.region_bounds = [(s, e) for _, s, e in area]
        ds = [dim_out // (2 ** (i + 1)) for i in range(len(kernel_sizes) - 1)]
        ds.append(dim_out - sum(ds))
        self.region_blocks = nn.ModuleDict()
        self._convs = []
        for key, _, _ in area:
            ml = nn.ModuleList()
            cl = List[nn.Conv2d]()
            for i in range(len(kernel_sizes)):
                c = nn.Conv2d(dim_in, ds[i], kernel_size=(kernel_sizes[i], 1), padding=(0, 0))
                ml.append(c)
                cl.append(c)
            self.region_blocks[key] = ml
            self._convs.append(cl)
        self.add_module("region_blocks", self.region_blocks)
        half = (max(kernel_sizes) - 1) // 2
        nt = 2 * half + 1
        self.tap_starts = [half - (k - 1) // 2 for k in kernel_sizes]
        nc = self.region_bounds[-1][1] if len(self.region_bounds) > 0 else 0
        cr = List[int]()
        ts = [0.0] * (nc * nt * nc)
        for i in range(len(self.region_bounds)):
            s, e = self.region_bounds[i]
            n = e - s
            for p in range(n):
                cr.append(i)
                ds2 = list(range(-half, half + 1)) if n > 1 else [0]
                for d in ds2:
                    row = (s + p) * nt + d + half
                    ts[row * nc + s + (p + d) % n] = 1.0
        self.chan_region = torch.tensor(cr, dtype=torch.long)
        self.register_buffer("chan_region", self.chan_region, False)
        self.tap_select = torch.tensor(ts, dtype=torch.float32).reshape(nc * nt, nc)
        self.register_buffer("tap_select", self.tap_select, False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, D = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        if len(self.region_bounds) == 0:
            return torch.zeros(B, C, T, self.dim_out, dtype=x.dtype)
        taps = torch.matmul(self.tap_select, x.reshape(B, C, T * D)).reshape(B, C, -1, T, D)
        outs = List[Tensor]()
        for j in range(len(self.tap_starts)):
            st = self.tap_starts[j]
            w = torch.stack([cl[j].weight for cl in self._convs]).index_select(0, self.chan_region)
            k = w.shape[3]
            w = w.squeeze(-1).permute(0, 3, 2, 1).reshape(C, k * D, -1)
            t = taps[:, :, st:st + k].permute(1, 0, 3, 2, 4)
            b = torch.stack([cl[j].bias.__val__() for cl in self._convs]).index_select(0, self.chan_region)
            outs.append(torch.bmm(t.reshape(C, B * T, k * D), w) + b.unsqueeze(1))
        out = torch.cat(outs, dim=-1).view(C, B, T, self.dim_out)
        return out.permute(1, 0, 2, 3).contiguous()

    def _get_name(self) -> str:
        return "_RegionEmbedding"

class _CSBrainEncoderLayer(nn.Module):
    d_model: int
    nhead: int
    inter_window_attn: nn.MultiheadAttention
    inter_region_attn: nn.MultiheadAttention
    global_fc: nn.Linear
    ff_block: FeedForwardBlock
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    norm3: nn.LayerNorm
    dropout1: nn.Dropout
    dropout2: nn.Dropout
    dropout3: nn.Dropout
    region_bounds: List[Tuple[int, int]]
    region_attn_mask: Optional[Tensor]
    chan_region: Tensor
    region_sizes: Tensor
    _avg: Optional[Tensor]

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 800, dropout: float = 0.1, activation = nn.GELU,
                 area: List[Tuple[str, int, int]] = []):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.inter_window_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.add_module("inter_window_attn", self.inter_window_attn)
        self.inter_region_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.add_module("inter_region_attn", self.inter_region_attn)
        self.global_fc = nn.Linear(d_model, d_model)
        self.add_module("global_fc", self.global_fc)
        self.ff_block = FeedForwardBlock(d_model, 1, dropout, activation, hidden_features=dim_feedforward)
        self.add_module("ff_block", self.ff_block)
        self.norm1 = nn.LayerNorm(d_model)
        self.add_module("norm1", self.norm1)
        self.norm2 = nn.LayerNorm(d_model)
        self.add_module("norm2", self.norm2)
        self.norm3 = nn.LayerNorm(d_model)
        self.add_module("norm3", self.norm3)
        self.dropout1 = nn.Dropout(dropout)
        self.add_module("dropout1", self.dropout1)
        self.dropout2 = nn.Dropout(dropout)
        self.add_module("dropout2", self.dropout2)
        self.dropout3 = nn.Dropout(dropout)
        self.add_module("dropout3", self.dropout3)
        self.region_bounds = [(s, e) for _, s, e in area]
        self.region_attn_mask = None
        self._avg = None
        sizes = [e - s for s, e in self.region_bounds]
        cr = List[int]()
        for i in range(len(sizes)):
            for _ in range(sizes[i]):
                cr.append(i)
        if len(area) > 0:
            n = sum(sizes)
            m = build_region_attention_mask(area, n)
            self.region_attn_mask = m
            self.register_buffer("region_attn_mask", m)
            # per-channel region mean as a matrix: avg[c, c'] = 1/|region| if same region
            av = [0.0] * (n * n)
            for c in range(n):
                for c2 in range(n):
                    if cr[c] == cr[c2]:
                        av[c * n + c2] = 1.0 / float(sizes[cr[c]])
            self._avg = torch.tensor(av, dtype=torch.float32).reshape(n, n)
        self.chan_region = torch.tensor(cr, dtype=torch.long)
        self.register_buffer("chan_region", self.chan_region, False)
        self.region_sizes = torch.tensor(sizes, dtype=torch.long).unsqueeze(1)
        self.register_buffer("region_sizes", self.region_sizes, False)

    def forward(self, src: Tensor) -> Tensor:
        x = src
        x = x + self.dropout1(self._inter_window_attention(self.norm1(x)))
        x = x + self.dropout2(self._inter_region_attention(self.norm2(x)))
        return x + self.dropout3(self._ff(self.norm3(x)))

    def _inter_window_attention(self, x: Tensor) -> Tensor:
        B, C, T, Fe = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        ws = min(T, 5)
        T0 = T
        if T % ws != 0:
            pl = ws - T % ws
            x = F.pad(x, [0, 0, 0, pl])
            T += pl
        nw = T // ws
        x = x.view(B, C, nw, ws, Fe).permute(0, 3, 1, 2, 4).reshape(B * ws * C, nw, Fe)
        y, _ = self.inter_window_attn.attend(x, x, x, need_weights=False)
        y = y.reshape(B, ws, C, nw, Fe).permute(0, 2, 3, 1, 4).reshape(B, C, T, Fe)
        if T != T0:
            y = y[:, :, :T0, :]
        return y

    def _inter_region_attention(self, x: Tensor) -> Tensor:
        B, C, T, Fe = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        xf = x.permute(0, 2, 1, 3).reshape(B * T, C, Fe)
        g = torch.zeros_like(xf)
        if self._avg is not None:
            g = torch.matmul(self._avg.__val__().to(xf.dtype), xf)
        xe = xf + self.global_fc(g)
        y, _ = self.inter_region_attn.attend(xe, xe, xe, attn_mask=self.region_attn_mask, need_weights=False)
        return y.reshape(B, T, C, Fe).permute(0, 2, 1, 3)

    def _ff(self, x: Tensor) -> Tensor:
        B, C, T, Fe = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        y = self.ff_block(x.permute(0, 2, 1, 3).reshape(B * T, C, Fe))
        return y.reshape(B, T, C, Fe).permute(0, 2, 1, 3)

    def _get_name(self) -> str:
        return "_CSBrainEncoderLayer"

def _reset_parameters(m):
    # nn.Linear / nn.Conv2d reset_parameters through the parameter handles
    w = _param_by_name(m, "weight").__val__()
    nn.init.kaiming_uniform_(w, a=math.sqrt(5.0))
    b = _param_by_name(m, "bias")
    if b is not None:
        fan_in = w.numel() // w.shape[0]
        bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
        nn.init.uniform_(b.__val__(), -bound, bound)

def _csbrain_init(m):
    if m._get_name() == "Linear":
        w = _param_by_name(m, "weight")
        if w is not None:
            nn.init.kaiming_normal_(w.__val__(), mode="fan_out", nonlinearity="relu")

class CSBrain(EEGModuleMixin, nn.Module):
    """CSBrain from Zhou et al (2025). Architecture only (no Hub weights).

    The patch embedding is CBraMod's with the fixed reference convolution
    stack (d_model follows from patch_size). Regions come from
    ``brain_regions`` or from the ``chs_info`` channel names."""
    patch_size: int
    emb_dim: int
    d_model: int
    on_non_divisible: str
    head_drop_prob: float
    head_hidden_dim: Optional[int]
    patch_embedding: _CBraModPatchEmbedding
    sorted_indices: Optional[Tensor]
    area_config: List[Tuple[str, int, int]]
    temporal_embed: _CrossScaleTemporalEmbedding
    region_embed: _RegionEmbedding
    encoder: nn.ModuleList
    _layers: List[_CSBrainEncoderLayer]
    proj_out: nn.Sequential
    final_layer: nn.Module

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, patch_size: int = 200,
                 on_non_divisible: str = "pad", dim_feedforward: int = 800, n_layer: int = 12, nhead: int = 8,
                 activation = nn.GELU, emb_dim: int = 200, temporal_kernel_sizes: List[int] = [1, 3, 5],
                 drop_prob: float = 0.1, head_drop_prob: Optional[float] = None,
                 brain_regions: Optional[List[int]] = None, channel_order: Optional[List[int]] = None,
                 head_hidden_dim: Optional[int] = None, return_encoder_output: bool = False):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        _check_on_non_divisible(on_non_divisible)
        self.patch_size = patch_size
        self.on_non_divisible = on_non_divisible
        patch_count(self.n_times, patch_size, on_non_divisible)
        # created under torch.random.fork_rng, then re-initialised
        st = torch.get_rng_state()
        self.patch_embedding = _CBraModPatchEmbedding(patch_size, [(25, 49, 25, 24, (5, 25)), (25, 3, 1, 1, (5, 25)),
                                                                   (25, 3, 1, 1, (5, 25))], drop_prob)
        torch.set_rng_state(st)
        for l in self.patch_embedding.proj_in.children() + self.patch_embedding.positional_encoding.children() + \
                self.patch_embedding.spectral_proj.children():
            if l._get_name() in ("Conv2d", "Linear"):
                _reset_parameters(l)
        self.add_module("patch_embedding", self.patch_embedding)
        d = self.patch_embedding.d_model
        self.d_model = d
        if brain_regions is not None and self._n_chans is not None and len(brain_regions.__val__()) != self.n_chans:
            raise ValueError(f"brain_regions has {len(brain_regions.__val__())} entries but n_chans={self.n_chans}.")
        self.sorted_indices = None
        self.area_config = []
        has = False
        ordered = List[int]()
        idx = List[int]()
        if brain_regions is not None:
            ordered, idx = order_regions(brain_regions.__val__(), channel_order)
            has = True
        elif self._chs_info is not None and len(self.chs_info) > 0:
            ordered, idx = order_regions([region_of_electrode(ch.get("ch_name", "")) for ch in self.chs_info], channel_order)
            has = True
        elif channel_order is not None:
            raise ValueError("channel_order needs brain_regions or chs_info.")
        if has:
            si = torch.tensor(idx, dtype=torch.long)
            self.sorted_indices = si
            self.register_buffer("sorted_indices", si)
            self.area_config = make_area_config(ordered)
        self.temporal_embed = _CrossScaleTemporalEmbedding(d, d, temporal_kernel_sizes)
        self.add_module("temporal_embed", self.temporal_embed)
        self.region_embed = _RegionEmbedding(d, d, self.area_config)
        self.add_module("region_embed", self.region_embed)
        s0 = torch.get_rng_state()
        self._layers = [_CSBrainEncoderLayer(d, nhead, dim_feedforward, drop_prob, activation, self.area_config)]
        s1 = torch.get_rng_state()
        for _ in range(1, n_layer):
            torch.set_rng_state(s0)
            self._layers.append(_CSBrainEncoderLayer(d, nhead, dim_feedforward, drop_prob, activation, self.area_config))
        torch.set_rng_state(s1)
        self.encoder = nn.ModuleList()
        for l in self._layers:
            self.encoder.append(l)
        self.add_module("encoder", self.encoder)
        self.proj_out = nn.Sequential(nn.Linear(d, emb_dim))
        self.add_module("proj_out", self.proj_out)
        self.emb_dim = emb_dim
        self.head_drop_prob = drop_prob if head_drop_prob is None else head_drop_prob.__val__()
        self.head_hidden_dim = head_hidden_dim
        self.apply(_csbrain_init)
        fl: nn.Module = nn.Identity()
        if not return_encoder_output:
            npch = patch_count(self.n_times, patch_size, on_non_divisible)
            hid = head_hidden_dim.__val__() if head_hidden_dim is not None else npch * emb_dim
            hp = self.head_drop_prob
            fl = nn.Sequential(nn.Flatten(), nn.Linear(self.n_chans * npch * emb_dim, hid), nn.ELU(), nn.Dropout(hp),
                               nn.Linear(hid, emb_dim), nn.ELU(), nn.Dropout(hp), nn.Linear(emb_dim, self.n_outputs))
        self.final_layer = fl
        self.add_module("final_layer", fl)

    def _tokenize(self, x: Tensor) -> Tensor:
        t = x.shape[-1]
        n = patch_count(t, self.patch_size, self.on_non_divisible)
        target = n * self.patch_size
        if target > t:
            x = F.pad(x, [0, target - t])
        elif target < t:
            x = x[:, :, :target]
        return x.reshape(x.shape[0], x.shape[1], -1, self.patch_size)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self._tokenize(x)
        if self.sorted_indices is not None:
            x = x.index_select(1, self.sorted_indices.__val__())
        emb = self.patch_embedding(x)
        for l in self._layers:
            emb = self.temporal_embed(emb) + emb
            emb = self.region_embed(emb) + emb
            emb = l(emb)
        return self.proj_out(emb)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x))

    def _get_name(self) -> str:
        return "CSBrain"
