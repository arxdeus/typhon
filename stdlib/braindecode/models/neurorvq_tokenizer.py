# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.models.labram import _LabramAttention
from braindecode.models.neurorvq import _preset, _MultiScaleTemporalConv
from braindecode.functional.initialization import _param_by_name
from braindecode.functional.patches import _check_on_non_divisible, patch_count, pad_or_crop_to_patches
from braindecode.modules.mlp import MLP
from braindecode.modules.timm import PreNormBlock

def _n_quantizers(modality: str) -> int:
    return 16 if modality == "emg" else 8

class _EMAEmbedding(nn.Module):
    weight: Tensor
    cluster_size: Tensor
    embed_avg: Tensor
    initted: Tensor
    def __init__(self, n_codes: int, code_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_codes, code_dim), requires_grad=False)
        self.register_parameter("weight", self.weight)
        self.cluster_size = nn.Parameter(torch.zeros(n_codes), requires_grad=False)
        self.register_parameter("cluster_size", self.cluster_size)
        self.embed_avg = nn.Parameter(torch.zeros(n_codes, code_dim), requires_grad=False)
        self.register_parameter("embed_avg", self.embed_avg)
        self.initted = torch.tensor([0.0])
        self.register_buffer("initted", self.initted)
    def initialize(self, v: Tensor):
        # spherical k-means on the first batch (braindecode _EMAEmbedding.initialize)
        if self.initted.item() != 0.0:
            return
        with torch.no_grad():
            nc, ns = self.weight.shape[0], v.shape[0]
            idx = torch.randperm(ns)[:nc] if ns >= nc else torch.randint(0, ns, [nc])
            means = v.index_select(0, idx)
            counts = torch.zeros(nc, dtype=torch.long)
            for _ in range(10):
                a = torch.matmul(v, means.t()).argmax(dim=-1)
                counts = torch.bincount(a, minlength=nc)
                sums = torch.zeros_like(means).index_add(0, a, v)
                up = F.normalize(sums / torch.clamp(counts, min=1).unsqueeze(-1).to(v.dtype), dim=-1)
                means = torch.where((counts == 0).unsqueeze(1), means, up)
            self.weight.copy_(means)
            self.cluster_size.copy_(counts.to(self.cluster_size.dtype))
            self.initted.fill_(1.0)
    def _get_name(self) -> str:
        return "_EMAEmbedding"

class _EMAVectorQuantizer(nn.Module):
    decay: float
    statistic_code_usage: bool
    embedding: _EMAEmbedding
    cluster_size: Tensor
    def __init__(self, n_codes: int, code_dim: int, statistic_code_usage: bool = False):
        super().__init__()
        self.decay = 0.99
        self.statistic_code_usage = statistic_code_usage
        self.embedding = _EMAEmbedding(n_codes, code_dim)
        self.add_module("embedding", self.embedding)
        self.cluster_size = torch.zeros(n_codes)
        self.register_buffer("cluster_size", self.cluster_size)
    def _indices(self, v: Tensor) -> Tensor:
        self.embedding.initialize(v)
        w = self.embedding.weight
        with torch.no_grad():
            d = v.pow(2.0).sum(dim=1, keepdim=True) + w.pow(2.0).sum(dim=1) - 2.0 * torch.matmul(v, w.t())
            return d.argmin(dim=1)
    def encode(self, z: Tensor) -> Tensor:
        z = F.normalize(z.permute(0, 2, 3, 1), dim=-1)
        return self._indices(z.reshape(-1, z.shape[-1]))
    def quantize(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        z = F.normalize(z.permute(0, 2, 3, 1), dim=-1)
        v = z.reshape(-1, z.shape[-1])
        idx = self._indices(v)
        q = F.embedding(idx, self.embedding.weight).view(z.shape)
        if self.training or self.statistic_code_usage:
            with torch.no_grad():
                w = self.embedding.weight
                vd = v.detach()
                ones = torch.ones(idx.shape[0], dtype=vd.dtype)
                cnt = torch.zeros(w.shape[0], dtype=vd.dtype).index_add(0, idx, ones)
                self.cluster_size.mul_(self.decay).add_(cnt * (1.0 - self.decay))
                if self.training:
                    sc = cnt.masked_fill(cnt == 0.0, 1.0)
                    es = torch.zeros(w.shape[0], w.shape[1], dtype=vd.dtype).index_add(0, idx, vd)
                    mn = F.normalize(es / sc.unsqueeze(1), dim=-1)
                    mn = torch.where((cnt == 0.0).unsqueeze(1), w, mn)
                    w.mul_(self.decay).add_(mn * (1.0 - self.decay))
                    w.copy_(F.normalize(w, dim=-1))
        q = z + (q - z).detach()
        return q.permute(0, 3, 1, 2).contiguous(), idx
    def _get_name(self) -> str:
        return "_EMAVectorQuantizer"

class _RVQ(nn.Module):
    layers: nn.ModuleList
    _layers: List[_EMAVectorQuantizer]
    def __init__(self, n_quantizers: int, n_codes: int, code_dim: int, statistic_code_usage: bool):
        super().__init__()
        self._layers = [_EMAVectorQuantizer(n_codes, code_dim, statistic_code_usage) for _ in range(n_quantizers)]
        self.layers = nn.ModuleList()
        for l in self._layers:
            self.layers.append(l)
        self.add_module("layers", self.layers)
    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = torch.zeros_like(x)
        res = x
        codes = List[Tensor]()
        for l in self._layers:
            q, i = l.quantize(res)
            res = res - q
            out = out + q
            codes.append(i)
        return out, torch.stack(codes)
    def encode(self, x: Tensor) -> Tensor:
        res = x
        codes = List[Tensor]()
        for l in self._layers:
            i = l.encode(res)
            q = F.embedding(i, l.embedding.weight).view(x.shape[0], x.shape[2], x.shape[3], x.shape[1])
            res = res - q.permute(0, 3, 1, 2).contiguous()
            codes.append(i)
        return torch.stack(codes)
    def _get_name(self) -> str:
        return "_ResidualVectorQuantizer"

class _PatchProjection(nn.Module):
    proj: nn.Conv2d
    def __init__(self, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        self.add_module("proj", self.proj)
    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
    def _get_name(self) -> str:
        return "_PatchProjection"

class _BranchTransformer(nn.Module):
    cls_token: Tensor
    pos_embed: Tensor
    time_embed: Tensor
    pos_drop: nn.Dropout
    blocks: nn.ModuleList
    _blocks: List[PreNormBlock[_LabramAttention, MLP]]
    _fc: List[nn.LayerNorm]
    def __init__(self, n_channels: int, max_patches: int, embed_dim: int, depth: int, num_heads: int, drop_prob: float,
                 attn_drop_rate: float, drop_path_rate: float, init_values: float):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_parameter("cls_token", self.cls_token)
        self.pos_embed = nn.Parameter(torch.zeros(n_channels + 1, embed_dim))
        self.register_parameter("pos_embed", self.pos_embed)
        self.time_embed = nn.Parameter(torch.zeros(max_patches, embed_dim))
        self.register_parameter("time_embed", self.time_embed)
        self.pos_drop = nn.Dropout(drop_prob)
        self._blocks = []
        self._fc = []
        self.blocks = nn.ModuleList()
    def build(self, embed_dim: int, depth: int, num_heads: int, drop_prob: float, attn_drop_rate: float,
              drop_path_rate: float, init_values: float):
        # called after the patch-embedding sub-modules are registered (torch order)
        self.add_module("pos_drop", self.pos_drop)
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist()
        ls: Optional[float] = None
        if init_values > 0:
            ls = init_values
        for i in range(depth):
            b = PreNormBlock(embed_dim, _LabramAttention(embed_dim, num_heads, True, True, None, attn_drop_rate, drop_prob),
                             MLP(embed_dim, [int(float(embed_dim) * 4.0)], embed_dim, drop=drop_prob),
                             drop_path=float(dpr[i]), layer_scale=ls)
            self._blocks.append(b)
            self.blocks.append(b)
        self.add_module("blocks", self.blocks)
        for i in range(1, 5):
            n = nn.LayerNorm(embed_dim)
            self._fc.append(n)
            self.add_module(f"fc_norm_{i}", n)
    def embeddings(self, time_idx: Tensor, spatial_idx: Tensor) -> Tuple[Tensor, Tensor]:
        si = torch.cat([torch.zeros(1, dtype=torch.long), spatial_idx])
        return self.pos_embed.index_select(0, si).unsqueeze(0), self.time_embed.index_select(0, time_idx).unsqueeze(0)
    def run(self, scales: List[Tensor], spatial: Tensor, temporal: Tensor) -> List[Tensor]:
        outs = List[Tensor]()
        for i in range(len(scales)):
            t = scales[i]
            t = torch.cat([self.cls_token.expand(t.shape[0], -1, -1), t], dim=1) + spatial
            t = torch.cat([t[:, :1], t[:, 1:] + temporal], dim=1)
            t = self.pos_drop(t)
            for b in self._blocks:
                t = b(t)
            outs.append(self._fc[i](t[:, 1:]))
        return outs
    def _get_name(self) -> str:
        return "_BranchTransformer"

def _task_init(m):
    if m._get_name() == "Linear":
        nn.init.trunc_normal_(_param_by_name(m, "weight").__val__(), std=0.02)
        nn.init.zeros_(_param_by_name(m, "bias").__val__())

def _standardize(x: Tensor) -> Tensor:
    m = x.mean(dim=[1, 2, 3], keepdim=True)
    s = torch.sqrt(torch.clamp(x.var(dim=[1, 2, 3], keepdim=True), min=1e-8))
    return (x - m) / s

class NeuroRVQTokenizer(EEGModuleMixin, nn.Module):
    """NeuroRVQ tokenizer from Barmpas et al (2025). Architecture only.

    ``forward`` returns the standardized reconstruction (braindecode's
    second output); ``targets_and_reconstruction`` returns both. No in-model
    resampling (input sfreq must match the modality preset)."""
    num_patches: int
    on_non_divisible: str
    patch_size: int
    max_patches: int
    code_dim: int
    num_quantizers: int
    channel_names: List[str]
    channel_keep_ix: Tensor
    spatial_embedding_ix: Tensor
    _drops_channels: bool
    encoder: _BranchTransformer
    _enc_embed: _MultiScaleTemporalConv
    decoder: _BranchTransformer
    _dec_embed: List[_PatchProjection]
    _quant: List[_RVQ]
    _task: List[nn.Sequential]
    decode_task_layer_amplitude: nn.Sequential
    decode_task_layer_angle_sin: nn.Sequential
    decode_task_layer_angle_cos: nn.Sequential

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 channel_names: Optional[List[str]] = None, modality: str = "eeg", patch_size: Optional[int] = None,
                 on_non_divisible: str = "pad", on_missing: str = "error", max_patches: Optional[int] = None,
                 out_chans: int = 8, encoder_depth: int = 12, decoder_depth: int = 3, num_heads: int = 10,
                 n_code: int = 8192, code_dim: int = 128, num_quantizers: Optional[int] = None, drop_prob: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0, init_values: float = 0.0,
                 activation = nn.GELU, statistic_code_usage: bool = False):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        pr = _preset(modality)
        if self._sfreq is not None and self.sfreq != pr.sfreq:
            raise ValueError(f"NeuroRVQTokenizer expects {pr.sfreq} Hz for modality {modality!r}; resample first.")
        ps = patch_size.__val__() if patch_size is not None else pr.patch_size
        mp = max_patches.__val__() if max_patches is not None else pr.max_patches
        nq = num_quantizers.__val__() if num_quantizers is not None else _n_quantizers(modality)
        _check_on_non_divisible(on_non_divisible)
        self.num_patches = patch_count(self.n_times, ps, on_non_divisible)
        if self.num_patches > mp:
            raise ValueError(f"n_times supports at most {mp} patches (got {self.num_patches}).")
        self.on_non_divisible = on_non_divisible
        E = out_chans * (ps // 8)
        if E % num_heads:
            raise ValueError("The derived embedding width must be divisible by num_heads.")
        if on_missing not in ("error", "drop"):
            raise ValueError(f'on_missing must be "error" or "drop", got {on_missing!r}.')
        names = List[str]()
        if channel_names is not None:
            names = channel_names.__val__()
        elif self._chs_info is not None:
            names = [ch["ch_name"] for ch in self.chs_info]
        else:
            names = pr.channels[:self.n_chans]
        if len(names) != self.n_chans:
            raise ValueError("channel_names must have one entry per input channel.")
        names = [n.strip().lower() for n in names]
        keep = [i for i in range(len(names)) if names[i] in pr.channels]
        if len(keep) != len(names) and (on_missing != "drop" or len(keep) == 0):
            raise ValueError('Some channel names are not in the NeuroRVQ vocabulary; pass on_missing="drop".')
        self._drops_channels = len(keep) != len(names)
        self.channel_names = [names[i] for i in keep]
        self.channel_keep_ix = torch.tensor(keep, dtype=torch.long)
        self.register_buffer("channel_keep_ix", self.channel_keep_ix, persistent=False)
        self.patch_size = ps
        self.max_patches = mp
        self.code_dim = code_dim
        self.num_quantizers = nq
        self.spatial_embedding_ix = torch.tensor([pr.channels.index(n) for n in self.channel_names], dtype=torch.long)
        self.register_buffer("spatial_embedding_ix", self.spatial_embedding_ix, persistent=False)
        nch = len(pr.channels)
        self.encoder = _BranchTransformer(nch, mp, E, encoder_depth, num_heads, drop_prob, attn_drop_rate, drop_path_rate, init_values)
        self._enc_embed = _MultiScaleTemporalConv(out_chans, activation, pr.kernels1, pr.kernels2)
        self.encoder.add_module("patch_embed", self._enc_embed)
        self.encoder.build(E, encoder_depth, num_heads, drop_prob, attn_drop_rate, drop_path_rate, init_values)
        self.add_module("encoder", self.encoder)
        self.decoder = _BranchTransformer(nch, mp, E, decoder_depth, num_heads, drop_prob, attn_drop_rate, drop_path_rate, init_values)
        self._dec_embed = []
        for i in range(1, 5):
            pp = _PatchProjection(code_dim, E)
            self._dec_embed.append(pp)
            self.decoder.add_module(f"patch_embed_{i}", pp)
        self.decoder.build(E, decoder_depth, num_heads, drop_prob, attn_drop_rate, drop_path_rate, init_values)
        self.add_module("decoder", self.decoder)
        self._quant = []
        self._task = []
        for i in range(1, 5):
            q = _RVQ(nq, n_code, code_dim, statistic_code_usage)
            self._quant.append(q)
            self.add_module(f"quantize_{i}", q)
            t = nn.Sequential(nn.Linear(E, E), nn.Tanh(), nn.Linear(E, code_dim))
            self._task.append(t)
            self.add_module(f"encode_task_layer_{i}", t)
        self.decode_task_layer_amplitude = nn.Sequential(nn.Linear(4 * E, E), nn.GELU(), nn.Linear(E, ps))
        self.add_module("decode_task_layer_amplitude", self.decode_task_layer_amplitude)
        self.decode_task_layer_angle_sin = nn.Sequential(nn.Linear(4 * E, E), nn.Tanh(), nn.Linear(E, ps), nn.Tanh())
        self.add_module("decode_task_layer_angle_sin", self.decode_task_layer_angle_sin)
        self.decode_task_layer_angle_cos = nn.Sequential(nn.Linear(4 * E, E), nn.Tanh(), nn.Linear(E, ps), nn.Tanh())
        self.add_module("decode_task_layer_angle_cos", self.decode_task_layer_angle_cos)
        self.add_module("final_layer", nn.Identity())
        for t in self._task:
            t.apply(_task_init)
        self.decode_task_layer_amplitude.apply(_task_init)
        self.decode_task_layer_angle_sin.apply(_task_init)
        self.decode_task_layer_angle_cos.apply(_task_init)

    def _indices(self) -> Tuple[Tensor, Tensor]:
        n = self.spatial_embedding_ix.numel()
        t = torch.arange(self.max_patches - self.num_patches, self.max_patches).repeat(n)
        s = torch.repeat_interleave(self.spatial_embedding_ix, self.num_patches)
        return t, s

    def _patches(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_chans or x.shape[2] != self.n_times:
            raise ValueError(f"Expected input shape (batch, {self.n_chans}, {self.n_times}), got {x.shape}.")
        if self._drops_channels:
            x = x.index_select(1, self.channel_keep_ix)
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        return x.reshape(x.shape[0], x.shape[1], -1, self.patch_size)

    def _branch_latents(self, p: Tensor, t: Tensor, s: Tensor) -> List[Tensor]:
        B, C, P = p.shape[0], p.shape[1], p.shape[2]
        sp, tm = self.encoder.embeddings(t, s)
        fs = self.encoder.run(self._enc_embed.branches(p), sp, tm)
        return [self._task[i](fs[i]).reshape(B, C, P, self.code_dim).permute(0, 3, 1, 2).contiguous() for i in range(4)]

    def tokenize(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            p = self._patches(x)
            t, s = self._indices()
            lat = self._branch_latents(p, t, s)
            return torch.stack([self._quant[i].encode(lat[i]).reshape(self.num_quantizers, x.shape[0], -1) for i in range(4)])

    def targets_and_reconstruction(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        p = self._patches(x)
        t, s = self._indices()
        amp = torch.log1p(torch.abs(torch.fft.fft(p, dim=-1))).to(p.dtype)
        am = amp.mean(dim=[1, 2, 3], keepdim=True)
        asd = torch.sqrt(torch.clamp(amp.var(dim=[1, 2, 3], keepdim=True), min=1e-8))
        lat = self._branch_latents(p, t, s)
        qs = List[Tensor]()
        for i in range(4):
            q, _ = self._quant[i].quantize(lat[i])
            qs.append(q)
        toks = [self._dec_embed[i](qs[i]) for i in range(4)]
        sp, tm = self.decoder.embeddings(t, s)
        dec = torch.cat(self.decoder.run(toks, sp, tm), dim=-1)
        ra = self.decode_task_layer_amplitude(dec)
        rs = self.decode_task_layer_angle_sin(dec).reshape(p.shape)
        rc = self.decode_task_layer_angle_cos(dec).reshape(p.shape)
        ra = torch.expm1(ra.reshape(p.shape) * asd + am)
        rec = torch.real(torch.fft.ifft(torch.complex(ra * rc, ra * rs), dim=-1)).to(p.dtype)
        B = x.shape[0]
        ts = _standardize(p)
        rstd = _standardize(rec.reshape(B, -1, 1, self.patch_size).contiguous())
        n = p.shape[1] * self.num_patches
        return ts.reshape(B, n, -1), rstd.reshape(B, n, -1)

    def forward(self, x: Tensor) -> Tensor:
        _, r = self.targets_and_reconstruction(x)
        return r

    def _get_name(self) -> str:
        return "NeuroRVQTokenizer"
