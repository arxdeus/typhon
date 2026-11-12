# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.models.labram import _LabramAttention
from braindecode.functional.initialization import _param_by_name
from braindecode.functional.patches import _check_on_non_divisible, patch_count, pad_or_crop_to_patches
from braindecode.modules.mlp import MLP
from braindecode.modules.timm import PreNormBlock

NEURORVQ_CHANNELS = ['a1', 'a2', 'af3', 'af4', 'af7', 'af8', 'afz', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'ccp1', 'ccp2', 'ccp3', 'ccp4', 'ccp5', 'ccp6', 'ccp7', 'ccp8', 'cfc1', 'cfc2', 'cfc3', 'cfc4', 'cfc5', 'cfc6', 'cfc7', 'cfc8', 'cp1', 'cp2', 'cp3', 'cp4', 'cp5', 'cp6', 'cpz', 'cz', 'eog', 'f1', 'f10', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fcz', 'fp1', 'fp2', 'fpz', 'ft7', 'ft8', 'fz', 'iz', 'loc', 'o1', 'o2', 'oz', 'p08', 'p1', 'p10', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'po1', 'po10', 'po2', 'po3', 'po4', 'po7', 'po8', 'po9', 'poz', 'pz', 'roc', 'sp1', 'sp2', 't1', 't10', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 'tp10', 'tp7', 'tp8', 'tp9']

@tuple
class _Preset:
    sfreq: float
    patch_size: int
    max_patches: int
    kernels1: List[int]
    kernels2: List[int]
    head_pooling: str
    channels: List[str]

def _preset(modality: str) -> _Preset:
    if modality == "eeg":
        return _Preset(200.0, 200, 256, [21, 15, 9, 5], [9, 7, 5, 3], "flatten", NEURORVQ_CHANNELS)
    if modality == "ecg":
        return _Preset(200.0, 40, 600, [21, 15, 9, 5], [9, 7, 5, 3], "mean",
                       "avf avl avr i ii iii v1 v2 v3 v4 v5 v6 vx vy vz".split())
    if modality == "emg":
        return _Preset(1000.0, 200, 256, [51, 17, 8, 5], [25, 9, 4, 3], "mean", sorted([f"c{i}" for i in range(1, 17)]))
    if modality == "ppg":
        return _Preset(100.0, 80, 12, [41, 31, 17, 9], [17, 13, 9, 5], "mean", ["ppg_c1"])
    raise ValueError("modality must be one of ['ecg', 'eeg', 'emg', 'ppg'].")

def _rvq_init(m):
    n = m._get_name()
    if n in ("Linear", "Conv2d"):
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

class _MultiScaleTemporalConv(nn.Module):
    _c1: List[nn.Conv2d]
    _n1: List[nn.GroupNorm]
    _c2: List[nn.Conv2d]
    _n2: List[nn.GroupNorm]
    gelu1: nn.Module
    gelu2: nn.Module
    def __init__(self, out_chans: int, activation, kernels1: List[int], kernels2: List[int]):
        super().__init__()
        self._c1 = []
        self._n1 = []
        self._c2 = []
        self._n2 = []
        for i in range(len(kernels1)):
            k = kernels1[i]
            c = nn.Conv2d(1, out_chans, (1, k), padding=(0, k // 2))
            g = nn.GroupNorm(4, out_chans)
            self.add_module(f"conv1_{i + 1}", c)
            self.add_module(f"norm1_{i + 1}", g)
            self.add_module(f"pool1_{i + 1}", nn.AvgPool2d((1, 2)))
            self._c1.append(c)
            self._n1.append(g)
        for i in range(len(kernels2)):
            k = kernels2[i]
            c = nn.Conv2d(out_chans, out_chans, (1, k), padding=(0, k // 2))
            g = nn.GroupNorm(4, out_chans)
            self.add_module(f"conv2_{i + 1}", c)
            self.add_module(f"norm2_{i + 1}", g)
            self.add_module(f"pool2_{i + 1}", nn.AvgPool2d((1, 4)))
            self._c2.append(c)
            self._n2.append(g)
        self.gelu1 = activation()
        self.add_module("gelu1", self.gelu1)
        self.gelu2 = activation()
        self.add_module("gelu2", self.gelu2)
    def branches(self, x: Tensor) -> List[Tensor]:
        B, C, P, S = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(B, C * P, S).unsqueeze(1)
        out = List[Tensor]()
        for i in range(len(self._c1)):
            h = F.avg_pool2d(self.gelu1(self._n1[i](self._c1[i](x))), (1, 2))
            h = F.avg_pool2d(self.gelu2(self._n2[i](self._c2[i](h))), (1, 4))
            out.append(h.permute(0, 2, 3, 1).reshape(B, C * P, -1))
        return out
    def _get_name(self) -> str:
        return "_MultiScaleTemporalConv"

class NeuroRVQ(EEGModuleMixin, nn.Module):
    """NeuroRVQ encoder from Barmpas et al (2025). Architecture only.

    In-model resampling to the modality sampling rate is not ported: the
    input sfreq must match the preset (or be unset)."""
    embed_dim: int
    patch_size: int
    on_non_divisible: str
    max_patches: int
    head_pooling: str
    num_patches: int
    channel_names: List[str]
    spatial_embedding_ix: Tensor
    channel_keep_ix: Tensor
    _drops_channels: bool
    cls_token: Tensor
    patch_embed: _MultiScaleTemporalConv
    pos_embed: Tensor
    time_embed: Tensor
    pos_drop: nn.Dropout
    blocks: nn.ModuleList
    _blocks: List[PreNormBlock[_LabramAttention, MLP]]
    norm: nn.Identity
    fc_norm: nn.LayerNorm
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 channel_names: Optional[List[str]] = None, modality: str = "eeg", patch_size: Optional[int] = None,
                 on_non_divisible: str = "pad", on_missing: str = "error", max_patches: Optional[int] = None,
                 depth: int = 12, num_heads: int = 10, mlp_ratio: float = 4.0, out_chans: int = 8,
                 qkv_bias: bool = True, qk_norm: bool = True, init_values: float = 1e-5, drop_prob: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0, activation = nn.GELU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if on_missing not in ("error", "drop"):
            raise ValueError(f'on_missing must be "error" or "drop", got {on_missing!r}.')
        pr = _preset(modality)
        ps = patch_size.__val__() if patch_size is not None else pr.patch_size
        mp = max_patches.__val__() if max_patches is not None else pr.max_patches
        _check_on_non_divisible(on_non_divisible)
        in_sf: Optional[float] = self._sfreq
        if in_sf is None and self._input_window_seconds is not None:
            in_sf = float(self.n_times) / self._input_window_seconds.__val__()
        if in_sf is not None and in_sf.__val__() != pr.sfreq:
            raise NotImplementedError(f"NeuroRVQ: in-model resampling ({in_sf} Hz -> {pr.sfreq} Hz) is not ported.")
        npch = patch_count(self.n_times, ps, on_non_divisible)
        if npch > mp:
            raise ValueError(f"n_times supports at most {mp} patches (got {npch}).")
        if depth <= 0 or num_heads <= 0 or out_chans <= 0 or mp <= 0:
            raise ValueError("depth, num_heads, out_chans and max_patches must be positive.")
        if out_chans % 4:
            raise ValueError("out_chans must be divisible by 4 for GroupNorm.")
        E = out_chans * (ps // 8)
        if E % num_heads:
            raise ValueError("The derived embedding width must be divisible by num_heads.")
        self.embed_dim = E
        self.patch_size = ps
        self.on_non_divisible = on_non_divisible
        self.max_patches = mp
        self.head_pooling = pr.head_pooling
        self.num_patches = npch
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
        if len(set(names)) != len(names):
            raise ValueError("channel_names must be unique.")
        keep = [i for i in range(len(names)) if names[i] in pr.channels]
        if len(keep) != len(names) and (on_missing != "drop" or len(keep) == 0):
            raise ValueError('Some channel names are not in the NeuroRVQ vocabulary; pass on_missing="drop" to leave them out.')
        self._drops_channels = len(keep) != len(names)
        self.channel_names = [names[i] for i in keep]
        self.spatial_embedding_ix = torch.tensor([pr.channels.index(n) for n in self.channel_names], dtype=torch.long)
        self.register_buffer("spatial_embedding_ix", self.spatial_embedding_ix, persistent=False)
        self.channel_keep_ix = torch.tensor(keep, dtype=torch.long)
        self.register_buffer("channel_keep_ix", self.channel_keep_ix, persistent=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, E))
        self.register_parameter("cls_token", self.cls_token)
        self.patch_embed = _MultiScaleTemporalConv(out_chans, activation, pr.kernels1, pr.kernels2)
        self.add_module("patch_embed", self.patch_embed)
        self.pos_embed = nn.Parameter(torch.zeros(len(pr.channels) + 1, E))
        self.register_parameter("pos_embed", self.pos_embed)
        self.time_embed = nn.Parameter(torch.zeros(mp, E))
        self.register_parameter("time_embed", self.time_embed)
        self.pos_drop = nn.Dropout(drop_prob)
        self.add_module("pos_drop", self.pos_drop)
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist()
        ls: Optional[float] = None
        if init_values > 0:
            ls = init_values
        self._blocks = [PreNormBlock(E, _LabramAttention(E, num_heads, qkv_bias, qk_norm, None, attn_drop_rate, drop_prob),
                                     MLP(E, [int(float(E) * mlp_ratio)], E, drop=drop_prob), drop_path=float(dpr[i]),
                                     layer_scale=ls) for i in range(depth)]
        self.blocks = nn.ModuleList()
        for b in self._blocks:
            self.blocks.append(b)
        self.add_module("blocks", self.blocks)
        self.norm = nn.Identity()
        self.add_module("norm", self.norm)
        hd = E * 4
        if self.head_pooling == "flatten":
            hd *= len(self.channel_names) * npch
        self.fc_norm = nn.LayerNorm(hd)
        self.add_module("fc_norm", self.fc_norm)
        self.final_layer = nn.Linear(hd, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        self.apply(_rvq_init)
        with torch.no_grad():
            for i in range(len(self._blocks)):
                s = math.sqrt(2.0 * float(i + 1))
                self._blocks[i].attn.proj.weight.div_(s)
                _param_by_name(self._blocks[i].mlp.children()[2], "weight").__val__().div_(s)

    def forward_features(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_chans or x.shape[2] != self.n_times:
            raise ValueError(f"Expected input shape (batch, {self.n_chans}, {self.n_times}), got {x.shape}.")
        if self._drops_channels:
            x = x.index_select(1, self.channel_keep_ix)
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        x = x.reshape(x.shape[0], x.shape[1], -1, self.patch_size)
        brs = self.patch_embed.branches(x)
        nsl = self.spatial_embedding_ix.numel()
        six = torch.cat([torch.zeros(1, dtype=torch.long), torch.repeat_interleave(self.spatial_embedding_ix, self.num_patches)])
        spatial = self.pos_embed.index_select(0, six).unsqueeze(0)
        tix = torch.arange(self.max_patches - self.num_patches, self.max_patches).repeat(nsl)
        temporal = self.time_embed.index_select(0, tix).unsqueeze(0)
        cls = self.cls_token.expand(brs[0].shape[0], -1, -1)
        scales = List[Tensor]()
        for br in brs:
            h = torch.cat([cls, br], dim=1) + spatial
            h = torch.cat([h[:, :1], h[:, 1:] + temporal], dim=1)
            h = self.pos_drop(h)
            for b in self._blocks:
                h = b(h)
            scales.append(h[:, 1:])
        tok = torch.cat(scales, dim=-1)
        f = tok.flatten(1) if self.head_pooling == "flatten" else tok.mean(1)
        return self.fc_norm(f)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x))

    def _get_name(self) -> str:
        return "NeuroRVQ"
