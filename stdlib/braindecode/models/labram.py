# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import scaled_dot_product_attention
from braindecode.functional.initialization import _param_by_name
from braindecode.modules import DropPath
from braindecode.modules.mlp import MLP

LABRAM_CHANNEL_ORDER = [
    'FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', 'F9', 'F7',
    'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
    'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', 'TP9', 'TP7', 'CP5',
    'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6',
    'P8', 'P10', 'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', 'O1', 'OZ', 'O2',
    'O9', 'CB1', 'CB2', 'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', 'CFC1', 'CFC2', 'CFC3', 'CFC4',
    'CFC5', 'CFC6', 'CFC7', 'CFC8', 'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', 'T1', 'T2',
    'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', 'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP2-F8', 'F8-T8',
    'T8-P8', 'P8-O2']
_LABRAM_N_TIME_SLOTS = 16

def _n_patches(n_times: int, patch_size: int, on_non_divisible: str) -> int:
    return -(-n_times // patch_size) if on_non_divisible == "pad" else n_times // patch_size

def _prepare_input(x: Tensor, patch_size: int, on_non_divisible: str) -> Tensor:
    t = x.shape[-1]
    r = t % patch_size
    if r == 0:
        return x
    if on_non_divisible == "pad":
        return F.pad(x, [0, patch_size - r])
    if on_non_divisible == "crop":
        return x[:, :, :t - r]
    raise ValueError(f"Time dimension ({t}) is not divisible by patch_size ({patch_size}).")

def _labram_init(m):
    # braindecode.functional.init_weights_ default (timm/ViT scheme)
    n = m._get_name()
    if n == "Linear":
        w = _param_by_name(m, "weight")
        if w is not None:
            nn.init.trunc_normal_(w.__val__(), std=0.02)
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

class _SegmentPatch(nn.Module):
    n_times: int
    patch_size: int
    n_patchs: int
    emb_dim: int
    n_chans: int
    learned_patcher: bool
    on_non_divisible: str
    patcher: Optional[nn.Conv1d]

    def __init__(self, n_times: int = 2000, patch_size: int = 200, n_chans: int = 1, emb_dim: int = 200,
                 learned_patcher: bool = True, on_non_divisible: str = "pad"):
        super().__init__()
        self.n_times = n_times
        self.patch_size = patch_size
        self.on_non_divisible = on_non_divisible
        self.n_patchs = _n_patches(n_times, patch_size, on_non_divisible)
        self.emb_dim = emb_dim
        self.n_chans = n_chans
        self.learned_patcher = learned_patcher
        self.patcher = None
        if learned_patcher:
            p = nn.Conv1d(1, emb_dim, patch_size, stride=patch_size)
            self.patcher = p
            self.add_module("patcher", p)

    def forward(self, x: Tensor) -> Tensor:
        x = _prepare_input(x, self.patch_size, self.on_non_divisible)
        b, c, t = x.shape[0], x.shape[1], x.shape[2]
        if self.patcher is not None:
            y = self.patcher.__val__()(x.reshape(b * c, 1, t))
            return y.reshape(b, c, y.shape[1], y.shape[2]).permute(0, 1, 3, 2)
        return x.view(b, c, t // self.patch_size, self.patch_size)

    def _get_name(self) -> str:
        return "_SegmentPatch"

class _PatchEmbed(nn.Module):
    n_times: int
    patch_size: int
    n_patchs: int
    emb_dim: int
    in_channels: int
    on_non_divisible: str
    proj: nn.Conv2d

    def __init__(self, n_times: int = 2000, patch_size: int = 200, in_channels: int = 1, emb_dim: int = 200,
                 on_non_divisible: str = "pad"):
        super().__init__()
        self.n_times = n_times
        self.patch_size = patch_size
        self.on_non_divisible = on_non_divisible
        self.n_patchs = _n_patches(n_times, patch_size, on_non_divisible)
        self.emb_dim = emb_dim
        self.in_channels = in_channels
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=(1, patch_size), stride=(1, patch_size))
        self.add_module("proj", self.proj)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        b, c = x.shape[0], x.shape[1]
        x = _prepare_input(x, self.patch_size, self.on_non_divisible)
        if c % self.in_channels != 0:
            raise ValueError(f"n_channels ({c}) must be divisible by in_channels ({self.in_channels}).")
        x = x.view(b, self.in_channels, c // self.in_channels, x.shape[-1])
        return self.proj(x).mean(dim=2).transpose(1, 2).contiguous()

    def _get_name(self) -> str:
        return "_PatchEmbed"

class _TemporalConv(nn.Module):
    conv1: nn.Conv2d
    act_layer_1: nn.Module
    norm1: nn.GroupNorm
    conv2: nn.Conv2d
    act_layer_2: nn.Module
    norm2: nn.GroupNorm
    conv3: nn.Conv2d
    norm3: nn.GroupNorm
    act_layer_3: nn.Module

    def __init__(self, in_channels: int = 1, out_channels: int = 8, num_groups: int = 4, activation = nn.GELU):
        super().__init__()
        self.add_module("channel_patch_flatten", Rearrange("Batch chs npat spatch -> Batch () (chs npat) spatch"))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.add_module("conv1", self.conv1)
        self.act_layer_1 = activation()
        self.add_module("act_layer_1", self.act_layer_1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.add_module("norm1", self.norm1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.add_module("conv2", self.conv2)
        self.act_layer_2 = activation()
        self.add_module("act_layer_2", self.act_layer_2)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.add_module("norm2", self.norm2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.add_module("conv3", self.conv3)
        self.norm3 = nn.GroupNorm(num_groups, out_channels)
        self.add_module("norm3", self.norm3)
        self.act_layer_3 = activation()
        self.add_module("act_layer_3", self.act_layer_3)
        self.add_module("transpose_temporal_channel", Rearrange("Batch C NA T -> Batch NA (T C)"))

    def forward(self, x: Tensor) -> Tensor:
        b, c, n, s = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(b, 1, c * n, s)
        x = self.act_layer_1(self.norm1(self.conv1(x)))
        x = self.act_layer_2(self.norm2(self.conv2(x)))
        x = self.act_layer_3(self.norm3(self.conv3(x)))
        # Batch C NA T -> Batch NA (T C)
        return x.permute(0, 2, 3, 1).flatten(2)

    def _get_name(self) -> str:
        return "_TemporalConv"

class _LabramAttention(nn.Module):
    num_heads: int
    scale: float
    qkv: nn.Linear
    q_bias: Optional[Tensor]
    v_bias: Optional[Tensor]
    q_norm: Optional[nn.LayerNorm]
    k_norm: Optional[nn.LayerNorm]
    window: int
    relative_position_bias_table: Optional[Tensor]
    relative_position_index: Optional[Tensor]
    attn_drop: nn.Dropout
    proj: nn.Linear
    proj_drop: nn.Dropout

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_norm: bool = True,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 window_size: Optional[Tuple[int, int]] = None, attn_head_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        hd = dim // num_heads
        if attn_head_dim is not None:
            hd = attn_head_dim.__val__()
        ahd = hd * num_heads
        self.scale = qk_scale.__val__() if qk_scale is not None and qk_scale.__val__() != 0.0 else float(hd) ** -0.5
        self.q_bias = None
        self.v_bias = None
        if qkv_bias:
            qb = nn.Parameter(torch.zeros(ahd))
            vb = nn.Parameter(torch.zeros(ahd))
            self.q_bias = qb
            self.v_bias = vb
            self.register_parameter("q_bias", qb)
            self.register_parameter("v_bias", vb)
        self.window = 0
        self.relative_position_bias_table = None
        self.relative_position_index = None
        if window_size is not None:
            wh, ww = window_size.__val__()
            nrd = (2 * wh - 1) * (2 * ww - 1) + 3
            tbl = nn.Parameter(torch.zeros(nrd, num_heads))
            self.relative_position_bias_table = tbl
            self.register_parameter("relative_position_bias_table", tbl)
            n = wh * ww
            vals = List[int](capacity=(n + 1) * (n + 1))
            for i in range(n + 1):
                for j in range(n + 1):
                    v = 0
                    if i == 0 and j == 0:
                        v = nrd - 1
                    elif j == 0:
                        v = nrd - 2
                    elif i == 0:
                        v = nrd - 3
                    else:
                        hi, wi = (i - 1) // ww, (i - 1) % ww
                        hj, wj = (j - 1) // ww, (j - 1) % ww
                        v = (hi - hj + wh - 1) * (2 * ww - 1) + (wi - wj + ww - 1)
                    vals.append(v)
            idx = torch.tensor(vals, dtype=torch.long).reshape(n + 1, n + 1)
            self.relative_position_index = idx
            self.register_buffer("relative_position_index", idx)
            self.window = n
        self.qkv = nn.Linear(dim, ahd * 3, bias=False)
        self.add_module("qkv", self.qkv)
        self.q_norm = None
        self.k_norm = None
        if qk_norm:
            qn = nn.LayerNorm(hd, eps=1e-6)
            kn = nn.LayerNorm(hd, eps=1e-6)
            self.q_norm = qn
            self.k_norm = kn
            self.add_module("q_norm", qn)
            self.add_module("k_norm", kn)
        self.attn_drop = nn.Dropout(attn_drop)
        self.add_module("attn_drop", self.attn_drop)
        self.proj = nn.Linear(ahd, dim)
        self.add_module("proj", self.proj)
        self.proj_drop = nn.Dropout(proj_drop)
        self.add_module("proj_drop", self.proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N = x.shape[0], x.shape[1]
        qkv = self.qkv(x)
        if self.q_bias is not None:
            qb = self.q_bias.__val__()
            vb = self.v_bias.__val__()
            qkv = qkv + torch.cat([qb, torch.zeros_like(vb), vb])
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        parts = qkv.unbind(0)
        q, k, v = parts[0], parts[1], parts[2]
        if self.q_norm is not None:
            q = self.q_norm.__val__()(q)
        if self.k_norm is not None:
            k = self.k_norm.__val__()(k)
        mask: Optional[Tensor] = None
        if self.relative_position_bias_table is not None:
            n1 = self.window + 1
            rpb = self.relative_position_bias_table.__val__().index_select(0, self.relative_position_index.__val__().view(-1))
            mask = rpb.view(n1, n1, -1).permute(2, 0, 1).contiguous().unsqueeze(0)
        p = self.attn_drop.p if self.training else 0.0
        out = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p, scale=self.scale, exact=True)
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, -1)))

    def _get_name(self) -> str:
        return "_Attention"

class _WindowsAttentionBlock(nn.Module):
    gamma_1: Optional[Tensor]
    gamma_2: Optional[Tensor]
    norm1: nn.LayerNorm
    attn: _LabramAttention
    drop_path: nn.Module
    norm2: nn.LayerNorm
    mlp: MLP

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, qk_norm: bool = True,
                 qk_scale: Optional[float] = None, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 init_values: float = 0.0, activation = nn.GELU, window_size: Optional[Tuple[int, int]] = None,
                 attn_head_dim: Optional[int] = None):
        super().__init__()
        self.gamma_1 = None
        self.gamma_2 = None
        if init_values > 0:
            g1 = nn.Parameter(init_values * torch.ones(dim))
            g2 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_1 = g1
            self.gamma_2 = g2
            self.register_parameter("gamma_1", g1)
            self.register_parameter("gamma_2", g2)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.add_module("norm1", self.norm1)
        self.attn = _LabramAttention(dim, num_heads, qkv_bias, qk_norm, qk_scale, attn_drop, drop, window_size, attn_head_dim)
        self.add_module("attn", self.attn)
        dp: nn.Module = nn.Identity()
        if drop_path > 0.0:
            dp = DropPath(drop_path)
        self.drop_path = dp
        self.add_module("drop_path", dp)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.add_module("norm2", self.norm2)
        self.mlp = MLP(dim, [int(float(dim) * mlp_ratio)], activation=activation, drop=drop)
        self.add_module("mlp", self.mlp)

    def forward(self, x: Tensor) -> Tensor:
        y = self.attn(self.norm1(x))
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            return x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.gamma_1.__val__() * y)
        return x + self.drop_path(self.gamma_2.__val__() * self.mlp(self.norm2(x)))

    def _get_name(self) -> str:
        return "_WindowsAttentionBlock"

class Labram(EEGModuleMixin, nn.Module):
    """LaBraM from Jiang et al (2024). Architecture only (no Hub weights).

    ``qk_norm`` is a bool here (True = nn.LayerNorm as in braindecode, False =
    None). Without ``ch_names`` the input must have the 128 canonical
    channels; use ``forward_ch(x, ch_names)`` for a subset."""
    patch_size: int
    embed_dim: int
    num_features: int
    neural_tokenizer: bool
    init_scale: float
    on_non_divisible: str
    n_path: int
    n_patchs: int
    cls_token: Tensor
    position_embedding: Optional[Tensor]
    temporal_embedding: Tensor
    patch_embed: nn.Sequential
    _seg: Optional[_SegmentPatch]
    _tconv: Optional[_TemporalConv]
    _pemb: Optional[_PatchEmbed]
    pos_drop: nn.Dropout
    blocks: nn.ModuleList
    _blocks: List[_WindowsAttentionBlock]
    norm: nn.Module
    fc_norm: Optional[nn.LayerNorm]
    final_layer: nn.Module
    _canonical_index: Dict[str, int]

    def __init__(self, n_times: Optional[int] = None, n_outputs: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_chans: Optional[int] = None,
                 sfreq: Optional[float] = None, input_window_seconds: Optional[float] = None, patch_size: int = 200,
                 learned_patcher: bool = False, embed_dim: int = 200, conv_in_channels: int = 1,
                 conv_out_channels: int = 8, num_layers: int = 12, num_heads: int = 10, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_norm: bool = True, qk_scale: Optional[float] = None, drop_prob: float = 0.0,
                 attn_drop_prob: float = 0.0, drop_path_prob: float = 0.0, init_values: float = 0.1,
                 use_abs_pos_emb: bool = True, use_mean_pooling: bool = True, init_scale: float = 0.001,
                 neural_tokenizer: bool = True, attn_head_dim: Optional[int] = None, activation = nn.GELU,
                 on_non_divisible: str = "pad"):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self._canonical_index = {LABRAM_CHANNEL_ORDER[i].upper(): i for i in range(len(LABRAM_CHANNEL_ORDER))}
        self.neural_tokenizer = neural_tokenizer
        self.init_scale = init_scale
        self.patch_size = patch_size
        if patch_size > self.n_times:
            self.patch_size = self.n_times
        self.on_non_divisible = on_non_divisible
        self.n_path = _n_patches(self.n_times, self.patch_size, on_non_divisible)
        self.patch_embed = nn.Sequential()
        self._seg = None
        self._tconv = None
        self._pemb = None
        if neural_tokenizer:
            seg = _SegmentPatch(self.n_times, self.patch_size, self.n_chans, self.patch_size, learned_patcher, on_non_divisible)
            tc = _TemporalConv(out_channels=conv_out_channels, activation=activation)
            self.patch_embed.add_module("segment_patch", seg)
            self.patch_embed.add_module("temporal_conv", tc)
            self._seg = seg
            self._tconv = tc
            self.n_patchs = seg.n_patchs
        else:
            pe = _PatchEmbed(self.n_times, patch_size, conv_in_channels, embed_dim, on_non_divisible)
            self.patch_embed.add_module("segment_patch", pe)
            self._pemb = pe
            self.n_patchs = pe.n_patchs
        self.add_module("patch_embed", self.patch_embed)
        if neural_tokenizer:
            w = (self.patch_size + 14 - 15) // 8 + 1
            self.embed_dim = w * conv_out_channels
        else:
            self.embed_dim = embed_dim
        self.num_features = self.embed_dim
        E = self.embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, E))
        self.register_parameter("cls_token", self.cls_token)
        self.position_embedding = None
        if use_abs_pos_emb:
            pe2 = nn.Parameter(torch.zeros(1, len(LABRAM_CHANNEL_ORDER) + 1, E))
            self.position_embedding = pe2
            self.register_parameter("position_embedding", pe2)
        nts = max(_LABRAM_N_TIME_SLOTS, self.n_patchs) if neural_tokenizer else self.n_patchs + 1
        self.temporal_embedding = nn.Parameter(torch.zeros(1, nts, E))
        self.register_parameter("temporal_embedding", self.temporal_embedding)
        self.pos_drop = nn.Dropout(drop_prob)
        self.add_module("pos_drop", self.pos_drop)
        dpr = torch.linspace(0.0, drop_path_prob, num_layers).tolist()
        ws: Optional[Tuple[int, int]] = None
        if not neural_tokenizer:
            ws = (1, self.n_patchs)
        self._blocks = [_WindowsAttentionBlock(E, num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_prob,
                                               attn_drop_prob, float(dpr[i]), init_values, activation, ws, attn_head_dim)
                        for i in range(num_layers)]
        self.blocks = nn.ModuleList()
        for b in self._blocks:
            self.blocks.append(b)
        self.add_module("blocks", self.blocks)
        nm: nn.Module = nn.Identity()
        self.fc_norm = None
        if not use_mean_pooling:
            nm = nn.LayerNorm(E, eps=1e-6)
        self.norm = nm
        self.add_module("norm", nm)
        if use_mean_pooling:
            fn = nn.LayerNorm(E, eps=1e-6)
            self.fc_norm = fn
            self.add_module("fc_norm", fn)
        fl: nn.Module = nn.Identity()
        lin: Optional[nn.Linear] = None
        if self.n_outputs > 0:
            l = nn.Linear(E, self.n_outputs)
            lin = l
            fl = l
        self.final_layer = fl
        self.add_module("final_layer", fl)
        self.apply(_labram_init)
        with torch.no_grad():
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
            if self.position_embedding is not None:
                nn.init.trunc_normal_(self.position_embedding.__val__(), std=0.02)
            if lin is not None:
                nn.init.trunc_normal_(lin.__val__().weight, std=0.02)
            for i in range(len(self._blocks)):
                blk = self._blocks[i]
                s = math.sqrt(2.0 * float(i + 1))
                blk.attn.proj.weight.div_(s)
                ml = blk.mlp.children()
                _param_by_name(ml[len(ml) - 2], "weight").__val__().div_(s)
            if lin is not None:
                lin.__val__().weight.mul_(self.init_scale)
                lin.__val__().bias.__val__().mul_(self.init_scale)

    def _embed(self, x: Tensor) -> Tensor:
        if self._seg is not None:
            return self._tconv.__val__()(self._seg.__val__()(x))
        return self._pemb.__val__()(x)

    def forward_features(self, x: Tensor, input_chans: Tensor, return_patch_tokens: bool = False,
                         return_all_tokens: bool = False) -> Tensor:
        B, C = x.shape[0], x.shape[1]
        x = self._embed(x)
        npc = x.shape[1]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        if self.position_embedding is not None:
            pemb = self.position_embedding.__val__()
            if self.neural_tokenizer:
                used = pemb.index_select(1, input_chans)
                rest = used[:, 1:, :].unsqueeze(2).expand(B, -1, self.n_patchs, -1).flatten(1, 2)
                x = x + torch.cat([used[:, 0:1, :].expand(B, -1, -1), rest], dim=1)
            else:
                n_pos = min(pemb.shape[1], npc + 1)
                x = x + pemb[:, :n_pos, :].expand(B, -1, -1)
        te: Tensor
        if self.neural_tokenizer:
            te = self.temporal_embedding[:, 0:self.n_patchs, :].unsqueeze(1).expand(B, C, -1, -1).flatten(1, 2)
        else:
            nt = min(npc, self.temporal_embedding.shape[1] - 1)
            te = self.temporal_embedding[:, 1:nt + 1, :].expand(B, -1, -1)
        x = torch.cat([x[:, 0:1, :], x[:, 1:, :] + te], dim=1)
        x = self.pos_drop(x)
        for blk in self._blocks:
            x = blk(x)
        x = self.norm(x)
        if self.fc_norm is not None:
            fn = self.fc_norm.__val__()
            if return_all_tokens:
                return fn(x)
            tokens = x[:, 1:, :]
            if return_patch_tokens:
                return fn(tokens)
            return fn(tokens.mean(1))
        if return_all_tokens:
            return x
        if return_patch_tokens:
            return x[:, 1:]
        return x[:, 0]

    def _input_chans(self, x: Tensor, ch_names: Optional[List[str]]) -> Tensor:
        nc = len(self._canonical_index)
        if ch_names is None:
            if x.shape[1] != nc:
                raise ValueError(f"Labram expects the {nc} canonical channels when ch_names is not given, got {x.shape[1]}.")
            return torch.arange(nc + 1, dtype=torch.long)
        names = ch_names.__val__()
        if len(names) != x.shape[1]:
            raise ValueError(f"len(ch_names)={len(names)} does not match the input channels {x.shape[1]}.")
        matched = [0]
        for n in names:
            u = n.upper()
            if u not in self._canonical_index:
                raise ValueError(f"Channel {n!r} is not in the LaBraM channel vocabulary.")
            matched.append(self._canonical_index[u] + 1)
        return torch.tensor(matched, dtype=torch.long)

    def forward_ch(self, x: Tensor, ch_names: Optional[List[str]] = None) -> Tensor:
        return self.final_layer(self.forward_features(x, self._input_chans(x, ch_names)))

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_ch(x, None)

    def _get_name(self) -> str:
        return "Labram"
