# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional.initialization import _param_by_name
from braindecode.functional.patches import pad_or_crop_to_patches
from braindecode.functional.wavelet import daubechies_filters, wavelet_decomposition

_FMIN = -3.4028234663852886e38

def _mvp_init(m):
    n = m._get_name()
    if n in ("Linear", "Embedding"):
        nn.init.normal_(_param_by_name(m, "weight").__val__(), mean=0.0, std=0.02)
        b = _param_by_name(m, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())

class _WaveletPatchEmbed(nn.Module):
    _filters: Tensor
    dwt_size: int
    ln: nn.RMSNorm
    proj: nn.Linear
    def __init__(self, segment_len: int, d_model: int):
        super().__init__()
        self._filters = daubechies_filters(4)
        self.register_buffer("_filters", self._filters, persistent=False)
        self.dwt_size = wavelet_decomposition(torch.zeros(1, segment_len), self._filters).shape[-1]
        self.ln = nn.RMSNorm(self.dwt_size, eps=1e-5)
        self.add_module("ln", self.ln)
        self.proj = nn.Linear(self.dwt_size, d_model, bias=False)
        self.add_module("proj", self.proj)
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.ln(wavelet_decomposition(x, self._filters)))
    def _get_name(self) -> str:
        return "_WaveletPatchEmbed"

def _mvpa_masks(S: int, C: int, lw: int) -> Tuple[Tensor, Tensor, Tensor]:
    n = S * C
    win = List[bool](capacity=n * n)
    cau = List[bool](capacity=n * n)
    for i in range(n):
        si = i // C
        for j in range(n):
            sj = j // C
            win.append(i >= n - C or (sj - si <= lw and si - sj <= lw))
            cau.append(sj <= si)
    sh = List[int](capacity=n * C)
    for i in range(n):
        c = i % C
        for c2 in range(C):
            sh.append(C - 1 - abs(c - c2))
    return (torch.tensor(win).reshape(n, n), torch.tensor(cau).reshape(n, n),
            torch.tensor(sh, dtype=torch.long).reshape(n, C))

class _MVPAttention(nn.Module):
    n_heads: int
    n_head_kv: int
    n_kv_groups: int
    head_dim: int
    kv_dim: int
    global_att: bool
    local_window: int
    scale_attn: bool
    scale_by_layer_idx: bool
    layer_idx: Optional[int]
    q_attn: nn.Linear
    c_attn: nn.Linear
    position_net: nn.Linear
    channel_net: nn.Linear
    c_proj: nn.Linear
    attn_bias: Tensor
    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout

    def __init__(self, d_model: int, n_heads: int, n_head_kv: int, global_att: bool = True, local_window: int = 10,
                 attn_drop: float = 0.0, resid_drop: float = 0.0, scale_attn: bool = True,
                 scale_by_layer_idx: bool = False, layer_idx: Optional[int] = None):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        if n_heads % n_head_kv:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_head_kv ({n_head_kv}).")
        self.n_heads = n_heads
        self.n_head_kv = n_head_kv
        self.n_kv_groups = n_heads // n_head_kv
        self.head_dim = d_model // n_heads
        self.kv_dim = self.head_dim * n_head_kv
        self.global_att = global_att
        self.local_window = local_window
        self.scale_attn = scale_attn
        self.scale_by_layer_idx = scale_by_layer_idx
        self.layer_idx = layer_idx
        self.q_attn = nn.Linear(d_model, d_model, bias=False)
        self.add_module("q_attn", self.q_attn)
        self.c_attn = nn.Linear(d_model, 2 * self.kv_dim, bias=False)
        self.add_module("c_attn", self.c_attn)
        self.position_net = nn.Linear(d_model, self.kv_dim, bias=False)
        self.add_module("position_net", self.position_net)
        self.channel_net = nn.Linear(d_model, self.kv_dim, bias=False)
        self.add_module("channel_net", self.channel_net)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.add_module("c_proj", self.c_proj)
        self.attn_bias = nn.Parameter(torch.empty(3 * self.kv_dim))
        self.register_parameter("attn_bias", self.attn_bias)
        nn.init.normal_(self.attn_bias, std=0.02)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.add_module("attn_dropout", self.attn_dropout)
        self.resid_dropout = nn.Dropout(resid_drop)
        self.add_module("resid_dropout", self.resid_dropout)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        r = self.n_kv_groups
        x = x.contiguous()
        b, kv, s, c, hd = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        return x.unsqueeze(2).expand(-1, -1, r, -1, -1, -1).reshape(b, kv * r, s, c, hd)

    def _split(self, t: Tensor, nh: int) -> Tensor:
        return t.unflatten(-1, [nh, -1]).permute(0, 3, 1, 2, 4)

    def _as_bias(self, b: Tensor) -> Tensor:
        return self._repeat_kv(b.reshape(1, self.n_head_kv, 1, 1, -1))

    @staticmethod
    def _rel_shift(x: Tensor) -> Tensor:
        b, h, nq, nk = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        z = torch.zeros(b, h, nk, 1, dtype=x.dtype)
        xp = torch.cat([z, x.contiguous().reshape(b, h, nk, nq)], dim=-1).reshape(b, h, nq + 1, nk)
        return xp[:, :, 1:, :].reshape(b, h, nq, nk)

    def forward_attn(self, hs: Tensor, pos: Tensor, chan: Tensor, masks: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        q = self._split(self.q_attn(hs), self.n_heads)
        kv = self.c_attn(hs).split(self.kv_dim, dim=-1)
        ck = self._split(kv[0], self.n_head_kv)
        v = self._split(kv[1], self.n_head_kv)
        tk = self._split(self.position_net(pos).unsqueeze(2), self.n_head_kv)
        chk = self._split(self.channel_net(chan).unsqueeze(1), self.n_head_kv)
        S, C = q.shape[2], q.shape[3]
        wm, cm, cs = masks
        bs = self.attn_bias.split(self.kv_dim, dim=0)
        tb = self._as_bias(bs[1])
        cb = self._as_bias(bs[2])
        tk = self._repeat_kv(tk).squeeze(3).permute(0, 1, 3, 2)
        chk = self._repeat_kv(chk).squeeze(2).permute(0, 1, 3, 2)
        tq = (q + tb).flatten(2, 3)
        cq = (q + cb).flatten(2, 3)
        ta = self._rel_shift(torch.matmul(tq, tk))
        ca0 = torch.matmul(cq, chk)
        ca = torch.gather(ca0, -1, cs.expand(ca0.shape[0], ca0.shape[1], cs.shape[0], cs.shape[1]))
        aw = ta.unsqueeze(-1).expand(-1, -1, -1, -1, C).flatten(3) + ca.unsqueeze(3).expand(-1, -1, -1, S, -1).flatten(3)
        if self.global_att:
            gb = self._as_bias(bs[0])
            gk = self._repeat_kv(ck).permute(0, 1, 4, 2, 3).flatten(3)
            ga = torch.matmul((q + gb).flatten(2, 3), gk)
            aw = aw + ga.masked_fill(wm.logical_not(), 0.0)
        if self.scale_attn:
            aw = aw / (float(v.size(-1)) ** 0.5)
        if self.scale_by_layer_idx and self.layer_idx is not None:
            aw = aw / float(self.layer_idx.__val__() + 1)
        aw = aw.masked_fill(cm.logical_not(), _FMIN)
        aw = self.attn_dropout(F.softmax(aw, dim=-1))
        out = torch.matmul(aw, self._repeat_kv(v).flatten(2, 3)).unflatten(2, [-1, C])
        return self.resid_dropout(self.c_proj(out.permute(0, 2, 3, 1, 4).flatten(3)))

    def _get_name(self) -> str:
        return "_MVPAttention"

class _SwiGLUMLP(nn.Module):
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear
    act: nn.Module
    def __init__(self, d_model: int, d_inner: int, activation = nn.SiLU):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.add_module("gate_proj", self.gate_proj)
        self.up_proj = nn.Linear(d_model, d_inner, bias=False)
        self.add_module("up_proj", self.up_proj)
        self.down_proj = nn.Linear(d_inner, d_model, bias=False)
        self.add_module("down_proj", self.down_proj)
        self.act = activation()
        self.add_module("act", self.act)
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
    def _get_name(self) -> str:
        return "_SwiGLUMLP"

class _MVPABlock(nn.Module):
    ln_1: nn.RMSNorm
    attn: _MVPAttention
    mlp: _SwiGLUMLP
    def __init__(self, d_model: int, n_heads: int, n_head_kv: int, d_inner: int, global_att: bool = True,
                 local_window: int = 10, attn_drop: float = 0.0, resid_drop: float = 0.0, eps: float = 1e-5,
                 layer_idx: Optional[int] = None, activation = nn.SiLU):
        super().__init__()
        self.ln_1 = nn.RMSNorm(d_model, eps=eps)
        self.add_module("ln_1", self.ln_1)
        self.attn = _MVPAttention(d_model, n_heads, n_head_kv, global_att, local_window, attn_drop, resid_drop,
                                  True, False, layer_idx)
        self.add_module("attn", self.attn)
        self.mlp = _SwiGLUMLP(d_model, d_inner, activation)
        self.add_module("mlp", self.mlp)
    def block(self, h: Tensor, pos: Tensor, chan: Tensor, masks: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        n = self.ln_1(h)
        return h + self.mlp(n) + self.attn.forward_attn(n, pos, chan, masks)
    def _get_name(self) -> str:
        return "_MVPABlock"

class MVPFormer(EEGModuleMixin, nn.Module):
    """MVPFormer from Carzaniga et al (2026). Architecture only (no Hub weights).

    The optional ``ch_mask`` of braindecode's forward is not ported."""
    segment_len: int
    d_model: int
    pooling: str
    local_window: int
    n_segments: int
    patch_embed: _WaveletPatchEmbed
    positional_embedding: nn.Embedding
    channel_embedding: nn.Embedding
    drop: nn.Dropout
    blocks: nn.ModuleList
    _blocks: List[_MVPABlock]
    ln_f: nn.RMSNorm
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 segment_len: int = 2560, d_model: int = 256, n_layers: int = 4, n_heads: int = 8,
                 n_head_kv: int = 4, d_inner: int = 512, local_window: int = 10, global_att: bool = True,
                 max_segments: int = 110, max_channels: int = 128, drop_prob: float = 0.1, activation = nn.SiLU,
                 pooling: str = "mean"):
        if segment_len < 1:
            raise ValueError(f"segment_len must be a positive int, got {segment_len}.")
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if pooling not in ("mean", "concat"):
            raise ValueError(f"pooling must be 'mean' or 'concat', got {pooling!r}.")
        self.segment_len = segment_len
        self.d_model = d_model
        self.pooling = pooling
        self.local_window = local_window
        self.n_segments = -(-self.n_times // segment_len)
        if self.n_segments > max_segments:
            raise ValueError(f"n_times={self.n_times} gives {self.n_segments} segments, more than max_segments={max_segments}.")
        if self.n_chans > max_channels:
            raise ValueError(f"n_chans={self.n_chans} exceeds max_channels={max_channels}.")
        self.patch_embed = _WaveletPatchEmbed(segment_len, d_model)
        self.add_module("patch_embed", self.patch_embed)
        self.positional_embedding = nn.Embedding(max_segments, d_model)
        self.add_module("positional_embedding", self.positional_embedding)
        self.channel_embedding = nn.Embedding(max_channels, d_model)
        self.add_module("channel_embedding", self.channel_embedding)
        self.drop = nn.Dropout(drop_prob)
        self.add_module("drop", self.drop)
        self._blocks = [_MVPABlock(d_model, n_heads, n_head_kv, d_inner, global_att, local_window, drop_prob, drop_prob,
                                   1e-5, i, activation) for i in range(n_layers)]
        self.blocks = nn.ModuleList()
        for b in self._blocks:
            self.blocks.append(b)
        self.add_module("blocks", self.blocks)
        self.ln_f = nn.RMSNorm(d_model, eps=1e-5)
        self.add_module("ln_f", self.ln_f)
        hi = d_model if pooling == "mean" else self.n_chans * d_model
        self.final_layer = nn.Linear(hi, self.n_outputs, bias=False)
        self.add_module("final_layer", self.final_layer)
        self.apply(_mvp_init)

    def forward_features(self, x: Tensor) -> Tensor:
        x = pad_or_crop_to_patches(x, self.segment_len, "pad")
        p = x.reshape(x.shape[0], x.shape[1], -1, self.segment_len)
        C, S = p.shape[1], p.shape[2]
        e = self.patch_embed(p).permute(0, 2, 1, 3)
        pos = self.positional_embedding.weight[:S].unsqueeze(0)
        ch = self.channel_embedding.weight[:C].unsqueeze(0)
        h = self.drop(e)
        masks = _mvpa_masks(S, C, self.local_window)
        for b in self._blocks:
            h = b.block(h, pos, ch, masks)
        pooled = self.ln_f(h)[:, -1]
        if self.pooling == "mean":
            return pooled.mean(dim=1)
        return pooled.flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x))

    def _get_name(self) -> str:
        return "MVPFormer"
