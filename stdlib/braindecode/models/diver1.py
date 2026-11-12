# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import apply_rotary, rotary_angles, rotary_frequencies
from braindecode.functional.patches import pad_or_crop_to_patches

class _GatedFFN(nn.Module):
    # braindecode FeedForwardBlock(gated=True)
    fc1: nn.Linear
    fc_gate: nn.Linear
    fc2: nn.Linear
    activation: nn.Module
    dropout1: nn.Dropout
    dropout2: nn.Dropout
    def __init__(self, d: int, hidden: int, drop_p: float, activation, bias: bool, output_drop_p: float):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden, bias=bias)
        self.add_module("fc1", self.fc1)
        self.fc_gate = nn.Linear(d, hidden, bias=bias)
        self.add_module("fc_gate", self.fc_gate)
        self.fc2 = nn.Linear(hidden, d, bias=bias)
        self.add_module("fc2", self.fc2)
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.dropout1 = nn.Dropout(drop_p)
        self.add_module("dropout1", self.dropout1)
        self.dropout2 = nn.Dropout(output_drop_p)
        self.add_module("dropout2", self.dropout2)
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout2(self.fc2(self.dropout1(self.activation(self.fc_gate(x)) * self.fc1(x))))
    def _get_name(self) -> str:
        return "FeedForwardBlock"

class _RotaryEmbedding(nn.Module):
    cos: Tensor
    sin: Tensor
    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim must be even, got {head_dim}.")
        a = rotary_angles(torch.arange(max_len, dtype=torch.float32), rotary_frequencies(head_dim, base))
        self.cos = a.cos()
        self.register_buffer("cos", self.cos, persistent=False)
        self.sin = a.sin()
        self.register_buffer("sin", self.sin, persistent=False)
    def rotate(self, x: Tensor, pid: Tensor) -> Tensor:
        return apply_rotary(x, self.cos.index_select(0, pid).to(x.dtype), self.sin.index_select(0, pid).to(x.dtype))
    def _get_name(self) -> str:
        return "_RotaryEmbedding"

class _AnyVariateAttention(nn.Module):
    num_heads: int
    head_dim: int
    scale: Optional[float]
    drop_prob: float
    rotary: _RotaryEmbedding
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    q_norm: nn.RMSNorm
    k_norm: nn.RMSNorm
    out_proj: nn.Linear
    channel_bias: nn.Embedding
    def __init__(self, d_model: int, num_heads: int, drop_prob: float, rotary: _RotaryEmbedding, mup_attention: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / float(self.head_dim) if mup_attention else None
        self.drop_prob = drop_prob
        self.rotary = rotary
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.add_module("q_proj", self.q_proj)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.add_module("k_proj", self.k_proj)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.add_module("v_proj", self.v_proj)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.add_module("q_norm", self.q_norm)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.add_module("k_norm", self.k_norm)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.add_module("out_proj", self.out_proj)
        self.channel_bias = nn.Embedding(2, num_heads)
        self.add_module("channel_bias", self.channel_bias)
    def _split(self, t: Tensor) -> Tensor:
        return t.unflatten(-1, [self.num_heads, -1]).transpose(1, 2)
    def attn(self, x: Tensor, var_id: Tensor, time_id: Tensor) -> Tensor:
        q = self.rotary.rotate(self.q_norm(self._split(self.q_proj(x))), time_id)
        k = self.rotary.rotate(self.k_norm(self._split(self.k_proj(x))), time_id)
        v = self._split(self.v_proj(x))
        same = (var_id.unsqueeze(-1) == var_id.unsqueeze(-2)).unsqueeze(0).unsqueeze(0)
        w = self.channel_bias.weight
        bias = torch.where(same, w[1].view(1, -1, 1, 1), w[0].view(1, -1, 1, 1)).to(q.dtype)
        p = self.drop_prob if self.training else 0.0
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=p, scale=self.scale)
        return self.out_proj(o.transpose(1, 2).flatten(2))
    def _get_name(self) -> str:
        return "_AnyVariateAttention"

class _AnyVariateEncoderLayer(nn.Module):
    norm1: nn.RMSNorm
    self_attn: _AnyVariateAttention
    dropout: nn.Dropout
    norm2: nn.RMSNorm
    ffn: _GatedFFN
    def __init__(self, d_model: int, num_heads: int, d_ff: int, drop_prob: float, activation, rotary: _RotaryEmbedding,
                 mup_attention: bool = True):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.add_module("norm1", self.norm1)
        self.self_attn = _AnyVariateAttention(d_model, num_heads, drop_prob, rotary, mup_attention)
        self.add_module("self_attn", self.self_attn)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.add_module("norm2", self.norm2)
        self.ffn = _GatedFFN(d_model, d_ff, drop_prob, activation, False, drop_prob)
        self.add_module("ffn", self.ffn)
    def layer(self, x: Tensor, var_id: Tensor, time_id: Tensor) -> Tensor:
        x = x + self.dropout(self.self_attn.attn(self.norm1(x), var_id, time_id))
        return x + self.ffn(self.norm2(x))
    def _get_name(self) -> str:
        return "_AnyVariateEncoderLayer"

class _AnyVariateEncoder(nn.Module):
    rotary: _RotaryEmbedding
    layers: nn.ModuleList
    _layers: List[_AnyVariateEncoderLayer]
    norm: nn.RMSNorm
    def __init__(self, d_model: int, n_layers: int, num_heads: int, d_ff: int, drop_prob: float, activation, max_len: int,
                 mup_attention: bool = True):
        super().__init__()
        self.rotary = _RotaryEmbedding(d_model // num_heads, max_len)
        self.add_module("rotary", self.rotary)
        self._layers = [_AnyVariateEncoderLayer(d_model, num_heads, d_ff, drop_prob, activation, self.rotary, mup_attention)
                        for _ in range(n_layers)]
        self.layers = nn.ModuleList()
        for l in self._layers:
            self.layers.append(l)
        self.add_module("layers", self.layers)
        self.norm = nn.RMSNorm(d_model, eps=1e-5)
        self.add_module("norm", self.norm)
    def encode(self, x: Tensor, var_id: Tensor, time_id: Tensor) -> Tensor:
        for l in self._layers:
            x = l.layer(x, var_id, time_id)
        return self.norm(x)
    def _get_name(self) -> str:
        return "_AnyVariateEncoder"

def _ids(n_rows: int, n_cols: int) -> Tuple[Tensor, Tensor]:
    v = [r for r in range(n_rows) for _ in range(n_cols)]
    t = [c for _ in range(n_rows) for c in range(n_cols)]
    return torch.tensor(v, dtype=torch.long), torch.tensor(t, dtype=torch.long)

class _PatchCNN(nn.Module):
    padl: int
    padr: int
    n_patches: int
    proj_in: nn.Sequential
    def __init__(self, n_patches: int, patch_size: int, d_model: int, stride: Optional[int] = None, kernel_size: int = 63,
                 depth: int = 3, out_size: Optional[int] = None):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"cnn_kernel_size must be odd, got {kernel_size}.")
        if depth < 1:
            raise ValueError(f"cnn_depth must be at least 1, got {depth}.")
        padded = 1
        while padded < patch_size:
            padded *= 2
        st = 0
        if out_size is not None:
            if stride is not None:
                raise ValueError("cnn_out_size and cnn_stride are mutually exclusive.")
            o = out_size.__val__()
            if o < 1 or padded % o:
                raise ValueError(f"cnn_out_size must divide {padded}.")
            st = padded // o
        elif stride is not None:
            st = stride.__val__()
        else:
            st = padded // (8 if patch_size >= 100 else 16)
        if st < 1 or padded % st:
            raise ValueError(f"cnn_stride must divide {padded}.")
        os_ = padded // st
        if d_model % os_:
            raise ValueError(f"d_model ({d_model}) must be divisible by the CNN output length ({os_}).")
        hidden = d_model // os_
        pt = padded - patch_size
        self.padl = pt // 2
        self.padr = pt - pt // 2
        self.n_patches = n_patches
        ng = math.gcd(os_, hidden)
        self.add_module("pad", nn.ZeroPad2d((self.padl, self.padr, 0, 0)))
        self.proj_in = nn.Sequential(nn.Conv2d(1, hidden, kernel_size=(1, kernel_size), stride=(1, st),
                                               padding=(0, (kernel_size - 1) // 2)), nn.GroupNorm(ng, hidden), nn.GELU())
        j = 3
        for _ in range(depth - 1):
            self.proj_in.add_module(str(j), nn.Conv2d(hidden, hidden, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)))
            self.proj_in.add_module(str(j + 1), nn.GroupNorm(ng, hidden))
            self.proj_in.add_module(str(j + 2), nn.GELU())
            j += 3
        self.add_module("proj_in", self.proj_in)
    def forward(self, x: Tensor) -> Tensor:
        B, C, P = x.shape[0], x.shape[1], x.shape[2]
        x = F.pad(x, [self.padl, self.padr]).reshape(B, 1, C * P, -1)
        y = self.proj_in(x)
        H, O = y.shape[1], y.shape[3]
        return y.reshape(B, H, C, P, O).permute(0, 2, 3, 1, 4).reshape(B, C, P, H * O)
    def _get_name(self) -> str:
        return "_PatchCNN"

class _SinusoidalCoordEmbedding(nn.Module):
    n_feats: int
    padding: int
    scale: float
    dim_t: Tensor
    def __init__(self, d_model: int, temperature: float = 2000.0, scale: float = 1.0 / 256.0):
        super().__init__()
        fpa = d_model // 3
        self.n_feats = (fpa // 2) * 2
        self.padding = d_model - self.n_feats * 3
        self.scale = scale * 2.0 * math.pi
        dt = torch.arange(0, self.n_feats, 2, dtype=torch.float32) / float(self.n_feats)
        self.dim_t = torch.pow(torch.full([dt.numel()], temperature), dt)
        self.register_buffer("dim_t", self.dim_t, persistent=False)
    def forward(self, xyz: Tensor) -> Tensor:
        a = (xyz * self.scale).unsqueeze(-1) / self.dim_t
        e = torch.stack([a.sin(), a.cos()], dim=-1).flatten(-3)
        return F.pad(e, [0, self.padding])
    def _get_name(self) -> str:
        return "_SinusoidalCoordEmbedding"

class _ChannelMetaEmbedding(nn.Module):
    coord_emb: _SinusoidalCoordEmbedding
    type_emb: nn.Embedding
    subtype_emb: nn.Embedding
    def __init__(self, d_model: int):
        super().__init__()
        dt = d_model // 4
        if dt < 1:
            raise ValueError(f"d_model ({d_model}) too small for the channel metadata embedding.")
        self.coord_emb = _SinusoidalCoordEmbedding(d_model - dt)
        self.add_module("coord_emb", self.coord_emb)
        self.type_emb = nn.Embedding(2, dt)
        self.add_module("type_emb", self.type_emb)
        self.subtype_emb = nn.Embedding(3, dt)
        self.add_module("subtype_emb", self.subtype_emb)
    def forward(self, md: Tensor) -> Tensor:
        co = md[..., :3]
        known = (torch.isfinite(co).all(dim=-1, keepdim=True) & (co != 0.0).any(dim=-1, keepdim=True)).to(co.dtype)
        st = md[..., 4].long()
        sk = (st >= 0).to(co.dtype).unsqueeze(-1)
        pos = self.coord_emb(torch.nan_to_num(co)) * known
        mod = self.type_emb(md[..., 3].long()) + self.subtype_emb(torch.clamp(st, min=0)) * sk
        return torch.cat([pos, mod], dim=-1)
    def _get_name(self) -> str:
        return "_ChannelMetaEmbedding"

class _STCPE(nn.Module):
    window: int
    n_patches: int
    n_windows: int
    window_index: Tensor
    down: nn.Linear
    encoder: _AnyVariateEncoder
    up: nn.Linear
    def __init__(self, d_model: int, ratio: int, window: int, activation, n_patches: int, mup_attention: bool = True):
        super().__init__()
        inner = d_model // ratio
        if inner < 2 or inner % 2:
            raise ValueError(f"d_model // stcpe_ratio must be even and >= 2, got {inner}.")
        ih = max(1, inner // 32)
        while ih > 1 and (inner % ih or (inner // ih) % 2):
            ih -= 1
        self.window = window
        self.n_patches = n_patches
        self.n_windows = n_patches + window - 1
        wi = [w + n for w in range(window) for n in range(self.n_windows)]
        self.window_index = torch.tensor(wi, dtype=torch.long)
        self.register_buffer("window_index", self.window_index, persistent=False)
        self.down = nn.Linear(d_model, inner)
        self.add_module("down", self.down)
        self.encoder = _AnyVariateEncoder(inner, 1, ih, 4 * inner, 0.0, activation, window, mup_attention)
        self.add_module("encoder", self.encoder)
        self.up = nn.Linear(inner, d_model)
        self.add_module("up", self.up)
    def forward(self, x: Tensor) -> Tensor:
        B, C, P = x.shape[0], x.shape[1], x.shape[2]
        W, NW = self.window, self.n_windows
        x = self.down(x)
        D = x.shape[-1]
        win = F.pad(x, [0, 0, W - 1, W - 1]).index_select(2, self.window_index).view(B, C, W, NW, D)
        win = win.permute(0, 3, 1, 2, 4).reshape(B * NW, C * W, D)
        vid, tid = _ids(C, W)
        enc = self.encoder.encode(win, vid, tid).view(B, NW, C, W, D).permute(0, 2, 3, 1, 4).reshape(B, C, W * NW, D)
        npad = P + 2 * (W - 1)
        folded = torch.zeros(B, C, npad, D, dtype=x.dtype).index_add(2, self.window_index, enc)
        ov = torch.zeros(npad, dtype=x.dtype).index_add(0, self.window_index, torch.ones(W * NW, dtype=x.dtype))
        s, e = W - 1, W - 1 + P
        return self.up(folded[:, :, s:e] / ov[s:e].unsqueeze(1))
    def _get_name(self) -> str:
        return "_STCPE"

class DIVER1(EEGModuleMixin, nn.Module):
    """DIVER-1 from Han et al (2025). Architecture only (no Hub weights).

    braindecode reads per-channel metadata (xyz in mm, modality, sub-modality)
    from chs_info; here it is passed as ``chan_metadata`` (n_chans, 5) in the
    same layout. ``ch_mask`` is not ported."""
    patch_size: int
    d_model: int
    num_heads: int
    pooling: str
    n_patches: int
    patch_cnn: _PatchCNN
    spectral_emb: Optional[nn.Sequential]
    default_chan_metadata: Tensor
    chan_emb: Optional[_ChannelMetaEmbedding]
    stcpe: Optional[_STCPE]
    patch_register: Tensor
    chan_register: Tensor
    global_register: Tensor
    encoder: _AnyVariateEncoder
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, patch_size: int = 500,
                 d_model: int = 256, n_layers: int = 12, num_heads: Optional[int] = None, d_ff: Optional[int] = None,
                 cnn_depth: int = 3, cnn_stride: Optional[int] = None, cnn_kernel_size: int = 63, use_stcpe: bool = True,
                 stcpe_window: int = 7, stcpe_ratio: int = 8, use_spectral_emb: bool = True,
                 use_position_emb: bool = True, pooling: str = "flatten", mup_attention: bool = True,
                 drop_prob: float = 0.1, activation = nn.SiLU, cnn_out_size: Optional[int] = None,
                 chan_metadata: Optional[Tensor] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if chan_metadata is None:
            raise ValueError("DIVER1 needs per-channel metadata: pass chan_metadata (n_chans, 5) = (x, y, z [mm], modality, sub-modality).")
        md = chan_metadata.__val__().to(torch.float32)
        if md.ndim != 2 or md.shape[0] != self.n_chans or md.shape[1] != 5:
            raise ValueError(f"chan_metadata must have shape ({self.n_chans}, 5).")
        if pooling not in ("flatten", "mean"):
            raise ValueError(f"pooling must be 'flatten' or 'mean', got {pooling!r}.")
        if patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        nh = num_heads.__val__() if num_heads is not None else max(1, d_model // 32)
        dff = d_ff.__val__() if d_ff is not None else 4 * d_model
        if d_model % nh or (d_model // nh) % 2:
            raise ValueError(f"d_model ({d_model}) must split into num_heads ({nh}) heads of even width.")
        if stcpe_window % 2 == 0:
            raise ValueError(f"stcpe_window must be odd, got {stcpe_window}.")
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = nh
        self.pooling = pooling
        self.n_patches = -(-self.n_times // patch_size)
        self.patch_register = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_parameter("patch_register", self.patch_register)
        self.chan_register = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_parameter("chan_register", self.chan_register)
        self.global_register = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_parameter("global_register", self.global_register)
        self.patch_cnn = _PatchCNN(self.n_patches, patch_size, d_model, cnn_stride, cnn_kernel_size, cnn_depth, cnn_out_size)
        self.add_module("patch_cnn", self.patch_cnn)
        self.spectral_emb = None
        if use_spectral_emb:
            se = nn.Sequential()
            se.add_module("spectral_proj", nn.Sequential(nn.Linear(d_model // 2 + 1, d_model), nn.Dropout(drop_prob)))
            self.spectral_emb = se
            self.add_module("spectral_emb", se)
        self.default_chan_metadata = md
        self.register_buffer("default_chan_metadata", md, persistent=False)
        self.chan_emb = None
        if use_position_emb:
            ce = _ChannelMetaEmbedding(d_model)
            self.chan_emb = ce
            self.add_module("chan_emb", ce)
        self.stcpe = None
        if use_stcpe:
            sc = _STCPE(d_model, stcpe_ratio, stcpe_window, activation, self.n_patches, mup_attention)
            self.stcpe = sc
            self.add_module("stcpe", sc)
        # torch draws the registers after these sub-modules (state_dict order is unaffected)
        with torch.no_grad():
            self.patch_register.copy_(torch.randn(1, 1, d_model) * 0.02)
            self.chan_register.copy_(torch.randn(1, 1, d_model) * 0.02)
            self.global_register.copy_(torch.randn(1, 1, d_model) * 0.02)
        self.encoder = _AnyVariateEncoder(d_model, n_layers, nh, dff, drop_prob, activation, max(512, self.n_patches + 1),
                                          mup_attention)
        self.add_module("encoder", self.encoder)
        hi = self.n_chans * self.n_patches * d_model if pooling == "flatten" else d_model
        self.final_layer = nn.Linear(hi, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        B, C = x.shape[0], x.shape[1]
        if C != self.n_chans:
            raise ValueError(f"DIVER1 expects {self.n_chans} channels, got {C}.")
        if -(-x.shape[-1] // self.patch_size) != self.n_patches:
            raise ValueError(f"Input length gives a different patch count than n_times={self.n_times}.")
        x = pad_or_crop_to_patches(x, self.patch_size, "pad")
        t = self.patch_cnn(x.reshape(B, C, -1, self.patch_size))
        if self.spectral_emb is not None:
            amp = torch.abs(torch.fft.rfft(t, dim=-1, norm="forward")).to(t.dtype)
            t = t + self.spectral_emb.__val__()(amp)
        if self.chan_emb is not None:
            t = t + self.chan_emb.__val__()(self.default_chan_metadata).unsqueeze(-2)
        if self.stcpe is not None:
            t = t + self.stcpe.__val__()(t)
        P = self.n_patches
        t = torch.cat([self.patch_register.unsqueeze(0).expand(B, C, -1, -1), t], dim=2)
        row = torch.cat([self.global_register.unsqueeze(0).expand(B, -1, -1, -1),
                         self.chan_register.unsqueeze(0).expand(B, -1, P, -1)], dim=2)
        t = torch.cat([row, t], dim=1)
        R, Cn = t.shape[1], t.shape[2]
        vid, tid = _ids(R, Cn)
        t = self.encoder.encode(t.flatten(1, 2), vid, tid).reshape(B, R, Cn, -1)[:, 1:, 1:]
        f = t.flatten(1) if self.pooling == "flatten" else t.mean(dim=[1, 2])
        return self.final_layer(f)

    def _get_name(self) -> str:
        return "DIVER1"
