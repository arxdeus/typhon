# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import apply_rotary, rotary_frequencies
from braindecode.functional.initialization import _param_by_name
from braindecode.modules import FeedForwardBlock
from braindecode.modules.quantization import ResidualVectorQuantizer

_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946

class _SELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        f = torch.where(x > 0.0, torch.full_like(x, _SELU_SCALE), torch.ones_like(x))
        return F.elu(x, _SELU_ALPHA * _SELU_SCALE) * f
    def _get_name(self) -> str:
        return "_SELU"

def _omni_init(m):
    n = m._get_name()
    if n in ("Linear", "Embedding"):
        nn.init.trunc_normal_(_param_by_name(m, "weight").__val__(), std=0.02)
        b = _param_by_name(m, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())
    elif n == "RMSNorm":
        w = _param_by_name(m, "weight")
        if w is not None:
            nn.init.ones_(w.__val__())

def sensor_geometry(positions: Tensor, sensor_type: List[int], orientations: Optional[Tensor] = None) -> Tensor:
    """(n_chans, 6) positions + orientations, centred and scaled per modality
    (EEG / MEG) as braindecode's _geometry_from_chs_info."""
    n = positions.shape[0]
    pos = torch.zeros(n, 6)
    vals = positions.to(torch.float32).reshape(-1).tolist()
    ori = orientations.__val__().to(torch.float32).reshape(-1).tolist() if orientations is not None else [0.0] * (3 * n)
    rows = [[0.0] * 6 for _ in range(n)]
    for i in range(n):
        for j in range(3):
            rows[i][j] = vals[3 * i + j]
            if sensor_type[i] > 0:
                rows[i][3 + j] = ori[3 * i + j]
    for meg in (False, True):
        idx = [i for i in range(n) if (sensor_type[i] > 0) == meg]
        if len(idx) == 0:
            continue
        # float32 arithmetic, as numpy on the float32 array
        sub = torch.tensor([[rows[i][j] for j in range(3)] for i in idx], dtype=torch.float32)
        c = sub - sub.mean(dim=0, keepdim=True)
        sc = torch.sqrt(3.0 * (c * c).sum(dim=1).mean()).item()
        c = c / (sc if sc > 0 else 1.0)
        cl = c.reshape(-1).tolist()
        for k in range(len(idx)):
            for j in range(3):
                rows[idx[k]][j] = cl[3 * k + j]
    return torch.tensor(rows, dtype=torch.float32)

def _attend(q: Tensor, k: Tensor, v: Tensor, n_head: int, p: float) -> Tensor:
    q = q.unflatten(-1, [n_head, -1]).transpose(1, 2)
    k = k.unflatten(-1, [n_head, -1]).transpose(1, 2)
    v = v.unflatten(-1, [n_head, -1]).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=p).transpose(1, 2).flatten(2)

class _SEANetConv1d(nn.Module):
    conv: nn.Conv1d
    k: int
    s: int
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        c = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        weight_norm(c)
        self.conv = c
        self.add_module("conv", c)
        self.k = kernel_size
        self.s = stride
    def forward(self, x: Tensor) -> Tensor:
        pt = self.k - self.s
        extra = (self.k - pt - x.shape[-1]) % self.s
        r = pt // 2
        pl, pr = pt - r, r + extra
        if max(pl, pr) == 0:
            return self.conv(x)
        short = max(max(pl, pr) - x.shape[-1] + 1, 0)
        if short > 0:
            x = F.pad(x, [0, short])
        x = F.pad(x, [pl, pr], mode="reflect")
        return self.conv(x[..., :x.shape[-1] - short])
    def _get_name(self) -> str:
        return "_SEANetConv1d"

class _SEANetConvTranspose1d(nn.Module):
    convtr: nn.ConvTranspose1d
    padding_total: int
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        c = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        weight_norm(c)
        self.convtr = c
        self.add_module("convtr", c)
        self.padding_total = kernel_size - stride
    def forward(self, x: Tensor) -> Tensor:
        y = self.convtr(x)
        r = self.padding_total // 2
        return y[..., self.padding_total - r:y.shape[-1] - r]
    def _get_name(self) -> str:
        return "_SEANetConvTranspose1d"

class _SEANetLSTM(nn.Module):
    lstm: nn.LSTM
    def __init__(self, dimension: int):
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, 2)
        self.add_module("lstm", self.lstm)
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x, None)
        return (y + x).permute(1, 2, 0)
    def _get_name(self) -> str:
        return "_SEANetLSTM"

class _SEANetResBlock(nn.Module):
    block: nn.Sequential
    shortcut: _SEANetConv1d
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(nn.ELU(), _SEANetConv1d(dim, dim // 2, 3), nn.ELU(), _SEANetConv1d(dim // 2, dim, 1))
        self.add_module("block", self.block)
        self.shortcut = _SEANetConv1d(dim, dim, 1)
        self.add_module("shortcut", self.shortcut)
    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.block(x)
    def _get_name(self) -> str:
        return "_SEANetResBlock"

def _seanet_encoder(dim: int, nf: int, ratios: List[int], ks: int, lks: int) -> nn.Sequential:
    s = nn.Sequential(_SEANetConv1d(1, nf, ks))
    i = 1
    m = 1
    for r in reversed(ratios):
        s.add_module(str(i), _SEANetResBlock(m * nf))
        s.add_module(str(i + 1), nn.ELU())
        s.add_module(str(i + 2), _SEANetConv1d(m * nf, 2 * m * nf, 2 * r, r))
        i += 3
        m *= 2
    s.add_module(str(i), _SEANetLSTM(m * nf))
    s.add_module(str(i + 1), nn.ELU())
    s.add_module(str(i + 2), _SEANetConv1d(m * nf, dim, lks))
    return s

def _seanet_decoder(dim: int, nf: int, ratios: List[int], ks: int, lks: int) -> nn.Sequential:
    m = 1 << len(ratios)
    s = nn.Sequential(_SEANetConv1d(dim, m * nf, ks), _SEANetLSTM(m * nf))
    i = 2
    for r in ratios:
        o = m * nf // 2
        s.add_module(str(i), nn.ELU())
        s.add_module(str(i + 1), _SEANetConvTranspose1d(m * nf, o, 2 * r, r))
        s.add_module(str(i + 2), _SEANetResBlock(o))
        i += 3
        m //= 2
    s.add_module(str(i), nn.ELU())
    s.add_module(str(i + 1), _SEANetConv1d(nf, 1, lks))
    return s

class _SensorEmbedding(nn.Module):
    sensor_embedding_layer: nn.Embedding
    pos_embedding_layer: nn.Sequential
    aggregate_mlp: FeedForwardBlock
    norm: nn.RMSNorm
    def __init__(self, n_dim: int):
        super().__init__()
        self.sensor_embedding_layer = nn.Embedding(3, n_dim)
        self.add_module("sensor_embedding_layer", self.sensor_embedding_layer)
        self.pos_embedding_layer = nn.Sequential(nn.Linear(6, n_dim // 2), _SELU(), nn.Linear(n_dim // 2, n_dim))
        self.add_module("pos_embedding_layer", self.pos_embedding_layer)
        self.aggregate_mlp = FeedForwardBlock(n_dim, 4, 0.0, _SELU)
        self.add_module("aggregate_mlp", self.aggregate_mlp)
        self.norm = nn.RMSNorm(n_dim, eps=1e-6)
        self.add_module("norm", self.norm)
    def embed(self, pos: Tensor, st: Tensor) -> Tensor:
        x = self.pos_embedding_layer(pos) + self.sensor_embedding_layer(st).to(pos.dtype)
        return self.norm(x + self.aggregate_mlp(x))
    def _get_name(self) -> str:
        return "_SensorEmbedding"

class _BackwardSolution(nn.Module):
    n_head: int
    dropout: float
    v: nn.Linear
    proj: nn.Linear
    def __init__(self, n_dim: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.dropout = dropout
        self.v = nn.Linear(n_dim, n_dim)
        self.add_module("v", self.v)
        self.proj = nn.Linear(n_dim, n_dim)
        self.add_module("proj", self.proj)
    def solve(self, q: Tensor, k: Tensor, x: Tensor) -> Tensor:
        return self.proj(_attend(q, k, self.v(x), self.n_head, self.dropout if self.training else 0.0))
    def _get_name(self) -> str:
        return "_BackwardSolution"

class _ForwardSolution(nn.Module):
    n_dim: int
    n_head: int
    dropout: float
    kv: nn.Linear
    proj: nn.Linear
    def __init__(self, n_dim: int, n_head: int, dropout: float):
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.kv = nn.Linear(n_dim, 2 * n_dim)
        self.add_module("kv", self.kv)
        self.proj = nn.Linear(n_dim, n_dim)
        self.add_module("proj", self.proj)
    def solve(self, se: Tensor, neurons: Tensor) -> Tensor:
        kv = self.kv(neurons).split(self.n_dim, dim=-1)
        return self.proj(_attend(se, kv[0], kv[1], self.n_head, self.dropout if self.training else 0.0))
    def _get_name(self) -> str:
        return "_ForwardSolution"

class _TokenizerEncoder(nn.Module):
    seanet_encoder: nn.Sequential
    neuros: Tensor
    backwardsolution: _BackwardSolution
    k_proj: nn.Linear
    def __init__(self, nf: int, ratios: List[int], ks: int, lks: int, n_dim: int, n_head: int, dropout: float, n_neuro: int):
        super().__init__()
        self.seanet_encoder = _seanet_encoder(n_dim, nf, ratios, ks, lks)
        self.add_module("seanet_encoder", self.seanet_encoder)
        self.neuros = nn.Parameter(torch.randn(n_neuro, n_dim))
        self.register_parameter("neuros", self.neuros)
        self.backwardsolution = _BackwardSolution(n_dim, n_head, dropout)
        self.add_module("backwardsolution", self.backwardsolution)
        self.k_proj = nn.Linear(n_dim, n_dim)
        self.add_module("k_proj", self.k_proj)
    def encode(self, x: Tensor, se: Tensor) -> Tensor:
        B, C, W, L = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = self.seanet_encoder(x.reshape(B * C * W, 1, L))
        x = x.reshape(B, C, W, x.shape[1], -1).permute(0, 1, 2, 4, 3).reshape(B, C, -1, x.shape[1])
        T, D = x.shape[2], x.shape[3]
        s = se.unsqueeze(2).repeat(1, 1, T, 1).permute(0, 2, 1, 3).reshape(-1, C, D)
        x = x.permute(0, 2, 1, 3).reshape(-1, C, D)
        nr = self.neuros.to(x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.backwardsolution.solve(nr, self.k_proj(x + s), x)
        x = x.reshape(B, W, -1, x.shape[1], D).permute(0, 3, 1, 2, 4)
        x = x.reshape(B, x.shape[1], -1, D)
        return x.reshape(B, x.shape[1], W, -1, D)
    def _get_name(self) -> str:
        return "_TokenizerEncoder"

class _TokenizerDecoder(nn.Module):
    forwardsolution: _ForwardSolution
    seanet_decoder: nn.Sequential
    def __init__(self, n_dim: int, n_head: int, nf: int, ratios: List[int], ks: int, lks: int, dropout: float):
        super().__init__()
        self.forwardsolution = _ForwardSolution(n_dim, n_head, dropout)
        self.add_module("forwardsolution", self.forwardsolution)
        self.seanet_decoder = _seanet_decoder(n_dim, nf, ratios, ks, lks)
        self.add_module("seanet_decoder", self.seanet_decoder)
    def decode(self, x: Tensor, se: Tensor) -> Tensor:
        B, S, W, T, D = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        x = x.permute(0, 2, 3, 1, 4).reshape(-1, S, D)
        s = se.view(B, -1, 1, 1, D)
        C = s.shape[1]
        s = s.repeat(1, 1, W, T, 1).permute(0, 2, 3, 1, 4).reshape(-1, C, D)
        x = self.forwardsolution.solve(s, x)
        x = x.reshape(B, W, T, C, D).permute(0, 3, 1, 4, 2)
        x = self.seanet_decoder(x.reshape(-1, D, T))
        return x.reshape(B, C, W, x.shape[-1])
    def _get_name(self) -> str:
        return "_TokenizerDecoder"

class BrainTokenizer(EEGModuleMixin, nn.Module):
    """BrainOmni's BrainTokenizer (Xiao et al 2025). Architecture only.

    braindecode derives sensor geometry from MNE ``chs_info`` (positions,
    coil types, orientations); here pass ``sensor_positions`` (n_chans, 3),
    ``sensor_types`` (0 EEG, 1 MAG, 2 GRAD) and for MEG
    ``sensor_orientations`` (n_chans, 3). ``forward`` returns the
    reconstruction; ``encode_decode`` also returns commitment loss and codes."""
    window_length: int
    pos: Tensor
    sensor_type: Tensor
    sensor_embed: _SensorEmbedding
    encoder: _TokenizerEncoder
    quantizer: ResidualVectorQuantizer
    final_layer: _TokenizerDecoder

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, emb_dim: int = 256,
                 n_neuro: int = 16, window_length: int = 512, n_filters: int = 32, ratios: List[int] = [8, 4, 2],
                 kernel_size: int = 5, last_kernel_size: int = 5, tokenizer_num_heads: int = 4, codebook_dim: int = 256,
                 codebook_size: int = 512, num_quantizers: int = 4, rotation_trick: bool = True, drop_prob: float = 0.0,
                 sensor_positions: Optional[Tensor] = None, sensor_types: Optional[List[int]] = None,
                 sensor_orientations: Optional[Tensor] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if n_filters < 2:
            raise ValueError(f"n_filters must be at least 2, got {n_filters}.")
        if emb_dim <= 0 or tokenizer_num_heads <= 0 or emb_dim % tokenizer_num_heads:
            raise ValueError("emb_dim must be a positive multiple of tokenizer_num_heads.")
        if sensor_positions is None:
            raise ValueError("BrainTokenizer needs sensor_positions (n_chans, 3).")
        sp = sensor_positions.__val__()
        if sp.ndim != 2 or sp.shape[0] != self.n_chans or sp.shape[1] != 3:
            raise ValueError(f"sensor_positions must have shape ({self.n_chans}, 3).")
        st = sensor_types.__val__() if sensor_types is not None else [0] * self.n_chans
        if len(st) != self.n_chans or any([t < 0 or t > 2 for t in st]):
            raise ValueError("sensor_types must give 0 (eeg), 1 (mag) or 2 (grad) per channel.")
        if any([t > 0 for t in st]) and sensor_orientations is None:
            raise ValueError("MEG sensors need sensor_orientations (n_chans, 3).")
        self.pos = sensor_geometry(sp, st, sensor_orientations)
        self.register_buffer("pos", self.pos, persistent=False)
        self.sensor_type = torch.tensor(st, dtype=torch.long)
        self.register_buffer("sensor_type", self.sensor_type, persistent=False)
        self.window_length = window_length
        self.sensor_embed = _SensorEmbedding(emb_dim)
        self.add_module("sensor_embed", self.sensor_embed)
        self.encoder = _TokenizerEncoder(n_filters, ratios, kernel_size, last_kernel_size, emb_dim, tokenizer_num_heads,
                                         drop_prob, n_neuro)
        self.add_module("encoder", self.encoder)
        self.quantizer = ResidualVectorQuantizer(emb_dim, codebook_dim, codebook_size, num_quantizers, rotation_trick)
        self.add_module("quantizer", self.quantizer)
        self.final_layer = _TokenizerDecoder(emb_dim, tokenizer_num_heads, n_filters, ratios, kernel_size,
                                             last_kernel_size, drop_prob)
        self.add_module("final_layer", self.final_layer)
        self.apply(_omni_init)

    def _encode_quantize(self, x: Tensor, overlap_ratio: float = 0.0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        step = int(float(self.window_length) * (1.0 - overlap_ratio))
        x = F.pad(x, [0, max(self.window_length - x.shape[-1], 0)])
        if step < self.window_length:
            x = F.pad(x, [0, (-(x.shape[-1] - self.window_length)) % step])
        x = x.unfold(-1, self.window_length, step)
        se = self.sensor_embed.embed(self.pos, self.sensor_type).unsqueeze(0).expand(x.shape[0], -1, -1)
        fq, idx, cm = self.quantizer.quantize(self.encoder.encode(x, se))
        return fq, idx, cm, se

    def encode_decode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        fq, idx, cm, se = self._encode_quantize(x)
        r = self.final_layer.decode(fq, se)[..., :self.window_length]
        r = r.reshape(r.shape[0], r.shape[1], -1)
        n = x.shape[-1]
        r = F.pad(r, [0, max(n - r.shape[-1], 0)])[..., :n]
        return r, cm, idx

    def tokenize(self, x: Tensor, overlap_ratio: float = 0.0) -> Tuple[Tensor, Tensor]:
        if overlap_ratio < 0 or int(float(self.window_length) * (1.0 - overlap_ratio)) < 1:
            raise ValueError(f"overlap_ratio {overlap_ratio} gives no positive window step.")
        was = self.training
        self.eval()
        f, i, _, _ = self._encode_quantize(x, overlap_ratio)
        self.train(was)
        return f.flatten(2, 3).detach(), i.flatten(2, 3)

    def forward(self, x: Tensor) -> Tensor:
        r, _, _ = self.encode_decode(x)
        return r

    def _get_name(self) -> str:
        return "BrainTokenizer"

class _RotaryPositionalEmbedding(nn.Module):
    n_dim: int
    freqs: Tensor
    rotate: Tensor
    def __init__(self, n_dim: int, base: float = 10000.0):
        super().__init__()
        self.n_dim = n_dim
        self.freqs = rotary_frequencies(n_dim, base)
        self.register_buffer("freqs", self.freqs)
        self.rotate = self._polar(240)
        self.register_buffer("rotate", self.rotate)
    def _polar(self, seq: int) -> Tensor:
        a = torch.outer(torch.arange(seq).to(self.freqs.dtype), self.freqs)
        return torch.stack([a.cos(), a.sin()], dim=-1)
    def rope(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        L, H = q.shape[1], q.shape[2]
        r = self.rotate if L <= self.rotate.shape[0] else self._polar(L)
        r = torch.repeat_interleave(r[:L], 2, dim=1).unflatten(1, [H, -1])
        c, s = r[..., 0], r[..., 1]
        return apply_rotary(q, c, s), apply_rotary(k, c, s)
    def _get_name(self) -> str:
        return "_RotaryPositionalEmbedding"

class _MultiHeadAttentionRoPE(nn.Module):
    dropout: float
    n_dim: int
    n_head: int
    qkv: nn.Linear
    proj: nn.Linear
    rope_embedding_layer: Optional[_RotaryPositionalEmbedding]
    def __init__(self, n_dim: int, n_head: int, dropout: float, rope: bool):
        super().__init__()
        self.dropout = dropout
        self.n_dim = n_dim
        self.n_head = n_head
        self.qkv = nn.Linear(n_dim, 3 * n_dim)
        self.add_module("qkv", self.qkv)
        self.proj = nn.Linear(n_dim, n_dim)
        self.add_module("proj", self.proj)
        self.rope_embedding_layer = None
        if rope:
            r = _RotaryPositionalEmbedding(n_dim)
            self.rope_embedding_layer = r
            self.add_module("rope_embedding_layer", r)
    def forward(self, x: Tensor) -> Tensor:
        c = self.qkv(x).split(self.n_dim, dim=-1)
        q, k = c[0], c[1]
        if self.rope_embedding_layer is not None:
            q2, k2 = self.rope_embedding_layer.__val__().rope(q.unflatten(-1, [self.n_head, -1]), k.unflatten(-1, [self.n_head, -1]))
            q, k = q2.flatten(2), k2.flatten(2)
        return self.proj(_attend(q, k, c[2], self.n_head, self.dropout if self.training else 0.0))
    def _get_name(self) -> str:
        return "_MultiHeadAttentionRoPE"

class _SpatialTemporalBlock(nn.Module):
    pre_attn_norm: nn.RMSNorm
    time_attn: _MultiHeadAttentionRoPE
    spatial_attn: _MultiHeadAttentionRoPE
    pre_ff_norm: nn.RMSNorm
    ff: FeedForwardBlock
    def __init__(self, n_dim: int, n_head: int, dropout: float):
        super().__init__()
        self.pre_attn_norm = nn.RMSNorm(n_dim, eps=1e-6)
        self.add_module("pre_attn_norm", self.pre_attn_norm)
        self.time_attn = _MultiHeadAttentionRoPE(n_dim // 2, n_head // 2, dropout, True)
        self.add_module("time_attn", self.time_attn)
        self.spatial_attn = _MultiHeadAttentionRoPE(n_dim // 2, n_head // 2, dropout, False)
        self.add_module("spatial_attn", self.spatial_attn)
        self.pre_ff_norm = nn.RMSNorm(n_dim, eps=1e-6)
        self.add_module("pre_ff_norm", self.pre_ff_norm)
        self.ff = FeedForwardBlock(n_dim, 4, 0.0, _SELU, output_drop_p=dropout)
        self.add_module("ff", self.ff)
    def forward(self, x: Tensor) -> Tensor:
        B, C, T, D = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        h = self.pre_attn_norm(x)
        xs = h[..., D // 2:].permute(0, 2, 1, 3).reshape(B * T, C, -1)
        xt = h[..., :D // 2].reshape(B * C, T, -1)
        xs = self.spatial_attn(xs).reshape(B, T, C, -1).permute(0, 2, 1, 3)
        xt = self.time_attn(xt).reshape(B, C, T, -1)
        x = x + torch.cat([xs, xt], dim=-1)
        return x + self.ff(self.pre_ff_norm(x))
    def _get_name(self) -> str:
        return "_SpatialTemporalBlock"

class BrainOmni(EEGModuleMixin, nn.Module):
    """BrainOmni from Xiao et al (2025). Architecture only (no Hub weights).

    Sensor geometry is passed explicitly (see BrainTokenizer). As in
    braindecode, the tokenizer is frozen and the last transformer block is
    unused; ``activation`` must be SELU (the default)."""
    lm_dim: int
    n_neuro: int
    overlap_ratio: float
    tokenizer: BrainTokenizer
    projection: nn.Module
    blocks: nn.ModuleList
    _blocks: List[_SpatialTemporalBlock]
    final_layer: nn.Sequential

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, emb_dim: int = 256,
                 n_neuro: int = 16, window_length: int = 512, overlap_ratio: float = 0.25, n_filters: int = 32,
                 ratios: List[int] = [8, 4, 2], kernel_size: int = 5, last_kernel_size: int = 5,
                 tokenizer_num_heads: int = 4, codebook_dim: int = 256, codebook_size: int = 512,
                 num_quantizers: int = 4, rotation_trick: bool = True, tokenizer_drop_prob: float = 0.0,
                 lm_dim: int = 256, num_heads: int = 8, depth: int = 12, drop_prob: float = 0.1,
                 sensor_positions: Optional[Tensor] = None, sensor_types: Optional[List[int]] = None,
                 sensor_orientations: Optional[Tensor] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}.")
        if lm_dim <= 0 or num_heads <= 0 or num_heads % 2 or lm_dim % num_heads or (lm_dim // num_heads) % 2:
            raise ValueError("lm_dim and num_heads must split into an even number of even-width heads.")
        self.lm_dim = lm_dim
        self.n_neuro = n_neuro
        self.overlap_ratio = overlap_ratio
        self.tokenizer = BrainTokenizer(n_chans=self.n_chans, n_times=self.n_times, sfreq=self._sfreq, emb_dim=emb_dim,
                                        n_neuro=n_neuro, window_length=window_length, n_filters=n_filters, ratios=ratios,
                                        kernel_size=kernel_size, last_kernel_size=last_kernel_size,
                                        tokenizer_num_heads=tokenizer_num_heads, codebook_dim=codebook_dim,
                                        codebook_size=codebook_size, num_quantizers=num_quantizers,
                                        rotation_trick=rotation_trick, drop_prob=tokenizer_drop_prob,
                                        sensor_positions=sensor_positions, sensor_types=sensor_types,
                                        sensor_orientations=sensor_orientations)
        self.add_module("tokenizer", self.tokenizer)
        pj: nn.Module = nn.Identity()
        if emb_dim != lm_dim:
            pj = nn.Linear(emb_dim, lm_dim)
        self.projection = pj
        self.add_module("projection", pj)
        self._blocks = [_SpatialTemporalBlock(lm_dim, num_heads, drop_prob) for _ in range(depth)]
        self.blocks = nn.ModuleList()
        for b in self._blocks:
            self.blocks.append(b)
        self.add_module("blocks", self.blocks)
        for c in self.children():
            c.apply(_omni_init)
        self.final_layer = nn.Sequential(nn.Dropout(0.1), nn.Linear(n_neuro * lm_dim, lm_dim), _SELU(),
                                         nn.Linear(lm_dim, self.n_outputs))
        self.add_module("final_layer", self.final_layer)
        self.tokenizer.requires_grad_(False)

    def encode(self, x: Tensor) -> Tensor:
        f, _ = self.tokenizer.tokenize(x, self.overlap_ratio)
        nr = self.tokenizer.encoder.neuros.detach().to(f.dtype)
        h = self.projection(f + nr.view(1, f.shape[1], 1, -1))
        for i in range(len(self._blocks) - 1):
            h = self._blocks[i](h)
        return F.normalize(h, p=2.0, dim=-1, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.encode(x).mean(dim=2).flatten(1))

    def _get_name(self) -> str:
        return "BrainOmni"
