# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import sinusoidal_positional_encoding
from braindecode.functional.initialization import _param_by_name

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

class FourierEmb(nn.Module):
    n_dims: int
    margin: float
    pos: Tensor
    def __init__(self, dimension: int = 2048, margin: float = 0.2):
        super().__init__()
        nf = math.sqrt(float(dimension) / 2.0)
        n = int(round(nf))
        if abs(nf - float(n)) > 1e-6:
            raise ValueError("(dimension / 2) ** (1 / 2) must be an integer.")
        self.n_dims = 2
        self.margin = margin
        self.pos = torch.arange(n).to(torch.float32) * (2.0 * math.pi) / (1.0 + 2.0 * margin)
        self.register_buffer("pos", self.pos)
    def forward(self, p: Tensor) -> Tensor:
        p = p + self.margin
        l = p.unsqueeze(-2) * self.pos.view(-1, 1)
        g = (l[..., 0].unsqueeze(-1) + l[..., 1].unsqueeze(-2)).flatten(p.dim() - 1)
        return torch.cat([torch.cos(g), torch.sin(g)], dim=-1)
    def _get_name(self) -> str:
        return "FourierEmb"

class ChannelMerger(nn.Module):
    heads: Tensor
    embedding: FourierEmb
    n_dims: int
    dropout: float
    invalid_value: float
    def __init__(self, out_channels: int = 270, pos_dim: int = 2048, dropout: float = 0.2, invalid_value: float = -0.1):
        super().__init__()
        self.heads = nn.Parameter(torch.randn(out_channels, pos_dim))
        self.register_parameter("heads", self.heads)
        with torch.no_grad():
            self.heads.div_(float(pos_dim) ** 0.5)
        self.embedding = FourierEmb(pos_dim)
        self.add_module("embedding", self.embedding)
        self.n_dims = 2
        self.dropout = dropout
        self.invalid_value = invalid_value
    def merge(self, x: Tensor, positions: Tensor) -> Tensor:
        B, C = positions.shape[0], positions.shape[1]
        emb = self.embedding(positions)
        off = torch.zeros(B, C, dtype=x.dtype)
        off = off.masked_fill((positions == self.invalid_value).all(dim=-1), -math.inf)
        if self.training and self.dropout != 0.0:
            ctr = torch.rand(self.n_dims)
            ban = torch.linalg.vector_norm(positions[:, :, :self.n_dims] - ctr, 2, dim=-1) <= self.dropout
            off = off.masked_fill(ban, -math.inf)
        sc = torch.matmul(self.heads.unsqueeze(0).expand(B, -1, -1), emb.transpose(1, 2)) + off.unsqueeze(1)
        w = torch.nan_to_num(torch.softmax(sc, dim=2))
        return torch.matmul(w, x)
    def _get_name(self) -> str:
        return "ChannelMerger"

class _DConvBlock(nn.Module):
    conv: nn.Conv1d
    act: Optional[nn.Module]
    d: int
    half: int
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, activation, last: bool):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=(kernel_size // 2) * dilation, dilation=dilation)
        self.add_module("conv", self.conv)
        self.d = dilation
        self.half = kernel_size // 2
        self.act = None
        if not last:
            a = activation()
            self.act = a
            self.add_module("act", a)
    def forward(self, x: Tensor) -> Tensor:
        reach = (x.shape[-1] - 1) // self.d
        if reach < self.half:
            w = self.conv.weight[:, :, self.half - reach:self.half + reach + 1]
            x = F.conv1d(x, w, self.conv.bias, padding=reach * self.d, dilation=self.d)
        else:
            x = self.conv(x)
        if self.act is not None:
            x = self.act.__val__()(x)
        return x
    def _get_name(self) -> str:
        return "_ConvBlock"

class SimpleConv(nn.Module):
    merger: Optional[ChannelMerger]
    initial_linear: nn.Sequential
    sequence: nn.ModuleList
    def __init__(self, in_channels: int, out_channels: int = 128, hidden: int = 512, depth: int = 10, kernel_size: int = 9,
                 dilation_growth: float = 2.5, initial_linear: int = 256, initial_depth: int = 1, drop_prob: float = 0.2,
                 activation = nn.ReLU, merger: Optional[ChannelMerger] = None):
        super().__init__()
        if dilation_growth > 1 and kernel_size % 2 == 0:
            raise ValueError("Odd kernel required with dilation.")
        self.merger = merger
        if merger is not None:
            self.add_module("merger", merger.__val__())
        ci = merger.__val__().heads.shape[0] if merger is not None else in_channels
        self.initial_linear = nn.Sequential()
        for i in range(initial_depth):
            self.initial_linear.add_module(str(i), nn.Conv1d(ci, initial_linear, 1))
            ci = initial_linear
        self.add_module("initial_linear", self.initial_linear)
        sizes = [initial_linear] + [hidden] * (depth - 1) + [out_channels]
        self.sequence = nn.ModuleList()
        dil = 1.0
        for i in range(depth):
            self.sequence.append(_DConvBlock(sizes[i], sizes[i + 1], kernel_size, int(dil), activation, i == depth - 1))
            dil *= dilation_growth
        self.add_module("sequence", self.sequence)
    def run(self, x: Tensor, positions: Optional[Tensor]) -> Tensor:
        L = x.shape[-1]
        if self.merger is not None:
            if positions is None:
                raise ValueError("SimpleConv with a merger requires positions.")
            x = self.merger.__val__().merge(x, positions.__val__())
        x = self.initial_linear(x)
        for b in self.sequence.children():
            x = b(x)
        return x[..., :L]
    def _get_name(self) -> str:
        return "SimpleConv"

class GatedLinearUnit(nn.Module):
    activation: nn.Module
    def __init__(self, activation = nn.GELU):
        super().__init__()
        self.activation = activation()
        self.add_module("activation", self.activation)
    def forward(self, x: Tensor) -> Tensor:
        c = x.chunk(2, dim=-1)
        return c[0] * self.activation(c[1])
    def _get_name(self) -> str:
        return "GatedLinearUnit"

class _DFeedForward(nn.Module):
    net: nn.Sequential
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GatedLinearUnit(), nn.Linear(dim * mult, dim))
        self.add_module("net", self.net)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    def _get_name(self) -> str:
        return "_FeedForward"

class _DAttention(nn.Module):
    heads: int
    to_q: nn.Linear
    to_kv: nn.Linear
    to_out: nn.Linear
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, heads: int = 2, dim_head: int = 64):
        super().__init__()
        inner = dim_head * heads
        cd = context_dim.__val__() if context_dim is not None else query_dim
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.add_module("to_q", self.to_q)
        self.to_kv = nn.Linear(cd, inner * 2, bias=False)
        self.add_module("to_kv", self.to_kv)
        self.to_out = nn.Linear(inner, query_dim)
        self.add_module("to_out", self.to_out)
    def attend(self, x: Tensor, ctx: Tensor) -> Tensor:
        q = self.to_q(x).unflatten(-1, [self.heads, -1]).transpose(1, 2)
        kv = self.to_kv(ctx).chunk(2, dim=-1)
        k = kv[0].unflatten(-1, [self.heads, -1]).transpose(1, 2)
        v = kv[1].unflatten(-1, [self.heads, -1]).transpose(1, 2)
        return self.to_out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))
    def forward(self, x: Tensor) -> Tensor:
        return self.attend(x, x)
    def _get_name(self) -> str:
        return "_Attention"

class _DPreNorm(nn.Module):
    fn: nn.Module
    norm: nn.LayerNorm
    norm_context: Optional[nn.LayerNorm]
    _attn: Optional[_DAttention]
    def __init__(self, dim: int, fn: nn.Module, context_dim: Optional[int] = None, attn: Optional[_DAttention] = None):
        super().__init__()
        self.fn = fn
        self.add_module("fn", fn)
        self.norm = nn.LayerNorm(dim)
        self.add_module("norm", self.norm)
        self.norm_context = None
        if context_dim is not None:
            nc = nn.LayerNorm(context_dim.__val__())
            self.norm_context = nc
            self.add_module("norm_context", nc)
        self._attn = attn
    def apply_ctx(self, x: Tensor, ctx: Tensor) -> Tensor:
        return self._attn.__val__().attend(self.norm(x), self.norm_context.__val__()(ctx))
    def forward(self, x: Tensor) -> Tensor:
        return self.fn(self.norm(x))
    def _get_name(self) -> str:
        return "_PreNorm"

def _fourier_encode(x: Tensor, max_freq: float, nb: int) -> Tensor:
    x = x.unsqueeze(-1)
    o = x
    sc = torch.linspace(1.0, max_freq / 2.0, nb).to(x.dtype).reshape([1] * (x.dim() - 1) + [nb])
    x = x * sc * math.pi
    return torch.cat([torch.cat([x.sin(), x.cos()], dim=-1), o], dim=-1)

class Perceiver(nn.Module):
    max_freq: float
    num_freq_bands: int
    latents: Tensor
    layers: nn.ModuleList
    _cross: List[_DPreNorm]
    _cff: List[_DPreNorm]
    _sa: List[_DPreNorm]
    _sff: List[_DPreNorm]
    def __init__(self, input_dim: int = 128, num_latents: int = 256, latent_dim: int = 128, depth: int = 6,
                 cross_heads: int = 2, latent_heads: int = 2, cross_dim_head: int = 64, latent_dim_head: int = 64,
                 max_freq: float = 10.0, num_freq_bands: int = 6):
        super().__init__()
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        cd = input_dim + num_freq_bands * 2 + 1
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.register_parameter("latents", self.latents)
        with torch.no_grad():
            self.latents.copy_(sinusoidal_positional_encoding(num_latents, latent_dim))
        self.layers = nn.ModuleList()
        self._cross = []
        self._cff = []
        self._sa = []
        self._sff = []
        for _ in range(depth):
            ca = _DAttention(latent_dim, cd, cross_heads, cross_dim_head)
            c = _DPreNorm(latent_dim, ca, cd, ca)
            cf = _DPreNorm(latent_dim, _DFeedForward(latent_dim))
            sa = _DPreNorm(latent_dim, _DAttention(latent_dim, None, latent_heads, latent_dim_head))
            sf = _DPreNorm(latent_dim, _DFeedForward(latent_dim))
            inner = nn.ModuleList()
            inner.append(sa)
            inner.append(sf)
            outer = nn.ModuleList()
            outer.append(inner)
            l = nn.ModuleList()
            l.append(c)
            l.append(cf)
            l.append(outer)
            self.layers.append(l)
            self._cross.append(c)
            self._cff.append(cf)
            self._sa.append(sa)
            self._sff.append(sf)
        self.add_module("layers", self.layers)
        self.add_module("to_logits", nn.Identity())
    def forward(self, data: Tensor) -> Tensor:
        B, T = data.shape[0], data.shape[1]
        pos = torch.linspace(-1.0, 1.0, T).to(data.dtype).unsqueeze(-1)
        ep = _fourier_encode(pos, self.max_freq, self.num_freq_bands).flatten(-2).unsqueeze(0).expand(B, -1, -1)
        data = torch.cat([data, ep], dim=-1)
        x = self.latents.unsqueeze(0).expand(B, -1, -1)
        for i in range(len(self._cross)):
            x = self._cross[i].apply_ctx(x, data) + x
            x = self._cff[i](x) + x
            x = self._sa[i](x) + x
            x = self._sff[i](x) + x
        return x
    def _get_name(self) -> str:
        return "Perceiver"

class _DecoderLayer(nn.Module):
    self_attn: _DAttention
    cross_attn: _DAttention
    ff: _DFeedForward
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    norm3: nn.LayerNorm
    drop: nn.Dropout
    def __init__(self, dim: int, heads: int, drop_prob: float):
        super().__init__()
        self.self_attn = _DAttention(dim, None, heads, dim // heads)
        self.add_module("self_attn", self.self_attn)
        self.cross_attn = _DAttention(dim, dim, heads, dim // heads)
        self.add_module("cross_attn", self.cross_attn)
        self.ff = _DFeedForward(dim)
        self.add_module("ff", self.ff)
        self.norm1 = nn.LayerNorm(dim)
        self.add_module("norm1", self.norm1)
        self.norm2 = nn.LayerNorm(dim)
        self.add_module("norm2", self.norm2)
        self.norm3 = nn.LayerNorm(dim)
        self.add_module("norm3", self.norm3)
        self.drop = nn.Dropout(drop_prob)
        self.add_module("drop", self.drop)
    def layer(self, q: Tensor, mem: Tensor) -> Tensor:
        q = q + self.drop(self.self_attn(self.norm1(q)))
        q = q + self.drop(self.cross_attn.attend(self.norm2(q), mem))
        return q + self.drop(self.ff(self.norm3(q)))
    def _get_name(self) -> str:
        return "_DecoderLayer"

@tuple
class DanceEvents:
    cls: Tensor
    start: Tensor
    end: Tensor
    dense: Tensor

class DanceDetrDecoder(nn.Module):
    input_proj: nn.Linear
    query_embed: Tensor
    layers: nn.ModuleList
    _layers: List[_DecoderLayer]
    class_head: nn.Linear
    start_head: nn.Linear
    end_head: nn.Linear
    def __init__(self, input_dim: int = 128, dim: int = 256, depth: int = 4, heads: int = 4, n_queries: int = 100,
                 n_outputs: int = 4, drop_prob: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.add_module("input_proj", self.input_proj)
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, dim))
        self.register_parameter("query_embed", self.query_embed)
        self._layers = [_DecoderLayer(dim, heads, drop_prob) for _ in range(depth)]
        self.layers = nn.ModuleList()
        for l in self._layers:
            self.layers.append(l)
        self.add_module("layers", self.layers)
        self.class_head = nn.Linear(dim, n_outputs)
        self.add_module("class_head", self.class_head)
        self.start_head = nn.Linear(dim, 1)
        self.add_module("start_head", self.start_head)
        self.end_head = nn.Linear(dim, 1)
        self.add_module("end_head", self.end_head)
    def decode(self, mem: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, T = mem.shape[0], mem.shape[1]
        mem = self.input_proj(mem)
        mem = mem + sinusoidal_positional_encoding(T, mem.shape[-1]).to(mem.dtype).unsqueeze(0)
        x = self.query_embed.expand(B, -1, -1)
        for l in self._layers:
            x = l.layer(x, mem)
        return (self.class_head(x), torch.sigmoid(self.start_head(x)).squeeze(-1), torch.sigmoid(self.end_head(x)).squeeze(-1))
    def _get_name(self) -> str:
        return "DanceDetrDecoder"

class DANCE(EEGModuleMixin, nn.Module):
    """DANCE from Klein et al (2025). Architecture only (no Hub weights).

    braindecode reads electrode xyz from chs_info; here pass
    ``channel_positions`` (n_chans, 3); the xy min-max normalization is the
    same. Without positions (or with all-zero positions) the ChannelMerger is
    disabled, as in braindecode. ``detect`` returns (class, start, end,
    dense)."""
    num_latents: int
    n_queries: int
    embed_dim: int
    use_channel_merger: bool
    min_n_times: int
    channel_positions: Optional[Tensor]
    input_drop: nn.Dropout
    conv: SimpleConv
    perceiver: Perceiver
    decoder: DanceDetrDecoder
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, n_queries: int = 100,
                 use_channel_merger: bool = True, n_virtual_channels: int = 270, fourier_emb_dim: int = 2048,
                 merger_drop_prob: float = 0.2, embed_dim: int = 128, conv_hidden: int = 512, conv_depth: int = 10,
                 conv_kernel_size: int = 9, conv_dilation_growth: float = 2.5, conv_initial_linear: int = 256,
                 conv_initial_depth: int = 1, conv_drop_prob: float = 0.2, num_latents: int = 256,
                 perceiver_depth: int = 6, cross_attn_heads: int = 2, latent_attn_heads: int = 2,
                 cross_dim_head: int = 64, latent_dim_head: int = 64, max_freq: float = 10.0, num_freq_bands: int = 6,
                 decoder_dim: int = 256, decoder_depth: int = 4, decoder_heads: int = 4, activation = nn.GELU,
                 drop_prob: float = 0.1, channel_positions: Optional[Tensor] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.num_latents = num_latents
        self.n_queries = n_queries
        self.embed_dim = embed_dim
        valid = False
        if channel_positions is not None:
            cp = channel_positions.__val__()
            valid = cp.numel() > 0 and bool(torch.isfinite(cp).all().item()) and torch.abs(cp).max().item() > 1e-8
        if use_channel_merger and not valid:
            print("DANCE: no usable electrode positions; disabling the ChannelMerger (use_channel_merger=False).")
            use_channel_merger = False
        self.use_channel_merger = use_channel_merger
        self.channel_positions = None
        merger: Optional[ChannelMerger] = None
        if use_channel_merger:
            xy = channel_positions.__val__().to(torch.float64)[:, :2]
            mn = xy.min(dim=0).values
            mx = xy.max(dim=0).values
            p = ((xy - mn) / torch.clamp(mx - mn, min=1e-9)).to(torch.float32)
            self.channel_positions = p
            self.register_buffer("channel_positions", p, persistent=False)
            merger = ChannelMerger(n_virtual_channels, fourier_emb_dim, merger_drop_prob)
        self.input_drop = nn.Dropout(drop_prob)
        self.add_module("input_drop", self.input_drop)
        self.conv = SimpleConv(self.n_chans, embed_dim, conv_hidden, conv_depth, conv_kernel_size, conv_dilation_growth,
                               conv_initial_linear, conv_initial_depth, conv_drop_prob, nn.ReLU, merger)
        self.add_module("conv", self.conv)
        self.min_n_times = conv_kernel_size
        self.perceiver = Perceiver(embed_dim, num_latents, embed_dim, perceiver_depth, cross_attn_heads, latent_attn_heads,
                                   cross_dim_head, latent_dim_head, max_freq, num_freq_bands)
        self.add_module("perceiver", self.perceiver)
        self.decoder = DanceDetrDecoder(embed_dim, decoder_dim, decoder_depth, decoder_heads, n_queries, self.n_outputs, drop_prob)
        self.add_module("decoder", self.decoder)
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        self.apply(_vit_init)

    def _encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-2] != self.n_chans:
            raise ValueError(f"Expected input (batch, {self.n_chans}, time), got {x.shape}.")
        if x.shape[-1] < self.min_n_times:
            raise ValueError(f"Need at least {self.min_n_times} samples, got {x.shape[-1]}.")
        x = self.input_drop(x)
        pos: Optional[Tensor] = None
        if self.channel_positions is not None:
            pos = self.channel_positions.__val__().unsqueeze(0).expand(x.size(0), -1, -1)
        x = self.conv.run(x, pos)
        return self.perceiver(x.transpose(2, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self._encode(x))

    def detect(self, x: Tensor) -> DanceEvents:
        lat = self._encode(x)
        c, s, e = self.decoder.decode(lat)
        return DanceEvents(c, s, e, self.final_layer(lat))

    def _get_name(self) -> str:
        return "DANCE"
