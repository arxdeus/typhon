# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import rotary_frequencies, rotary_angles, apply_rotary
from braindecode.functional.initialization import _param_by_name
from braindecode.functional.patches import _check_on_non_divisible, pad_or_crop_to_patches
from braindecode.modules.timm import TimmMlp, PreNormBlock

def _luna_init(m):
    # init_weights_(weights={nn.Linear: xavier_normal_}) (isinstance: also
    # MultiheadAttention.out_proj), LayerNorm -> 1/0
    n = m._get_name()
    if n in ("Linear", "NonDynamicallyQuantizableLinear"):
        nn.init.xavier_normal_(_param_by_name(m, "weight").__val__())
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

def nerf_positional_encoding(coords: Tensor, embed_size: int) -> Tensor:
    N, C, dim = coords.shape[0], coords.shape[1], coords.shape[2]
    freqs = embed_size // (2 * dim)
    left = embed_size - freqs * 2 * dim
    fb = torch.pow(torch.full([freqs], 2.0), torch.arange(freqs, dtype=torch.float32))
    sc = coords.unsqueeze(-1) * fb.view(1, 1, 1, -1)
    enc = torch.stack([torch.sin(sc), torch.cos(sc)], dim=-1).permute(0, 1, 3, 2, 4).reshape(N, C, freqs * dim * 2)
    if left > 0:
        enc = torch.cat([enc, torch.zeros(N, C, left, dtype=coords.dtype)], dim=-1)
    return enc.to(coords.dtype)

class _FrequencyFeatureEmbedder(nn.Module):
    patch_size: int
    embed_dim: int
    frequency_to_embed: TimmMlp
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        fin = 2 * (patch_size // 2 + 1)
        self.frequency_to_embed = TimmMlp(fin, 4 * fin, embed_dim)
        self.add_module("frequency_to_embed", self.frequency_to_embed)
    def forward(self, x: Tensor) -> Tensor:
        B, C, T = x.shape[0], x.shape[1], x.shape[2]
        if T % self.patch_size != 0:
            x = F.pad(x, [0, self.patch_size - T % self.patch_size])
            T = x.size(-1)
        x = x.view(B, C, T // self.patch_size, self.patch_size)
        fr = torch.fft.rfft(x, dim=-1)
        feats = torch.cat([torch.abs(fr), torch.angle(fr)], dim=-1).to(x.dtype)
        return self.frequency_to_embed(feats).flatten(1, 2)
    def _get_name(self) -> str:
        return "_FrequencyFeatureEmbedder"

class _RotaryEmbedding(nn.Module):
    freqs: Tensor
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.freqs = nn.Parameter(rotary_frequencies(dim, theta), requires_grad=False)
        self.register_parameter("freqs", self.freqs)
    def forward(self, positions: Tensor) -> Tensor:
        return rotary_angles(positions.to(self.freqs.dtype), self.freqs)
    def _get_name(self) -> str:
        return "_RotaryEmbedding"

class _RotarySelfAttentionBlock(nn.Module):
    dim: int
    num_heads: int
    scale: float
    attn_drop: float
    rotary_emb: _RotaryEmbedding
    qkv_proj: nn.Linear
    proj: nn.Linear
    proj_drop: nn.Dropout
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        hd = dim // num_heads
        self.rotary_emb = _RotaryEmbedding(hd)
        self.add_module("rotary_emb", self.rotary_emb)
        self.scale = qk_scale.__val__() if qk_scale is not None else float(hd) ** -0.5
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.add_module("qkv_proj", self.qkv_proj)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.add_module("proj", self.proj)
        self.proj_drop = nn.Dropout(proj_drop)
        self.add_module("proj_drop", self.proj_drop)
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape[0], x.shape[1], x.shape[2]
        h = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        fr = self.rotary_emb(torch.arange(N, dtype=torch.float32).to(x.dtype))
        cos, sin = fr.cos(), fr.sin()
        q = apply_rotary(h[0], cos, sin)
        k = apply_rotary(h[1], cos, sin)
        a = F.scaled_dot_product_attention(q, k, h[2], dropout_p=self.attn_drop if self.training else 0.0, scale=self.scale)
        return self.proj_drop(self.proj(a.transpose(1, 2).flatten(2)))
    def _get_name(self) -> str:
        return "_RotarySelfAttentionBlock"

class _LunaFeedForwardBlock(nn.Module):
    fc1: nn.Linear
    activation: nn.GELU
    dropout1: nn.Dropout
    dropout2: nn.Dropout
    fc2: nn.Linear
    norm: nn.LayerNorm
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.add_module("fc1", self.fc1)
        self.activation = nn.GELU()
        self.add_module("activation", self.activation)
        self.dropout1 = nn.Dropout(dropout)
        self.add_module("dropout1", self.dropout1)
        self.dropout2 = nn.Dropout(dropout)
        self.add_module("dropout2", self.dropout2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.add_module("fc2", self.fc2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.add_module("norm", self.norm)
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout2(self.fc2(self.norm(self.dropout1(self.activation(self.fc1(x))))))
    def _get_name(self) -> str:
        return "_FeedForwardBlock"

class _ClassificationHeadWithQueries(nn.Module):
    embed_dim: int
    decoder_attn: nn.MultiheadAttention
    decoder_ffn: TimmMlp
    learned_agg: Tensor
    def __init__(self, input_dim: int = 8, embed_dim: int = 768, num_queries: int = 8, num_heads: int = 8,
                 num_classes: int = 2, drop_decoder: float = 0.15, drop_ffn: float = 0.15):
        super().__init__()
        self.embed_dim = embed_dim * num_queries
        E = self.embed_dim
        self.decoder_attn = nn.MultiheadAttention(E, num_heads, batch_first=True, dropout=drop_decoder)
        self.add_module("decoder_attn", self.decoder_attn)
        self.decoder_ffn = TimmMlp(E, E * 4, num_classes, drop=drop_ffn)
        self.add_module("decoder_ffn", self.decoder_ffn)
        self.learned_agg = nn.Parameter(torch.randn(1, 1, E))
        self.register_parameter("learned_agg", self.learned_agg)
    def forward(self, x: Tensor) -> Tensor:
        q = self.learned_agg.repeat(x.shape[0], 1, 1)
        y, _ = self.decoder_attn.attend(q, x, x, need_weights=False)
        return self.decoder_ffn(y[:, 0, :])
    def _get_name(self) -> str:
        return "_ClassificationHeadWithQueries"

class _CrossAttentionBlock(nn.Module):
    num_queries: int
    query_embed: Tensor
    cross_attention: nn.MultiheadAttention
    temperature: Tensor
    ffn: TimmMlp
    keys_norm: nn.LayerNorm
    values_norm: nn.LayerNorm
    queries_norm: nn.LayerNorm
    query_self_attn: nn.TransformerEncoder
    def __init__(self, num_queries: int, input_embed_dim: int, output_embed_dim: int, num_heads: int,
                 dropout_p: float = 0.1, ff_dim: int = 2048):
        super().__init__()
        D = input_embed_dim
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, D))
        self.register_parameter("query_embed", self.query_embed)
        self.cross_attention = nn.MultiheadAttention(D, num_heads, dropout=dropout_p, batch_first=True)
        self.add_module("cross_attention", self.cross_attention)
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.register_parameter("temperature", self.temperature)
        self.ffn = TimmMlp(D, ff_dim, output_embed_dim, drop=dropout_p)
        self.add_module("ffn", self.ffn)
        self.keys_norm = nn.LayerNorm(D)
        self.add_module("keys_norm", self.keys_norm)
        self.values_norm = nn.LayerNorm(D)
        self.add_module("values_norm", self.values_norm)
        self.queries_norm = nn.LayerNorm(D)
        self.add_module("queries_norm", self.queries_norm)
        self.query_self_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, nhead=num_heads, activation="gelu",
                                                                                 dim_feedforward=ff_dim, batch_first=True,
                                                                                 norm_first=True), num_layers=3)
        self.add_module("query_self_attn", self.query_self_attn)
    def initialize_weights(self):
        nn.init.orthogonal_(self.query_embed, gain=1.0)
        self.apply(_luna_init)
    def forward(self, x: Tensor) -> Tensor:
        q = self.queries_norm(self.query_embed.repeat(x.size(0), 1, 1))
        y, _ = self.cross_attention.attend(q, self.keys_norm(x), self.values_norm(x), need_weights=False)
        y = self.ffn(y) + y
        return self.query_self_attn(y)
    def _get_name(self) -> str:
        return "_CrossAttentionBlock"

class _PatchEmbedNetwork(nn.Module):
    patch_size: int
    embed_dim: int
    on_non_divisible: str
    proj_in: nn.Sequential
    def __init__(self, embed_dim: int = 64, patch_size: int = 40, on_non_divisible: str = "pad"):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.on_non_divisible = on_non_divisible
        oc = embed_dim // 4
        k = patch_size // 2
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, oc, kernel_size=(1, k - 1), stride=(1, k // 2), padding=(0, k // 2 - 1)),
            nn.GroupNorm(4, oc), nn.GELU(),
            nn.Conv2d(oc, oc, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)), nn.GroupNorm(4, oc), nn.GELU(),
            nn.Conv2d(oc, oc, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)), nn.GroupNorm(4, oc), nn.GELU())
        self.add_module("proj_in", self.proj_in)
    def forward(self, x: Tensor) -> Tensor:
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        x = x.reshape(x.shape[0], x.shape[1], -1, self.patch_size).flatten(1, 2).unsqueeze(1)
        return self.proj_in(x).permute(0, 2, 3, 1).flatten(2)
    def _get_name(self) -> str:
        return "_PatchEmbedNetwork"

class LUNA(EEGModuleMixin, nn.Module):
    """LUNA from Doner et al (2025). Architecture only (no Hub weights).

    Classification mode only (n_outputs > 0): the masked-reconstruction
    decoder (n_outputs=0) needs nn.TransformerDecoder and is not ported.
    Default channel locations are the linspace template; pass explicit
    locations with ``forward_locs(x, channel_locations)``."""
    num_classes: int
    embed_dim: int
    num_queries: int
    patch_size: int
    patch_embed_size: int
    on_non_divisible: str
    mask_token: Tensor
    patch_embed: _PatchEmbedNetwork
    freq_embed: _FrequencyFeatureEmbedder
    channel_location_embedder: TimmMlp
    cross_attn: _CrossAttentionBlock
    blocks: nn.ModuleList
    _blocks: List[PreNormBlock[_RotarySelfAttentionBlock, _LunaFeedForwardBlock]]
    norm: nn.LayerNorm
    final_layer: _ClassificationHeadWithQueries

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None, n_times: Optional[int] = None,
                 sfreq: Optional[float] = None, chs_info: Optional[List[Dict[str, str]]] = None,
                 input_window_seconds: Optional[float] = None, patch_size: int = 40, num_queries: int = 4,
                 embed_dim: int = 64, depth: int = 8, num_heads: int = 2, mlp_ratio: float = 4.0,
                 drop_path: float = 0.0, drop_prob_chan: float = 0.0, attn_drop: float = 0.0, activation = nn.GELU,
                 on_non_divisible: str = "pad"):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        _check_on_non_divisible(on_non_divisible)
        self.mapping = {"cross_attn.temparature": "cross_attn.temperature"}
        for s in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "norm.weight", "norm.bias"):
            self.mapping["channel_location_embedder.0." + s] = "channel_location_embedder." + s
        self.num_classes = self.n_outputs
        if self.num_classes <= 0:
            raise NotImplementedError("LUNA reconstruction mode (n_outputs=0) is not ported.")
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.patch_size = patch_size
        self.patch_embed_size = embed_dim
        self.on_non_divisible = on_non_divisible
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_parameter("mask_token", self.mask_token)
        self.patch_embed = _PatchEmbedNetwork(embed_dim, patch_size, on_non_divisible)
        self.add_module("patch_embed", self.patch_embed)
        self.freq_embed = _FrequencyFeatureEmbedder(patch_size, embed_dim)
        self.add_module("freq_embed", self.freq_embed)
        self.channel_location_embedder = TimmMlp(embed_dim, embed_dim * 2, embed_dim, act_layer=activation,
                                                 drop=drop_prob_chan)
        self.add_module("channel_location_embedder", self.channel_location_embedder)
        self.cross_attn = _CrossAttentionBlock(num_queries, embed_dim, embed_dim, num_heads,
                                               ff_dim=int(mlp_ratio * float(embed_dim)))
        self.add_module("cross_attn", self.cross_attn)
        D = embed_dim * num_queries
        self._blocks = [PreNormBlock(D, _RotarySelfAttentionBlock(D, num_heads * num_queries, True, None, attn_drop, 0.0),
                                     _LunaFeedForwardBlock(D, int(float(D) * mlp_ratio), 0.0), drop_path=drop_path)
                        for _ in range(depth)]
        self.blocks = nn.ModuleList()
        for b in self._blocks:
            self.blocks.append(b)
        self.add_module("blocks", self.blocks)
        self.norm = nn.LayerNorm(D)
        self.add_module("norm", self.norm)
        self.final_layer = _ClassificationHeadWithQueries(patch_size, embed_dim, num_queries, num_heads, self.num_classes)
        self.add_module("final_layer", self.final_layer)
        self.mask_token.requires_grad = False
        self.cross_attn.initialize_weights()
        nn.init.trunc_normal_(self.mask_token, std=0.02, a=-0.02, b=0.02)
        self.apply(_luna_init)
        with torch.no_grad():
            for i in range(len(self._blocks)):
                s = math.sqrt(2.0 * float(i + 1))
                self._blocks[i].attn.proj.weight.div_(s)
                self._blocks[i].mlp.fc2.weight.div_(s)

    def channel_location_template(self, n: int) -> Tensor:
        pos = torch.linspace(-1.0, 1.0, n)
        z = torch.zeros_like(pos)
        return torch.stack([pos, z, z], dim=-1)

    def forward_locs(self, x: Tensor, channel_locations: Tensor) -> Tensor:
        B, C = x.shape[0], x.shape[1]
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        npc = x.shape[-1] // self.patch_size
        xp = self.patch_embed(x) + self.freq_embed(x)
        cl = channel_locations
        cmin = cl.min(dim=1, keepdim=True)[0]
        cmax = cl.max(dim=1, keepdim=True)[0]
        cl = (cl - cmin) / (cmax - cmin + 1e-8)
        cle = self.channel_location_embedder(nerf_positional_encoding(cl, self.patch_embed_size).to(x.dtype))
        xt = xp.unflatten(1, [C, -1]).transpose(1, 2).flatten(0, 1)
        xt = xt + torch.repeat_interleave(cle, npc, dim=0)
        h = self.cross_attn(xt).unflatten(0, [B, -1]).flatten(2)
        for b in self._blocks:
            h = b(h)
        return self.final_layer(self.norm(h))

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_locs(x, self.channel_location_template(x.shape[1]).to(x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1))

    def _get_name(self) -> str:
        return "LUNA"
