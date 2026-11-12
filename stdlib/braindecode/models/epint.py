# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import apply_rotary, rotary_angles, rotary_frequencies
from braindecode.modules import FeedForwardBlock

class _RoPEAttention(nn.Module):
    n_heads: int
    w_q: nn.Linear
    w_k: nn.Linear
    w_v: nn.Linear
    w_concat: nn.Linear
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.add_module("w_q", self.w_q)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.add_module("w_k", self.w_k)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.add_module("w_v", self.w_v)
        self.w_concat = nn.Linear(embed_dim, embed_dim)
        self.add_module("w_concat", self.w_concat)
    def attn(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        q = apply_rotary(self.w_q(x).unflatten(-1, [self.n_heads, -1]).transpose(1, 2), cos, sin)
        k = apply_rotary(self.w_k(x).unflatten(-1, [self.n_heads, -1]).transpose(1, 2), cos, sin)
        v = self.w_v(x).unflatten(-1, [self.n_heads, -1]).transpose(1, 2)
        return self.w_concat(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))
    def _get_name(self) -> str:
        return "_RoPEAttention"

class _EpiNTLayer(nn.Module):
    attention: _RoPEAttention
    norm1: nn.LayerNorm
    dropout1: nn.Dropout
    ffn: FeedForwardBlock
    norm2: nn.LayerNorm
    dropout2: nn.Dropout
    def __init__(self, n_tokens: int, embed_dim: int, n_heads: int, ffn_dim: int, drop_prob: float, activation):
        super().__init__()
        self.attention = _RoPEAttention(embed_dim, n_heads)
        self.add_module("attention", self.attention)
        self.norm1 = nn.LayerNorm([n_tokens, embed_dim])
        self.add_module("norm1", self.norm1)
        self.dropout1 = nn.Dropout(drop_prob)
        self.add_module("dropout1", self.dropout1)
        self.ffn = FeedForwardBlock(embed_dim, 1, drop_prob, activation, hidden_features=ffn_dim, output_drop_p=drop_prob)
        self.add_module("ffn", self.ffn)
        self.norm2 = nn.LayerNorm([n_tokens, embed_dim])
        self.add_module("norm2", self.norm2)
        self.dropout2 = nn.Dropout(drop_prob)
        self.add_module("dropout2", self.dropout2)
    def layer(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = self.norm1(x + self.dropout1(self.attention.attn(x, cos, sin)))
        return self.norm2(x + self.dropout2(self.ffn(x)))
    def _get_name(self) -> str:
        return "_EpiNTLayer"

class EpiNT(EEGModuleMixin, nn.Module):
    """EpiNT from Yuan et al (2025). Architecture only (no Hub weights).

    The optional ``ch_mask`` of braindecode's forward is not ported."""
    patch_size: int
    n_patches: int
    instance_norm: nn.InstanceNorm1d
    patch_embedding: nn.Linear
    cls_token: Tensor
    rope_cos: Tensor
    rope_sin: Tensor
    encoder: nn.ModuleList
    _layers: List[_EpiNTLayer]
    final_layer: nn.Sequential

    def __init__(self, patch_size: int = 256, embed_dim: int = 512, n_layers: int = 6, n_heads: int = 8,
                 ffn_dim: int = 2048, drop_prob: float = 0.1, head_drop_prob: float = 0.1, rope_theta: float = 10000.0,
                 activation = nn.ReLU, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if embed_dim % n_heads or (embed_dim // n_heads) % 2:
            raise ValueError(f"embed_dim ({embed_dim}) must split into n_heads ({n_heads}) heads of even width for RoPE.")
        self.patch_size = patch_size
        self.n_patches = self.n_times // patch_size
        if self.n_patches < 1:
            raise ValueError(f"n_times={self.n_times} is shorter than patch_size={patch_size}.")
        nt = self.n_patches + 1
        self.instance_norm = nn.InstanceNorm1d(1, affine=True)
        self.add_module("instance_norm", self.instance_norm)
        self.patch_embedding = nn.Linear(patch_size, embed_dim)
        self.add_module("patch_embedding", self.patch_embedding)
        self.cls_token = nn.Parameter(torch.empty(embed_dim))
        self.register_parameter("cls_token", self.cls_token)
        nn.init.normal_(self.cls_token, std=0.1)
        ang = rotary_angles(torch.arange(nt).float(), rotary_frequencies(embed_dim // n_heads, rope_theta))
        self.rope_cos = ang.cos()
        self.register_buffer("rope_cos", self.rope_cos, persistent=False)
        self.rope_sin = ang.sin()
        self.register_buffer("rope_sin", self.rope_sin, persistent=False)
        self._layers = [_EpiNTLayer(nt, embed_dim, n_heads, ffn_dim, drop_prob, activation) for _ in range(n_layers)]
        self.encoder = nn.ModuleList()
        for l in self._layers:
            self.encoder.append(l)
        self.add_module("encoder", self.encoder)
        head = nn.Linear(embed_dim, self.n_outputs)
        self.final_layer = nn.Sequential(nn.Dropout(head_drop_prob), head)
        self.add_module("final_layer", self.final_layer)
        nn.init.kaiming_uniform_(head.weight)
        nn.init.zeros_(head.bias.__val__())
        self.mapping = {"norm.weight": "instance_norm.weight", "norm.bias": "instance_norm.bias",
                        "embed.proj.weight": "patch_embedding.weight", "embed.proj.bias": "patch_embedding.bias",
                        "embed.cls_embed": "cls_token", "head.weight": "final_layer.1.weight",
                        "head.bias": "final_layer.1.bias"}
        for key in self.encoder.state_dict().keys():
            up = key.replace("ffn.0.", "ffn.linear1.").replace("ffn.3.", "ffn.linear2.")
            self.mapping["transformer_encoder." + up] = "encoder." + key

    def forward_features(self, x: Tensor) -> Tensor:
        if x.shape[2] != self.n_times:
            raise ValueError(f"EpiNT expects {self.n_times} samples, got {x.shape[2]}.")
        C = x.shape[1]
        z = self.instance_norm(x.flatten(0, 1).unsqueeze(1))
        z = z[..., :self.n_patches * self.patch_size].reshape(z.shape[0], self.n_patches, self.patch_size)
        z = self.patch_embedding(z)
        z = torch.cat([self.cls_token.expand(z.shape[0], 1, -1), z], dim=1)
        for l in self._layers:
            z = l.layer(z, self.rope_cos, self.rope_sin)
        return z[:, 0].unflatten(0, [-1, C]).mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x))

    def _get_name(self) -> str:
        return "EpiNT"
