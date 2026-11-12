# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.base import EEGModuleMixin
from braindecode.functional.initialization import _param_by_name

def _normal_linear(std_emb: float):
    def f(m):
        n = m._get_name()
        if n == "Linear" or (n == "Embedding" and std_emb > 0.0):
            nn.init.normal_(_param_by_name(m, "weight").__val__(), mean=0.0, std=0.02 if n == "Linear" else std_emb)
            b = _param_by_name(m, "bias")
            if b is not None:
                nn.init.zeros_(b.__val__())
    return f

class _PBTLayerNorm(nn.Module):
    weight: Tensor
    bias: Optional[Tensor]
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.register_parameter("weight", self.weight)
        self.bias = None
        if bias:
            b = nn.Parameter(torch.zeros(ndim))
            self.bias = b
            self.register_parameter("bias", b)
        else:
            self.register_parameter("bias", None)
    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, [self.weight.shape[0]], self.weight, self.bias, 1e-5)
    def _get_name(self) -> str:
        return "_LayerNorm"

class _MHSA(nn.Module):
    attn: nn.Linear
    proj: nn.Linear
    attn_drop_prob: nn.Dropout
    resid_drop_prob: nn.Dropout
    n_head: int
    embed_dim: int
    drop_prob: float
    def __init__(self, embed_dim: int, n_head: int, bias: bool, drop_prob: float = 0.0):
        super().__init__()
        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"
        self.attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.add_module("attn", self.attn)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.add_module("proj", self.proj)
        self.attn_drop_prob = nn.Dropout(drop_prob)
        self.add_module("attn_drop_prob", self.attn_drop_prob)
        self.resid_drop_prob = nn.Dropout(drop_prob)
        self.add_module("resid_drop_prob", self.resid_drop_prob)
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.drop_prob = drop_prob
    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape[0], x.shape[1], x.shape[2]
        s = self.attn(x).split(self.embed_dim, dim=2)
        q = s[0].view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = s[1].view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = s[2].view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_prob if self.training else 0.0)
        return self.resid_drop_prob(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))
    def _get_name(self) -> str:
        return "_MHSA"

class _PBTFeedForward(nn.Module):
    proj_in: nn.Linear
    activation: nn.Module
    proj: nn.Linear
    drop_prob: nn.Dropout
    drop_prob1: nn.Dropout
    def __init__(self, embed_dim: int, dim_feedforward: int, drop_prob: float = 0.0, bias: bool = False, activation = nn.GELU):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, dim_feedforward, bias=bias)
        self.add_module("proj_in", self.proj_in)
        self.activation = activation()
        self.add_module("activation", self.activation)
        self.proj = nn.Linear(dim_feedforward, embed_dim, bias=bias)
        self.add_module("proj", self.proj)
        self.drop_prob = nn.Dropout(drop_prob)
        self.add_module("drop_prob", self.drop_prob)
        self.drop_prob1 = nn.Dropout(drop_prob)
        self.add_module("drop_prob1", self.drop_prob1)
    def forward(self, x: Tensor) -> Tensor:
        return self.drop_prob(self.proj(self.drop_prob1(self.activation(self.proj_in(x)))))
    def _get_name(self) -> str:
        return "_FeedForward"

class _PBTEncoderLayer(nn.Module):
    layer_norm_att: _PBTLayerNorm
    mhsa: _MHSA
    layer_norm_ff: _PBTLayerNorm
    feed_forward: _PBTFeedForward
    def __init__(self, embed_dim: int, n_head: int, drop_prob: float, bias: bool, activation):
        super().__init__()
        self.layer_norm_att = _PBTLayerNorm(embed_dim, bias)
        self.add_module("layer_norm_att", self.layer_norm_att)
        self.mhsa = _MHSA(embed_dim, n_head, bias, drop_prob)
        self.add_module("mhsa", self.mhsa)
        self.layer_norm_ff = _PBTLayerNorm(embed_dim, bias)
        self.add_module("layer_norm_ff", self.layer_norm_ff)
        self.feed_forward = _PBTFeedForward(embed_dim, 4 * embed_dim, drop_prob, bias, activation)
        self.add_module("feed_forward", self.feed_forward)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mhsa(self.layer_norm_att(x))
        return x + self.feed_forward(self.layer_norm_ff(x))
    def _get_name(self) -> str:
        return "_TransformerEncoderLayer"

class _PBTEncoder(nn.Module):
    encoder_block: nn.ModuleList
    _layers: List[_PBTEncoderLayer]
    def __init__(self, num_layers: int, embed_dim: int, n_head: int, drop_prob: float, bias: bool, activation):
        super().__init__()
        self._layers = [_PBTEncoderLayer(embed_dim, n_head, drop_prob, bias, activation) for _ in range(num_layers)]
        self.encoder_block = nn.ModuleList()
        for l in self._layers:
            self.encoder_block.append(l)
        self.add_module("encoder_block", self.encoder_block)
        self.apply(_normal_linear(0.0))
        sd = 0.02 / math.sqrt(2.0 * float(num_layers))
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=sd)
    def forward(self, x: Tensor) -> Tensor:
        for l in self._layers:
            x = l(x)
        return x
    def _get_name(self) -> str:
        return "_TransformerEncoder"

class PBT(EEGModuleMixin, nn.Module):
    """Patched Brain Transformer from Klein et al (2025). Architecture only."""
    d_input: int
    embed_dim: int
    num_embeddings: int
    learnable_cls: bool
    cls_token: Tensor
    patching_projection: nn.Linear
    pos_embedding: nn.Embedding
    transformer_encoder: _PBTEncoder
    final_layer: nn.Linear

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, d_input: int = 64, embed_dim: int = 128, num_layers: int = 4,
                 num_heads: int = 4, drop_prob: float = 0.1, learnable_cls: bool = True, bias_transformer: bool = False,
                 activation = nn.GELU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.d_input = d_input
        self.embed_dim = embed_dim
        self.num_embeddings = (self.n_chans * self.n_times) // d_input
        self.learnable_cls = learnable_cls
        if learnable_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.002)
            self.register_parameter("cls_token", self.cls_token)
        else:
            self.cls_token = torch.zeros(1, 1, embed_dim)
        self.add_module("patch_signal", nn.Identity())
        self.patching_projection = nn.Linear(d_input, embed_dim, bias=False)
        self.add_module("patching_projection", self.patching_projection)
        self.pos_embedding = nn.Embedding(self.num_embeddings + 1, embed_dim)
        self.add_module("pos_embedding", self.pos_embedding)
        self.transformer_encoder = _PBTEncoder(num_layers, embed_dim, num_heads, drop_prob, bias_transformer, activation)
        self.add_module("transformer_encoder", self.transformer_encoder)
        self.final_layer = nn.Linear(embed_dim, self.n_outputs, bias=True)
        self.add_module("final_layer", self.final_layer)
        self.apply(_normal_linear(0.002))

    def forward(self, X: Tensor) -> Tensor:
        B = X.size(0)
        xp = X.reshape(B, -1)[:, :self.num_embeddings * self.d_input].reshape(B, self.num_embeddings, self.d_input)
        tok = torch.cat([self.cls_token.expand(B, 1, -1), self.patching_projection(xp)], dim=1)
        out = self.transformer_encoder(tok + self.pos_embedding.weight[:self.num_embeddings + 1])
        return self.final_layer(out[:, 0])

    def _get_name(self) -> str:
        return "PBT"
