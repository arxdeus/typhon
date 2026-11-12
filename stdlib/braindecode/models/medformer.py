# Ported from braindecode 2.0.0 (BSD-3-Clause)

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from braindecode.models.base import EEGModuleMixin
from braindecode.modules.positional import SinusoidalPositionalEncoding

class _CrossChannelTokenEmbedding(nn.Module):
    token_conv: nn.Conv2d
    l_patch: int
    def __init__(self, c_in: int, l_patch: int, d_model: int):
        super().__init__()
        self.l_patch = l_patch
        self.token_conv = nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=(c_in, l_patch),
                                    stride=(1, l_patch), padding=0, bias=False)
        self.add_module("token_conv", self.token_conv)
        init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")
    def forward(self, x: Tensor) -> Tensor:
        lp = self.l_patch
        n_patches = x.shape[-1] // lp
        x = x[:, 0, :, :n_patches * lp].reshape(x.shape[0], x.shape[2], n_patches, lp)
        x = torch.matmul(x.transpose(1, 2).flatten(2), self.token_conv.weight.flatten(1).t())
        return x.transpose(1, 2).unsqueeze(2)
    def _get_name(self) -> str:
        return "_CrossChannelTokenEmbedding"

class _ListPatchEmbedding(nn.Module):
    paddings: nn.ModuleList
    _pads: List[nn.ReplicationPad1d]
    value_embeddings: nn.ModuleList
    _ves: List[_CrossChannelTokenEmbedding]
    position_embedding: SinusoidalPositionalEncoding
    channel_embedding: SinusoidalPositionalEncoding
    dropout: nn.Dropout
    learnable_embeddings: nn.ParameterList
    _les: List[Tensor]
    single_channel: bool
    n_chans: int
    n_times: int
    def __init__(self, enc_in: int, d_model: int, seq_len: int, patch_len_list: List[int], stride_list: List[int],
                 dropout: float, single_channel: bool, n_chans: int, n_times: int):
        super().__init__()
        self.single_channel = single_channel
        self.n_chans = n_chans
        self.n_times = n_times
        self._pads = [nn.ReplicationPad1d((0, s)) for s in stride_list]
        self.paddings = nn.ModuleList()
        for p in self._pads:
            self.paddings.append(p)
        self.add_module("paddings", self.paddings)
        self._ves = [_CrossChannelTokenEmbedding(1 if single_channel else enc_in, pl, d_model) for pl in patch_len_list]
        self.value_embeddings = nn.ModuleList()
        for v in self._ves:
            self.value_embeddings.append(v)
        self.add_module("value_embeddings", self.value_embeddings)
        self.position_embedding = SinusoidalPositionalEncoding(5000, d_model, persistent=True)
        self.add_module("position_embedding", self.position_embedding)
        self.channel_embedding = SinusoidalPositionalEncoding(enc_in, seq_len, persistent=True)
        self.add_module("channel_embedding", self.channel_embedding)
        self.dropout = nn.Dropout(dropout)
        self.add_module("dropout", self.dropout)
        self._les = [nn.Parameter(torch.randn(1, d_model)) for _ in patch_len_list]
        self.learnable_embeddings = nn.ParameterList()
        for p in self._les:
            self.learnable_embeddings.append(p)
        self.add_module("learnable_embeddings", self.learnable_embeddings)
    def embed(self, x: Tensor) -> List[Tensor]:
        if self.single_channel:
            x = x.reshape(x.shape[0] * self.n_chans, 1, self.n_times)
        else:
            x = self.channel_embedding(x)
        out = List[Tensor]()
        for i in range(len(self._ves)):
            xn = self._ves[i](self._pads[i](x).unsqueeze(1))
            xn = xn.squeeze(2).transpose(1, 2)
            out.append(self.position_embedding(xn + self._les[i]))
        return out
    def _get_name(self) -> str:
        return "_ListPatchEmbedding"

class _FullAttention(nn.Module):
    dropout: nn.Dropout
    def __init__(self, attention_dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.add_module("dropout", self.dropout)
    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        e = q.shape[3]
        scale = 1.0 / math.sqrt(e)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        a = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.matmul(a, v).transpose(1, 2).contiguous()
    def _get_name(self) -> str:
        return "_FullAttention"

class _AttentionLayer(nn.Module):
    inner_attention: _FullAttention
    query_projection: nn.Linear
    key_projection: nn.Linear
    value_projection: nn.Linear
    out_projection: nn.Linear
    num_heads: int
    def __init__(self, attention: _FullAttention, d_model: int, num_heads: int):
        super().__init__()
        dk = d_model // num_heads
        self.inner_attention = attention
        self.add_module("inner_attention", self.inner_attention)
        self.query_projection = nn.Linear(d_model, dk * num_heads)
        self.add_module("query_projection", self.query_projection)
        self.key_projection = nn.Linear(d_model, dk * num_heads)
        self.add_module("key_projection", self.key_projection)
        self.value_projection = nn.Linear(d_model, dk * num_heads)
        self.add_module("value_projection", self.value_projection)
        self.out_projection = nn.Linear(dk * num_heads, d_model)
        self.add_module("out_projection", self.out_projection)
        self.num_heads = num_heads
    def self_attn(self, x: Tensor) -> Tensor:
        b, l = x.shape[0], x.shape[1]
        h = self.num_heads
        q = self.query_projection(x).view(b, l, h, -1)
        k = self.key_projection(x).view(b, l, h, -1)
        v = self.value_projection(x).view(b, l, h, -1)
        out = self.inner_attention.attend(q, k, v).view(b, l, -1)
        return self.out_projection(out)
    def _get_name(self) -> str:
        return "_AttentionLayer"

class _MedformerLayer(nn.Module):
    intra_attentions: nn.ModuleList
    _intra: List[_AttentionLayer]
    inter_attention: Optional[_AttentionLayer]
    def __init__(self, num_blocks: int, d_model: int, num_heads: int, dropout: float = 0.1, no_inter: bool = False):
        super().__init__()
        self._intra = [_AttentionLayer(_FullAttention(dropout), d_model, num_heads) for _ in range(num_blocks)]
        self.intra_attentions = nn.ModuleList()
        for a in self._intra:
            self.intra_attentions.append(a)
        self.add_module("intra_attentions", self.intra_attentions)
        if no_inter or num_blocks <= 1:
            self.inter_attention = None
        else:
            ia = _AttentionLayer(_FullAttention(dropout), d_model, num_heads)
            self.inter_attention = ia
            self.add_module("inter_attention", ia)
    def run(self, x: List[Tensor]) -> List[Tensor]:
        xi = [self._intra[i].self_attn(x[i]) for i in range(len(self._intra))]
        if self.inter_attention is not None:
            routers = torch.cat([t[:, -1:] for t in xi], dim=1)
            xin = self.inter_attention.__val__().self_attn(routers)
            return [torch.cat([xi[i][:, :-1], xin[:, i:i + 1]], dim=1) for i in range(len(xi))]
        return xi
    def _get_name(self) -> str:
        return "_MedformerLayer"

class _EncoderLayer(nn.Module):
    attention: _MedformerLayer
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    dropout: nn.Dropout
    activation: nn.Module
    def __init__(self, attention: _MedformerLayer, d_model: int, dim_feedforward: int, dropout: float, activation: nn.Module):
        super().__init__()
        dff = dim_feedforward if dim_feedforward else 4 * d_model
        self.attention = attention
        self.add_module("attention", self.attention)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=dff, kernel_size=1)
        self.add_module("conv1", self.conv1)
        self.conv2 = nn.Conv1d(in_channels=dff, out_channels=d_model, kernel_size=1)
        self.add_module("conv2", self.conv2)
        self.norm1 = nn.LayerNorm(d_model)
        self.add_module("norm1", self.norm1)
        self.norm2 = nn.LayerNorm(d_model)
        self.add_module("norm2", self.norm2)
        self.dropout = nn.Dropout(dropout)
        self.add_module("dropout", self.dropout)
        self.activation = activation
        self.add_module("activation", self.activation)
    def run(self, x: List[Tensor]) -> List[Tensor]:
        new_x = self.attention.run(x)
        lengths = [t.shape[1] for t in x]
        x_all = self.norm1(torch.cat([x[i] + self.dropout(new_x[i]) for i in range(len(x))], dim=1))
        ys = self.activation(self.conv1(x_all.transpose(-1, 1))).split(lengths, dim=-1)
        y = [self.dropout(self.conv2(self.dropout(t)).transpose(-1, 1)) for t in ys]
        return list(self.norm2(x_all + torch.cat(y, dim=1)).split(lengths, dim=1))
    def _get_name(self) -> str:
        return "_EncoderLayer"

class _Encoder(nn.Module):
    attn_layers: nn.ModuleList
    _layers: List[_EncoderLayer]
    norm: nn.LayerNorm
    def __init__(self, attn_layers: List[_EncoderLayer], norm_layer: nn.LayerNorm):
        super().__init__()
        self._layers = attn_layers
        self.attn_layers = nn.ModuleList()
        for l in attn_layers:
            self.attn_layers.append(l)
        self.add_module("attn_layers", self.attn_layers)
        self.norm = norm_layer
        self.add_module("norm", self.norm)
    def run(self, x: List[Tensor]) -> Tensor:
        for l in self._layers:
            x = l.run(x)
        y = torch.cat([t[:, -1, :].unsqueeze(1) for t in x], dim=1)
        return self.norm(y)
    def _get_name(self) -> str:
        return "_Encoder"

class MEDFormer(EEGModuleMixin, nn.Module):
    """MEDFormer from Wang et al (2024)."""
    embed_dim: int
    num_heads: int
    drop_prob: float
    no_inter_attn: bool
    num_layers: int
    single_channel: bool
    patch_len_list: List[int]
    patch_num_list: List[int]
    enc_embedding: _ListPatchEmbedding
    encoder: _Encoder
    activation_layer: nn.Module
    dropout: nn.Dropout
    final_layer: nn.Linear

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, patch_len_list: Optional[List[int]] = None, embed_dim: int = 128,
                 num_heads: int = 8, drop_prob: float = 0.1, no_inter_attn: bool = False, num_layers: int = 6,
                 dim_feedforward: int = 256, activation_trans = nn.ReLU, single_channel: bool = False,
                 output_attention: bool = True, activation_class = nn.GELU):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.no_inter_attn = no_inter_attn
        self.num_layers = num_layers
        self.single_channel = single_channel
        pl = patch_len_list if patch_len_list is not None else [2, 8, 16]
        self.patch_len_list = pl
        self.patch_num_list = [int((self.n_times - p) / p + 2) for p in pl]
        self.enc_embedding = _ListPatchEmbedding(self.n_chans, embed_dim, self.n_times, pl, pl, drop_prob,
                                                 single_channel, self.n_chans, self.n_times)
        self.add_module("enc_embedding", self.enc_embedding)
        layers = [_EncoderLayer(_MedformerLayer(len(pl), embed_dim, num_heads, drop_prob, no_inter_attn),
                                embed_dim, dim_feedforward, drop_prob, activation_trans()) for _ in range(num_layers)]
        self.encoder = _Encoder(layers, nn.LayerNorm(embed_dim))
        self.add_module("encoder", self.encoder)
        self.activation_layer = activation_class()
        self.add_module("activation_layer", self.activation_layer)
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.final_layer = nn.Linear(embed_dim * len(self.patch_num_list) * (self.n_chans if single_channel else 1),
                                     self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        enc = self.encoder.run(self.enc_embedding.embed(x))
        if self.single_channel:
            b = enc.shape[0] // self.n_chans
            enc = enc.reshape(b, self.n_chans, enc.shape[1], enc.shape[2])
        out = self.dropout(self.activation_layer(enc))
        return self.final_layer(out.reshape(out.shape[0], -1))

    def _get_name(self) -> str:
        return "MEDFormer"
