# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.modules.positional import SinusoidalPositionalEncoding
from braindecode.modules.linear_attention import LinearAttentionTransformer

class _PatchFrequencyEmbedding(nn.Module):
    projection: nn.Linear
    def __init__(self, emb_size: int = 256, n_freq: int = 101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)
        self.add_module("projection", self.projection)
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x.permute(0, 2, 1))
    def _get_name(self) -> str:
        return "_PatchFrequencyEmbedding"

class _BIOTClassificationHead(nn.Sequential):
    activation_layer: nn.Module
    classification_head: nn.Linear
    def __init__(self, emb_size: int, n_outputs: int, activation = nn.ELU):
        super().__init__()
        self.activation_layer = activation()
        self.add_module("activation_layer", self.activation_layer)
        self.classification_head = nn.Linear(emb_size, n_outputs)
        self.add_module("classification_head", self.classification_head)
    def forward(self, x: Tensor) -> Tensor:
        return self.classification_head(self.activation_layer(x))
    def _get_name(self) -> str:
        return "_ClassificationHead"

class _BIOTEncoder(nn.Module):
    n_fft: int
    hop_length: int
    patch_embedding: _PatchFrequencyEmbedding
    transformer: LinearAttentionTransformer
    positional_encoding: SinusoidalPositionalEncoding
    channel_tokens: nn.Embedding
    index: Tensor

    def __init__(self, emb_size: int = 256, num_heads: int = 8, n_chans: int = 16, n_layers: int = 4,
                 n_fft: int = 200, hop_length: int = 100, drop_prob: float = 0.1, max_seq_len: int = 1024,
                 attn_dropout: float = 0.2, attn_layer_dropout: float = 0.2):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_embedding = _PatchFrequencyEmbedding(emb_size, n_fft // 2 + 1)
        self.add_module("patch_embedding", self.patch_embedding)
        self.transformer = LinearAttentionTransformer(emb_size, n_layers, num_heads, attn_layer_dropout)
        self.add_module("transformer", self.transformer)
        self.positional_encoding = SinusoidalPositionalEncoding(1000, emb_size, drop_prob, persistent=True)
        self.add_module("positional_encoding", self.positional_encoding)
        self.channel_tokens = nn.Embedding(n_chans, emb_size)
        self.add_module("channel_tokens", self.channel_tokens)
        self.index = torch.arange(n_chans, dtype=torch.long)
        self.register_buffer("index", self.index, persistent=False)

    def stft(self, sample: Tensor) -> Tensor:
        spec = torch.stft(sample.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, center=False,
                          onesided=True, return_complex=True)
        return torch.abs(spec).to(sample.dtype)

    def forward(self, x: Tensor) -> Tensor:
        b, c, t = x.shape[0], x.shape[1], x.shape[2]
        spec = self.stft(x.transpose(0, 1).reshape(c * b, 1, t)).unflatten(0, [c, b])
        seq = List[Tensor]()
        for i in range(c):
            e = self.patch_embedding(spec[i])
            ts = e.shape[1]
            tok = self.channel_tokens.weight[i].unsqueeze(0).unsqueeze(0).repeat(b, ts, 1)
            seq.append(self.positional_encoding(e + tok))
        emb = torch.cat(seq, dim=1)
        return self.transformer(emb).mean(dim=1)

    def _get_name(self) -> str:
        return "_BIOTEncoder"

class BIOT(EEGModuleMixin, nn.Module):
    """BIOT from Yang et al (2023). Architecture only (no Hub weights)."""
    embed_dim: int
    hop_length: int
    num_heads: int
    num_layers: int
    return_feature: bool
    n_fft: int
    encoder: _BIOTEncoder
    final_layer: _BIOTClassificationHead

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 4, sfreq: Optional[float] = None,
                 hop_length: int = 100, return_feature: bool = False, n_outputs: Optional[int] = None,
                 n_chans: Optional[int] = None, chs_info: Optional[List[Dict[str, str]]] = None,
                 n_times: Optional[int] = None, input_window_seconds: Optional[float] = None, activation = nn.ELU,
                 drop_prob: float = 0.5, max_seq_len: int = 1024, att_drop_prob: float = 0.2,
                 att_layer_drop_prob: float = 0.2):
        sf = sfreq
        if sf is None and (n_times is None or input_window_seconds is None):
            sf = 200.0
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sf)
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.hop_length = hop_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.return_feature = return_feature
        if float(self.hop_length) > self.sfreq:
            self.hop_length = int(self.sfreq // 2.0)
        if self.input_window_seconds < 1.0:
            self.n_fft = self.n_times
        else:
            self.n_fft = int(self.sfreq)
        self.encoder = _BIOTEncoder(self.embed_dim, self.num_heads, self.n_chans, self.num_layers, self.n_fft,
                                    self.hop_length, drop_prob, max_seq_len, att_drop_prob, att_layer_drop_prob)
        self.add_module("encoder", self.encoder)
        self.final_layer = _BIOTClassificationHead(self.embed_dim, self.n_outputs, activation)
        self.add_module("final_layer", self.final_layer)

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.encoder(x))

    def forward_features(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def _get_name(self) -> str:
        return "BIOT"
