# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.functional.patches import pad_or_crop_to_patches
from braindecode.modules.timm import TimmAttention, PreNormBlock

def _dino_emb_dim(patch_size: int, cksp: List[Tuple[int, int, int, int, Tuple[int, int]]]) -> int:
    r = patch_size
    for _, k, s, p, _ in cksp:
        r = (r + 2 * p - k) // s + 1
    return cksp[-1][0] * r

class _DinoPatchEmbedding(nn.Module):
    patch_size: int
    n_channel_embeddings: int
    emb_dim: int
    proj_in: nn.Sequential
    spectral_proj: nn.Sequential
    channel_embedding: nn.Linear
    time_encoding: nn.Sequential
    def __init__(self, patch_size: int, cksp: List[Tuple[int, int, int, int, Tuple[int, int]]], n_channel_embeddings: int,
                 activation = nn.GELU, drop_prob: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_channel_embeddings = n_channel_embeddings
        self.emb_dim = _dino_emb_dim(patch_size, cksp)
        E = self.emb_dim
        self.proj_in = nn.Sequential()
        ic = 1
        i = 0
        for oc, k, s, p, g in cksp:
            self.proj_in.add_module(str(i), nn.Conv2d(ic, oc, (1, k), (1, s), (0, p)))
            self.proj_in.add_module(str(i + 1), nn.GroupNorm(g[0], g[1]))
            self.proj_in.add_module(str(i + 2), activation())
            i += 3
            ic = oc
        self.add_module("proj_in", self.proj_in)
        self.spectral_proj = nn.Sequential(nn.Linear(patch_size // 2 + 1, E), nn.Dropout(drop_prob))
        self.add_module("spectral_proj", self.spectral_proj)
        self.channel_embedding = nn.Linear(n_channel_embeddings, E)
        self.add_module("channel_embedding", self.channel_embedding)
        self.time_encoding = nn.Sequential(nn.Conv2d(E, E, (1, 5), (1, 1), (0, 2), groups=E))
        self.add_module("time_encoding", self.time_encoding)
    def forward(self, x: Tensor) -> Tensor:
        C = x.shape[1]
        tt = self.proj_in(x.flatten(1, 2).unsqueeze(1))
        B, D, L = tt.shape[0], tt.shape[1], tt.shape[3]
        tt = tt.reshape(B, D, C, -1, L).permute(0, 2, 3, 1, 4).flatten(3, 4)
        spec = torch.abs(torch.fft.rfft(x, dim=-1, norm="forward")).to(x.dtype)
        pe = tt + self.spectral_proj(spec)
        oh = torch.eye(C, self.n_channel_embeddings, dtype=self.channel_embedding.weight.dtype)
        pe = pe + self.channel_embedding(oh).unsqueeze(0).unsqueeze(2)
        return pe + self.time_encoding(pe.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    def _get_name(self) -> str:
        return "_PatchEmbedding"

class _DinoMLP(nn.Module):
    fc1: nn.Linear
    act: nn.Module
    fc2: nn.Linear
    drop: nn.Dropout
    def __init__(self, emb_dim: int, hidden_dim: int, activation = nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.add_module("fc1", self.fc1)
        self.act = activation()
        self.add_module("act", self.act)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.add_module("fc2", self.fc2)
        self.drop = nn.Dropout(drop)
        self.add_module("drop", self.drop)
    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))
    def _get_name(self) -> str:
        return "_MLP"

class _DinoHead(nn.Module):
    token_proj: nn.Sequential
    time_proj: nn.Sequential
    classifier: nn.Sequential
    def __init__(self, emb_dim: int, n_outputs: int, activation = nn.GELU, drop_prob_1: float = 0.5, drop_prob_2: float = 0.3):
        super().__init__()
        self.token_proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), activation())
        self.add_module("token_proj", self.token_proj)
        self.time_proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), activation())
        self.add_module("time_proj", self.time_proj)
        self.classifier = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2), activation(), nn.Dropout(drop_prob_1),
                                        nn.Linear(emb_dim // 2, emb_dim // 4), activation(), nn.Dropout(drop_prob_2),
                                        nn.Linear(emb_dim // 4, n_outputs))
        self.add_module("classifier", self.classifier)
    def head(self, t: Tensor, C: int) -> Tensor:
        x = self.token_proj(t)
        x = x.reshape(x.shape[0], C, -1, x.shape[2]).mean(dim=1)
        return self.classifier(self.time_proj(x).mean(dim=1))
    def _get_name(self) -> str:
        return "_ClassificationHead"

class EEGDINO(EEGModuleMixin, nn.Module):
    """EEG-DINO from Wang et al (2025). Architecture only (no Hub weights).

    ``return_encoder_output=True`` gives the pooled encoder features;
    ``forward_features`` returns (patch tokens, mean global token)."""
    patch_size: int
    n_global_tokens: int
    global_token_layer: int
    return_encoder_output: bool
    emb_dim: int
    patch_embedding: _DinoPatchEmbedding
    encoder_layers: nn.ModuleList
    _layers: List[PreNormBlock[TimmAttention, _DinoMLP]]
    global_tokens: Tensor
    final_layer: nn.Module
    _head: Optional[_DinoHead]

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, patch_size: int = 200,
                 n_layer: int = 12, nhead: int = 8, dim_feedforward: int = 512,
                 channels_kernel_stride_padding_norm: List[Tuple[int, int, int, int, Tuple[int, int]]] = [
                     (25, 49, 25, 24, (5, 25)), (25, 3, 1, 1, (5, 25)), (25, 3, 1, 1, (5, 25))],
                 n_channel_embeddings: int = 19, n_global_tokens: int = 1, global_token_layer: int = 1,
                 activation = nn.GELU, patch_activation = nn.GELU, drop_prob: float = 0.1,
                 return_encoder_output: bool = False):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if not (1 <= global_token_layer <= n_layer):
            raise ValueError(f"global_token_layer must be in [1, n_layer={n_layer}], got {global_token_layer}.")
        if n_global_tokens < 1:
            raise ValueError(f"n_global_tokens must be >= 1, got {n_global_tokens}.")
        if self.n_chans > n_channel_embeddings:
            raise ValueError(f"n_chans={self.n_chans} exceeds n_channel_embeddings={n_channel_embeddings}.")
        self.patch_size = patch_size
        self.n_global_tokens = n_global_tokens
        self.global_token_layer = global_token_layer
        self.return_encoder_output = return_encoder_output
        self.patch_embedding = _DinoPatchEmbedding(patch_size, channels_kernel_stride_padding_norm, n_channel_embeddings,
                                                   patch_activation, drop_prob)
        self.add_module("patch_embedding", self.patch_embedding)
        E = self.patch_embedding.emb_dim
        self.emb_dim = E
        self._layers = [PreNormBlock(E, TimmAttention(E, nhead, qv_bias=True, attn_drop=drop_prob, proj_drop=drop_prob, exact=True),
                                     _DinoMLP(E, dim_feedforward, activation, drop_prob), drop_path=drop_prob)
                        for _ in range(n_layer)]
        self.encoder_layers = nn.ModuleList()
        for l in self._layers:
            self.encoder_layers.append(l)
        self.add_module("encoder_layers", self.encoder_layers)
        self.global_tokens = nn.Parameter(torch.zeros(1, n_global_tokens, E))
        self.register_parameter("global_tokens", self.global_tokens)
        fl: nn.Module = nn.Identity()
        self._head = None
        if not return_encoder_output:
            h = _DinoHead(E, self.n_outputs, activation)
            self._head = h
            fl = h
        self.final_layer = fl
        self.add_module("final_layer", fl)
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

    def _encode(self, x: Tensor) -> Tuple[Tensor, int]:
        x = pad_or_crop_to_patches(x, self.patch_size, "pad")
        x = x.reshape(x.shape[0], x.shape[1], -1, self.patch_size)
        pe = self.patch_embedding(x)
        C = pe.shape[1]
        t = pe.flatten(1, 2)
        g = self.global_tokens.expand(t.shape[0], -1, -1)
        for i in range(len(self._layers)):
            t = self._layers[i](t)
            if i + 1 == self.global_token_layer:
                t = torch.cat([g, t], dim=1)
        return t, C

    def forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        t, _ = self._encode(x)
        return t[:, self.n_global_tokens:], t[:, :self.n_global_tokens].mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        t, C = self._encode(x)
        p = t[:, self.n_global_tokens:]
        if self._head is None:
            return p.mean(dim=1)
        return self._head.__val__().head(p, C)

    def _get_name(self) -> str:
        return "EEGDINO"
