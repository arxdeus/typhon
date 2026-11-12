# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import sinusoidal_positional_encoding
from braindecode.functional.patches import _check_on_non_divisible, patch_count, pad_or_crop_to_patches

class _SleepFMTokenizer(nn.Module):
    embed_dim: int
    patch_size: int
    on_non_divisible: str
    tokenizer: nn.Sequential
    def __init__(self, patch_size: int, embed_dim: int, activation, n_times: int, max_patches: int,
                 on_non_divisible: str = "pad"):
        super().__init__()
        if patch_size < 64 or patch_size % 64:
            raise ValueError("patch_size must be at least 64 and divisible by 64.")
        _check_on_non_divisible(on_non_divisible)
        n = patch_count(n_times, patch_size, on_non_divisible)
        if not (0 < n <= max_patches):
            raise ValueError(f"n_times={n_times} gives {n} patches; at most {max_patches} are supported.")
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.on_non_divisible = on_non_divisible
        self.tokenizer = nn.Sequential()
        ic = 1
        i = 0
        bi = 1
        for oc in (4, 8, 16, 32, 64, 128):
            self.tokenizer.add_module(str(i), nn.Conv1d(ic, oc, 5, stride=2, padding=2))
            self.tokenizer.add_module(str(i + 1), nn.BatchNorm1d(oc))
            self.tokenizer.add_module(str(i + 2), activation())
            self.tokenizer.add_module(str(i + 3), nn.LayerNorm([oc, patch_size // (1 << bi)]))
            i += 4
            bi += 1
            ic = oc
        self.tokenizer.add_module(str(i), nn.AdaptiveAvgPool1d(1))
        self.tokenizer.add_module(str(i + 1), nn.Flatten())
        self.tokenizer.add_module(str(i + 2), nn.Linear(128, embed_dim))
        self.add_module("tokenizer", self.tokenizer)
    def n_patches(self, n_times: int) -> int:
        return patch_count(n_times, self.patch_size, self.on_non_divisible)
    def forward(self, x: Tensor) -> Tensor:
        B, C = x.shape[0], x.shape[1]
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        t = self.tokenizer(x.reshape(-1, 1, self.patch_size))
        return t.reshape(B, C, -1, t.shape[-1])
    def _get_name(self) -> str:
        return "_SleepFMTokenizer"

class _SleepFMAttentionPooling(nn.Module):
    transformer_layer: nn.TransformerEncoderLayer
    def __init__(self, input_dim: int, num_heads: int, drop_prob: float):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=drop_prob,
                                                            batch_first=True)
        self.add_module("transformer_layer", self.transformer_layer)
    def pool(self, x: Tensor, kpm: Optional[Tensor]) -> Tensor:
        if kpm is None:
            return self.transformer_layer(x).mean(dim=1)
        m = kpm.__val__()
        am = m.all(dim=1, keepdim=True)
        m = m & am.logical_not()
        out: Tensor
        if x.shape[1] == 1:
            out = x[:, 0]
        else:
            o = self.transformer_layer(x, src_key_padding_mask=m)
            valid = m.logical_not().unsqueeze(-1).to(o.dtype)
            out = (o * valid).sum(dim=1) / valid.sum(dim=1)
        return out.masked_fill(am, 0.0)
    def per_patch(self, tokens: Tensor, mask: Tensor) -> Tensor:
        C, P, E = tokens.shape[1], tokens.shape[2], tokens.shape[3]
        t = tokens.permute(0, 2, 1, 3).reshape(-1, C, E)
        m = mask.unsqueeze(1).expand(-1, P, -1).reshape(-1, C)
        return self.pool(t, m).reshape(-1, P, E)
    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x, None)
    def _get_name(self) -> str:
        return "_SleepFMAttentionPooling"

class _Context(nn.Module):
    # _add_context_layers: spatial_pooling, positional_encoding, layer_norm,
    # transformer_encoder (registered on the owner, see _ctx_register)
    pass

def _ctx_modules(embed_dim: int, num_heads: int, num_layers: int, pooling_heads: int, drop_prob: float,
                 n_positions: int) -> Tuple[_SleepFMAttentionPooling, Tensor, nn.LayerNorm, nn.TransformerEncoder]:
    sp = _SleepFMAttentionPooling(embed_dim, pooling_heads, drop_prob)
    pe = sinusoidal_positional_encoding(n_positions, embed_dim).unsqueeze(0)
    ln = nn.LayerNorm(embed_dim)
    te = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=drop_prob,
                                                          batch_first=True, norm_first=True),
                               num_layers=num_layers, enable_nested_tensor=False)
    return sp, pe, ln, te

class SleepFM(EEGModuleMixin, nn.Module):
    """SleepFM from Thapa et al (2026). Architecture only (no Hub weights).

    The optional ``channel_mask`` of braindecode's forward is not ported."""
    patch_size: int
    embed_dim: int
    patch_embedding: _SleepFMTokenizer
    spatial_pooling: _SleepFMAttentionPooling
    positional_encoding: Tensor
    layer_norm: nn.LayerNorm
    transformer_encoder: nn.TransformerEncoder
    temporal_pooling: _SleepFMAttentionPooling
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, patch_size: int = 640,
                 embed_dim: int = 128, num_heads: int = 8, num_layers: int = 6, pooling_heads: int = 8,
                 drop_prob: float = 0.3, max_seq_length: int = 128, activation = nn.ELU, on_non_divisible: str = "pad"):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embedding = _SleepFMTokenizer(patch_size, embed_dim, activation, self.n_times, max_seq_length,
                                                 on_non_divisible)
        self.add_module("patch_embedding", self.patch_embedding)
        sp, pe, ln, te = _ctx_modules(embed_dim, num_heads, num_layers, pooling_heads, drop_prob, max_seq_length)
        self.spatial_pooling = sp
        self.add_module("spatial_pooling", sp)
        self.positional_encoding = pe
        self.register_buffer("positional_encoding", pe)
        self.layer_norm = ln
        self.add_module("layer_norm", ln)
        self.transformer_encoder = te
        self.add_module("transformer_encoder", te)
        self.temporal_pooling = _SleepFMAttentionPooling(embed_dim, pooling_heads, drop_prob)
        self.add_module("temporal_pooling", self.temporal_pooling)
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool)
        t = self.spatial_pooling.per_patch(self.patch_embedding(x), mask)
        t = self.layer_norm(t + self.positional_encoding[:, :t.shape[1]])
        ct = self.transformer_encoder(t)
        return self.temporal_pooling(ct), ct

    def forward(self, x: Tensor) -> Tensor:
        p, _ = self.encode(x)
        return self.final_layer(p)

    def _get_name(self) -> str:
        return "SleepFM"

class _SleepFMStagingHead(nn.Module):
    spatial_pooling: _SleepFMAttentionPooling
    positional_encoding: Tensor
    layer_norm: nn.LayerNorm
    transformer_encoder: nn.TransformerEncoder
    lstm: nn.LSTM
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, pooling_heads: int, drop_prob: float,
                 max_seq_length: int):
        super().__init__()
        sp, pe, ln, te = _ctx_modules(embed_dim, num_heads, num_layers, pooling_heads, drop_prob, max_seq_length)
        self.spatial_pooling = sp
        self.add_module("spatial_pooling", sp)
        self.positional_encoding = pe
        self.register_buffer("positional_encoding", pe)
        self.layer_norm = ln
        self.add_module("layer_norm", ln)
        self.transformer_encoder = te
        self.add_module("transformer_encoder", te)
        self.lstm = nn.LSTM(embed_dim, embed_dim // 2, num_layers=num_layers, batch_first=True,
                            dropout=drop_prob if num_layers > 1 else 0.0, bidirectional=True)
        self.add_module("lstm", self.lstm)
    def head(self, tokens: Tensor, modality_mask: Tensor) -> Tensor:
        f = self.spatial_pooling.per_patch(tokens, modality_mask)
        tm = torch.zeros(f.shape[0], f.shape[1], dtype=torch.bool)
        f = self.layer_norm(f + self.positional_encoding[:, :f.shape[1]])
        f = self.transformer_encoder(f, src_key_padding_mask=tm)
        out, _ = self.lstm(f, None)
        return out
    def _get_name(self) -> str:
        return "_SleepFMStagingHead"

class SleepFMStager(EEGModuleMixin, nn.Module):
    """SleepFM sleep stager from Thapa et al (2026). Architecture only.

    ``channel_modalities`` gives one integer modality label per channel (at
    most 4 modalities). The optional channel / temporal masks of
    braindecode's forward are not ported. Output: (batch, n_outputs, n_patches)."""
    patch_size: int
    embed_dim: int
    encoder_chunk_patches: int
    _modality_sizes: List[int]
    _channel_order: Tensor
    patch_embedding: _SleepFMTokenizer
    spatial_pooling: _SleepFMAttentionPooling
    positional_encoding: Tensor
    layer_norm: nn.LayerNorm
    transformer_encoder: nn.TransformerEncoder
    staging_head: _SleepFMStagingHead
    final_layer: nn.Linear

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None,
                 channel_modalities: Optional[List[int]] = None, patch_size: int = 640, embed_dim: int = 128,
                 encoder_num_heads: int = 8, encoder_num_layers: int = 6, encoder_pooling_heads: int = 8,
                 encoder_drop_prob: float = 0.0, encoder_chunk_patches: int = 60, staging_num_heads: int = 4,
                 staging_num_layers: int = 1, staging_pooling_heads: int = 4, drop_prob: float = 0.3,
                 max_seq_length: int = 8196, activation = nn.ELU, on_non_divisible: str = "pad"):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        labels = [0] * self.n_chans
        if channel_modalities is not None:
            labels = channel_modalities.__val__()
            if len(labels) != self.n_chans:
                raise ValueError(f"channel_modalities has {len(labels)} entries but n_chans={self.n_chans}.")
        mods = List[int]()
        for l in labels:
            if l not in mods:
                mods.append(l)
        if len(mods) > 4:
            raise ValueError("SleepFMStager supports at most 4 modalities.")
        order = [i for m in mods for i in range(len(labels)) if labels[i] == m]
        self._modality_sizes = [labels.count(m) for m in mods]
        self._channel_order = torch.tensor(order, dtype=torch.long)
        self.register_buffer("_channel_order", self._channel_order, persistent=False)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_chunk_patches = encoder_chunk_patches
        self.patch_embedding = _SleepFMTokenizer(patch_size, embed_dim, activation, self.n_times, max_seq_length,
                                                 on_non_divisible)
        self.add_module("patch_embedding", self.patch_embedding)
        sp, pe, ln, te = _ctx_modules(embed_dim, encoder_num_heads, encoder_num_layers, encoder_pooling_heads,
                                      encoder_drop_prob, max(encoder_chunk_patches, 128))
        self.spatial_pooling = sp
        self.add_module("spatial_pooling", sp)
        self.positional_encoding = pe
        self.register_buffer("positional_encoding", pe)
        self.layer_norm = ln
        self.add_module("layer_norm", ln)
        self.transformer_encoder = te
        self.add_module("transformer_encoder", te)
        self.staging_head = _SleepFMStagingHead(embed_dim, staging_num_heads, staging_num_layers, staging_pooling_heads,
                                                drop_prob, max_seq_length)
        self.add_module("staging_head", self.staging_head)
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)
        self.add_module("final_layer", self.final_layer)

    def forward_features(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        npch = self.patch_embedding.n_patches(x.shape[-1])
        ch = self.encoder_chunk_patches
        nfull = npch // ch
        xs = x.index_select(1, self._channel_order).split(self._modality_sizes, 1)
        embs = List[Tensor]()
        miss = List[Tensor]()
        for sig in xs:
            tok = self.spatial_pooling.per_patch(self.patch_embedding(sig), torch.zeros(B, sig.shape[1], dtype=torch.bool))
            enc = List[Tensor]()
            for start, stop, nc in ((0, nfull * ch, nfull), (nfull * ch, npch, 1)):
                if stop == start:
                    continue
                seg = tok[:, start:stop].reshape(B * nc, -1, tok.shape[-1])
                seg = self.layer_norm(seg + self.positional_encoding[:, :seg.shape[1]])
                seg = self.transformer_encoder(seg)
                enc.append(seg.reshape(B, -1, seg.shape[-1]))
            embs.append(torch.cat(enc, dim=1))
            miss.append(torch.zeros(B, dtype=torch.bool))
        while len(embs) < 4:
            embs.append(torch.zeros_like(embs[0]))
            miss.append(torch.ones(B, dtype=torch.bool))
        return self.staging_head.head(torch.stack(embs, dim=1), torch.stack(miss, dim=1))

    def forward(self, x: Tensor) -> Tensor:
        return self.final_layer(self.forward_features(x)).transpose(1, 2)

    def _get_name(self) -> str:
        return "SleepFMStager"
