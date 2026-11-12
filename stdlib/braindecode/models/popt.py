# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# PopulationTransformer (PopT). Codon notes: chs_info "loc" holds whitespace
# separated floats; activation must be nn.GELU or nn.ReLU (mapped onto the
# TransformerEncoderLayer activation); forward(x) uses the stored electrode
# coordinates, seq_id 0 and no channel mask; channel_strategy="native" only.

import math
import torch
from torch import Tensor
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin
from braindecode.functional import sinusoidal_positional_encoding

class _PopTSpatialPositionalEncoding(nn.Module):
    pe: Tensor
    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        if hidden_dim % 8 != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by 8 for the spatial positional encoding.")
        self.pe = sinusoidal_positional_encoding(max_len, hidden_dim // 4).unsqueeze(0)
        self.register_buffer("pe", self.pe, persistent=False)
    def encode(self, seq: Tensor, coords: Tensor, seq_id: Tensor) -> Tensor:
        table = self.pe[0]
        b, e = coords.shape[0], coords.shape[1]
        p = table.index_select(0, coords.reshape(-1)).reshape(b, e, -1)
        s = table.index_select(0, seq_id.reshape(-1)).reshape(b, e, -1)
        elec = torch.cat([p, s], dim=-1)
        cls = table[0].repeat(b, 4).unsqueeze(1)
        return seq + torch.cat([cls, elec], dim=1)
    def _get_name(self) -> str:
        return "_PopTSpatialPositionalEncoding"

class _PopTInputEmbedding(nn.Module):
    input_dim: int
    in_proj: nn.Linear
    positional_encoding: _PopTSpatialPositionalEncoding
    layer_norm: nn.LayerNorm
    dropout: nn.Dropout
    def __init__(self, input_dim: int, hidden_dim: int, max_len: int = 5000, drop_prob: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.add_module("in_proj", self.in_proj)
        self.positional_encoding = _PopTSpatialPositionalEncoding(hidden_dim, max_len)
        self.add_module("positional_encoding", self.positional_encoding)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.add_module("layer_norm", self.layer_norm)
        self.dropout = nn.Dropout(p=drop_prob)
        self.add_module("dropout", self.dropout)
    def embed(self, features: Tensor, coords: Tensor, seq_id: Tensor) -> Tensor:
        b = features.shape[0]
        cls = torch.ones(b, 1, self.input_dim, dtype=features.dtype)
        x = self.in_proj(torch.cat([cls, features], dim=1))
        x = self.positional_encoding.encode(x, coords, seq_id)
        return self.dropout(self.layer_norm(x))
    def _get_name(self) -> str:
        return "_PopTInputEmbedding"

class _PopTSpecPredictionHead(nn.Module):
    hidden_layer: nn.Linear
    act_fn: nn.GELU
    layer_norm: nn.LayerNorm
    output: nn.Linear
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.add_module("hidden_layer", self.hidden_layer)
        self.act_fn = nn.GELU()
        self.add_module("act_fn", self.act_fn)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.add_module("layer_norm", self.layer_norm)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.add_module("output", self.output)
    def forward(self, h: Tensor) -> Tensor:
        return self.output(self.layer_norm(self.act_fn(self.hidden_layer(h))))
    def _get_name(self) -> str:
        return "_PopTSpecPredictionHead"

class PopulationTransformer(EEGModuleMixin, nn.Module):
    """Population Transformer from Chau et al (2025)."""
    input_dim: int
    hidden_dim: int
    max_len: int
    coord_units: str
    shift_coords: bool
    input_embedding: _PopTInputEmbedding
    transformer_encoder: nn.TransformerEncoder
    spec_prediction_head: _PopTSpecPredictionHead
    final_layer: nn.Linear
    electrode_coords: Tensor

    def __init__(self, n_outputs: Optional[int] = None, n_chans: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, n_times: Optional[int] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None, hidden_dim: int = 512,
                 ffn_dim: int = 2048, n_layers: int = 6, n_heads: int = 8, max_len: int = 5000, coord_units: str = "m",
                 shift_coords: bool = False, activation = nn.GELU, drop_prob: float = 0.1,
                 channel_strategy: str = "native"):
        if channel_strategy != "native":
            raise NotImplementedError("PopulationTransformer: only channel_strategy='native' is supported in Codon")
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if coord_units != "m" and coord_units != "raw":
            raise ValueError(f"coord_units must be 'm' or 'raw', got {coord_units!r}.")
        act = activation()._get_name()
        if act != "GELU" and act != "ReLU":
            raise NotImplementedError("PopulationTransformer: activation must be nn.GELU or nn.ReLU in Codon")
        self.input_dim = self.n_times
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.coord_units = coord_units
        self.shift_coords = shift_coords
        self.input_embedding = _PopTInputEmbedding(self.input_dim, hidden_dim, max_len, drop_prob)
        self.add_module("input_embedding", self.input_embedding)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=ffn_dim,
                                           activation=act.lower(), dropout=drop_prob, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.add_module("transformer_encoder", self.transformer_encoder)
        self.spec_prediction_head = _PopTSpecPredictionHead(hidden_dim, self.input_dim)
        self.add_module("spec_prediction_head", self.spec_prediction_head)
        self.final_layer = nn.Linear(hidden_dim, self.n_outputs)
        self.add_module("final_layer", self.final_layer)
        self.electrode_coords = self._coords_from_chs_info()
        self.register_buffer("electrode_coords", self.electrode_coords, persistent=False)

    def _coords_from_chs_info(self) -> Tensor:
        n = self.n_chans
        vals = List[float]()
        ok = self._chs_info is not None
        if ok:
            chs = self._chs_info.__val__()
            for i in range(n):
                if i >= len(chs) or "loc" not in chs[i]:
                    ok = False
                    break
                parts = chs[i]["loc"].split()
                if len(parts) < 3:
                    ok = False
                    break
                for k in range(3):
                    v = float(float32(float(parts[k])))
                    if not math.isfinite(v):
                        ok = False
                    vals.append(v)
            if ok and all(abs(v) <= 1e-8 for v in vals):
                ok = False
        coords = torch.arange(n).unsqueeze(1).repeat(1, 3)
        if ok:
            lt = torch.tensor(vals, dtype=torch.float32).reshape(n, 3)
            if self.coord_units == "m":
                lt = lt * 1000.0
            coords = lt.round().long()
            if self.shift_coords:
                coords = coords - coords.amin(dim=0, keepdim=True)
            if ((coords < 0) | (coords >= self.max_len)).any().item():
                print(f"UserWarning: Some electrode coordinates from chs_info fall outside [0, {self.max_len - 1}] and are clamped.")
        return coords.clamp(0, self.max_len - 1)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        rm = Dict[str, Tensor]()
        for k, v in state_dict.items():
            if k.startswith("final_layer.norm."):
                continue
            if k.startswith("final_layer.fc."):
                rm["final_layer." + k[len("final_layer.fc."):]] = v
            else:
                rm[k] = v
        return nn.Module.load_state_dict(self, rm, strict, assign)

    def forward(self, x: Tensor) -> Tensor:
        b, n = x.shape[0], x.shape[1]
        if n != self.electrode_coords.shape[0]:
            raise ValueError(f"x has {n} electrodes but the model has coordinates for {self.electrode_coords.shape[0]}.")
        coords = self.electrode_coords.unsqueeze(0).expand(b, -1, -1)
        seq_id = torch.zeros(b, n, dtype=torch.int64)
        h = self.input_embedding.embed(x, coords, seq_id)
        z = self.transformer_encoder.encode(h)
        return self.final_layer(z[:, 0, :])

    def _get_name(self) -> str:
        return "PopulationTransformer"
