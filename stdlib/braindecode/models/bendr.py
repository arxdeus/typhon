# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from braindecode.models.base import EEGModuleMixin
from braindecode.functional.initialization import _param_by_name

BENDR_CHANNEL_ORDER = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'T5', 'P3', 'PZ',
                       'P4', 'T6', 'O1', 'O2', 'SCALE']

class _ConvEncoderBENDR(nn.Module):
    encoder_h: int
    in_features: int
    encoder: nn.Sequential

    def __init__(self, in_features: int, encoder_h: int = 512, enc_width: List[int] = [3, 2, 2, 2, 2, 2],
                 dropout: float = 0.0, projection_head: bool = False, enc_downsample: List[int] = [3, 2, 2, 2, 2, 2],
                 activation = nn.GELU):
        super().__init__()
        self.encoder_h = encoder_h
        self.in_features = in_features
        if len(enc_downsample) != len(enc_width):
            raise ValueError("Encoder width and downsampling factors must have the same length.")
        cur = in_features
        self.encoder = nn.Sequential()
        for i in range(len(enc_width)):
            w = enc_width[i] if enc_width[i] % 2 != 0 else enc_width[i] + 1
            self.encoder.add_module(f"Encoder_{i}", nn.Sequential(
                nn.Conv1d(cur, encoder_h, w, stride=enc_downsample[i], padding=w // 2), nn.Dropout1d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h), activation()))
            cur = encoder_h
        if projection_head:
            self.encoder.add_module("projection_head", nn.Conv1d(encoder_h, encoder_h, 1))
        self.add_module("encoder", self.encoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def _get_name(self) -> str:
        return "_ConvEncoderBENDR"

class _TFixupEncoderLayer(nn.Module):
    # nn.TransformerEncoderLayer (seq-first, post-norm) with norm1/norm2
    # replaced by Identity, as in braindecode.
    self_attn: nn.MultiheadAttention
    linear1: nn.Linear
    dropout: nn.Dropout
    linear2: nn.Linear
    dropout1: nn.Dropout
    dropout2: nn.Dropout
    activation: nn.Module

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: nn.Module):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, True, batch_first=False)
        self.add_module("self_attn", self.self_attn)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.add_module("linear1", self.linear1)
        self.dropout = nn.Dropout(dropout)
        self.add_module("dropout", self.dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.add_module("linear2", self.linear2)
        self.add_module("norm1", nn.Identity())
        self.add_module("norm2", nn.Identity())
        self.dropout1 = nn.Dropout(dropout)
        self.add_module("dropout1", self.dropout1)
        self.dropout2 = nn.Dropout(dropout)
        self.add_module("dropout2", self.dropout2)
        self.activation = activation
        self.add_module("activation", activation)

    def forward(self, src: Tensor) -> Tensor:
        y, _ = self.self_attn.attend(src, src, src, need_weights=False)
        x = src + self.dropout1(y)
        return x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def _get_name(self) -> str:
        return "_TFixupEncoderLayer"

def _bert_init(scale: float):
    def f(m):
        n = m._get_name()
        if n in ("Linear", "NonDynamicallyQuantizableLinear"):
            w = _param_by_name(m, "weight")
            if w is not None:
                with torch.no_grad():
                    nn.init.xavier_uniform_(w.__val__())
                    w.__val__().mul_(scale)
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
    return f

class _BENDRContextualizer(nn.Module):
    dropout: float
    layer_drop: float
    start_token: Optional[float]
    transformer_dim: int
    in_features: int
    relative_position: Optional[nn.Sequential]
    input_conditioning: nn.Sequential
    transformer_layers: nn.ModuleList
    _layers: List[_TFixupEncoderLayer]
    norm: nn.LayerNorm
    output_layer: nn.Conv1d

    def __init__(self, in_features: int, hidden_feedforward: int = 3076, heads: int = 8, layers: int = 8,
                 dropout: float = 0.1, activation = nn.GELU, position_encoder: int = 25, layer_drop: float = 0.0,
                 start_token: Optional[float] = -5.0):
        super().__init__()
        self.dropout = dropout
        self.layer_drop = layer_drop
        self.start_token = start_token
        self.transformer_dim = 3 * in_features
        self.in_features = in_features
        self.relative_position = None
        if position_encoder > 0:
            conv = nn.Conv1d(in_features, in_features, kernel_size=position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0.0, std=2.0 / float(self.transformer_dim))
            nn.init.constant_(conv.bias.__val__(), 0.0)
            weight_norm(conv, name="weight", dim=2)
            rp = nn.Sequential(conv, activation())
            self.relative_position = rp
            self.add_module("relative_position", rp)
        self.input_conditioning = nn.Sequential(nn.Identity(), nn.LayerNorm(in_features), nn.Dropout(dropout),
                                                nn.Linear(in_features, self.transformer_dim), nn.Identity())
        self.add_module("input_conditioning", self.input_conditioning)
        # copy.deepcopy(encoder_layer): identical weights, replayed RNG state
        s0 = torch.get_rng_state()
        self._layers = List[_TFixupEncoderLayer]()
        s1 = s0
        for i in range(layers):
            if i > 0:
                torch.set_rng_state(s0)
            self._layers.append(_TFixupEncoderLayer(self.transformer_dim, heads, hidden_feedforward, dropout, activation()))
            if i == 0:
                s1 = torch.get_rng_state()
        if layers == 0:
            _TFixupEncoderLayer(self.transformer_dim, heads, hidden_feedforward, dropout, activation())
            s1 = torch.get_rng_state()
        torch.set_rng_state(s1)
        self.transformer_layers = nn.ModuleList()
        for l in self._layers:
            self.transformer_layers.append(l)
        self.add_module("transformer_layers", self.transformer_layers)
        self.norm = nn.LayerNorm(self.transformer_dim)
        self.norm.requires_grad_(False)
        self.add_module("norm", self.norm)
        self.output_layer = nn.Conv1d(self.transformer_dim, in_features, 1)
        self.add_module("output_layer", self.output_layer)
        self.apply(_bert_init(0.67 * float(layers) ** (-0.25)))

    def forward(self, x: Tensor) -> Tensor:
        if self.relative_position is not None:
            x = x + self.relative_position.__val__()(x)
        x = self.input_conditioning(x.permute(0, 2, 1)).permute(1, 0, 2)
        if self.start_token is not None:
            tok = torch.full([1, x.shape[1], x.shape[2]], self.start_token.__val__(), dtype=x.dtype)
            x = torch.cat([tok, x], dim=0)
        for l in self._layers:
            if not self.training or torch.rand(1).item() > self.layer_drop:
                x = l(x)
        return self.output_layer(x.permute(1, 2, 0))

    def _get_name(self) -> str:
        return "_BENDRContextualizer"

class BENDR(EEGModuleMixin, nn.Module):
    """BENDR from Kostas et al (2021). Architecture only (no Hub weights).

    Remapping arbitrary ``chs_info`` montages onto the 20 BENDR channels
    (braindecode's spline ChannelLayer) is not ported: chs_info, when given,
    must already be in BENDR_CHANNEL_ORDER."""
    encoder_h: int
    contextualizer_hidden: int
    include_final_layer: bool
    encoder_only: bool
    encoder: _ConvEncoderBENDR
    contextualizer: _BENDRContextualizer
    final_layer: Optional[nn.Linear]

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 chs_info: Optional[List[Dict[str, str]]] = None, input_window_seconds: Optional[float] = None,
                 sfreq: Optional[float] = None, encoder_h: int = 512, contextualizer_hidden: int = 3076,
                 projection_head: bool = False, drop_prob: float = 0.1, layer_drop: float = 0.0, activation = nn.GELU,
                 transformer_layers: int = 8, transformer_heads: int = 8, position_encoder_length: int = 25,
                 enc_width: List[int] = [3, 2, 2, 2, 2, 2], enc_downsample: List[int] = [3, 2, 2, 2, 2, 2],
                 start_token: Optional[float] = -5.0, final_layer: bool = True, encoder_only: bool = False):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Module.__init__(self)
        if self._chs_info is not None:
            names = [ch["ch_name"].lower() for ch in self.chs_info]
            if names != [n.lower() for n in BENDR_CHANNEL_ORDER]:
                raise NotImplementedError("BENDR: channel remapping (ChannelLayer) is not supported; "
                                          "pass channels in BENDR_CHANNEL_ORDER or use n_chans.")
        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden
        self.include_final_layer = final_layer
        self.encoder_only = encoder_only
        self.encoder = _ConvEncoderBENDR(self.n_chans, encoder_h, enc_width, drop_prob, projection_head,
                                         enc_downsample, activation)
        self.add_module("encoder", self.encoder)
        self.contextualizer = _BENDRContextualizer(encoder_h, contextualizer_hidden, transformer_heads,
                                                   transformer_layers, drop_prob, activation, position_encoder_length,
                                                   layer_drop, start_token)
        self.add_module("contextualizer", self.contextualizer)
        self.final_layer = None
        if final_layer:
            lin = nn.Linear(encoder_h * 4 if encoder_only else encoder_h, self.n_outputs)
            weight_norm(lin, name="weight", dim=1)
            self.final_layer = lin
            self.add_module("final_layer", lin)

    def forward_features(self, x: Tensor) -> Tensor:
        enc = self.encoder(x)
        if self.encoder_only:
            if enc.shape[2] < 4:
                raise RuntimeError(f"encoder_only needs at least 4 encoded time steps, got {enc.shape[2]}.")
            chunks = torch.tensor_split(enc, 4, dim=2)
            return torch.cat([c.mean(dim=2) for c in chunks], dim=1)
        return self.contextualizer(enc)[:, :, 0]

    def forward(self, x: Tensor) -> Tensor:
        f = self.forward_features(x)
        if self.final_layer is not None:
            f = self.final_layer.__val__()(f)
        return f

    def _get_name(self) -> str:
        return "BENDR"
