# Ported from torch 2.15.0 (BSD-3-Clause)
#
# TransformerDecoderLayer, TransformerDecoder, Transformer (slow path).

from torch.nn._util import _or
from torch._tensor import Tensor, _DType
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.layers import Linear, LayerNorm, Dropout
from torch.nn.modules.transformer import MultiheadAttention, TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F
import torch.nn.init as init

class TransformerDecoderLayer(Module):
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout_p: float
    activation_name: str
    layer_norm_eps: float
    batch_first: bool
    norm_first: bool
    has_bias: bool
    self_attn: MultiheadAttention
    multihead_attn: MultiheadAttention
    linear1: Linear
    dropout: Dropout
    linear2: Linear
    norm1: LayerNorm
    norm2: LayerNorm
    norm3: LayerNorm
    dropout1: Dropout
    dropout2: Dropout
    dropout3: Dropout

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu",
                 layer_norm_eps: float = 1e-05, batch_first: bool = False, norm_first: bool = False, bias: bool = True,
                 device = None, dtype: Optional[_DType] = None):
        super().__init__()
        if activation != "relu" and activation != "gelu":
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.activation_name = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.has_bias = bias
        self.self_attn = MultiheadAttention(d_model, nhead, dropout, bias, batch_first=batch_first, dtype=dtype)
        self.add_module("self_attn", self.self_attn)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout, bias, batch_first=batch_first, dtype=dtype)
        self.add_module("multihead_attn", self.multihead_attn)
        self.linear1 = Linear(d_model, dim_feedforward, bias, dtype=dtype)
        self.add_module("linear1", self.linear1)
        self.dropout = Dropout(dropout)
        self.add_module("dropout", self.dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias, dtype=dtype)
        self.add_module("linear2", self.linear2)
        self.norm1 = LayerNorm(d_model, layer_norm_eps, bias=bias, dtype=dtype)
        self.add_module("norm1", self.norm1)
        self.norm2 = LayerNorm(d_model, layer_norm_eps, bias=bias, dtype=dtype)
        self.add_module("norm2", self.norm2)
        self.norm3 = LayerNorm(d_model, layer_norm_eps, bias=bias, dtype=dtype)
        self.add_module("norm3", self.norm3)
        self.dropout1 = Dropout(dropout)
        self.add_module("dropout1", self.dropout1)
        self.dropout2 = Dropout(dropout)
        self.add_module("dropout2", self.dropout2)
        self.dropout3 = Dropout(dropout)
        self.add_module("dropout3", self.dropout3)

    def _get_name(self) -> str:
        return "TransformerDecoderLayer"

    def _act(self, x: Tensor) -> Tensor:
        return F.gelu(x) if self.activation_name == "gelu" else F.relu(x)

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool) -> Tensor:
        y, _ = self.self_attn.attend(x, x, x, key_padding_mask, False, attn_mask, True, is_causal)
        return self.dropout1(y)

    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                   is_causal: bool) -> Tensor:
        y, _ = self.multihead_attn.attend(x, mem, mem, key_padding_mask, False, attn_mask, True, is_causal)
        return self.dropout2(y)

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout3(self.linear2(self.dropout(self._act(self.linear1(x)))))

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
               tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
               tgt_is_causal: bool = False, memory_is_causal: bool = False) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))
        return x

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError("TransformerDecoderLayer needs (tgt, memory): use decode()")

    def __call__(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                 tgt_is_causal: bool = False, memory_is_causal: bool = False) -> Tensor:
        return self.decode(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                           tgt_is_causal, memory_is_causal)

    def _clone(self) -> TransformerDecoderLayer:
        from torch import get_rng_state, set_rng_state
        st = get_rng_state()
        c = TransformerDecoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout_p, self.activation_name,
                                    self.layer_norm_eps, self.batch_first, self.norm_first, self.has_bias,
                                    dtype=self.linear1.weight.dtype)
        set_rng_state(st)
        c.load_state_dict(self.state_dict())
        c.train(self.training)
        return c

class TransformerDecoder(Module):
    layers: ModuleList
    _layers: List[TransformerDecoderLayer]
    num_layers: int
    norm: Optional[Module]

    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int, norm: Optional[Module] = None):
        super().__init__()
        self._layers = [decoder_layer._clone() for _ in range(num_layers)]
        self.layers = ModuleList()
        for l in self._layers:
            self.layers.append(l)
        self.add_module("layers", self.layers)
        self.num_layers = num_layers
        self.norm = norm
        if norm is not None:
            self.add_module("norm", norm)

    def _get_name(self) -> str:
        return "TransformerDecoder"

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
               tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
               tgt_is_causal: Optional[bool] = None, memory_is_causal: bool = False) -> Tensor:
        x = tgt
        ic = _or(tgt_is_causal, False)
        for l in self._layers:
            x = l.decode(x, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, ic,
                         memory_is_causal)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError("TransformerDecoder needs (tgt, memory): use decode()")

    def __call__(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                 tgt_is_causal: Optional[bool] = None, memory_is_causal: bool = False) -> Tensor:
        return self.decode(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                           tgt_is_causal, memory_is_causal)

class Transformer(Module):
    encoder: TransformerEncoder
    decoder: TransformerDecoder
    d_model: int
    nhead: int
    batch_first: bool

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu", layer_norm_eps: float = 1e-05,
                 batch_first: bool = False, norm_first: bool = False, bias: bool = True, device = None,
                 dtype: Optional[_DType] = None):
        super().__init__()
        el = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first,
                                     norm_first, bias, dtype=dtype)
        en = LayerNorm(d_model, layer_norm_eps, bias=bias, dtype=dtype)
        self.encoder = TransformerEncoder(el, num_encoder_layers, en)
        self.add_module("encoder", self.encoder)
        dl = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first,
                                     norm_first, bias, dtype=dtype)
        dn = LayerNorm(d_model, layer_norm_eps, bias=bias, dtype=dtype)
        self.decoder = TransformerDecoder(dl, num_decoder_layers, dn)
        self.add_module("decoder", self.decoder)
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def _get_name(self) -> str:
        return "Transformer"

    def run(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        memory = self.encoder.encode(src, src_mask, src_key_padding_mask)
        return self.decoder.decode(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError("Transformer needs (src, tgt): use run()")

    def __call__(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return self.run(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask,
                        memory_key_padding_mask)
