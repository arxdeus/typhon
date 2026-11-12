# Ported from torch 2.15.0 (BSD-3-Clause)
#
# MultiheadAttention, TransformerEncoderLayer, TransformerEncoder.
# Math follows torch's slow (non-fused) path, which is numerically what CPU
# torch computes in training mode.

import math
from torch.nn._util import _or
from torch._tensor import Tensor, no_grad, _DType
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.layers import Linear, LayerNorm, Dropout, _P, _empty, Parameter
import torch.nn.functional as F
import torch.nn.init as init

class NonDynamicallyQuantizableLinear(Linear):
    _f_NonDynamicallyQuantizableLinear: int
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device = None, dtype: Optional[_DType] = None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self._f_NonDynamicallyQuantizableLinear = 0

class MultiheadAttention(Module):
    embed_dim: int
    kdim: int
    vdim: int
    _qkv_same_embed_dim: bool
    num_heads: int
    dropout: float
    batch_first: bool
    head_dim: int
    in_proj_weight: Optional[Tensor]
    q_proj_weight: Optional[Tensor]
    k_proj_weight: Optional[Tensor]
    v_proj_weight: Optional[Tensor]
    in_proj_bias: Optional[Tensor]
    out_proj: Linear
    bias_k: Optional[Tensor]
    bias_v: Optional[Tensor]
    add_zero_attn: bool

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True, add_bias_kv: bool = False,
                 add_zero_attn: bool = False, kdim: Optional[int] = None, vdim: Optional[int] = None, batch_first: bool = False,
                 device = None, dtype: Optional[_DType] = None):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(f"embed_dim and num_heads must be greater than 0, got embed_dim={embed_dim} and num_heads={num_heads} instead")
        self.embed_dim = embed_dim
        self.kdim = _or(kdim, embed_dim)
        self.vdim = _or(vdim, embed_dim)
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise AssertionError("embed_dim must be divisible by num_heads")
        self.in_proj_weight = None
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        if not self._qkv_same_embed_dim:
            q = _P([embed_dim, embed_dim], dtype)
            k = _P([embed_dim, self.kdim], dtype)
            v = _P([embed_dim, self.vdim], dtype)
            self.q_proj_weight = q
            self.k_proj_weight = k
            self.v_proj_weight = v
            self.register_parameter("q_proj_weight", q)
            self.register_parameter("k_proj_weight", k)
            self.register_parameter("v_proj_weight", v)
            self.register_parameter("in_proj_weight", None)
        else:
            w = _P([3 * embed_dim, embed_dim], dtype)
            self.in_proj_weight = w
            self.register_parameter("in_proj_weight", w)
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
        self.in_proj_bias = None
        if bias:
            b = _P([3 * embed_dim], dtype)
            self.in_proj_bias = b
            self.register_parameter("in_proj_bias", b)
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias, device, dtype)
        self.add_module("out_proj", self.out_proj)
        self.bias_k = None
        self.bias_v = None
        if add_bias_kv:
            bk = _P([1, 1, embed_dim], dtype)
            bv = _P([1, 1, embed_dim], dtype)
            self.bias_k = bk
            self.bias_v = bv
            self.register_parameter("bias_k", bk)
            self.register_parameter("bias_v", bv)
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            init.xavier_uniform_(self.in_proj_weight)
        else:
            init.xavier_uniform_(self.q_proj_weight)
            init.xavier_uniform_(self.k_proj_weight)
            init.xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            init.constant_(self.in_proj_bias, 0.0)
            init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            init.xavier_normal_(self.bias_v)

    def _get_name(self) -> str:
        return "MultiheadAttention"

    def forward(self, input: Tensor) -> Tensor:
        out, _ = self.attend(input, input, input, need_weights=False)
        return out

    def __call__(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                 need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True,
                 is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        return self.attend(query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal)

    def attend(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
               need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True,
               is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """Multi-arg forward (torch's MultiheadAttention.forward). Returns
        (attn_output, attn_weights or None)."""
        from torch import tensor, bool as tbool, cat, zeros
        is_batched = query.dim() == 3
        q = query
        k = key
        v = value
        if self.batch_first and is_batched:
            q = q.transpose(1, 0)
            k = k.transpose(1, 0)
            v = v.transpose(1, 0)
        if not is_batched:
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
        # q: [L, N, E]
        L = q.size(0)
        N = q.size(1)
        E = self.embed_dim
        H = self.num_heads
        D = self.head_dim
        qp: Tensor
        kp: Tensor
        vp: Tensor
        b_q: Optional[Tensor] = None
        b_k: Optional[Tensor] = None
        b_v: Optional[Tensor] = None
        if self.in_proj_bias is not None:
            ib = self.in_proj_bias
            b_q = ib.narrow(0, 0, E)
            b_k = ib.narrow(0, E, E)
            b_v = ib.narrow(0, 2 * E, E)
        if self._qkv_same_embed_dim:
            w = self.in_proj_weight
            qp = F.linear(q, w.narrow(0, 0, E), b_q)
            kp = F.linear(k, w.narrow(0, E, E), b_k)
            vp = F.linear(v, w.narrow(0, 2 * E, E), b_v)
        else:
            qp = F.linear(q, self.q_proj_weight, b_q)
            kp = F.linear(k, self.k_proj_weight, b_k)
            vp = F.linear(v, self.v_proj_weight, b_v)
        am: Optional[Tensor] = attn_mask
        kpm: Optional[Tensor] = key_padding_mask
        if self.bias_k is not None and self.bias_v is not None:
            kp = cat([kp, self.bias_k.expand(1, N, E)], 0)
            vp = cat([vp, self.bias_v.expand(1, N, E)], 0)
            if am is not None:
                am = F.pad(am, [0, 1])
            if kpm is not None:
                kpm = F.pad(kpm, [0, 1])
        S = kp.size(0)
        # [L, N, E] -> [N*H, L, D]
        qh = qp.reshape([L, N * H, D]).transpose(0, 1)
        kh = kp.reshape([S, N * H, D]).transpose(0, 1)
        vh = vp.reshape([S, N * H, D]).transpose(0, 1)
        if self.add_zero_attn:
            kh = cat([kh, zeros([N * H, 1, D], dtype=kh.dtype)], 1)
            vh = cat([vh, zeros([N * H, 1, D], dtype=vh.dtype)], 1)
            if am is not None:
                am = F.pad(am, [0, 1])
            if kpm is not None:
                kpm = F.pad(kpm, [0, 1])
            S += 1
        # merge masks into float additive mask [N*H or 1, L, S]
        mask: Optional[Tensor] = None
        if am is not None:
            m = am
            if m.dtype == tbool:
                m = zeros(m._shape, dtype=qh.dtype).masked_fill(m, -float("inf"))
            if m.dim() == 2:
                m = m.unsqueeze(0)
            mask = m
        if kpm is not None:
            km = kpm
            if km.dtype == tbool:
                km = zeros(km._shape, dtype=qh.dtype).masked_fill(km, -float("inf"))
            km = km.reshape([N, 1, 1, S]).expand(N, H, 1, S).reshape([N * H, 1, S])
            if mask is None:
                mask = km
            else:
                mask = mask + km
        if is_causal and mask is None:
            from torch import ones
            cm = ones([L, S], dtype=tbool).triu(1)
            mask = zeros([L, S], dtype=qh.dtype).masked_fill(cm, -float("inf")).unsqueeze(0)
        dp = self.dropout if self.training else 0.0
        if need_weights:
            qs = qh * (1.0 / math.sqrt(float(D)))
            aw = qs.matmul(kh.transpose(-2, -1))
            if mask is not None:
                aw = aw + mask
            aw = aw.softmax(-1)
            if dp > 0.0:
                aw = F.dropout(aw, dp, True)
            ao = aw.matmul(vh)
            ao = ao.transpose(0, 1).reshape([L * N, E])
            ao = F.linear(ao, self.out_proj.weight, self.out_proj.bias).reshape([L, N, E])
            awv = aw.reshape([N, H, L, S])
            if average_attn_weights:
                awv = awv.mean(1)
            if not is_batched:
                ao = ao.squeeze(1)
                awv = awv.squeeze(0)
            if self.batch_first and is_batched:
                ao = ao.transpose(1, 0)
            return (ao, Optional(awv))
        else:
            qh4 = qh.reshape([N, H, L, D])
            kh4 = kh.reshape([N, H, S, D])
            vh4 = vh.reshape([N, H, S, D])
            m4: Optional[Tensor] = None
            if mask is not None:
                mm = mask
                if mm.size(0) == 1:
                    m4 = mm.unsqueeze(0)
                else:
                    m4 = mm.reshape([N, H, mm.size(1), S])  # L, or 1 for a key-padding-only mask
            ao = F.scaled_dot_product_attention(qh4, kh4, vh4, m4, dp, False)
            ao = ao.permute(2, 0, 1, 3).reshape([L * N, E])
            ao = F.linear(ao, self.out_proj.weight, self.out_proj.bias).reshape([L, N, E])
            if not is_batched:
                ao = ao.squeeze(1)
            if self.batch_first and is_batched:
                ao = ao.transpose(1, 0)
            return (ao, None)

class TransformerEncoderLayer(Module):
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
    linear1: Linear
    dropout: Dropout
    linear2: Linear
    norm1: LayerNorm
    norm2: LayerNorm
    dropout1: Dropout
    dropout2: Dropout

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
        self.dropout1 = Dropout(dropout)
        self.add_module("dropout1", self.dropout1)
        self.dropout2 = Dropout(dropout)
        self.add_module("dropout2", self.dropout2)

    def _get_name(self) -> str:
        return "TransformerEncoderLayer"

    def _act(self, x: Tensor) -> Tensor:
        return F.gelu(x) if self.activation_name == "gelu" else F.relu(x)

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool) -> Tensor:
        y, _ = self.self_attn.attend(x, x, x, key_padding_mask, False, attn_mask, True, is_causal)
        return self.dropout1(y)

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self._act(self.linear1(x)))))

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
               is_causal: bool = False) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def forward(self, input: Tensor) -> Tensor:
        return self.encode(input)

    def __call__(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                 is_causal: bool = False) -> Tensor:
        return self.encode(src, src_mask, src_key_padding_mask, is_causal)

    def _clone(self) -> TransformerEncoderLayer:
        from torch import get_rng_state, set_rng_state
        # deepcopy semantics: same values, no RNG consumption
        st = get_rng_state()
        c = TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout_p, self.activation_name,
                                    self.layer_norm_eps, self.batch_first, self.norm_first, self.has_bias,
                                    dtype=self.linear1.weight.dtype)
        set_rng_state(st)
        c.load_state_dict(self.state_dict())
        c.train(self.training)
        return c

class TransformerEncoder(Module):
    layers: ModuleList
    _layers: List[TransformerEncoderLayer]
    num_layers: int
    norm: Optional[Module]

    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int, norm: Optional[Module] = None,
                 enable_nested_tensor: bool = True, mask_check: bool = True):
        super().__init__()
        self._layers = [encoder_layer._clone() for _ in range(num_layers)]
        self.layers = ModuleList()
        for l in self._layers:
            self.layers.append(l)
        self.add_module("layers", self.layers)
        self.num_layers = num_layers
        self.norm = norm
        if norm is not None:
            self.add_module("norm", norm)

    def _get_name(self) -> str:
        return "TransformerEncoder"

    def encode(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
               is_causal: Optional[bool] = None) -> Tensor:
        x = src
        ic = _or(is_causal, False)
        for l in self._layers:
            x = l.encode(x, mask, src_key_padding_mask, ic)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def forward(self, input: Tensor) -> Tensor:
        return self.encode(input)

    def __call__(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                 is_causal: Optional[bool] = None) -> Tensor:
        return self.encode(src, mask, src_key_padding_mask, is_causal)
