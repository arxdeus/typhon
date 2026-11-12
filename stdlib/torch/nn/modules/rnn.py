# Ported from torch 2.15.0 (BSD-3-Clause)
#
# RNN, LSTM, GRU (multi-layer, bidirectional, batch_first, dropout between
# layers). Parameters are registered in torch's order with torch's names
# (weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, ..._reverse) and
# initialized by uniform_(-1/sqrt(H), 1/sqrt(H)) in that order. The step math
# is composed from autograd-aware Tensor ops (gate layouts: LSTM i,f,g,o;
# GRU r,z,n).

import math
from torch._tensor import Tensor, _DType
from torch.nn.modules.module import Module
from torch.nn.modules.layers import _P
import torch.nn.functional as F
import torch.nn.init as init

class _RNNBase(Module):
    """Typhon note: `m(x)` (one argument) returns only the output sequence
    (Module's single-tensor call protocol, so RNNs work inside Sequential).
    `m(x, None)` / `m(x, h0)` returns torch's tuple (output, h_n) or, for
    LSTM, (output, (h_n, c_n))."""
    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    _w: List[List[Tensor]]   # per (layer, direction): [w_ih, w_hh, (b_ih, b_hh)]

    def __init__(self):
        super().__init__()

    def _setup(self, mode: str, input_size: int, hidden_size: int, num_layers: int, bias: bool, batch_first: bool,
               dropout: float, bidirectional: bool, dtype: Optional[_DType]):
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self._w = []
        gate = hidden_size
        if mode == "LSTM":
            gate = 4 * hidden_size
        elif mode == "GRU":
            gate = 3 * hidden_size
        nd = 2 if bidirectional else 1
        for layer in range(num_layers):
            for d in range(nd):
                lin = input_size if layer == 0 else hidden_size * nd
                suffix = "_reverse" if d == 1 else ""
                ps = [_P([gate, lin], dtype), _P([gate, hidden_size], dtype)]
                names = [f"weight_ih_l{layer}{suffix}", f"weight_hh_l{layer}{suffix}"]
                if bias:
                    ps.append(_P([gate], dtype))
                    ps.append(_P([gate], dtype))
                    names.append(f"bias_ih_l{layer}{suffix}")
                    names.append(f"bias_hh_l{layer}{suffix}")
                for i in range(len(ps)):
                    self.register_parameter(names[i], ps[i])
                self._w.append(ps)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        for w in self.parameters():
            init.uniform_(w, -stdv, stdv)

    def _cell(self, x: Tensor, h: Tensor, c: Tensor, ps: List[Tensor]) -> Tuple[Tensor, Tensor]:
        b_ih: Optional[Tensor] = ps[2] if self.bias else None
        b_hh: Optional[Tensor] = ps[3] if self.bias else None
        gi = F.linear(x, ps[0], b_ih)
        gh = F.linear(h, ps[1], b_hh)
        H = self.hidden_size
        if self.mode == "LSTM":
            g = gi + gh
            i = g.narrow(-1, 0, H).sigmoid()
            f = g.narrow(-1, H, H).sigmoid()
            gg = g.narrow(-1, 2 * H, H).tanh()
            o = g.narrow(-1, 3 * H, H).sigmoid()
            c2 = f * c + i * gg
            return (o * c2.tanh(), c2)
        elif self.mode == "GRU":
            r = (gi.narrow(-1, 0, H) + gh.narrow(-1, 0, H)).sigmoid()
            z = (gi.narrow(-1, H, H) + gh.narrow(-1, H, H)).sigmoid()
            n = (gi.narrow(-1, 2 * H, H) + r * gh.narrow(-1, 2 * H, H)).tanh()
            return ((1.0 - z) * n + z * h, c)
        elif self.mode == "RNN_RELU":
            return ((gi + gh).relu(), c)
        else:
            return ((gi + gh).tanh(), c)

    def run(self, input: Tensor, hx: Optional[Tensor] = None, cx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (output, h_n, c_n); c_n is meaningful only for LSTM."""
        from torch import zeros, stack, cat
        batched = input.dim() == 3
        x = input if batched else input.unsqueeze(1 if self.batch_first else 1)
        if not batched and self.batch_first:
            x = input.unsqueeze(0)
        if self.batch_first:
            x = x.transpose(0, 1)
        # x: [T, N, I]
        T = x.size(0)
        N = x.size(1)
        nd = 2 if self.bidirectional else 1
        H = self.hidden_size
        h0: Tensor
        c0: Tensor
        if hx is not None:
            h0 = hx.__val__()
            if not batched:
                h0 = h0.unsqueeze(1)
        else:
            h0 = zeros([self.num_layers * nd, N, H], dtype=x.dtype)
        if cx is not None:
            c0 = cx.__val__()
            if not batched:
                c0 = c0.unsqueeze(1)
        else:
            c0 = zeros([self.num_layers * nd, N, H], dtype=x.dtype)
        hn = List[Tensor]()
        cn = List[Tensor]()
        layer_in = x
        for layer in range(self.num_layers):
            outs_dir = List[Tensor]()
            for d in range(nd):
                idx = layer * nd + d
                ps = self._w[idx]
                h = h0[idx]
                c = c0[idx]
                seq = List[Tensor]()
                for k in range(T):
                    t = T - 1 - k if d == 1 else k
                    h, c = self._cell(layer_in[t], h, c, ps)
                    seq.append(h)
                if d == 1:
                    seq = seq[::-1]
                outs_dir.append(stack(seq, 0))
                hn.append(h)
                cn.append(c)
            layer_out = outs_dir[0] if nd == 1 else cat(outs_dir, 2)
            if self.dropout > 0.0 and self.training and layer < self.num_layers - 1:
                layer_out = F.dropout(layer_out, self.dropout, True)
            layer_in = layer_out
        out = layer_in
        if self.batch_first:
            out = out.transpose(0, 1)
        h_n = stack(hn, 0)
        c_n = stack(cn, 0)
        if not batched:
            out = out.squeeze(0 if self.batch_first else 1)
            h_n = h_n.squeeze(1)
            c_n = c_n.squeeze(1)
        return (out, h_n, c_n)

    def forward(self, input: Tensor) -> Tensor:
        out, _, _ = self.run(input)
        return out

    def extra_repr(self) -> str:
        s = f"{self.input_size}, {self.hidden_size}"
        if self.num_layers != 1:
            s += f", num_layers={self.num_layers}"
        if not self.bias:
            s += ", bias=False"
        if self.batch_first:
            s += ", batch_first=True"
        if self.dropout != 0.0:
            s += f", dropout={self.dropout}"
        if self.bidirectional:
            s += ", bidirectional=True"
        return s

class RNN(_RNNBase):
    nonlinearity: str

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, nonlinearity: str = "tanh", bias: bool = True,
                 batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self.nonlinearity = nonlinearity
        mode = "RNN_TANH" if nonlinearity == "tanh" else "RNN_RELU"
        if nonlinearity != "tanh" and nonlinearity != "relu":
            raise ValueError(f"Unknown nonlinearity '{nonlinearity}'. Select from 'tanh' or 'relu'.")
        self._setup(mode, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, dtype)

    def __call__(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        out, h, _ = self.run(input, hx)
        return (out, h)

    def _get_name(self) -> str: return "RNN"

class GRU(_RNNBase):
    _gru: int

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self._gru = 0
        self._setup("GRU", input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, dtype)

    def __call__(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        out, h, _ = self.run(input, hx)
        return (out, h)

    def _get_name(self) -> str: return "GRU"

class LSTM(_RNNBase):
    _lstm: int

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False, proj_size: int = 0, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self._lstm = 0
        if proj_size != 0:
            raise NotImplementedError("LSTM proj_size is not supported")
        self._setup("LSTM", input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, dtype)

    def __call__(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h0: Optional[Tensor] = None
        c0: Optional[Tensor] = None
        if hx is not None:
            hh = hx.__val__()
            h0 = hh[0]
            c0 = hh[1]
        out, h, c = self.run(input, h0, c0)
        return (out, (h, c))

    def _get_name(self) -> str: return "LSTM"
