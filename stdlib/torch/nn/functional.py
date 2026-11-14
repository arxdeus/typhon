# Ported from torch 2.15.0 (BSD-3-Clause)
#
# torch.nn.functional for Typhon. Composite ops are built from autograd-aware
# Tensor methods; convolution, pooling and padding use native kernels in
# torch.nn._ops with explicit backward nodes.

import math as _math
from torch.nn._util import _or
from torch._tensor import Tensor, _F32, _F64, _new
import torch._tensor as _T
from torch._random import Generator
from torch.nn._ops import conv_transpose2d_raw, conv2d_raw, max_pool2d_raw, avg_pool2d_raw, pad_raw, pool_out_size

# ---------------------------------------------------------------- arg helpers

def _pair(v) -> Tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    elif isinstance(v, Tuple[int, int]):
        return v
    elif isinstance(v, Tuple[int]):
        return (v[0], v[0])
    elif isinstance(v, List[int]):
        if len(v) == 1:
            return (v[0], v[0])
        return (v[0], v[1])
    else:
        compile_error("expected int or pair of ints")

def _single(v) -> int:
    if isinstance(v, Optional[int]):
        return v.__val__()
    elif isinstance(v, int):
        return v
    elif isinstance(v, List[int]):
        return v[0]
    else:
        return v[0]

def _opt_pair(v, default: Tuple[int, int]) -> Tuple[int, int]:
    if v is None:
        return default
    else:
        return _pair(v)

def _wrap(d: int, n: int) -> int:
    if d < -n or d >= n:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-n}, {n-1}], but got {d})")
    return d + n if d < 0 else d

# ---------------------------------------------------------------- linear

def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    if (input.ndim == 2 and weight.ndim == 2 and input._dt == _F32 and weight._dt == _F32
            and (bias is None or (bias.__val__()._dt == _F32 and bias.__val__().ndim == 1))):
        # ATen addmm on CPU BLAS (bit-exact, incl. backward)
        from torch.nn._linear import linear_f32
        return linear_f32(input, weight, bias)
    out = input.matmul(weight.t())
    if bias is not None:
        out = out + bias.__val__()
    return out

def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    # out[..., o] = sum_ij x1_i W[o,i,j] x2_j
    O = weight.size(0)
    outs = List[Tensor]()
    y = input1.unsqueeze(-2).matmul(weight.unsqueeze(0) if input1.ndim > 1 else weight)
    out = (y * input2.unsqueeze(-2)).sum(-1)
    if bias is not None:
        out = out + bias.__val__()
    return out

# ---------------------------------------------------------------- convolution

def _same_pad(L: int, k: int, d: int) -> Tuple[int, int]:
    total = d * (k - 1)
    left = total // 2
    return (left, total - left)

def _conv2d_impl(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: Tuple[int, int],
                 padding_mode: str, pad_hw: Tuple[int, int], same: bool, dilation: Tuple[int, int], groups: int) -> Tensor:
    batched = input.ndim == 4
    if input.ndim != 3 and input.ndim != 4:
        raise RuntimeError(f"Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: {input.shape}")
    x = input if batched else input.unsqueeze(0)
    if same:
        if stride[0] != 1 or stride[1] != 1:
            raise RuntimeError("padding='same' is not supported for strided convolutions")
        lh, rh = _same_pad(x.size(2), weight.size(2), dilation[0])
        lw, rw = _same_pad(x.size(3), weight.size(3), dilation[1])
        if lh == rh and lw == rw:
            out = conv2d_raw(x, weight, bias, stride, (lh, lw), dilation, groups, x._shape.copy(), weight._shape.copy())
        else:
            x = pad_raw(x, [lw, rw, lh, rh], 0, 0.0)
            out = conv2d_raw(x, weight, bias, stride, (0, 0), dilation, groups, x._shape.copy(), weight._shape.copy())
    else:
        out = conv2d_raw(x, weight, bias, stride, pad_hw, dilation, groups, x._shape.copy(), weight._shape.copy())
    return out if batched else out.squeeze(0)

def _parse_padding2(padding) -> Tuple[Tuple[int, int], bool]:
    if isinstance(padding, str):
        if padding == "same":
            return ((0, 0), True)
        elif padding == "valid":
            return ((0, 0), False)
        else:
            raise ValueError(f"Invalid padding string '{padding}', should be one of {{'valid', 'same'}}")
    else:
        return (_pair(padding), False)

def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride = 1, padding = 0, dilation = 1, groups: int = 1) -> Tensor:
    p, same = _parse_padding2(padding)
    return _conv2d_impl(input, weight, bias, _pair(stride), "zeros", p, same, _pair(dilation), groups)

def conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride = 1, padding = 0, dilation = 1, groups: int = 1) -> Tensor:
    batched = input.ndim == 3
    if input.ndim != 2 and input.ndim != 3:
        raise RuntimeError(f"Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: {input.shape}")
    x = (input if batched else input.unsqueeze(0)).unsqueeze(2)
    w = weight.unsqueeze(2)
    p = 0
    same = False
    if isinstance(padding, str):
        if padding == "same":
            same = True
        elif padding != "valid":
            raise ValueError(f"Invalid padding string '{padding}', should be one of {{'valid', 'same'}}")
    else:
        p = _single(padding)
    out = _conv2d_impl(x, w, bias, (1, _single(stride)), "zeros", (0, p), same, (1, _single(dilation)), groups)
    out = out.squeeze(2)
    return out if batched else out.squeeze(0)

def conv_transpose2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride = 1, padding = 0, output_padding = 0,
                     groups: int = 1, dilation = 1) -> Tensor:
    batched = input.ndim == 4
    if input.ndim != 3 and input.ndim != 4:
        raise RuntimeError(f"Expected 3D (unbatched) or 4D (batched) input to conv_transpose2d, but got input of size: {input.shape}")
    x = input if batched else input.unsqueeze(0)
    out = conv_transpose2d_raw(x, weight, bias, _pair(stride), _pair(padding), _pair(output_padding), _pair(dilation), groups,
                               x._shape.copy(), weight._shape.copy())
    return out if batched else out.squeeze(0)

def conv_transpose1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride = 1, padding = 0, output_padding = 0,
                     groups: int = 1, dilation = 1) -> Tensor:
    batched = input.ndim == 3
    if input.ndim != 2 and input.ndim != 3:
        raise RuntimeError(f"Expected 2D (unbatched) or 3D (batched) input to conv_transpose1d, but got input of size: {input.shape}")
    x = (input if batched else input.unsqueeze(0)).unsqueeze(2)
    w = weight.unsqueeze(2)
    out = conv_transpose2d_raw(x, w, bias, (1, _single(stride)), (0, _single(padding)), (0, _single(output_padding)),
                               (1, _single(dilation)), groups, x._shape.copy(), w._shape.copy()).squeeze(2)
    return out if batched else out.squeeze(0)

def conv3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride = 1, padding = 0, dilation = 1, groups: int = 1) -> Tensor:
    from torch.nn._conv3d import conv3d as _c3
    return _c3(input, weight, bias, stride, padding, dilation, groups)

# ---------------------------------------------------------------- padding

def _pad_mode(mode: str) -> int:
    if mode == "constant":
        return 0
    elif mode == "reflect":
        return 1
    elif mode == "replicate":
        return 2
    elif mode == "circular":
        return 3
    raise NotImplementedError(f"Unrecognised padding mode {mode}")

def pad(input: Tensor, pad, mode: str = "constant", value: Optional[float] = None) -> Tensor:
    pads = List[int]()
    for p in pad:
        pads.append(int(p))
    m = _pad_mode(mode)
    if m != 0:
        n = len(pads) // 2
        nd = input.ndim
        if not (nd == n + 1 or nd == n + 2):
            raise NotImplementedError(f"Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now")
    v = _or(value, 0.0)
    # negative constant pads crop (torch supports this); handled by maps
    return pad_raw(input, pads, m, v)

# ---------------------------------------------------------------- pooling

def _pool_args(kernel_size, stride, padding, dilation) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    k = _pair(kernel_size)
    s = k if stride is None else _pair(stride)
    if isinstance(stride, List[int]):
        if len(stride) == 0:
            s = k
    return (k, s, _pair(padding), _pair(dilation))

def max_pool2d(input: Tensor, kernel_size, stride = None, padding = 0, dilation = 1, ceil_mode: bool = False, return_indices: bool = False) -> Tensor:
    k, s, p, d = _pool_args(kernel_size, stride, padding, dilation)
    if input.ndim != 3 and input.ndim != 4:
        raise RuntimeError("non-empty 3D or 4D (batch mode) tensor expected for input")
    lead = input._shape[:input.ndim - 2]
    out, _ = max_pool2d_raw(input, k, s, p, d, ceil_mode, lead)
    return out

def max_pool1d(input: Tensor, kernel_size, stride = None, padding = 0, dilation = 1, ceil_mode: bool = False, return_indices: bool = False) -> Tensor:
    k = _single(kernel_size)
    s = k if stride is None else _single(stride)
    x = input.unsqueeze(-2)
    lead = input._shape[:input.ndim - 1]
    out, _ = max_pool2d_raw(x, (1, k), (1, s), (0, _single(padding)), (1, _single(dilation)), ceil_mode, lead)
    return out.squeeze(-2)

def avg_pool2d(input: Tensor, kernel_size, stride = None, padding = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None) -> Tensor:
    k, s, p, _ = _pool_args(kernel_size, stride, padding, 1)
    if input.ndim != 3 and input.ndim != 4:
        raise RuntimeError("non-empty 3D or 4D (batch mode) tensor expected for input")
    lead = input._shape[:input.ndim - 2]
    dov = _or(divisor_override, 0)
    return avg_pool2d_raw(input, k, s, p, ceil_mode, count_include_pad, dov, lead)

def avg_pool1d(input: Tensor, kernel_size, stride = None, padding = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> Tensor:
    k = _single(kernel_size)
    s = k if stride is None else _single(stride)
    x = input.unsqueeze(-2)
    lead = input._shape[:input.ndim - 1]
    out = avg_pool2d_raw(x, (1, k), (1, s), (0, _single(padding)), ceil_mode, count_include_pad, 0, lead)
    return out.squeeze(-2)

def _adaptive_bounds(i: int, insz: int, outsz: int) -> Tuple[int, int]:
    start = (i * insz) // outsz
    end = ((i + 1) * insz + outsz - 1) // outsz
    return (start, end)

def _adaptive_avg_last(x: Tensor, out_size: int) -> Tensor:
    # pools along last dim using slices + mean (autograd through core ops)
    L = x.size(-1)
    if out_size == L:
        return x
    if L % out_size == 0:
        k = L // out_size
        sh = x._shape[:x.ndim - 1] + [out_size, k]
        return x.reshape(sh).mean(-1)
    parts = List[Tensor]()
    for i in range(out_size):
        a, b = _adaptive_bounds(i, L, out_size)
        parts.append(x.narrow(-1, a, b - a).mean(-1, keepdim=True))
    from torch import cat
    return cat(parts, -1)

def adaptive_avg_pool1d(input: Tensor, output_size) -> Tensor:
    return _adaptive_avg_last(input, _single(output_size))

def adaptive_avg_pool2d(input: Tensor, output_size) -> Tensor:
    oh, ow = _pair(output_size)
    x = _adaptive_avg_last(input, ow)
    x = _adaptive_avg_last(x.transpose(-1, -2), oh).transpose(-1, -2)
    return x

def _adaptive_max_last(x: Tensor, out_size: int) -> Tensor:
    L = x.size(-1)
    if L % out_size == 0:
        k = L // out_size
        sh = x._shape[:x.ndim - 1] + [out_size, k]
        return x.reshape(sh).amax(-1)
    parts = List[Tensor]()
    for i in range(out_size):
        a, b = _adaptive_bounds(i, L, out_size)
        parts.append(x.narrow(-1, a, b - a).amax(-1, keepdim=True))
    from torch import cat
    return cat(parts, -1)

def adaptive_max_pool1d(input: Tensor, output_size, return_indices: bool = False) -> Tensor:
    return _adaptive_max_last(input, _single(output_size))

def adaptive_max_pool2d(input: Tensor, output_size, return_indices: bool = False) -> Tensor:
    oh, ow = _pair(output_size)
    x = _adaptive_max_last(input, ow)
    return _adaptive_max_last(x.transpose(-1, -2), oh).transpose(-1, -2)

# ---------------------------------------------------------------- activations

def relu(input: Tensor, inplace: bool = False) -> Tensor:
    return input.relu()

def relu6(input: Tensor, inplace: bool = False) -> Tensor:
    return input.clamp(0.0, 6.0)

def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    from torch import where
    if input._dt == _F32 and not (_T._GRAD_ENABLED[0] and input.requires_grad) and input.is_contiguous():
        # fused inference path = ATen elu kernel: SLEEF expm1f on full 4-lane blocks,
        # libm expm1f on the n % 4 tail (cpu_kernel_vec)
        from torch.nn._act import _elu_fwd
        out = _new(input._shape.copy(), _F32)
        _elu_fwd(Ptr[float32](input._st.ptr) + input._off, Ptr[float32](out._st.ptr),
                 input.numel(), float32(alpha))
        return out
    if input._dt == _F32 and alpha == 1.0:
        # ATen elu (training): SLEEF expm1f/expf kernels, bit-exact on aarch64
        from torch.nn._act import elu_f32
        return elu_f32(input, alpha)
    return where(input > 0.0, input, input.expm1() * alpha)

def selu(input: Tensor, inplace: bool = False) -> Tensor:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return elu(input, alpha) * scale

def celu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    from torch import where
    return where(input > 0.0, input, (input / alpha).expm1() * alpha)

def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    from torch import where
    return where(input > 0.0, input, input * negative_slope)

def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    if approximate == "tanh":
        k = _math.sqrt(2.0 / _math.pi)
        inner = (input + input * input * input * 0.044715) * k
        return input * 0.5 * (inner.tanh() + 1.0)
    elif approximate == "none":
        return input * 0.5 * ((input * _math.sqrt(0.5)).erf() + 1.0)
    raise RuntimeError(f"approximate argument must be either none or tanh.")

def silu(input: Tensor, inplace: bool = False) -> Tensor:
    return input * input.sigmoid()

def mish(input: Tensor, inplace: bool = False) -> Tensor:
    return input * softplus(input).tanh()

def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()

def tanh(input: Tensor) -> Tensor:
    return input.tanh()

def hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> Tensor:
    return input.clamp(min_val, max_val)

def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    return (input / 6.0 + 0.5).clamp(0.0, 1.0)

def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    return input * (input + 3.0).clamp(0.0, 6.0) / 6.0

def softplus(input: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    from torch import where
    bx = input * beta
    return where(bx > threshold, input, (bx.exp().log1p()) / beta)

def softsign(input: Tensor) -> Tensor:
    return input / (input.abs() + 1.0)

def logsigmoid(input: Tensor) -> Tensor:
    return -softplus(-input)

def tanhshrink(input: Tensor) -> Tensor:
    return input - input.tanh()

def threshold(input: Tensor, threshold: float, value: float, inplace: bool = False) -> Tensor:
    from torch import where, full_like
    return where(input > threshold, input, full_like(input, value))

def _softmax_dim(ndim: int) -> int:
    return 0 if ndim == 0 or ndim == 1 or ndim == 3 else 1

def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype = None) -> Tensor:
    d = _or(dim, _softmax_dim(input.ndim))
    return input.softmax(d)

def log_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype = None) -> Tensor:
    d = _or(dim, _softmax_dim(input.ndim))
    return input.log_softmax(d)

def softmin(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype = None) -> Tensor:
    return softmax(-input, dim)

def glu(input: Tensor, dim: int = -1) -> Tensor:
    n = input.size(dim) // 2
    a = input.narrow(dim, 0, n)
    b = input.narrow(dim, n, n)
    return a * b.sigmoid()

def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    denom = input.norm(p, dim, keepdim=True).clamp(eps, None).expand_as(input)
    return input / denom

# ---------------------------------------------------------------- dropout

def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, generator: Optional[Generator] = None) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    if not training or p == 0.0:
        return input
    if p == 1.0:
        return input * 0.0
    from torch import empty_like
    from torch._tensor import no_grad
    mask = empty_like(input)
    with no_grad():
        mask.bernoulli_(1.0 - p, generator)
        mask.div_(1.0 - p)
    return input * mask

def _feature_dropout(input: Tensor, p: float, training: bool, batched_dims: int, generator: Optional[Generator]) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    if not training or p == 0.0:
        return input
    from torch import empty
    from torch._tensor import no_grad
    nd = input.ndim
    shp = List[int]()
    for i in range(nd):
        shp.append(input.size(i) if i < batched_dims else 1)
    mask = empty(shp, dtype=input.dtype)
    with no_grad():
        if p == 1.0:
            mask.fill_(0.0)
        else:
            mask.bernoulli_(1.0 - p, generator)
            mask.div_(1.0 - p)
    return input * mask

def dropout1d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if input.ndim != 2 and input.ndim != 3:
        raise RuntimeError(f"dropout1d: Expected 2D or 3D input, but received a {input.ndim}D input.")
    if input.ndim == 2:
        return _feature_dropout(input.unsqueeze(0), p, training, 2, None).squeeze(0)
    return _feature_dropout(input, p, training, 2, None)

def dropout2d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if input.ndim == 3:
        # torch treats 3D input as unbatched (C, H, W) with a warning only for
        # historical reasons; it actually applies 1D channel-wise dropout.
        return _feature_dropout(input, p, training, 1, None)
    return _feature_dropout(input, p, training, 2, None)

def dropout3d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if input.ndim == 4:
        return _feature_dropout(input.unsqueeze(0), p, training, 2, None).squeeze(0)
    return _feature_dropout(input, p, training, 2, None)

def alpha_dropout(input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False) -> Tensor:
    if not training or p == 0.0:
        return input
    raise NotImplementedError("alpha_dropout in training mode")

# ---------------------------------------------------------------- normalization

def _bshape(x: Tensor, C: int) -> List[int]:
    shp = [1] * x.ndim
    shp[1] = C
    return shp

def batch_norm(input: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor], weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None, training: bool = False, momentum: float = 0.1, eps: float = 1e-05) -> Tensor:
    from torch._tensor import no_grad
    C = input.size(1)
    dims = [0] + list(range(2, input.ndim))
    shp = _bshape(input, C)
    out: Tensor
    if training:
        n = input.numel() // C
        if n <= 1:
            raise ValueError(f"Expected more than 1 value per channel when training, got input size {input.shape}")
        if (input._dt == _F32 and (weight is None or weight.__val__()._dt == _F32)
                and (bias is None or bias.__val__()._dt == _F32)
                and (running_mean is None or (running_mean.__val__()._dt == _F32 and running_mean.__val__().is_contiguous()))
                and (running_var is None or (running_var.__val__()._dt == _F32 and running_var.__val__().is_contiguous()))):
            from torch.nn._bn import batch_norm_train_f32
            return batch_norm_train_f32(input, running_mean, running_var, weight, bias, momentum, eps)
        mean = input.mean(dims, keepdim=True)
        xc = input - mean
        var = (xc * xc).mean(dims, keepdim=True)
        out = xc * (var + eps).rsqrt()
        if running_mean is not None:
            with no_grad():
                rm = running_mean.__val__()
                rm.mul_(1.0 - momentum).add_(mean.detach().reshape([C]) * momentum)
        if running_var is not None:
            with no_grad():
                rv = running_var.__val__()
                unb = var.detach().reshape([C]) * (float(n) / float(n - 1))
                rv.mul_(1.0 - momentum).add_(unb * momentum)
    else:
        if running_mean is None or running_var is None:
            raise RuntimeError("running_mean and running_var must be defined in evaluation mode")
        rm = running_mean.__val__()
        rv = running_var.__val__()
        r = (rv + eps).rsqrt()
        if (input._dt == _F32 and rm._dt == _F32 and r._dt == _F32 and input.is_contiguous() and rm.is_contiguous()
                and not (_T._GRAD_ENABLED[0] and (input.requires_grad or (weight is not None and weight.__val__().requires_grad)
                                                  or (bias is not None and bias.__val__().requires_grad)))
                and (weight is None or (weight.__val__()._dt == _F32 and weight.__val__().is_contiguous()))
                and (bias is None or (bias.__val__()._dt == _F32 and bias.__val__().is_contiguous()))):
            # fused inference path: same per-element ops and order as below
            # ((x - mean) * rsqrt(var + eps)) * weight + bias, in float32
            res = _new(input._shape.copy(), _F32)
            N0 = input.size(0)
            inner = input.numel() // (N0 * C) if N0 * C > 0 else 0
            px = Ptr[float32](input._st.ptr) + input._off
            po = Ptr[float32](res._st.ptr)
            pm = Ptr[float32](rm._st.ptr) + rm._off
            pr = Ptr[float32](r._st.ptr) + r._off
            for c in range(C):
                mc = pm[c]
                rc = pr[c]
                hw = weight is not None
                hb = bias is not None
                wc = float32(0.0)
                bc = float32(0.0)
                if weight is not None:
                    wv = weight.__val__()
                    wc = (Ptr[float32](wv._st.ptr) + wv._off)[c]
                if bias is not None:
                    bv = bias.__val__()
                    bc = (Ptr[float32](bv._st.ptr) + bv._off)[c]
                for n in range(N0):
                    base = (n * C + c) * inner
                    for i in range(inner):
                        v = (px[base + i] - mc) * rc
                        if hw:
                            v = v * wc
                        if hb:
                            v = v + bc
                        po[base + i] = v
            return res
        out = (input - rm.reshape(shp)) * r.reshape(shp)
    if weight is not None:
        out = out * weight.__val__().reshape(shp)
    if bias is not None:
        out = out + bias.__val__().reshape(shp)
    return out

def instance_norm(input: Tensor, running_mean: Optional[Tensor] = None, running_var: Optional[Tensor] = None,
                  weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, use_input_stats: bool = True,
                  momentum: float = 0.1, eps: float = 1e-05) -> Tensor:
    C = input.size(1)
    dims = list(range(2, input.ndim))
    shp = _bshape(input, C)
    mean = input.mean(dims, keepdim=True)
    xc = input - mean
    var = (xc * xc).mean(dims, keepdim=True)
    out = xc * (var + eps).rsqrt()
    if weight is not None:
        out = out * weight.__val__().reshape(shp)
    if bias is not None:
        out = out + bias.__val__().reshape(shp)
    return out

def layer_norm(input: Tensor, normalized_shape, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: float = 1e-05) -> Tensor:
    ns = List[int]()
    if isinstance(normalized_shape, int):
        ns.append(normalized_shape)
    else:
        for v in normalized_shape:
            ns.append(int(v))
    k = len(ns)
    nd = input.ndim
    for i in range(k):
        if input.size(nd - k + i) != ns[i]:
            raise RuntimeError(f"Given normalized_shape={ns}, expected input with shape [*, {', '.join([str(a) for a in ns])}], but got input of size{input.shape}")
    dims = list(range(nd - k, nd))
    mean = input.mean(dims, keepdim=True)
    xc = input - mean
    var = (xc * xc).mean(dims, keepdim=True)
    out = xc * (var + eps).rsqrt()
    if weight is not None:
        out = out * weight.__val__()
    if bias is not None:
        out = out + bias.__val__()
    return out

def rms_norm(input: Tensor, normalized_shape, weight: Optional[Tensor] = None, eps: Optional[float] = None) -> Tensor:
    ns = List[int]()
    if isinstance(normalized_shape, int):
        ns.append(normalized_shape)
    else:
        for v in normalized_shape:
            ns.append(int(v))
    nd = input.ndim
    dims = list(range(nd - len(ns), nd))
    e = _or(eps, (1.1920928955078125e-07 if input._dt == _F32 else 2.220446049250313e-16))
    out = input * ((input * input).mean(dims, keepdim=True) + e).rsqrt()
    if weight is not None:
        out = out * weight.__val__()
    return out

def group_norm(input: Tensor, num_groups: int, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: float = 1e-05) -> Tensor:
    N = input.size(0)
    C = input.size(1)
    if C % num_groups != 0:
        raise RuntimeError(f"Expected number of channels in input to be divisible by num_groups, but got input of shape {input.shape} and num_groups={num_groups}")
    x = input.reshape([N, num_groups, -1])
    mean = x.mean(-1, keepdim=True)
    xc = x - mean
    var = (xc * xc).mean(-1, keepdim=True)
    out = (xc * (var + eps).rsqrt()).reshape(input._shape)
    shp = _bshape(input, C)
    if weight is not None:
        out = out * weight.__val__().reshape(shp)
    if bias is not None:
        out = out + bias.__val__().reshape(shp)
    return out

# ---------------------------------------------------------------- losses

def _reduce(l: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return l.mean()
    elif reduction == "sum":
        return l.sum()
    elif reduction == "none":
        return l
    raise ValueError(f"{reduction} is not a valid value for reduction")

def mse_loss(input: Tensor, target: Tensor, size_average = None, reduce = None, reduction: str = "mean", weight: Optional[Tensor] = None) -> Tensor:
    d = input - target
    l = d * d
    if weight is not None:
        l = l * weight.__val__()
    return _reduce(l, reduction)

def l1_loss(input: Tensor, target: Tensor, size_average = None, reduce = None, reduction: str = "mean", weight: Optional[Tensor] = None) -> Tensor:
    l = (input - target).abs()
    if weight is not None:
        l = l * weight.__val__()
    return _reduce(l, reduction)

def smooth_l1_loss(input: Tensor, target: Tensor, size_average = None, reduce = None, reduction: str = "mean", beta: float = 1.0) -> Tensor:
    from torch import where
    d = (input - target).abs()
    if beta == 0.0:
        return _reduce(d, reduction)
    l = where(d < beta, d * d * (0.5 / beta), d - 0.5 * beta)
    return _reduce(l, reduction)

def huber_loss(input: Tensor, target: Tensor, reduction: str = "mean", delta: float = 1.0, weight: Optional[Tensor] = None) -> Tensor:
    from torch import where
    d = (input - target).abs()
    l = where(d < delta, d * d * 0.5, (d - 0.5 * delta) * delta)
    return _reduce(l, reduction)

def _gather_class(logp: Tensor, target: Tensor) -> Tensor:
    # logp: [N, C, *]; target: [N, *] long -> [N, *]
    return logp.gather(1, target.unsqueeze(1)).squeeze(1)

def nll_loss(input: Tensor, target: Tensor, weight: Optional[Tensor] = None, size_average = None, ignore_index: int = -100,
             reduce = None, reduction: str = "mean") -> Tensor:
    from torch import where, zeros_like
    x = input
    t = target
    if x.ndim == 1:
        x = x.unsqueeze(0)
        t = t.unsqueeze(0)
    C = x.size(1)
    valid = t != ignore_index
    tsafe = where(valid, t, zeros_like(t))
    picked = -_gather_class(x, tsafe)
    w: Tensor
    if weight is not None:
        w = weight.__val__().index_select(0, tsafe.reshape([-1])).reshape(tsafe._shape)
    else:
        from torch import ones_like
        w = ones_like(picked)
    w = w * valid.to(w.dtype)
    l = picked * w
    if reduction == "none":
        return l if input.ndim > 1 else l.squeeze(0)
    elif reduction == "sum":
        return l.sum()
    elif reduction == "mean":
        return l.sum() / w.sum()
    raise ValueError(f"{reduction} is not a valid value for reduction")

def cross_entropy(input: Tensor, target: Tensor, weight: Optional[Tensor] = None, size_average = None, ignore_index: int = -100,
                  reduce = None, reduction: str = "mean", label_smoothing: float = 0.0) -> Tensor:
    cdim = 0 if input.ndim == 1 else 1
    if (input.ndim == 2 and input._dt == _F32 and not target.is_floating_point() and target.ndim == 1
            and target._dt == _T._I64 and label_smoothing <= 0.0
            and (weight is None or weight.__val__()._dt == _F32) and reduction in ("mean", "sum", "none")):
        # ATen log_softmax + nll_loss kernels (bit-exact incl. backward)
        from torch.nn._ce import cross_entropy_f32
        return cross_entropy_f32(input, target, weight, ignore_index, reduction)
    logp = input.log_softmax(cdim)
    if target.is_floating_point():
        # class probabilities
        C = input.size(cdim)
        tgt = target
        if label_smoothing > 0.0:
            tgt = tgt * (1.0 - label_smoothing) + label_smoothing / float(C)
        prod = logp * tgt
        if weight is not None:
            shp = [1] * input.ndim
            shp[cdim] = C
            prod = prod * weight.__val__().reshape(shp)
        l = -prod.sum(cdim)
        if reduction == "mean":
            return l.sum() / float(input.numel() // C)
        return _reduce(l, reduction)
    nll = nll_loss(logp, target, weight, None, ignore_index, None, reduction)
    if label_smoothing <= 0.0:
        return nll
    # smoothing term
    from torch import where, zeros_like
    C = input.size(cdim)
    x = logp if input.ndim > 1 else logp.unsqueeze(0)
    t = target if input.ndim > 1 else target.unsqueeze(0)
    valid = (t != ignore_index).to(x.dtype)
    s: Tensor
    if weight is not None:
        shp = [1] * x.ndim
        shp[1] = C
        s = -(x * weight.__val__().reshape(shp)).sum(1)
    else:
        s = -x.sum(1)
    s = s * valid
    smooth: Tensor
    if reduction == "none":
        smooth = s if input.ndim > 1 else s.squeeze(0)
    elif reduction == "sum":
        smooth = s.sum()
    else:
        if weight is not None:
            tsafe = where(t != ignore_index, t, zeros_like(t))
            wsum = (weight.__val__().index_select(0, tsafe.reshape([-1])).reshape(tsafe._shape) * valid).sum()
            smooth = s.sum() / wsum
        else:
            smooth = s.sum() / valid.sum()
    return nll * (1.0 - label_smoothing) + smooth * (label_smoothing / float(C))

def binary_cross_entropy(input: Tensor, target: Tensor, weight: Optional[Tensor] = None, size_average = None, reduce = None, reduction: str = "mean") -> Tensor:
    l = -(target * input.log().clamp(-100.0, None) + (1.0 - target) * (1.0 - input).log().clamp(-100.0, None))
    if weight is not None:
        l = l * weight.__val__()
    return _reduce(l, reduction)

def binary_cross_entropy_with_logits(input: Tensor, target: Tensor, weight: Optional[Tensor] = None, size_average = None, reduce = None,
                                     reduction: str = "mean", pos_weight: Optional[Tensor] = None) -> Tensor:
    # (1 - t) * x + (1 + (pw - 1) * t) * log(1 + exp(-x)), stable form
    ls = logsigmoid(input)
    l: Tensor
    if pos_weight is not None:
        lw = (pos_weight.__val__() - 1.0) * target + 1.0
        l = (1.0 - target) * input - lw * ls
    else:
        l = (1.0 - target) * input - ls
    if weight is not None:
        l = l * weight.__val__()
    return _reduce(l, reduction)

def kl_div(input: Tensor, target: Tensor, size_average = None, reduce = None, reduction: str = "mean", log_target: bool = False) -> Tensor:
    from torch import xlogy
    l: Tensor
    if log_target:
        l = target.exp() * (target - input)
    else:
        l = xlogy(target, target) - target * input
    if reduction == "batchmean":
        return l.sum() / float(input.size(0)) if input.ndim > 0 else l.sum()
    return _reduce(l, reduction)

# ---------------------------------------------------------------- misc

def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    from torch import zeros, int64
    n = num_classes
    if n < 0:
        n = int(tensor.max().item()) + 1
    out = zeros(tensor._shape + [n], dtype=int64)
    out.scatter_(-1, tensor.unsqueeze(-1), 1)
    return out

def embedding(input: Tensor, weight: Tensor, padding_idx: Optional[int] = None, max_norm: Optional[float] = None,
              norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False) -> Tensor:
    flat = input.reshape([-1])
    out = weight.index_select(0, flat)
    if padding_idx is not None and weight.requires_grad:
        # torch: the padding row receives no gradient
        pi = padding_idx.__val__()
        if pi < 0:
            pi += weight.size(0)
        pad = (flat == pi).unsqueeze(1)
        out = out.masked_fill(pad, 0.0) + weight.detach().index_select(0, flat).masked_fill(pad.logical_not(), 0.0)
    return out.reshape(input._shape + [weight.size(1)])

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None,
                                 dropout_p: float = 0.0, is_causal: bool = False, scale: Optional[float] = None,
                                 enable_gqa: bool = False) -> Tensor:
    from torch import ones, tril, where, full_like, bool as tbool
    L = query.size(-2)
    S = key.size(-2)
    sf = 1.0 / _math.sqrt(float(query.size(-1)))
    if scale is not None:
        sf = scale.__val__()
    attn = query.matmul(key.transpose(-2, -1)) * sf
    if is_causal:
        m = ones([L, S], dtype=tbool).tril(0)
        attn = attn.masked_fill(m.logical_not(), -float("inf"))
    if attn_mask is not None:
        am = attn_mask.__val__()
        if am.dtype == tbool:
            attn = attn.masked_fill(am.logical_not(), -float("inf"))
        else:
            attn = attn + am
    attn = attn.softmax(-1)
    if dropout_p > 0.0:
        attn = dropout(attn, dropout_p, True)
    return attn.matmul(value)

def pixel_shuffle(input: Tensor, upscale_factor: int) -> Tensor:
    r = upscale_factor
    nd = input.ndim
    lead = input._shape[:nd - 3]
    C, H, W = input.size(-3), input.size(-2), input.size(-1)
    oc = C // (r * r)
    x = input.reshape(lead + [oc, r, r, H, W])
    L = len(lead)
    perm = list(range(L)) + [L, L + 3, L + 1, L + 4, L + 2]
    return x.permute(perm).reshape(lead + [oc, H * r, W * r])

def interpolate(input: Tensor, size = None, scale_factor = None, mode: str = "nearest", align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None, antialias: bool = False) -> Tensor:
    # 1-D (3-D input) nearest / linear along the last dim
    if input.ndim != 3:
        raise NotImplementedError("interpolate: only 3-D input supported")
    L = input.size(-1)
    O = 0
    sc = 0.0
    if size is not None:
        O = _single(size)
        sc = float(L) / float(O)
    else:
        f = 0.0
        if isinstance(scale_factor, List[float]):
            f = scale_factor[0]
        elif isinstance(scale_factor, Optional[float]):
            f = scale_factor.__val__()
        else:
            f = float(scale_factor)
        O = int(_math.floor(float(L) * f))
        sc = 1.0 / f if recompute_scale_factor is None or not recompute_scale_factor else float(L) / float(O)
    from torch import tensor as _tensor, int64
    if mode == "nearest":
        idx = [min(int(_math.floor(float(i) * sc)), L - 1) for i in range(O)]
        return input.index_select(-1, _tensor(idx, dtype=int64))
    elif mode == "linear":
        ac = align_corners is not None and align_corners
        i0 = List[int]()
        i1 = List[int]()
        lam = List[float]()
        for i in range(O):
            src = 0.0
            if ac:
                src = float(i) * (float(L - 1) / float(O - 1)) if O > 1 else 0.0
            else:
                src = max((float(i) + 0.5) * sc - 0.5, 0.0)
            a = int(_math.floor(src))
            a = min(a, L - 1)
            b = min(a + 1, L - 1)
            i0.append(a)
            i1.append(b)
            lam.append(src - float(a))
        x0 = input.index_select(-1, _tensor(i0, dtype=int64))
        x1 = input.index_select(-1, _tensor(i1, dtype=int64))
        w = _tensor(lam, dtype=input.dtype)
        return x0 + (x1 - x0) * w
    raise NotImplementedError("interpolate mode " + mode)
