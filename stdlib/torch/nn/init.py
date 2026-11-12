# Ported from torch 2.15.0 (BSD-3-Clause)
#
# torch.nn.init. Every initializer draws from the torch RNG (core Generator)
# in torch's exact order, so `torch.manual_seed(s)` + default layer init is
# bit-identical to torch.

import math
from torch.nn._util import _or
from torch._tensor import Tensor, no_grad
from torch._random import Generator

def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    linear_fns = ["linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        ns = _or(param, 0.01)
        return math.sqrt(2.0 / (1 + ns ** 2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

def _calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[int, int]:
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if dimensions > 2:
        for i in range(2, dimensions):
            receptive_field_size *= tensor.size(i)
    return (num_input_fmaps * receptive_field_size, num_output_fmaps * receptive_field_size)

def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    mode = mode.lower()
    if mode != "fan_in" and mode != "fan_out":
        raise ValueError(f"Mode {mode} not supported, please use one of fan_in, fan_out")
    fi, fo = _calculate_fan_in_and_fan_out(tensor)
    return fi if mode == "fan_in" else fo

def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0, generator: Optional[Generator] = None) -> Tensor:
    with no_grad():
        tensor.uniform_(a, b, generator)
    return tensor

def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, generator: Optional[Generator] = None) -> Tensor:
    with no_grad():
        tensor.normal_(mean, std, generator)
    return tensor

def constant_(tensor: Tensor, val: float) -> Tensor:
    with no_grad():
        tensor.fill_(val)
    return tensor

def ones_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 1.0)

def zeros_(tensor: Tensor) -> Tensor:
    with no_grad():
        tensor.zero_()
    return tensor

def eye_(tensor: Tensor) -> Tensor:
    if tensor.ndim != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    with no_grad():
        tensor.zero_()
        n = min(tensor.size(0), tensor.size(1))
        for i in range(n):
            tensor[i, i] = 1.0
    return tensor

def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    d = tensor.ndim
    if d < 3 or d > 5:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")
    sizes = tensor._shape
    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])
    with no_grad():
        tensor.zero_()
        for g in range(groups):
            for i in range(min_dim):
                idx = [g * out_chans_per_grp + i, i] + [sizes[k] // 2 for k in range(2, d)]
                flat = 0
                st = tensor.stride()
                for k in range(d):
                    flat += idx[k] * st[k]
                tensor.reshape([-1])[flat] = 1.0
    return tensor

def xavier_uniform_(tensor: Tensor, gain: float = 1.0, generator: Optional[Generator] = None) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a, generator)

def xavier_normal_(tensor: Tensor, gain: float = 1.0, generator: Optional[Generator] = None) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std, generator)

def kaiming_uniform_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu",
                     generator: Optional[Generator] = None) -> Tensor:
    if 0 in tensor._shape:
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(float(fan))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound, generator)

def kaiming_normal_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu",
                    generator: Optional[Generator] = None) -> Tensor:
    if 0 in tensor._shape:
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(float(fan))
    return normal_(tensor, 0.0, std, generator)

def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0,
                  generator: Optional[Generator] = None) -> Tensor:
    from torch import where, empty_like, logical_or
    with no_grad():
        p = _norm_cdf((b - mean) / std) - _norm_cdf((a - mean) / std)
        if p > 0.3:
            lo = float(tensor.new_tensor(a).item())
            hi = float(tensor.new_tensor(b).item())
            result = tensor.normal_(mean, std, generator)
            while True:
                mask = logical_or(result < lo, result > hi)
                if not bool(mask.any()):
                    break
                result = where(mask, empty_like(result).normal_(mean, std, generator), result)
            if result is not tensor:
                tensor.copy_(result)
        else:
            mode = max(a, min(mean, b))
            log_peak = -0.5 * ((mode - mean) / std) ** 2
            candidates = empty_like(tensor)
            accept_buf = empty_like(tensor)
            tensor.uniform_(a, b, generator)
            candidates.copy_(tensor)
            candidates.sub_(mean).div_(std).pow_(2).mul_(-0.5).sub_(log_peak)
            pending = accept_buf.uniform_(0.0, 1.0, generator).log().gt(candidates)
            if bool(pending.any()):
                result = tensor
                while True:
                    candidates.uniform_(a, b, generator)
                    result = where(pending, candidates, result)
                    candidates.sub_(mean).div_(std).pow_(2).mul_(-0.5).sub_(log_peak)
                    pending = where(pending, accept_buf.uniform_(0.0, 1.0, generator).log().gt(candidates), pending)
                    if not bool(pending.any()):
                        break
                tensor.copy_(result)
    return tensor

def orthogonal_(tensor: Tensor, gain: float = 1.0, generator: Optional[Generator] = None) -> Tensor:
    if tensor.ndim < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    if tensor.numel() == 0:
        return tensor
    from torch import linalg, diag
    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new_empty(rows, cols).normal_(0.0, 1.0, generator)
    if rows < cols:
        flattened = flattened.t()
    q, r = linalg.qr(flattened)
    d = diag(r, 0)
    ph = d.sign()
    q = q * ph
    if rows < cols:
        q = q.t()
    with no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

def sparse_(tensor: Tensor, sparsity: float, std: float = 0.01, generator: Optional[Generator] = None) -> Tensor:
    raise NotImplementedError("nn.init.sparse_")

# deprecated aliases
def uniform(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor: return uniform_(tensor, a, b)
def normal(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor: return normal_(tensor, mean, std)
def constant(tensor: Tensor, val: float) -> Tensor: return constant_(tensor, val)
def xavier_uniform(tensor: Tensor, gain: float = 1.0) -> Tensor: return xavier_uniform_(tensor, gain)
def xavier_normal(tensor: Tensor, gain: float = 1.0) -> Tensor: return xavier_normal_(tensor, gain)
def kaiming_uniform(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    return kaiming_uniform_(tensor, a, mode, nonlinearity)
def kaiming_normal(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    return kaiming_normal_(tensor, a, mode, nonlinearity)
