# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Layers of torch.nn. Default initialization follows torch's reset_parameters
# exactly (same initializers, same draw order).

import math
from torch.nn._util import _or
from torch._tensor import Tensor, no_grad, _DType
from torch.nn.modules.module import Module
from torch.nn.modules.container import Sequential, ModuleList, Identity
import torch.nn.functional as F
import torch.nn.init as init

def Parameter(data: Optional[Tensor] = None, requires_grad: bool = True) -> Tensor:
    """torch.nn.Parameter: a leaf Tensor sharing storage with `data`."""
    if data is None:
        from torch import empty
        t = empty([0])
        t.requires_grad = requires_grad
        return t
    d = data
    t = d.detach()
    t.requires_grad = requires_grad
    return t

def _dt(dtype: Optional[_DType]) -> _DType:
    if dtype is None:
        from torch import get_default_dtype
        return get_default_dtype()
    return dtype

def _empty(shape: List[int], dtype: Optional[_DType]) -> Tensor:
    from torch import empty
    return empty(shape, dtype=_dt(dtype))

def _P(shape: List[int], dtype: Optional[_DType]) -> Tensor:
    return Parameter(_empty(shape, dtype))

def _tuple_n(v, n: int) -> List[int]:
    if isinstance(v, int):
        return [v] * n
    else:
        out = List[int]()
        for a in v:
            out.append(int(a))
        if len(out) == 1 and n > 1:
            return out * n
        return out

def _fmt_t(v: List[int]) -> str:
    if len(v) == 1:
        return "(" + str(v[0]) + ",)"
    return "(" + ", ".join([str(a) for a in v]) + ")"

def _fbool(b: bool) -> str:
    return "True" if b else "False"

def _ffloat(x: float) -> str:
    return str(x)

# ====================================================================== Linear

class Linear(Module):
    in_features: int
    out_features: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _P([out_features, in_features], dtype)
        self.register_parameter("weight", self.weight)
        if bias:
            b = _P([out_features], dtype)
            self.bias = b
            self.register_parameter("bias", b)
        else:
            self.bias = None
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def _get_name(self) -> str:
        return "Linear"

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={_fbool(self.bias is not None)}"

class Bilinear(Module):
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = _P([out_features, in1_features, in2_features], dtype)
        self.register_parameter("weight", self.weight)
        if bias:
            b = _P([out_features], dtype)
            self.bias = b
            self.register_parameter("bias", b)
        else:
            self.bias = None
            self.register_parameter("bias", None)
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def _fwd2(self, input1: Tensor, input2: Tensor) -> Tensor:
        return F.bilinear(input1, input2, self.weight, self.bias)

    def _get_name(self) -> str:
        return "Bilinear"

# ====================================================================== Conv

class _ConvNd(Module):
    in_channels: int
    out_channels: int
    kernel_size: List[int]
    stride: List[int]
    padding: List[int]
    padding_str: str
    dilation: List[int]
    groups: int
    padding_mode: str
    transposed: bool
    output_padding: List[int]
    weight: Tensor
    bias: Optional[Tensor]
    _nd: int
    _rev_pad: List[int]

    def __init__(self):
        super().__init__()

    def _setup(self, nd: int, in_channels: int, out_channels: int, kernel_size, stride, padding, dilation, groups: int,
               bias: bool, padding_mode: str, dtype: Optional[_DType], transposed: bool, output_padding):
        self._nd = nd
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_modes = ["zeros", "reflect", "replicate", "circular"]
        if padding_mode not in valid_modes:
            raise ValueError(f"padding_mode must be one of {valid_modes}, but got padding_mode='{padding_mode}'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tuple_n(kernel_size, nd)
        self.stride = _tuple_n(stride, nd)
        self.dilation = _tuple_n(dilation, nd)
        self.groups = groups
        self.padding_mode = padding_mode
        self.transposed = transposed
        self.output_padding = _tuple_n(output_padding, nd)
        self.padding_str = ""
        if isinstance(padding, str):
            if padding != "same" and padding != "valid":
                raise ValueError(f"Invalid padding string '{padding}', should be one of {{'valid', 'same'}}")
            if padding == "same" and any(s != 1 for s in self.stride):
                raise ValueError("padding='same' is not supported for strided convolutions")
            self.padding_str = padding
            self.padding = [0] * nd
            self._rev_pad = [0] * (2 * nd)
            if padding == "same":
                for i in range(nd):
                    d = self.dilation[i]
                    k = self.kernel_size[i]
                    total = d * (k - 1)
                    left = total // 2
                    j = nd - 1 - i
                    self._rev_pad[2 * j] = left
                    self._rev_pad[2 * j + 1] = total - left
        else:
            self.padding = _tuple_n(padding, nd)
            self._rev_pad = List[int]()
            for i in range(nd - 1, -1, -1):
                self._rev_pad.append(self.padding[i])
                self._rev_pad.append(self.padding[i])
        wshape: List[int]
        if transposed:
            wshape = [in_channels, out_channels // groups] + self.kernel_size
        else:
            wshape = [out_channels, in_channels // groups] + self.kernel_size
        self.weight = _P(wshape, dtype)
        self.register_parameter("weight", self.weight)
        if bias:
            b = _P([out_channels], dtype)
            self.bias = b
            self.register_parameter("bias", b)
        else:
            self.bias = None
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={_fmt_t(self.kernel_size)}, stride={_fmt_t(self.stride)}"
        if self.padding_str:
            s += f", padding={self.padding_str}"
        elif any(p != 0 for p in self.padding):
            s += f", padding={_fmt_t(self.padding)}"
        if any(d != 1 for d in self.dilation):
            s += f", dilation={_fmt_t(self.dilation)}"
        if any(p != 0 for p in self.output_padding):
            s += f", output_padding={_fmt_t(self.output_padding)}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode}"
        return s

class Conv2d(_ConvNd):
    _f_Conv2d: int
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1,
                 groups: int = 1, bias: bool = True, padding_mode: str = "zeros", device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self._f_Conv2d = 0
        self._setup(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, dtype, False, 0)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(input, self._rev_pad, self.padding_mode)
            return F.conv2d(x, weight, bias, (self.stride[0], self.stride[1]), 0, (self.dilation[0], self.dilation[1]), self.groups)
        if self.padding_str == "same":
            return F.conv2d(input, weight, bias, (self.stride[0], self.stride[1]), "same", (self.dilation[0], self.dilation[1]), self.groups)
        return F.conv2d(input, weight, bias, (self.stride[0], self.stride[1]), (self.padding[0], self.padding[1]),
                        (self.dilation[0], self.dilation[1]), self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

    def _get_name(self) -> str:
        return "Conv2d"

class Conv1d(_ConvNd):
    _f_Conv1d: int
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1,
                 groups: int = 1, bias: bool = True, padding_mode: str = "zeros", device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self._f_Conv1d = 0
        self._setup(1, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, dtype, False, 0)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(input, self._rev_pad, self.padding_mode)
            return F.conv1d(x, weight, bias, self.stride[0], 0, self.dilation[0], self.groups)
        if self.padding_str == "same":
            return F.conv1d(input, weight, bias, self.stride[0], "same", self.dilation[0], self.groups)
        return F.conv1d(input, weight, bias, self.stride[0], self.padding[0], self.dilation[0], self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

    def _get_name(self) -> str:
        return "Conv1d"

def _out_pad_list(output_padding, nd: int) -> List[int]:
    return _tuple_n(output_padding, nd)

class ConvTranspose2d(_ConvNd):
    _ct2: int

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, output_padding = 0,
                 groups: int = 1, bias: bool = True, dilation = 1, padding_mode: str = "zeros", device = None,
                 dtype: Optional[_DType] = None):
        super().__init__()
        self._ct2 = 0
        if padding_mode != "zeros":
            raise ValueError(f'Only "zeros" padding mode is supported for ConvTranspose2d')
        self._setup(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, dtype, True, output_padding)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv_transpose2d(input, self.weight, self.bias, (self.stride[0], self.stride[1]), (self.padding[0], self.padding[1]),
                                  (self.output_padding[0], self.output_padding[1]), self.groups, (self.dilation[0], self.dilation[1]))

    def _get_name(self) -> str:
        return "ConvTranspose2d"

class ConvTranspose1d(_ConvNd):
    _ct1: int

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, output_padding = 0,
                 groups: int = 1, bias: bool = True, dilation = 1, padding_mode: str = "zeros", device = None,
                 dtype: Optional[_DType] = None):
        super().__init__()
        self._ct1 = 0
        if padding_mode != "zeros":
            raise ValueError(f'Only "zeros" padding mode is supported for ConvTranspose1d')
        self._setup(1, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, dtype, True, output_padding)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv_transpose1d(input, self.weight, self.bias, self.stride[0], self.padding[0], self.output_padding[0],
                                  self.groups, self.dilation[0])

    def _get_name(self) -> str:
        return "ConvTranspose1d"

# ====================================================================== Normalization

class _NormBase(Module):
    num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool
    weight: Optional[Tensor]
    bias: Optional[Tensor]
    running_mean: Optional[Tensor]
    running_var: Optional[Tensor]
    num_batches_tracked: Optional[Tensor]

    def __init__(self):
        super().__init__()

    def _setup(self, num_features: int, eps: float, momentum: Optional[float], affine: bool, track_running_stats: bool, dtype: Optional[_DType]):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            w = _P([num_features], dtype)
            b = _P([num_features], dtype)
            self.weight = w
            self.bias = b
            self.register_parameter("weight", w)
            self.register_parameter("bias", b)
        else:
            self.weight = None
            self.bias = None
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if track_running_stats:
            from torch import zeros, ones, tensor, int64
            rm = zeros([num_features], dtype=_dt(dtype))
            rv = ones([num_features], dtype=_dt(dtype))
            nbt = tensor(0, dtype=int64)
            self.running_mean = rm
            self.running_var = rv
            self.num_batches_tracked = nbt
            self.register_buffer("running_mean", rm)
            self.register_buffer("running_var", rv)
            self.register_buffer("num_batches_tracked", nbt)
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            with no_grad():
                self.running_mean.zero_()
                self.running_var.fill_(1.0)
                self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input: Tensor):
        pass

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        eaf = _or(self.momentum, 0.0)
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                with no_grad():
                    self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    eaf = 1.0 / float(self.num_batches_tracked.item())
        bn_training = True
        if not self.training:
            bn_training = self.running_mean is None and self.running_var is None
        rm: Optional[Tensor] = None
        rv: Optional[Tensor] = None
        if not self.training or self.track_running_stats:
            rm = self.running_mean
            rv = self.running_var
        return F.batch_norm(input, rm, rv, self.weight, self.bias, bn_training, eaf, self.eps)

    def extra_repr(self) -> str:
        m = "None" if self.momentum is None else str(self.momentum)
        return f"{self.num_features}, eps={self.eps}, momentum={m}, affine={_fbool(self.affine)}, track_running_stats={_fbool(self.track_running_stats)}"

class BatchNorm1d(_NormBase):
    _f_BatchNorm1d: int
    def __init__(self, num_features: int, eps: float = 1e-05, momentum: Optional[float] = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self._f_BatchNorm1d = 0
        self._setup(num_features, eps, momentum, affine, track_running_stats, dtype)

    def _check_input_dim(self, input: Tensor):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")

    def _get_name(self) -> str:
        return "BatchNorm1d"

class BatchNorm2d(_NormBase):
    _f_BatchNorm2d: int
    def __init__(self, num_features: int, eps: float = 1e-05, momentum: Optional[float] = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self._f_BatchNorm2d = 0
        self._setup(num_features, eps, momentum, affine, track_running_stats, dtype)

    def _check_input_dim(self, input: Tensor):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")

    def _get_name(self) -> str:
        return "BatchNorm2d"

class BatchNorm3d(_NormBase):
    _f_BatchNorm3d: int
    def __init__(self, num_features: int, eps: float = 1e-05, momentum: Optional[float] = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self._f_BatchNorm3d = 0
        self._setup(num_features, eps, momentum, affine, track_running_stats, dtype)

    def _check_input_dim(self, input: Tensor):
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")

    def _get_name(self) -> str:
        return "BatchNorm3d"

class _InstanceNorm(_NormBase):
    _f__InstanceNorm: int

    def __init__(self):
        super().__init__()
        self._f__InstanceNorm = 0
    def _get_no_batch_dim(self) -> int:
        return 2

    def forward(self, input: Tensor) -> Tensor:
        nb = self._get_no_batch_dim()
        if input.dim() == nb:
            return self._apply_in(input.unsqueeze(0)).squeeze(0)
        return self._apply_in(input)

    def _apply_in(self, input: Tensor) -> Tensor:
        return F.instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                               self.training or not self.track_running_stats, _or(self.momentum, 0.1), self.eps)

class InstanceNorm1d(_InstanceNorm):
    _f_InstanceNorm1d: int
    def __init__(self, num_features: int, eps: float = 1e-05, momentum: Optional[float] = 0.1, affine: bool = False,
                 track_running_stats: bool = False, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self._f_InstanceNorm1d = 0
        self._setup(num_features, eps, momentum, affine, track_running_stats, dtype)

    def _get_name(self) -> str:
        return "InstanceNorm1d"

class InstanceNorm2d(_InstanceNorm):
    _f_InstanceNorm2d: int
    def __init__(self, num_features: int, eps: float = 1e-05, momentum: Optional[float] = 0.1, affine: bool = False,
                 track_running_stats: bool = False, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self._f_InstanceNorm2d = 0
        self._setup(num_features, eps, momentum, affine, track_running_stats, dtype)

    def _get_no_batch_dim(self) -> int:
        return 3

    def _get_name(self) -> str:
        return "InstanceNorm2d"

class LayerNorm(Module):
    normalized_shape: List[int]
    eps: float
    elementwise_affine: bool
    weight: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(self, normalized_shape, eps: float = 1e-05, elementwise_affine: bool = True, bias: bool = True,
                 device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self.normalized_shape = _tuple_n(normalized_shape, 1)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = None
        self.bias = None
        if elementwise_affine:
            w = _P(self.normalized_shape, dtype)
            self.weight = w
            self.register_parameter("weight", w)
            if bias:
                b = _P(self.normalized_shape, dtype)
                self.bias = b
                self.register_parameter("bias", b)
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def _get_name(self) -> str:
        return "LayerNorm"

    def extra_repr(self) -> str:
        return f"{_fmt_t(self.normalized_shape)}, eps={self.eps}, elementwise_affine={_fbool(self.elementwise_affine)}"

class RMSNorm(Module):
    normalized_shape: List[int]
    eps: Optional[float]
    elementwise_affine: bool
    weight: Optional[Tensor]

    def __init__(self, normalized_shape, eps: Optional[float] = None, elementwise_affine: bool = True, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self.normalized_shape = _tuple_n(normalized_shape, 1)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = None
        if elementwise_affine:
            w = _P(self.normalized_shape, dtype)
            init.ones_(w)
            self.weight = w
            self.register_parameter("weight", w)
        else:
            self.register_parameter("weight", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.rms_norm(input, self.normalized_shape, self.weight, self.eps)

    def _get_name(self) -> str:
        return "RMSNorm"

class GroupNorm(Module):
    num_groups: int
    num_channels: int
    eps: float
    affine: bool
    weight: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-05, affine: bool = True, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.weight = None
        self.bias = None
        if affine:
            w = _P([num_channels], dtype)
            b = _P([num_channels], dtype)
            init.ones_(w)
            init.zeros_(b)
            self.weight = w
            self.bias = b
            self.register_parameter("weight", w)
            self.register_parameter("bias", b)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def _get_name(self) -> str:
        return "GroupNorm"

    def extra_repr(self) -> str:
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={_fbool(self.affine)}"

# ====================================================================== Dropout

class _DropoutNd(Module):
    p: float
    inplace: bool

    def __init__(self):
        super().__init__()

    def _setup(self, p: float, inplace: bool):
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={_fbool(self.inplace)}"

class Dropout(_DropoutNd):
    _f_Dropout: int
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self._f_Dropout = 0
        self._setup(p, inplace)

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)

    def _get_name(self) -> str:
        return "Dropout"

class Dropout1d(_DropoutNd):
    _f_Dropout1d: int
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self._f_Dropout1d = 0
        self._setup(p, inplace)

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout1d(input, self.p, self.training, self.inplace)

    def _get_name(self) -> str:
        return "Dropout1d"

class Dropout2d(_DropoutNd):
    _f_Dropout2d: int
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self._f_Dropout2d = 0
        self._setup(p, inplace)

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout2d(input, self.p, self.training, self.inplace)

    def _get_name(self) -> str:
        return "Dropout2d"

class Dropout3d(_DropoutNd):
    _f_Dropout3d: int
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self._f_Dropout3d = 0
        self._setup(p, inplace)

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout3d(input, self.p, self.training, self.inplace)

    def _get_name(self) -> str:
        return "Dropout3d"

class AlphaDropout(_DropoutNd):
    _f_AlphaDropout: int
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self._f_AlphaDropout = 0
        self._setup(p, inplace)

    def forward(self, input: Tensor) -> Tensor:
        return F.alpha_dropout(input, self.p, self.training)

    def _get_name(self) -> str:
        return "AlphaDropout"

# ====================================================================== Pooling

class MaxPool2d(Module):
    kernel_size: List[int]
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    ceil_mode: bool
    return_indices: bool

    def __init__(self, kernel_size, stride = None, padding = 0, dilation = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = _tuple_n(kernel_size, 2)
        self.stride = self.kernel_size if stride is None else _tuple_n(stride, 2)
        self.padding = _tuple_n(padding, 2)
        self.dilation = _tuple_n(dilation, 2)
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool2d(input, (self.kernel_size[0], self.kernel_size[1]), (self.stride[0], self.stride[1]),
                            (self.padding[0], self.padding[1]), (self.dilation[0], self.dilation[1]), self.ceil_mode)

    def _get_name(self) -> str:
        return "MaxPool2d"

class MaxPool1d(Module):
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    ceil_mode: bool
    return_indices: bool

    def __init__(self, kernel_size, stride = None, padding = 0, dilation = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = _tuple_n(kernel_size, 1)[0]
        self.stride = self.kernel_size if stride is None else _tuple_n(stride, 1)[0]
        self.padding = _tuple_n(padding, 1)[0]
        self.dilation = _tuple_n(dilation, 1)[0]
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool1d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)

    def _get_name(self) -> str:
        return "MaxPool1d"

class AvgPool2d(Module):
    kernel_size: List[int]
    stride: List[int]
    padding: List[int]
    ceil_mode: bool
    count_include_pad: bool
    divisor_override: Optional[int]

    def __init__(self, kernel_size, stride = None, padding = 0, ceil_mode: bool = False, count_include_pad: bool = True,
                 divisor_override: Optional[int] = None):
        super().__init__()
        self.kernel_size = _tuple_n(kernel_size, 2)
        self.stride = self.kernel_size if stride is None else _tuple_n(stride, 2)
        self.padding = _tuple_n(padding, 2)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool2d(input, (self.kernel_size[0], self.kernel_size[1]), (self.stride[0], self.stride[1]),
                            (self.padding[0], self.padding[1]), self.ceil_mode, self.count_include_pad, self.divisor_override)

    def _get_name(self) -> str:
        return "AvgPool2d"

class AvgPool1d(Module):
    kernel_size: int
    stride: int
    padding: int
    ceil_mode: bool
    count_include_pad: bool

    def __init__(self, kernel_size, stride = None, padding = 0, ceil_mode: bool = False, count_include_pad: bool = True):
        super().__init__()
        self.kernel_size = _tuple_n(kernel_size, 1)[0]
        self.stride = self.kernel_size if stride is None else _tuple_n(stride, 1)[0]
        self.padding = _tuple_n(padding, 1)[0]
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool1d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)

    def _get_name(self) -> str:
        return "AvgPool1d"

class AdaptiveAvgPool1d(Module):
    output_size: int

    def __init__(self, output_size):
        super().__init__()
        self.output_size = _tuple_n(output_size, 1)[0]

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool1d(input, self.output_size)

    def _get_name(self) -> str:
        return "AdaptiveAvgPool1d"

class AdaptiveAvgPool2d(Module):
    output_size: List[int]

    def __init__(self, output_size):
        super().__init__()
        self.output_size = _tuple_n(output_size, 2)

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(input, (self.output_size[0], self.output_size[1]))

    def _get_name(self) -> str:
        return "AdaptiveAvgPool2d"

class AdaptiveMaxPool1d(Module):
    output_size: int

    def __init__(self, output_size, return_indices: bool = False):
        super().__init__()
        self.output_size = _tuple_n(output_size, 1)[0]

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_max_pool1d(input, self.output_size)

    def _get_name(self) -> str:
        return "AdaptiveMaxPool1d"

class AdaptiveMaxPool2d(Module):
    output_size: List[int]

    def __init__(self, output_size, return_indices: bool = False):
        super().__init__()
        self.output_size = _tuple_n(output_size, 2)

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_max_pool2d(input, (self.output_size[0], self.output_size[1]))

    def _get_name(self) -> str:
        return "AdaptiveMaxPool2d"

# ====================================================================== Padding

class _PadNd(Module):
    padding: List[int]
    mode: str
    value: float

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, self.mode, self.value)

    def extra_repr(self) -> str:
        return _fmt_t(self.padding)

def _pad_list(padding, n: int) -> List[int]:
    if isinstance(padding, int):
        return [padding] * (2 * n)
    out = List[int]()
    for p in padding:
        out.append(int(p))
    return out

class ZeroPad1d(_PadNd):
    _f_ZeroPad1d: int
    def __init__(self, padding):
        super().__init__()
        self._f_ZeroPad1d = 0
        self.padding = _pad_list(padding, 1)
        self.mode = "constant"
        self.value = 0.0
    def _get_name(self) -> str: return "ZeroPad1d"

class ZeroPad2d(_PadNd):
    _f_ZeroPad2d: int
    def __init__(self, padding):
        super().__init__()
        self._f_ZeroPad2d = 0
        self.padding = _pad_list(padding, 2)
        self.mode = "constant"
        self.value = 0.0
    def _get_name(self) -> str: return "ZeroPad2d"

class ConstantPad1d(_PadNd):
    _f_ConstantPad1d: int
    def __init__(self, padding, value: float):
        super().__init__()
        self._f_ConstantPad1d = 0
        self.padding = _pad_list(padding, 1)
        self.mode = "constant"
        self.value = value
    def _get_name(self) -> str: return "ConstantPad1d"

class ConstantPad2d(_PadNd):
    _f_ConstantPad2d: int
    def __init__(self, padding, value: float):
        super().__init__()
        self._f_ConstantPad2d = 0
        self.padding = _pad_list(padding, 2)
        self.mode = "constant"
        self.value = value
    def _get_name(self) -> str: return "ConstantPad2d"

class ReflectionPad1d(_PadNd):
    _f_ReflectionPad1d: int
    def __init__(self, padding):
        super().__init__()
        self._f_ReflectionPad1d = 0
        self.padding = _pad_list(padding, 1)
        self.mode = "reflect"
        self.value = 0.0
    def _get_name(self) -> str: return "ReflectionPad1d"

class ReflectionPad2d(_PadNd):
    _f_ReflectionPad2d: int
    def __init__(self, padding):
        super().__init__()
        self._f_ReflectionPad2d = 0
        self.padding = _pad_list(padding, 2)
        self.mode = "reflect"
        self.value = 0.0
    def _get_name(self) -> str: return "ReflectionPad2d"

class ReplicationPad1d(_PadNd):
    _f_ReplicationPad1d: int
    def __init__(self, padding):
        super().__init__()
        self._f_ReplicationPad1d = 0
        self.padding = _pad_list(padding, 1)
        self.mode = "replicate"
        self.value = 0.0
    def _get_name(self) -> str: return "ReplicationPad1d"

class ReplicationPad2d(_PadNd):
    _f_ReplicationPad2d: int
    def __init__(self, padding):
        super().__init__()
        self._f_ReplicationPad2d = 0
        self.padding = _pad_list(padding, 2)
        self.mode = "replicate"
        self.value = 0.0
    def _get_name(self) -> str: return "ReplicationPad2d"

class CircularPad1d(_PadNd):
    _f_CircularPad1d: int
    def __init__(self, padding):
        super().__init__()
        self._f_CircularPad1d = 0
        self.padding = _pad_list(padding, 1)
        self.mode = "circular"
        self.value = 0.0
    def _get_name(self) -> str: return "CircularPad1d"

class CircularPad2d(_PadNd):
    _f_CircularPad2d: int
    def __init__(self, padding):
        super().__init__()
        self._f_CircularPad2d = 0
        self.padding = _pad_list(padding, 2)
        self.mode = "circular"
        self.value = 0.0
    def _get_name(self) -> str: return "CircularPad2d"

# ====================================================================== Activations

class ReLU(Module):
    inplace: bool
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    def forward(self, input: Tensor) -> Tensor: return F.relu(input)
    def _get_name(self) -> str: return "ReLU"

class ReLU6(Module):
    inplace: bool
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    def forward(self, input: Tensor) -> Tensor: return F.relu6(input)
    def _get_name(self) -> str: return "ReLU6"

class ELU(Module):
    alpha: float
    inplace: bool
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
    def forward(self, input: Tensor) -> Tensor: return F.elu(input, self.alpha)
    def _get_name(self) -> str: return "ELU"
    def extra_repr(self) -> str: return f"alpha={self.alpha}"

class SELU(Module):
    inplace: bool
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    def forward(self, input: Tensor) -> Tensor: return F.selu(input)
    def _get_name(self) -> str: return "SELU"

class CELU(Module):
    alpha: float
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
    def forward(self, input: Tensor) -> Tensor: return F.celu(input, self.alpha)
    def _get_name(self) -> str: return "CELU"

class LeakyReLU(Module):
    negative_slope: float
    inplace: bool
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
    def forward(self, input: Tensor) -> Tensor: return F.leaky_relu(input, self.negative_slope)
    def _get_name(self) -> str: return "LeakyReLU"
    def extra_repr(self) -> str: return f"negative_slope={self.negative_slope}"

class PReLU(Module):
    num_parameters: int
    weight: Tensor
    def __init__(self, num_parameters: int = 1, init: float = 0.25, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = _P([num_parameters], dtype)
        with no_grad():
            self.weight.fill_(init)
        self.register_parameter("weight", self.weight)
    def forward(self, input: Tensor) -> Tensor:
        from torch import where
        w = self.weight
        if self.num_parameters > 1 and input.ndim >= 2:
            shp = [1] * input.ndim
            shp[1] = self.num_parameters
            w = w.reshape(shp)
        return where(input > 0.0, input, input * w)
    def _get_name(self) -> str: return "PReLU"

class GELU(Module):
    approximate: str
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
    def forward(self, input: Tensor) -> Tensor: return F.gelu(input, self.approximate)
    def _get_name(self) -> str: return "GELU"
    def extra_repr(self) -> str: return f"approximate='{self.approximate}'"

class SiLU(Module):
    inplace: bool
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    def forward(self, input: Tensor) -> Tensor: return F.silu(input)
    def _get_name(self) -> str: return "SiLU"

class Mish(Module):
    inplace: bool
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    def forward(self, input: Tensor) -> Tensor: return F.mish(input)
    def _get_name(self) -> str: return "Mish"

class Tanh(Module):
    _t: int
    def __init__(self):
        super().__init__()
        self._t = 0
    def forward(self, input: Tensor) -> Tensor: return input.tanh()
    def _get_name(self) -> str: return "Tanh"

class Sigmoid(Module):
    _t: int
    def __init__(self):
        super().__init__()
        self._t = 0
    def forward(self, input: Tensor) -> Tensor: return input.sigmoid()
    def _get_name(self) -> str: return "Sigmoid"

class Hardswish(Module):
    inplace: bool
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    def forward(self, input: Tensor) -> Tensor: return F.hardswish(input)
    def _get_name(self) -> str: return "Hardswish"

class Hardsigmoid(Module):
    inplace: bool
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    def forward(self, input: Tensor) -> Tensor: return F.hardsigmoid(input)
    def _get_name(self) -> str: return "Hardsigmoid"

class Hardtanh(Module):
    min_val: float
    max_val: float
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    def forward(self, input: Tensor) -> Tensor: return F.hardtanh(input, self.min_val, self.max_val)
    def _get_name(self) -> str: return "Hardtanh"

class Softplus(Module):
    beta: float
    threshold: float
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
    def forward(self, input: Tensor) -> Tensor: return F.softplus(input, self.beta, self.threshold)
    def _get_name(self) -> str: return "Softplus"

class Softsign(Module):
    _t: int
    def __init__(self):
        super().__init__()
        self._t = 0
    def forward(self, input: Tensor) -> Tensor: return F.softsign(input)
    def _get_name(self) -> str: return "Softsign"

class LogSigmoid(Module):
    _t: int
    def __init__(self):
        super().__init__()
        self._t = 0
    def forward(self, input: Tensor) -> Tensor: return F.logsigmoid(input)
    def _get_name(self) -> str: return "LogSigmoid"

class Tanhshrink(Module):
    _t: int
    def __init__(self):
        super().__init__()
        self._t = 0
    def forward(self, input: Tensor) -> Tensor: return F.tanhshrink(input)
    def _get_name(self) -> str: return "Tanhshrink"

class Threshold(Module):
    threshold: float
    value: float
    def __init__(self, threshold: float, value: float, inplace: bool = False):
        super().__init__()
        self.threshold = threshold
        self.value = value
    def forward(self, input: Tensor) -> Tensor: return F.threshold(input, self.threshold, self.value)
    def _get_name(self) -> str: return "Threshold"

class GLU(Module):
    dim: int
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    def forward(self, input: Tensor) -> Tensor: return F.glu(input, self.dim)
    def _get_name(self) -> str: return "GLU"

class Softmax(Module):
    dim: Optional[int]
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
    def forward(self, input: Tensor) -> Tensor: return F.softmax(input, self.dim)
    def _get_name(self) -> str: return "Softmax"
    def extra_repr(self) -> str: return f"dim={self.dim}"

class LogSoftmax(Module):
    dim: Optional[int]
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
    def forward(self, input: Tensor) -> Tensor: return F.log_softmax(input, self.dim)
    def _get_name(self) -> str: return "LogSoftmax"
    def extra_repr(self) -> str: return f"dim={self.dim}"

class Softmin(Module):
    dim: Optional[int]
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
    def forward(self, input: Tensor) -> Tensor: return F.softmin(input, self.dim)
    def _get_name(self) -> str: return "Softmin"

# ====================================================================== Shape

class Flatten(Module):
    start_dim: int
    end_dim: int
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    def forward(self, input: Tensor) -> Tensor: return input.flatten(self.start_dim, self.end_dim)
    def _get_name(self) -> str: return "Flatten"
    def extra_repr(self) -> str: return f"start_dim={self.start_dim}, end_dim={self.end_dim}"

class Unflatten(Module):
    dim: int
    unflattened_size: List[int]
    def __init__(self, dim: int, unflattened_size):
        super().__init__()
        self.dim = dim
        self.unflattened_size = _tuple_n(unflattened_size, 1)
    def forward(self, input: Tensor) -> Tensor: return input.unflatten(self.dim, self.unflattened_size)
    def _get_name(self) -> str: return "Unflatten"

class PixelShuffle(Module):
    upscale_factor: int
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor
    def forward(self, input: Tensor) -> Tensor: return F.pixel_shuffle(input, self.upscale_factor)
    def _get_name(self) -> str: return "PixelShuffle"

class Upsample(Module):
    size: Optional[int]
    scale_factor: Optional[float]
    mode: str
    align_corners: Optional[bool]
    def __init__(self, size: Optional[int] = None, scale_factor: Optional[float] = None, mode: str = "nearest", align_corners: Optional[bool] = None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, input: Tensor) -> Tensor:
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
    def _get_name(self) -> str: return "Upsample"

# ====================================================================== Embedding

class Embedding(Module):
    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    weight: Tensor

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, max_norm: Optional[float] = None,
                 norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False, _weight: Optional[Tensor] = None,
                 _freeze: bool = False, device = None, dtype: Optional[_DType] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        pi: Optional[int] = None
        if padding_idx is not None:
            p = padding_idx
            if p < 0:
                p += num_embeddings
            pi = p
        self.padding_idx = pi
        if _weight is not None:
            self.weight = Parameter(_weight, not _freeze)
            self.register_parameter("weight", self.weight)
        else:
            self.weight = Parameter(_empty([num_embeddings, embedding_dim], dtype), not _freeze)
            self.register_parameter("weight", self.weight)
            init.normal_(self.weight)
            if self.padding_idx is not None:
                with no_grad():
                    self.weight[self.padding_idx.__val__()].fill_(0.0)

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.weight, self.padding_idx)

    def _get_name(self) -> str:
        return "Embedding"

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"

# ====================================================================== Losses

class _Loss(Module):
    reduction: str

    def __init__(self):
        super().__init__()

    def _fwd2(self, input: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError("loss")

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self._fwd2(input, target)

    def __call__(self, input: Tensor) -> Tensor:
        raise TypeError("loss modules take (input, target)")

class MSELoss(_Loss):
    _f_MSELoss: int
    def __init__(self, size_average = None, reduce = None, reduction: str = "mean"):
        super().__init__()
        self._f_MSELoss = 0
        self.reduction = reduction
    def _fwd2(self, input: Tensor, target: Tensor) -> Tensor: return F.mse_loss(input, target, reduction=self.reduction)
    def forward(self, input: Tensor, target: Tensor) -> Tensor: return self._fwd2(input, target)
    def _get_name(self) -> str: return "MSELoss"

class L1Loss(_Loss):
    _f_L1Loss: int
    def __init__(self, size_average = None, reduce = None, reduction: str = "mean"):
        super().__init__()
        self._f_L1Loss = 0
        self.reduction = reduction
    def _fwd2(self, input: Tensor, target: Tensor) -> Tensor: return F.l1_loss(input, target, reduction=self.reduction)
    def forward(self, input: Tensor, target: Tensor) -> Tensor: return self._fwd2(input, target)
    def _get_name(self) -> str: return "L1Loss"

class SmoothL1Loss(_Loss):
    beta: float
    def __init__(self, size_average = None, reduce = None, reduction: str = "mean", beta: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.beta = beta
    def _fwd2(self, input: Tensor, target: Tensor) -> Tensor: return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)
    def forward(self, input: Tensor, target: Tensor) -> Tensor: return self._fwd2(input, target)
    def _get_name(self) -> str: return "SmoothL1Loss"

class HuberLoss(_Loss):
    delta: float
    def __init__(self, reduction: str = "mean", delta: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.delta = delta
    def _fwd2(self, input: Tensor, target: Tensor) -> Tensor: return F.huber_loss(input, target, self.reduction, self.delta)
    def forward(self, input: Tensor, target: Tensor) -> Tensor: return self._fwd2(input, target)
    def _get_name(self) -> str: return "HuberLoss"

class NLLLoss(_Loss):
    weight: Optional[Tensor]
    ignore_index: int
    def __init__(self, weight: Optional[Tensor] = None, size_average = None, ignore_index: int = -100, reduce = None, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer("weight", weight)
    def _fwd2(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input, target, self.weight, None, self.ignore_index, None, self.reduction)
    def forward(self, input: Tensor, target: Tensor) -> Tensor: return self._fwd2(input, target)
    def _get_name(self) -> str: return "NLLLoss"

class CrossEntropyLoss(_Loss):
    weight: Optional[Tensor]
    ignore_index: int
    label_smoothing: float
    def __init__(self, weight: Optional[Tensor] = None, size_average = None, ignore_index: int = -100, reduce = None,
                 reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
    def _fwd2(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, self.weight, None, self.ignore_index, None, self.reduction, self.label_smoothing)
    def forward(self, input: Tensor, target: Tensor) -> Tensor: return self._fwd2(input, target)
    def _get_name(self) -> str: return "CrossEntropyLoss"

class BCELoss(_Loss):
    weight: Optional[Tensor]
    def __init__(self, weight: Optional[Tensor] = None, size_average = None, reduce = None, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
    def _fwd2(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, self.weight, reduction=self.reduction)
    def forward(self, input: Tensor, target: Tensor) -> Tensor: return self._fwd2(input, target)
    def _get_name(self) -> str: return "BCELoss"

class BCEWithLogitsLoss(_Loss):
    weight: Optional[Tensor]
    pos_weight: Optional[Tensor]
    def __init__(self, weight: Optional[Tensor] = None, size_average = None, reduce = None, reduction: str = "mean",
                 pos_weight: Optional[Tensor] = None):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
    def _fwd2(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, self.weight, reduction=self.reduction, pos_weight=self.pos_weight)
    def forward(self, input: Tensor, target: Tensor) -> Tensor: return self._fwd2(input, target)
    def _get_name(self) -> str: return "BCEWithLogitsLoss"

class KLDivLoss(_Loss):
    log_target: bool
    def __init__(self, size_average = None, reduce = None, reduction: str = "mean", log_target: bool = False):
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target
    def _fwd2(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)
    def forward(self, input: Tensor, target: Tensor) -> Tensor: return self._fwd2(input, target)
    def _get_name(self) -> str: return "KLDivLoss"
