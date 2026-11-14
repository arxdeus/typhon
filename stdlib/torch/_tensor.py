# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Native Typhon implementation of torch.Tensor.
#
# Design: one static `Tensor` class. dtype is a RUNTIME tag (see _DType) and
# shape/strides are runtime List[int], exactly like torch, so that modules can
# take and return a single static type independent of rank. Element kernels
# are generic functions dispatched per dtype through `_dispatch`, which passes
# a zero value of the concrete element type (bool, u8, i32, int, float32,
# float). Reverse-mode autograd uses `Node` objects (see _autograd section).

import internal.static as static
import numpy as np
import numpy.util as _U
import math as _math
from torch._random import Generator, default_generator

# ---------------------------------------------------------------- dtypes

_BOOL = 0
_U8 = 1
_I32 = 2
_I64 = 3
_F32 = 4
_F64 = 5
_I16 = 6
_I8 = 7
_C64 = 8
_C128 = 9

@tuple
class _DType:
    code: int

    def __repr__(self) -> str:
        return "torch." + self._name()

    def __str__(self) -> str:
        return self.__repr__()

    def _name(self) -> str:
        c = self.code
        if c == _BOOL: return "bool"
        if c == _U8: return "uint8"
        if c == _I32: return "int32"
        if c == _I64: return "int64"
        if c == _F32: return "float32"
        if c == _F64: return "float64"
        if c == _I16: return "int16"
        if c == _C64: return "complex64"
        if c == _C128: return "complex128"
        return "int8"

    def __eq__(self, o: _DType) -> bool:
        return self.code == o.code

    def __ne__(self, o: _DType) -> bool:
        return self.code != o.code

    def __hash__(self) -> int:
        return self.code

    @property
    def is_floating_point(self) -> bool:
        return self.code == _F32 or self.code == _F64

    @property
    def is_complex(self) -> bool:
        return self.code == _C64 or self.code == _C128

    @property
    def is_signed(self) -> bool:
        return self.code != _BOOL and self.code != _U8

    @property
    def itemsize(self) -> int:
        return _itemsize(self.code)

def _itemsize(c: int) -> int:
    if c == _BOOL or c == _U8 or c == _I8: return 1
    if c == _I16: return 2
    if c == _I32 or c == _F32: return 4
    if c == _C128: return 16
    return 8

def _is_float(c: int) -> bool:
    return c == _F32 or c == _F64

def _is_complex(c: int) -> bool:
    return c == _C64 or c == _C128

def _category(c: int) -> int:
    if c == _BOOL: return 0
    if _is_float(c): return 2
    if _is_complex(c): return 3
    return 1

# standard promotion table between two dtype codes
def _promote2(a: int, b: int) -> int:
    if a == b: return a
    ca = _category(a)
    cb = _category(b)
    if ca != cb:
        if ca == 3 and b == _F64: return _C128
        if cb == 3 and a == _F64: return _C128
        return a if ca > cb else b
    if ca == 2:
        return _F64 if (a == _F64 or b == _F64) else _F32
    if ca == 3:
        return _C128 if (a == _C128 or b == _C128) else _C64
    if ca == 1:
        # integer lattice: u8 with signed -> next signed big enough
        def rank(x: int) -> int:
            if x == _I8: return 1
            if x == _U8: return 2
            if x == _I16: return 3
            if x == _I32: return 4
            return 5
        ra = rank(a)
        rb = rank(b)
        if (a == _U8 and b == _I8) or (a == _I8 and b == _U8):
            return _I16
        return a if ra > rb else b
    return _BOOL

def _dtype_of_type(T: type) -> int:
    if T is bool: return _BOOL
    elif T is u8: return _U8
    elif T is i8: return _I8
    elif T is i16: return _I16
    elif T is i32: return _I32
    elif T is int: return _I64
    elif T is float32: return _F32
    elif T is float: return _F64
    else:
        compile_error("unsupported element type for torch.Tensor")

# Calls f(zero_of_element_type, *args) for runtime dtype code `c`.
def _dispatch(c: int, f, *args):
    if c == _C64 or c == _C128:
        raise NotImplementedError("torch: operation not supported for complex tensors")
    if c == _F32: return f(float32(0.0), *args)
    elif c == _F64: return f(0.0, *args)
    elif c == _I64: return f(0, *args)
    elif c == _I32: return f(i32(0), *args)
    elif c == _BOOL: return f(False, *args)
    elif c == _U8: return f(u8(0), *args)
    elif c == _I16: return f(i16(0), *args)
    else: return f(i8(0), *args)

def _dispatch_float(c: int, f, *args):
    if c == _F64: return f(0.0, *args)
    else: return f(float32(0.0), *args)

# value conversion with torch/C++ semantics (float -> int truncates)
def _cv(v, T: type):
    if T is bool:
        return v != type(v)()
    elif isinstance(v, bool):
        return T(1) if v else T(0)
    elif (T is int or T is i32 or T is u8 or T is i16 or T is i8) and (isinstance(v, float) or isinstance(v, float32)):
        return T(int(v))
    else:
        return T(v)

# ---------------------------------------------------------------- Size

class Size(List[int]):
    def __init__(self, d: List[int]):
        super().__init__(d)

    def __init__(self):
        super().__init__()

    def __eq__(self, o) -> bool:
        if len(self) != len(o): return False
        i = 0
        for v in o:
            if self[i] != v: return False
            i += 1
        return True

    def __ne__(self, o) -> bool:
        return not (self == o)

    def __repr__(self) -> str:
        return "torch.Size(" + List[int].__repr__(self) + ")"

    def __str__(self) -> str:
        return self.__repr__()

    def numel(self) -> int:
        n = 1
        for v in self:
            n *= v
        return n

# ---------------------------------------------------------------- grad mode

_GRAD_ENABLED = [True]

def is_grad_enabled() -> bool:
    return _GRAD_ENABLED[0]

def set_grad_enabled(mode: bool):
    _GRAD_ENABLED[0] = mode

class no_grad:
    _prev: bool
    def __init__(self):
        self._prev = True
    def __enter__(self):
        self._prev = _GRAD_ENABLED[0]
        _GRAD_ENABLED[0] = False
    def __exit__(self):
        _GRAD_ENABLED[0] = self._prev
    def __call__(self, f):
        return f

class enable_grad:
    _prev: bool
    def __init__(self):
        self._prev = True
    def __enter__(self):
        self._prev = _GRAD_ENABLED[0]
        _GRAD_ENABLED[0] = True
    def __exit__(self):
        _GRAD_ENABLED[0] = self._prev

class inference_mode:
    _prev: bool
    _mode: bool
    def __init__(self, mode: bool = True):
        self._prev = True
        self._mode = mode
    def __enter__(self):
        self._prev = _GRAD_ENABLED[0]
        if self._mode:
            _GRAD_ENABLED[0] = False
    def __exit__(self):
        _GRAD_ENABLED[0] = self._prev

# ---------------------------------------------------------------- storage

class _Storage:
    ptr: Ptr[byte]
    nbytes: int
    def __init__(self, nbytes: int):
        self.ptr = Ptr[byte](nbytes if nbytes > 0 else 1)
        self.nbytes = nbytes
    def __init__(self, ptr: Ptr[byte], nbytes: int):
        self.ptr = ptr
        self.nbytes = nbytes

def _coalesce(shape: List[int], strides: List[int]) -> Tuple[List[int], List[int]]:
    # drop size-1 dims and merge dims that are contiguous with each other
    shp = List[int]()
    st = List[int]()
    for i in range(len(shape)):
        if shape[i] != 1:
            shp.append(shape[i])
            st.append(strides[i])
    if len(shp) == 0:
        return ([1], [0])
    os = [shp[0]]
    ot = [st[0]]
    for i in range(1, len(shp)):
        if ot[-1] == st[i] * shp[i]:
            os[-1] *= shp[i]
            ot[-1] = st[i]
        else:
            os.append(shp[i])
            ot.append(st[i])
    return (os, ot)

def _contig_strides(shape: List[int]) -> List[int]:
    n = len(shape)
    st = [0] * n
    s = 1
    i = n - 1
    while i >= 0:
        st[i] = s
        d = shape[i]
        s *= d if d > 1 else 1
        i -= 1
    return st

def _numel(shape: List[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n

# ---------------------------------------------------------------- autograd node

class _NodeBase[T]:
    """Autograd graph node. `inputs[i]` receives `apply(grad)[i]`.

    Custom autograd: subclass `torch.autograd.Node`, set `self.inputs`
    (List[Tensor]) and override `apply(self, grad) -> List[Optional[Tensor]]`
    (and `name`). Attach with `torch.autograd.attach(out, node)`.
    """
    inputs: List[T]
    saved: List[T]
    ints: List[int]
    ints2: List[int]
    fl: float
    fl2: float
    kind: int
    _name: str
    _freed: bool

    def __init__(self):
        self.inputs = []
        self.saved = []
        self.ints = []
        self.ints2 = []
        self.fl = 0.0
        self.fl2 = 0.0
        self.kind = 0
        self._name = "Node"
        self._freed = False

    def __init__(self, name: str, inputs: List[T]):
        self.inputs = inputs
        self.saved = []
        self.ints = []
        self.ints2 = []
        self.fl = 0.0
        self.fl2 = 0.0
        self.kind = 0
        self._name = name
        self._freed = False

    def name(self) -> str:
        return self._name

    def apply(self, grad: T) -> List[Optional[T]]:
        return _backward_builtin(self, grad)

    def __repr__(self) -> str:
        return "<" + self.name() + ">"

# ---------------------------------------------------------------- Tensor

class Tensor:
    _st: _Storage
    _off: int
    _shape: List[int]
    _strides: List[int]
    _dt: int
    requires_grad: bool
    grad: Optional[Tensor]
    grad_fn: Optional[_NodeBase[Tensor]]
    _retain: bool
    _wrapped: bool

    def __init__(self, st: _Storage, off: int, shape: List[int], strides: List[int], dt: int):
        self._st = st
        self._off = off
        self._shape = shape
        self._strides = strides
        self._dt = dt
        self.requires_grad = False
        self.grad = None
        self.grad_fn = None
        self._retain = False
        self._wrapped = False

    # ---- metadata
    @property
    def shape(self) -> Size:
        return Size(self._shape.copy())

    def size(self) -> Size:
        return Size(self._shape.copy())

    def size(self, dim: int) -> int:
        return self._shape[_wrap(dim, len(self._shape))]

    def stride(self) -> List[int]:
        return self._strides.copy()

    def stride(self, dim: int) -> int:
        return self._strides[_wrap(dim, len(self._shape))]

    def storage_offset(self) -> int:
        return self._off

    @property
    def dtype(self) -> _DType:
        return _DType(self._dt)

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def dim(self) -> int:
        return len(self._shape)

    def ndimension(self) -> int:
        return len(self._shape)

    def numel(self) -> int:
        return _numel(self._shape)

    def nelement(self) -> int:
        return _numel(self._shape)

    def element_size(self) -> int:
        return _itemsize(self._dt)

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def is_cuda(self) -> bool:
        return False

    @property
    def is_leaf(self) -> bool:
        return self.grad_fn is None

    @property
    def is_floating_point_(self) -> bool:
        return _is_float(self._dt)

    def is_floating_point(self) -> bool:
        return _is_float(self._dt)

    def is_contiguous(self) -> bool:
        return _is_contig(self._shape, self._strides)

    def data_ptr(self) -> int:
        return int(self._st.ptr.as_byte() + self._off * _itemsize(self._dt))

    def _ptr(self, T: type) -> Ptr[T]:
        """Raw typed pointer to element [0,...,0] (honours storage offset)."""
        if _dtype_of_type(T) != self._dt:
            raise TypeError("torch: _ptr element type does not match tensor dtype " + str(self.dtype))
        return Ptr[T](self._st.ptr) + self._off

    @property
    def data(self) -> Tensor:
        return self.detach()

    @property
    def T(self) -> Tensor:
        n = len(self._shape)
        p = [n - 1 - i for i in range(n)]
        return permute(self, p)

    @property
    def mT(self) -> Tensor:
        return transpose(self, -2, -1)

    def requires_grad_(self, requires_grad: bool = True) -> Tensor:
        self.requires_grad = requires_grad
        return self

    def retain_grad(self):
        self._retain = True

    def __len__(self) -> int:
        if len(self._shape) == 0:
            raise TypeError("len() of a 0-d tensor")
        return self._shape[0]

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]

    def __hash__(self) -> int:
        return int(self._st.ptr.as_byte()) + self._off

    def __repr__(self) -> str:
        return _format(self)

    def __str__(self) -> str:
        return _format(self)

    def __bool__(self) -> bool:
        return _item_f64(self) != 0.0

    def __float__(self) -> float:
        return _item_f64(self)

    def __int__(self) -> int:
        return int(_item_f64(self)) if _is_float(self._dt) else _item_i64(self)

    def __index__(self) -> int:
        return _item_i64(self)

    def item(self):
        """Python scalar; returns float for floating dtypes else int (bool -> int)."""
        return _item_f64(self)

    def tolist(self) -> List[float]:
        """Flattened values as float (static typing: no nested lists)."""
        c = self.contiguous()
        if _is_complex(c._dt):
            c = _as_real_raw(c)
        n = c.numel()
        out = List[float](n)
        _dispatch(c._dt, _k_tolist, c, out)
        return out

    # ---- autograd
    def backward(self, gradient: Optional[Tensor] = None, retain_graph: bool = False, create_graph: bool = False, inputs: Optional[List[Tensor]] = None):
        g: Tensor
        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            g = ones_like(self)
        else:
            g = gradient
        _run_backward(self, g, create_graph, retain_graph or create_graph)

    def detach(self) -> Tensor:
        t = Tensor(self._st, self._off, self._shape, self._strides, self._dt)
        return t

    def detach_(self) -> Tensor:
        self.grad_fn = None
        self.requires_grad = False
        return self

    def zero_(self) -> Tensor:
        return self.fill_(0.0)

    # ---- dtype conversions
    def to(self, dtype: _DType, non_blocking: bool = False, copy: bool = False) -> Tensor:
        if dtype.code == self._dt and not copy:
            return self
        return _to_dtype(self, dtype.code)

    def to(self, device: str, dtype: Optional[_DType] = None, non_blocking: bool = False, copy: bool = False) -> Tensor:
        if dtype is not None:
            return self.to(dtype)
        return self

    def to(self, other: Tensor, non_blocking: bool = False, copy: bool = False) -> Tensor:
        return self.to(other.dtype)

    def float(self) -> Tensor:
        return self.to(_DType(_F32))

    def double(self) -> Tensor:
        return self.to(_DType(_F64))

    def long(self) -> Tensor:
        return self.to(_DType(_I64))

    def int(self) -> Tensor:
        return self.to(_DType(_I32))

    def bool(self) -> Tensor:
        return self.to(_DType(_BOOL))

    def byte(self) -> Tensor:
        return self.to(_DType(_U8))

    def cpu(self) -> Tensor:
        return self

    def cuda(self) -> Tensor:
        return self

    def type_as(self, other: Tensor) -> Tensor:
        return self.to(other.dtype)

    def __deepcopy__(self) -> Tensor:
        # torch: leaf tensors only; new storage, same requires_grad, detached from graph.
        if self.grad_fn is not None:
            raise RuntimeError("Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.  If you were attempting to deepcopy a module, this may be because of a torch.nn.utils.weight_norm usage, see https://github.com/pytorch/pytorch/pull/103001")
        r = clone(self.detach())
        if self.requires_grad:
            r.requires_grad_(True)
        return r

    def __copy__(self) -> Tensor:
        return self

    def clone(self, memory_format=None) -> Tensor:
        return clone(self)

    def contiguous(self, memory_format=None) -> Tensor:
        if self.is_contiguous():
            return self
        return clone(self)

    def numpy(self, dtype: type = float32, ndim: Literal[int] = 1, force: bool = False):
        """ndarray[dtype, ndim] (static, must match runtime dtype/rank).

        Shares memory when contiguous, like torch."""
        return _to_numpy(self, dtype, ndim)

    # ---- views
    def view(self, *shape) -> Tensor:
        return view(self, _size_args(shape))

    def view(self, dtype: _DType) -> Tensor:
        if dtype.code != self._dt:
            raise NotImplementedError("view(dtype) reinterpretation not supported")
        return self

    def view_as(self, other: Tensor) -> Tensor:
        return view(self, other._shape)

    def reshape(self, *shape) -> Tensor:
        return reshape(self, _size_args(shape))

    def reshape_as(self, other: Tensor) -> Tensor:
        return reshape(self, other._shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        return flatten(self, start_dim, end_dim)

    def unflatten(self, dim: int, sizes) -> Tensor:
        return unflatten(self, dim, sizes)

    def permute(self, *dims) -> Tensor:
        return permute(self, _size_args(dims))

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        return transpose(self, dim0, dim1)

    def swapaxes(self, dim0: int, dim1: int) -> Tensor:
        return transpose(self, dim0, dim1)

    def t(self) -> Tensor:
        return t(self)

    def unsqueeze(self, dim: int) -> Tensor:
        return unsqueeze(self, dim)

    def squeeze(self, dim = None) -> Tensor:
        return squeeze(self, dim)

    def expand(self, *sizes) -> Tensor:
        return expand(self, _size_args(sizes))

    def expand_as(self, other: Tensor) -> Tensor:
        return expand(self, other._shape)

    def repeat(self, *sizes) -> Tensor:
        return repeat(self, _size_args(sizes))

    def split(self, split_size, dim: int = 0) -> List[Tensor]:
        return split(self, split_size, dim)

    def chunk(self, chunks: int, dim: int = 0) -> List[Tensor]:
        return chunk(self, chunks, dim)

    def unbind(self, dim: int = 0) -> List[Tensor]:
        return unbind(self, dim)

    def narrow(self, dim: int, start: int, length: int) -> Tensor:
        return narrow(self, dim, start, length)

    def select(self, dim: int, index: int) -> Tensor:
        return select(self, dim, index)

    def unfold(self, dimension: int, size: int, step: int) -> Tensor:
        return _unfold(self, dimension, size, step)

    def flip(self, *dims) -> Tensor:
        return flip(self, _size_args(dims))

    def roll(self, shifts, dims = None) -> Tensor:
        return roll(self, shifts, dims)

    def movedim(self, source: int, destination: int) -> Tensor:
        return movedim(self, source, destination)

    # ---- indexing
    def __getitem__(self, idx) -> Tensor:
        return _getitem(self, idx)

    def __setitem__(self, idx, value):
        _setitem(self, idx, value)

    def index_select(self, dim: int, index: Tensor) -> Tensor:
        return index_select(self, dim, index)

    def gather(self, dim: int, index: Tensor) -> Tensor:
        return gather(self, dim, index)

    def masked_fill(self, mask: Tensor, value) -> Tensor:
        return masked_fill(self, mask, value)

    def masked_fill_(self, mask: Tensor, value) -> Tensor:
        r = masked_fill(self.detach(), mask, value)
        self.copy_(r)
        return self

    def masked_select(self, mask: Tensor) -> Tensor:
        return masked_select(self, mask)

    def scatter(self, dim: int, index: Tensor, src) -> Tensor:
        return scatter(self, dim, index, src)

    def scatter_add(self, dim: int, index: Tensor, src: Tensor) -> Tensor:
        return scatter_add(self, dim, index, src)

    def index_add(self, dim: int, index: Tensor, source: Tensor, alpha: float = 1.0) -> Tensor:
        return index_add(self, dim, index, source, alpha)

    def nonzero(self, as_tuple: bool = False) -> Tensor:
        return nonzero(self)

    # ---- arithmetic
    def __add__(self, o) -> Tensor: return add(self, o)
    def __radd__(self, o) -> Tensor: return add(_scalar_like(o, self), self)
    def __sub__(self, o) -> Tensor: return sub(self, o)
    def __rsub__(self, o) -> Tensor: return sub(_scalar_like(o, self), self)
    def __mul__(self, o) -> Tensor: return mul(self, o)
    def __rmul__(self, o) -> Tensor: return mul(_scalar_like(o, self), self)
    def __truediv__(self, o) -> Tensor: return div(self, o)
    def __rtruediv__(self, o) -> Tensor: return div(_scalar_like(o, self), self)
    def __floordiv__(self, o) -> Tensor: return floor_divide(self, o)
    def __mod__(self, o) -> Tensor: return remainder(self, o)
    def __pow__(self, o) -> Tensor: return pow(self, o)
    def __rpow__(self, o) -> Tensor: return pow(_scalar_like(o, self), self)
    def __matmul__(self, o: Tensor) -> Tensor: return matmul(self, o)
    def __neg__(self) -> Tensor: return neg(self)
    def __pos__(self) -> Tensor: return self
    def __abs__(self) -> Tensor: return abs(self)
    def __invert__(self) -> Tensor: return logical_not(self) if self._dt == _BOOL else bitwise_not(self)
    def __and__(self, o) -> Tensor: return _bitop(self, o, 0)
    def __or__(self, o) -> Tensor: return _bitop(self, o, 1)
    def __xor__(self, o) -> Tensor: return _bitop(self, o, 2)
    def __eq__(self, o) -> Tensor: return eq(self, o)
    def __ne__(self, o) -> Tensor: return ne(self, o)
    def __lt__(self, o) -> Tensor: return lt(self, o)
    def __le__(self, o) -> Tensor: return le(self, o)
    def __gt__(self, o) -> Tensor: return gt(self, o)
    def __ge__(self, o) -> Tensor: return ge(self, o)

    def __iadd__(self, o) -> Tensor: return self.add_(o)
    def __isub__(self, o) -> Tensor: return self.sub_(o)
    def __imul__(self, o) -> Tensor: return self.mul_(o)
    def __itruediv__(self, o) -> Tensor: return self.div_(o)

    def add(self, o, alpha = 1) -> Tensor: return add(self, o, alpha)
    def sub(self, o, alpha = 1) -> Tensor: return sub(self, o, alpha)
    def mul(self, o) -> Tensor: return mul(self, o)
    def div(self, o, rounding_mode: Optional[str] = None) -> Tensor: return div(self, o, rounding_mode)
    def pow(self, o) -> Tensor: return pow(self, o)
    def matmul(self, o: Tensor) -> Tensor: return matmul(self, o)
    def mm(self, o: Tensor) -> Tensor: return mm(self, o)
    def bmm(self, o: Tensor) -> Tensor: return bmm(self, o)
    def neg(self) -> Tensor: return neg(self)
    def eq(self, o) -> Tensor: return eq(self, o)
    def ne(self, o) -> Tensor: return ne(self, o)
    def lt(self, o) -> Tensor: return lt(self, o)
    def le(self, o) -> Tensor: return le(self, o)
    def gt(self, o) -> Tensor: return gt(self, o)
    def ge(self, o) -> Tensor: return ge(self, o)
    def maximum(self, o: Tensor) -> Tensor: return maximum(self, o)
    def minimum(self, o: Tensor) -> Tensor: return minimum(self, o)
    def equal(self, o: Tensor) -> bool: return equal(self, o)
    def allclose(self, o: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool:
        return allclose(self, o, rtol, atol, equal_nan)
    def outer(self, o: Tensor) -> Tensor: return outer(self, o)
    def dot(self, o: Tensor) -> Tensor: return dot(self, o)

    # ---- unary math
    def exp(self) -> Tensor: return exp(self)
    def log(self) -> Tensor: return log(self)
    def log1p(self) -> Tensor: return log1p(self)
    def log2(self) -> Tensor: return log2(self)
    def log10(self) -> Tensor: return log10(self)
    def expm1(self) -> Tensor: return expm1(self)
    def sqrt(self) -> Tensor: return sqrt(self)
    def rsqrt(self) -> Tensor: return rsqrt(self)
    def abs(self) -> Tensor: return abs(self)
    def sign(self) -> Tensor: return sign(self)
    def sin(self) -> Tensor: return sin(self)
    def cos(self) -> Tensor: return cos(self)
    def tan(self) -> Tensor: return tan(self)
    def tanh(self) -> Tensor: return tanh(self)
    def sinh(self) -> Tensor: return sinh(self)
    def cosh(self) -> Tensor: return cosh(self)
    def atan(self) -> Tensor: return atan(self)
    def asin(self) -> Tensor: return asin(self)
    def acos(self) -> Tensor: return acos(self)
    def sigmoid(self) -> Tensor: return sigmoid(self)
    def relu(self) -> Tensor: return relu(self)
    def erf(self) -> Tensor: return erf(self)
    def floor(self) -> Tensor: return floor(self)
    def ceil(self) -> Tensor: return ceil(self)
    def round(self, decimals: int = 0) -> Tensor: return round(self, decimals)
    def trunc(self) -> Tensor: return trunc(self)
    def frac(self) -> Tensor: return frac(self)
    def reciprocal(self) -> Tensor: return reciprocal(self)
    def square(self) -> Tensor: return square(self)
    def clamp(self, min = None, max = None) -> Tensor: return clamp(self, min, max)
    def clip(self, min = None, max = None) -> Tensor: return clamp(self, min, max)
    def clamp_(self, min = None, max = None) -> Tensor:
        self.copy_(clamp(self.detach(), min, max))
        return self
    def isnan(self) -> Tensor: return isnan(self)
    def isinf(self) -> Tensor: return isinf(self)
    def isfinite(self) -> Tensor: return isfinite(self)
    def nan_to_num(self, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None) -> Tensor:
        return nan_to_num(self, nan, posinf, neginf)
    def logical_not(self) -> Tensor: return logical_not(self)
    def softmax(self, dim: int) -> Tensor: return softmax(self, dim)
    def log_softmax(self, dim: int) -> Tensor: return log_softmax(self, dim)

    # ---- reductions
    def sum(self, dim = None, keepdim: bool = False, dtype: Optional[_DType] = None) -> Tensor:
        return sum(self, dim, keepdim, dtype)
    def mean(self, dim = None, keepdim: bool = False, dtype: Optional[_DType] = None) -> Tensor:
        return mean(self, dim, keepdim, dtype)
    def prod(self, dim = None, keepdim: bool = False) -> Tensor:
        return prod(self, dim, keepdim)
    def var(self, dim = None, unbiased: bool = True, keepdim: bool = False, correction: Optional[int] = None) -> Tensor:
        return var(self, dim, unbiased, keepdim, correction)
    def std(self, dim = None, unbiased: bool = True, keepdim: bool = False, correction: Optional[int] = None) -> Tensor:
        return std(self, dim, unbiased, keepdim, correction)
    def max(self, dim = None, keepdim: bool = False):
        return max(self, dim, keepdim)
    def max(self, other: Tensor) -> Tensor:
        return maximum(self, other)
    def min(self, dim = None, keepdim: bool = False):
        return min(self, dim, keepdim)
    def min(self, other: Tensor) -> Tensor:
        return minimum(self, other)
    def amax(self, dim = None, keepdim: bool = False) -> Tensor:
        return amax(self, dim, keepdim)
    def amin(self, dim = None, keepdim: bool = False) -> Tensor:
        return amin(self, dim, keepdim)
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        return argmax(self, dim, keepdim)
    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        return argmin(self, dim, keepdim)
    def norm(self, p = 2, dim = None, keepdim: bool = False) -> Tensor:
        return norm(self, p, dim, keepdim)
    def logsumexp(self, dim, keepdim: bool = False) -> Tensor:
        return logsumexp(self, dim, keepdim)
    def cumprod(self, dim: int, dtype: Optional[_DType] = None) -> Tensor:
        return cumprod(self, dim, dtype)
    def cummax(self, dim: int):
        return cummax(self, dim)
    def cummin(self, dim: int):
        return cummin(self, dim)
    def logcumsumexp(self, dim: int) -> Tensor:
        return logcumsumexp(self, dim)
    def cumsum(self, dim: int) -> Tensor:
        return cumsum(self, dim)
    def any(self, dim = None, keepdim: bool = False) -> Tensor:
        return any(self, dim, keepdim)
    def all(self, dim = None, keepdim: bool = False) -> Tensor:
        return all(self, dim, keepdim)
    def count_nonzero(self, dim = None) -> Tensor:
        return count_nonzero(self, dim)
    def sort(self, dim: int = -1, descending: bool = False):
        return sort(self, dim, descending)
    def argsort(self, dim: int = -1, descending: bool = False) -> Tensor:
        return argsort(self, dim, descending)
    def topk(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        return topk(self, k, dim, largest, sorted)
    def diag(self, diagonal: int = 0) -> Tensor:
        return diag(self, diagonal)
    def triu(self, diagonal: int = 0) -> Tensor:
        return triu(self, diagonal)
    def tril(self, diagonal: int = 0) -> Tensor:
        return tril(self, diagonal)
    def where(self, condition: Tensor, other) -> Tensor:
        return _where3(condition, self, other)

    # ---- in-place (not recorded by autograd; use under no_grad or on leaves)
    def fill_(self, value) -> Tensor:
        _check_inplace(self)
        _fill(self, float(value))
        return self
    def copy_(self, src: Tensor, non_blocking: bool = False) -> Tensor:
        _check_inplace(self)
        _copy_into(self, expand(src.detach(), self._shape))
        return self
    def add_(self, o, alpha = 1) -> Tensor:
        return self.copy_(add(self.detach(), _detach_any(o), alpha))
    def sub_(self, o, alpha = 1) -> Tensor:
        return self.copy_(sub(self.detach(), _detach_any(o), alpha))
    def mul_(self, o) -> Tensor:
        return self.copy_(mul(self.detach(), _detach_any(o)))
    def div_(self, o) -> Tensor:
        return self.copy_(div(self.detach(), _detach_any(o)))
    def pow_(self, o) -> Tensor:
        return self.copy_(pow(self.detach(), _detach_any(o)))
    def neg_(self) -> Tensor:
        return self.copy_(neg(self.detach()))
    def exp_(self) -> Tensor:
        return self.copy_(exp(self.detach()))
    def sqrt_(self) -> Tensor:
        return self.copy_(sqrt(self.detach()))
    def abs_(self) -> Tensor:
        return self.copy_(abs(self.detach()))
    def relu_(self) -> Tensor:
        return self.copy_(relu(self.detach()))
    def addcmul_(self, t1: Tensor, t2: Tensor, value = 1) -> Tensor:
        return self.copy_(add(self.detach(), mul(t1.detach(), t2.detach()), value))
    def addcdiv_(self, t1: Tensor, t2: Tensor, value = 1) -> Tensor:
        return self.copy_(add(self.detach(), div(t1.detach(), t2.detach()), value))
    def lerp_(self, end: Tensor, weight) -> Tensor:
        d = self.detach()
        return self.copy_(add(d, mul(sub(end.detach(), d), weight)))
    def uniform_(self, a: float = 0.0, b: float = 1.0, generator: Optional[Generator] = None) -> Tensor:
        _check_inplace(self)
        _uniform_fill(self, a, b, _gen(generator))
        return self
    def normal_(self, mean: float = 0.0, std: float = 1.0, generator: Optional[Generator] = None) -> Tensor:
        _check_inplace(self)
        _normal_fill(self, mean, std, _gen(generator))
        return self
    def bernoulli_(self, p: float = 0.5, generator: Optional[Generator] = None) -> Tensor:
        _check_inplace(self)
        _bernoulli_fill(self, p, _gen(generator))
        return self
    def random_(self, from_: int, to: int, generator: Optional[Generator] = None) -> Tensor:
        _check_inplace(self)
        _randint_fill(self, from_, to, _gen(generator))
        return self
    def exponential_(self, lambd: float = 1.0, generator: Optional[Generator] = None) -> Tensor:
        _check_inplace(self)
        _exponential_fill(self, lambd, _gen(generator))
        return self
    def bernoulli(self, generator: Optional[Generator] = None) -> Tensor:
        return bernoulli(self, generator=generator)
    def new_zeros(self, *size) -> Tensor:
        return zeros(_size_args(size), dtype=self.dtype)
    def new_ones(self, *size) -> Tensor:
        return ones(_size_args(size), dtype=self.dtype)
    def new_empty(self, *size) -> Tensor:
        return empty(_size_args(size), dtype=self.dtype)
    def new_full(self, size, fill_value) -> Tensor:
        return full(_size_args((size,)), fill_value, dtype=self.dtype)
    def new_tensor(self, data) -> Tensor:
        return tensor(data, dtype=self.dtype)
    def type(self, dtype: _DType) -> Tensor:
        return self.to(dtype)
    @property
    def real(self) -> Tensor:
        return real(self)
    @property
    def imag(self) -> Tensor:
        return imag(self)
    def angle(self) -> Tensor:
        return angle(self)
    def conj(self) -> Tensor:
        return conj(self)
    def is_complex(self) -> bool:
        return _is_complex(self._dt)

Node = _NodeBase[Tensor]

def _wrap(d: int, n: int) -> int:
    m = n if n > 0 else 1
    if d < -m or d >= m:
        raise IndexError("Dimension out of range (expected to be in range of [" + str(-m) + ", " + str(m - 1) + "], but got " + str(d) + ")")
    return d + m if d < 0 else d

def _is_contig(shape: List[int], strides: List[int]) -> bool:
    s = 1
    i = len(shape) - 1
    while i >= 0:
        d = shape[i]
        if d != 1:
            if strides[i] != s:
                return False
            s *= d
        i -= 1
    return True

def _size_args(args) -> List[int]:
    if static.len(args) == 1:
        a = args[0]
        if isinstance(a, int):
            return [a]
        elif isinstance(a, Tensor):
            return [int(v) for v in a.tolist()]
        else:
            return [int(v) for v in a]
    else:
        return [int(v) for v in args]

def _to_list(dims) -> List[int]:
    if isinstance(dims, int):
        return [dims]
    else:
        return [int(v) for v in dims]

def _gen(g: Optional[Generator]) -> Generator:
    if g is None:
        return default_generator
    return g

def _detach_any(o):
    if isinstance(o, Tensor):
        return o.detach()
    else:
        return o

# raw allocation
def _new(shape: List[int], dt: int) -> Tensor:
    n = _numel(shape)
    st = _Storage(n * _itemsize(dt))
    return Tensor(st, 0, shape, _contig_strides(shape), dt)

# offsets of every element of (shape, strides, off) in C order
def _offsets(shape: List[int], strides: List[int], off: int) -> List[int]:
    n = _numel(shape)
    nd = len(shape)
    out = List[int](n)
    if n == 0:
        return out
    if nd == 0:
        out.append(off)
        return out
    idx = [0] * nd
    cur = off
    last = nd - 1
    sl = strides[last]
    dl = shape[last]
    k = 0
    while k < n:
        for j in range(dl):
            out.append(cur + j * sl)
        k += dl
        # increment higher dims
        d = last - 1
        while d >= 0:
            idx[d] += 1
            cur += strides[d]
            if idx[d] < shape[d]:
                break
            cur -= strides[d] * shape[d]
            idx[d] = 0
            d -= 1
    return out

# ================================================================ C math

from C import erf(float) -> float as _erf64
from C import erff(float32) -> float32 as _erff
from C import erfc(float) -> float as _erfc64
from C import erfcf(float32) -> float32 as _erfcf
from C import lgamma(float) -> float as _lgamma64
from C import lgammaf(float32) -> float32 as _lgammaf

def _erf(v):
    if isinstance(v, float32):
        return _erff(v)
    else:
        return _erf64(float(v))

def _erfc(v):
    if isinstance(v, float32):
        return _erfcf(v)
    else:
        return _erfc64(float(v))

def _lgamma(v):
    if isinstance(v, float32):
        return _lgammaf(v)
    else:
        return _lgamma64(float(v))

def _imax(a: int, b: int) -> int:
    return a if a > b else b

def _imin(a: int, b: int) -> int:
    return a if a < b else b

def _unw(x):
    return x.__val__()

# ================================================================ basic kernels

def _k_tolist(z, c: Tensor, out: List[float]):
    T = type(z)
    p = Ptr[T](c._st.ptr) + c._off
    for i in range(c.numel()):
        out.append(_cv(p[i], float))

def _k_read(z, t: Tensor, off: int) -> float:
    T = type(z)
    return _cv(Ptr[T](t._st.ptr)[off], float)

def _k_read_i(z, t: Tensor, off: int) -> int:
    T = type(z)
    return _cv(Ptr[T](t._st.ptr)[off], int)

def _item_f64(t: Tensor) -> float:
    if t.numel() != 1:
        raise RuntimeError("a Tensor with " + str(t.numel()) + " elements cannot be converted to Scalar")
    return _dispatch(t._dt, _k_read, t, t._off)

def _item_i64(t: Tensor) -> int:
    if t.numel() != 1:
        raise RuntimeError("a Tensor with " + str(t.numel()) + " elements cannot be converted to Scalar")
    return _dispatch(t._dt, _k_read_i, t, t._off)

def _k_fill(z, t: Tensor, v: float):
    T = type(z)
    x = _cv(v, T)
    p = Ptr[T](t._st.ptr)
    if t.is_contiguous():
        q = p + t._off
        for i in range(t.numel()):
            q[i] = x
    else:
        for o in _offsets(t._shape, t._strides, t._off):
            p[o] = x

def _fill(t: Tensor, v: float):
    if _is_complex(t._dt):
        r = _as_real_raw(t)
        _dispatch(r._dt, _k_fill, _select_raw(r, len(r._shape) - 1, 0), v)
        _dispatch(r._dt, _k_fill, _select_raw(r, len(r._shape) - 1, 1), 0.0)
        return
    _dispatch(t._dt, _k_fill, t, v)

def _k_copy2(zs, zd, dst: Tensor, src: Tensor):
    S = type(zs)
    D = type(zd)
    ps = Ptr[S](src._st.ptr)
    pd = Ptr[D](dst._st.ptr)
    n = dst.numel()
    if dst.is_contiguous() and src.is_contiguous():
        a = ps + src._off
        b = pd + dst._off
        for i in range(n):
            b[i] = _cv(a[i], D)
    elif dst.is_contiguous():
        b = pd + dst._off
        nd = len(src._shape)
        if n == 0:
            return
        if nd == 0:
            b[0] = _cv(ps[src._off], D)
            return
        shp, st = _coalesce(src._shape, src._strides)
        nd = len(shp)
        last = nd - 1
        dl = shp[last]
        sl = st[last]
        idx = [0] * nd
        cur = src._off
        i = 0
        while i < n:
            if sl == 0:
                v = _cv(ps[cur], D)
                for j in range(dl):
                    b[i + j] = v
            elif sl == 1:
                q = ps + cur
                for j in range(dl):
                    b[i + j] = _cv(q[j], D)
            else:
                for j in range(dl):
                    b[i + j] = _cv(ps[cur + j * sl], D)
            i += dl
            d = last - 1
            while d >= 0:
                idx[d] += 1
                cur += st[d]
                if idx[d] < shp[d]:
                    break
                cur -= st[d] * shp[d]
                idx[d] = 0
                d -= 1
    else:
        od = _offsets(dst._shape, dst._strides, dst._off)
        os = _offsets(src._shape, src._strides, src._off)
        for i in range(n):
            pd[od[i]] = _cv(ps[os[i]], D)

def _k_copy(zd, dst: Tensor, src: Tensor):
    _dispatch(src._dt, _k_copy2, zd, dst, src)

def _copy_into(dst: Tensor, src: Tensor):
    if dst._shape != src._shape:
        raise RuntimeError("copy_: shape mismatch " + str(dst._shape) + " vs " + str(src._shape))
    if _is_complex(dst._dt) or _is_complex(src._dt):
        dr = _as_real_raw(dst) if _is_complex(dst._dt) else dst
        if not _is_complex(dst._dt):
            _dispatch(dst._dt, _k_copy, dst, _select_raw(_as_real_raw(src), len(src._shape), 0))
        elif _is_complex(src._dt):
            _dispatch(dr._dt, _k_copy, dr, _as_real_raw(src))
        else:
            # real -> complex: real part = src, imag = 0
            re = _select_raw(dr, len(dr._shape) - 1, 0)
            im = _select_raw(dr, len(dr._shape) - 1, 1)
            _dispatch(dr._dt, _k_copy, re, src)
            _dispatch(dr._dt, _k_fill, im, 0.0)
        return
    _dispatch(dst._dt, _k_copy, dst, src)

def _real_code(c: int) -> int:
    return _F64 if c == _C128 else _F32

def _complex_code(c: int) -> int:
    return _C128 if c == _F64 else _C64

# complex tensor -> float view with trailing dim 2 (no autograd)
def _as_real_raw(x: Tensor) -> Tensor:
    return Tensor(x._st, x._off * 2, x._shape + [2], [s * 2 for s in x._strides] + [1], _real_code(x._dt))

def _select_raw(x: Tensor, d: int, i: int) -> Tensor:
    shp = x._shape.copy()
    st = x._strides.copy()
    off = x._off + i * st[d]
    shp.pop(d)
    st.pop(d)
    return Tensor(x._st, off, shp, st, x._dt)

# contiguous tensor of dtype dt holding the values of t (no copy if possible)
def _contig_as(t: Tensor, dt: int) -> Tensor:
    if t._dt == dt and t.is_contiguous():
        return t
    r = _new(t._shape.copy(), dt)
    _copy_into(r, t)
    return r

# ================================================================ autograd plumbing

def _needs(ts: List[Tensor]) -> bool:
    if not _GRAD_ENABLED[0]:
        return False
    for x in ts:
        if x.requires_grad:
            return True
    return False

def _node(name: str, kind: int, inputs: List[Tensor]) -> Node:
    n = Node(name, inputs)
    n.kind = kind
    return n

_LEAF_INPLACE_MSG = "a leaf Variable that requires grad is being used in an in-place operation."
_FREED_MSG = "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."

# nodes whose torch counterpart saves no tensors/scalars: re-running backward
# through them after the graph was freed is legal in torch
_NOSAVE_NODES = Set[str](["AddBackward0", "SubBackward0", "NegBackward0", "SumBackward0", "SumBackward1",
    "MeanBackward0", "ViewBackward0", "UnsafeViewBackward0", "ExpandBackward0", "PermuteBackward0",
    "TransposeBackward0", "SqueezeBackward0", "UnsqueezeBackward0", "SliceBackward0", "SelectBackward0",
    "CatBackward0", "CloneBackward0", "ToCopyBackward0", "FlipBackward0", "CumsumBackward0",
    "DiagonalBackward0", "DiagEmbedBackward0", "TrilBackward0", "TriuBackward0", "UnfoldBackward0",
    "SignBackward0", "FloorBackward0", "CeilBackward0", "RoundBackward0", "TruncBackward0",
    "FracBackward0", "ConstantPadNdBackward0", "ViewAsComplexBackward0", "ViewAsRealBackward0"])

def _check_inplace(t: Tensor):
    if _GRAD_ENABLED[0] and t.requires_grad and t.grad_fn is None:
        raise RuntimeError(_LEAF_INPLACE_MSG)

def _attach(out: Tensor, n: Node) -> Tensor:
    out.grad_fn = n
    out.requires_grad = True
    return out

def _sum_to(g: Tensor, shape: List[int]) -> Tensor:
    if g._shape == shape:
        return g
    nd = len(g._shape)
    ns = len(shape)
    lead = nd - ns
    dims = List[int]()
    for i in range(lead):
        dims.append(i)
    for i in range(ns):
        if shape[i] == 1 and g._shape[lead + i] != 1:
            dims.append(lead + i)
    r = g
    if len(dims) > 0:
        r = sum(g, dims, True)
    if lead > 0:
        r = reshape(r, shape)
    elif r._shape != shape:
        r = reshape(r, shape)
    return r

def _accum_leaf(t: Tensor, g: Tensor):
    tg = t.grad
    if tg is None:
        t.grad = _contig_as(g, t._dt).clone() if id(g._st) == id(t._st) else _contig_as(g, t._dt).detach()
        # ensure grad owns memory distinct from intermediates reused later
        if t.grad is not None:
            gg = _unw(t.grad)
            if not gg.is_contiguous() or gg._shape != t._shape:
                t.grad = clone(reshape(gg, t._shape))
    else:
        old = _unw(tg)
        t.grad = add(old, _contig_as(g, t._dt))

def _run_backward(root: Tensor, grad: Tensor, create_graph: bool = False, retain_graph: bool = False):
    prev = _GRAD_ENABLED[0]
    _GRAD_ENABLED[0] = create_graph
    try:
        _run_backward_impl(root, grad, retain_graph)
    finally:
        _GRAD_ENABLED[0] = prev

def _run_backward_impl(root: Tensor, grad: Tensor, retain_graph: bool = False):
    if root.grad_fn is None:
        if root.requires_grad:
            _accum_leaf(root, grad)
        return
    rn = _unw(root.grad_fn)
    # iterative DFS post-order
    order = List[Node]()
    visited = Set[int]()
    stack = [rn]
    pos = [0]
    visited.add(id(rn))
    while len(stack) > 0:
        n = stack[-1]
        i = pos[-1]
        if i < len(n.inputs):
            pos[-1] = i + 1
            inp = n.inputs[i]
            fn = inp.grad_fn
            if fn is not None:
                c = _unw(fn)
                if id(c) not in visited:
                    visited.add(id(c))
                    stack.append(c)
                    pos.append(0)
        else:
            order.append(n)
            stack.pop()
            pos.pop()
    grads = Dict[int, Tensor]()
    grads[id(rn)] = grad
    if root._retain:
        _accum_leaf(root, grad)
    k = len(order) - 1
    while k >= 0:
        n = order[k]
        k -= 1
        key = id(n)
        if key not in grads:
            continue
        gn = grads.pop(key)
        if n._freed:
            raise RuntimeError(_FREED_MSG)
        outs = n.apply(gn)
        if not retain_graph and n._name not in _NOSAVE_NODES:
            n._freed = True
            n.saved = []
        for i in range(_imin(len(outs), len(n.inputs))):
            go = outs[i]
            if go is None:
                continue
            inp = n.inputs[i]
            if not inp.requires_grad:
                continue
            gg = _unw(go)
            if gg._dt != inp._dt and _is_float(inp._dt):
                gg = _to_dtype(gg, inp._dt)
            if gg._shape != inp._shape:
                gg = _sum_to(gg, inp._shape)
            fn = inp.grad_fn
            if fn is not None:
                cid = id(_unw(fn))
                if cid in grads:
                    grads[cid] = add(grads[cid], gg)
                else:
                    grads[cid] = gg
                if inp._retain:
                    _accum_leaf(inp, gg)
            else:
                _accum_leaf(inp, gg)

# ================================================================ dtype / clone

def _to_dtype(x: Tensor, dt: int) -> Tensor:
    r = _new(x._shape.copy(), dt)
    _copy_into(r, x)
    if _is_float(dt) and _is_float(x._dt) and _needs([x]):
        _attach(r, _node("ToCopyBackward0", _K_TOCOPY, [x]))
    return r

def clone(input: Tensor, memory_format=None) -> Tensor:
    r = _new(input._shape.copy(), input._dt)
    _copy_into(r, input)
    if _needs([input]):
        _attach(r, _node("CloneBackward0", _K_IDENT, [input]))
    return r

def _view_of(x: Tensor, shape: List[int], strides: List[int], off: int) -> Tensor:
    return Tensor(x._st, off, shape, strides, x._dt)

# ================================================================ backward kinds

_K_IDENT = 1
_K_TOCOPY = 2
_K_ADD = 3
_K_SUB = 4
_K_MUL = 5
_K_DIV = 6
_K_POW = 7
_K_POWS = 8
_K_RPOWS = 9
_K_NEG = 10
_K_UNARY = 11
_K_MAXIMUM = 12
_K_MINIMUM = 13
_K_WHERE = 14
_K_SUM = 15
_K_AMAX = 16
_K_MAXDIM = 17
_K_VIEW = 18
_K_PERMUTE = 19
_K_EXPAND = 20
_K_SELECT = 21
_K_SLICE = 22
_K_CAT = 23
_K_INDEX_SELECT = 24
_K_GATHER = 25
_K_MASKED_FILL = 26
_K_MM = 27
_K_BMM = 28
_K_FLIP = 29
_K_UNFOLD = 30
_K_CUMSUM = 31
_K_PROD = 32
_K_CLAMP = 33
_K_SCATTER_ADD = 34
_K_INDEX_ADD = 35
_K_DIAG = 36
_K_TRIL = 37
_K_STRIDED = 38
_K_ATAN2 = 39
_K_VAR = 40
_K_VAC = 41
_K_CUSTOMFN = 42

# unary sub-kinds (node.ints[0])
_U_EXP = 1
_U_LOG = 2
_U_LOG1P = 3
_U_EXPM1 = 4
_U_SQRT = 5
_U_RSQRT = 6
_U_ABS = 7
_U_SIN = 8
_U_COS = 9
_U_TAN = 10
_U_TANH = 11
_U_SIGMOID = 12
_U_RELU = 13
_U_ERF = 14
_U_ZERO = 15
_U_RECIP = 16
_U_SQUARE = 17
_U_SINH = 18
_U_COSH = 19
_U_ASIN = 20
_U_ACOS = 21
_U_ATAN = 22
_U_LOG2 = 23
_U_LOG10 = 24
_U_ERFC = 25
_U_LGAMMA = 26

def _O(t: Tensor) -> Optional[Tensor]:
    return Optional[Tensor](t)

def _backward_builtin(n: Node, g: Tensor) -> List[Optional[Tensor]]:
    k = n.kind
    if k == _K_IDENT:
        return [_O(g)]
    if k == _K_TOCOPY:
        return [_O(g)]
    if k == _K_ADD:
        return [_O(g), _O(mul(g, n.fl) if n.fl != 1.0 else g)]
    if k == _K_SUB:
        return [_O(g), _O(neg(g) if n.fl == 1.0 else mul(g, -n.fl))]
    if k == _K_MUL:
        a = n.inputs[0]
        b = n.inputs[1]
        return [_O(mul(g, b.detach())), _O(mul(g, a.detach()))]
    if k == _K_DIV:
        a = n.inputs[0].detach()
        b = n.inputs[1].detach()
        ga = div(g, b)
        return [_O(ga), _O(neg(div(mul(ga, a), b)))]
    if k == _K_POW:
        a = n.inputs[0].detach()
        b = n.inputs[1].detach()
        out = n.saved[0]
        ga = mul(mul(g, b), pow(a, sub(b, 1.0)))
        la = log(a)
        gb = mul(mul(g, out), masked_fill(la, eq(a, 0.0), 0.0))
        return [_O(ga), _O(gb)]
    if k == _K_POWS:
        a = n.inputs[0].detach()
        e = n.fl
        if e == 0.0:
            return [_O(zeros_like(a))]
        return [_O(mul(g, mul(pow(a, e - 1.0), e)))]
    if k == _K_RPOWS:
        # out = base ** x
        out = n.saved[0]
        return [_O(mul(mul(g, out), _math.log(n.fl)))]
    if k == _K_NEG:
        return [_O(neg(g))]
    if k == _K_UNARY:
        return [_O(_unary_backward(n.ints[0], g, n.inputs[0].detach(), n.saved[0]))]
    if k == _K_CLAMP:
        mask = n.saved[0]
        return [_O(mul(g, mask))]
    if k == _K_MAXIMUM or k == _K_MINIMUM:
        a = n.inputs[0].detach()
        b = n.inputs[1].detach()
        ma = gt(a, b) if k == _K_MAXIMUM else lt(a, b)
        mb = lt(a, b) if k == _K_MAXIMUM else gt(a, b)
        me = eq(a, b)
        ga = add(mul(g, ma), mul(mul(g, me), 0.5))
        gb = add(mul(g, mb), mul(mul(g, me), 0.5))
        return [_O(ga), _O(gb)]
    if k == _K_WHERE:
        c = n.saved[0]
        zg = zeros_like(g)
        return [_O(_where3(c, g, zg)), _O(_where3(c, zg, g))]
    if k == _K_SUM:
        shp = n.ints
        gg = g
        if n.fl == 0.0:  # keepdim False: reinsert reduced dims
            gg = reshape(g, n.ints2)
        return [_O(expand(gg, shp))]
    if k == _K_AMAX:
        x = n.inputs[0].detach()
        outk = n.saved[0]  # keepdim result
        mask = _contig_as(eq(x, outk), x._dt)
        cnt = sum(mask, n.ints2, True)
        gg = g
        if n.fl == 0.0:
            gg = reshape(g, outk._shape)
        return [_O(div(mul(mask, gg), cnt))]
    if k == _K_MAXDIM:
        idx = n.saved[0]  # keepdim indices
        d = n.ints[0]
        gg = g
        if n.fl == 0.0:
            gg = unsqueeze(g, d)
        z = zeros(n.ints2, dtype=_DType(g._dt))
        return [_O(scatter(z, d, idx, gg))]
    if k == _K_VIEW:
        return [_O(reshape(g, n.ints))]
    if k == _K_PERMUTE:
        p = n.ints
        inv = [0] * len(p)
        for i in range(len(p)):
            inv[p[i]] = i
        return [_O(permute(g, inv))]
    if k == _K_EXPAND:
        return [_O(_sum_to(g, n.ints))]
    if k == _K_SELECT:
        z = zeros(n.ints, dtype=_DType(g._dt))
        select(z, n.ints2[0], n.ints2[1]).copy_(g)
        return [_O(z)]
    if k == _K_SLICE:
        z = zeros(n.ints, dtype=_DType(g._dt))
        _slice(z, n.ints2[0], n.ints2[1], n.ints2[2], n.ints2[3]).copy_(g)
        return [_O(z)]
    if k == _K_CAT:
        d = n.ints2[0]
        outs = List[Optional[Tensor]]()
        st = 0
        for sz in n.ints:
            outs.append(_O(narrow(g, d, st, sz)))
            st += sz
        return outs
    if k == _K_INDEX_SELECT:
        idx = n.saved[0]
        z = zeros(n.ints, dtype=_DType(g._dt))
        return [_O(index_add(z, n.ints2[0], idx, g))]
    if k == _K_GATHER:
        idx = n.saved[0]
        z = zeros(n.ints, dtype=_DType(g._dt))
        return [_O(scatter_add(z, n.ints2[0], idx, g))]
    if k == _K_SCATTER_ADD:
        idx = n.saved[0]
        return [_O(g), _O(gather(g, n.ints2[0], idx))]
    if k == _K_INDEX_ADD:
        idx = n.saved[0]
        return [_O(g), _O(mul(index_select(g, n.ints2[0], idx), n.fl))]
    if k == _K_MASKED_FILL:
        m = n.saved[0]
        return [_O(masked_fill(g, m, 0.0))]
    if k == _K_MM:
        a = n.inputs[0].detach()
        b = n.inputs[1].detach()
        return [_O(mm(g, t(b))), _O(mm(t(a), g))]
    if k == _K_BMM:
        a = n.inputs[0].detach()
        b = n.inputs[1].detach()
        return [_O(bmm(g, transpose(b, 1, 2))), _O(bmm(transpose(a, 1, 2), g))]
    if k == _K_FLIP:
        return [_O(flip(g, n.ints))]
    if k == _K_UNFOLD:
        # ints = input shape, ints2 = [dim, size, step]
        d = n.ints2[0]
        size = n.ints2[1]
        step = n.ints2[2]
        z = zeros(n.ints, dtype=_DType(g._dt))
        nw = g._shape[d]
        for i in range(nw):
            win = narrow(z, d, i * step, size)
            gi = movedim(select(g, d, i), len(g._shape) - 2, d)
            win.copy_(add(win, gi))
        return [_O(z)]
    if k == _K_CUMSUM:
        d = n.ints[0]
        return [_O(flip(cumsum(flip(g, [d]), d), [d]))]
    if k == _K_PROD:
        x = n.inputs[0].detach()
        out = n.saved[0]
        gg = g
        if n.fl == 0.0:
            gg = reshape(g, out._shape)
        return [_O(div(mul(gg, out), x))]
    if k == _K_DIAG:
        if len(n.ints) == 1:
            return [_O(diag(g, n.ints2[0]))]
        z = zeros(n.ints, dtype=_DType(g._dt))
        return [_O(add(z, diag(g, n.ints2[0])))]
    if k == _K_TRIL:
        if n.ints2[1] == 1:
            return [_O(tril(g, n.ints2[0]))]
        return [_O(triu(g, n.ints2[0]))]
    if k == _K_STRIDED:
        # generic view of a contiguous-base region: ints=input shape,
        # ints2=[off, ndim, shape..., strides...] relative to contiguous input
        z = zeros(n.ints, dtype=_DType(g._dt))
        nd = n.ints2[1]
        shp = n.ints2[2:2 + nd]
        sts = n.ints2[2 + nd:2 + 2 * nd]
        v = Tensor(z._st, n.ints2[0], shp, sts, z._dt)
        _scatter_add_view(v, g)
        return [_O(z)]
    if k == _K_ATAN2:
        y = n.inputs[0].detach()
        x = n.inputs[1].detach()
        den = add(mul(x, x), mul(y, y))
        return [_O(div(mul(g, x), den)), _O(neg(div(mul(g, y), den)))]
    if k == _K_VAR:
        return [_O(view_as_complex(g.contiguous()))]
    if k == _K_VAC:
        return [_O(view_as_real(g))]
    raise NotImplementedError("backward not implemented for " + n.name())

def _addv(x, y):
    if isinstance(x, bool): return x or y
    else: return x + y

def _scatter_add_view(v: Tensor, g: Tensor):
    _dispatch(v._dt, _k_addview, v, _contig_as(g, v._dt))

def _k_addview(z, v: Tensor, g: Tensor):
    T = type(z)
    p = Ptr[T](v._st.ptr)
    q = Ptr[T](g._st.ptr) + g._off
    i = 0
    for o in _offsets(v._shape, v._strides, v._off):
        p[o] = _addv(p[o], q[i])
        i += 1

def _unary_backward(u: int, g: Tensor, x: Tensor, out: Tensor) -> Tensor:
    if u == _U_EXP: return mul(g, out)
    if u == _U_LOG: return div(g, x)
    if u == _U_LOG1P: return div(g, add(x, 1.0))
    if u == _U_EXPM1: return mul(g, add(out, 1.0))
    if u == _U_SQRT: return div(g, mul(out, 2.0))
    if u == _U_RSQRT: return mul(mul(g, pow(out, 3.0)), -0.5)
    if u == _U_ABS: return mul(g, sign(x))
    if u == _U_SIN: return mul(g, cos(x))
    if u == _U_COS: return neg(mul(g, sin(x)))
    if u == _U_TAN: return mul(g, add(mul(out, out), 1.0))
    if u == _U_TANH: return mul(g, rsub(mul(out, out), 1.0))
    if u == _U_SIGMOID: return mul(g, mul(out, rsub(out, 1.0)))
    if u == _U_RELU: return mul(g, _contig_as(gt(out, 0.0), g._dt))
    if u == _U_ERF: return mul(g, mul(exp(neg(mul(x, x))), 1.1283791670955126))
    if u == _U_ERFC: return mul(g, mul(exp(neg(mul(x, x))), -1.1283791670955126))
    if u == _U_ZERO: return zeros_like(g)
    if u == _U_RECIP: return neg(mul(g, mul(out, out)))
    if u == _U_SQUARE: return mul(g, mul(x, 2.0))
    if u == _U_SINH: return mul(g, cosh(x))
    if u == _U_COSH: return mul(g, sinh(x))
    if u == _U_ASIN: return div(g, sqrt(rsub(mul(x, x), 1.0)))
    if u == _U_ACOS: return neg(div(g, sqrt(rsub(mul(x, x), 1.0))))
    if u == _U_ATAN: return div(g, add(mul(x, x), 1.0))
    if u == _U_LOG2: return div(g, mul(x, 0.6931471805599453))
    if u == _U_LOG10: return div(g, mul(x, 2.302585092994046))
    raise NotImplementedError("unary backward")

# ================================================================ scalars / promotion

def _scalar_tensor(v, dt: int) -> Tensor:
    r = _new(List[int](), dt)
    _dispatch(dt, _k_set_scalar, r, v)
    r._wrapped = True
    return r

def _k_set_scalar(z, r: Tensor, v):
    T = type(z)
    Ptr[T](r._st.ptr)[0] = _cv(v, T)

def _scalar_like(o, ref: Tensor) -> Tensor:
    if isinstance(o, Tensor):
        return o
    elif isinstance(o, Optional[Tensor]):
        if o is None:
            raise TypeError("unsupported operand: NoneType (tensor.grad is None?)")
        return o.__val__()
    elif isinstance(o, bool):
        return _scalar_tensor(o, _BOOL)
    elif isinstance(o, int):
        return _scalar_tensor(o, _I64)
    elif isinstance(o, float) or isinstance(o, float32):
        return _scalar_tensor(float(o), _F64)
    else:
        return _scalar_tensor(o, _I64)

def _as_t(o, ref: Tensor) -> Tensor:
    return _scalar_like(o, ref)

# result dtype of a binary op (torch type promotion with 0-dim/wrapped weak typing)
def _result_dt(a: Tensor, b: Tensor) -> int:
    wa = a._wrapped or len(a._shape) == 0
    wb = b._wrapped or len(b._shape) == 0
    if wa == wb:
        if a._wrapped and b._wrapped:
            return _promote2(a._dt, b._dt)
        return _promote2(a._dt, b._dt)
    strong = b if wa else a
    weak = a if wa else b
    cs = _category(strong._dt)
    cw = _category(weak._dt)
    if cw > cs:
        if weak._wrapped:
            return _F32 if cw == 2 else _I64
        return weak._dt
    return strong._dt

def _broadcast_shapes(a: List[int], b: List[int]) -> List[int]:
    na = len(a)
    nb = len(b)
    n = _imax(na, nb)
    out = [0] * n
    for i in range(n):
        da = a[na - n + i] if na - n + i >= 0 else 1
        db = b[nb - n + i] if nb - n + i >= 0 else 1
        if da == db or db == 1:
            out[i] = da
        elif da == 1:
            out[i] = db
        else:
            raise RuntimeError("The size of tensor a (" + str(da) + ") must match the size of tensor b (" + str(db) + ") at non-singleton dimension " + str(i))
    return out

def broadcast_shapes(*shapes) -> Size:
    r = List[int]()
    for s in shapes:
        r = _broadcast_shapes(r, _to_list(s))
    return Size(r)

# materialize x broadcast to shape, contiguous in dtype dt
def _bcast_contig(x: Tensor, shape: List[int], dt: int) -> Tensor:
    if x._shape == shape:
        return _contig_as(x, dt)
    return _contig_as(_expand_raw(x, shape), dt)

def _expand_raw(x: Tensor, shape: List[int]) -> Tensor:
    nd = len(shape)
    nx = len(x._shape)
    st = [0] * nd
    shp = shape.copy()
    for i in range(nd):
        j = i - (nd - nx)
        if shape[i] == -1:
            if j < 0:
                raise RuntimeError("expand: -1 not allowed in leading, non-existing dimension")
            shp[i] = x._shape[j]
        if j >= 0:
            if x._shape[j] == shp[i]:
                st[i] = x._strides[j]
            elif x._shape[j] == 1:
                st[i] = 0
            else:
                raise RuntimeError("The expanded size of the tensor (" + str(shp[i]) + ") must match the existing size (" + str(x._shape[j]) + ") at non-singleton dimension " + str(i) + ".  Target sizes: " + str(shape) + ".  Tensor sizes: " + str(x._shape))
    return Tensor(x._st, x._off, shp, st, x._dt)

# ================================================================ elementwise kernels

def _k_bin(z, a: Tensor, b: Tensor, out: Tensor, f):
    T = type(z)
    n = out.numel()
    pa = Ptr[T](a._st.ptr) + a._off
    pb = Ptr[T](b._st.ptr) + b._off
    po = Ptr[T](out._st.ptr)
    if b.numel() == 1 and n != 1:
        bv = pb[0]
        for i in range(n):
            po[i] = T(f(pa[i], bv))
    elif a.numel() == 1 and n != 1:
        av = pa[0]
        for i in range(n):
            po[i] = T(f(av, pb[i]))
    else:
        for i in range(n):
            po[i] = T(f(pa[i], pb[i]))

def _k_cmp(z, a: Tensor, b: Tensor, out: Tensor, f):
    T = type(z)
    n = out.numel()
    pa = Ptr[T](a._st.ptr) + a._off
    pb = Ptr[T](b._st.ptr) + b._off
    po = Ptr[bool](out._st.ptr)
    if b.numel() == 1 and n != 1:
        bv = pb[0]
        for i in range(n):
            po[i] = f(pa[i], bv)
    else:
        for i in range(n):
            po[i] = f(pa[i], pb[i])

def _prep2(a: Tensor, b: Tensor, dt: int):
    shape = _broadcast_shapes(a._shape, b._shape)
    ca = _contig_as(a, dt) if a.numel() == 1 else _bcast_contig(a, shape, dt)
    cb = _contig_as(b, dt) if b.numel() == 1 else _bcast_contig(b, shape, dt)
    return shape, ca, cb

def _binary(a: Tensor, b: Tensor, f, dt: int) -> Tensor:
    shape, ca, cb = _prep2(a, b, dt)
    out = _new(shape, dt)
    _dispatch(dt, _k_bin, ca, cb, out, f)
    return out

def _compare(a, b, f) -> Tensor:
    ta = _as_t(a, _scalar_tensor(0, _I64))
    tb = _as_t(b, ta)
    dt = _result_dt(ta, tb)
    shape, ca, cb = _prep2(ta, tb, dt)
    out = _new(shape, _BOOL)
    _dispatch(dt, _k_cmp, ca, cb, out, f)
    return out

def _f_add(x, y):
    if isinstance(x, bool): return x or y
    else: return x + y
def _f_sub(x, y): return x - y
def _f_mul(x, y):
    if isinstance(x, bool): return x and y
    else: return x * y
def _f_div(x, y):
    if isinstance(x, float) or isinstance(x, float32): return x.cdiv(y)
    else: return x / y
def _f_eq(x, y): return x == y
def _f_ne(x, y): return x != y
def _f_lt(x, y): return x < y
def _f_le(x, y): return x <= y
def _f_gt(x, y): return x > y
def _f_ge(x, y): return x >= y
def _f_max(x, y):
    if isinstance(x, float) or isinstance(x, float32):
        if x != x: return x
        if y != y: return y
    return x if x > y else y
def _f_min(x, y):
    if isinstance(x, float) or isinstance(x, float32):
        if x != x: return x
        if y != y: return y
    return x if x < y else y
def _f_pow(x, y):
    if isinstance(x, float) or isinstance(x, float32):
        return _U.pow(x, y)
    elif isinstance(x, bool):
        return x or not y
    else:
        if y < type(y)(0):
            return type(x)(0) if x != type(x)(1) else x
        r = type(x)(1)
        e = y
        bb = x
        while e > type(y)(0):
            if e & type(y)(1):
                r *= bb
            bb *= bb
            e >>= type(y)(1)
        return r
def _f_floordiv(x, y):
    if isinstance(x, float) or isinstance(x, float32):
        return _U.floor(x.cdiv(y))
    else:
        q = x // y
        return q
def _f_rem(x, y):
    if isinstance(x, float) or isinstance(x, float32):
        m = x - _U.floor(x.cdiv(y)) * y
        return m
    else:
        return x % y
def _f_and(x, y):
    if isinstance(x, float) or isinstance(x, float32): return x
    else: return x & y
def _f_or(x, y):
    if isinstance(x, float) or isinstance(x, float32): return x
    else: return x | y
def _f_xor(x, y):
    if isinstance(x, float) or isinstance(x, float32): return x
    else: return x ^ y

def _bool_add_dt(dt: int) -> int:
    return dt

def add(input, other, alpha = 1, out = None) -> Tensor:
    a = _as_t(input, _scalar_tensor(0, _I64))
    b = _as_t(other, a)
    if _is_complex(a._dt) or _is_complex(b._dt):
        return _cbinary(a, mul(b, alpha) if alpha != 1 else b, "add")
    dt = _result_dt(a, b)
    if alpha != 1:
        b2 = mul(b, alpha)
        r = _binary(a, b2.detach(), _f_add, dt)
        if _needs([a, b]):
            nd = _node("AddBackward0", _K_ADD, [a, b])
            nd.fl = float(alpha)
            _attach(r, nd)
        return r
    r = _binary(a, b, _f_add, dt)
    if _needs([a, b]):
        nd = _node("AddBackward0", _K_ADD, [a, b])
        nd.fl = 1.0
        _attach(r, nd)
    return r

def sub(input, other, alpha = 1, out = None) -> Tensor:
    a = _as_t(input, _scalar_tensor(0, _I64))
    b = _as_t(other, a)
    if _is_complex(a._dt) or _is_complex(b._dt):
        return _cbinary(a, mul(b, alpha) if alpha != 1 else b, "sub")
    dt = _result_dt(a, b)
    if dt == _BOOL:
        raise RuntimeError("Subtraction, the `-` operator, with two bool tensors is not supported.")
    bb = b if alpha == 1 else mul(b.detach(), alpha)
    r = _binary(a, bb, _f_sub, dt)
    if _needs([a, b]):
        nd = _node("SubBackward0", _K_SUB, [a, b])
        nd.fl = float(alpha)
        _attach(r, nd)
    return r

def rsub(input, other, alpha = 1) -> Tensor:
    a = _as_t(input, _scalar_tensor(0, _I64))
    return sub(_as_t(other, a), a, alpha)

def subtract(*args, **kwargs):
    return sub(*args, **kwargs)

def mul(input, other, out = None) -> Tensor:
    a = _as_t(input, _scalar_tensor(0, _I64))
    b = _as_t(other, a)
    if _is_complex(a._dt) or _is_complex(b._dt):
        return _cbinary(a, b, "mul")
    dt = _result_dt(a, b)
    r = _binary(a, b, _f_mul, dt)
    if _needs([a, b]):
        _attach(r, _node("MulBackward0", _K_MUL, [a, b]))
    return r

def multiply(*args, **kwargs):
    return mul(*args, **kwargs)

def div(input, other, rounding_mode: Optional[str] = None, out = None) -> Tensor:
    a = _as_t(input, _scalar_tensor(0, _I64))
    b = _as_t(other, a)
    if _is_complex(a._dt) or _is_complex(b._dt):
        return _cbinary(a, b, "div")
    dt = _result_dt(a, b)
    if rounding_mode is not None:
        rm = rounding_mode
        if rm == "floor":
            return floor_divide(a, b)
        # trunc
        if _is_float(dt):
            return trunc(div(a, b))
        return _binary(a, b, _f_div_trunc_int, dt)
    if not _is_float(dt):
        dt = _F32
    if True:
        r = _binary(a, b, _f_div, dt)
    if _needs([a, b]):
        _attach(r, _node("DivBackward0", _K_DIV, [a, b]))
    return r

def _f_div_trunc_int(x, y):
    q = x // y
    if (x % y != type(x)(0)) and ((x < type(x)(0)) != (y < type(y)(0))):
        q += type(x)(1)
    return q

def divide(*args, **kwargs):
    return div(*args, **kwargs)
def true_divide(*args, **kwargs):
    return div(*args, **kwargs)

def floor_divide(input, other) -> Tensor:
    a = _as_t(input, _scalar_tensor(0, _I64))
    b = _as_t(other, a)
    dt = _result_dt(a, b)
    return _binary(a, b, _f_floordiv, dt)

def remainder(input, other) -> Tensor:
    a = _as_t(input, _scalar_tensor(0, _I64))
    b = _as_t(other, a)
    dt = _result_dt(a, b)
    return _binary(a, b, _f_rem, dt)

def fmod(input, other) -> Tensor:
    a = _as_t(input, _scalar_tensor(0, _I64))
    b = _as_t(other, a)
    dt = _result_dt(a, b)
    return _binary(a, b, _f_fmod, dt)

def _f_fmod(x, y):
    if isinstance(x, float) or isinstance(x, float32):
        return x - _U.trunc(x.cdiv(y)) * y
    else:
        r = x % y
        if r != type(x)(0) and ((r < type(x)(0)) != (x < type(x)(0))):
            r -= y
        return r

def pow(input, exponent) -> Tensor:
    if isinstance(input, Tensor) and not isinstance(exponent, Tensor):
        a = input
        e = _scalar_like(exponent, a)
        dt = _result_dt(a, e)
        if (isinstance(exponent, float) or isinstance(exponent, int)) and _is_float(dt):
            ev = float(exponent)
            if ev == 2.0:
                r = _binary(a, a, _f_mul, dt)
            elif ev == 1.0:
                r = _contig_as(a, dt).clone().detach() if a._dt == dt else _to_dtype(a.detach(), dt)
            elif ev == 0.5:
                r = _unary_rawf(_contig_as(a, dt), _U.sqrt, dt)
            else:
                r = _binary(a, e, _f_pow, dt)
        else:
            r = _binary(a, e, _f_pow, dt)
        if _needs([a]):
            nd = _node("PowBackward0", _K_POWS, [a])
            nd.fl = float(exponent)
            _attach(r, nd)
        return r
    elif isinstance(input, Tensor):
        a = input
        b = exponent
        dt = _result_dt(a, b)
        r = _binary(a, b, _f_pow, dt)
        if _needs([a, b]):
            nd = _node("PowBackward1", _K_POW, [a, b])
            nd.saved = [r.detach()]
            _attach(r, nd)
        return r
    else:
        b = exponent
        a = _scalar_like(input, b)
        dt = _result_dt(a, b)
        r = _binary(a, b, _f_pow, dt)
        if _needs([b]):
            nd = _node("PowBackward2", _K_RPOWS, [b])
            nd.fl = float(input)
            nd.saved = [r.detach()]
            _attach(r, nd)
        return r

def float_power(*args, **kwargs):
    return pow(*args, **kwargs)

def maximum(input: Tensor, other: Tensor) -> Tensor:
    dt = _result_dt(input, other)
    r = _binary(input, other, _f_max, dt)
    if _needs([input, other]):
        _attach(r, _node("MaximumBackward0", _K_MAXIMUM, [input, other]))
    return r

def minimum(input: Tensor, other: Tensor) -> Tensor:
    dt = _result_dt(input, other)
    r = _binary(input, other, _f_min, dt)
    if _needs([input, other]):
        _attach(r, _node("MinimumBackward0", _K_MINIMUM, [input, other]))
    return r

def fmax(*args, **kwargs):
    return maximum(*args, **kwargs)
def fmin(*args, **kwargs):
    return minimum(*args, **kwargs)

def eq(input, other) -> Tensor: return _compare(input, other, _f_eq)
def ne(input, other) -> Tensor: return _compare(input, other, _f_ne)
def lt(input, other) -> Tensor: return _compare(input, other, _f_lt)
def le(input, other) -> Tensor: return _compare(input, other, _f_le)
def gt(input, other) -> Tensor: return _compare(input, other, _f_gt)
def ge(input, other) -> Tensor: return _compare(input, other, _f_ge)
def greater(*args, **kwargs):
    return gt(*args, **kwargs)
def greater_equal(*args, **kwargs):
    return ge(*args, **kwargs)
def less(*args, **kwargs):
    return lt(*args, **kwargs)
def less_equal(*args, **kwargs):
    return le(*args, **kwargs)
def not_equal(*args, **kwargs):
    return ne(*args, **kwargs)

def _bitop(a: Tensor, o, which: int) -> Tensor:
    b = _as_t(o, a)
    dt = _result_dt(a, b)
    if _is_float(dt):
        raise RuntimeError("bitwise ops are only supported for integer and bool tensors")
    if which == 0:
        return _binary(a, b, _f_and, dt)
    elif which == 1:
        return _binary(a, b, _f_or, dt)
    return _binary(a, b, _f_xor, dt)

def logical_and(input: Tensor, other: Tensor) -> Tensor:
    return _bitop(input.bool(), other.bool(), 0)

def logical_or(input: Tensor, other: Tensor) -> Tensor:
    return _bitop(input.bool(), other.bool(), 1)

def logical_xor(input: Tensor, other: Tensor) -> Tensor:
    return _bitop(input.bool(), other.bool(), 2)

def bitwise_and(input: Tensor, other) -> Tensor: return _bitop(input, other, 0)
def bitwise_or(input: Tensor, other) -> Tensor: return _bitop(input, other, 1)
def bitwise_xor(input: Tensor, other) -> Tensor: return _bitop(input, other, 2)

def _k_not(z, a: Tensor, out: Tensor):
    T = type(z)
    pa = Ptr[T](a._st.ptr) + a._off
    po = Ptr[bool](out._st.ptr)
    for i in range(out.numel()):
        po[i] = pa[i] == T()

def logical_not(input: Tensor) -> Tensor:
    c = input.contiguous()
    out = _new(c._shape.copy(), _BOOL)
    _dispatch(c._dt, _k_not, c, out)
    return out

def bitwise_not(input: Tensor) -> Tensor:
    if input._dt == _BOOL:
        return logical_not(input)
    if _is_float(input._dt):
        raise RuntimeError("bitwise_not is only supported for integer and bool tensors")
    return _unary_raw(input.contiguous(), _f_bnot, input._dt)

def _f_bnot(x):
    if isinstance(x, float) or isinstance(x, float32): return x
    else: return ~x

# ================================================================ complex

def view_as_real(input: Tensor) -> Tensor:
    if not _is_complex(input._dt):
        raise RuntimeError("view_as_real is only supported for complex tensors")
    r = _as_real_raw(input)
    if _needs([input]):
        _attach(r, _node("ViewAsRealBackward0", _K_VAR, [input]))
    return r

def view_as_complex(input: Tensor) -> Tensor:
    if not _is_float(input._dt):
        raise RuntimeError("view_as_complex is only supported for float and double tensors, but got a tensor of scalar type: " + input.dtype._name())
    x = input
    if len(x._shape) == 0 or x._shape[-1] != 2 or x._strides[-1] != 1 or x._off % 2 != 0:
        x = clone(x) if len(x._shape) > 0 and x._shape[-1] == 2 else x
        if len(x._shape) == 0 or x._shape[-1] != 2:
            raise RuntimeError("Tensor must have a last dimension of size 2")
    for st in x._strides[:-1]:
        if st % 2 != 0:
            x = clone(x)
            break
    r = Tensor(x._st, x._off // 2, x._shape[:-1], [st // 2 for st in x._strides[:-1]], _complex_code(x._dt))
    if _needs([x]):
        _attach(r, _node("ViewAsComplexBackward0", _K_VAC, [x]))
    return r

def complex(real: Tensor, imag: Tensor) -> Tensor:
    return view_as_complex(stack([real, imag], -1))

def polar(abs: Tensor, angle: Tensor) -> Tensor:
    return complex(mul(abs, cos(angle)), mul(abs, sin(angle)))

def is_complex(input: Tensor) -> bool:
    return _is_complex(input._dt)

def real(input: Tensor) -> Tensor:
    if not _is_complex(input._dt):
        return input
    return select(view_as_real(input), -1, 0)

def imag(input: Tensor) -> Tensor:
    if not _is_complex(input._dt):
        raise RuntimeError("imag is not implemented for tensors with non-complex dtypes.")
    return select(view_as_real(input), -1, 1)

def angle(input: Tensor) -> Tensor:
    if not _is_complex(input._dt):
        z = zeros_like(input)
        pi_ = full_like(input, 3.141592653589793)
        return _where3(lt(input, 0), pi_, z)
    return atan2(imag(input), real(input))

def conj(input: Tensor) -> Tensor:
    if not _is_complex(input._dt):
        return input
    return complex(real(input), neg(imag(input)))

def _to_cplx(x: Tensor, dt: int) -> Tensor:
    if _is_complex(x._dt):
        if x._dt == dt:
            return x
        return complex(real(x).to(_DType(_real_code(dt))), imag(x).to(_DType(_real_code(dt))))
    xr = x.to(_DType(_real_code(dt)))
    return complex(xr, zeros_like(xr.detach()))

def _cbinary(a: Tensor, b: Tensor, op: str) -> Tensor:
    dt = _promote2(a._dt, b._dt)
    if not _is_complex(dt):
        dt = _complex_code(dt)
    if not _is_complex(b._dt) and (op == "mul" or op == "div"):
        ca = _to_cplx(a, dt)
        bb = b.to(_DType(_real_code(dt)))
        ar = view_as_real(ca)
        return view_as_complex(mul(ar, unsqueeze(bb, -1)) if op == "mul" else div(ar, unsqueeze(bb, -1)))
    ca = _to_cplx(a, dt)
    cb = _to_cplx(b, dt)
    ar = real(ca)
    ai = imag(ca)
    br = real(cb)
    bi = imag(cb)
    if op == "add":
        return complex(add(ar, br), add(ai, bi))
    if op == "sub":
        return complex(sub(ar, br), sub(ai, bi))
    if op == "mul":
        return complex(sub(mul(ar, br), mul(ai, bi)), add(mul(ar, bi), mul(ai, br)))
    den = add(mul(br, br), mul(bi, bi))
    return complex(div(add(mul(ar, br), mul(ai, bi)), den), div(sub(mul(ai, br), mul(ar, bi)), den))

# ---------------------------------------------------------------- unary

def _k_map(z, x: Tensor, out: Tensor, f):
    T = type(z)
    p = Ptr[T](x._st.ptr) + x._off
    q = Ptr[T](out._st.ptr)
    for i in range(out.numel()):
        q[i] = T(f(p[i]))

def _unary_raw(x: Tensor, f, dt: int) -> Tensor:
    c = _contig_as(x, dt)
    out = _new(c._shape.copy(), dt)
    _dispatch(dt, _k_map, c, out, f)
    return out

def _unary_rawf(x: Tensor, f, dt: int) -> Tensor:
    c = _contig_as(x, dt)
    out = _new(c._shape.copy(), dt)
    _dispatch_float(dt, _k_map, c, out, f)
    return out

def _float_dt(dt: int) -> int:
    return dt if _is_float(dt) else _F32

def _unary(x: Tensor, f, u: int, name: str, keep_dtype: Literal[bool] = False) -> Tensor:
    if keep_dtype:
        dt = x._dt
        r = _unary_raw(x, f, dt)
    else:
        dt = _float_dt(x._dt)
        r = _unary_rawf(x, f, dt)
    if _needs([x]):
        nd = _node(name, _K_UNARY, [x])
        nd.ints = [u]
        nd.saved = [r.detach()]
        _attach(r, nd)
    return r

def _f_neg(x):
    if isinstance(x, bool): return x
    else: return -x

def neg(input: Tensor) -> Tensor:
    if input._dt == _BOOL:
        raise RuntimeError("Negation, the `-` operator, on a bool tensor is not supported.")
    r = _unary_raw(input, _f_neg, input._dt)
    if _needs([input]):
        _attach(r, _node("NegBackward0", _K_NEG, [input]))
    return r

def negative(*args, **kwargs):
    return neg(*args, **kwargs)

def _f_sigmoid(x):
    one = type(x)(1)
    return one.cdiv(one + _U.exp(-x))
def _f_relu(x):
    if isinstance(x, float) or isinstance(x, float32):
        if x != x: return x
    return x if x > type(x)(0) else type(x)(0)
def _f_abs(x):
    if isinstance(x, bool): return x
    elif isinstance(x, float) or isinstance(x, float32): return _U.fabs(x)
    else: return x if x >= type(x)(0) else -x
def _f_sign(x):
    if isinstance(x, bool): return x
    elif x > type(x)(0): return type(x)(1)
    elif x < type(x)(0): return type(x)(-1)
    else: return x
def _f_rsqrt(x): return type(x)(1).cdiv(_U.sqrt(x))
def _f_recip(x): return type(x)(1).cdiv(x)
def _f_square(x): return x * x
def _f_round(x): return _U.rint(x)
def _f_frac(x): return x - _U.trunc(x)

def exp(input: Tensor) -> Tensor: return _unary(input, _U.exp, _U_EXP, "ExpBackward0")
def log(input: Tensor) -> Tensor: return _unary(input, _U.log, _U_LOG, "LogBackward0")
def log1p(input: Tensor) -> Tensor: return _unary(input, _U.log1p, _U_LOG1P, "Log1PBackward0")
def log2(input: Tensor) -> Tensor: return _unary(input, _U.log2, _U_LOG2, "Log2Backward0")
def log10(input: Tensor) -> Tensor: return _unary(input, _U.log10, _U_LOG10, "Log10Backward0")
def expm1(input: Tensor) -> Tensor: return _unary(input, _U.expm1, _U_EXPM1, "Expm1Backward0")
def sqrt(input: Tensor) -> Tensor: return _unary(input, _U.sqrt, _U_SQRT, "SqrtBackward0")
def rsqrt(input: Tensor) -> Tensor: return _unary(input, _f_rsqrt, _U_RSQRT, "RsqrtBackward0")
def sin(input: Tensor) -> Tensor: return _unary(input, _U.sin, _U_SIN, "SinBackward0")
def cos(input: Tensor) -> Tensor: return _unary(input, _U.cos, _U_COS, "CosBackward0")
def tan(input: Tensor) -> Tensor: return _unary(input, _U.tan, _U_TAN, "TanBackward0")
def tanh(input: Tensor) -> Tensor: return _unary(input, _U.tanh, _U_TANH, "TanhBackward0")
def sinh(input: Tensor) -> Tensor: return _unary(input, _U.sinh, _U_SINH, "SinhBackward0")
def cosh(input: Tensor) -> Tensor: return _unary(input, _U.cosh, _U_COSH, "CoshBackward0")
def asin(input: Tensor) -> Tensor: return _unary(input, _U.asin, _U_ASIN, "AsinBackward0")
def acos(input: Tensor) -> Tensor: return _unary(input, _U.acos, _U_ACOS, "AcosBackward0")
def atan(input: Tensor) -> Tensor: return _unary(input, _U.atan, _U_ATAN, "AtanBackward0")
def sigmoid(input: Tensor) -> Tensor: return _unary(input, _f_sigmoid, _U_SIGMOID, "SigmoidBackward0")
def erf(input: Tensor) -> Tensor: return _unary(input, _erf, _U_ERF, "ErfBackward0")
def erfc(input: Tensor) -> Tensor: return _unary(input, _erfc, _U_ERFC, "ErfcBackward0")
def lgamma(input: Tensor) -> Tensor: return _unary(input, _lgamma, _U_ZERO, "LgammaBackward0")
def reciprocal(input: Tensor) -> Tensor: return _unary(input, _f_recip, _U_RECIP, "ReciprocalBackward0")
def arcsin(*args, **kwargs):
    return asin(*args, **kwargs)
def arccos(*args, **kwargs):
    return acos(*args, **kwargs)
def arctan(*args, **kwargs):
    return atan(*args, **kwargs)

def relu(input: Tensor) -> Tensor:
    return _unary(input, _f_relu, _U_RELU, "ReluBackward0", True)

def abs(input: Tensor) -> Tensor:
    if _is_complex(input._dt):
        r = view_as_real(input)
        re = select(r, -1, 0)
        im = select(r, -1, 1)
        return sqrt(add(mul(re, re), mul(im, im)))
    return _unary(input, _f_abs, _U_ABS, "AbsBackward0", True)

def absolute(*args, **kwargs):
    return abs(*args, **kwargs)

def square(input: Tensor) -> Tensor:
    return _unary(input, _f_square, _U_SQUARE, "PowBackward0", True)

def sign(input: Tensor) -> Tensor:
    return _unary(input, _f_sign, _U_ZERO, "SignBackward0", True)

def sgn(*args, **kwargs):
    return sign(*args, **kwargs)

def _int_passthrough(input: Tensor, f, name: str) -> Tensor:
    if not _is_float(input._dt):
        return input.clone()
    return _unary(input, f, _U_ZERO, name, False)

def floor(input: Tensor) -> Tensor: return _int_passthrough(input, _U.floor, "FloorBackward0")
def ceil(input: Tensor) -> Tensor: return _int_passthrough(input, _U.ceil, "CeilBackward0")
def trunc(input: Tensor) -> Tensor: return _int_passthrough(input, _U.trunc, "TruncBackward0")
def fix(*args, **kwargs):
    return trunc(*args, **kwargs)

def round(input: Tensor, decimals: int = 0) -> Tensor:
    if decimals == 0:
        return _int_passthrough(input, _f_round, "RoundBackward0")
    s = 10.0 ** decimals
    return div(round(mul(input, s)), s)

def frac(input: Tensor) -> Tensor:
    r = _unary_rawf(input, _f_frac, _float_dt(input._dt))
    if _needs([input]):
        _attach(r, _node("FracBackward0", _K_IDENT, [input]))
    return r

def _k_isnan(z, x: Tensor, out: Tensor, which: int):
    T = type(z)
    p = Ptr[T](x._st.ptr) + x._off
    q = Ptr[bool](out._st.ptr)
    for i in range(out.numel()):
        v = p[i]
        if isinstance(v, float) or isinstance(v, float32):
            if which == 0:
                q[i] = v != v
            elif which == 1:
                q[i] = (v == v) and (v - v != v - v)
            else:
                q[i] = v - v == v - v
        else:
            q[i] = which == 2

def _classify(x: Tensor, which: int) -> Tensor:
    c = x.contiguous()
    out = _new(c._shape.copy(), _BOOL)
    _dispatch(c._dt, _k_isnan, c, out, which)
    return out

def isnan(input: Tensor) -> Tensor: return _classify(input, 0)
def isinf(input: Tensor) -> Tensor: return _classify(input, 1)
def isfinite(input: Tensor) -> Tensor: return _classify(input, 2)

def isposinf(input: Tensor) -> Tensor: return logical_and(isinf(input), gt(input, 0))
def isneginf(input: Tensor) -> Tensor: return logical_and(isinf(input), lt(input, 0))

def nan_to_num(input: Tensor, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None) -> Tensor:
    if not _is_float(input._dt):
        return input.clone()
    big = 3.4028234663852886e38 if input._dt == _F32 else 1.7976931348623157e308
    pi = posinf if posinf is not None else big
    ni = neginf if neginf is not None else -big
    r = where(isnan(input), _scalar_tensor(nan, input._dt), input)
    r = where(isposinf(input), _scalar_tensor(pi, input._dt), r)
    r = where(isneginf(input), _scalar_tensor(ni, input._dt), r)
    return r

def clamp(input: Tensor, min = None, max = None) -> Tensor:
    x = input
    r = x
    dt = x._dt
    mask = ones_like(x)
    if min is not None:
        mn = _as_t(min, x)
        dt = _result_dt(x, mn)
        r = _binary(r, mn, _f_max, dt)
        # torch: scalar bounds give zero gradient at the boundary
        if isinstance(min, Tensor):
            mask = mul(mask, ge(x.detach(), mn))
        else:
            mask = mul(mask, gt(x.detach(), mn))
    if max is not None:
        mx = _as_t(max, x)
        dt = _result_dt(r, mx)
        r = _binary(r, mx, _f_min, dt)
        if isinstance(max, Tensor):
            mask = mul(mask, le(x.detach(), mx))
        else:
            mask = mul(mask, lt(x.detach(), mx))
    if min is None and max is None:
        raise RuntimeError("torch.clamp: At least one of 'min' or 'max' must not be None")
    if _needs([x]):
        nd = _node("ClampBackward1", _K_CLAMP, [x])
        nd.saved = [_contig_as(mask, _float_dt(r._dt))]
        _attach(r, nd)
    return r

def clip(*args, **kwargs):
    return clamp(*args, **kwargs)

def clamp_min(input: Tensor, min) -> Tensor:
    return clamp(input, min, None)

def clamp_max(input: Tensor, max) -> Tensor:
    return clamp(input, None, max)

def where(condition: Tensor, input = None, other = None):
    if input is None:
        return unbind(nonzero(condition), 1)
    else:
        return _where3(condition, input, other)

def _where3(condition: Tensor, input, other) -> Tensor:
    if not isinstance(input, Tensor) and not isinstance(other, Tensor):
        a0 = _scalar_like(input, condition)
        return _where3(condition, a0, _scalar_like(other, a0))
    a = _as_t(input, condition) if isinstance(input, Tensor) else _scalar_like(input, _unw_t(other))
    b = _as_t(other, a)
    dt = _result_dt(a, b)
    shape = _broadcast_shapes(_broadcast_shapes(condition._shape, a._shape), b._shape)
    c = _bcast_contig(condition, shape, _BOOL)
    ca = _bcast_contig(a, shape, dt)
    cb = _bcast_contig(b, shape, dt)
    out = _new(shape, dt)
    _dispatch(dt, _k_where, c, ca, cb, out)
    if _needs([a, b]):
        nd = _node("WhereBackward0", _K_WHERE, [a, b])
        nd.saved = [c]
        _attach(out, nd)
    return out

def _unw_t(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    else:
        return _scalar_like(x, _scalar_tensor(0, _I64))


def _k_where(z, c: Tensor, a: Tensor, b: Tensor, out: Tensor):
    T = type(z)
    pc = Ptr[bool](c._st.ptr) + c._off
    pa = Ptr[T](a._st.ptr) + a._off
    pb = Ptr[T](b._st.ptr) + b._off
    po = Ptr[T](out._st.ptr)
    for i in range(out.numel()):
        po[i] = pa[i] if pc[i] else pb[i]

def masked_fill(input: Tensor, mask: Tensor, value) -> Tensor:
    if mask._dt != _BOOL:
        raise RuntimeError("masked_fill only supports boolean masks")
    v = _scalar_tensor(float(value) if isinstance(value, Tensor) else value, input._dt)
    c = _bcast_contig(mask, input._shape, _BOOL)
    ci = _contig_as(input, input._dt)
    out = _new(input._shape.copy(), input._dt)
    vv = _contig_as(_expand_raw(v, input._shape), input._dt)
    _dispatch(input._dt, _k_where, c, vv, ci, out)
    if _needs([input]):
        nd = _node("MaskedFillBackward0", _K_MASKED_FILL, [input])
        nd.saved = [c]
        _attach(out, nd)
    return out

# ================================================================ shape ops

def _infer_shape(shape: List[int], n: int) -> List[int]:
    s = shape.copy()
    neg_i = -1
    p = 1
    for i in range(len(s)):
        if s[i] == -1:
            if neg_i >= 0:
                raise RuntimeError("only one dimension can be inferred")
            neg_i = i
        else:
            p *= s[i]
    if neg_i >= 0:
        if p == 0 or n % p != 0:
            raise RuntimeError("shape '" + str(shape) + "' is invalid for input of size " + str(n))
        s[neg_i] = n // p
    elif p != n:
        raise RuntimeError("shape '" + str(shape) + "' is invalid for input of size " + str(n))
    return s

# strides for viewing (shape, strides) as new shape without copying, or None
def _view_strides(oshape: List[int], ostrides: List[int], nshape: List[int]) -> Optional[List[int]]:
    # port of at::detail::computeStride
    numel = _numel(oshape)
    nst = [0] * len(nshape)
    if numel == 0:
        if oshape == nshape:
            return ostrides.copy()
        return _contig_strides(nshape)
    view_d = len(nshape) - 1
    chunk_base_stride = ostrides[-1] if len(ostrides) > 0 else 1
    tensor_numel = 1
    view_numel = 1
    tensor_d = len(oshape) - 1
    while tensor_d >= 0:
        tensor_numel *= oshape[tensor_d]
        if tensor_d == 0 or (oshape[tensor_d - 1] != 1 and ostrides[tensor_d - 1] != tensor_numel * chunk_base_stride):
            while view_d >= 0 and (view_numel < tensor_numel or nshape[view_d] == 1):
                nst[view_d] = view_numel * chunk_base_stride
                view_numel *= nshape[view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                return None
            if tensor_d > 0:
                chunk_base_stride = ostrides[tensor_d - 1]
                tensor_numel = 1
                view_numel = 1
        tensor_d -= 1
    if view_d != -1:
        return None
    return nst

def view(input: Tensor, shape: List[int]) -> Tensor:
    s = _infer_shape(shape, input.numel())
    st = _view_strides(input._shape, input._strides, s)
    if st is None:
        raise RuntimeError("view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.")
    r = Tensor(input._st, input._off, s, _unw(st), input._dt)
    if _needs([input]):
        nd = _node("ViewBackward0", _K_VIEW, [input])
        nd.ints = input._shape.copy()
        _attach(r, nd)
    return r

def reshape(input: Tensor, shape) -> Tensor:
    sh = _to_list(shape)
    s = _infer_shape(sh, input.numel())
    st = _view_strides(input._shape, input._strides, s)
    if st is not None:
        r = Tensor(input._st, input._off, s, _unw(st), input._dt)
    else:
        c = _new(input._shape.copy(), input._dt)
        _copy_into(c, input)
        r = Tensor(c._st, 0, s, _contig_strides(s), input._dt)
    if _needs([input]):
        nd = _node("ViewBackward0" if st is not None else "UnsafeViewBackward0", _K_VIEW, [input])
        nd.ints = input._shape.copy()
        _attach(r, nd)
    return r

def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    nd = len(input._shape)
    if nd == 0:
        return reshape(input, [1])
    s = _wrap(start_dim, nd)
    e = _wrap(end_dim, nd)
    if s > e:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")
    if s == e:
        return input
    shp = List[int]()
    for i in range(s):
        shp.append(input._shape[i])
    p = 1
    for i in range(s, e + 1):
        p *= input._shape[i]
    shp.append(p)
    for i in range(e + 1, nd):
        shp.append(input._shape[i])
    return reshape(input, shp)

def unflatten(input: Tensor, dim: int, sizes) -> Tensor:
    d = _wrap(dim, len(input._shape))
    sz = _infer_shape(_to_list(sizes), input._shape[d])
    shp = input._shape[:d] + sz + input._shape[d + 1:]
    return view(input, shp) if input.is_contiguous() else reshape(input, shp)

def ravel(input: Tensor) -> Tensor:
    return reshape(input, [-1])

def permute(input: Tensor, dims) -> Tensor:
    p = _to_list(dims)
    nd = len(input._shape)
    if len(p) != nd:
        raise RuntimeError("permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = " + str(nd) + " is not equal to len(dims) = " + str(len(p)))
    for i in range(nd):
        p[i] = _wrap(p[i], nd)
    shp = [input._shape[i] for i in p]
    st = [input._strides[i] for i in p]
    r = Tensor(input._st, input._off, shp, st, input._dt)
    if _needs([input]):
        n = _node("PermuteBackward0", _K_PERMUTE, [input])
        n.ints = p
        _attach(r, n)
    return r

def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    nd = len(input._shape)
    if nd == 0:
        return input
    a = _wrap(dim0, nd)
    b = _wrap(dim1, nd)
    p = list(range(nd))
    p[a] = b
    p[b] = a
    r = permute(input, p)
    if r.grad_fn is not None:
        _unw(r.grad_fn)._name = "TransposeBackward0"
    return r

def swapaxes(*args, **kwargs):
    return transpose(*args, **kwargs)
def swapdims(*args, **kwargs):
    return transpose(*args, **kwargs)

def t(input: Tensor) -> Tensor:
    if len(input._shape) < 2:
        return input
    return transpose(input, 0, 1)

def adjoint(input: Tensor) -> Tensor:
    return transpose(input, -2, -1)

def movedim(input: Tensor, source, destination) -> Tensor:
    nd = len(input._shape)
    src = [_wrap(v, nd) for v in _to_list(source)]
    dst = [_wrap(v, nd) for v in _to_list(destination)]
    order = [-1] * nd
    used = [False] * nd
    for i in range(len(src)):
        order[dst[i]] = src[i]
        used[src[i]] = True
    rest = [i for i in range(nd) if not used[i]]
    k = 0
    for i in range(nd):
        if order[i] == -1:
            order[i] = rest[k]
            k += 1
    return permute(input, order)

def moveaxis(*args, **kwargs):
    return movedim(*args, **kwargs)

def unsqueeze(input: Tensor, dim: int) -> Tensor:
    nd = len(input._shape)
    d = _wrap(dim, nd + 1)
    shp = input._shape.copy()
    st = input._strides.copy()
    ns = st[d] * shp[d] if d < nd else 1
    shp.insert(d, 1)
    st.insert(d, ns)
    r = Tensor(input._st, input._off, shp, st, input._dt)
    if _needs([input]):
        n = _node("UnsqueezeBackward0", _K_VIEW, [input])
        n.ints = input._shape.copy()
        _attach(r, n)
    return r

def squeeze(input: Tensor, dim = None) -> Tensor:
    nd = len(input._shape)
    keep = [True] * nd
    if dim is None:
        for i in range(nd):
            if input._shape[i] == 1:
                keep[i] = False
    else:
        if nd == 0:
            return input
        for d0 in _to_list(dim):
            d = _wrap(d0, nd)
            if input._shape[d] == 1:
                keep[d] = False
    shp = List[int]()
    st = List[int]()
    for i in range(nd):
        if keep[i]:
            shp.append(input._shape[i])
            st.append(input._strides[i])
    r = Tensor(input._st, input._off, shp, st, input._dt)
    if _needs([input]):
        n = _node("SqueezeBackward0", _K_VIEW, [input])
        n.ints = input._shape.copy()
        _attach(r, n)
    return r

def expand(input: Tensor, shape) -> Tensor:
    sh = _to_list(shape)
    r = _expand_raw(input, sh)
    if _needs([input]):
        n = _node("ExpandBackward0", _K_EXPAND, [input])
        n.ints = input._shape.copy()
        _attach(r, n)
    return r

def broadcast_to(input: Tensor, size) -> Tensor:
    return expand(input, size)

def broadcast_tensors(*tensors) -> List[Tensor]:
    shp = List[int]()
    for x in tensors:
        shp = _broadcast_shapes(shp, x._shape)
    return [expand(x, shp) for x in tensors]

def select(input: Tensor, dim: int, index: int) -> Tensor:
    nd = len(input._shape)
    d = _wrap(dim, nd)
    n = input._shape[d]
    i = index
    if i < -n or i >= n:
        raise IndexError("index " + str(index) + " is out of bounds for dimension " + str(d) + " with size " + str(n))
    if i < 0:
        i += n
    shp = input._shape.copy()
    st = input._strides.copy()
    off = input._off + i * st[d]
    shp.pop(d)
    st.pop(d)
    r = Tensor(input._st, off, shp, st, input._dt)
    if _needs([input]):
        nd2 = _node("SelectBackward0", _K_SELECT, [input])
        nd2.ints = input._shape.copy()
        nd2.ints2 = [d, i]
        _attach(r, nd2)
    return r

# slice along dim with python semantics (start/stop already resolved)
def _slice(input: Tensor, d: int, start: int, stop: int, step: int) -> Tensor:
    n = input._shape[d]
    length = 0
    if step > 0 and stop > start:
        length = (stop - start + step - 1) // step
    shp = input._shape.copy()
    st = input._strides.copy()
    off = input._off + (start * st[d] if length > 0 else 0)
    shp[d] = length
    st[d] = st[d] * step
    r = Tensor(input._st, off, shp, st, input._dt)
    if _needs([input]):
        nd2 = _node("SliceBackward0", _K_SLICE, [input])
        nd2.ints = input._shape.copy()
        nd2.ints2 = [d, start, stop, step]
        _attach(r, nd2)
    return r

def narrow(input: Tensor, dim: int, start: int, length: int) -> Tensor:
    d = _wrap(dim, len(input._shape))
    s = start if start >= 0 else start + input._shape[d]
    if s < 0 or s + length > input._shape[d] or length < 0:
        raise RuntimeError("start (" + str(start) + ") + length (" + str(length) + ") exceeds dimension size (" + str(input._shape[d]) + ").")
    return _slice(input, d, s, s + length, 1)

def _unfold(input: Tensor, dimension: int, size: int, step: int) -> Tensor:
    nd = len(input._shape)
    d = _wrap(dimension, nd) if nd > 0 else 0
    n = input._shape[d] if nd > 0 else 1
    if size > n:
        raise RuntimeError("maximum size for tensor at dimension " + str(d) + " is " + str(n) + " but size is " + str(size))
    shp = input._shape.copy()
    st = input._strides.copy()
    nw = (n - size) // step + 1
    sd = st[d] if nd > 0 else 1
    if nd > 0:
        shp[d] = nw
        st[d] = sd * step
    shp.append(size)
    st.append(sd)
    r = Tensor(input._st, input._off, shp, st, input._dt)
    if _needs([input]):
        nd2 = _node("UnfoldBackward0", _K_UNFOLD, [input])
        nd2.ints = input._shape.copy()
        nd2.ints2 = [d, size, step]
        _attach(r, nd2)
    return r

def cat(tensors, dim: int = 0) -> Tensor:
    if not isinstance(tensors, List[Tensor]):
        # `torch.cat((a, b, c), dim=1)`: a tuple (or other sequence) of tensors
        return cat(List[Tensor]([t for t in tensors]), dim)
    if len(tensors) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    # legacy: skip 1-d empty tensors
    ts = List[Tensor]()
    for x in tensors:
        if not (len(x._shape) == 1 and x._shape[0] == 0):
            ts.append(x)
    if len(ts) == 0:
        return tensors[0]
    ref = ts[0]
    nd = len(ref._shape)
    d = _wrap(dim, nd)
    dt = ref._dt
    for x in ts:
        dt = _promote2(dt, x._dt)
        if len(x._shape) != nd:
            raise RuntimeError("Tensors must have same number of dimensions: got " + str(nd) + " and " + str(len(x._shape)))
    shp = ref._shape.copy()
    tot = 0
    for x in ts:
        for i in range(nd):
            if i != d and x._shape[i] != shp[i]:
                raise RuntimeError("Sizes of tensors must match except in dimension " + str(d) + ". Expected size " + str(shp[i]) + " but got size " + str(x._shape[i]) + " for tensor number 1 in the list.")
        tot += x._shape[d]
    shp[d] = tot
    out = _new(shp, dt)
    st = 0
    with no_grad():
        for x in ts:
            w = x._shape[d]
            if w > 0:
                _copy_into(_slice(out, d, st, st + w, 1), x)
            st += w
    if _needs(ts):
        n = _node("CatBackward0", _K_CAT, ts)
        n.ints = [x._shape[d] for x in ts]
        n.ints2 = [d]
        _attach(out, n)
    return out

def concat(*args, **kwargs):
    return cat(*args, **kwargs)
def concatenate(*args, **kwargs):
    return cat(*args, **kwargs)

def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    if len(tensors) == 0:
        raise RuntimeError("stack expects a non-empty TensorList")
    nd = len(tensors[0]._shape)
    d = _wrap(dim, nd + 1)
    for x in tensors:
        if x._shape != tensors[0]._shape:
            raise RuntimeError("stack expects each tensor to be equal size, but got " + str(tensors[0]._shape) + " at entry 0 and " + str(x._shape) + " at entry 1")
    return cat([unsqueeze(x, d) for x in tensors], d)

def hstack(tensors: List[Tensor]) -> Tensor:
    if len(tensors[0]._shape) == 1:
        return cat(tensors, 0)
    return cat(tensors, 1)

def vstack(tensors: List[Tensor]) -> Tensor:
    return cat([x if len(x._shape) > 1 else unsqueeze(x, 0) for x in tensors], 0)

def split(tensor: Tensor, split_size_or_sections, dim: int = 0) -> List[Tensor]:
    d = _wrap(dim, len(tensor._shape))
    n = tensor._shape[d]
    out = List[Tensor]()
    if isinstance(split_size_or_sections, int):
        s = split_size_or_sections
        st = 0
        if n == 0:
            out.append(tensor)
        while st < n:
            w = _imin(s, n - st)
            out.append(narrow(tensor, d, st, w))
            st += w
    else:
        st = 0
        for w in split_size_or_sections:
            out.append(narrow(tensor, d, st, w))
            st += w
        if st != n:
            raise RuntimeError("split_with_sizes expects split_sizes to sum exactly to " + str(n))
    return out

def chunk(input: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
    d = _wrap(dim, len(input._shape))
    n = input._shape[d]
    s = (n + chunks - 1) // chunks
    if s == 0:
        s = 1
    return split(input, s, d)

def tensor_split(input: Tensor, sections: int, dim: int = 0) -> List[Tensor]:
    d = _wrap(dim, len(input._shape))
    n = input._shape[d]
    q = n // sections
    r = n % sections
    out = List[Tensor]()
    st = 0
    for i in range(sections):
        w = q + (1 if i < r else 0)
        out.append(narrow(input, d, st, w))
        st += w
    return out

def unbind(input: Tensor, dim: int = 0) -> List[Tensor]:
    d = _wrap(dim, len(input._shape))
    return [select(input, d, i) for i in range(input._shape[d])]

def repeat(input: Tensor, sizes) -> Tensor:
    reps = _to_list(sizes)
    nd = len(reps)
    if nd < len(input._shape):
        raise RuntimeError("Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
    x = input
    while len(x._shape) < nd:
        x = unsqueeze(x, 0)
    # expand trick: [a,b] -> [r0,a,r1,b] -> reshape
    shp = List[int]()
    exp_shape = List[int]()
    for i in range(nd):
        shp.append(1)
        shp.append(x._shape[i])
        exp_shape.append(reps[i])
        exp_shape.append(x._shape[i])
    y = reshape(x, shp)
    y = expand(y, exp_shape)
    final = [reps[i] * x._shape[i] for i in range(nd)]
    return reshape(contiguous(y), final)

def contiguous(input: Tensor) -> Tensor:
    if input.is_contiguous():
        return input
    return clone(input)

def tile(input: Tensor, dims) -> Tensor:
    reps = _to_list(dims)
    while len(reps) < len(input._shape):
        reps.insert(0, 1)
    return repeat(input, reps)

def repeat_interleave(input: Tensor, repeats: int, dim: Optional[int] = None) -> Tensor:
    if dim is None:
        x = reshape(input, [-1])
        d = 0
    else:
        x = input
        d = _wrap(dim, len(input._shape))
    n = x._shape[d]
    idx = _new([n * repeats], _I64)
    p = Ptr[int](idx._st.ptr)
    for i in range(n):
        for j in range(repeats):
            p[i * repeats + j] = i
    return index_select(x, d, idx)

def flip(input: Tensor, dims) -> Tensor:
    ds = [_wrap(v, len(input._shape)) for v in _to_list(dims)]
    shp = input._shape.copy()
    st = input._strides.copy()
    off = input._off
    for d in ds:
        if shp[d] > 0:
            off += (shp[d] - 1) * st[d]
        st[d] = -st[d]
    v = Tensor(input._st, off, shp, st, input._dt)
    r = _new(shp.copy(), input._dt)
    _copy_into(r, v)
    if _needs([input]):
        n = _node("FlipBackward0", _K_FLIP, [input])
        n.ints = ds
        _attach(r, n)
    return r

def fliplr(input: Tensor) -> Tensor:
    return flip(input, [1])

def flipud(input: Tensor) -> Tensor:
    return flip(input, [0])

def roll(input: Tensor, shifts, dims = None) -> Tensor:
    if dims is None:
        flat = reshape(input, [-1])
        return reshape(roll(flat, shifts, 0), input._shape)
    else:
        sh = _to_list(shifts)
        ds = _to_list(dims)
        x = input
        for k in range(len(ds)):
            d = _wrap(ds[k], len(x._shape))
            n = x._shape[d]
            if n == 0:
                continue
            s = sh[k] % n
            if s == 0:
                continue
            x = cat([narrow(x, d, n - s, s), narrow(x, d, 0, n - s)], d)
        return x

# ================================================================ indexing

def _k_index_select(z, x: Tensor, d: int, idx: List[int], out: Tensor):
    T = type(z)
    # x contiguous
    outer = 1
    for i in range(d):
        outer *= x._shape[i]
    inner = 1
    for i in range(d + 1, len(x._shape)):
        inner *= x._shape[i]
    n = x._shape[d]
    px = Ptr[T](x._st.ptr) + x._off
    po = Ptr[T](out._st.ptr)
    m = len(idx)
    for o in range(outer):
        for j in range(m):
            src = px + (o * n + idx[j]) * inner
            dst = po + (o * m + j) * inner
            for i in range(inner):
                dst[i] = src[i]

def _index_list(index: Tensor, n: int) -> List[int]:
    c = _contig_as(index, _I64)
    p = Ptr[int](c._st.ptr) + c._off
    out = List[int](c.numel())
    for i in range(c.numel()):
        v = p[i]
        if v < -n or v >= n:
            raise IndexError("index " + str(v) + " is out of bounds for dimension with size " + str(n))
        out.append(v + n if v < 0 else v)
    return out

def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    nd = len(input._shape)
    d = _wrap(dim, nd) if nd > 0 else 0
    x = input.contiguous() if nd > 0 else reshape(input, [1])
    n = x._shape[d]
    idx = _index_list(index, n)
    shp = x._shape.copy()
    shp[d] = len(idx)
    out = _new(shp, input._dt)
    _dispatch(input._dt, _k_index_select, x, d, idx, out)
    if nd == 0:
        out = reshape(out.detach(), List[int]()) if len(idx) == 1 else out
    if _needs([input]):
        nn = _node("IndexSelectBackward0", _K_INDEX_SELECT, [input])
        nn.ints = input._shape.copy()
        nn.ints2 = [d]
        nn.saved = [_contig_as(index, _I64)]
        _attach(out, nn)
    return out

def _k_index_add(z, out: Tensor, d: int, idx: List[int], src: Tensor, alpha: float):
    T = type(z)
    outer = 1
    for i in range(d):
        outer *= out._shape[i]
    inner = 1
    for i in range(d + 1, len(out._shape)):
        inner *= out._shape[i]
    n = out._shape[d]
    m = len(idx)
    po = Ptr[T](out._st.ptr)
    ps = Ptr[T](src._st.ptr) + src._off
    a = _cv(alpha, T)
    for o in range(outer):
        for j in range(m):
            dst = po + (o * n + idx[j]) * inner
            s = ps + (o * m + j) * inner
            for i in range(inner):
                if alpha == 1.0:
                    dst[i] = _cv(dst[i] + s[i], T) if not isinstance(z, bool) else (dst[i] or s[i])
                else:
                    dst[i] = _cv(dst[i] + s[i] * a, T) if not isinstance(z, bool) else (dst[i] or s[i])

def index_add(input: Tensor, dim: int, index: Tensor, source: Tensor, alpha: float = 1.0) -> Tensor:
    d = _wrap(dim, len(input._shape))
    out = _new(input._shape.copy(), input._dt)
    _copy_into(out, input)
    idx = _index_list(index, input._shape[d])
    src = _contig_as(source, input._dt)
    _dispatch(input._dt, _k_index_add, out, d, idx, src, alpha)
    if _needs([input, source]):
        nn = _node("IndexAddBackward0", _K_INDEX_ADD, [input, source])
        nn.ints2 = [d]
        nn.fl = alpha
        nn.saved = [_contig_as(index, _I64)]
        _attach(out, nn)
    return out

def index_copy(input: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
    d = _wrap(dim, len(input._shape))
    out = clone(input.detach())
    idx = _index_list(index, input._shape[d])
    for j in range(len(idx)):
        select(out, d, idx[j]).copy_(select(source.detach(), d, j))
    return out

# gather / scatter (index has same ndim as input)
def _k_gather(z, x: Tensor, d: int, idx: Tensor, out: Tensor):
    T = type(z)
    px = Ptr[T](x._st.ptr)
    pi = Ptr[int](idx._st.ptr) + idx._off
    po = Ptr[T](out._st.ptr)
    nd = len(idx._shape)
    n = idx.numel()
    coord = [0] * nd
    for k in range(n):
        off = x._off
        for j in range(nd):
            c = coord[j]
            if j == d:
                c = pi[k]
                if c < 0:
                    c += x._shape[d]
                if c < 0 or c >= x._shape[d]:
                    raise RuntimeError("index " + str(pi[k]) + " is out of bounds for dimension " + str(d) + " with size " + str(x._shape[d]))
            off += c * x._strides[j]
        po[k] = px[off]
        j = nd - 1
        while j >= 0:
            coord[j] += 1
            if coord[j] < idx._shape[j]:
                break
            coord[j] = 0
            j -= 1

def gather(input: Tensor, dim: int, index: Tensor, sparse_grad: bool = False) -> Tensor:
    d = _wrap(dim, len(input._shape))
    idx = _contig_as(index, _I64)
    out = _new(idx._shape.copy(), input._dt)
    _dispatch(input._dt, _k_gather, input, d, idx, out)
    if _needs([input]):
        nn = _node("GatherBackward0", _K_GATHER, [input])
        nn.ints = input._shape.copy()
        nn.ints2 = [d]
        nn.saved = [idx]
        _attach(out, nn)
    return out

def _k_scatter(z, out: Tensor, d: int, idx: Tensor, src: Tensor, accumulate: bool):
    T = type(z)
    po = Ptr[T](out._st.ptr)
    pi = Ptr[int](idx._st.ptr) + idx._off
    ps = Ptr[T](src._st.ptr)
    nd = len(idx._shape)
    n = idx.numel()
    coord = [0] * nd
    for k in range(n):
        off = 0
        soff = src._off
        for j in range(nd):
            c = coord[j]
            soff += c * src._strides[j]
            if j == d:
                c = pi[k]
                if c < 0:
                    c += out._shape[d]
            off += c * out._strides[j]
        if accumulate:
            if isinstance(z, bool):
                po[off] = po[off] or ps[soff]
            else:
                po[off] = _addv(po[off], ps[soff])
        else:
            po[off] = ps[soff]
        j = nd - 1
        while j >= 0:
            coord[j] += 1
            if coord[j] < idx._shape[j]:
                break
            coord[j] = 0
            j -= 1

def scatter(input: Tensor, dim: int, index: Tensor, src) -> Tensor:
    d = _wrap(dim, len(input._shape))
    out = _new(input._shape.copy(), input._dt)
    _copy_into(out, input)
    idx = _contig_as(index, _I64)
    if isinstance(src, Tensor):
        s = _contig_as(src, input._dt)
    else:
        s = _contig_as(_expand_raw(_scalar_tensor(src, input._dt), idx._shape), input._dt)
    _dispatch(input._dt, _k_scatter, out, d, idx, s, False)
    return out

def scatter_add(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    d = _wrap(dim, len(input._shape))
    out = _new(input._shape.copy(), input._dt)
    _copy_into(out, input)
    idx = _contig_as(index, _I64)
    s = _contig_as(src, input._dt)
    _dispatch(input._dt, _k_scatter, out, d, idx, s, True)
    if _needs([input, src]):
        nn = _node("ScatterAddBackward0", _K_SCATTER_ADD, [input, src])
        nn.ints2 = [d]
        nn.saved = [idx]
        _attach(out, nn)
    return out

def take_along_dim(input: Tensor, indices: Tensor, dim: int) -> Tensor:
    return gather(input, dim, indices)

def _k_nonzero(z, x: Tensor, out: List[int]):
    T = type(z)
    p = Ptr[T](x._st.ptr) + x._off
    for i in range(x.numel()):
        if p[i] != T():
            out.append(i)

def nonzero(input: Tensor, as_tuple: bool = False) -> Tensor:
    c = input.contiguous()
    lin = List[int]()
    _dispatch(c._dt, _k_nonzero, c, lin)
    nd = len(c._shape)
    out = _new([len(lin), nd], _I64)
    p = Ptr[int](out._st.ptr)
    for k in range(len(lin)):
        v = lin[k]
        j = nd - 1
        while j >= 0:
            p[k * nd + j] = v % c._shape[j]
            v //= c._shape[j]
            j -= 1
    return out

def argwhere(input: Tensor) -> Tensor:
    return nonzero(input)

def masked_select(input: Tensor, mask: Tensor) -> Tensor:
    shape = _broadcast_shapes(input._shape, mask._shape)
    x = reshape(expand(input, shape), [-1])
    m = _bcast_contig(mask, shape, _BOOL)
    lin = List[int]()
    _k_nonzero(False, m, lin)
    idx = _new([len(lin)], _I64)
    p = Ptr[int](idx._st.ptr)
    for i in range(len(lin)):
        p[i] = lin[i]
    return index_select(x, 0, idx)

def _slice_int(v):
    # Tensor.item() is statically float, while torch returns int for integer dtypes:
    # accept integral floats as slice bounds (t[:, :, :-shift] with shift = randint(...).item()).
    if isinstance(v, float):
        if v != float(int(v)):
            raise TypeError("slice indices must be integers or None or have an __index__ method")
        return Optional[int](int(v))
    elif isinstance(v, Optional[float]):
        if v is None:
            return Optional[int]()
        return _slice_int(v.__val__())
    else:
        return v

def _norm_slice(sl, n: int):
    if isinstance(sl.start, float) or isinstance(sl.stop, float) or isinstance(sl.step, float) or isinstance(sl.start, Optional[float]) or isinstance(sl.stop, Optional[float]) or isinstance(sl.step, Optional[float]):
        return _norm_slice(Slice(_slice_int(sl.start), _slice_int(sl.stop), _slice_int(sl.step)), n)
    start, stop, step, length = sl.adjust_indices(n)
    if step <= 0:
        raise ValueError("step must be greater than zero")
    if length <= 0:
        return (start if start <= n else n, start if start <= n else n, 1)
    return (start, stop, step)

def _getitem(x: Tensor, idx) -> Tensor:
    if isinstance(idx, Tuple):
        return _getitem_tuple(x, idx)
    else:
        return _getitem_tuple(x, (idx,))

def _count_consumed(idx) -> int:
    n = 0
    for e in idx:
        if isinstance(e, NoneType) or isinstance(e, ellipsis):
            pass
        elif isinstance(e, Tensor):
            if e._dt == _BOOL:
                n += len(e._shape)
            else:
                n += 1
        else:
            n += 1
    return n

def _getitem_tuple(x: Tensor, idx) -> Tensor:
    r = x
    d = 0
    nd = len(x._shape)
    consumed = _count_consumed(idx)
    adv_dims = List[int]()
    adv_idx = List[Tensor]()
    for e in idx:
        if isinstance(e, int):
            r = select(r, d, e)
        elif isinstance(e, Slice):
            st, sp, stp = _norm_slice(e, r._shape[d])
            r = _slice(r, d, st, sp, stp)
            d += 1
        elif isinstance(e, NoneType):
            r = unsqueeze(r, d)
            d += 1
        elif isinstance(e, ellipsis):
            d += nd - consumed
        elif isinstance(e, Tensor):
            if e._dt == _BOOL:
                # boolean mask over the next len(e.shape) dims
                k = len(e._shape)
                if k == 0:
                    r = unsqueeze(r, d)
                    if not _item_f64(e) != 0.0:
                        r = _slice(r, d, 0, 0, 1)
                    d += 1
                else:
                    shp = r._shape[:d] + [-1] + r._shape[d + k:]
                    flat = reshape(r, shp)
                    lin = List[int]()
                    m = _contig_as(e, _BOOL)
                    _k_nonzero(False, m, lin)
                    ii = _new([len(lin)], _I64)
                    p = Ptr[int](ii._st.ptr)
                    for i in range(len(lin)):
                        p[i] = lin[i]
                    r = index_select(flat, d, ii)
                    d += 1
            elif len(e._shape) == 0:
                r = select(r, d, _item_i64(e))
            else:
                adv_dims.append(d)
                adv_idx.append(e)
                d += 1
        elif isinstance(e, List[int]):
            ii = tensor(e, dtype=_DType(_I64))
            adv_dims.append(d)
            adv_idx.append(ii)
            d += 1
        elif isinstance(e, bool):
            r = unsqueeze(r, d)
            if not e:
                r = _slice(r, d, 0, 0, 1)
            d += 1
        else:
            compile_error("unsupported index type")
    if len(adv_idx) == 1:
        e = adv_idx[0]
        ad = adv_dims[0]
        flat = index_select(r, ad, reshape(e, [-1]))
        shp = r._shape[:ad] + e._shape + r._shape[ad + 1:]
        r = reshape(flat, shp)
    elif len(adv_idx) > 1:
        r = _multi_adv(r, adv_dims, adv_idx)
    return r

# advanced indexing with several index tensors (adjacent or not)
def _multi_adv(r: Tensor, dims: List[int], idxs: List[Tensor]) -> Tensor:
    bshape = List[int]()
    for e in idxs:
        bshape = _broadcast_shapes(bshape, e._shape)
    # move indexed dims to front
    nd = len(r._shape)
    rest = [i for i in range(nd) if i not in dims]
    p = permute(r, dims + rest)
    lead = [r._shape[i] for i in dims]
    tail = [r._shape[i] for i in rest]
    flat = reshape(p, [-1] + tail)
    m = _numel(bshape)
    lin = _new([m], _I64)
    pl = Ptr[int](lin._st.ptr)
    for i in range(m):
        pl[i] = 0
    for k in range(len(idxs)):
        e = _bcast_contig(idxs[k], bshape, _I64)
        pe = Ptr[int](e._st.ptr)
        n = lead[k]
        for i in range(m):
            v = pe[i]
            if v < 0:
                v += n
            pl[i] = pl[i] * n + v
    out = index_select(flat, 0, lin)
    out = reshape(out, bshape + tail)
    adjacent = True
    for k in range(1, len(dims)):
        if dims[k] != dims[k - 1] + 1:
            adjacent = False
    if adjacent and dims[0] > 0:
        nb = len(bshape)
        perm = List[int]()
        for i in range(dims[0]):
            perm.append(nb + i)
        for i in range(nb):
            perm.append(i)
        for i in range(dims[0], len(tail)):
            perm.append(nb + i)
        out = permute(out, perm)
    return out

def _setitem(x: Tensor, idx, value):
    _check_inplace(x)
    if isinstance(idx, Tensor):
        if idx._dt == _BOOL:
            m = _bcast_contig(idx, x._shape, _BOOL)
            if isinstance(value, Tensor):
                if value.numel() == 1:
                    r = masked_fill(x.detach(), m, _item_f64(value))
                    x.copy_(r)
                else:
                    lin = List[int]()
                    _k_nonzero(False, m, lin)
                    fx = x.contiguous()
                    v = _contig_as(value, x._dt)
                    _dispatch(x._dt, _k_put, fx, lin, v)
                    if fx is not x:
                        x.copy_(fx)
            else:
                r = masked_fill(x.detach(), m, value)
                x.copy_(r)
            return
    with no_grad():
        v = x.detach()[idx]
        if isinstance(value, Tensor):
            _copy_into(v, _expand_raw(value.detach(), v._shape))
        else:
            _fill(v, float(value))

def _k_put(z, x: Tensor, lin: List[int], v: Tensor):
    T = type(z)
    p = Ptr[T](x._st.ptr) + x._off
    q = Ptr[T](v._st.ptr) + v._off
    for i in range(len(lin)):
        p[lin[i]] = q[i]

# ================================================================ reductions

def _norm_dims(dim, nd: int) -> List[int]:
    if dim is None:
        return list(range(nd))
    else:
        ds = _to_list(dim)
        if len(ds) == 0:
            return list(range(nd))
        out = List[int]()
        for v in ds:
            w = _wrap(v, nd) if nd > 0 else 0
            if w not in out:
                out.append(w)
        out.sort()
        return out

# Move reduced dims last and make contiguous; returns (tensor, outer, inner, out_shape_keep, out_shape)
def _reduce_prep(x: Tensor, dims: List[int]):
    nd = len(x._shape)
    keep = [i for i in range(nd) if i not in dims]
    perm = keep + dims
    xp = Tensor(x._st, x._off, [x._shape[i] for i in perm], [x._strides[i] for i in perm], x._dt)
    c = _contig_as(xp, x._dt)
    inner = 1
    for d in dims:
        inner *= x._shape[d]
    outer = 1
    for d in keep:
        outer *= x._shape[d]
    kshape = x._shape.copy()
    for d in dims:
        kshape[d] = 1
    oshape = [x._shape[i] for i in keep]
    return c, outer, inner, kshape, oshape

def _k_sum(z, c: Tensor, outer: int, inner: int, out: Tensor):
    T = type(z)
    p = Ptr[T](c._st.ptr) + c._off
    if out._dt == _F32:
        q = Ptr[float32](out._st.ptr)
        for o in range(outer):
            q[o] = _cascade_sum_f32(Ptr[float32]((p + o * inner).as_byte()), inner)
    elif out._dt == _F64:
        q = Ptr[float](out._st.ptr)
        for o in range(outer):
            b = p + o * inner
            q[o] = _pairwise_sum_f64(b, inner)
    else:
        q = Ptr[int](out._st.ptr)
        for o in range(outer):
            acc = 0
            b = p + o * inner
            for i in range(inner):
                acc += _cv(b[i], int)
            q[o] = acc

def _cascade_sum_f32(p: Ptr[float32], n: int) -> float32:
    """Bit-exact float32 contiguous sum of ATen CPU (ReduceOps SumKernel cascade_sum,
    inner reduction): 8-lane vectors, 4 interleaved rows, 4 cascade levels, then the
    scalar tail followed by a horizontal add of the lanes."""
    W = 8
    ILP = 4
    L = 4
    vs = n // W
    sz = vs // ILP
    lp = 4
    if sz > 1:
        cl = 0
        while (1 << cl) < sz:
            cl += 1
        if cl // L > lp:
            lp = cl // L
    step = 1 << lp
    mask = step - 1
    acc = Ptr[float32](L * ILP * W)
    for k in range(L * ILP * W):
        acc[k] = float32(0.0)
    i = 0
    while i + step <= sz:
        for _ in range(step):
            for r in range(ILP):
                src = p + (i * ILP + r) * W
                dst = acc + r * W
                for l in range(W):
                    dst[l] = dst[l] + src[l]
            i += 1
        for j in range(1, L):
            a = acc + j * ILP * W
            b = acc + (j - 1) * ILP * W
            for k in range(ILP * W):
                a[k] = a[k] + b[k]
                b[k] = float32(0.0)
            if (i & (mask << (j * lp))) != 0:
                break
    while i < sz:
        for r in range(ILP):
            src = p + (i * ILP + r) * W
            dst = acc + r * W
            for l in range(W):
                dst[l] = dst[l] + src[l]
        i += 1
    for j in range(1, L):
        a = acc + j * ILP * W
        for k in range(ILP * W):
            acc[k] = acc[k] + a[k]
    for v in range(sz * ILP, vs):
        src = p + v * W
        for l in range(W):
            acc[l] = acc[l] + src[l]
    for r in range(1, ILP):
        for l in range(W):
            acc[l] = acc[l] + acc[r * W + l]
    fa = float32(0.0)
    for k in range(vs * W, n):
        fa = fa + p[k]
    for l in range(W):
        fa = fa + acc[l]
    return fa

def _multi_row_sum_f32(p: Ptr[float32], row_stride: int, col_stride: int, size: int, nrows: int, out: Ptr[float32]):
    # ATen SumKernel multi_row_sum<float, nrows> (4 cascade levels)
    L = 4
    lp = 4
    if size > 1:
        cl = 0
        while (1 << cl) < size:
            cl += 1
        if cl // L > lp:
            lp = cl // L
    step = 1 << lp
    mask = step - 1
    acc = Ptr[float32](L * nrows)
    for k in range(L * nrows):
        acc[k] = float32(0.0)
    i = 0
    while i + step <= size:
        for _ in range(step):
            base = p + i * row_stride
            for k in range(nrows):
                acc[k] = acc[k] + base[k * col_stride]
            i += 1
        for j in range(1, L):
            for k in range(nrows):
                acc[j * nrows + k] = acc[j * nrows + k] + acc[(j - 1) * nrows + k]
                acc[(j - 1) * nrows + k] = float32(0.0)
            if (i & (mask << (j * lp))) != 0:
                break
    while i < size:
        base = p + i * row_stride
        for k in range(nrows):
            acc[k] = acc[k] + base[k * col_stride]
        i += 1
    for j in range(1, L):
        for k in range(nrows):
            acc[k] = acc[k] + acc[j * nrows + k]
    for k in range(nrows):
        out[k] = acc[k]

def _row_sum_f32(p: Ptr[float32], stride: int, size: int) -> float32:
    # ATen SumKernel row_sum (ilp_factor 4)
    si = size // 4
    ps = Ptr[float32](4)
    _multi_row_sum_f32(p, stride * 4, stride, si, 4, ps)
    for i in range(si * 4, size):
        ps[0] = ps[0] + p[i * stride]
    for k in range(1, 4):
        ps[0] = ps[0] + ps[k]
    return ps[0]

def _outer_sum_f32(p: Ptr[float32], N: int, C: int, q: Ptr[float32]):
    # ATen vectorized_outer_sum on aarch64 (nrows 8 x 4 lanes): sum over rows of a
    # contiguous (N, C) block, per column.
    j = 0
    while j + 32 <= C:
        _multi_row_sum_f32(p + j, C, 1, N, 32, q + j)
        j += 32
    while j < C:
        q[j] = _row_sum_f32(p + j, C, N)
        j += 1

def _pairwise_sum_f64(p, n: int) -> float:
    if n <= 8:
        s = 0.0
        for i in range(n):
            s += _cv(p[i], float)
        return s
    if n <= 128:
        r = [0.0] * 8
        i = 0
        while i + 8 <= n:
            for j in range(8):
                r[j] += _cv(p[i + j], float)
            i += 8
        s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n:
            s += _cv(p[i], float)
            i += 1
        return s
    h = (n // 2) & ~7
    return _pairwise_sum_f64(p, h) + _pairwise_sum_f64(p + h, n - h)

def sum(input: Tensor, dim = None, keepdim: bool = False, dtype: Optional[_DType] = None) -> Tensor:
    if _is_complex(input._dt):
        r = view_as_real(input)
        nd0 = len(input._shape)
        return view_as_complex(sum(r, _norm_dims(dim, nd0) if nd0 > 0 else List[int](), keepdim))
    x = input
    if dtype is not None:
        x = x.to(dtype)
    nd = len(x._shape)
    dims = _norm_dims(dim, nd)
    if x._dt == _F32 and nd >= 2 and len(dims) == 1 and dims[0] == 0 and x.is_contiguous():
        # contiguous outer reduction (e.g. linear bias grad): ATen vectorized_outer_sum
        rest = x.numel() // x._shape[0] if x._shape[0] > 0 else 0
        kshape0 = x._shape.copy()
        kshape0[0] = 1
        out0 = _new(kshape0 if keepdim else x._shape[1:], _F32)
        _outer_sum_f32(Ptr[float32](x._st.ptr) + x._off, x._shape[0], rest, Ptr[float32](out0._st.ptr))
        if _needs([x]):
            n0 = _node("SumBackward1", _K_SUM, [x])
            n0.ints = x._shape.copy()
            n0.ints2 = kshape0
            n0.fl = 1.0 if keepdim else 0.0
            _attach(out0, n0)
        return out0
    c, outer, inner, kshape, oshape = _reduce_prep(x, dims)
    odt = x._dt if _is_float(x._dt) else _I64
    out = _new(kshape if keepdim else oshape, odt)
    _dispatch(c._dt, _k_sum, c, outer, inner, out)
    if _needs([x]):
        n = _node("SumBackward1" if dim is not None else "SumBackward0", _K_SUM, [x])
        n.ints = x._shape.copy()
        n.ints2 = kshape
        n.fl = 1.0 if keepdim else 0.0
        _attach(out, n)
    return out

def mean(input: Tensor, dim = None, keepdim: bool = False, dtype: Optional[_DType] = None) -> Tensor:
    x = input
    if dtype is not None:
        x = x.to(dtype)
    if _is_complex(x._dt):
        r = view_as_real(x)
        nd0 = len(x._shape)
        return view_as_complex(mean(r, _norm_dims(dim, nd0) if nd0 > 0 else List[int](), keepdim))
    if not _is_float(x._dt):
        raise RuntimeError("mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: " + _DType(x._dt)._name().capitalize())
    nd = len(x._shape)
    dims = _norm_dims(dim, nd)
    cnt = 1
    for d in dims:
        if nd > 0:
            cnt *= x._shape[d]
    s = sum(x, dims if nd > 0 else None, keepdim)
    r = div(s, float(cnt))
    if r.grad_fn is not None:
        _unw(r.grad_fn)._name = "MeanBackward0"
    return r

def nansum(input: Tensor, dim = None, keepdim: bool = False) -> Tensor:
    return sum(where(isnan(input), zeros_like(input), input), dim, keepdim)

def nanmean(input: Tensor, dim = None, keepdim: bool = False) -> Tensor:
    m = logical_not(isnan(input))
    return div(nansum(input, dim, keepdim), sum(m, dim, keepdim))

def _k_prod(z, c: Tensor, outer: int, inner: int, out: Tensor):
    T = type(z)
    p = Ptr[T](c._st.ptr) + c._off
    if _is_float(out._dt):
        for o in range(outer):
            acc = 1.0
            b = p + o * inner
            for i in range(inner):
                acc *= _cv(b[i], float)
            if out._dt == _F32:
                Ptr[float32](out._st.ptr)[o] = float32(acc)
            else:
                Ptr[float](out._st.ptr)[o] = acc
    else:
        q = Ptr[int](out._st.ptr)
        for o in range(outer):
            acc = 1
            b = p + o * inner
            for i in range(inner):
                acc *= _cv(b[i], int)
            q[o] = acc

def prod(input: Tensor, dim = None, keepdim: bool = False, dtype: Optional[_DType] = None) -> Tensor:
    x = input if dtype is None else input.to(_unw(dtype))
    dims = _norm_dims(dim, len(x._shape))
    c, outer, inner, kshape, oshape = _reduce_prep(x, dims)
    odt = x._dt if _is_float(x._dt) else _I64
    out = _new(kshape if keepdim else oshape, odt)
    _dispatch(c._dt, _k_prod, c, outer, inner, out)
    if _needs([x]):
        n = _node("ProdBackward0", _K_PROD, [x])
        n.saved = [reshape(out.detach(), kshape)]
        n.fl = 1.0 if keepdim else 0.0
        _attach(out, n)
    return out

def _welford_f32(x: Tensor, dims: List[int], corr: int, keepdim: bool, take_sqrt: bool) -> Tensor:
    # ATen CPU std_var kernel: per-row Welford in float64 (sqrt in float64 for std),
    # then cast to float32.
    c, outer, inner, kshape, oshape = _reduce_prep(x, dims)
    out = _new(kshape if keepdim else oshape, _F32)
    pc = Ptr[float32](c._st.ptr) + c._off
    q = Ptr[float32](out._st.ptr)
    den = inner - corr
    for o in range(outer):
        b = pc + o * inner
        mu = 0.0
        m2 = 0.0
        for i in range(inner):
            v = float(b[i])
            d = v - mu
            mu += d / float(i + 1)
            m2 += d * (v - mu)
        r = m2 / float(den) if den > 0 else (float("nan") if inner == 0 or m2 == 0.0 else float("inf"))
        q[o] = float32(_math.sqrt(r) if take_sqrt else r)
    return out

def var(input: Tensor, dim = None, unbiased: bool = True, keepdim: bool = False, correction: Optional[int] = None) -> Tensor:
    corr = (1 if unbiased else 0) if correction is None else correction
    x = input
    nd = len(x._shape)
    dims = _norm_dims(dim, nd)
    if x._dt == _F32 and nd > 0 and not _needs([x]):
        return _welford_f32(x, dims, corr, keepdim, False)
    cnt = 1
    for d in dims:
        if nd > 0:
            cnt *= x._shape[d]
    m = mean(x, dims if nd > 0 else None, True)
    dev = sub(x, m)
    ss = sum(mul(dev, dev), dims if nd > 0 else None, keepdim)
    den = cnt - corr
    return div(ss, float(den) if den > 0 else 0.0)

def std(input: Tensor, dim = None, unbiased: bool = True, keepdim: bool = False, correction: Optional[int] = None) -> Tensor:
    nd = len(input._shape)
    if input._dt == _F32 and nd > 0 and not _needs([input]):
        corr = (1 if unbiased else 0) if correction is None else correction
        return _welford_f32(input, _norm_dims(dim, nd), corr, keepdim, True)
    return sqrt(var(input, dim, unbiased, keepdim, correction))

def var_mean(input: Tensor, dim = None, unbiased: bool = True, keepdim: bool = False, correction: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    return (var(input, dim, unbiased, keepdim, correction), mean(input, dim, keepdim))

def std_mean(input: Tensor, dim = None, unbiased: bool = True, keepdim: bool = False, correction: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    return (std(input, dim, unbiased, keepdim, correction), mean(input, dim, keepdim))

def _k_minmax(z, c: Tensor, outer: int, inner: int, vals: Tensor, idx: Tensor, is_max: bool):
    T = type(z)
    p = Ptr[T](c._st.ptr) + c._off
    pv = Ptr[T](vals._st.ptr)
    pi = Ptr[int](idx._st.ptr)
    if inner == 0:
        raise IndexError("max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.")
    for o in range(outer):
        b = p + o * inner
        best = b[0]
        bi = 0
        isnanbest = False
        if isinstance(z, float) or isinstance(z, float32):
            isnanbest = best != best
        for i in range(1, inner):
            if isnanbest:
                break
            v = b[i]
            if isinstance(z, float) or isinstance(z, float32):
                if v != v:
                    best = v
                    bi = i
                    isnanbest = True
                    break
            if (v > best) if is_max else (v < best):
                best = v
                bi = i
        pv[o] = best
        pi[o] = bi

@tuple
class return_types_max:
    values: Tensor
    indices: Tensor

def _minmax_dim(x: Tensor, dim: int, keepdim: bool, is_max: bool) -> return_types_max:
    nd = len(x._shape)
    d = _wrap(dim, nd) if nd > 0 else 0
    dims = [d] if nd > 0 else List[int]()
    c, outer, inner, kshape, oshape = _reduce_prep(x, dims)
    vals = _new(kshape if keepdim else oshape, x._dt)
    idx = _new(kshape if keepdim else oshape, _I64)
    _dispatch(c._dt, _k_minmax, c, outer, inner, vals, idx, is_max)
    if _needs([x]):
        n = _node("MaxBackward0" if is_max else "MinBackward0", _K_MAXDIM, [x])
        n.ints = [d]
        n.ints2 = x._shape.copy()
        n.fl = 1.0 if keepdim else 0.0
        n.saved = [reshape(idx, kshape)]
        _attach(vals, n)
    return return_types_max(vals, idx)

def _amaxmin(x: Tensor, dim, keepdim: bool, is_max: bool) -> Tensor:
    nd = len(x._shape)
    dims = _norm_dims(dim, nd)
    c, outer, inner, kshape, oshape = _reduce_prep(x, dims)
    vals = _new(kshape if keepdim else oshape, x._dt)
    idx = _new(kshape if keepdim else oshape, _I64)
    _dispatch(c._dt, _k_minmax, c, outer, inner, vals, idx, is_max)
    if _needs([x]):
        n = _node(("AmaxBackward0" if is_max else "AminBackward0") if dim is not None else ("MaxBackward1" if is_max else "MinBackward1"), _K_AMAX, [x])
        n.saved = [reshape(vals.detach(), kshape)]
        n.ints2 = dims
        n.fl = 1.0 if keepdim else 0.0
        _attach(vals, n)
    return vals

def max(input: Tensor, dim = None, keepdim: bool = False):
    if dim is None:
        return _amaxmin(input, None, False, True)
    elif isinstance(dim, Tensor):
        return maximum(input, dim)
    else:
        return _minmax_dim(input, dim, keepdim, True)

def min(input: Tensor, dim = None, keepdim: bool = False):
    if dim is None:
        return _amaxmin(input, None, False, False)
    elif isinstance(dim, Tensor):
        return minimum(input, dim)
    else:
        return _minmax_dim(input, dim, keepdim, False)

def amax(input: Tensor, dim = None, keepdim: bool = False) -> Tensor:
    return _amaxmin(input, dim, keepdim, True)

def amin(input: Tensor, dim = None, keepdim: bool = False) -> Tensor:
    return _amaxmin(input, dim, keepdim, False)

def aminmax(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tuple[Tensor, Tensor]:
    return (amin(input, dim, keepdim), amax(input, dim, keepdim))

def argmax(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    if dim is None:
        r = _minmax_dim(reshape(input.detach(), [-1]), 0, False, True).indices
        if keepdim:
            return reshape(r, [1] * len(input._shape))
        return r
    return _minmax_dim(input.detach(), dim, keepdim, True).indices

def argmin(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    if dim is None:
        r = _minmax_dim(reshape(input.detach(), [-1]), 0, False, False).indices
        if keepdim:
            return reshape(r, [1] * len(input._shape))
        return r
    return _minmax_dim(input.detach(), dim, keepdim, False).indices

def any(input: Tensor, dim = None, keepdim: bool = False) -> Tensor:
    b = _contig_as(ne(input, 0), _I64)
    return gt(sum(b, dim, keepdim), 0)

def all(input: Tensor, dim = None, keepdim: bool = False) -> Tensor:
    b = _contig_as(eq(input, 0), _I64)
    return eq(sum(b, dim, keepdim), 0)

def count_nonzero(input: Tensor, dim = None) -> Tensor:
    return sum(_contig_as(ne(input, 0), _I64), dim)

def logsumexp(input: Tensor, dim, keepdim: bool = False) -> Tensor:
    x = input if _is_float(input._dt) else input.float()
    m = amax(x.detach(), dim, True)
    m = where(isinf(m), zeros_like(m), m)
    s = log(sum(exp(sub(x, m)), dim, True))
    r = add(s, m)
    if not keepdim:
        r = squeeze(r, _norm_dims(dim, len(x._shape)))
    return r

def softmax(input: Tensor, dim: int, dtype: Optional[_DType] = None) -> Tensor:
    x = input if dtype is None else input.to(_unw(dtype))
    if not _is_float(x._dt):
        x = x.float()
    if (x._dt == _F32 and x.ndim >= 1 and (dim == -1 or dim == x.ndim - 1) and x.is_contiguous()
            and not (_GRAD_ENABLED[0] and x.requires_grad) and x.numel() > 0):
        # ATen _vec_softmax_lastdim: max via reduce_all, out = Sleef exp(x - max),
        # sum via reduce_all, out *= 1 / sum (no autograd needed here)
        from torch.nn._ce import _max_all, _sum_all
        from torch._sleef import expf as _sexp
        C = x._shape[x.ndim - 1]
        R = x.numel() // C
        res = _new(x._shape.copy(), _F32)
        px = Ptr[float32](x._st.ptr) + x._off
        po = Ptr[float32](res._st.ptr)
        acc = Ptr[float32](4)
        for r in range(R):
            row = px + r * C
            o = po + r * C
            mx = _max_all(row, C, acc)
            for j in range(C):
                o[j] = _sexp(row[j] - mx)
            inv = float32(1.0) / _sum_all(o, C, acc)
            for j in range(C):
                o[j] = o[j] * inv
        return res
    m = amax(x.detach(), dim, True)
    e = exp(sub(x, m))
    return div(e, sum(e, dim, True))

def log_softmax(input: Tensor, dim: int, dtype: Optional[_DType] = None) -> Tensor:
    x = input if dtype is None else input.to(_unw(dtype))
    if not _is_float(x._dt):
        x = x.float()
    m = amax(x.detach(), dim, True)
    sh = sub(x, m)
    return sub(sh, log(sum(exp(sh), dim, True)))

def _k_cumsum(z, c: Tensor, outer: int, n: int, inner: int, out: Tensor):
    T = type(z)
    p = Ptr[T](c._st.ptr) + c._off
    if _is_float(out._dt):
        for o in range(outer):
            for i in range(inner):
                acc = 0.0
                for j in range(n):
                    k = (o * n + j) * inner + i
                    acc += _cv(p[k], float)
                    if out._dt == _F32:
                        Ptr[float32](out._st.ptr)[k] = float32(acc)
                    else:
                        Ptr[float](out._st.ptr)[k] = acc
    else:
        q = Ptr[int](out._st.ptr)
        for o in range(outer):
            for i in range(inner):
                acc = 0
                for j in range(n):
                    k = (o * n + j) * inner + i
                    acc += _cv(p[k], int)
                    q[k] = acc

def cumsum(input: Tensor, dim: int, dtype: Optional[_DType] = None) -> Tensor:
    x = input if dtype is None else input.to(_unw(dtype))
    nd = len(x._shape)
    d = _wrap(dim, nd) if nd > 0 else 0
    c = x.contiguous()
    outer = 1
    for i in range(d):
        outer *= c._shape[i]
    inner = 1
    for i in range(d + 1, nd):
        inner *= c._shape[i]
    n = c._shape[d] if nd > 0 else 1
    odt = x._dt if _is_float(x._dt) else _I64
    out = _new(c._shape.copy(), odt)
    _dispatch(c._dt, _k_cumsum, c, outer, n, inner, out)
    if _needs([x]):
        nn = _node("CumsumBackward0", _K_CUMSUM, [x])
        nn.ints = [d]
        _attach(out, nn)
    return out

def cumprod(input: Tensor, dim: int, dtype: Optional[_DType] = None) -> Tensor:
    x = input if dtype is None else input.to(_unw(dtype))
    if not _is_float(x._dt) and not _is_complex(x._dt) and x._dt != _I64:
        x = x.to(_DType(_I64))
    if len(x._shape) == 0:
        return clone(x)
    d = _wrap(dim, len(x._shape))
    if x._shape[d] == 0:
        return clone(x)
    parts = List[Tensor]()
    acc = select(x, d, 0)
    parts.append(acc)
    for i in range(1, x._shape[d]):
        acc = mul(acc, select(x, d, i))
        parts.append(acc)
    r = stack(parts, d)
    if r.grad_fn is not None:
        _unw(r.grad_fn)._name = "CumprodBackward0"
    return r

def _cumext(input: Tensor, dim: int, is_max: bool) -> return_types_max:
    x = input
    if len(x._shape) == 0:
        return return_types_max(clone(x), _new(List[int](), _I64))
    d = _wrap(dim, len(x._shape))
    n = x._shape[d]
    if n == 0:
        return return_types_max(clone(x), _new(x._shape.copy(), _I64))
    acc = select(x, d, 0)
    ai = zeros(acc._shape.copy(), dtype=_DType(_I64))
    vals = [acc]
    idxs = [ai]
    flt = _is_float(x._dt)
    for i in range(1, n):
        cur = select(x, d, i)
        take = ge(cur, acc) if is_max else le(cur, acc)
        if flt:
            cn = isnan(cur)
            an = isnan(acc)
            take = logical_and(logical_or(take, cn), logical_not(logical_and(an, logical_not(cn))))
        acc = where(take, cur, acc)
        ai = where(take, full(ai._shape.copy(), i, dtype=_DType(_I64)), ai)
        vals.append(acc)
        idxs.append(ai)
    v = stack(vals, d)
    if v.grad_fn is not None:
        _unw(v.grad_fn)._name = "CummaxBackward0" if is_max else "CumminBackward0"
    return return_types_max(v, stack(idxs, d))

def cummax(input: Tensor, dim: int) -> return_types_max:
    return _cumext(input, dim, True)

def cummin(input: Tensor, dim: int) -> return_types_max:
    return _cumext(input, dim, False)

def logcumsumexp(input: Tensor, dim: int) -> Tensor:
    x = input
    if len(x._shape) == 0:
        return clone(x)
    d = _wrap(dim, len(x._shape))
    if x._shape[d] == 0:
        return clone(x)
    # running logaddexp with a per-step detached shift (the shift cancels in
    # the math, so gradients are exact); -inf shifts are mapped to 0
    acc = select(x, d, 0)
    parts = [acc]
    for i in range(1, x._shape[d]):
        cur = select(x, d, i)
        m = maximum(acc.detach(), cur.detach())
        m = where(isinf(m), zeros_like(m), m)
        acc = add(log(add(exp(sub(acc, m)), exp(sub(cur, m)))), m)
        parts.append(acc)
    r = stack(parts, d)
    if r.grad_fn is not None:
        _unw(r.grad_fn)._name = "LogcumsumexpBackward0"
    return r

def diff(input: Tensor, n: int = 1, dim: int = -1) -> Tensor:
    x = input
    d = _wrap(dim, len(x._shape))
    for _ in range(n):
        L = x._shape[d]
        x = sub(narrow(x, d, 1, L - 1), narrow(x, d, 0, L - 1))
    return x

def norm(input: Tensor, p = 2, dim = None, keepdim: bool = False, dtype: Optional[_DType] = None) -> Tensor:
    x = input if _is_float(input._dt) else input.float()
    if isinstance(p, str):
        if p == "fro":
            return sqrt(sum(mul(x, x), dim, keepdim))
        raise NotImplementedError("norm p=" + p)
    else:
        pv = float(p)
        if pv == 2.0:
            return sqrt(sum(mul(x, x), dim, keepdim))
        if pv == 1.0:
            return sum(abs(x), dim, keepdim)
        if pv == float("inf"):
            return amax(abs(x), dim, keepdim)
        if pv == -float("inf"):
            return amin(abs(x), dim, keepdim)
        if pv == 0.0:
            return sum(_contig_as(ne(x, 0.0), x._dt), dim, keepdim)
        return pow(sum(pow(abs(x), pv), dim, keepdim), 1.0 / pv)

def dist(input: Tensor, other: Tensor, p: float = 2.0) -> Tensor:
    return norm(sub(input, other), p)

# sort / topk
@tuple
class return_types_sort:
    values: Tensor
    indices: Tensor

def _k_sortidx(z, c: Tensor, outer: int, n: int, inner: int, idx: Tensor, descending: bool, stable: bool):
    T = type(z)
    p = Ptr[T](c._st.ptr) + c._off
    q = Ptr[int](idx._st.ptr)
    for o in range(outer):
        for i in range(inner):
            keys = List[Tuple[float, int]](n)
            for j in range(n):
                v = _cv(p[(o * n + j) * inner + i], float)
                if v != v:
                    v = float("inf") if not descending else float("inf")
                    keys.append((v, j))
                    continue
                keys.append((-v if descending else v, j))
            keys.sort()
            # NaN handling: NaN is largest in ascending, first in descending
            if descending:
                nans = [kv for kv in keys if kv[0] == float("inf") and p[(o * n + kv[1]) * inner + i] != p[(o * n + kv[1]) * inner + i]]
                if len(nans) > 0:
                    rest = [kv for kv in keys if not (kv[0] == float("inf") and p[(o * n + kv[1]) * inner + i] != p[(o * n + kv[1]) * inner + i])]
                    keys = nans + rest
            for j in range(n):
                q[(o * n + j) * inner + i] = keys[j][1]

def argsort(input: Tensor, dim: int = -1, descending: bool = False, stable: bool = False) -> Tensor:
    nd = len(input._shape)
    if nd == 0:
        return zeros(List[int](), dtype=_DType(_I64))
    d = _wrap(dim, nd)
    c = input.contiguous()
    outer = 1
    for i in range(d):
        outer *= c._shape[i]
    inner = 1
    for i in range(d + 1, nd):
        inner *= c._shape[i]
    idx = _new(c._shape.copy(), _I64)
    _dispatch(c._dt, _k_sortidx, c, outer, c._shape[d], inner, idx, descending, stable)
    return idx

def sort(input: Tensor, dim: int = -1, descending: bool = False, stable: bool = False) -> return_types_sort:
    idx = argsort(input, dim, descending, stable)
    if len(input._shape) == 0:
        return return_types_sort(input.clone(), idx)
    return return_types_sort(gather(input, dim, idx), idx)

def msort(input: Tensor) -> Tensor:
    return sort(input, 0).values

def topk(input: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> return_types_sort:
    d = _wrap(dim, len(input._shape))
    idx = narrow(argsort(input, d, largest, True), d, 0, k).contiguous()
    return return_types_sort(gather(input, d, idx), idx)

def kthvalue(input: Tensor, k: int, dim: int = -1, keepdim: bool = False) -> return_types_sort:
    d = _wrap(dim, len(input._shape))
    idx = narrow(argsort(input, d, False, True), d, k - 1, 1).contiguous()
    v = gather(input, d, idx)
    if not keepdim:
        return return_types_sort(squeeze(v, d), squeeze(idx, d))
    return return_types_sort(v, idx)

def median(input: Tensor) -> Tensor:
    f = reshape(input, [-1])
    n = f._shape[0]
    return kthvalue(f, (n + 1) // 2, 0).values

def median(input: Tensor, dim: int, keepdim: bool = False) -> return_types_sort:
    d = _wrap(dim, len(input._shape))
    n = input._shape[d]
    return kthvalue(input, (n + 1) // 2, d, keepdim)

def quantile(input: Tensor, q: float, dim: Optional[int] = None, keepdim: bool = False, interpolation: str = "linear") -> Tensor:
    x = reshape(input, [-1]) if dim is None else input
    d = 0 if dim is None else _wrap(dim, len(input._shape))
    s = sort(x, d).values
    n = s._shape[d]
    pos = q * (n - 1)
    lo = int(_math.floor(pos))
    hi = _imin(lo + 1, n - 1)
    frac_ = pos - lo
    a = select(s, d, lo)
    b = select(s, d, hi)
    r = add(a, mul(sub(b, a), frac_))
    if keepdim:
        r = unsqueeze(r, d)
    return r

def unique(input: Tensor, sorted: bool = True, return_inverse: bool = False, return_counts: bool = False, dim: Optional[int] = None) -> Tensor:
    v = sort(reshape(input.detach(), [-1])).values
    vals = v.tolist()
    keep = List[int]()
    for i in range(len(vals)):
        if i == 0 or vals[i] != vals[i - 1]:
            keep.append(i)
    idx = _new([len(keep)], _I64)
    p = Ptr[int](idx._st.ptr)
    for i in range(len(keep)):
        p[i] = keep[i]
    return index_select(v, 0, idx)

def bincount(input: Tensor, weights: Optional[Tensor] = None, minlength: int = 0) -> Tensor:
    vals = input.tolist()
    m = minlength
    for v in vals:
        m = _imax(m, int(v) + 1)
    if weights is None:
        out = zeros([m], dtype=_DType(_I64))
        p = Ptr[int](out._st.ptr)
        for v in vals:
            p[int(v)] += 1
        return out
    w = _unw(weights).tolist()
    out = zeros([m], dtype=_DType(_F64 if _unw(weights)._dt == _F64 else _F32))
    acc = [0.0] * m
    for i in range(len(vals)):
        acc[int(vals[i])] += w[i]
    for i in range(m):
        if out._dt == _F32:
            Ptr[float32](out._st.ptr)[i] = float32(acc[i])
        else:
            Ptr[float](out._st.ptr)[i] = acc[i]
    return out

def equal(input: Tensor, other: Tensor) -> bool:
    if input._shape != other._shape:
        return False
    return _item_f64(all(eq(input, other))) != 0.0

def allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool:
    return _item_f64(all(isclose(input, other, rtol, atol, equal_nan))) != 0.0

def isclose(input: Tensor, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> Tensor:
    a = input.double()
    b = other.double()
    close = le(abs(sub(a, b)), add(mul(abs(b), rtol), atol))
    close = logical_or(close, eq(a, b))
    if equal_nan:
        close = logical_or(close, logical_and(isnan(a), isnan(b)))
    return close

# ================================================================ linear algebra core

from numpy.linalg.blas import cblas_sgemm, cblas_dgemm, fint

def _gemm(z, a: Tensor, b: Tensor, out: Tensor, M: int, K: int, N: int, ao: int, bo: int, oo: int):
    T = type(z)
    pa = Ptr[T](a._st.ptr) + a._off + ao
    pb = Ptr[T](b._st.ptr) + b._off + bo
    po = Ptr[T](out._st.ptr) + oo
    if M == 0 or N == 0:
        return
    if K == 0:
        for i in range(M * N):
            po[i] = T()
        return
    if isinstance(z, float32):
        cblas_sgemm(fint(101), fint(111), fint(111), fint(M), fint(N), fint(K), float32(1.0), pa, fint(K), pb, fint(N), float32(0.0), po, fint(N))
    elif isinstance(z, float):
        cblas_dgemm(fint(101), fint(111), fint(111), fint(M), fint(N), fint(K), 1.0, pa, fint(K), pb, fint(N), 0.0, po, fint(N))
    else:
        for i in range(M):
            for j in range(N):
                acc = T()
                for k in range(K):
                    if isinstance(z, bool):
                        acc = acc or (pa[i * K + k] and pb[k * N + j])
                    else:
                        acc += pa[i * K + k] * pb[k * N + j]
                po[i * N + j] = acc

def _gemm_bt(z, a: Tensor, b: Tensor, out: Tensor, M: int, K: int, N: int):
    # out[M,N] = a[M,K] @ b^T where b is a contiguous [N,K] buffer (b = w.t()
    # of a contiguous weight): BLAS transposed operand, no copy of the weight.
    T = type(z)
    pa = Ptr[T](a._st.ptr) + a._off
    pb = Ptr[T](b._st.ptr) + b._off
    po = Ptr[T](out._st.ptr)
    if isinstance(z, float32):
        cblas_sgemm(fint(101), fint(111), fint(112), fint(M), fint(N), fint(K), float32(1.0), pa, fint(K), pb, fint(K), float32(0.0), po, fint(N))
    else:
        cblas_dgemm(fint(101), fint(111), fint(112), fint(M), fint(N), fint(K), 1.0, pa, fint(K), pb, fint(K), 0.0, po, fint(N))

def mm(input: Tensor, mat2: Tensor) -> Tensor:
    if len(input._shape) != 2 or len(mat2._shape) != 2:
        raise RuntimeError("self must be a matrix")
    M = input._shape[0]
    K = input._shape[1]
    if mat2._shape[0] != K:
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (" + str(M) + "x" + str(K) + " and " + str(mat2._shape[0]) + "x" + str(mat2._shape[1]) + ")")
    N = mat2._shape[1]
    if input._dt != mat2._dt:
        raise RuntimeError("expected m1 and m2 to have the same dtype, but got: " + str(input.dtype) + " != " + str(mat2.dtype))
    dt = input._dt
    a = _contig_as(input, dt)
    out = _new([M, N], dt)
    if (dt == _F32 or dt == _F64) and M > 0 and N > 0 and K > 0 and mat2._strides[0] == 1 and mat2._strides[1] == K:
        if dt == _F32:
            _gemm_bt(float32(), a, mat2, out, M, K, N)
        else:
            _gemm_bt(0.0, a, mat2, out, M, K, N)
    else:
        b = _contig_as(mat2, dt)
        _dispatch(dt, _gemm, a, b, out, M, K, N, 0, 0, 0)
    if _needs([input, mat2]):
        _attach(out, _node("MmBackward0", _K_MM, [input, mat2]))
    return out

def bmm(input: Tensor, mat2: Tensor) -> Tensor:
    if len(input._shape) != 3 or len(mat2._shape) != 3:
        raise RuntimeError("batch1 must be a 3D tensor")
    B = input._shape[0]
    M = input._shape[1]
    K = input._shape[2]
    N = mat2._shape[2]
    if mat2._shape[0] != B or mat2._shape[1] != K:
        raise RuntimeError("Expected size for first two dimensions of batch2 tensor to be: [" + str(B) + ", " + str(K) + "] but got: [" + str(mat2._shape[0]) + ", " + str(mat2._shape[1]) + "].")
    if input._dt != mat2._dt:
        raise RuntimeError("expected scalar type " + input.dtype._name() + " but found " + mat2.dtype._name())
    dt = input._dt
    a = _contig_as(input, dt)
    b = _contig_as(mat2, dt)
    out = _new([B, M, N], dt)
    for i in range(B):
        _dispatch(dt, _gemm, a, b, out, M, K, N, i * M * K, i * K * N, i * M * N)
    if _needs([input, mat2]):
        _attach(out, _node("BmmBackward0", _K_BMM, [input, mat2]))
    return out

def matmul(input: Tensor, other: Tensor) -> Tensor:
    na = len(input._shape)
    nb = len(other._shape)
    if na == 0 or nb == 0:
        raise RuntimeError("both arguments to matmul need to be at least 1D, but they are " + str(na) + "D and " + str(nb) + "D")
    if na == 1 and nb == 1:
        return dot(input, other)
    if na == 2 and nb == 1:
        return squeeze(mm(input, unsqueeze(other, 1)), 1)
    if na == 1 and nb == 2:
        return squeeze(mm(unsqueeze(input, 0), other), 0)
    if na == 2 and nb == 2:
        return mm(input, other)
    if na >= 3 and nb <= 2:
        # fold batch into rows
        b2 = other if nb == 2 else unsqueeze(other, 1)
        K = input._shape[-1]
        lead = input._shape[:-1]
        r = mm(reshape(input, [-1, K]), b2)
        if nb == 2:
            return reshape(r, lead + [b2._shape[1]])
        return reshape(r, lead)
    a = input if na > 1 else unsqueeze(input, 0)
    b = other if nb > 1 else unsqueeze(other, 1)
    ba = a._shape[:-2]
    bb = b._shape[:-2]
    bs = _broadcast_shapes(ba, bb)
    M = a._shape[-2]
    K = a._shape[-1]
    N = b._shape[-1]
    ae = reshape(expand(a, bs + [M, K]), [-1, M, K])
    be = reshape(expand(b, bs + [K, N]), [-1, K, N])
    r = reshape(bmm(ae, be), bs + [M, N])
    if na == 1:
        r = squeeze(r, -2)
    if nb == 1:
        r = squeeze(r, -1)
    return r

def dot(input: Tensor, other: Tensor) -> Tensor:
    if len(input._shape) != 1 or len(other._shape) != 1:
        raise RuntimeError("1D tensors expected, but got " + str(len(input._shape)) + "D and " + str(len(other._shape)) + "D tensors")
    return reshape(mm(unsqueeze(input, 0), unsqueeze(other, 1)), List[int]())

def inner(*args, **kwargs):
    return dot(*args, **kwargs)
def vdot(*args, **kwargs):
    return dot(*args, **kwargs)

def mv(input: Tensor, vec: Tensor) -> Tensor:
    return matmul(input, vec)

def outer(input: Tensor, vec2: Tensor) -> Tensor:
    return mul(unsqueeze(input, 1), unsqueeze(vec2, 0))

def ger(*args, **kwargs):
    return outer(*args, **kwargs)

def addmm(input: Tensor, mat1: Tensor, mat2: Tensor, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
    r = mm(mat1, mat2)
    if alpha != 1.0:
        r = mul(r, alpha)
    if beta == 0.0:
        return r
    return add(mul(input, beta) if beta != 1.0 else input, r)

def baddbmm(input: Tensor, batch1: Tensor, batch2: Tensor, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
    r = bmm(batch1, batch2)
    if alpha != 1.0:
        r = mul(r, alpha)
    if beta == 0.0:
        return r
    return add(mul(input, beta) if beta != 1.0 else input, r)

def diag(input: Tensor, diagonal: int = 0) -> Tensor:
    nd = len(input._shape)
    if nd == 1:
        n = input._shape[0]
        m = n + (diagonal if diagonal > 0 else -diagonal)
        out = zeros([m, m], dtype=input.dtype)
        st = diagonal * out._strides[1] if diagonal >= 0 else -diagonal * out._strides[0]
        v = Tensor(out._st, st, [n], [out._strides[0] + out._strides[1]], out._dt)
        _copy_into(v, input)
        if _needs([input]):
            nn = _node("DiagEmbedBackward0", _K_DIAG, [input])
            nn.ints = [n]
            nn.ints2 = [diagonal]
            _attach(out, nn)
        return out
    elif nd == 2:
        r = diagonal_(input, diagonal)
        out = clone(r)
        return out
    raise RuntimeError("diag(): Supports 1D or 2D tensors. Got " + str(nd) + "D")

def diagonal_(input: Tensor, offset: int = 0) -> Tensor:
    R = input._shape[0]
    C = input._shape[1]
    if offset >= 0:
        n = _imax(0, _imin(R, C - offset))
        off = input._off + offset * input._strides[1]
    else:
        n = _imax(0, _imin(R + offset, C))
        off = input._off - offset * input._strides[0]
    r = Tensor(input._st, off if n > 0 else input._off, [n], [input._strides[0] + input._strides[1]], input._dt)
    if _needs([input]):
        nn = _node("DiagonalBackward0", _K_STRIDED, [input])
        nn.ints = input._shape.copy()
        cs = _contig_strides(input._shape)
        o2 = (offset * cs[1]) if offset >= 0 else (-offset * cs[0])
        nn.ints2 = [o2 if n > 0 else 0, 1, n, cs[0] + cs[1]]
        _attach(r, nn)
    return r

def diagonal(input: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor:
    nd = len(input._shape)
    a = _wrap(dim1, nd)
    b = _wrap(dim2, nd)
    if nd == 2 and a == 0 and b == 1:
        return diagonal_(input, offset)
    # general: move dims last and use batched gather via strided view
    x = movedim(input, [a, b], [nd - 2, nd - 1]).contiguous()
    R = x._shape[-2]
    C = x._shape[-1]
    n = _imax(0, _imin(R, C - offset)) if offset >= 0 else _imax(0, _imin(R + offset, C))
    shp = x._shape[:-2] + [n]
    st = x._strides[:-2] + [x._strides[-2] + x._strides[-1]]
    off = x._off + (offset * x._strides[-1] if offset >= 0 else -offset * x._strides[-2])
    r = Tensor(x._st, off, shp, st, x._dt)
    if _needs([x]):
        nn = _node("DiagonalBackward0", _K_STRIDED, [x])
        nn.ints = x._shape.copy()
        nn.ints2 = [off - x._off, len(shp)] + shp + st
        _attach(r, nn)
    return r

def diag_embed(input: Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> Tensor:
    lead = input._shape[:-1]
    n = input._shape[-1]
    flat = reshape(input, [-1, n])
    rows = [diag(select(flat, 0, i), offset) for i in range(flat._shape[0])]
    m = n + (offset if offset >= 0 else -offset)
    return reshape(stack(rows, 0), lead + [m, m])

def trace(input: Tensor) -> Tensor:
    return sum(diagonal_(input, 0))

def _tri(input: Tensor, diagonal: int, upper: bool) -> Tensor:
    R = input._shape[-2]
    C = input._shape[-1]
    m = _new([R, C], _BOOL)
    p = Ptr[bool](m._st.ptr)
    for i in range(R):
        for j in range(C):
            keep = (j - i >= diagonal) if upper else (j - i <= diagonal)
            p[i * C + j] = not keep
    out = masked_fill(input.detach(), m, 0)
    if _needs([input]):
        nn = _node("TriuBackward0" if upper else "TrilBackward0", _K_TRIL, [input])
        nn.ints2 = [diagonal, 0 if upper else 1]
        _attach(out, nn)
    return out

def triu(input: Tensor, diagonal: int = 0) -> Tensor:
    return _tri(input, diagonal, True)

def tril(input: Tensor, diagonal: int = 0) -> Tensor:
    return _tri(input, diagonal, False)

def kron(input: Tensor, other: Tensor) -> Tensor:
    a = input
    b = other
    if len(a._shape) == 1 and len(b._shape) == 1:
        return reshape(outer(a, b), [-1])
    ra = a._shape[0]
    ca = a._shape[1]
    rb = b._shape[0]
    cb = b._shape[1]
    r = mul(reshape(a, [ra, 1, ca, 1]), reshape(b, [1, rb, 1, cb]))
    return reshape(r, [ra * rb, ca * cb])

def cross(input: Tensor, other: Tensor, dim: int = -1) -> Tensor:
    d = _wrap(dim, len(input._shape))
    a0 = select(input, d, 0); a1 = select(input, d, 1); a2 = select(input, d, 2)
    b0 = select(other, d, 0); b1 = select(other, d, 1); b2 = select(other, d, 2)
    return stack([sub(mul(a1, b2), mul(a2, b1)), sub(mul(a2, b0), mul(a0, b2)), sub(mul(a0, b1), mul(a1, b0))], d)

def cdist(x1: Tensor, x2: Tensor, p: float = 2.0) -> Tensor:
    d = sub(unsqueeze(x1, -2), unsqueeze(x2, -3))
    return norm(d, p, -1)

def einsum(equation: str, *operands) -> Tensor:
    ops = List[Tensor]()
    for o in operands:
        if isinstance(o, List[Tensor]):
            for q in o:
                ops.append(q)
        else:
            ops.append(o)
    return _einsum(equation, ops)

def _einsum(equation: str, ops: List[Tensor]) -> Tensor:
    eq_ = equation.replace(" ", "")
    lhs = eq_
    rhs = ""
    has_arrow = "->" in eq_
    if has_arrow:
        k = eq_.find("->")
        lhs = eq_[:k]
        rhs = eq_[k + 2:]
    terms = lhs.split(",")
    if len(terms) != len(ops):
        raise RuntimeError("einsum(): more operands were provided than specified in the equation")
    # expand ellipsis to capital-free placeholder letters
    ell_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    maxell = 0
    for i in range(len(terms)):
        if "..." in terms[i]:
            maxell = _imax(maxell, len(ops[i]._shape) - (len(terms[i]) - 3))
    for i in range(len(terms)):
        if "..." in terms[i]:
            ne_ = len(ops[i]._shape) - (len(terms[i]) - 3)
            terms[i] = terms[i].replace("...", ell_letters[maxell - ne_:maxell])
    if not has_arrow:
        cnt = Dict[str, int]()
        for tm in terms:
            for ch in tm:
                cnt[ch] = cnt.get(ch, 0) + 1
        ell = "".join([c for c in ell_letters[:maxell]])
        rest = sorted([c for c, v in cnt.items() if v == 1 and c not in ell_letters])
        rhs = ell + "".join(rest)
    else:
        rhs = rhs.replace("...", ell_letters[:maxell])
    # gather letter sizes
    sizes = Dict[str, int]()
    for i in range(len(terms)):
        tm = terms[i]
        if len(tm) != len(ops[i]._shape):
            raise RuntimeError("einsum(): the number of subscripts in the equation (" + str(len(tm)) + ") does not match the number of dimensions (" + str(len(ops[i]._shape)) + ") for operand " + str(i))
        for j in range(len(tm)):
            s = ops[i]._shape[j]
            if tm[j] in sizes and sizes[tm[j]] != s and s != 1:
                if sizes[tm[j]] == 1:
                    sizes[tm[j]] = s
                else:
                    raise RuntimeError("einsum(): operands do not broadcast with remapped shapes")
            elif tm[j] not in sizes:
                sizes[tm[j]] = s
    # handle repeated letters within a term by taking diagonals
    cur_terms = List[str]()
    cur_ops = List[Tensor]()
    for i in range(len(terms)):
        tm = terms[i]
        x = ops[i]
        changed = True
        while changed:
            changed = False
            for a in range(len(tm)):
                b = tm.find(tm[a], a + 1)
                if b >= 0:
                    x = diagonal(x, 0, a, b)
                    tm = tm[:a] + tm[a + 1:b] + tm[b + 1:] + tm[a]
                    changed = True
                    break
        cur_terms.append(tm)
        cur_ops.append(x)
    # contract left to right
    acc_t = cur_terms[0]
    acc = cur_ops[0]
    for i in range(1, len(cur_terms)):
        bt = cur_terms[i]
        b = cur_ops[i]
        later = rhs + "".join(cur_terms[i + 1:])
        acc, acc_t = _einsum_pair(acc, acc_t, b, bt, later, sizes)
    # final: sum out letters not in rhs, permute
    sumd = [j for j in range(len(acc_t)) if acc_t[j] not in rhs]
    if len(sumd) > 0:
        acc = sum(acc, sumd)
        acc_t = "".join([c for c in acc_t if c in rhs])
    perm = [acc_t.find(c) for c in rhs]
    if len(perm) > 0:
        acc = permute(acc, perm)
    return acc

def _einsum_pair(a: Tensor, at: str, b: Tensor, bt: str, later: str, sizes: Dict[str, int]) -> Tuple[Tensor, str]:
    # classify letters
    batch = "".join([c for c in at if c in bt and c in later])
    contr = "".join([c for c in at if c in bt and c not in later])
    a_only = "".join([c for c in at if c not in bt])
    b_only = "".join([c for c in bt if c not in at])
    # sum out a-only/b-only letters that are not needed later
    a_keep = "".join([c for c in a_only if c in later])
    b_keep = "".join([c for c in b_only if c in later])
    a_sum = [at.find(c) for c in a_only if c not in later]
    if len(a_sum) > 0:
        a = sum(a, a_sum)
        at = "".join([c for c in at if c not in a_only or c in later])
    b_sum = [bt.find(c) for c in b_only if c not in later]
    if len(b_sum) > 0:
        b = sum(b, b_sum)
        bt = "".join([c for c in bt if c not in b_only or c in later])
    def sz(s: str) -> int:
        p = 1
        for c in s:
            p *= sizes[c]
        return p
    ap = permute(a, [at.find(c) for c in batch + a_keep + contr])
    bp = permute(b, [bt.find(c) for c in batch + contr + b_keep])
    ap = expand(ap, [sizes[c] for c in batch + a_keep + contr])
    bp = expand(bp, [sizes[c] for c in batch + contr + b_keep])
    A = reshape(ap, [sz(batch), sz(a_keep), sz(contr)])
    B = reshape(bp, [sz(batch), sz(contr), sz(b_keep)])
    if A._dt != B._dt:
        dt = _promote2(A._dt, B._dt)
        A = A.to(_DType(dt))
        B = B.to(_DType(dt))
    R = bmm(A, B)
    outt = batch + a_keep + b_keep
    return (reshape(R, [sizes[c] for c in outt]), outt)

def tensordot(a: Tensor, b: Tensor, dims = 2) -> Tensor:
    if isinstance(dims, int):
        na = len(a._shape)
        da = list(range(na - dims, na))
        db = list(range(dims))
    else:
        da = _to_list(dims[0])
        db = _to_list(dims[1])
    ka = [i for i in range(len(a._shape)) if i not in da]
    kb = [i for i in range(len(b._shape)) if i not in db]
    A = reshape(permute(a, ka + da), [_numel([a._shape[i] for i in ka]), -1])
    B = reshape(permute(b, db + kb), [-1, _numel([b._shape[i] for i in kb])])
    return reshape(mm(A, B), [a._shape[i] for i in ka] + [b._shape[i] for i in kb])

# ================================================================ padding (core kernel for F.pad)

def _pad(input: Tensor, pad: List[int], mode: str = "constant", value: float = 0.0) -> Tensor:
    if len(pad) % 2 != 0:
        raise RuntimeError("Padding length must be divisible by 2")
    x = input
    nd = len(x._shape)
    npairs = len(pad) // 2
    if npairs > nd:
        raise RuntimeError("Padding length too large")
    for k in range(npairs):
        d = nd - 1 - k
        lo = pad[2 * k]
        hi = pad[2 * k + 1]
        if lo == 0 and hi == 0:
            continue
        n = x._shape[d]
        if mode == "constant":
            if lo < 0:
                x = narrow(x, d, -lo, x._shape[d] + lo)
                lo = 0
            if hi < 0:
                x = narrow(x, d, 0, x._shape[d] + hi)
                hi = 0
            parts = List[Tensor]()
            if lo > 0:
                s = x._shape.copy()
                s[d] = lo
                parts.append(full(s, value, dtype=x.dtype))
            parts.append(x)
            if hi > 0:
                s = x._shape.copy()
                s[d] = hi
                parts.append(full(s, value, dtype=x.dtype))
            x = cat(parts, d)
        else:
            idx = _new([n + lo + hi], _I64)
            p = Ptr[int](idx._st.ptr)
            for i in range(n + lo + hi):
                j = i - lo
                if mode == "reflect":
                    if n == 1:
                        j = 0
                    else:
                        per = 2 * (n - 1)
                        j = j % per
                        if j < 0:
                            j += per
                        if j >= n:
                            j = per - j
                elif mode == "replicate":
                    j = _imin(_imax(j, 0), n - 1)
                elif mode == "circular":
                    j = j % n
                    if j < 0:
                        j += n
                else:
                    raise NotImplementedError("pad mode " + mode)
                p[i] = j
            x = index_select(x, d, idx)
    return x

# ================================================================ formatting

def _fmt_float(v: float, sci: bool, dec: int) -> str:
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == -float("inf"):
        return "-inf"
    if sci:
        return f"{v:.{dec}e}"
    return f"{v:.{dec}f}"

def _format(x: Tensor) -> str:
    if _is_complex(x._dt):
        r = _as_real_raw(x.contiguous())
        v = r.tolist()
        parts = List[str]()
        for i in range(len(v) // 2):
            im = v[2 * i + 1]
            parts.append(f"{v[2*i]:.4f}" + ("+" if im >= 0 else "-") + f"{(im if im >= 0 else -im):.4f}j")
        return "tensor(" + str(x._shape) + ": [" + ", ".join(parts) + "], dtype=" + str(x.dtype) + ")"
    vals = x.tolist()
    nd = len(x._shape)
    isf = _is_float(x._dt)
    strs = List[str]()
    sci = False
    allint = True
    if isf:
        mx = 0.0
        mn = float("inf")
        for v in vals:
            if v == v and v != float("inf") and v != -float("inf"):
                av = v if v >= 0 else -v
                if av > mx:
                    mx = av
                if av != 0.0 and av < mn:
                    mn = av
                if v != _math.floor(v):
                    allint = False
        if mx >= 1e8 or (mn != float("inf") and mx / mn > 1000.0 and mn < 1e-4):
            sci = True
    for v in vals:
        if x._dt == _BOOL:
            strs.append("True" if v != 0.0 else "False")
        elif isf:
            if sci:
                strs.append(_fmt_float(v, True, 4))
            elif allint:
                strs.append(_fmt_float(v, False, 0) + ".")
            else:
                strs.append(_fmt_float(v, False, 4))
        else:
            strs.append(str(int(v)))
    w = 0
    for s in strs:
        w = _imax(w, len(s))
    strs = [" " * (w - len(s)) + s for s in strs]
    body = ""
    if nd == 0:
        body = strs[0].strip() if len(strs) > 0 else ""
    else:
        body = _fmt_rec(strs, x._shape, 0, 0, 7)[0]
    extra = ""
    if x._dt != _F32 and x._dt != _I64 and x._dt != _BOOL:
        extra += ", dtype=" + str(x.dtype)
    elif x.numel() == 0 and x._dt != _F32:
        extra += ", dtype=" + str(x.dtype)
    if x.grad_fn is not None:
        extra += ", grad_fn=<" + _unw(x.grad_fn).name() + ">"
    elif x.requires_grad:
        extra += ", requires_grad=True"
    return "tensor(" + body + extra + ")"

def _fmt_rec(strs: List[str], shape: List[int], d: int, start: int, indent: int) -> Tuple[str, int]:
    n = shape[d]
    if d == len(shape) - 1:
        parts = [strs[start + i] for i in range(n)]
        return ("[" + ", ".join(parts) + "]", start + n)
    sub_ = List[str]()
    pos = start
    for i in range(n):
        s, pos = _fmt_rec(strs, shape, d + 1, pos, indent + 1)
        sub_.append(s)
    sep = "," + "\n" * (len(shape) - d - 1) + " " * (indent + d + 1 - 7 + 7)
    return ("[" + sep.join(sub_) + "]", pos)

# ================================================================ factories

def _dt_or(dtype: Optional[_DType], default: int) -> int:
    if dtype is None:
        return default
    return _unw(dtype).code

def _finish(t: Tensor, requires_grad: bool) -> Tensor:
    if requires_grad:
        t.requires_grad = True
    return t

def empty(*size, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None, layout = None, pin_memory: bool = False, memory_format = None) -> Tensor:
    return _finish(_new(_size_args(size), _dt_or(dtype, _F32)), requires_grad)

def zeros(*size, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None, layout = None) -> Tensor:
    t = _new(_size_args(size), _dt_or(dtype, _F32))
    _fill(t, 0.0)
    return _finish(t, requires_grad)

def ones(*size, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None, layout = None) -> Tensor:
    t = _new(_size_args(size), _dt_or(dtype, _F32))
    _fill(t, 1.0)
    return _finish(t, requires_grad)

def full(size, fill_value, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None, layout = None) -> Tensor:
    dflt = _F32
    if isinstance(fill_value, bool):
        dflt = _BOOL
    elif isinstance(fill_value, int):
        dflt = _I64
    t = _new(_to_list(size), _dt_or(dtype, dflt))
    _dispatch(t._dt, _k_fill_any, t, fill_value)
    return _finish(t, requires_grad)

def _k_fill_any(z, t: Tensor, v):
    T = type(z)
    x = _cv(v, T)
    p = Ptr[T](t._st.ptr) + t._off
    for i in range(t.numel()):
        p[i] = x

def zeros_like(input: Tensor, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None, memory_format = None) -> Tensor:
    return zeros(input._shape.copy(), dtype=_DType(_dt_or(dtype, input._dt)), requires_grad=requires_grad)

def ones_like(input: Tensor, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None, memory_format = None) -> Tensor:
    return ones(input._shape.copy(), dtype=_DType(_dt_or(dtype, input._dt)), requires_grad=requires_grad)

def empty_like(input: Tensor, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None, memory_format = None) -> Tensor:
    return empty(input._shape.copy(), dtype=_DType(_dt_or(dtype, input._dt)), requires_grad=requires_grad)

def full_like(input: Tensor, fill_value, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None, memory_format = None) -> Tensor:
    return full(input._shape.copy(), fill_value, dtype=_DType(_dt_or(dtype, input._dt)), requires_grad=requires_grad)

def rand_like(input: Tensor, dtype: Optional[_DType] = None, generator: Optional[Generator] = None) -> Tensor:
    return rand(input._shape.copy(), dtype=_DType(_dt_or(dtype, input._dt)), generator=generator)

def randn_like(input: Tensor, dtype: Optional[_DType] = None, generator: Optional[Generator] = None) -> Tensor:
    return randn(input._shape.copy(), dtype=_DType(_dt_or(dtype, input._dt)), generator=generator)

def scalar_tensor(s, dtype: Optional[_DType] = None) -> Tensor:
    return full(List[int](), float(s), dtype=_DType(_dt_or(dtype, _F32)))

def eye(n: int, m: Optional[int] = None, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None) -> Tensor:
    mm_ = n if m is None else m
    t = zeros([n, mm_], dtype=_DType(_dt_or(dtype, _F32)))
    for i in range(_imin(n, mm_)):
        _dispatch(t._dt, _k_set_at, t, i * mm_ + i, 1.0)
    return _finish(t, requires_grad)

def _k_set_at(z, t: Tensor, off: int, v: float):
    T = type(z)
    Ptr[T](t._st.ptr)[off] = _cv(v, T)

def arange(start, end = None, step = 1, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None) -> Tensor:
    if end is None:
        return arange(0, start, step, dtype=dtype, requires_grad=requires_grad)
    else:
        isint = isinstance(start, int) and isinstance(end, int) and isinstance(step, int)
        dflt = _I64 if isint else _F32
        dt = _dt_or(dtype, dflt)
        s = float(start)
        e = float(end)
        st = float(step)
        if st == 0.0:
            raise RuntimeError("step must be nonzero")
        n = int(_math.ceil((e - s) / st))
        if isint:
            n = (int(end) - int(start) + int(step) - (1 if int(step) > 0 else -1)) // int(step)
        if n < 0:
            n = 0
        t = _new([n], dt)
        _dispatch(dt, _k_arange, t, start, step, n)
        return _finish(t, requires_grad)

def _k_arange(z, t: Tensor, start, step, n: int):
    T = type(z)
    p = Ptr[T](t._st.ptr)
    if isinstance(z, float) or isinstance(z, float32):
        s = float(start)
        st = float(step)
        for i in range(n):
            p[i] = T(s + i * st)
    else:
        for i in range(n):
            p[i] = _cv(start + i * step, T)

def range_(start: float, end: float, step: float = 1.0, dtype: Optional[_DType] = None) -> Tensor:
    n = int(_math.floor((end - start) / step)) + 1
    t = _new([n], _dt_or(dtype, _F32))
    _dispatch(t._dt, _k_arange, t, start, step, n)
    return t

def linspace(start: float, end: float, steps: int, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None) -> Tensor:
    dt = _dt_or(dtype, _F32)
    t = _new([steps], dt)
    _dispatch(dt, _k_linspace, t, float(start), float(end), steps)
    return _finish(t, requires_grad)

def _k_linspace(z, t: Tensor, start: float, end: float, steps: int):
    # torch: first half from start, second half from end (symmetric)
    T = type(z)
    p = Ptr[T](t._st.ptr)
    if steps == 1:
        p[0] = _cv(start, T)
        return
    if isinstance(z, float32):
        stp = (float32(end) - float32(start)) / float32(steps - 1)
        half = steps // 2
        for i in range(steps):
            if i < half:
                p[i] = float32(start) + stp * float32(i)
            else:
                p[i] = float32(end) - stp * float32(steps - i - 1)
    else:
        stp = (end - start) / (steps - 1)
        half = steps // 2
        for i in range(steps):
            if i < half:
                p[i] = _cv(start + stp * i, T)
            else:
                p[i] = _cv(end - stp * (steps - i - 1), T)

def logspace(start: float, end: float, steps: int, base: float = 10.0, dtype: Optional[_DType] = None) -> Tensor:
    return pow(base, linspace(start, end, steps, dtype=_DType(_F64))).to(_DType(_dt_or(dtype, _F32)))

def meshgrid(*tensors, indexing: str = "ij") -> List[Tensor]:
    ts = List[Tensor]()
    for x in tensors:
        if isinstance(x, List[Tensor]):
            for q in x:
                ts.append(q)
        else:
            ts.append(x)
    n = len(ts)
    shp = [x.numel() for x in ts]
    if indexing == "xy" and n >= 2:
        shp[0], shp[1] = shp[1], shp[0]
    out = List[Tensor]()
    for i in range(n):
        v = [1] * n
        k = i
        if indexing == "xy" and n >= 2 and i < 2:
            k = 1 - i
        v[k] = ts[i].numel()
        out.append(expand(reshape(ts[i], v), shp))
    return out

# ---------------------------------------------------------------- tensor()/numpy

def _flatten_py(x, out: List[float], shp: List[int], depth: int):
    if isinstance(x, List) or isinstance(x, Tuple):
        if len(shp) <= depth:
            shp.append(len(x))
        elif shp[depth] != len(x):
            raise ValueError("expected sequence of length " + str(shp[depth]) + " at dim " + str(depth) + " (got " + str(len(x)) + ")")
        for e in x:
            _flatten_py(e, out, shp, depth + 1)
    elif isinstance(x, Tensor):
        for s in x._shape:
            if len(shp) <= depth:
                shp.append(s)
            depth += 1
        for v in x.tolist():
            out.append(v)
    else:
        out.append(float(x))

def _elem_code(x) -> int:
    if isinstance(x, List):
        if len(x) > 0:
            return _elem_code(x[0])
        return _elem_code(type(x).T())
    elif isinstance(x, Tuple):
        if static.len(x) > 0:
            return _elem_code(x[0])
        return _F32
    elif isinstance(x, Tensor):
        return x._dt
    elif isinstance(x, bool):
        return _BOOL
    elif isinstance(x, int):
        return _I64
    elif isinstance(x, float32):
        return _F32
    elif isinstance(x, float):
        return _F32
    else:
        return _dtype_of_type(type(x))

def tensor(data, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None) -> Tensor:
    if isinstance(data, Tensor):
        r = _new(data._shape.copy(), _dt_or(dtype, data._dt))
        _copy_into(r, data)
        return _finish(r, requires_grad)
    elif isinstance(data, np.ndarray):
        src = from_numpy(data)
        r = _new(src._shape.copy(), _dt_or(dtype, src._dt))
        _copy_into(r, src)
        return _finish(r, requires_grad)
    elif isinstance(data, List) or isinstance(data, Tuple):
        vals = List[float]()
        shp = List[int]()
        _flatten_py(data, vals, shp, 0)
        dt = _dt_or(dtype, _elem_code(data))
        if dtype is None and _is_int_list_with_large(data):
            dt = _I64
        r = _new(shp, dt)
        if dt == _I64 and _is_int_like(data):
            ivals = List[int]()
            _flatten_int(data, ivals)
            p = Ptr[int](r._st.ptr)
            for i in range(len(ivals)):
                p[i] = ivals[i]
        else:
            _dispatch(dt, _k_from_list, r, vals)
        return _finish(r, requires_grad)
    else:
        dflt = _F32
        if isinstance(data, bool):
            dflt = _BOOL
        elif isinstance(data, int):
            dflt = _I64
        elif isinstance(data, float32):
            dflt = _F32
        elif not isinstance(data, float):
            dflt = _dtype_of_type(type(data))
        r = _new(List[int](), _dt_or(dtype, dflt))
        _dispatch(r._dt, _k_set_scalar, r, data)
        return _finish(r, requires_grad)

def _is_int_list_with_large(x) -> bool:
    return False

def _is_int_like(x) -> bool:
    if isinstance(x, List):
        return _is_int_like(type(x).T())
    elif isinstance(x, Tuple):
        if static.len(x) > 0:
            return _is_int_like(x[0])
        return False
    else:
        return isinstance(x, int)

def _flatten_int(x, out: List[int]):
    if isinstance(x, List) or isinstance(x, Tuple):
        for e in x:
            _flatten_int(e, out)
    elif isinstance(x, int):
        out.append(x)
    else:
        out.append(int(x))

def _k_from_list(z, r: Tensor, vals: List[float]):
    T = type(z)
    p = Ptr[T](r._st.ptr)
    for i in range(len(vals)):
        p[i] = _cv(vals[i], T)

def as_tensor(data, dtype: Optional[_DType] = None, device = None) -> Tensor:
    if isinstance(data, Tensor):
        if dtype is None or _unw(dtype).code == data._dt:
            return data
        return data.to(_unw(dtype))
    elif isinstance(data, np.ndarray):
        t = from_numpy(data)
        if dtype is None or _unw(dtype).code == t._dt:
            return t
        return t.to(_unw(dtype))
    else:
        return tensor(data, dtype=dtype)

def from_numpy(ndarray) -> Tensor:
    """Shares memory with the ndarray (like torch). Negative strides are copied."""
    a = ndarray
    T = a.dtype
    dt = _dtype_of_type(T)
    nd = static.len(a.shape)
    shp = List[int](nd)
    sts = List[int](nd)
    neg = False
    for i in static.range(static.len(a.shape)):
        shp.append(a.shape[i])
        s = a.strides[i]
        if s < 0:
            neg = True
        sts.append(s // _itemsize(dt))
    if neg:
        b = np.ascontiguousarray(a)
        return from_numpy(b)
    n = 1
    for v in shp:
        n *= v
    st = _Storage(a.data.as_byte(), n * _itemsize(dt))
    return Tensor(st, 0, shp, sts, dt)

def _to_numpy(x: Tensor, T: type, ndim: Literal[int]):
    if _dtype_of_type(T) != x._dt:
        raise TypeError("Tensor.numpy(): requested element type does not match tensor dtype " + str(x.dtype) + " (pass e.g. t.numpy(float, ndim) for float64)")
    if len(x._shape) != ndim:
        raise TypeError("Tensor.numpy(): requested ndim " + str(ndim) + " but tensor has " + str(len(x._shape)) + " dims")
    if x.requires_grad:
        raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
    sh = (0,) * ndim
    st = (0,) * ndim
    ps = Ptr[int](__ptr__(sh).as_byte())
    pt = Ptr[int](__ptr__(st).as_byte())
    isz = _itemsize(x._dt)
    for i in range(ndim):
        ps[i] = x._shape[i]
        pt[i] = x._strides[i] * isz
    return np.ndarray[T, ndim](sh, st, Ptr[T](x._st.ptr) + x._off)

# ================================================================ random

def _k_uniform(z, t: Tensor, a: float, b: float, g: Generator):
    T = type(z)
    p = Ptr[T](t._st.ptr)
    if t.is_contiguous() and (isinstance(z, float32) or isinstance(z, float)):
        # same draws and order as the offsets path, without materialising offsets
        q = p + t._off
        n = t.numel()
        if isinstance(z, float32):
            fa = float32(a)
            fb = float32(b)
            for i in range(n):
                u = g._uniform_f32()
                v = _fma32(u, fb - fa, fa)
                q[i] = fa if v == fb else v
        else:
            for i in range(n):
                u = g._uniform_f64()
                v = _fma64(u, b - a, a)
                q[i] = a if v == b else v
        return
    offs = _offsets(t._shape, t._strides, t._off)
    if isinstance(z, float32):
        fa = float32(a)
        fb = float32(b)
        for o in offs:
            u = g._uniform_f32()
            v = _fma32(u, fb - fa, fa)
            p[o] = fa if v == fb else v
    elif isinstance(z, float):
        for o in offs:
            u = g._uniform_f64()
            v = _fma64(u, b - a, a)
            p[o] = a if v == b else v
    else:
        raise RuntimeError("uniform_ expects a floating point tensor")

def _uniform_fill(t: Tensor, a: float, b: float, g: Generator):
    if not _is_float(t._dt):
        _randint_fill(t, int(a), int(b), g)
        return
    _dispatch_float(t._dt, _k_uniform, t, a, b, g)

@pure
@llvm
def _fma64(a: float, b: float, c: float) -> float:
    declare double @llvm.fma.f64(double, double, double)
    %y = call double @llvm.fma.f64(double %a, double %b, double %c)
    ret double %y

def _k_normal(z, t: Tensor, mean: float, std: float, g: Generator):
    T = type(z)
    n = t.numel()
    if not (isinstance(z, float32) or isinstance(z, float)):
        raise RuntimeError("normal_ expects a floating point tensor")
    if n >= 16 and t.is_contiguous():
        p = Ptr[T](t._st.ptr) + t._off
        for i in range(n):
            if isinstance(z, float32):
                p[i] = g._uniform_f32()
            else:
                p[i] = g._uniform_f64()
        i = 0
        while i < n - 15:
            _normal16(p + i, mean, std)
            i += 16
        if n % 16 != 0:
            q = p + (n - 16)
            for j in range(16):
                if isinstance(z, float32):
                    q[j] = g._uniform_f32()
                else:
                    q[j] = g._uniform_f64()
            _normal16(q, mean, std)
    else:
        p = Ptr[T](t._st.ptr)
        for o in _offsets(t._shape, t._strides, t._off):
            p[o] = T(g._normal_f64(mean, std))

# NormalFill16 scalar template (non-AVX2 CPU path, as used on arm64)
def _normal16(data, mean: float, std: float):
    T = type(data[0])
    if T is float32:
        m = float32(mean)
        s = float32(std)
        for j in range(8):
            u1 = float32(1.0) - data[j]
            u2 = data[j + 8]
            radius = _U.sqrt32(float32(-2.0) * _U.log32(u1))
            theta = float32(float(float32(2.0)) * 3.141592653589793 * float(u2))
            data[j] = float32(_fma64(float(radius * _U.cos32(theta)), float(s), float(m))) if False else _fma32(radius * _U.cos32(theta), s, m)
            data[j + 8] = _fma32(radius * _U.sin32(theta), s, m)
    else:
        for j in range(8):
            u1 = 1.0 - data[j]
            u2 = data[j + 8]
            radius = _math.sqrt(-2.0 * _math.log(u1))
            theta = 2.0 * 3.141592653589793 * u2
            data[j] = _fma64(radius * _math.cos(theta), std, mean)
            data[j + 8] = _fma64(radius * _math.sin(theta), std, mean)

@pure
@llvm
def _fma32(a: float32, b: float32, c: float32) -> float32:
    declare float @llvm.fma.f32(float, float, float)
    %y = call float @llvm.fma.f32(float %a, float %b, float %c)
    ret float %y

def _normal_fill(t: Tensor, mean: float, std: float, g: Generator):
    _dispatch_float(t._dt, _k_normal, t, mean, std, g)

def _k_bernoulli(z, t: Tensor, p_: float, g: Generator):
    T = type(z)
    p = Ptr[T](t._st.ptr)
    for o in _offsets(t._shape, t._strides, t._off):
        u = g._uniform_f64()
        p[o] = _cv(u < p_, T)

def _bernoulli_fill(t: Tensor, p: float, g: Generator):
    _dispatch(t._dt, _k_bernoulli, t, p, g)

def _k_bernoulli_t(z, t: Tensor, probs: Tensor, g: Generator):
    T = type(z)
    p = Ptr[T](t._st.ptr)
    offs = _offsets(t._shape, t._strides, t._off)
    pr = _contig_as(_expand_raw(probs, t._shape), probs._dt)
    if probs._dt == _F64:
        q = Ptr[float](pr._st.ptr)
        for i in range(len(offs)):
            u = g._uniform_f64()
            p[offs[i]] = _cv(u < q[i], T)
    else:
        q2 = Ptr[float32](_contig_as(pr, _F32)._st.ptr)
        for i in range(len(offs)):
            u = g._uniform_f32()
            p[offs[i]] = _cv(u < q2[i], T)

def bernoulli(input: Tensor, p: Optional[float] = None, generator: Optional[Generator] = None) -> Tensor:
    out = _new(input._shape.copy(), input._dt)
    if p is not None:
        _bernoulli_fill(out, _unw(p), _gen(generator))
    else:
        _dispatch(out._dt, _k_bernoulli_t, out, input, _gen(generator))
    return out

def _k_randint(z, t: Tensor, low: int, high: int, g: Generator):
    T = type(z)
    p = Ptr[T](t._st.ptr)
    rng = high - low
    if rng <= 0:
        raise RuntimeError("random_ expects 'from' to be less than 'to', but got from=" + str(low) + " >= to=" + str(high))
    big = rng >= (1 << 28)
    ur = u64(rng)
    for o in _offsets(t._shape, t._strides, t._off):
        v = 0
        if big:
            v = int(g.random64() % ur) + low
        else:
            v = int(u64(g.random()) % ur) + low
        p[o] = _cv(v, T)

def _randint_fill(t: Tensor, low: int, high: int, g: Generator):
    _dispatch(t._dt, _k_randint, t, low, high, g)

def _k_exponential(z, t: Tensor, lambd: float, g: Generator):
    T = type(z)
    p = Ptr[T](t._st.ptr)
    for o in _offsets(t._shape, t._strides, t._off):
        u = g._uniform_f64()
        p[o] = T(-1.0 / lambd * _math.log1p(-u))

def _exponential_fill(t: Tensor, lambd: float, g: Generator):
    _dispatch_float(t._dt, _k_exponential, t, lambd, g)

def manual_seed(seed: int) -> Generator:
    default_generator.manual_seed(seed)
    return default_generator

def seed() -> int:
    return default_generator.seed()

def initial_seed() -> int:
    return default_generator.initial_seed()

def get_rng_state():
    return default_generator.get_state()

def set_rng_state(new_state):
    default_generator.set_state(new_state)

def rand(*size, generator: Optional[Generator] = None, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None) -> Tensor:
    t = _new(_size_args(size), _dt_or(dtype, _F32))
    _uniform_fill(t, 0.0, 1.0, _gen(generator))
    return _finish(t, requires_grad)

def randn(*size, generator: Optional[Generator] = None, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None) -> Tensor:
    t = _new(_size_args(size), _dt_or(dtype, _F32))
    _normal_fill(t, 0.0, 1.0, _gen(generator))
    return _finish(t, requires_grad)

def randint(low: int, high = None, size = None, generator: Optional[Generator] = None, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None) -> Tensor:
    """randint(low, high, size) or randint(high, size)."""
    lo = low
    hi = 0
    shp = List[int]()
    if size is None:
        lo = 0
        hi = low
        shp = _to_list(high)
    else:
        hi = int(high)
        shp = _to_list(size)
    t = _new(shp, _dt_or(dtype, _I64))
    _randint_fill(t, lo, hi, _gen(generator))
    return t

def randint_like(input: Tensor, low: int, high: int, dtype: Optional[_DType] = None) -> Tensor:
    return randint(low, high, input._shape.copy(), dtype=_DType(_dt_or(dtype, input._dt)))

def randperm(n: int, generator: Optional[Generator] = None, dtype: Optional[_DType] = None, requires_grad: bool = False, device = None) -> Tensor:
    g = _gen(generator)
    r = List[int](n)
    for i in range(n):
        r.append(i)
    # randperm_cpu: Fisher-Yates with generator->random() % (n - i)
    for i in range(n - 1):
        z = int(u64(g.random()) % u64(n - i))
        tmp = r[i]
        r[i] = r[z + i]
        r[z + i] = tmp
    t = _new([n], _dt_or(dtype, _I64))
    _dispatch(t._dt, _k_from_ints, t, r)
    return t

def _k_from_ints(z, t: Tensor, vals: List[int]):
    T = type(z)
    p = Ptr[T](t._st.ptr)
    for i in range(len(vals)):
        p[i] = _cv(vals[i], T)

def normal(mean, std, size = None, generator: Optional[Generator] = None, dtype: Optional[_DType] = None, requires_grad: bool = False) -> Tensor:
    if isinstance(mean, Tensor):
        if isinstance(std, Tensor):
            shp = _broadcast_shapes(mean._shape, std._shape)
            z = _new(shp, mean._dt)
            _normal_fill(z, 0.0, 1.0, _gen(generator))
            return add(mul(z, std.detach()), mean.detach())
        else:
            z = _new(mean._shape.copy(), mean._dt)
            _normal_fill(z, 0.0, float(std), _gen(generator))
            return add(z, mean.detach())
    elif isinstance(std, Tensor):
        z = _new(std._shape.copy(), std._dt)
        _normal_fill(z, 0.0, 1.0, _gen(generator))
        return add(mul(z, std.detach()), float(mean))
    else:
        if size is None:
            raise TypeError("normal() missing required argument 'size'")
        t = _new(_to_list(size), _dt_or(dtype, _F32))
        _normal_fill(t, float(mean), float(std), _gen(generator))
        return _finish(t, requires_grad)

def multinomial(input: Tensor, num_samples: int, replacement: bool = False, generator: Optional[Generator] = None) -> Tensor:
    g = _gen(generator)
    probs = reshape(input.double(), [-1, input._shape[-1]])
    R = probs._shape[0]
    K = probs._shape[1]
    out = _new([R, num_samples], _I64)
    po = Ptr[int](out._st.ptr)
    for r in range(R):
        w = select(probs, 0, r).tolist()
        for s in range(num_samples):
            tot = 0.0
            for v in w:
                tot += v
            u = g._uniform_f64() * tot
            acc = 0.0
            pick = K - 1
            for k in range(K):
                acc += w[k]
                if u < acc:
                    pick = k
                    break
            po[r * num_samples + s] = pick
            if not replacement:
                w[pick] = 0.0
    if len(input._shape) == 1:
        return reshape(out, [num_samples])
    return out

# ================================================================ misc

def numel(input: Tensor) -> int:
    return input.numel()

def is_tensor(obj) -> bool:
    return isinstance(obj, Tensor)

def is_floating_point(input: Tensor) -> bool:
    return _is_float(input._dt)

def detach(input: Tensor) -> Tensor:
    return input.detach()

def squeeze_(input: Tensor) -> Tensor:
    return squeeze(input)

def addcmul(input: Tensor, tensor1: Tensor, tensor2: Tensor, value = 1) -> Tensor:
    return add(input, mul(mul(tensor1, tensor2), value))

def addcdiv(input: Tensor, tensor1: Tensor, tensor2: Tensor, value = 1) -> Tensor:
    return add(input, mul(div(tensor1, tensor2), value))

def lerp(input: Tensor, end: Tensor, weight) -> Tensor:
    return add(input, mul(sub(end, input), weight))

def atan2(input: Tensor, other: Tensor) -> Tensor:
    dt = _float_dt(_result_dt(input, other))
    shape, ca, cb = _prep2(input, other, dt)
    out = _new(shape, dt)
    _dispatch_float(dt, _k_bin, ca, cb, out, _U.atan2)
    if _needs([input, other]):
        _attach(out, _node("Atan2Backward0", _K_ATAN2, [input, other]))
    return out

def arctan2(*args, **kwargs):
    return atan2(*args, **kwargs)

def hypot(input: Tensor, other: Tensor) -> Tensor:
    return sqrt(add(mul(input, input), mul(other, other)))

def logaddexp(input: Tensor, other: Tensor) -> Tensor:
    m = maximum(input, other).detach()
    return add(m, log(add(exp(sub(input, m)), exp(sub(other, m)))))

def xlogy(input: Tensor, other: Tensor) -> Tensor:
    r = mul(input, log(other))
    return _where3(eq(input, 0), zeros_like(r), r)

def sinc(input: Tensor) -> Tensor:
    px = mul(input, 3.141592653589793)
    r = div(sin(px), px)
    return _where3(eq(input, 0), ones_like(r), r)

def softplus_(input: Tensor) -> Tensor:
    return log1p(exp(input))

def heaviside(input: Tensor, values: Tensor) -> Tensor:
    return _where3(eq(input, 0), values, _contig_as(gt(input, 0), input._dt))

def searchsorted(sorted_sequence: Tensor, input: Tensor, right: bool = False) -> Tensor:
    s = sorted_sequence.tolist()
    v = input.tolist()
    out = _new(input._shape.copy(), _I64)
    p = Ptr[int](out._st.ptr)
    for i in range(len(v)):
        lo = 0
        hi = len(s)
        while lo < hi:
            mid = (lo + hi) // 2
            if (s[mid] <= v[i]) if right else (s[mid] < v[i]):
                lo = mid + 1
            else:
                hi = mid
        p[i] = lo
    return out

def bucketize(input: Tensor, boundaries: Tensor, right: bool = False) -> Tensor:
    return searchsorted(boundaries, input, right)

def histc(input: Tensor, bins: int = 100, min: float = 0.0, max: float = 0.0) -> Tensor:
    v = input.tolist()
    lo = min
    hi = max
    if lo == 0.0 and hi == 0.0 and len(v) > 0:
        lo = v[0]
        hi = v[0]
        for x in v:
            if x < lo: lo = x
            if x > hi: hi = x
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    out = zeros([bins], dtype=input.dtype)
    cnt = [0.0] * bins
    for x in v:
        if x < lo or x > hi:
            continue
        b = int((x - lo) * bins / (hi - lo))
        if b == bins:
            b -= 1
        cnt[b] += 1.0
    _dispatch(out._dt, _k_from_list, out, cnt)
    return out

def cov(input: Tensor, correction: int = 1) -> Tensor:
    x = input if len(input._shape) == 2 else unsqueeze(input, 0)
    n = x._shape[1]
    xc = sub(x, mean(x, 1, True))
    c = div(mm(xc, t(xc)), float(n - correction))
    if len(input._shape) == 1:
        return reshape(c, List[int]())
    return c

def corrcoef(input: Tensor) -> Tensor:
    c = cov(input)
    d = sqrt(diagonal_(c, 0))
    return div(div(c, unsqueeze(d, 1)), unsqueeze(d, 0))

_DEFAULT_DT = [_F32]

def _random_int64_full(generator: Optional[Generator] = None) -> int:
    """torch.empty((), dtype=int64).random_(generator=g).item()."""
    g = _gen(generator)
    v = g.random64()
    return int(v % u64(0x8000000000000000))




def stft(input: Tensor, n_fft: int, hop_length: Optional[int] = None, win_length: Optional[int] = None,
         window: Optional[Tensor] = None, center: bool = True, pad_mode: str = "reflect", normalized: bool = False,
         onesided: Optional[bool] = None, return_complex: Optional[bool] = None) -> Tensor:
    from torch.fft import _stft
    return _stft(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided, return_complex)

def istft(input: Tensor, n_fft: int, hop_length: Optional[int] = None, win_length: Optional[int] = None,
          window: Optional[Tensor] = None, center: bool = True, normalized: bool = False,
          onesided: Optional[bool] = None, length: Optional[int] = None, return_complex: bool = False) -> Tensor:
    from torch.fft import _istft
    return _istft(input, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex)

def hann_window(window_length: int, periodic: bool = True, dtype: Optional[_DType] = None) -> Tensor:
    return hamming_window(window_length, periodic, 0.5, 0.5, dtype)

def hamming_window(window_length: int, periodic: bool = True, alpha: float = 0.54, beta: float = 0.46, dtype: Optional[_DType] = None) -> Tensor:
    # torch: alpha - beta * cos(arange(L) * (2*pi / N)) computed in the window dtype
    dt = _dt_or(dtype, _F32)
    if window_length == 1:
        return ones([1], dtype=_DType(dt))
    N = window_length if periodic else window_length - 1
    a = mul(arange(0, window_length, dtype=_DType(dt)), 2.0 * 3.141592653589793 / N)
    return sub(alpha, mul(cos(a), beta))
