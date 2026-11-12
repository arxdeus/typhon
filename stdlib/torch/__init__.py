# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Native Codon implementation of a torch (CPU) subset. See docs/libraries/torch.md.

from torch._random import Generator, default_generator
from torch._tensor import *
from torch._tensor import _DType, _F32, _F64, _I64, _I32, _BOOL, _U8, _I16, _I8, _C64, _C128, _NodeBase, _new, _random_int64_full, _DEFAULT_DT
from torch._tensor import range_ as range
from torch._misc import *
from torch._serialization import save, load
import torch.cuda as cuda

float32 = _DType(_F32)
float = _DType(_F32)
float64 = _DType(_F64)
double = _DType(_F64)
int64 = _DType(_I64)
long = _DType(_I64)
int32 = _DType(_I32)
int = _DType(_I32)
int16 = _DType(_I16)
short = _DType(_I16)
int8 = _DType(_I8)
uint8 = _DType(_U8)
bool = _DType(_BOOL)
complex64 = _DType(_C64)
cfloat = _DType(_C64)
complex128 = _DType(_C128)
cdouble = _DType(_C128)
dtype = _DType

__version__ = "2.15.0"

import torch.autograd as autograd
import torch.linalg as linalg
import torch.fft as fft
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
