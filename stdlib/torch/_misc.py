# Ported from torch 2.15.0 (BSD-3-Clause)

from torch._tensor import _DType, _DEFAULT_DT

def get_default_dtype() -> _DType:
    return _DType(_DEFAULT_DT[0])

def set_default_dtype(d: _DType):
    _DEFAULT_DT[0] = d.code

def device(name: str) -> str:
    return name

def use_deterministic_algorithms(mode: bool, warn_only: bool = False):
    pass

def set_num_threads(n: int):
    pass

def get_num_threads() -> int:
    return 1

