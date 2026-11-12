# Ported from scipy 1.18.1 (BSD-3-Clause)

# Private helpers shared by scipy.linalg: dtype promotion and conversion
# between ndarrays and column-major (Fortran) LAPACK work buffers.

import numpy as np
from numpy.linalg.blas import fint
from numpy.linalg import LinAlgError

def _is_complex(T: type) -> Literal[bool]:
    return T is complex or T is complex64

def _lapack_type(T: type):
    # scipy computes float32/complex64 in single precision; this port always
    # promotes to double precision (float64 / complex128).
    if _is_complex(T):
        return complex()
    else:
        return float()

def _arr(a):
    if isinstance(a, np.ndarray):
        return a
    else:
        return np.asarray(a)

def _arr_lapack(a, check_finite: bool = True):
    x = _arr(a)
    T = type(_lapack_type(x.dtype))
    y = x.astype(T)
    if check_finite:
        for v in y.flat:
            if _is_complex(T):
                if not (np.isfinite(v.real) and np.isfinite(v.imag)):
                    raise ValueError("array must not contain infs or NaNs")
            else:
                if not np.isfinite(v):
                    raise ValueError("array must not contain infs or NaNs")
    return y

def _square2(a, name: str = "a"):
    if a.ndim != 2:
        compile_error("expected a 2-D square matrix")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f'expected square "{name}" matrix')
    return a.shape[0]

def _to_f(a: np.ndarray[T, 2], T: type):
    # column-major copy, leading dimension = rows
    m, n = a.shape
    p = Ptr[T](max(m * n, 1))
    for j in range(n):
        for i in range(m):
            p[i + j * m] = a[i, j]
    return p

def _to_f_as(a, U: type):
    m, n = a.shape
    p = Ptr[U](max(m * n, 1))
    for j in range(n):
        for i in range(m):
            p[i + j * m] = U(a[i, j])
    return p

def _from_f(p: Ptr[T], m: int, n: int, ld: int, T: type):
    r = np.empty((m, n), dtype=T)
    for i in range(m):
        for j in range(n):
            r[i, j] = p[i + j * ld]
    return r

def _vec(p: Ptr[T], n: int, T: type):
    r = np.empty(n, dtype=T)
    for i in range(n):
        r[i] = p[i]
    return r

def _work_size(w) -> int:
    if isinstance(w, complex):
        return int(w.real)
    else:
        return int(w)

def _promote(A: type, B: type):
    if _is_complex(A) or _is_complex(B):
        return complex()
    else:
        return float()

@tuple
class _Pair[A, B]:
    a: A
    b: B
    def __getitem__(self, idx: Literal[int]):
        if idx == 0 or idx == -2:
            return self.a
        elif idx == 1 or idx == -1:
            return self.b
        else:
            compile_error("tuple index out of range")

@tuple
class _Triple[A, B, C]:
    a: A
    b: B
    c: C
    def __getitem__(self, idx: Literal[int]):
        if idx == 0 or idx == -3:
            return self.a
        elif idx == 1 or idx == -2:
            return self.b
        elif idx == 2 or idx == -1:
            return self.c
        else:
            compile_error("tuple index out of range")
