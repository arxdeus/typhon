# Ported from torch 2.15.0 (BSD-3-Clause)
#
# torch.linalg on top of the native numpy.linalg (LAPACK). Inputs may be
# batched (..., n, n). Autograd: inv, solve, cholesky, eigh, det, slogdet,
# svd (singular values and vectors), qr is forward only.

import numpy as np
from numpy.util import sizeof
import internal.static as static
from torch._tensor import Tensor, Node, _O, _new, _needs, _attach, _is_float, _F32, _F64, _I64, _DType, _numel, reshape, matmul, mul, div, sub, add, neg, transpose, diag_embed, diagonal as _tdiag, unsqueeze, eye, zeros_like, norm as _norm, sqrt, sum, abs, log, _contig_as, tril, triu, stack, select, expand, ones_like, where, _where3, gt, amax, sign, _to_list, cat, narrow, squeeze, clone, _wrap, _imax, _imin

def _mT(x: Tensor) -> Tensor:
    return transpose(x, -2, -1)

def _batch_mats(A: Tensor) -> Tuple[Tensor, int, int, int]:
    if len(A._shape) < 2:
        raise RuntimeError("linalg: The input tensor A must have at least 2 dimensions.")
    m = A._shape[-2]
    n = A._shape[-1]
    dt = A._dt if _is_float(A._dt) else _F32
    c = _contig_as(A, dt)
    B = _numel(A._shape[:-2])
    return (c, B, m, n)

def _np_mat(z, c: Tensor, b: int, m: int, n: int):
    T = type(z)
    p = Ptr[T](c._st.ptr) + c._off + b * m * n
    return np.ndarray[T, 2]((m, n), (n * sizeof(T), sizeof(T)), p)

def _put(z, out: Tensor, b: int, a):
    T = type(z)
    p = Ptr[T](out._st.ptr) + b * a.size
    k = 0
    for i in range(a.shape[0]):
        if static.len(a.shape) == 2:
            for j in range(a.shape[1]):
                p[k] = T(a[i, j])
                k += 1
        else:
            p[k] = T(a[i])
            k += 1

def _apply_mat(A: Tensor, oshape: List[int], f) -> Tensor:
    c, B, m, n = _batch_mats(A)
    out = _new(A._shape[:-2] + oshape, c._dt)
    for b in range(B):
        if c._dt == _F64:
            _put(0.0, out, b, f(_np_mat(0.0, c, b, m, n)))
        else:
            _put(float32(0.0), out, b, f(_np_mat(float32(0.0), c, b, m, n)))
    return out

# ---------------------------------------------------------------- inv

class _InvBackward(Node):
    _pad0: int
    def __init__(self, A: Tensor, Ainv: Tensor):
        super().__init__("LinalgInvExBackward0", [A])
        self._pad0 = 0
        self.saved = [Ainv]
    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        Ai = self.saved[0]
        return [_O(neg(matmul(matmul(_mT(Ai), g), _mT(Ai))))]

def inv(A: Tensor) -> Tensor:
    n = A._shape[-1]
    out = _apply_mat(A.detach(), [n, n], np.linalg.inv)
    if _needs([A]):
        _attach(out, _InvBackward(A, out.detach()))
    return out

def inv_ex(A: Tensor) -> Tuple[Tensor, Tensor]:
    return (inv(A), _new(A._shape[:-2], _I64))

def pinv(A: Tensor, rcond: Optional[float] = None, hermitian: bool = False) -> Tensor:
    U, S, Vh = svd(A, False)
    mx = amax(S, -1, True)
    eps = 1.1920928955078125e-07 if S._dt == _F32 else 2.220446049250313e-16
    rc = rcond if rcond is not None else eps * float(_imax(A._shape[-1], A._shape[-2]))
    big = gt(S, mul(mx, rc))
    Sinv = _where3(big, div(ones_like(S), _where3(big, S, ones_like(S))), zeros_like(S))
    return matmul(mul(_mT(Vh), unsqueeze(Sinv, -2)), _mT(U))

# ---------------------------------------------------------------- solve

class _SolveBackward(Node):
    _pad0: int
    def __init__(self, A: Tensor, B: Tensor, X: Tensor):
        super().__init__("LinalgSolveExBackward0", [A, B])
        self._pad0 = 0
        self.saved = [A.detach(), X]
    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        A = self.saved[0]
        X = self.saved[1]
        gB = solve(_mT(A), g)
        gA = neg(matmul(gB, _mT(X)))
        return [_O(gA), _O(gB)]

def solve(A: Tensor, B: Tensor, left: bool = True) -> Tensor:
    if not left:
        return _mT(solve(_mT(A), _mT(B)))
    vec = len(B._shape) == len(A._shape) - 1
    Bm = unsqueeze(B, -1) if vec else B
    X = matmul(inv(A.detach()), Bm.detach())
    if vec:
        X = squeeze(X, -1)
    X = X.detach()
    if _needs([A, B]):
        if vec:
            _attach(X, _SolveVecBackward(A, B, X))
        else:
            _attach(X, _SolveBackward(A, B, X))
    return X

class _SolveVecBackward(Node):
    _pad0: int
    def __init__(self, A: Tensor, B: Tensor, X: Tensor):
        super().__init__("LinalgSolveExBackward0", [A, B])
        self._pad0 = 0
        self.saved = [A.detach(), X]
    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        A = self.saved[0]
        X = self.saved[1]
        gB = solve(_mT(A), g)
        gA = neg(mul(unsqueeze(gB, -1), unsqueeze(X, -2)))
        return [_O(gA), _O(gB)]

# ---------------------------------------------------------------- cholesky

class _CholBackward(Node):
    upper: bool
    def __init__(self, A: Tensor, L: Tensor, upper: bool):
        super().__init__("LinalgCholeskyExBackward0", [A])
        self.saved = [L]
        self.upper = upper
    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        # torch cholesky_backward: L lower
        L = self.saved[0]
        gL = g
        if self.upper:
            L = _mT(L)
            gL = _mT(g)
        n = L._shape[-1]
        phi = matmul(_mT(L), gL)
        phi = tril(phi)
        dg = _tdiag(phi, 0, -2, -1)
        phi = sub(phi, mul(diag_embed(dg), 0.5))
        Linv = inv(L)
        gA = matmul(matmul(_mT(Linv), phi), Linv)
        gA = mul(add(gA, _mT(gA)), 0.5)
        return [_O(gA)]

def cholesky(A: Tensor, upper: bool = False) -> Tensor:
    n = A._shape[-1]
    L = _apply_mat(A.detach(), [n, n], np.linalg.cholesky)
    if upper:
        L = _mT(L).contiguous()
    if _needs([A]):
        _attach(L, _CholBackward(A, L.detach(), upper))
    return L

def cholesky_ex(A: Tensor, upper: bool = False) -> Tuple[Tensor, Tensor]:
    return (cholesky(A, upper), _new(A._shape[:-2], _I64))

# ---------------------------------------------------------------- eigh

@tuple
class return_types_eigh:
    eigenvalues: Tensor
    eigenvectors: Tensor

def _eigh_raw(A: Tensor, UPLO: str) -> Tuple[Tensor, Tensor]:
    c, B, m, n = _batch_mats(A.detach())
    L = _new(A._shape[:-2] + [n], c._dt)
    V = _new(A._shape[:-2] + [n, n], c._dt)
    for b in range(B):
        if c._dt == _F64:
            r = np.linalg.eigh(_np_mat(0.0, c, b, n, n), UPLO)
            _put(0.0, L, b, r[0])
            _put(0.0, V, b, r[1])
        else:
            r2 = np.linalg.eigh(_np_mat(float32(0.0), c, b, n, n), UPLO)
            _put(float32(0.0), L, b, r2[0])
            _put(float32(0.0), V, b, r2[1])
    return (L, V)

def eigh(A: Tensor, UPLO: str = "L") -> return_types_eigh:
    L, V = _eigh_raw(A, UPLO)
    if _needs([A]):
        # Express outputs through a shared helper so that both grads reach A.
        st = _EighState(L, V)
        Lo = L.detach()
        Vo = V.detach()
        _attach(Lo, _EighValsNode(A, st))
        _attach(Vo, _EighVecsNode(A, st))
        return return_types_eigh(Lo, Vo)
    return return_types_eigh(L, V)

class _EighState:
    L: Tensor
    V: Tensor
    def __init__(self, L: Tensor, V: Tensor):
        self.L = L
        self.V = V

class _EighValsNode(Node):
    st: _EighState
    def __init__(self, A: Tensor, st: _EighState):
        super().__init__("LinalgEighBackward0", [A])
        self.st = st
    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        V = self.st.V
        gA = matmul(mul(V, unsqueeze(g, -2)), _mT(V))
        return [_O(gA)]

class _EighVecsNode(Node):
    st: _EighState
    def __init__(self, A: Tensor, st: _EighState):
        super().__init__("LinalgEighBackward0", [A])
        self.st = st
    def apply(self, gV: Tensor) -> List[Optional[Tensor]]:
        # torch linalg_eig_backward (hermitian): gA = V (VhgV - diag-real-part) / E V^H
        L = self.st.L
        V = self.st.V
        Ediff = sub(unsqueeze(L, -2), unsqueeze(L, -1))  # E[i,j] = L[j]-L[i]
        n = L._shape[-1]
        I = eye(n, dtype=_DType(L._dt))
        Ediff = add(Ediff, I)  # avoid div by zero on diagonal (masked below)
        VhgV = matmul(_mT(V), gV)
        # skew part: (VhgV - VhgV^H)/2 is what matters for real symmetric
        Ft = div(VhgV, Ediff)
        Ft = mul(Ft, sub(ones_like(I), I))
        gA = matmul(matmul(V, Ft), _mT(V))
        gA = mul(add(gA, _mT(gA)), 0.5)
        return [_O(gA)]

def eigvalsh(A: Tensor, UPLO: str = "L") -> Tensor:
    return eigh(A, UPLO).eigenvalues

# ---------------------------------------------------------------- svd

@tuple
class return_types_svd:
    U: Tensor
    S: Tensor
    Vh: Tensor

def _svd_raw(A: Tensor, full_matrices: bool) -> Tuple[Tensor, Tensor, Tensor]:
    c, B, m, n = _batch_mats(A.detach())
    k = _imin(m, n)
    um = m
    un = m if full_matrices else k
    vm = n if full_matrices else k
    U = _new(A._shape[:-2] + [um, un], c._dt)
    S = _new(A._shape[:-2] + [k], c._dt)
    Vh = _new(A._shape[:-2] + [vm, n], c._dt)
    for b in range(B):
        if c._dt == _F64:
            r = np.linalg.svd(_np_mat(0.0, c, b, m, n), full_matrices=full_matrices)
            _put(0.0, U, b, r[0])
            _put(0.0, S, b, r[1])
            _put(0.0, Vh, b, r[2])
        else:
            r2 = np.linalg.svd(_np_mat(float32(0.0), c, b, m, n), full_matrices=full_matrices)
            _put(float32(0.0), U, b, r2[0])
            _put(float32(0.0), S, b, r2[1])
            _put(float32(0.0), Vh, b, r2[2])
    return (U, S, Vh)

class _SvdValsNode(Node):
    _pad0: int
    def __init__(self, A: Tensor, U: Tensor, Vh: Tensor):
        super().__init__("LinalgSvdBackward0", [A])
        self._pad0 = 0
        self.saved = [U, Vh]
    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        U = self.saved[0]
        Vh = self.saved[1]
        return [_O(matmul(mul(U, unsqueeze(g, -2)), Vh))]

def svd(A: Tensor, full_matrices: bool = True, driver: Optional[str] = None) -> return_types_svd:
    U, S, Vh = _svd_raw(A, full_matrices)
    if _needs([A]):
        k = S._shape[-1]
        Uk = narrow(U, -1, 0, k).contiguous()
        Vk = narrow(Vh, -2, 0, k).contiguous()
        _attach(S, _SvdValsNode(A, Uk, Vk))
    return return_types_svd(U, S, Vh)

def svdvals(A: Tensor, driver: Optional[str] = None) -> Tensor:
    return svd(A, False).S

# ---------------------------------------------------------------- det / slogdet

@tuple
class return_types_slogdet:
    sign: Tensor
    logabsdet: Tensor

def slogdet(A: Tensor) -> return_types_slogdet:
    c, B, m, n = _batch_mats(A.detach())
    sg = _new(A._shape[:-2], c._dt)
    la = _new(A._shape[:-2], c._dt)
    for b in range(B):
        if c._dt == _F64:
            r = np.linalg.slogdet(_np_mat(0.0, c, b, n, n))
            Ptr[float](sg._st.ptr)[b] = float(r[0])
            Ptr[float](la._st.ptr)[b] = float(r[1])
        else:
            r2 = np.linalg.slogdet(_np_mat(float32(0.0), c, b, n, n))
            Ptr[float32](sg._st.ptr)[b] = float32(r2[0])
            Ptr[float32](la._st.ptr)[b] = float32(r2[1])
    if _needs([A]):
        _attach(la, _LogdetNode(A))
    return return_types_slogdet(sg, la)

class _LogdetNode(Node):
    _pad0: int
    def __init__(self, A: Tensor):
        super().__init__("LinalgSlogdetBackward0", [A])
        self._pad0 = 0
    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        A = self.inputs[0].detach()
        return [_O(mul(_mT(inv(A)), unsqueeze(unsqueeze(g, -1), -1)))]

def det(A: Tensor) -> Tensor:
    r = slogdet(A.detach())
    from torch._tensor import exp
    d = mul(r.sign, exp(r.logabsdet))
    if _needs([A]):
        _attach(d, _DetNode(A, d.detach()))
    return d

class _DetNode(Node):
    _pad0: int
    def __init__(self, A: Tensor, d: Tensor):
        super().__init__("LinalgDetBackward0", [A])
        self._pad0 = 0
        self.saved = [d]
    def apply(self, g: Tensor) -> List[Optional[Tensor]]:
        A = self.inputs[0].detach()
        gd = mul(g, self.saved[0])
        return [_O(mul(_mT(inv(A)), unsqueeze(unsqueeze(gd, -1), -1)))]

def logdet(A: Tensor) -> Tensor:
    return slogdet(A).logabsdet

# ---------------------------------------------------------------- qr

@tuple
class return_types_qr:
    Q: Tensor
    R: Tensor

def qr(A: Tensor, mode: str = "reduced") -> return_types_qr:
    c, B, m, n = _batch_mats(A.detach())
    k = _imin(m, n)
    qc = m if mode == "complete" else k
    Q = _new(A._shape[:-2] + [m, qc], c._dt)
    R = _new(A._shape[:-2] + [qc, n], c._dt)
    for b in range(B):
        if c._dt == _F64:
            if mode == "complete":
                r = np.linalg.qr(_np_mat(0.0, c, b, m, n), "complete")
                _put(0.0, Q, b, r[0])
                _put(0.0, R, b, r[1])
            else:
                r1 = np.linalg.qr(_np_mat(0.0, c, b, m, n), "reduced")
                _put(0.0, Q, b, r1[0])
                _put(0.0, R, b, r1[1])
        else:
            if mode == "complete":
                r2 = np.linalg.qr(_np_mat(float32(0.0), c, b, m, n), "complete")
                _put(float32(0.0), Q, b, r2[0])
                _put(float32(0.0), R, b, r2[1])
            else:
                r3 = np.linalg.qr(_np_mat(float32(0.0), c, b, m, n), "reduced")
                _put(float32(0.0), Q, b, r3[0])
                _put(float32(0.0), R, b, r3[1])
    return return_types_qr(Q, R)

# ---------------------------------------------------------------- norms / misc

def vector_norm(x: Tensor, ord = 2, dim = None, keepdim: bool = False, dtype = None) -> Tensor:
    return _norm(x, ord, dim, keepdim)

def matrix_norm(A: Tensor, ord = "fro", dim = (-2, -1), keepdim: bool = False) -> Tensor:
    if isinstance(ord, str):
        if ord == "fro":
            return _norm(A, "fro", dim, keepdim)
        if ord == "nuc":
            return sum(svdvals(A), -1)
        raise NotImplementedError("matrix_norm ord=" + ord)
    else:
        o = float(ord)
        if o == 2.0:
            return select(svdvals(A), -1, 0)
        if o == 1.0:
            return amax(sum(abs(A), -2), -1)
        if o == float("inf"):
            return amax(sum(abs(A), -1), -1)
        raise NotImplementedError("matrix_norm ord")

def norm(A: Tensor, ord = None, dim = None, keepdim: bool = False) -> Tensor:
    if ord is None:
        return _norm(A, 2, dim, keepdim)
    elif isinstance(ord, str):
        return matrix_norm(A, ord, (-2, -1) if dim is None else dim, keepdim)
    else:
        if dim is None and len(A._shape) == 2:
            return matrix_norm(A, ord)
        return _norm(A, ord, dim, keepdim)


def matrix_power(A: Tensor, n: int) -> Tensor:
    if n == 0:
        return expand(eye(A._shape[-1], dtype=_DType(A._dt)), A._shape)
    if n < 0:
        return matrix_power(inv(A), -n)
    r = A
    for _ in range(n - 1):
        r = matmul(r, A)
    return r

def eigvals_sym(A: Tensor) -> Tensor:
    return eigvalsh(A)

def diagonal(A: Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> Tensor:
    return _tdiag(A, offset, dim1, dim2)

def cond(A: Tensor) -> Tensor:
    s = svdvals(A)
    return div(select(s, -1, 0), select(s, -1, s._shape[-1] - 1))

def matrix_rank(A: Tensor, tol: Optional[float] = None, hermitian: bool = False) -> Tensor:
    s = svdvals(A)
    eps = 1.1920928955078125e-07 if s._dt == _F32 else 2.220446049250313e-16
    t = tol if tol is not None else 0.0
    mx = amax(s, -1, True)
    thr = mul(mx, eps * float(_imax(A._shape[-1], A._shape[-2]))) if tol is None else mul(ones_like(mx), t)
    return sum(_contig_as(gt(s, thr), _I64), -1)

def lstsq_solution(A: Tensor, B: Tensor) -> Tensor:
    return matmul(pinv(A), B)
