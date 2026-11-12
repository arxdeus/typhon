# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# LOBPCG (scipy/sparse/linalg/_eigen/lobpcg/lobpcg.py), a line-by-line port.
# A, B, M may be sparse arrays, ndarrays, LinearOperators or functions acting
# on (n, k) blocks. retLambdaHistory / retResidualNormsHistory must be
# compile-time constants (they change the number of returned values).
# Deviations: real (float64) data only; warnings are not emitted;
# verbosityLevel is ignored.

import numpy as np
from scipy.linalg import eigh, cholesky, inv, cho_factor, cho_solve
from .._base import _issparse

def _apply_block(A, X):
    # A(X) for the supported operator kinds
    if _issparse(A) or isinstance(A, np.ndarray):
        return np.asarray(A @ X)
    elif hasattr(A, "matmat"):
        return A.matmat(X)
    else:
        return A(X)

def _dense_of(A, n: int):
    if _issparse(A):
        return A.toarray().astype(float)
    elif isinstance(A, np.ndarray):
        return A.astype(float)
    else:
        return np.asarray(_apply_block(A, np.eye(n))).astype(float)

def _applyConstraints(V, factYBY, BY, Y):
    YBV = BY.T @ V
    tmp = cho_solve(factYBY, YBV)
    return V - Y @ tmp

def _cols(X, mask: List[int]):
    out = np.empty((X.shape[0], len(mask)), dtype=float)
    for c in range(len(mask)):
        for r in range(X.shape[0]):
            out[r, c] = X[r, mask[c]]
    return out

def _get_indx(lam, num: int, largest: bool) -> List[int]:
    ii = np.argsort(lam)
    out = List[int]()
    if largest:
        for t in range(num):
            out.append(int(ii[len(ii) - 1 - t]))
    else:
        for t in range(num):
            out.append(int(ii[t]))
    return out

def _take(lam, ii: List[int]):
    return np.array([lam[i] for i in ii], dtype=float)

def _block2(a, b, c, d):
    r0 = a.shape[0]
    c0 = a.shape[1]
    R = np.zeros((r0 + c.shape[0], c0 + b.shape[1]), dtype=float)
    R[:r0, :c0] = a
    R[:r0, c0:] = b
    R[r0:, :c0] = c
    R[r0:, c0:] = d
    return R

def _block3(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    a = m00.shape[0]
    b = m11.shape[0]
    c = m22.shape[0]
    R = np.zeros((a + b + c, a + b + c), dtype=float)
    R[:a, :a] = m00
    R[:a, a:a + b] = m01
    R[:a, a + b:] = m02
    R[a:a + b, :a] = m10
    R[a:a + b, a:a + b] = m11
    R[a:a + b, a + b:] = m12
    R[a + b:, :a] = m20
    R[a + b:, a:a + b] = m21
    R[a + b:, a + b:] = m22
    return R

def _b_orthonormalize(B, V, BV, haveB: Literal[bool]):
    # returns (ok, V, BV, invR)
    if haveB:
        BVv = BV
    else:
        BVv = V
    VBV = V.T @ BVv
    try:
        Rm = cholesky(VBV)
        Ri = inv(Rm)
        V2 = V @ Ri
        if haveB:
            BV2 = BVv @ Ri
        else:
            BV2 = V2
        return True, V2, BV2, Ri
    except:
        return False, V, BVv, VBV

def _resid(AX, BXorX, lam):
    R = AX - BXorX * lam[None, :]
    k = R.shape[1]
    out = np.empty(k, dtype=float)
    for c in range(k):
        s = 0.0
        for r in range(R.shape[0]):
            s += R[r, c] * R[r, c]
        out[c] = abs(s) ** 0.5
    return R, out

def _lobpcg_dense(A, B, n: int, sizeX: int, largest: bool):
    Ad = _dense_of(A, n)
    if largest:
        sub = (n - sizeX, n - 1)
    else:
        sub = (0, sizeX - 1)
    if B is None:
        vals, vecs = eigh(Ad, subset_by_index=sub)
    else:
        vals, vecs = eigh(Ad, _dense_of(B, n), subset_by_index=sub)
    if largest:
        k = len(vals)
        v2 = np.array([vals[k - 1 - i] for i in range(k)], dtype=float)
        V2 = np.ascontiguousarray(vecs[:, ::-1])
        return v2, V2
    return vals, vecs

def lobpcg(A, X, B = None, M = None, Y = None, tol = None, maxiter = None,
           largest: bool = True, verbosityLevel: int = 0,
           retLambdaHistory: Literal[bool] = False,
           retResidualNormsHistory: Literal[bool] = False,
           restartControl: int = 20):
    blockVectorX = np.asarray(X).astype(float)
    if blockVectorX.ndim != 2:
        compile_error("expected rank-2 array for argument X")
    mi = 20 if maxiter is None else int(maxiter)
    bestIterationNumber = mi
    n = blockVectorX.shape[0]
    sizeX = blockVectorX.shape[1]
    sizeY = 0
    if Y is not None:
        sizeY = Y.shape[1]
    lambdaHistory = List[np.ndarray[float, 1]]()
    residualNormsHistory = List[np.ndarray[float, 1]]()
    if (n - sizeY) < (5 * sizeX):
        # dense fallback (scipy: eigh with subset_by_index)
        if Y is not None:
            raise NotImplementedError("The dense eigensolver does not support constraints.")
        vals, vecs = _lobpcg_dense(A, B, n, min(sizeX, n), largest)
        if retLambdaHistory:
            if retResidualNormsHistory:
                return vals, vecs, lambdaHistory, residualNormsHistory
            else:
                return vals, vecs, lambdaHistory
        else:
            if retResidualNormsHistory:
                return vals, vecs, residualNormsHistory
            else:
                return vals, vecs
    if tol is None or float(tol) <= 0.0:
        residualTolerance = (2.220446049250313e-16 ** 0.5) * n
    else:
        residualTolerance = float(tol)
    if Y is not None:
        Yv = np.asarray(Y).astype(float)
        if B is not None:
            BY = _apply_block(B, Yv)
        else:
            BY = Yv
        gramYBY = cho_factor(Yv.T @ BY)
        blockVectorX = _applyConstraints(blockVectorX, gramYBY, BY, Yv)
    if B is not None:
        ok, blockVectorX, blockVectorBX, _ = _b_orthonormalize(
            B, blockVectorX, _apply_block(B, blockVectorX), True)
    else:
        ok, blockVectorX, blockVectorBX, _ = _b_orthonormalize(
            B, blockVectorX, blockVectorX, False)
    if not ok:
        raise ValueError("Linearly dependent initial approximations")
    blockVectorAX = _apply_block(A, blockVectorX)
    gramXAX = blockVectorX.T @ blockVectorAX
    _lambda, eigBlockVector = eigh(gramXAX)
    ii = _get_indx(_lambda, sizeX, largest)
    _lambda = _take(_lambda, ii)
    if retLambdaHistory:
        lambdaHistory.append(_lambda.copy())
    eigBlockVector = _cols(eigBlockVector, ii)
    blockVectorX = blockVectorX @ eigBlockVector
    blockVectorAX = blockVectorAX @ eigBlockVector
    if B is not None:
        blockVectorBX = blockVectorBX @ eigBlockVector
    activeMask = [True] * sizeX
    blockVectorP = np.zeros((n, sizeX), dtype=float)
    blockVectorAP = np.zeros((n, sizeX), dtype=float)
    blockVectorBP = np.zeros((n, sizeX), dtype=float)
    smallestResidualNorm = 1.7976931348623157e308
    bestblockVectorX = blockVectorX
    iterationNumber = -1
    restart = True
    forcedRestart = False
    explicitGramFlag = False
    while iterationNumber < mi:
        iterationNumber += 1
        if B is not None:
            blockVectorR, residualNorms = _resid(blockVectorAX, blockVectorBX, _lambda)
        else:
            blockVectorR, residualNorms = _resid(blockVectorAX, blockVectorX, _lambda)
        if retResidualNormsHistory:
            residualNormsHistory.append(residualNorms.copy())
        residualNorm = 0.0
        for c in range(sizeX):
            residualNorm += abs(residualNorms[c])
        residualNorm /= sizeX
        if residualNorm < smallestResidualNorm:
            smallestResidualNorm = residualNorm
            bestIterationNumber = iterationNumber
            bestblockVectorX = blockVectorX
        elif residualNorm > float(2 ** restartControl) * smallestResidualNorm:
            forcedRestart = True
            blockVectorAX = _apply_block(A, blockVectorX)
            if B is not None:
                blockVectorBX = _apply_block(B, blockVectorX)
        for c in range(sizeX):
            activeMask[c] = activeMask[c] and residualNorms[c] > residualTolerance
        act = List[int]()
        for c in range(sizeX):
            if activeMask[c]:
                act.append(c)
        currentBlockSize = len(act)
        if currentBlockSize == 0:
            break
        activeBlockVectorR = _cols(blockVectorR, act)
        if iterationNumber > 0:
            activeBlockVectorP = _cols(blockVectorP, act)
            activeBlockVectorAP = _cols(blockVectorAP, act)
            activeBlockVectorBP = _cols(blockVectorBP, act)
        else:
            activeBlockVectorP = np.zeros((n, currentBlockSize), dtype=float)
            activeBlockVectorAP = np.zeros((n, currentBlockSize), dtype=float)
            activeBlockVectorBP = np.zeros((n, currentBlockSize), dtype=float)
        if M is not None:
            activeBlockVectorR = np.asarray(_apply_block(M, activeBlockVectorR)).astype(float)
        if Y is not None:
            activeBlockVectorR = _applyConstraints(activeBlockVectorR, gramYBY, BY, Yv)
        if B is not None:
            activeBlockVectorR = activeBlockVectorR - (blockVectorX @ (blockVectorBX.T @ activeBlockVectorR))
            ok, activeBlockVectorR, activeBlockVectorBR, _ = _b_orthonormalize(
                B, activeBlockVectorR, _apply_block(B, activeBlockVectorR), True)
        else:
            activeBlockVectorR = activeBlockVectorR - (blockVectorX @ (blockVectorX.T @ activeBlockVectorR))
            ok, activeBlockVectorR, activeBlockVectorBR, _ = _b_orthonormalize(
                B, activeBlockVectorR, activeBlockVectorR, False)
        if not ok:
            break
        activeBlockVectorAR = _apply_block(A, activeBlockVectorR)
        if iterationNumber > 0:
            if B is not None:
                okp, activeBlockVectorP, activeBlockVectorBP, invR = _b_orthonormalize(
                    B, activeBlockVectorP, activeBlockVectorBP, True)
            else:
                okp, activeBlockVectorP, _bp, invR = _b_orthonormalize(
                    B, activeBlockVectorP, activeBlockVectorP, False)
            if okp:
                activeBlockVectorAP = activeBlockVectorAP @ invR
                restart = forcedRestart
            else:
                restart = True
        myeps = 2.220446049250313e-16 ** 0.5
        rmax = residualNorms[0]
        for c in range(sizeX):
            rmax = max(rmax, residualNorms[c])
        if rmax > myeps and not explicitGramFlag:
            explicitGramFlag = False
        else:
            explicitGramFlag = True
        if B is None:
            blockVectorBX = blockVectorX
            activeBlockVectorBR = activeBlockVectorR
            if not restart:
                activeBlockVectorBP = activeBlockVectorP
        gramXAR = blockVectorX.T @ activeBlockVectorAR
        gramRAR = activeBlockVectorR.T @ activeBlockVectorAR
        if explicitGramFlag:
            gramRAR = (gramRAR + gramRAR.T) / 2
            gramXAX = blockVectorX.T @ blockVectorAX
            gramXAX = (gramXAX + gramXAX.T) / 2
            gramXBX = blockVectorX.T @ blockVectorBX
            gramRBR = activeBlockVectorR.T @ activeBlockVectorBR
            gramXBR = blockVectorX.T @ activeBlockVectorBR
        else:
            gramXAX = np.diag(_lambda)
            gramXBX = np.eye(sizeX)
            gramRBR = np.eye(currentBlockSize)
            gramXBR = np.zeros((sizeX, currentBlockSize), dtype=float)
        eig_ok = False
        if not restart:
            gramXAP = blockVectorX.T @ activeBlockVectorAP
            gramRAP = activeBlockVectorR.T @ activeBlockVectorAP
            gramPAP = activeBlockVectorP.T @ activeBlockVectorAP
            gramXBP = blockVectorX.T @ activeBlockVectorBP
            gramRBP = activeBlockVectorR.T @ activeBlockVectorBP
            if explicitGramFlag:
                gramPAP = (gramPAP + gramPAP.T) / 2
                gramPBP = activeBlockVectorP.T @ activeBlockVectorBP
            else:
                gramPBP = np.eye(currentBlockSize)
            gramA = _block3(gramXAX, gramXAR, gramXAP, np.ascontiguousarray(gramXAR.T),
                            gramRAR, gramRAP, np.ascontiguousarray(gramXAP.T),
                            np.ascontiguousarray(gramRAP.T), gramPAP)
            gramB = _block3(gramXBX, gramXBR, gramXBP, np.ascontiguousarray(gramXBR.T),
                            gramRBR, gramRBP, np.ascontiguousarray(gramXBP.T),
                            np.ascontiguousarray(gramRBP.T), gramPBP)
            try:
                _lam2, eigBV2 = eigh(gramA, gramB)
                _lambda_all = _lam2
                eigBlockVectorAll = eigBV2
                eig_ok = True
            except:
                restart = True
        if restart:
            gramA = _block2(gramXAX, gramXAR, np.ascontiguousarray(gramXAR.T), gramRAR)
            gramB = _block2(gramXBX, gramXBR, np.ascontiguousarray(gramXBR.T), gramRBR)
            try:
                _lam3, eigBV3 = eigh(gramA, gramB)
                _lambda_all = _lam3
                eigBlockVectorAll = eigBV3
                eig_ok = True
            except:
                break
        if not eig_ok:
            break
        ii = _get_indx(_lambda_all, sizeX, largest)
        _lambda = _take(_lambda_all, ii)
        eigBlockVector = _cols(eigBlockVectorAll, ii)
        if retLambdaHistory:
            lambdaHistory.append(_lambda.copy())
        eigBlockVectorX = np.ascontiguousarray(eigBlockVector[:sizeX])
        if not restart:
            eigBlockVectorR = np.ascontiguousarray(eigBlockVector[sizeX:sizeX + currentBlockSize])
            eigBlockVectorP = np.ascontiguousarray(eigBlockVector[sizeX + currentBlockSize:])
            pp = activeBlockVectorR @ eigBlockVectorR
            pp += activeBlockVectorP @ eigBlockVectorP
            app = activeBlockVectorAR @ eigBlockVectorR
            app += activeBlockVectorAP @ eigBlockVectorP
            if B is not None:
                bpp = activeBlockVectorBR @ eigBlockVectorR
                bpp += activeBlockVectorBP @ eigBlockVectorP
        else:
            eigBlockVectorR = np.ascontiguousarray(eigBlockVector[sizeX:])
            pp = activeBlockVectorR @ eigBlockVectorR
            app = activeBlockVectorAR @ eigBlockVectorR
            if B is not None:
                bpp = activeBlockVectorBR @ eigBlockVectorR
        blockVectorX = blockVectorX @ eigBlockVectorX + pp
        blockVectorAX = blockVectorAX @ eigBlockVectorX + app
        if B is not None:
            blockVectorBX = blockVectorBX @ eigBlockVectorX + bpp
            blockVectorBP = bpp
        blockVectorP = pp
        blockVectorAP = app
    if B is not None:
        blockVectorR, residualNorms = _resid(blockVectorAX, blockVectorBX, _lambda)
    else:
        blockVectorR, residualNorms = _resid(blockVectorAX, blockVectorX, _lambda)
    if retLambdaHistory:
        # row iterationNumber + 1 (already written when the loop ran out)
        if len(lambdaHistory) > iterationNumber + 1:
            lambdaHistory[iterationNumber + 1] = _lambda.copy()
        else:
            lambdaHistory.append(_lambda.copy())
    if retResidualNormsHistory:
        if len(residualNormsHistory) > iterationNumber + 1:
            residualNormsHistory[iterationNumber + 1] = residualNorms.copy()
        else:
            residualNormsHistory.append(residualNorms.copy())
    residualNorm = 0.0
    for c in range(sizeX):
        residualNorm += abs(residualNorms[c])
    residualNorm /= sizeX
    if residualNorm < smallestResidualNorm:
        smallestResidualNorm = residualNorm
        bestIterationNumber = iterationNumber + 1
        bestblockVectorX = blockVectorX
    blockVectorX = bestblockVectorX
    if Y is not None:
        blockVectorX = _applyConstraints(blockVectorX, gramYBY, BY, Yv)
    blockVectorAX = _apply_block(A, blockVectorX)
    gramXAX = blockVectorX.T @ blockVectorAX
    if B is not None:
        blockVectorBX = _apply_block(B, blockVectorX)
    else:
        blockVectorBX = blockVectorX
    gramXBX = blockVectorX.T @ blockVectorBX
    gramXAX = (gramXAX + gramXAX.T) / 2
    gramXBX = (gramXBX + gramXBX.T) / 2
    _lambda_f, eigBlockVector_f = eigh(gramXAX, gramXBX)
    ii = _get_indx(_lambda_f, sizeX, largest)
    _lambda = _take(_lambda_f, ii)
    eigBlockVector = _cols(eigBlockVector_f, ii)
    blockVectorX = blockVectorX @ eigBlockVector
    blockVectorAX = blockVectorAX @ eigBlockVector
    if B is not None:
        blockVectorBX = blockVectorBX @ eigBlockVector
        blockVectorR, residualNorms = _resid(blockVectorAX, blockVectorBX, _lambda)
    else:
        blockVectorR, residualNorms = _resid(blockVectorAX, blockVectorX, _lambda)
    if retLambdaHistory:
        # scipy writes the final values at row bestIterationNumber + 1 and
        # truncates the history there
        while len(lambdaHistory) < bestIterationNumber + 2:
            lambdaHistory.append(_lambda.copy())
        lambdaHistory[bestIterationNumber + 1] = _lambda.copy()
        lambdaHistory = lambdaHistory[:bestIterationNumber + 2]
    if retResidualNormsHistory:
        while len(residualNormsHistory) < bestIterationNumber + 2:
            residualNormsHistory.append(residualNorms.copy())
        residualNormsHistory[bestIterationNumber + 1] = residualNorms.copy()
        residualNormsHistory = residualNormsHistory[:bestIterationNumber + 2]
    if retLambdaHistory:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, lambdaHistory, residualNormsHistory
        else:
            return _lambda, blockVectorX, lambdaHistory
    else:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, residualNormsHistory
        else:
            return _lambda, blockVectorX
