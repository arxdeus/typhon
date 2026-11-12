# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.utils._indexing / _chunking / murmurhash / Bunch.

import numpy as np
from numpy.random.legacy import RandomState, check_random_state

def _safe_indexing(X, indices, *, axis: int = 0):
    """Rows (axis=0) or columns (axis=1) of X selected by an int/bool
    index array, a slice or a scalar. Lists are indexed element-wise."""
    if isinstance(indices, NoneType):
        return X
    if isinstance(X, List):
        idx = np.asarray(indices)
        if idx.dtype is bool:
            return [X[i] for i in range(len(X)) if idx[i]]
        return [X[int(i)] for i in idx]
    elif hasattr(X, "tocsr") and hasattr(X, "indptr"):
        if axis == 0:
            return X[np.asarray(indices), :]
        return X[:, np.asarray(indices)]
    else:
        A = np.asarray(X)
        if static.len(A.shape) == 1:
            if axis != 0:
                raise ValueError("'X' should be a 2D NumPy array, 2D sparse matrix or dataframe when indexing the columns (i.e. 'axis=1').")
            if isinstance(indices, int) or isinstance(indices, slice):
                return A[indices]
            return A[np.asarray(indices)]
        else:
            if isinstance(indices, int) or isinstance(indices, slice):
                if axis == 0:
                    return A[indices]
                return A[:, indices]
            else:
                idx = np.asarray(indices)
                if axis == 0:
                    return A[idx]
                return A[:, idx]

def _approximate_mode(class_counts, n_draws: int, rng):
    """Most likely class-count draw of n_draws from a multivariate
    hypergeometric (ties broken at random, as sklearn)."""
    rs = check_random_state(rng)
    cc = np.asarray(class_counts, dtype=float)
    continuous = cc / float(np.sum(cc)) * float(n_draws)
    floored = np.floor(continuous)
    need = int(n_draws - float(np.sum(floored)))
    if need > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        for value in values:
            inds = np.where(remainder == value)[0]
            add_now = min(inds.shape[0], need)
            chosen = rs.choice(inds, size=add_now, replace=False)
            for c in chosen:
                floored[c] += 1.0
            need -= add_now
            if need == 0:
                break
    return floored.astype(int)

def _resample_indices(n_samples: int, replace: bool, max_n: int, rs, stratify, sample_weight):
    if isinstance(stratify, NoneType):
        if replace:
            if isinstance(sample_weight, NoneType):
                return rs.choice(n_samples, size=max_n, replace=True)
            else:
                sw = np.asarray(sample_weight, dtype=float)
                return rs.choice(n_samples, size=max_n, p=sw / np.sum(sw), replace=True)
        else:
            idx = np.arange(n_samples)
            rs.shuffle(idx)
            return idx[:max_n].copy()
    else:
        y = np.asarray(stratify)
        if static.len(y.shape) != 1:
            compile_error("resample: 2-D stratify is not supported")
        classes, y_idx = np.unique(y, return_inverse=True)
        y_idx = np.asarray(y_idx).reshape(-1)
        k = classes.shape[0]
        counts = np.bincount(y_idx, minlength=k)
        order = np.argsort(y_idx, kind="mergesort")
        n_i = _approximate_mode(counts, max_n, rs)
        out = List[int]()
        start = 0
        for i in range(k):
            cls_idx = order[start:start + counts[i]].copy()
            start += counts[i]
            sel = rs.choice(cls_idx, int(n_i[i]), replace=replace)
            for s in sel:
                out.append(int(s))
        return rs.permutation(np.array(out, dtype=int))

def resample(*arrays, replace: bool = True, n_samples = None, random_state = None,
             stratify = None, sample_weight = None):
    """Resample arrays consistently (bootstrap by default). Returns a
    single array when one input is given, else a tuple."""
    rs = check_random_state(random_state)
    n = _num(arrays[0])
    max_n = n if isinstance(n_samples, NoneType) else int(n_samples)
    if max_n > n and not replace:
        raise ValueError(f"Cannot sample {max_n} out of arrays with dim {n} when replace is False")
    for a in arrays:
        if _num(a) != n:
            raise ValueError("Found input variables with inconsistent numbers of samples")
    if not isinstance(sample_weight, NoneType) and not replace:
        raise NotImplementedError("Resampling with sample_weight is only implemented for replace=True.")
    idx = _resample_indices(n, replace, max_n, rs, stratify, sample_weight)
    if static.len(arrays) == 1:
        return _safe_indexing(arrays[0], idx)
    else:
        return tuple(_safe_indexing(a, idx) for a in arrays)

def shuffle(*arrays, random_state = None, n_samples = None):
    return resample(*arrays, replace=False, n_samples=n_samples, random_state=random_state)

def _num(x) -> int:
    if isinstance(x, List):
        return len(x)
    return x.shape[0]

def gen_batches(n: int, batch_size: int, *, min_batch_size: int = 0):
    """Slices of batch_size elements from 0 to n (last one possibly
    smaller, merged when below min_batch_size)."""
    if batch_size <= 0:
        raise ValueError(f"gen_batches got batch_size={batch_size}, must be positive")
    start = 0
    for _ in range(n // batch_size):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)

def gen_even_slices(n: int, n_packs: int, *, n_samples = None):
    if n_packs < 1:
        raise ValueError(f"gen_even_slices got n_packs={n_packs}, must be >=1")
    start = 0
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if not isinstance(n_samples, NoneType):
                end = min(int(n_samples), end)
            yield slice(start, end)
            start = end

def get_chunk_n_rows(row_bytes: int, *, max_n_rows = None, working_memory = None) -> int:
    wm = 1024.0 if isinstance(working_memory, NoneType) else float(working_memory)
    chunk = int(wm * (2 ** 20) // row_bytes)
    if not isinstance(max_n_rows, NoneType):
        chunk = min(chunk, int(max_n_rows))
    if chunk < 1:
        chunk = 1
    return chunk

# ------------------------------------------------------------ murmurhash

def _rotl32(x: u32, r: int) -> u32:
    return (x << u32(r)) | (x >> u32(32 - r))

def _fmix32(h: u32) -> u32:
    h ^= h >> u32(16)
    h *= u32(0x85ebca6b)
    h ^= h >> u32(13)
    h *= u32(0xc2b2ae35)
    h ^= h >> u32(16)
    return h

def _murmur_bytes(data: List[int], seed: int) -> u32:
    c1 = u32(0xcc9e2d51)
    c2 = u32(0x1b873593)
    h = u32(seed & 0xffffffff)
    n = len(data)
    nblocks = n // 4
    for b in range(nblocks):
        k = u32(data[4 * b]) | (u32(data[4 * b + 1]) << u32(8)) | (u32(data[4 * b + 2]) << u32(16)) | (u32(data[4 * b + 3]) << u32(24))
        k *= c1
        k = _rotl32(k, 15)
        k *= c2
        h ^= k
        h = _rotl32(h, 13)
        h = h * u32(5) + u32(0xe6546b64)
    tail = nblocks * 4
    rem = n & 3
    k1 = u32(0)
    if rem == 3:
        k1 ^= u32(data[tail + 2]) << u32(16)
    if rem >= 2:
        k1 ^= u32(data[tail + 1]) << u32(8)
    if rem >= 1:
        k1 ^= u32(data[tail])
        k1 *= c1
        k1 = _rotl32(k1, 15)
        k1 *= c2
        h ^= k1
    h ^= u32(n)
    return _fmix32(h)

def _murmur_one(key, seed: int, positive: bool) -> int:
    data = List[int]()
    if isinstance(key, str):
        bs = key.encode("utf-8")
        for i in range(len(bs)):
            data.append(int(bs[i]) & 0xff)
    else:
        v = int(key) & 0xffffffff
        for s in range(4):
            data.append((v >> (8 * s)) & 0xff)
    h = _murmur_bytes(data, seed)
    if positive:
        return int(h)
    return int(i32(h))

def murmurhash3_32(key, seed: int = 0, positive: bool = False):
    """MurmurHash3_x86_32 of an int (as 4 little-endian bytes), a str
    (UTF-8 bytes) or an int array (elementwise)."""
    if isinstance(key, np.ndarray):
        out = np.empty(key.shape, dtype=int)
        kf = key.ravel()
        of = out.reshape(-1)
        for i in range(kf.shape[0]):
            of[i] = _murmur_one(kf[i], seed, positive)
        return out
    else:
        return _murmur_one(key, seed, positive)
