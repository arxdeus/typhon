# Ported from joblib 1.6.0 (BSD-3-Clause)

# Native Typhon subset of joblib:
#   Parallel(n_jobs, backend)(delayed(f)(*args) for ...)  -> list of results
#   Memory(location).cache(f)                              -> memoizing wrapper
#   hash(ndarray)                                          -> joblib.hash md5 hex
#   parallel_config / parallel_backend                     -> context managers
# Deviations (see docs/libraries/sklearn.md): Parallel runs the tasks
# sequentially in submission order (results identical to any backend);
# Memory caches in process memory, keyed by joblib.hash of the arguments, and
# does not persist to `location`; hash() supports numeric ndarrays, int, float,
# str, bool, None and tuples/lists of them is NOT implemented (ndarray only).

import numpy as np
import hashlib

__version__: Literal[str] = "1.6.0"

# ---------------------------------------------------------------- hashing

def _dtype_code(T: type) -> str:
    if T is float:
        return "f8"
    elif T is float32:
        return "f4"
    elif T is int:
        return "i8"
    elif T is i32:
        return "i4"
    elif T is i16:
        return "i2"
    elif T is i8:
        return "i1"
    elif T is u64:
        return "u8"
    elif T is u32:
        return "u4"
    elif T is u16:
        return "u2"
    elif T is u8:
        return "u1"
    elif T is bool:
        return "b1"
    elif T is complex:
        return "c16"
    else:
        compile_error("joblib.hash: unsupported dtype")

class _Buf:
    # raw byte builder (Typhon str escapes above 0x7f are not single bytes)
    b: List[int]
    memo: int
    def __init__(self):
        self.b = []
        self.memo = 0
    def raw(self, *vals):
        for v in vals:
            self.b.append(v)
    def ascii(self, s: str):
        for c in s:
            self.b.append(ord(c))
    def le(self, v: int, n: int):
        for k in range(n):
            self.b.append((v >> (8 * k)) & 0xff)
    def put(self):
        # protocol 3 BINPUT (memo index < 256)
        self.raw(0x71, self.memo)
        self.memo += 1
    def pint(self, v: int):
        # Pickler.save_long, protocol >= 2
        if 0 <= v < 256:
            self.raw(0x4b, v)
        elif 0 <= v < 65536:
            self.raw(0x4d, v & 0xff, v >> 8)
        elif -0x80000000 <= v <= 0x7fffffff:
            self.raw(0x4a)
            self.le(v & 0xffffffff, 4)
        else:
            raise NotImplementedError("joblib.hash: integer out of 32-bit range")
    def int_tuple(self, t: List[int]):
        if len(t) == 0:
            self.raw(0x29)          # EMPTY_TUPLE (not memoized)
            return
        if len(t) <= 3:
            for v in t:
                self.pint(v)
            self.raw(0x84 + len(t))  # TUPLE1..3
        else:
            self.raw(0x28)           # MARK
            for v in t:
                self.pint(v)
            self.raw(0x74)           # TUPLE
        self.put()
    def to_bytes(self) -> bytes:
        n = len(self.b)
        p = Ptr[byte](n)
        for i in range(n):
            p[i] = byte(self.b[i])
        return bytes(p, n)

def _dtype_pickle(T: type) -> bytes:
    # pickle.dumps(np.dtype(T)) with numpy 2.x (protocol 5, one frame)
    code = _dtype_code(T)
    order = "|" if (T is bool or T is i8 or T is u8) else "<"
    b = _Buf()
    b.raw(0x8c, 5); b.ascii("numpy"); b.raw(0x94)
    b.raw(0x8c, 5); b.ascii("dtype"); b.raw(0x94, 0x93, 0x94)
    b.raw(0x8c, len(code)); b.ascii(code); b.raw(0x94, 0x89, 0x88, 0x87, 0x94)
    b.ascii("R"); b.raw(0x94); b.ascii("(K"); b.raw(3)
    b.raw(0x8c, 1); b.ascii(order); b.raw(0x94)
    b.ascii("NNNJ"); b.raw(0xff, 0xff, 0xff, 0xff); b.ascii("J"); b.raw(0xff, 0xff, 0xff, 0xff)
    b.ascii("K"); b.raw(0); b.ascii("t"); b.raw(0x94); b.ascii("b.")
    h = _Buf()
    h.raw(0x80, 4, 0x95)
    h.le(len(b.b), 8)
    h.b.extend(b.b)
    return h.to_bytes()

def hash(obj, hash_name: str = "md5", coerce_mmap: bool = False) -> str:
    """joblib.hash for numpy arrays: identical hex digest to joblib 1.6 /
    numpy 2.x (NumpyHasher, pickle protocol 3)."""
    if not isinstance(obj, np.ndarray):
        compile_error("joblib.hash: only numpy arrays are supported natively")
    h = hashlib.new(hash_name)
    a = obj
    T = a.dtype
    # raw data bytes as NumpyHasher feeds them (C order, F-contiguous as a.T)
    if a.ndim != 0 and not a.flags.c_contiguous and a.flags.f_contiguous:
        c = np.ascontiguousarray(a.T)
        h.update(bytes(c.data.as_byte(), c.size * T.__elemsize__))
    else:
        c = np.ascontiguousarray(a)
        h.update(bytes(c.data.as_byte(), c.size * T.__elemsize__))
    h.update(b"_HASHED_DTYPE")
    h.update(_dtype_pickle(T))
    b = _Buf()
    b.raw(0x80, 3)
    b.ascii("cnumpy\nndarray\n"); b.put()
    b.ascii("(X"); b.le(6, 4); b.ascii("HASHED")
    b.int_tuple([int(v) for v in a.shape])
    b.int_tuple([int(v) for v in a.strides])
    b.ascii("t"); b.put()
    b.raw(0x86); b.put()
    b.ascii(".")
    h.update(b.to_bytes())
    return h.hexdigest()

# ---------------------------------------------------------------- Parallel

class _Delayed[F]:
    f: F
    def __init__(self, f: F):
        self.f = f
    def __call__(self, *args, **kwargs):
        return (self.f, args, kwargs)

def delayed(function):
    return _Delayed(function)

class Parallel:
    n_jobs: Optional[int]
    backend: Optional[str]
    verbose: int
    prefer: Optional[str]

    def __init__(self, n_jobs: Optional[int] = None, backend: Optional[str] = None,
                 return_as: str = "list", verbose: int = 0, timeout = None,
                 pre_dispatch: str = "2 * n_jobs", batch_size = "auto",
                 temp_folder = None, max_nbytes = "1M", mmap_mode: str = "r",
                 prefer: Optional[str] = None, require = None):
        if return_as != "list":
            raise NotImplementedError("joblib.Parallel: only return_as='list'")
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
        self.prefer = prefer

    def _n_workers(self, n_tasks: int) -> int:
        nj = self.n_jobs
        if nj is None:
            nj = _active_n_jobs()
        k = 1 if nj is None else nj.__val__()
        if k < 0:
            k = max(cpu_count() + 1 + k, 1)
        if k == 0:
            raise ValueError("n_jobs == 0 in Parallel has no meaning")
        if self._backend() == "sequential":
            k = 1
        return min(k, max(n_tasks, 1))

    def _backend(self) -> str:
        if self.backend is not None:
            return self.backend.__val__()
        return _active_backend()

    def __call__(self, iterable):
        """Run the delayed calls; results are returned in submission order.
        n_jobs > 1 runs them on OpenMP threads (any backend name), so the
        functions must be thread safe, as with joblib's threading backend."""
        tasks = [t for t in iterable]
        n = len(tasks)
        if n == 0:
            return [_first(tasks) for _ in range(0)]
        # task 0 runs first on the calling thread (it also fixes the type)
        out = [_first(tasks)] * n
        k = self._n_workers(n)
        if k <= 1:
            for i in range(1, n):
                out[i] = tasks[i][0](*tasks[i][1], **tasks[i][2])
        else:
            @par(num_threads=k, schedule="dynamic", chunk_size=1)
            for i in range(1, n):
                out[i] = tasks[i][0](*tasks[i][1], **tasks[i][2])
        return out

    def __enter__(self):
        return self

    def __exit__(self):
        pass

def _first(tasks):
    # task 0 is evaluated eagerly (it also fixes the result type)
    return tasks[0][0](*tasks[0][1], **tasks[0][2])

_cfg_backend = [""]
_cfg_n_jobs = [0]

def _active_backend() -> str:
    return _cfg_backend[-1] if _cfg_backend[-1] else "loky"

def _active_n_jobs() -> Optional[int]:
    v = _cfg_n_jobs[-1]
    return None if v == 0 else Optional[int](v)

def cpu_count(only_physical_cores: bool = False) -> int:
    from openmp import get_num_procs
    return get_num_procs()

class parallel_config:
    backend: Optional[str]
    n_jobs: Optional[int]

    def __init__(self, backend: Optional[str] = None, *, n_jobs: Optional[int] = None,
                 verbose = None, temp_folder = None, max_nbytes = None,
                 mmap_mode = None, prefer = None, require = None,
                 inner_max_num_threads = None, **backend_params):
        self.backend = backend
        self.n_jobs = n_jobs

    def __enter__(self):
        _cfg_backend.append(self.backend.__val__() if self.backend is not None else _cfg_backend[-1])
        _cfg_n_jobs.append(self.n_jobs.__val__() if self.n_jobs is not None else _cfg_n_jobs[-1])
        return self

    def __exit__(self):
        _cfg_backend.pop()
        _cfg_n_jobs.pop()

def parallel_backend(backend: str, n_jobs: int = -1, inner_max_num_threads = None, **kw):
    return parallel_config(backend, n_jobs=n_jobs)

# ---------------------------------------------------------------- Memory

def _args_key(args) -> str:
    k = ""
    for a in args:
        if isinstance(a, np.ndarray):
            k += hash(a) + ";"
        else:
            k += repr(a) + ";"
    return k

class MemorizedFunc[F]:
    func: F
    _store: Dict[str, Ptr[byte]]
    location: Optional[str]

    def __init__(self, func: F, location: Optional[str]):
        self.func = func
        self._store = {}
        self.location = location

    def __call__(self, *args):
        R = type(self.func(*args))
        key = _args_key(args)
        if key in self._store:
            return Ptr[R](self._store[key])[0]
        r = self.func(*args)
        p = Ptr[R](1)
        p[0] = r
        self._store[key] = p.as_byte()
        return r

    def clear(self, warn: bool = True):
        self._store = {}

class Memory:
    location: Optional[str]
    verbose: int

    def __init__(self, location: Optional[str] = None, backend: str = "local",
                 mmap_mode = None, compress = False, verbose: int = 1,
                 backend_options = None):
        self.location = location
        self.verbose = verbose

    def cache(self, func, ignore = None, verbose = None, mmap_mode = False,
              cache_validation_callback = None):
        return MemorizedFunc(func, self.location)

    def clear(self, warn: bool = True):
        pass
