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

_NUM_THREADS = [0]

def _default_num_threads() -> int:
    # torch (OpenMP build): OMP_NUM_THREADS, else the physical (performance) core count
    # (macOS: hw.perflevel0.physicalcpu). Kernels whose float results depend on the
    # parallel_for chunking (e.g. channels-last batch_norm) follow this value.
    import os
    v = os.getenv("OMP_NUM_THREADS")
    if v is not None:
        try:
            n = int(v.__val__())
            if n > 0:
                return n
        except:
            pass
    if __apple__:
        from C import sysctlbyname(cobj, Ptr[i32], Ptr[int], cobj, int) -> i32
        for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            out = i32(0)
            sz = 4
            if sysctlbyname(key.c_str(), __ptr__(out), __ptr__(sz), cobj(), 0) == i32(0) and int(out) > 0:
                return int(out)
        return 1
    else:
        from C import sysconf(i32) -> int
        n = sysconf(i32(84))   # _SC_NPROCESSORS_ONLN (Linux/Android)
        return n if n > 0 else 1

def set_num_threads(n: int):
    if n <= 0:
        raise RuntimeError("set_num_threads expects a positive integer")
    _NUM_THREADS[0] = n

def get_num_threads() -> int:
    if _NUM_THREADS[0] == 0:
        _NUM_THREADS[0] = _default_num_threads()
    return _NUM_THREADS[0]


_NUM_WORKERS = [0]

def _fast_cores() -> int:
    # big + mid cores: macOS/iOS hw.perflevel0 (performance cores); Android/Linux: CPUs
    # whose cpuinfo_max_freq is above the slowest cluster (little cores excluded).
    if __apple__:
        from C import sysctlbyname(cobj, Ptr[i32], Ptr[int], cobj, int) -> i32
        out = i32(0)
        sz = 4
        if sysctlbyname("hw.perflevel0.physicalcpu".c_str(), __ptr__(out), __ptr__(sz), cobj(), 0) == i32(0) and int(out) > 0:
            return int(out)
        return 0
    freqs = List[int]()
    i = 0
    while i < 64:
        try:
            with open(f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq") as fh:
                freqs.append(int(fh.read().strip()))
        except:
            break
        i += 1
    if len(freqs) == 0:
        return 0
    lo = min(freqs)
    n = 0
    for v in freqs:
        if v > lo:
            n += 1
    return n if n > 0 else len(freqs)

def _num_workers() -> int:
    # Physical worker threads for Typhon's own kernels (results never depend on it):
    # TYPHON_NUM_WORKERS, else min(torch.get_num_threads(), 4) on desktop (a 5th worker
    # only adds spin time); on Android/iOS the big + mid core count.
    if _NUM_WORKERS[0] == 0:
        import os
        v = os.getenv("TYPHON_NUM_WORKERS")
        n = 0
        if v is not None:
            try:
                n = int(v.__val__())
            except:
                n = 0
        if n <= 0:
            n = get_num_threads()
            if n > 4:
                n = 4
            if __android__ or __ios__:
                f = _fast_cores()
                if f > 0:
                    n = f
        _NUM_WORKERS[0] = n
    return _NUM_WORKERS[0]
