import numpy as np
import mne

FEXP = [
@@FEXP@@
]
SEXP = [
@@SEXP@@
]


def _fget(name: str) -> List[float]:
    for n, v in FEXP:
        if n == name:
            return v
    raise ValueError("missing " + name)


def _sget(name: str) -> List[str]:
    for n, v in SEXP:
        if n == name:
            return v
    raise ValueError("missing " + name)


def _flat(got) -> List[float]:
    g = List[float]()
    a = np.asarray(got, dtype=float).ravel()
    for i in range(a.shape[0]):
        g.append(float(a[i]))
    return g


def check(name: str, got, rtol: float = 1e-7, atol: float = 1e-15):
    want = _fget(name)
    g = _flat(got)
    ok = len(g) == len(want)
    if ok:
        for i in range(len(want)):
            if np.isnan(want[i]):
                if not np.isnan(g[i]):
                    ok = False
            elif abs(g[i] - want[i]) > atol + rtol * abs(want[i]):
                ok = False
    if not ok:
        print("TEST FAILED:", name, g[:12], want[:12], len(g), len(want))


def checks(name: str, got):
    want = _sget(name)
    g = [str(v) for v in got]
    if g != want:
        print("TEST FAILED:", name, g, want)



def V(x):
    return x.__val__()


def TS(x):
    return x.__val__()
