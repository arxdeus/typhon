# Generates test_estimation.py and test_spatial.py from real pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import covariance as C, kernel as K
from pyriemann import estimation as E, spatialfilters as S
D = os.path.dirname(__file__)
def signals(n_trials, n_ch, n_t):
    X = np.zeros((n_trials, n_ch, n_t))
    for i in range(n_trials):
        for c in range(n_ch):
            for t in range(n_t):
                v = (i * 1009 + c * 7919 + t * 104729 + i * c * t * 13
                     + t * t * (c + 1) * 3 + c * c * t * 7) % 101
                X[i, c, t] = (v - 50.0) / 25.0 + (0.5 * (i % 2)) * ((c + t) % 3 == 0)
    return X
def spd(n_mats, n):
    out = np.zeros((n_mats, n, n))
    for k in range(n_mats):
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = ((i + 1) * (j + 2) * (k + 3) % 7 - 3.0) / 4.0
        out[k] = A @ A.T + (k + 1.0) * np.eye(n)
    return out
def lit(a): return repr(np.asarray(a, dtype=float).ravel().tolist())
src = open(os.path.join(D, "test_pipeline.py")).read()
head = src.split("P = spd(5, 4)")[0]

# ---- estimation
X = signals(6, 4, 64)
y = np.array([0, 1, 0, 1, 0, 1])
L = []; T = []
def add(name, arr, atol=1e-10, call=None):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close({call}, E_{name}, atol={atol})")
def per(f):
    return np.array([f(x) for x in X])
for e in ("hub", "stu", "tyl"):
    add("cov_"+e, per(lambda x: C.covariances(x, e)), call=f'covariances(X, "{e}")')
add("mest_hub_q", per(lambda x: C.covariance_mest(x, "hub", q=0.7)), call='covariance_mest(X, "hub", q=0.7)')
add("mest_stu_nu", per(lambda x: C.covariance_mest(x, "stu", nu=3)), call='covariance_mest(X, "stu", nu=3.0)')
add("mest_tyl_det", per(lambda x: C.covariance_mest(x, "tyl", norm="determinant")), call='covariance_mest(X, "tyl", norm="determinant")')
Pm = X[0][:2]
add("EP", C.covariances_EP(X, Pm, "scm"), call='covariances_EP(X, X[0][:2], "scm")')
add("CX", C.covariances_X(X, "scm", alpha=0.3), call='covariances_X(X, "scm", 0.3)')
add("block", E.BlockCovariances([1, 3], "lwf").transform(X), call='BlockCovariances([1, 3], "lwf").transform(X)')
add("block2", E.BlockCovariances(2, "oas").transform(X), call='BlockCovariances(2, "oas").transform(X)')
Sc, f = C.cross_spectrum(X[:2], window=16, overlap=0.5)
add("cs_re", Sc.real, call='cross_spectrum(X[:2], 16, 0.5)[0]')
add("cs_im", Sc.imag, call='cross_spectrum(X[:2], 16, 0.5)[1]')
Sc, f = C.cross_spectrum(X[:2], window=15, overlap=0.5, fs=32.0, fmin=2.0, fmax=12.0)
add("csf_re", Sc.real, call='cross_spectrum(X[:2], 15, 0.5, 2.0, 12.0, 32.0)[0]')
add("csf_f", f, call='cross_spectrum(X[:2], 15, 0.5, 2.0, 12.0, 32.0)[2]')
for coh in ("ordinary", "instantaneous", "lagged", "imaginary"):
    Cc, f = C.coherence(X[:2], window=16, overlap=0.5, coh=coh)
    add("coh_"+coh, Cc, call=f'coherence(X[:2], 16, 0.5, None, None, None, "{coh}")[0]')
Cc, f = C.coherence(X[:2], window=16, overlap=0.5, fs=64.0, coh="lagged")
add("coh_lag_fs", Cc, call='coherence(X[:2], 16, 0.5, None, None, 64.0, "lagged")[0]')
add("ERPc", E.ERPCovariances(estimator="lwf").fit_transform(X, y), call='ERPCovariances("lwf").fit_transform(X, y)')
add("ERPsvd", E.ERPCovariances(estimator="scm", svd=2).fit_transform(X, y), call='ERPCovariances("scm", 2).fit_transform(X, y)')
add("XDC", E.XdawnCovariances(nfilter=2).fit_transform(X, y), 1e-9, call='XdawnCovariances(2).fit_transform(X, y)')
add("XDC2", E.XdawnCovariances(nfilter=3, applyfilters=False, estimator="oas").fit_transform(X, y), 1e-9, call='XdawnCovariances(3, False, "oas").fit_transform(X, y)')
add("TD", E.TimeDelayCovariances(delays=3).transform(X), call='TimeDelayCovariances(3).transform(X)')
add("TD2", E.TimeDelayCovariances(delays=[1, 4], estimator="lwf").transform(X), call='TimeDelayCovariances([1, 4], "lwf").transform(X)')
P = spd(5, 4)
add("shr", E.Shrinkage(0.3).transform(P), call='Shrinkage(0.3).transform(spd(5, 4))')
add("kern_rbf", E.Kernels("rbf").transform(X[:2]), call='Kernels("rbf").transform(X[:2])')
add("kern_lin", E.Kernels("linear").transform(X[:2]), call='Kernels("linear").transform(X[:2])')
for m in ("euclid", "logeuclid", "riemann"):
    add("k_"+m, K.kernel(P, metric=m), call=f'kernel(spd(5, 4), None, None, "{m}")')
    add("k2_"+m, K.kernel(P[:3], P[2:], Cref=P[0], metric=m), call=f'kernel(spd(5, 4)[:3], spd(5, 4)[2:], spd(5, 4)[0], "{m}")')
def emit(fname, keys, imp, setup):
    Ls = [l for l in L if any(l.startswith("E_"+k) for k in keys)]
    Ts = [t for t in T if any(t.split(", E_")[1].split(",")[0].startswith(k) for k in keys)]
    h = head.replace("import numpy as np\n", imp, 1)
    body = setup + "\n".join(Ls) + "\n\n@test\ndef test_x():\n" + "\n".join(Ts) + "\ntest_x()\n"
    open(os.path.join(D, fname), "w").write(h + body)
setup = "X = signals(6, 4, 64)\ny = np.array([0, 1, 0, 1, 0, 1])\n"
emit("test_estimation.py", ["EP", "TD"],
 '''import numpy as np
from pyriemann.geometry.covariance import covariances_EP
from pyriemann.estimation import TimeDelayCovariances
''', setup)
emit("test_estimators2.py", ["block", "ERPc", "ERPsvd", "shr"],
 '''import numpy as np
from pyriemann.estimation import BlockCovariances, ERPCovariances, Shrinkage
''', setup)
emit("test_covx.py", ["CX"],
 '''import numpy as np
from pyriemann.geometry.covariance import covariances_X
''', setup)
emit("test_mest.py", ["cov_", "mest_"],
 '''import numpy as np
from pyriemann.geometry.covariance import covariances, covariance_mest
''', setup)
emit("test_spectral.py", ["cs_", "csf_", "coh_", "k_", "k2_", "kern_", "XDC"],
 '''import numpy as np
from pyriemann.geometry.covariance import cross_spectrum, coherence
from pyriemann.geometry.kernel import kernel
from pyriemann.estimation import XdawnCovariances, Kernels
''', setup)
# ---- spatial filters
L = []; T = []
X = signals(12, 4, 48)
y2 = np.array([i % 2 for i in range(12)])
y3 = np.array([i % 3 for i in range(12)])
cov = C.covariances(X, "scm")
yr = np.array([(i * 7) % 5 + 0.5 for i in range(12)])
xd = S.Xdawn(nfilter=2).fit(X, y2)
add("xd_f", xd.filters_, 1e-8, call='Xdawn(2).fit(X, y2).filters_')
add("xd_e", xd.evokeds_, 1e-8, call='Xdawn(2).fit(X, y2).evokeds_')
add("xd_p", xd.patterns_, 1e-8, call='Xdawn(2).fit(X, y2).patterns_')
for nm, y_, M in (("b", y2, "euclid"), ("m", y3, "euclid"), ("br", y2, "riemann")):
    csp = S.CSP(nfilter=3, metric=M).fit(cov, y_)
    add("csp_"+nm+"_f", csp.filters_, 1e-8, call=f'CSP(3, "{M}").fit(cov, y{"2" if nm!="m" else "3"}).filters_')
    add("csp_"+nm+"_t", csp.transform(cov), 1e-8, call=f'CSP(3, "{M}").fit_transform(cov, y{"2" if nm!="m" else "3"})')
csp = S.CSP(nfilter=2, log=False).fit(cov, y2)
add("csp_nolog", csp.filters_ @ cov @ csp.filters_.T, 1e-8, call='CSP(2, "euclid", False).fit(cov, y2).transform_cov(cov)')
sp = S.SPoC(nfilter=2).fit(cov, yr)
add("spoc_f", sp.filters_, 1e-8, call='SPoC(2).fit(cov, yr).filters_')
add("spoc_t", sp.transform(cov), 1e-8, call='SPoC(2).fit_transform(cov, yr)')
imp = '''import numpy as np
from pyriemann.geometry.covariance import covariances
from pyriemann.spatialfilters import Xdawn, CSP, SPoC
'''
h = head.replace("import numpy as np\n", imp, 1)
body = ("X = signals(12, 4, 48)\ny2 = np.array([i % 2 for i in range(12)])\ny3 = np.array([i % 3 for i in range(12)])\n"
        "yr = np.array([float((i * 7) % 5) + 0.5 for i in range(12)])\ncov = covariances(X, \"scm\")\n"
        + "\n".join(L) + "\n\n@test\ndef test_spatial():\n" + "\n".join(T) + "\ntest_spatial()\n")
open(os.path.join(D, "test_spatial.py"), "w").write(h + body)
