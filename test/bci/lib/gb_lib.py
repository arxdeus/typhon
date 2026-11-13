import numpy as np
from mne.filter import IIRParams, filter_data
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

SFREQ = 125.0
N_CH = 6
N_T = 500
NF = 12

def log_power_features(X: np.ndarray[float, 3]) -> np.ndarray[float, 2]:
    n = X.shape[0]
    V = X * 1e-6
    out = np.empty((n, NF))
    tiny = np.finfo(np.float64).tiny
    bi = 0
    for lo, hi in ((8.0, 13.0), (13.0, 30.0)):
        F = filter_data(V, sfreq=SFREQ, l_freq=lo, h_freq=hi, method='iir',
                        iir_params=IIRParams(order=4, ftype='butter', output='sos'),
                        phase='zero', copy=True, verbose=False)
        for i in range(n):
            for c in range(N_CH):
                s = 0.0
                for t in range(N_T):
                    s += F[i, c, t] * F[i, c, t]
                p = s / float(N_T)
                out[i, bi * N_CH + c] = np.log(max(p, tiny))
        bi += 1
    return out

class Model:
    feats: List[float]
    labels: List[int]
    lda: HGB
    fitted: bool
    def __init__(self):
        self.feats = List[float]()
        self.labels = List[int]()
        self.lda = HGB(max_iter=50)
        self.fitted = False

models = List[Model]()

def _read(p: Ptr[float], n: int) -> np.ndarray[float, 3]:
    X = np.empty((n, N_CH, N_T))
    k = 0
    for i in range(n):
        for c in range(N_CH):
            for t in range(N_T):
                X[i, c, t] = p[k]
                k += 1
    return X

def _append(m: Model, F: np.ndarray[float, 2], y: Ptr[int], n: int):
    for i in range(n):
        for j in range(NF):
            m.feats.append(F[i, j])
        m.labels.append(y[i])

def _refit(m: Model) -> int:
    n = len(m.labels)
    if n < 4:
        return -1
    F = np.empty((n, NF))
    k = 0
    for i in range(n):
        for j in range(NF):
            F[i, j] = m.feats[k]
            k += 1
    yy = np.empty(n, dtype=int)
    for i in range(n):
        yy[i] = m.labels[i]
    m.lda = HGB(max_iter=50)
    m.lda.fit(F, yy)
    m.fitted = True
    return 0

@export
def create() -> int:
    models.append(Model())
    return len(models) - 1

@export
def feat(p: Ptr[float], n: int, out: Ptr[float]) -> int:
    F = log_power_features(_read(p, n))
    k = 0
    for i in range(n):
        for j in range(NF):
            out[k] = F[i, j]
            k += 1
    return 0

@export
def fit(h: int, p: Ptr[float], y: Ptr[int], n: int) -> int:
    m = models[h]
    m.feats = List[float]()
    m.labels = List[int]()
    _append(m, log_power_features(_read(p, n)), y, n)
    return _refit(m)

@export
def add_trials(h: int, p: Ptr[float], y: Ptr[int], n: int) -> int:
    m = models[h]
    _append(m, log_power_features(_read(p, n)), y, n)
    return len(m.labels)

@export
def add_features(h: int, f: Ptr[float], y: Ptr[int], n: int) -> int:
    m = models[h]
    for i in range(n):
        for j in range(NF):
            m.feats.append(f[i * NF + j])
        m.labels.append(y[i])
    return len(m.labels)

@export
def refit(h: int) -> int:
    return _refit(models[h])

def _predict_f(m: Model, F: np.ndarray[float, 2], n: int, out_labels: Ptr[int], out_proba: Ptr[float]) -> int:
    pred = m.lda.predict(F)
    pr = m.lda.predict_proba(F)
    nc = pr.shape[1]
    for i in range(n):
        out_labels[i] = pred[i]
        for k in range(nc):
            out_proba[i * nc + k] = pr[i, k]
    return nc

@export
def predict(h: int, p: Ptr[float], n: int, out_labels: Ptr[int], out_proba: Ptr[float]) -> int:
    m = models[h]
    if not m.fitted:
        return -1
    return _predict_f(m, log_power_features(_read(p, n)), n, out_labels, out_proba)

# Tree ensembles have no compact public parameter set, so the model file stores the
# training data (features + labels) and load() refits (deterministic, ~0.1 s for 600 trials).
@export
def save(h: int, cpath: cobj) -> int:
    path = str.from_ptr(cpath)
    m = models[h]
    with open(path, "w") as f:
        f.write("n " + str(len(m.labels)) + "\n")
        f.write("labels " + " ".join([str(v) for v in m.labels]) + "\n")
        f.write("feats " + " ".join([repr(v) for v in m.feats]) + "\n")
    return 0

@export
def load(cpath: cobj) -> int:
    path = str.from_ptr(cpath)
    m = Model()
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ")
            key = parts[0]
            if key == "labels":
                for i in range(1, len(parts)):
                    m.labels.append(int(parts[i]))
            elif key == "feats":
                for i in range(1, len(parts)):
                    m.feats.append(float(parts[i]))
    _refit(m)
    models.append(m)
    return len(models) - 1

@export
def n_trials(h: int) -> int:
    return len(models[h].labels)
