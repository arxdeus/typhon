# Ported from scikit-learn 1.9.1 (BSD-3-Clause)
#
# SAG / SAGA solver (sklearn/linear_model/_sag.py + _sag_fast.pyx.tp),
# dense float64 only. The sample stream reproduces sklearn's
# ArrayDataset.random(): our_rand_r xorshift seeded by
# check_random_state(random_state).randint(1, 2**31 - 1).

import math
import numpy as np

def _our_rand_r(seed: Ptr[u32]) -> int:
    if seed[0] == u32(0):
        seed[0] = u32(1)
    seed[0] ^= seed[0] << u32(13)
    seed[0] ^= seed[0] >> u32(17)
    seed[0] ^= seed[0] << u32(5)
    return int(seed[0] % u32(2147483648))

def _dataset_seed(random_state) -> int:
    if isinstance(random_state, NoneType):
        return np.random.randint(1, 2147483647)
    elif isinstance(random_state, Optional):
        if random_state is None:
            return np.random.randint(1, 2147483647)
        return np.random.RandomState(random_state.__val__()).randint(1, 2147483647)
    elif isinstance(random_state, int):
        return np.random.RandomState(random_state).randint(1, 2147483647)
    else:
        return random_state.randint(1, 2147483647)

def get_auto_step_size(max_squared_sum: float, alpha_scaled: float, loss: str,
                       fit_intercept: bool, n_samples: int = 0, is_saga: bool = False) -> float:
    L = 0.0
    if loss == "log" or loss == "multinomial":
        L = 0.25 * (max_squared_sum + float(int(fit_intercept))) + alpha_scaled
    elif loss == "squared":
        L = max_squared_sum + float(int(fit_intercept)) + alpha_scaled
    else:
        raise ValueError(f"Unknown loss function for SAG solver, got {loss} instead of 'log' or 'squared'")
    if is_saga:
        mun = min(2.0 * float(n_samples) * alpha_scaled, L)
        return 1.0 / (2.0 * L + mun)
    else:
        return 1.0 / L

def _soft(x: float, s: float) -> float:
    return max(x - s, 0.0) - max(-x - s, 0.0)

def _grad_binomial(y: float, raw: float) -> float:
    if raw > -37.0:
        e = math.exp(-raw)
        return ((1.0 - y) - y * e) / (1.0 + e)
    else:
        return math.exp(raw) - y

class _SagState:
    w: Ptr[float]
    sum_gradient: Ptr[float]
    cumulative_sums: Ptr[float]
    cumulative_sums_prox: Ptr[float]
    feature_hist: Ptr[int]
    n_samples: int
    n_classes: int
    n_features: int
    prox: bool

    def __init__(self, w: Ptr[float], sg: Ptr[float], n_samples: int, n_features: int,
                 n_classes: int, prox: bool):
        self.w = w
        self.sum_gradient = sg
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.prox = prox
        self.cumulative_sums = Ptr[float](n_samples)
        self.cumulative_sums_prox = Ptr[float](n_samples)
        self.feature_hist = Ptr[int](n_features)
        for j in range(n_features):
            self.feature_hist[j] = 0

    def lagged_update(self, wscale: float, xnnz: int, sample_itr: int, reset: bool) -> int:
        nc = self.n_classes
        w = self.w
        sg = self.sum_gradient
        cs = self.cumulative_sums
        csp = self.cumulative_sums_prox
        prox = self.prox
        for feature_ind in range(xnnz):
            f_idx = feature_ind * nc
            cum_sum = cs[sample_itr - 1]
            cum_sum_prox = 0.0
            if prox:
                cum_sum_prox = csp[sample_itr - 1]
            fh = self.feature_hist[feature_ind]
            if fh != 0:
                cum_sum -= cs[fh - 1]
                if prox:
                    cum_sum_prox -= csp[fh - 1]
            if not prox:
                for class_ind in range(nc):
                    idx = f_idx + class_ind
                    w[idx] -= cum_sum * sg[idx]
                    if reset:
                        w[idx] *= wscale
                        if not math.isfinite(w[idx]):
                            return -1
            else:
                for class_ind in range(nc):
                    idx = f_idx + class_ind
                    if abs(sg[idx] * cum_sum) < cum_sum_prox:
                        w[idx] -= cum_sum * sg[idx]
                        w[idx] = _soft(w[idx], cum_sum_prox)
                    else:
                        last_update_ind = fh
                        if last_update_ind == -1:
                            last_update_ind = sample_itr - 1
                        lagged_ind = sample_itr - 1
                        while lagged_ind > last_update_ind - 1:
                            grad_step = 0.0
                            prox_step = 0.0
                            if lagged_ind > 0:
                                grad_step = cs[lagged_ind] - cs[lagged_ind - 1]
                                prox_step = csp[lagged_ind] - csp[lagged_ind - 1]
                            else:
                                grad_step = cs[lagged_ind]
                                prox_step = csp[lagged_ind]
                            w[idx] -= sg[idx] * grad_step
                            w[idx] = _soft(w[idx], prox_step)
                            lagged_ind -= 1
                    if reset:
                        w[idx] *= wscale
                        if not math.isfinite(w[idx]):
                            return -1
            if reset:
                self.feature_hist[feature_ind] = sample_itr % self.n_samples
            else:
                self.feature_hist[feature_ind] = sample_itr
        if reset:
            cs[sample_itr - 1] = 0.0
            if prox:
                csp[sample_itr - 1] = 0.0
        return 0

class SagMemory:
    """warm_start_mem of sklearn's sag_solver (kept across a regularization path)."""
    sum_gradient: np.ndarray[float, 1]
    intercept_sum_gradient: np.ndarray[float, 1]
    gradient_memory: np.ndarray[float, 1]
    seen: np.ndarray[bool, 1]
    num_seen: int
    has: bool
    def __init__(self):
        self.sum_gradient = np.zeros(0)
        self.intercept_sum_gradient = np.zeros(0)
        self.gradient_memory = np.zeros(0)
        self.seen = np.zeros(0, dtype=bool)
        self.num_seen = 0
        self.has = False

def sag_solver(X, y, sample_weight, loss: str, alpha: float, beta: float, max_iter: int,
               tol: float, random_state, coef_init: np.ndarray[float, 2],
               intercept_init: np.ndarray[float, 1], fit_intercept: bool,
               is_saga: bool, max_squared_sum: float = -1.0, mem: Optional[SagMemory] = None,
               seed: int = -1):
    """Runs sag64 in place on coef_init (n_features, n_classes) and
    intercept_init (n_classes,). y holds labels 0..K-1 for 'multinomial',
    {0,1} for 'log', targets for 'squared'. Returns n_iter."""
    Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
    ya = np.asarray(y, dtype=float)
    n_samples, n_features = Xa.shape
    alpha_scaled = alpha / float(n_samples)
    beta_scaled = beta / float(n_samples)
    n_classes = coef_init.shape[1]
    sw = np.asarray(sample_weight, dtype=float)
    seedv = u32(max(_dataset_seed(random_state), 1)) if seed < 0 else u32(max(seed, 1))
    if max_squared_sum < 0.0:
        max_squared_sum = 0.0
        for i in range(n_samples):
            s = 0.0
            for j in range(n_features):
                s += Xa[i, j] * Xa[i, j]
            max_squared_sum = max(max_squared_sum, s)
    step_size = get_auto_step_size(max_squared_sum, alpha_scaled, loss, fit_intercept,
                                   n_samples, is_saga)
    if step_size * alpha_scaled == 1.0:
        raise ZeroDivisionError("Current sag implementation does not handle the case step_size * alpha_scaled == 1")
    multinomial = loss == "multinomial"
    is_log = loss == "log"
    nc = n_classes
    w = Ptr[float](n_features * nc)
    for j in range(n_features):
        for k in range(nc):
            w[j * nc + k] = coef_init[j, k]
    intercept = Ptr[float](nc)
    for k in range(nc):
        intercept[k] = intercept_init[k]
    sum_gradient = Ptr[float](n_features * nc)
    for i in range(n_features * nc):
        sum_gradient[i] = 0.0
    gradient_memory = Ptr[float](n_samples * nc)
    for i in range(n_samples * nc):
        gradient_memory[i] = 0.0
    seen = Ptr[bool](n_samples)
    for i in range(n_samples):
        seen[i] = False
    num_seen = 0
    use_mem = mem is not None and mem.__val__().has
    if use_mem:
        m0 = mem.__val__()
        for i in range(n_features * nc):
            sum_gradient[i] = m0.sum_gradient[i]
        for i in range(n_samples * nc):
            gradient_memory[i] = m0.gradient_memory[i]
        for i in range(n_samples):
            seen[i] = m0.seen[i]
        num_seen = m0.num_seen
    isg = Ptr[float](nc)
    prev = Ptr[float](n_features * nc)
    for i in range(n_features * nc):
        prev[i] = 0.0
    prediction = Ptr[float](nc)
    gradient = Ptr[float](nc)
    prox = beta > 0.0 and is_saga
    st = _SagState(w, sum_gradient, n_samples, n_features, nc, prox)
    for k in range(nc):
        isg[k] = 0.0
    if use_mem:
        for k in range(nc):
            isg[k] = mem.__val__().intercept_sum_gradient[k]
    cs = st.cumulative_sums
    csp = st.cumulative_sums_prox
    cs[0] = 0.0
    wscale_update = 1.0 - step_size * alpha_scaled
    wscale = 1.0
    status = 0
    xp = Xa.data
    n_iter = 0
    for it in range(max_iter):
        n_iter = it
        for sample_itr in range(n_samples):
            sample_ind = _our_rand_r(__ptr__(seedv)) % n_samples
            yv = ya[sample_ind]
            swv = sw[sample_ind]
            x = xp + sample_ind * n_features
            s_idx = sample_ind * nc
            if not seen[sample_ind]:
                num_seen += 1
                seen[sample_ind] = True
            if sample_itr > 0:
                status = st.lagged_update(wscale, n_features, sample_itr, False)
                if status == -1:
                    break
            for k in range(nc):
                inner = 0.0
                for j in range(n_features):
                    inner += w[j * nc + k] * x[j]
                prediction[k] = wscale * inner + intercept[k]
            if multinomial:
                mx = prediction[0]
                for k in range(1, nc):
                    if mx < prediction[k]:
                        mx = prediction[k]
                se = 0.0
                for k in range(nc):
                    gradient[k] = math.exp(prediction[k] - mx)
                    se += gradient[k]
                for k in range(nc):
                    gradient[k] /= se
                    gradient[k] = (gradient[k] - (1.0 if yv == float(k) else 0.0)) * swv
            elif is_log:
                gradient[0] = _grad_binomial(yv, prediction[0]) * swv
            else:
                gradient[0] = (prediction[0] - yv) * swv
            wscale *= wscale_update
            for j in range(n_features):
                val = x[j]
                f_idx = j * nc
                for k in range(nc):
                    gc = val * (gradient[k] - gradient_memory[s_idx + k])
                    if is_saga:
                        w[f_idx + k] -= gc * step_size * (1.0 - 1.0 / float(num_seen)) / wscale
                    sum_gradient[f_idx + k] += gc
            if fit_intercept:
                for k in range(nc):
                    gc = gradient[k] - gradient_memory[s_idx + k]
                    isg[k] += gc
                    gc *= step_size * (1.0 - 1.0 / float(num_seen))
                    if is_saga:
                        intercept[k] -= (step_size * isg[k] / float(num_seen) * 1.0) + gc
                    else:
                        intercept[k] -= step_size * isg[k] / float(num_seen) * 1.0
                    if not math.isfinite(intercept[k]):
                        status = -1
                        break
                if status == -1:
                    break
            for k in range(nc):
                gradient_memory[s_idx + k] = gradient[k]
            if sample_itr == 0:
                cs[0] = step_size / (wscale * float(num_seen))
                if prox:
                    csp[0] = step_size * beta_scaled / wscale
            else:
                cs[sample_itr] = cs[sample_itr - 1] + step_size / (wscale * float(num_seen))
                if prox:
                    csp[sample_itr] = csp[sample_itr - 1] + step_size * beta_scaled / wscale
            if wscale < 1e-9:
                status = st.lagged_update(wscale, n_features, sample_itr + 1, True)
                if status == 0:
                    wscale = 1.0
                if status == -1:
                    break
        if status == -1:
            break
        status = st.lagged_update(wscale, n_features, n_samples, True)
        if status == 0:
            wscale = 1.0
        if status == -1:
            break
        max_change = 0.0
        max_weight = 0.0
        for idx in range(n_features * nc):
            max_weight = max(max_weight, abs(w[idx]))
            max_change = max(max_change, abs(w[idx] - prev[idx]))
            prev[idx] = w[idx]
        if (max_weight != 0.0 and max_change / max_weight <= tol) or (max_weight == 0.0 and max_change == 0.0):
            break
    n_iter += 1
    if status == -1:
        raise ValueError(f"Floating-point under-/overflow occurred at epoch #{n_iter}. Scaling input data with StandardScaler or MinMaxScaler might help.")
    for j in range(n_features):
        for k in range(nc):
            coef_init[j, k] = w[j * nc + k]
    for k in range(nc):
        intercept_init[k] = intercept[k]
    if mem is not None:
        m1 = mem.__val__()
        m1.sum_gradient = np.empty(n_features * nc)
        for i in range(n_features * nc):
            m1.sum_gradient[i] = sum_gradient[i]
        m1.gradient_memory = np.empty(n_samples * nc)
        for i in range(n_samples * nc):
            m1.gradient_memory[i] = gradient_memory[i]
        m1.seen = np.empty(n_samples, dtype=bool)
        for i in range(n_samples):
            m1.seen[i] = seen[i]
        m1.intercept_sum_gradient = np.empty(nc)
        for k in range(nc):
            m1.intercept_sum_gradient[k] = isg[k]
        m1.num_seen = num_seen
        m1.has = True
    return n_iter
