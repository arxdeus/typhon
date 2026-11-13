# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.datasets._samples_generator. Every generator draws from a legacy
# numpy RandomState in the same order as sklearn, so the outputs equal the
# reference for the same integer seed.

import math
import numpy as np
from numpy.random.legacy import RandomState, check_random_state

def _shuffle_rows(rs, n: int):
    # sklearn.utils.shuffle(X, y, random_state=rs): resample(replace=False)
    idx = np.arange(n)
    rs.shuffle(idx)
    return idx

def sample_without_replacement(n_population: int, n_samples: int, method: str = "auto",
                               random_state = None):
    rng = check_random_state(random_state)
    if n_population < 0:
        raise ValueError(f"n_population should be greater than 0, got {n_population}.")
    if n_samples > n_population:
        raise ValueError(f"n_population should be greater or equal than n_samples, got n_samples > n_population ({n_samples} > {n_population})")
    ratio = float(n_samples) / float(n_population) if n_population != 0 else 1.0
    if method == "auto" and ratio > 0.01 and ratio < 0.99:
        return rng.permutation(n_population)[:n_samples].copy()
    out = np.empty(n_samples, dtype=int)
    if (method == "auto" or method == "tracking_selection") and ratio < 0.2:
        selected = Set[int]()
        for i in range(n_samples):
            j = int(rng.randint(n_population))
            while j in selected:
                j = int(rng.randint(n_population))
            selected.add(j)
            out[i] = j
    elif method == "auto" or method == "tracking_selection" or method == "reservoir_sampling":
        for i in range(n_samples):
            out[i] = i
        for i in range(n_samples, n_population):
            j = int(rng.randint(0, i + 1))
            if j < n_samples:
                out[j] = i
    elif method == "pool":
        pool = np.arange(n_population)
        for i in range(n_samples):
            j = int(rng.randint(n_population - i))
            out[i] = pool[j]
            pool[j] = pool[n_population - i - 1]
    else:
        raise ValueError(f"Expected a method name in ('auto', 'tracking_selection', 'reservoir_sampling', 'pool'), got {method}. ")
    return out

def _generate_hypercube(samples: int, dimensions: int, rng) -> np.ndarray[float, 2]:
    if dimensions > 30:
        a = rng.randint(2, size=(samples, dimensions - 30))
        b = _generate_hypercube(samples, 30, rng)
        out = np.empty((samples, dimensions))
        for i in range(samples):
            for j in range(dimensions - 30):
                out[i, j] = float(a[i, j])
            for j in range(30):
                out[i, dimensions - 30 + j] = b[i, j]
        return out
    v = sample_without_replacement(1 << dimensions, samples, random_state=rng)
    out = np.empty((samples, dimensions))
    for i in range(samples):
        x = int(v[i])
        for k in range(dimensions):
            out[i, k] = float((x >> (dimensions - 1 - k)) & 1)
    return out

def make_classification(n_samples: int = 100, n_features: int = 20, *, n_informative: int = 2,
                        n_redundant: int = 2, n_repeated: int = 0, n_classes: int = 2,
                        n_clusters_per_class: int = 2, weights = None, flip_y: float = 0.01,
                        class_sep: float = 1.0, hypercube: bool = True, shift = 0.0,
                        scale = 1.0, shuffle: bool = True, random_state = None):
    generator = check_random_state(random_state)
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError("Number of informative, redundant and repeated features must sum to less than the number of total features")
    if float(n_informative) < math.log2(float(n_classes * n_clusters_per_class)):
        raise ValueError(f"n_classes({n_classes}) * n_clusters_per_class({n_clusters_per_class}) must be smaller or equal 2**n_informative({n_informative})={1 << n_informative}")
    weights_ = List[float]()
    if weights is None:
        weights_ = [1.0 / float(n_classes)] * n_classes
    else:
        w = [float(v) for v in weights]
        if len(w) != n_classes and len(w) != n_classes - 1:
            raise ValueError("Weights specified but incompatible with number of classes.")
        if len(w) == n_classes - 1:
            s = 0.0
            for v in w:
                s += v
            w.append(1.0 - s)
        weights_ = w
    n_random = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class
    nspc = [int(float(n_samples) * weights_[k % n_classes] / float(n_clusters_per_class)) for k in range(n_clusters)]
    tot = 0
    for v in nspc:
        tot += v
    for i in range(n_samples - tot):
        nspc[i % n_clusters] += 1
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    centroids = _generate_hypercube(n_clusters, n_informative, generator)
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.uniform(size=(n_clusters, 1))
        centroids *= generator.uniform(size=(1, n_informative))
    X[:, :n_informative] = generator.standard_normal(size=(n_samples, n_informative))
    stop = 0
    for k in range(n_clusters):
        start = stop
        stop = stop + nspc[k]
        for r in range(start, stop):
            y[r] = k % n_classes
        Xk = np.ascontiguousarray(X[start:stop, :n_informative])
        A = 2 * generator.uniform(size=(n_informative, n_informative)) - 1
        P = np.dot(Xk, A)
        for r in range(stop - start):
            for j in range(n_informative):
                X[start + r, j] = P[r, j] + centroids[k, j]
    if n_redundant > 0:
        B = 2 * generator.uniform(size=(n_informative, n_redundant)) - 1
        X[:, n_informative:n_informative + n_redundant] = np.dot(np.ascontiguousarray(X[:, :n_informative]), B)
    n = n_informative + n_redundant
    if n_repeated > 0:
        ind = ((n - 1) * generator.uniform(size=n_repeated) + 0.5).astype(int)
        for t in range(n_repeated):
            for r in range(n_samples):
                X[r, n + t] = X[r, int(ind[t])]
    if n_random > 0:
        X[:, n_features - n_random:] = generator.standard_normal(size=(n_samples, n_random))
    if flip_y >= 0.0:
        flip = generator.uniform(size=n_samples) < flip_y
        cnt = 0
        for r in range(n_samples):
            if flip[r]:
                cnt += 1
        nv = generator.randint(n_classes, size=cnt)
        t = 0
        for r in range(n_samples):
            if flip[r]:
                y[r] = nv[t]
                t += 1
    if shift is None:
        X += (2 * generator.uniform(size=n_features) - 1) * class_sep
    else:
        X += float(shift)
    if scale is None:
        X *= 1 + 100 * generator.uniform(size=n_features)
    else:
        X *= float(scale)
    if shuffle:
        idx = _shuffle_rows(generator, n_samples)
        X = X[idx]
        y = y[idx]
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X = np.ascontiguousarray(X[:, indices])
    return X, y

def make_multilabel_classification(n_samples: int = 100, n_features: int = 20, *, n_classes: int = 5,
                                   n_labels: int = 2, length: int = 50, allow_unlabeled: bool = True,
                                   random_state = None):
    # dense X, dense indicator Y (sparse / return_distributions are not ported)
    generator = check_random_state(random_state)
    p_c = generator.uniform(size=n_classes)
    p_c /= p_c.sum()
    cumulative_p_c = np.cumsum(p_c)
    p_w_c = generator.uniform(size=(n_features, n_classes))
    p_w_c /= np.sum(p_w_c, axis=0)
    X = np.zeros((n_samples, n_features))
    Y = np.zeros((n_samples, n_classes), dtype=int)
    for i in range(n_samples):
        y_size = n_classes + 1
        while (not allow_unlabeled and y_size == 0) or y_size > n_classes:
            y_size = int(generator.poisson(float(n_labels)))
        ys = Set[int]()
        while len(ys) != y_size:
            c = np.searchsorted(cumulative_p_c, generator.uniform(size=y_size - len(ys)))
            for v in c:
                ys.add(int(v))
        # CPython iterates a small-int set in ascending order
        yl = sorted(list(ys))
        n_words = 0
        while n_words == 0:
            n_words = int(generator.poisson(float(length)))
        if len(yl) == 0:
            words = generator.randint(n_features, size=n_words)
            for w in words:
                X[i, int(w)] += 1.0
            continue
        cw = np.empty(n_features)
        acc = 0.0
        for f in range(n_features):
            s = 0.0
            for c2 in yl:
                s += p_w_c[f, c2]
            acc += s
            cw[f] = acc
        cw /= cw[n_features - 1]
        words = np.searchsorted(cw, generator.uniform(size=n_words))
        for w in words:
            X[i, int(w)] += 1.0
        for c2 in yl:
            Y[i, c2] = 1
    return X, Y

def make_hastie_10_2(n_samples: int = 12000, *, random_state = None):
    rs = check_random_state(random_state)
    X = rs.normal(size=(n_samples, 10))
    y = np.empty(n_samples)
    for i in range(n_samples):
        s = 0.0
        for j in range(10):
            s += X[i, j] ** 2.0
        y[i] = 1.0 if s > 9.34 else -1.0
    return X, y

def make_regression(n_samples: int = 100, n_features: int = 100, *, n_informative: int = 10,
                    n_targets: Literal[int] = 1, bias: float = 0.0, effective_rank = None,
                    tail_strength: float = 0.5, noise: float = 0.0, shuffle: bool = True,
                    coef: Literal[bool] = False, random_state = None):
    n_informative = min(n_features, n_informative)
    generator = check_random_state(random_state)
    X = np.empty((0, 0))
    if effective_rank is None:
        X = generator.standard_normal(size=(n_samples, n_features))
    else:
        X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features,
                                 effective_rank=effective_rank, tail_strength=tail_strength,
                                 random_state=generator)
    gt = np.zeros((n_features, n_targets))
    gt[:n_informative, :] = 100 * generator.uniform(size=(n_informative, n_targets))
    y = np.dot(X, gt) + bias
    if noise > 0.0:
        y += generator.normal(scale=noise, size=(n_samples, n_targets))
    if shuffle:
        idx = _shuffle_rows(generator, n_samples)
        X = X[idx]
        y = y[idx]
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X = np.ascontiguousarray(X[:, indices])
        gt = gt[indices]
    if n_targets == 1:
        if coef:
            return X, np.ascontiguousarray(y[:, 0]), np.ascontiguousarray(gt[:, 0])
        else:
            return X, np.ascontiguousarray(y[:, 0])
    else:
        if coef:
            return X, y, gt
        else:
            return X, y

def _split_n(n_samples) -> Tuple[int, int]:
    if isinstance(n_samples, int):
        return n_samples // 2, n_samples - n_samples // 2
    else:
        if len(n_samples) != 2:
            raise ValueError("When a tuple, n_samples must have exactly two elements.")
        return int(n_samples[0]), int(n_samples[1])

def _two_rows(ox, oy, ix, iy, n_out: int, n_in: int):
    n = n_out + n_in
    X = np.empty((n, 2))
    y = np.empty(n, dtype=int)
    for i in range(n_out):
        X[i, 0] = ox[i]; X[i, 1] = oy[i]; y[i] = 0
    for i in range(n_in):
        X[n_out + i, 0] = ix[i]; X[n_out + i, 1] = iy[i]; y[n_out + i] = 1
    return X, y

def make_circles(n_samples = 100, *, shuffle: bool = True, noise = None, random_state = None,
                 factor: float = 0.8):
    n_out, n_in = _split_n(n_samples)
    generator = check_random_state(random_state)
    lo = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    li = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
    X, y = _two_rows(np.cos(lo), np.sin(lo), np.cos(li) * factor, np.sin(li) * factor, n_out, n_in)
    if shuffle:
        idx = _shuffle_rows(generator, n_out + n_in)
        X = X[idx]
        y = y[idx]
    if noise is not None:
        X += generator.normal(scale=float(noise), size=X.shape)
    return X, y

def make_moons(n_samples = 100, *, shuffle: bool = True, noise = None, random_state = None):
    n_out, n_in = _split_n(n_samples)
    generator = check_random_state(random_state)
    lo = np.linspace(0, np.pi, n_out)
    li = np.linspace(0, np.pi, n_in)
    X, y = _two_rows(np.cos(lo), np.sin(lo), 1 - np.cos(li), 1 - np.sin(li) - 0.5, n_out, n_in)
    if shuffle:
        idx = _shuffle_rows(generator, n_out + n_in)
        X = X[idx]
        y = y[idx]
    if noise is not None:
        X += generator.normal(scale=float(noise), size=X.shape)
    return X, y

def make_blobs(n_samples = 100, n_features: int = 2, *, centers = None, cluster_std = 1.0,
               center_box = (-10.0, 10.0), shuffle: bool = True, random_state = None,
               return_centers: Literal[bool] = False):
    generator = check_random_state(random_state)
    C = np.empty((0, 0))
    per = List[int]()
    if isinstance(n_samples, int):
        if centers is None or isinstance(centers, int):
            nc = 3 if centers is None else int(centers)
            C = generator.uniform(float(center_box[0]), float(center_box[1]), size=(nc, n_features))
        else:
            C = np.asarray(centers, dtype=float)
        nc = C.shape[0]
        per = [n_samples // nc] * nc
        for i in range(n_samples % nc):
            per[i] += 1
    else:
        per = [int(v) for v in n_samples]
        if centers is None:
            C = generator.uniform(float(center_box[0]), float(center_box[1]), size=(len(per), n_features))
        else:
            C = np.asarray(centers, dtype=float)
        if C.shape[0] != len(per):
            raise ValueError(f"Length of `n_samples` not consistent with number of centers. Got n_samples = {n_samples} and centers = {C}")
    nc = C.shape[0]
    nf = C.shape[1]
    stds = List[float]()
    if isinstance(cluster_std, float) or isinstance(cluster_std, int):
        stds = [float(cluster_std)] * nc
    else:
        stds = [float(v) for v in cluster_std]
        if len(stds) != nc:
            raise ValueError(f"Length of `clusters_std` not consistent with number of centers. Got centers = {C} and cluster_std = {cluster_std}")
    tot = 0
    for v in per:
        tot += v
    X = np.empty((tot, nf))
    y = np.empty(tot, dtype=int)
    r = 0
    for i in range(nc):
        for _ in range(per[i]):
            for j in range(nf):
                X[r, j] = float(generator.normal(C[i, j], stds[i]))
            y[r] = i
            r += 1
    if shuffle:
        idx = _shuffle_rows(generator, tot)
        X = X[idx]
        y = y[idx]
    if return_centers:
        return X, y, C
    else:
        return X, y

def make_friedman1(n_samples: int = 100, n_features: int = 10, *, noise: float = 0.0,
                   random_state = None):
    generator = check_random_state(random_state)
    X = generator.uniform(size=(n_samples, n_features))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4] + noise * generator.standard_normal(size=n_samples))
    return X, y

def _friedman_X(generator, n_samples: int):
    X = generator.uniform(size=(n_samples, 4))
    for i in range(n_samples):
        X[i, 0] *= 100
        X[i, 1] *= 520 * np.pi
        X[i, 1] += 40 * np.pi
        X[i, 3] *= 10
        X[i, 3] += 1
    return X

def make_friedman2(n_samples: int = 100, *, noise: float = 0.0, random_state = None):
    generator = check_random_state(random_state)
    X = _friedman_X(generator, n_samples)
    y = (X[:, 0] ** 2 + (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5 \
        + noise * generator.standard_normal(size=n_samples)
    return X, y

def make_friedman3(n_samples: int = 100, *, noise: float = 0.0, random_state = None):
    generator = check_random_state(random_state)
    X = _friedman_X(generator, n_samples)
    y = np.arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]) \
        + noise * generator.standard_normal(size=n_samples)
    return X, y

def make_low_rank_matrix(n_samples: int = 100, n_features: int = 100, *, effective_rank = 10,
                         tail_strength: float = 0.5, random_state = None):
    import scipy.linalg as sl
    generator = check_random_state(random_state)
    n = min(n_samples, n_features)
    u, _ = sl.qr(generator.standard_normal(size=(n_samples, n)), mode="economic", check_finite=False)
    v, _ = sl.qr(generator.standard_normal(size=(n_features, n)), mode="economic", check_finite=False)
    er = float(effective_rank)
    si = np.arange(n).astype(float)
    low_rank = (1 - tail_strength) * np.exp(-1.0 * (si / er) ** 2)
    tail = tail_strength * np.exp(-0.1 * si / er)
    s = np.identity(n) * (low_rank + tail)
    return np.dot(np.dot(u, s), v.T)

def make_sparse_uncorrelated(n_samples: int = 100, n_features: int = 10, *, random_state = None):
    generator = check_random_state(random_state)
    X = generator.normal(loc=0, scale=1, size=(n_samples, n_features))
    loc = X[:, 0] + 2 * X[:, 1] - 2 * X[:, 2] - 1.5 * X[:, 3]
    y = np.empty(n_samples)
    for i in range(n_samples):
        y[i] = float(generator.normal(loc[i], 1.0))
    return X, y

def make_sparse_coded_signal(n_samples: int, *, n_components: int, n_features: int,
                             n_nonzero_coefs: int, random_state = None):
    generator = check_random_state(random_state)
    D = generator.standard_normal(size=(n_features, n_components))
    D /= np.sqrt(np.sum(D ** 2, axis=0))
    X = np.zeros((n_components, n_samples))
    for i in range(n_samples):
        idx = np.arange(n_components)
        generator.shuffle(idx)
        v = generator.standard_normal(size=n_nonzero_coefs)
        for k in range(n_nonzero_coefs):
            X[int(idx[k]), i] = v[k]
    Y = np.dot(D, X)
    return np.ascontiguousarray(Y.T), np.ascontiguousarray(D.T), np.ascontiguousarray(X.T)

def make_spd_matrix(n_dim: int, *, random_state = None):
    import scipy.linalg as sl
    generator = check_random_state(random_state)
    A = generator.uniform(size=(n_dim, n_dim))
    U, _, Vt = sl.svd(np.dot(A.T, A), check_finite=False)
    X = np.dot(np.dot(U, 1.0 + np.diag(generator.uniform(size=n_dim))), Vt)
    return X

def make_sparse_spd_matrix(n_dim: int = 1, *, alpha: float = 0.95, norm_diag: bool = False,
                           smallest_coef: float = 0.1, largest_coef: float = 0.9,
                           sparse_format = None, random_state = None):
    # dense result (sparse_format is not ported)
    rs = check_random_state(random_state)
    tot = n_dim * n_dim
    size = int(round((1 - alpha) * float(tot)))
    # scipy.sparse.random_array: choice(tot, size, replace=False) on a legacy
    # RandomState is permutation(tot)[:size]; then the data sampler
    raveled = rs.permutation(tot)[:size].copy()
    vals = rs.uniform(low=smallest_coef, high=largest_coef, size=size)
    aux = np.zeros((n_dim, n_dim))
    for k in range(size):
        r = int(raveled[k]) % n_dim
        c = int(raveled[k]) // n_dim
        if r > c:
            aux[r, c] = vals[k]
    perm = rs.permutation(n_dim)
    chol = np.zeros((n_dim, n_dim))
    for i in range(n_dim):
        for j in range(n_dim):
            chol[i, j] = aux[int(perm[j]), int(perm[i])]
        chol[i, i] -= 1.0
    prec = np.dot(chol.T, chol)
    if norm_diag:
        d = 1.0 / np.sqrt(np.diag(prec))
        for i in range(n_dim):
            for j in range(n_dim):
                prec[i, j] = d[i] * prec[i, j] * d[j]
    return prec

def make_swiss_roll(n_samples: int = 100, *, noise: float = 0.0, random_state = None,
                    hole: bool = False):
    generator = check_random_state(random_state)
    t = np.empty(n_samples)
    yv = np.empty(n_samples)
    if not hole:
        t = 1.5 * np.pi * (1 + 2 * generator.uniform(size=n_samples))
        yv = 21 * generator.uniform(size=n_samples)
    else:
        cx = List[float]()
        cy = List[float]()
        for i in range(3):
            for j in range(3):
                if i * 3 + j != 4:
                    cx.append(np.pi * (1.5 + i))
                    cy.append(float(j * 7))
        ci = generator.choice(8, n_samples)
        par = generator.uniform(size=(2, n_samples))
        for k in range(n_samples):
            t[k] = cx[int(ci[k])] + par[0, k] * np.pi
            yv[k] = cy[int(ci[k])] + par[1, k] * 7.0
    x = t * np.cos(t)
    z = t * np.sin(t)
    nz = noise * generator.standard_normal(size=(3, n_samples))
    X = np.empty((n_samples, 3))
    for k in range(n_samples):
        X[k, 0] = x[k] + nz[0, k]
        X[k, 1] = yv[k] + nz[1, k]
        X[k, 2] = z[k] + nz[2, k]
    return X, t

def make_s_curve(n_samples: int = 100, *, noise: float = 0.0, random_state = None):
    generator = check_random_state(random_state)
    t = 3 * np.pi * (generator.uniform(size=n_samples) - 0.5)
    X = np.empty((n_samples, 3))
    u = 2.0 * generator.uniform(size=n_samples)
    for k in range(n_samples):
        X[k, 0] = np.sin(t[k])
        X[k, 1] = u[k]
        X[k, 2] = np.sign(t[k]) * (np.cos(t[k]) - 1)
    nz = noise * generator.standard_normal(size=(3, n_samples))
    for k in range(n_samples):
        for j in range(3):
            X[k, j] += nz[j, k]
    return X, t

def make_gaussian_quantiles(*, mean = None, cov: float = 1.0, n_samples: int = 100,
                            n_features: int = 2, n_classes: int = 3, shuffle: bool = True,
                            random_state = None):
    if n_samples < n_classes:
        raise ValueError("n_samples must be at least n_classes")
    generator = check_random_state(random_state)
    m = np.zeros(n_features) if mean is None else np.array(mean, dtype=float)
    X = generator.multivariate_normal(m, cov * np.identity(n_features), (n_samples,))
    idx = np.argsort(np.sum((X - m) ** 2, axis=1))
    X = X[idx, :]
    step = n_samples // n_classes
    y = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        y[i] = min(i // step, n_classes - 1) if step > 0 else n_classes - 1
    if shuffle:
        sidx = _shuffle_rows(generator, n_samples)
        X = X[sidx]
        y = y[sidx]
    return X, y

def _labels_from_sizes(sizes) -> np.ndarray[int, 1]:
    out = List[int]()
    for c in range(sizes.shape[0]):
        for _ in range(int(sizes[c])):
            out.append(c)
    return np.array(out, dtype=int)

def make_biclusters(shape, n_clusters: int, *, noise: float = 0.0, minval = 10, maxval = 100,
                    shuffle: bool = True, random_state = None):
    generator = check_random_state(random_state)
    n_rows = int(shape[0]); n_cols = int(shape[1])
    consts = generator.uniform(float(minval), float(maxval), n_clusters)
    pv = np.full(n_clusters, 1.0 / float(n_clusters))
    row_labels = _labels_from_sizes(generator.multinomial(n_rows, pv))
    col_labels = _labels_from_sizes(generator.multinomial(n_cols, pv))
    result = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            if row_labels[i] == col_labels[j]:
                result[i, j] += consts[row_labels[i]]
    if noise > 0:
        result += generator.normal(scale=noise, size=(n_rows, n_cols))
    if shuffle:
        # _shuffle(result, random_state): a fresh RandomState from the seed
        g2 = check_random_state(random_state)
        ri = g2.permutation(n_rows)
        cj = g2.permutation(n_cols)
        result = np.ascontiguousarray(result[ri][:, cj])
        row_labels = row_labels[ri]
        col_labels = col_labels[cj]
    rows = np.empty((n_clusters, n_rows), dtype=bool)
    cols = np.empty((n_clusters, n_cols), dtype=bool)
    for c in range(n_clusters):
        for i in range(n_rows):
            rows[c, i] = row_labels[i] == c
        for j in range(n_cols):
            cols[c, j] = col_labels[j] == c
    return result, rows, cols

def make_checkerboard(shape, n_clusters, *, noise: float = 0.0, minval = 10, maxval = 100,
                      shuffle: bool = True, random_state = None):
    generator = check_random_state(random_state)
    nrc = 0; ncc = 0
    if isinstance(n_clusters, int):
        nrc = n_clusters; ncc = n_clusters
    else:
        nrc = int(n_clusters[0]); ncc = int(n_clusters[1])
    n_rows = int(shape[0]); n_cols = int(shape[1])
    row_labels = _labels_from_sizes(generator.multinomial(n_rows, np.full(nrc, 1.0 / float(nrc))))
    col_labels = _labels_from_sizes(generator.multinomial(n_cols, np.full(ncc, 1.0 / float(ncc))))
    result = np.zeros((n_rows, n_cols))
    for i in range(nrc):
        for j in range(ncc):
            v = float(generator.uniform(float(minval), float(maxval)))
            for a in range(n_rows):
                if row_labels[a] == i:
                    for b in range(n_cols):
                        if col_labels[b] == j:
                            result[a, b] += v
    if noise > 0:
        result += generator.normal(scale=noise, size=(n_rows, n_cols))
    if shuffle:
        g2 = check_random_state(random_state)
        ri = g2.permutation(n_rows)
        cj = g2.permutation(n_cols)
        result = np.ascontiguousarray(result[ri][:, cj])
        row_labels = row_labels[ri]
        col_labels = col_labels[cj]
    rows = np.empty((nrc * ncc, n_rows), dtype=bool)
    cols = np.empty((nrc * ncc, n_cols), dtype=bool)
    for i in range(nrc):
        for j in range(ncc):
            for a in range(n_rows):
                rows[i * ncc + j, a] = row_labels[a] == i
            for b in range(n_cols):
                cols[i * ncc + j, b] = col_labels[b] == j
    return result, rows, cols
