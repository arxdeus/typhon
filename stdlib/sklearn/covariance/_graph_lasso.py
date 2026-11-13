# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# GraphicalLasso / GraphicalLassoCV / graphical_lasso(_path) with the 'cd'
# mode (enet_coordinate_descent_gram from linear_model._cd_fast).
# GraphicalLassoCV: cv=None or an int (unshuffled KFold, as check_cv).

import numpy as np
import scipy.linalg as sl
from . import EmpiricalCovariance, empirical_covariance, log_likelihood

_EPS = 2.220446049250313e-16

def _objective(mle, precision_, alpha: float) -> float:
    p = precision_.shape[0]
    cost = -2.0 * log_likelihood(mle, precision_) + float(p) * np.log(2 * np.pi)
    cost += alpha * (float(np.abs(precision_).sum()) - float(np.abs(np.diag(precision_)).sum()))
    return cost

def _dual_gap(emp_cov, precision_, alpha: float) -> float:
    gap = float(np.sum(emp_cov * precision_))
    gap -= float(precision_.shape[0])
    gap += alpha * (float(np.abs(precision_).sum()) - float(np.abs(np.diag(precision_)).sum()))
    return gap

def _graphical_lasso(emp_cov: np.ndarray[float, 2], alpha: float, cov_init: Optional[np.ndarray[float, 2]],
                     tol: float, enet_tol: float, max_iter: int, eps: float):
    from ..linear_model._cd_fast import enet_coordinate_descent_gram
    from numpy.random.legacy import check_random_state
    p = emp_cov.shape[1]
    costs = List[Tuple[float, float]]()
    if alpha == 0:
        precision_ = np.linalg.inv(emp_cov)
        cost = -2.0 * log_likelihood(emp_cov, precision_) + float(p) * np.log(2 * np.pi)
        d_gap = float(np.sum(emp_cov * precision_)) - float(p)
        costs.append((cost, d_gap))
        return emp_cov.copy(), precision_, costs, 0
    covariance_ = emp_cov.copy() if cov_init is None else cov_init.__val__().copy()
    covariance_ *= 0.95
    for k in range(p):
        covariance_[k, k] = emp_cov[k, k]
    precision_ = sl.pinvh(covariance_)
    sub = np.ascontiguousarray(covariance_[1:, 1:])
    i = 0
    for it in range(max_iter):
        i = it
        for idx in range(p):
            others = [k for k in range(p) if k != idx]
            if idx > 0:
                di = idx - 1
                for t in range(p - 1):
                    sub[di, t] = covariance_[di, others[t]]
                for t in range(p - 1):
                    sub[t, di] = covariance_[others[t], di]
            else:
                for a in range(p - 1):
                    for b in range(p - 1):
                        sub[a, b] = covariance_[a + 1, b + 1]
            row = np.empty(p - 1)
            coefs = np.empty(p - 1)
            for t in range(p - 1):
                row[t] = emp_cov[idx, others[t]]
                coefs[t] = -(precision_[others[t], idx] / (precision_[idx, idx] + 1000 * eps))
            coefs, _, _, _ = enet_coordinate_descent_gram(coefs, alpha, 0.0, sub, row, row,
                                                          max(20, max_iter), enet_tol,
                                                          check_random_state(None), False, False, True)
            s = 0.0
            cvec = np.empty(p - 1)
            for t in range(p - 1):
                cvec[t] = covariance_[others[t], idx]
            precision_[idx, idx] = 1.0 / (covariance_[idx, idx] - float(np.dot(cvec, coefs)))
            for t in range(p - 1):
                precision_[others[t], idx] = -precision_[idx, idx] * coefs[t]
                precision_[idx, others[t]] = -precision_[idx, idx] * coefs[t]
            c2 = np.dot(sub, coefs)
            for t in range(p - 1):
                covariance_[idx, others[t]] = c2[t]
                covariance_[others[t], idx] = c2[t]
        if not np.isfinite(precision_.sum()):
            raise FloatingPointError("The system is too ill-conditioned for this solver. The system is too ill-conditioned for this solver")
        d_gap = _dual_gap(emp_cov, precision_, alpha)
        cost = _objective(emp_cov, precision_, alpha)
        costs.append((cost, d_gap))
        if abs(d_gap) < tol:
            break
        if not np.isfinite(cost) and it > 0:
            raise FloatingPointError("Non SPD result: the system is too ill-conditioned for this solver. The system is too ill-conditioned for this solver")
    return covariance_, precision_, costs, i + 1

def alpha_max(emp_cov) -> float:
    A = np.asarray(emp_cov, dtype=float).copy()
    for k in range(A.shape[0]):
        A[k, k] = 0.0
    return float(np.max(np.abs(A)))

def graphical_lasso(emp_cov, alpha: float, *, mode: str = "cd", tol: float = 1e-4,
                    enet_tol: float = 1e-4, max_iter: int = 100, verbose: bool = False,
                    return_costs: Literal[bool] = False, eps: float = _EPS,
                    return_n_iter: Literal[bool] = False):
    c, pr, costs, it = _graphical_lasso(np.asarray(emp_cov, dtype=float).copy(), alpha, None, tol, enet_tol, max_iter, eps)
    if return_costs and return_n_iter:
        return c, pr, costs, it
    elif return_costs:
        return c, pr, costs
    elif return_n_iter:
        return c, pr, it
    else:
        return c, pr

def graphical_lasso_path(X, alphas, cov_init = None, X_test = None, mode: str = "cd",
                         tol: float = 1e-4, enet_tol: float = 1e-4, max_iter: int = 100,
                         verbose: bool = False, eps: float = _EPS):
    """Returns (covariances, precisions, scores); scores is empty when X_test is None."""
    emp_cov = empirical_covariance(np.asarray(X, dtype=float))
    covariance_ = emp_cov.copy() if isinstance(cov_init, NoneType) else np.asarray(cov_init, dtype=float)
    covs = List[np.ndarray[float, 2]]()
    precs = List[np.ndarray[float, 2]]()
    scores = List[float]()
    has_test = not isinstance(X_test, NoneType)
    test_emp = emp_cov
    if not isinstance(X_test, NoneType):
        test_emp = empirical_covariance(np.asarray(X_test, dtype=float))
    for a in alphas:
        this_score = 0.0
        try:
            c, pr, _, _ = _graphical_lasso(emp_cov, float(a), covariance_, tol, enet_tol, max_iter, eps)
            covariance_ = c
            covs.append(c)
            precs.append(pr)
            if has_test:
                this_score = log_likelihood(test_emp, pr)
        except FloatingPointError:
            this_score = -np.inf
            covs.append(np.full(emp_cov.shape, np.nan))
            precs.append(np.full(emp_cov.shape, np.nan))
        if has_test:
            if not np.isfinite(this_score):
                this_score = -np.inf
            scores.append(this_score)
    return covs, precs, scores

class GraphicalLasso(EmpiricalCovariance):
    alpha: float
    mode: str
    covariance: str
    tol: float
    enet_tol: float
    max_iter: int
    eps: float
    costs_: List[Tuple[float, float]]
    n_iter_: int

    def __init__(self, alpha: float = 0.01, *, mode: str = "cd", covariance = None,
                 tol: float = 1e-4, enet_tol: float = 1e-4, max_iter: int = 100,
                 verbose: bool = False, eps: float = _EPS, assume_centered: bool = False):
        super().__init__(assume_centered=assume_centered)
        if mode != "cd":
            raise NotImplementedError("GraphicalLasso: only mode='cd' is supported")
        self.alpha = alpha
        self.mode = mode
        self.covariance = "" if isinstance(covariance, NoneType) else str(covariance)
        self.tol = tol
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.eps = eps
        self.costs_ = []
        self.n_iter_ = 0

    def fit(self, X, y = None):
        Xa = np.asarray(X, dtype=float)
        emp = np.empty((0, 0))
        if self.covariance == "precomputed":
            emp = Xa.copy()
            self.location_ = np.zeros(Xa.shape[1])
        else:
            emp = empirical_covariance(Xa, assume_centered=self.assume_centered)
            self.location_ = np.zeros(Xa.shape[1]) if self.assume_centered else Xa.mean(0)
        c, pr, costs, it = _graphical_lasso(emp, self.alpha, None, self.tol, self.enet_tol, self.max_iter, self.eps)
        self.covariance_ = c
        self.precision_ = pr
        self.costs_ = costs
        self.n_iter_ = it
        return self

class GraphicalLassoCV(EmpiricalCovariance):
    n_alphas: int
    alphas_list: List[float]
    n_refinements: int
    cv: int
    tol: float
    enet_tol: float
    max_iter: int
    eps: float
    alpha_: float
    cv_alphas_: List[float]
    mean_test_score_: List[float]
    costs_: List[Tuple[float, float]]
    n_iter_: int

    def __init__(self, *, alphas = 4, n_refinements: int = 4, cv = None, tol: float = 1e-4,
                 enet_tol: float = 1e-4, max_iter: int = 100, mode: str = "cd", n_jobs = None,
                 verbose: bool = False, eps: float = _EPS, assume_centered: bool = False):
        super().__init__(assume_centered=assume_centered)
        if isinstance(alphas, int):
            self.n_alphas = alphas
            self.alphas_list = []
        else:
            self.n_alphas = 0
            self.alphas_list = [float(a) for a in alphas]
        self.n_refinements = n_refinements
        self.cv = 5 if isinstance(cv, NoneType) else int(cv)
        self.tol = tol
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.eps = eps
        self.alpha_ = 0.0
        self.cv_alphas_ = []
        self.mean_test_score_ = []
        self.costs_ = []
        self.n_iter_ = 0

    def fit(self, X, y = None):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        n = Xa.shape[0]
        self.location_ = np.zeros(Xa.shape[1]) if self.assume_centered else Xa.mean(0)
        emp = empirical_covariance(Xa, assume_centered=self.assume_centered)
        k = self.cv
        # KFold(k) unshuffled folds
        folds = List[Tuple[List[int], List[int]]]()
        start = 0
        for f in range(k):
            sz = n // k + (1 if f < n % k else 0)
            te = list(range(start, start + sz))
            tr = [i for i in range(n) if i < start or i >= start + sz]
            folds.append((tr, te))
            start += sz
        explicit = self.n_alphas == 0
        alphas = List[float]()
        n_ref = 1
        if explicit:
            alphas = self.alphas_list.copy()
        else:
            n_ref = self.n_refinements
            a1 = alpha_max(emp)
            a0 = 0.01 * a1
            ls = np.logspace(np.log10(a0), np.log10(a1), self.n_alphas)[::-1]
            alphas = [float(v) for v in ls]
        path = List[Tuple[float, List[float]]]()
        best_index = 0
        for _ in range(n_ref):
            per = [List[float]() for _ in range(len(alphas))]
            for tr, te in folds:
                _, _, sc = graphical_lasso_path(Xa[np.array(tr)], alphas, X_test=Xa[np.array(te)],
                                                tol=self.tol, enet_tol=self.enet_tol,
                                                max_iter=int(0.1 * self.max_iter), eps=self.eps)
                for a in range(len(alphas)):
                    per[a].append(sc[a])
            for a in range(len(alphas)):
                path.append((alphas[a], per[a]))
            # sorted(..., key=alpha, reverse=True): stable
            path = sorted(path, key=lambda t: -t[0])
            best_score = -np.inf
            last_finite_idx = 0
            for index in range(len(path)):
                this_score = float(np.mean(np.array(path[index][1])))
                if this_score >= 0.1 / _EPS:
                    this_score = np.nan
                if np.isfinite(this_score):
                    last_finite_idx = index
                if this_score >= best_score:
                    best_score = this_score
                    best_index = index
            a1 = 0.0; a0 = 0.0
            if best_index == 0:
                a1 = path[0][0]; a0 = path[1][0]
            elif best_index == last_finite_idx and not best_index == len(path) - 1:
                a1 = path[best_index][0]; a0 = path[best_index + 1][0]
            elif best_index == len(path) - 1:
                a1 = path[best_index][0]; a0 = 0.01 * path[best_index][0]
            else:
                a1 = path[best_index - 1][0]; a0 = path[best_index + 1][0]
            if not explicit:
                ls = np.logspace(np.log10(a1), np.log10(a0), self.n_alphas + 2)
                alphas = [float(v) for v in ls[1:-1]]
        self.cv_alphas_ = [t[0] for t in path] + [0.0]
        self.mean_test_score_ = [float(np.mean(np.array(t[1]))) for t in path]
        self.alpha_ = path[best_index][0]
        c, pr, costs, it = _graphical_lasso(emp, self.alpha_, None, self.tol, self.enet_tol, self.max_iter, self.eps)
        self.covariance_ = c
        self.precision_ = pr
        self.costs_ = costs
        self.n_iter_ = it
        return self
