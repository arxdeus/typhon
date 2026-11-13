# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# SelfTrainingClassifier over a statically typed base estimator E with
# predict_proba. Integer labels, -1 marks unlabeled samples.

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ..base import clone, _accuracy

class SelfTrainingClassifier[E]:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    estimator: E
    threshold: float
    criterion: str
    k_best: int
    max_iter: Optional[int]
    estimator_: E
    transduction_: np.ndarray[int, 1]
    labeled_iter_: np.ndarray[int, 1]
    n_iter_: int
    termination_condition_: str
    classes_: np.ndarray[int, 1]

    def __init__(self, estimator: E, threshold: float = 0.75, criterion: str = "threshold",
                 k_best: int = 10, max_iter: Optional[int] = 10, verbose: bool = False):
        if criterion != "threshold" and criterion != "k_best":
            raise ValueError("criterion must be 'threshold' or 'k_best'")
        self.estimator = estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter
        self.estimator_ = estimator
        self.transduction_ = np.empty(0, dtype=int)
        self.labeled_iter_ = np.empty(0, dtype=int)
        self.n_iter_ = 0
        self.termination_condition_ = ""
        self.classes_ = np.empty(0, dtype=int)

    def fit(self, X, y):
        Xa = np.ascontiguousarray(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=int)
        n = ya.shape[0]
        self.estimator_ = clone(self.estimator)
        has = ya != -1
        self.transduction_ = ya.copy()
        self.labeled_iter_ = np.full(n, -1, dtype=int)
        for i in range(n):
            if has[i]:
                self.labeled_iter_[i] = 0
        self.n_iter_ = 0
        mi = -1 if self.max_iter is None else self.max_iter.__val__()
        while not np.all(has) and (mi < 0 or self.n_iter_ < mi):
            self.n_iter_ += 1
            self.estimator_.fit(Xa[has], self.transduction_[has])
            unl = np.nonzero(~has)[0]
            prob = self.estimator_.predict_proba(Xa[unl])
            ec = self.estimator_.classes_
            am = np.argmax(prob, axis=1)
            mp = np.max(prob, axis=1)
            sel = List[int]()
            if self.criterion == "threshold":
                for t in range(mp.shape[0]):
                    if mp[t] > self.threshold:
                        sel.append(t)
            else:
                nts = min(self.k_best, mp.shape[0])
                if nts == mp.shape[0]:
                    sel = list(range(nts))
                else:
                    part = np.argpartition(-mp, nts)[:nts]
                    sel = [int(v) for v in part]
            for t in sel:
                j = int(unl[t])
                self.transduction_[j] = ec[am[t]]
                has[j] = True
                self.labeled_iter_[j] = self.n_iter_
            if len(sel) == 0:
                self.termination_condition_ = "no_change"
                break
        if mi >= 0 and self.n_iter_ == mi:
            self.termination_condition_ = "max_iter"
        if np.all(has):
            self.termination_condition_ = "all_labeled"
        self.estimator_.fit(Xa[has], self.transduction_[has])
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    def score(self, X, y, sample_weight = None) -> float:
        return _accuracy(y, self.predict(X))
