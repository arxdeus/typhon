# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# DecisionTreeClassifier / DecisionTreeRegressor / ExtraTree* (sklearn/tree/_classes.py)
# Single-output targets; classification labels are integers.
# Inputs are converted to float32 exactly like sklearn (validate_data dtype=np.float32),
# so thresholds, impurities and predictions match sklearn bit for bit.

import math
import numpy as np
from ._criterion import Criterion, GINI, ENTROPY, MSE, MAE, POISSON
from ._splitter import Splitter
from ._tree import (Tree, build_depth_first, build_best_first, build_pruned_tree_ccp,
                    ccp_pruning_path, TREE_LEAF, TREE_UNDEFINED)
from ._utils import RAND_R_MAX

def _check_rs(random_state: Optional[int]):
    # check_random_state; None seeds from the clock (a function returning
    # both RandomState() and RandomState(int) miscompiles, see report)
    seed = 0
    if random_state is None:
        import time
        seed = int(time.time() * 1000000.0) % 4294967296
    else:
        seed = random_state
    return np.random.RandomState(seed)

def _to_f32_colmajor(X) -> Tuple[List[float32], int, int]:
    Xa = np.asarray(X, dtype=float)
    n = Xa.shape[0]
    p = Xa.shape[1]
    out = [float32(0.0)] * (n * p)
    for j in range(p):
        for i in range(n):
            out[j * n + i] = float32(Xa[i, j])
    return (out, n, p)

def _missing_mask(Xf: List[float32], n: int, p: int) -> List[bool]:
    any_nan = False
    m = [False] * p
    for j in range(p):
        for i in range(n):
            if math.isnan(float(Xf[j * n + i])):
                m[j] = True
                any_nan = True
                break
    return m if any_nan else List[bool]()

def _encode_classes(y) -> Tuple[np.ndarray[int, 1], List[float]]:
    ya = np.asarray(y)
    n = ya.shape[0]
    vals = sorted(set([int(ya[i]) for i in range(n)]))
    classes = np.array(vals)
    lut = Dict[int, int]()
    for k in range(len(vals)):
        lut[vals[k]] = k
    enc = [float(lut[int(ya[i])]) for i in range(n)]
    return (classes, enc)

def _balanced_weight(yenc: List[float], n_classes: int) -> List[float]:
    # compute_class_weight("balanced"): n_samples / (n_classes * bincount(y))
    n = len(yenc)
    cnt = [0] * n_classes
    for v in yenc:
        cnt[int(v)] += 1
    cw = [float(n) / (float(n_classes) * float(cnt[k])) if cnt[k] > 0 else 0.0
          for k in range(n_classes)]
    return [cw[int(v)] for v in yenc]

class BaseDecisionTree:
    criterion: str
    splitter: str
    max_depth: Optional[int]
    min_samples_split: float
    _mss_int: bool
    min_samples_leaf: float
    _msl_int: bool
    min_weight_fraction_leaf: float
    max_features: str          # "none", "sqrt", "log2", "int", "float"
    _mf_value: float
    random_state: Optional[int]
    max_leaf_nodes: Optional[int]
    min_impurity_decrease: float
    class_weight: str          # "" or "balanced"
    _class_weight_dict: Dict[int, float]
    ccp_alpha: float
    monotonic_cst: List[int]
    _is_clf: bool
    # fitted
    tree_: Tree
    n_features_in_: int
    n_outputs_: int
    max_features_: int
    classes_: np.ndarray[int, 1]
    n_classes_: int
    _fitted: bool

    def __init__(self, is_clf: bool, criterion: str, splitter: str, max_depth: Optional[int],
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf: float,
                 max_features, random_state: Optional[int], max_leaf_nodes: Optional[int],
                 min_impurity_decrease: float, class_weight, ccp_alpha: float, monotonic_cst):
        self._is_clf = is_clf
        if criterion == "friedman_mse":
            criterion = "squared_error"
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = float(min_samples_split)
        self._mss_int = isinstance(min_samples_split, int)
        self.min_samples_leaf = float(min_samples_leaf)
        self._msl_int = isinstance(min_samples_leaf, int)
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self._mf_value = 0.0
        if max_features is None:
            self.max_features = "none"
        elif isinstance(max_features, str):
            self.max_features = max_features
        elif isinstance(max_features, int):
            self.max_features = "int"
            self._mf_value = float(max_features)
        else:
            self.max_features = "float"
            self._mf_value = float(max_features)
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self._class_weight_dict = Dict[int, float]()
        if class_weight is None:
            self.class_weight = ""
        elif isinstance(class_weight, str):
            self.class_weight = class_weight
        else:
            self.class_weight = "dict"
            for k, v in class_weight.items():
                self._class_weight_dict[int(k)] = float(v)
        self.ccp_alpha = ccp_alpha
        if monotonic_cst is None:
            self.monotonic_cst = List[int]()
        else:
            self.monotonic_cst = [int(v) for v in monotonic_cst]
        self.n_features_in_ = 0
        self.n_outputs_ = 1
        self.max_features_ = 0
        self.classes_ = np.empty(0, dtype=int)
        self.n_classes_ = 0
        self.tree_ = Tree(0, [1], 1)
        self._fitted = False

    # ------------------------------------------------------------- fit

    def _criterion_kind(self) -> int:
        c = self.criterion
        if self._is_clf:
            if c == "gini":
                return GINI
            elif c == "entropy" or c == "log_loss":
                return ENTROPY
            raise ValueError(f"unknown criterion '{c}'")
        if c == "squared_error":
            return MSE
        elif c == "absolute_error":
            return MAE
        elif c == "poisson":
            return POISSON
        raise ValueError(f"unknown criterion '{c}'")

    def _resolve_max_features(self, p: int) -> int:
        mf = self.max_features
        if mf == "sqrt":
            return max(1, int(math.sqrt(float(p))))
        elif mf == "log2":
            return max(1, int(math.log2(float(p))))
        elif mf == "none":
            return p
        elif mf == "int":
            return int(self._mf_value)
        else:
            return max(1, int(self._mf_value * float(p))) if self._mf_value > 0.0 else 0

    def _fit_encoded(self, Xf: List[float32], n: int, p: int, yv: List[float],
                     n_classes: int, sw: List[float], has_w: bool, seed_rs,
                     missing: List[bool]):
        """Core of BaseDecisionTree._fit after validation: yv holds encoded
        classes (classifier) or targets (regressor), sw the sample weights."""
        self.n_features_in_ = p
        self.n_outputs_ = 1
        if self.criterion == "poisson":
            s = 0.0
            for v in yv:
                if v < 0.0:
                    raise ValueError("Some value(s) of y are negative which is not allowed for Poisson regression.")
                s += v
            if s <= 0.0:
                raise ValueError("Sum of y is not positive which is necessary for Poisson regression.")
        max_depth = 2147483647 if self.max_depth is None else self.max_depth
        msl = int(self.min_samples_leaf) if self._msl_int else int(math.ceil(self.min_samples_leaf * float(n)))
        mss = 0
        if self._mss_int:
            mss = int(self.min_samples_split)
        else:
            mss = max(2, int(math.ceil(self.min_samples_split * float(n))))
        mss = max(mss, 2 * msl)
        self.max_features_ = self._resolve_max_features(p)
        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes
        min_weight_leaf = 0.0
        if not has_w:
            min_weight_leaf = self.min_weight_fraction_leaf * float(n)
        else:
            tw = 0.0
            for w in sw:
                tw += w
            min_weight_leaf = self.min_weight_fraction_leaf * tw
        ncl = [n_classes] if self._is_clf else [1]
        crit = Criterion(self._criterion_kind(), 1, ncl, n)
        mono = List[int]()
        if len(self.monotonic_cst) > 0:
            if len(self.monotonic_cst) != p:
                raise ValueError("monotonic_cst has the wrong number of features")
            for v in self.monotonic_cst:
                if v != -1 and v != 0 and v != 1:
                    raise ValueError("monotonic_cst must be None or an array-like of -1, 0 or 1")
            if self._is_clf:
                if n_classes > 2:
                    raise ValueError("Monotonicity constraints are not supported with multiclass classification")
                mono = [-v for v in self.monotonic_cst]
            else:
                mono = list(self.monotonic_cst)
        seed = seed_rs.randint(0, RAND_R_MAX)
        splitter = Splitter(crit, self.splitter == "random", self.max_features_, msl,
                            min_weight_leaf, seed, mono)
        tree = Tree(p, ncl, 1)
        splitter.init(Xf, n, p, yv, sw, has_w, missing)
        if max_leaf_nodes < 0:
            build_depth_first(tree, splitter, mss, msl, min_weight_leaf, max_depth,
                              self.min_impurity_decrease)
        else:
            build_best_first(tree, splitter, mss, msl, min_weight_leaf, max_depth,
                             max_leaf_nodes, self.min_impurity_decrease)
        if self.ccp_alpha != 0.0:
            tree = build_pruned_tree_ccp(tree, self.ccp_alpha)
        self.tree_ = tree
        self._fitted = True

    def _prepare(self, X, y, sample_weight) -> Tuple[List[float32], int, int, List[float], int, List[float], bool]:
        Xf, n, p = _to_f32_colmajor(X)
        ya = np.asarray(y)
        if ya.shape[0] != n:
            raise ValueError(f"Number of labels={ya.shape[0]} does not match number of samples={n}")
        yv = List[float]()
        nc = 1
        if self._is_clf:
            classes, enc = _encode_classes(ya)
            self.classes_ = classes
            nc = classes.shape[0]
            self.n_classes_ = nc
            yv = enc
        else:
            yv = [float(ya[i]) for i in range(n)]
        has_w = sample_weight is not None
        sw = [1.0] * n
        if sample_weight is not None:
            swa = np.asarray(sample_weight, dtype=float)
            for i in range(n):
                sw[i] = float(swa[i])
        if self._is_clf and self.class_weight != "":
            cw = List[float]()
            if self.class_weight == "balanced":
                cw = _balanced_weight(yv, nc)
            else:
                cw = [self._class_weight_dict.get(int(self.classes_[int(v)]), 1.0) for v in yv]
            for i in range(n):
                sw[i] = sw[i] * cw[i] if has_w else cw[i]
            has_w = True
        return (Xf, n, p, yv, nc, sw, has_w)

    def fit(self, X, y, sample_weight = None, check_input: bool = True):
        rs = _check_rs(self.random_state)
        Xf, n, p, yv, nc, sw, has_w = self._prepare(X, y, sample_weight)
        self._fit_encoded(Xf, n, p, yv, nc, sw, has_w, rs, _missing_mask(Xf, n, p))
        return self

    # --------------------------------------------------------- predict

    def _check_X(self, X):
        if not self._fitted:
            raise ValueError("This estimator is not fitted yet. Call 'fit' first.")
        Xa = np.asarray(X, dtype=float)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but the tree is expecting {self.n_features_in_} features as input.")
        return Xa

    def apply(self, X, check_input: bool = True):
        return self.tree_.apply(self._check_X(X))

    def decision_path(self, X, check_input: bool = True):
        """Dense (n_samples, n_nodes) int indicator matrix (sklearn returns CSR)."""
        return self.tree_.decision_path(self._check_X(X))

    def get_depth(self) -> int:
        return self.tree_.max_depth

    def get_n_leaves(self) -> int:
        return self.tree_.n_leaves

    @property
    def feature_importances_(self):
        return self.tree_.compute_feature_importances()

    def cost_complexity_pruning_path(self, X, y, sample_weight = None):
        """-> (ccp_alphas, impurities)"""
        saved = self.ccp_alpha
        self.ccp_alpha = 0.0
        self.fit(X, y, sample_weight)
        self.ccp_alpha = saved
        t = self.tree_
        self._fitted = False
        return ccp_pruning_path(t)

class _ClfTree(BaseDecisionTree):
    def predict_proba(self, X, check_input: bool = True):
        Xa = self._check_X(X)
        return self.tree_.predict(Xa)[:, :self.n_classes_]

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X, check_input: bool = True):
        proba = self.predict_proba(X)
        n = proba.shape[0]
        out = np.empty(n, dtype=int)
        for i in range(n):
            best = 0
            for c in range(1, self.n_classes_):
                if proba[i, c] > proba[i, best]:
                    best = c
            out[i] = self.classes_[best]
        return out

    def score(self, X, y, sample_weight = None) -> float:
        from ..base import _accuracy
        return _accuracy(y, self.predict(X))

class _RegTree(BaseDecisionTree):
    def predict(self, X, check_input: bool = True):
        Xa = self._check_X(X)
        return self.tree_.predict(Xa)[:, 0]

    def score(self, X, y, sample_weight = None) -> float:
        return _r2(y, self.predict(X))

def _r2(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = yt.shape[0]
    m = float(np.mean(yt))
    num = 0.0
    den = 0.0
    for i in range(n):
        num += (yt[i] - yp[i]) ** 2
        den += (yt[i] - m) ** 2
    if den == 0.0:
        return 1.0 if num == 0.0 else 0.0
    return 1.0 - num / den

class DecisionTreeClassifier(_ClfTree):
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, *, criterion: str = "gini", splitter: str = "best",
                 max_depth: Optional[int] = None, min_samples_split = 2, min_samples_leaf = 1,
                 min_weight_fraction_leaf: float = 0.0, max_features = None,
                 random_state: Optional[int] = None, max_leaf_nodes: Optional[int] = None,
                 min_impurity_decrease: float = 0.0, class_weight = None,
                 ccp_alpha: float = 0.0, monotonic_cst = None):
        BaseDecisionTree.__init__(self, True, criterion, splitter, max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf, max_features,
                         random_state, max_leaf_nodes, min_impurity_decrease,
                         class_weight, ccp_alpha, monotonic_cst)


class DecisionTreeRegressor(_RegTree):
    _estimator_type: Literal[str] = "regressor"

    def __init__(self, *, criterion: str = "squared_error", splitter: str = "best",
                 max_depth: Optional[int] = None, min_samples_split = 2, min_samples_leaf = 1,
                 min_weight_fraction_leaf: float = 0.0, max_features = None,
                 random_state: Optional[int] = None, max_leaf_nodes: Optional[int] = None,
                 min_impurity_decrease: float = 0.0, ccp_alpha: float = 0.0,
                 monotonic_cst = None):
        BaseDecisionTree.__init__(self, False, criterion, splitter, max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf, max_features,
                         random_state, max_leaf_nodes, min_impurity_decrease,
                         None, ccp_alpha, monotonic_cst)


class ExtraTreeClassifier(_ClfTree):
    _estimator_type: Literal[str] = "classifier"

    def __init__(self, *, criterion: str = "gini", splitter: str = "random",
                 max_depth: Optional[int] = None, min_samples_split = 2, min_samples_leaf = 1,
                 min_weight_fraction_leaf: float = 0.0, max_features = "sqrt",
                 random_state: Optional[int] = None, max_leaf_nodes: Optional[int] = None,
                 min_impurity_decrease: float = 0.0, class_weight = None,
                 ccp_alpha: float = 0.0, monotonic_cst = None):
        BaseDecisionTree.__init__(self, True, criterion, splitter, max_depth, min_samples_split,
                                  min_samples_leaf, min_weight_fraction_leaf, max_features,
                                  random_state, max_leaf_nodes, min_impurity_decrease,
                                  class_weight, ccp_alpha, monotonic_cst)

class ExtraTreeRegressor(_RegTree):
    _estimator_type: Literal[str] = "regressor"

    def __init__(self, *, criterion: str = "squared_error", splitter: str = "random",
                 max_depth: Optional[int] = None, min_samples_split = 2, min_samples_leaf = 1,
                 min_weight_fraction_leaf: float = 0.0, max_features = 1.0,
                 random_state: Optional[int] = None, min_impurity_decrease: float = 0.0,
                 max_leaf_nodes: Optional[int] = None, ccp_alpha: float = 0.0,
                 monotonic_cst = None):
        BaseDecisionTree.__init__(self, False, criterion, splitter, max_depth, min_samples_split,
                                  min_samples_leaf, min_weight_fraction_leaf, max_features,
                                  random_state, max_leaf_nodes, min_impurity_decrease,
                                  None, ccp_alpha, monotonic_cst)
