# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Parameter values / grids shared by the search estimators and
# validation_curve (sklearn.model_selection._search ParameterGrid /
# ParameterSampler). A parameter value is a small tagged scalar (_PV) so that
# grids may mix float / int / bool / str / None values; it compares equal to
# the plain Python value.

import numpy as np

class _PV:
    kind: int   # 0 float, 1 int, 2 bool, 3 str, 4 None
    f: float
    s: str

    def __init__(self, kind: int, f: float, s: str):
        self.kind = kind
        self.f = f
        self.s = s

    def __eq__(self, o) -> bool:
        return self.kind <= 1 and self.f == float(o)
    def __eq__(self, o: _PV) -> bool:
        return self.kind == o.kind and self.f == o.f and self.s == o.s
    def __eq__(self, o: str) -> bool:
        return self.kind == 3 and self.s == o
    def __eq__(self, o: bool) -> bool:
        return self.kind == 2 and self.f == (1.0 if o else 0.0)
    def __eq__(self, o: NoneType) -> bool:
        return self.kind == 4
    def __ne__(self, o) -> bool:
        return not (self == o)

    def __float__(self) -> float:
        return self.f
    def __int__(self) -> int:
        return int(self.f)
    def __bool__(self) -> bool:
        return self.f != 0.0 if self.kind <= 2 else (self.kind == 3 and self.s != "")

    def __repr__(self) -> str:
        if self.kind == 0:
            return repr(self.f)
        elif self.kind == 1:
            return str(int(self.f))
        elif self.kind == 2:
            return "True" if self.f != 0.0 else "False"
        elif self.kind == 3:
            return repr(self.s)
        return "None"
    def __str__(self) -> str:
        return self.s if self.kind == 3 else self.__repr__()

def _pv(x) -> _PV:
    if isinstance(x, _PV):
        return x
    elif isinstance(x, NoneType):
        return _PV(4, 0.0, "")
    elif isinstance(x, str):
        return _PV(3, 0.0, x)
    elif isinstance(x, bool):
        return _PV(2, 1.0 if x else 0.0, "")
    elif isinstance(x, int):
        return _PV(1, float(x), "")
    elif isinstance(x, Optional):
        if x is None:
            return _PV(4, 0.0, "")
        return _pv(x.__val__())
    else:
        return _PV(0, float(x), "")

def _pv_list(v) -> List[_PV]:
    out = List[_PV]()
    if isinstance(v, Union):
        for tag, T in static.vars_types(v, with_index=True):
            if Union._get_tag(v) == tag:
                if hasattr(T, "rvs"):
                    raise ValueError("Parameter grid values must be sequences, not distributions")
                else:
                    for x in Union._get_data(v, T):
                        out.append(_pv(x))
    elif hasattr(v, "rvs"):
        raise ValueError("Parameter grid values must be sequences, not distributions")
    else:
        for x in v:
            out.append(_pv(x))
    return out

# A normalized grid: list of sub-grids; each sub-grid = sorted (key, values)
_Grid = List[List[Tuple[str, List[_PV]]]]

def _norm_one(d) -> List[Tuple[str, List[_PV]]]:
    if isinstance(d, Union):
        for tag, T in static.vars_types(d, with_index=True):
            if Union._get_tag(d) == tag:
                return _norm_one(Union._get_data(d, T))
        return List[Tuple[str, List[_PV]]]()
    else:
        return _norm_one_impl(d)

def _norm_one_impl(d) -> List[Tuple[str, List[_PV]]]:
    out = List[Tuple[str, List[_PV]]]()
    for k in sorted(d.keys()):
        vals = _pv_list(d[k])
        if len(vals) == 0:
            raise ValueError(f"Parameter grid for parameter {k!r} need to be a non-empty sequence, got: []")
        out.append((k, vals))
    return out

def _norm_grid(param_grid) -> _Grid:
    g = _Grid()
    if isinstance(param_grid, List) or isinstance(param_grid, Tuple):
        for d in param_grid:
            g.append(_norm_one(d))
    else:
        g.append(_norm_one(param_grid))
    return g

def _grid_size(g: _Grid) -> int:
    tot = 0
    for sub in g:
        p = 1
        for kv in sub:
            p *= len(kv[1])
        tot += p
    return tot

def _grid_iter(g: _Grid) -> List[Dict[str, _PV]]:
    # itertools.product over sorted keys: last key varies fastest
    out = List[Dict[str, _PV]]()
    for sub in g:
        combos = [Dict[str, _PV]()]
        for kv in sub:
            nxt = List[Dict[str, _PV]]()
            for base in combos:
                for v in kv[1]:
                    c = base.copy()
                    c[kv[0]] = v
                    nxt.append(c)
            combos = nxt
        for c in combos:
            out.append(c)
    return out

def _grid_getitem(g: _Grid, ind: int) -> Dict[str, _PV]:
    # ParameterGrid.__getitem__: reverse-sorted keys, divmod decomposition
    for sub in g:
        if len(sub) == 0:
            if ind == 0:
                return Dict[str, _PV]()
            ind -= 1
            continue
        total = 1
        for kv in sub:
            total *= len(kv[1])
        if ind >= total:
            ind -= total
        else:
            out = Dict[str, _PV]()
            i = len(sub) - 1
            while i >= 0:
                n = len(sub[i][1])
                off = ind % n
                ind = ind // n
                out[sub[i][0]] = sub[i][1][off]
                i -= 1
            # present keys in sorted order like the dict sklearn builds
            res = Dict[str, _PV]()
            for kv in sub:
                res[kv[0]] = out[kv[0]]
            return res
    raise IndexError("ParameterGrid index out of range")

class ParameterGrid:
    _g: _Grid

    def __init__(self, param_grid):
        self._g = _norm_grid(param_grid)

    def __iter__(self):
        for c in _grid_iter(self._g):
            yield c

    def __len__(self) -> int:
        return _grid_size(self._g)

    def __getitem__(self, ind: int) -> Dict[str, _PV]:
        return _grid_getitem(self._g, ind)

def _sample_without_replacement(n_population: int, n_samples: int, rs):
    ratio = float(n_samples) / float(n_population) if n_population != 0 else 1.0
    if ratio > 0.01 and ratio < 0.99:
        return rs.permutation(n_population)[:n_samples]
    out = np.empty(n_samples, dtype=int)
    if ratio < 0.2:
        selected = Set[int]()
        for i in range(n_samples):
            j = int(rs.randint(n_population))
            while j in selected:
                j = int(rs.randint(n_population))
            selected.add(j)
            out[i] = j
    else:
        for i in range(n_samples):
            out[i] = i
        for i in range(n_samples, n_population):
            j = int(rs.randint(0, i + 1))
            if j < n_samples:
                out[j] = i
    return out

def _all_lists(d) -> bool:
    if isinstance(d, Union):
        for tag, T in static.vars_types(d, with_index=True):
            if Union._get_tag(d) == tag:
                return _all_lists(Union._get_data(d, T))
        return True
    else:
        return _all_lists_impl(d)

def _all_lists_impl(d) -> bool:
    ok = True
    for k, v in d.items():
        if isinstance(v, Union):
            for tag, T in static.vars_types(v, with_index=True):
                if Union._get_tag(v) == tag:
                    if hasattr(T, "rvs"):
                        ok = False
        elif hasattr(v, "rvs"):
            ok = False
    return ok

def _draw(v, rng) -> _PV:
    if isinstance(v, Union):
        for tag, T in static.vars_types(v, with_index=True):
            if Union._get_tag(v) == tag:
                return _draw(Union._get_data(v, T), rng)
        return _PV(4, 0.0, "")
    elif hasattr(v, "rvs"):
        return _pv(v.rvs(random_state=rng))
    else:
        return _pv(v[int(rng.randint(len(v)))])

def _sample_one(d, rng) -> Dict[str, _PV]:
    if isinstance(d, Union):
        for tag, T in static.vars_types(d, with_index=True):
            if Union._get_tag(d) == tag:
                return _sample_one(Union._get_data(d, T), rng)
        return Dict[str, _PV]()
    else:
        return _sample_one_impl(d, rng)

def _sample_one_impl(d, rng) -> Dict[str, _PV]:
    out = Dict[str, _PV]()
    for k in sorted(d.keys()):
        out[k] = _draw(d[k], rng)
    return out

def _sample_params(dists, n_iter: int, random_state: Optional[int]) -> List[Dict[str, _PV]]:
    """ParameterSampler(dists, n_iter, random_state) materialized."""
    rng = np.random.RandomState(random_state)
    all_lists = True
    if isinstance(dists, List) or isinstance(dists, Tuple):
        for d in dists:
            if not _all_lists(d):
                all_lists = False
    else:
        all_lists = _all_lists(dists)
    out = List[Dict[str, _PV]]()
    if all_lists:
        g = _Grid()
        if isinstance(dists, List) or isinstance(dists, Tuple):
            for d in dists:
                g.append(_norm_one(d))
        else:
            g.append(_norm_one(dists))
        gs = _grid_size(g)
        n = n_iter if n_iter <= gs else gs
        for i in _sample_without_replacement(gs, n, rng):
            out.append(_grid_getitem(g, int(i)))
    else:
        for _ in range(n_iter):
            if isinstance(dists, List) or isinstance(dists, Tuple):
                which = int(rng.randint(len(dists))) if len(dists) > 1 else 0
                out.append(_sample_one(dists[which], rng))
            else:
                out.append(_sample_one(dists, rng))
    return out

class ParameterSampler[D]:
    param_distributions: D
    n_iter: int
    random_state: Optional[int]

    def __init__(self, param_distributions: D, n_iter: int, *, random_state: Optional[int] = None):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def __iter__(self):
        for p in _sample_params(self.param_distributions, self.n_iter, self.random_state):
            yield p

    def __len__(self) -> int:
        if _all_lists(self.param_distributions):
            return min(self.n_iter, _grid_size(_norm_grid(self.param_distributions)))
        return self.n_iter

# ---------------------------------------------------------- set_params

def _set_param_rt(est, key: str, val: _PV) -> bool:
    """Set parameter `key` (runtime string, `step__param` nesting) on est."""
    if isinstance(est, str) or isinstance(est, NoneType) or isinstance(est, tuple):
        return False
    elif hasattr(est, "_set_param_rt"):
        return est._set_param_rt(key, val)
    elif hasattr(est, "_sk_clone") and hasattr(est, "steps"):
        i = key.find("__")
        if i < 0:
            return False
        head = key[:i]
        rest = key[i + 2:]
        ok = False
        for st in est.steps:
            if st[0] == head:
                ok = _set_param_rt(st[1], rest, val)
        return ok
    elif hasattr(est, "_sk_clone") and hasattr(est, "transformers"):
        # ColumnTransformer: (name, transformer, columns) entries
        i = key.find("__")
        if i < 0:
            return False
        head = key[:i]
        rest = key[i + 2:]
        ok = False
        for st in est.transformers:
            if st[0] == head:
                ok = _set_param_rt(st[1], rest, val)
        if not ok and head == "remainder":
            ok = _set_param_rt(est.remainder, rest, val)
        return ok
    elif hasattr(est, "_sk_clone") and hasattr(est, "transformer_list"):
        # FeatureUnion: (name, transformer) entries
        i = key.find("__")
        if i < 0:
            return False
        head = key[:i]
        rest = key[i + 2:]
        ok = False
        for st in est.transformer_list:
            if st[0] == head:
                ok = _set_param_rt(st[1], rest, val)
        return ok
    elif key.find("__") >= 0:
        # nested estimator field (GridSearchCV.estimator, TransformedTargetRegressor.regressor, ...)
        i = key.find("__")
        head = key[:i]
        rest = key[i + 2:]
        ok = False
        for k, v in static.vars(est):
            if k == head and (hasattr(v, "fit") or hasattr(v, "steps")):
                ok = _set_param_rt(v, rest, val)
        return ok
    else:
        ok = False
        for k, v in static.vars(est):
            if k == key:
                if isinstance(v, bool):
                    setattr(est, k, val.f != 0.0); ok = True
                elif isinstance(v, float):
                    setattr(est, k, val.f); ok = True
                elif isinstance(v, int):
                    setattr(est, k, int(val.f)); ok = True
                elif isinstance(v, str):
                    if val.kind == 3:
                        setattr(est, k, val.s); ok = True
                elif isinstance(v, Optional[float]):
                    setattr(est, k, Optional[float]() if val.kind == 4 else Optional[float](val.f)); ok = True
                elif isinstance(v, Optional[int]):
                    setattr(est, k, Optional[int]() if val.kind == 4 else Optional[int](int(val.f))); ok = True
                elif isinstance(v, np.ndarray[float, 1]):
                    # scalar-or-array parameters (Ridge alpha) store a 1-d array
                    setattr(est, k, np.array([val.f])); ok = True
                elif isinstance(v, Optional[str]):
                    setattr(est, k, Optional[str]() if val.kind == 4 else Optional[str](val.s)); ok = True
        if not ok and val.kind <= 1:
            # str|float parameters stored as (`key`: str, `key_value`: float)
            kv = key + "_value"
            for k, v in static.vars(est):
                if k == kv:
                    if isinstance(v, float):
                        setattr(est, k, val.f); ok = True
            if ok:
                for k, v in static.vars(est):
                    if k == key:
                        if isinstance(v, str):
                            setattr(est, k, "")
        return ok

def _apply_params(est, params: Dict[str, _PV]):
    for k, v in params.items():
        if not _set_param_rt(est, k, v):
            raise ValueError(f"Invalid parameter {k!r} for estimator. Valid parameters are the estimator's constructor arguments.")
