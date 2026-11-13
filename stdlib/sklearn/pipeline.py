# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from .base import estimator_repr as _estimator_repr
import numpy as np
from .base import clone

def _name_of(est: T, T: type) -> str:
    # generic estimators (e.g. LogisticRegression[CW]) report "Name[...]"
    nm = T.__name__
    b = nm.find("[")
    if b >= 0:
        nm = nm[:b]
    return nm.lower()

def _clone_steps(steps):
    if static.len(steps) == 0:
        return ()
    else:
        return ((steps[0][0], clone(steps[0][1])),) + _clone_steps(steps[1:])

def _is_pass(est) -> Literal[bool]:
    return isinstance(est, str) or isinstance(est, NoneType)

def _ft1(est, X, y):
    if _is_pass(est):
        return X
    elif hasattr(est, "fit_transform"):
        return est.fit_transform(X, y)
    else:
        return est.fit(X, y).transform(X)

def _t1(est, X):
    if _is_pass(est):
        return X
    else:
        return est.transform(X)

def _fit_transform_head(steps, X, y):
    # fit_transform every step except the last; returns the transformed X
    if static.len(steps) <= 1:
        return X
    else:
        return _fit_transform_head(steps[1:], _ft1(steps[0][1], X, y), y)

def _drop_last(steps):
    if static.len(steps) <= 1:
        return ()
    else:
        return (steps[0],) + _drop_last(steps[1:])

def _transform_head(steps, X):
    if static.len(steps) <= 1:
        return X
    else:
        return _transform_head(steps[1:], _t1(steps[0][1], X))

def _transform_all(steps, X):
    if static.len(steps) == 0:
        return X
    else:
        return _transform_all(steps[1:], _t1(steps[0][1], X))

def _inverse_all(steps, X):
    if static.len(steps) == 0:
        return X
    else:
        last = steps[-1][1]
        if _is_pass(last):
            return _inverse_all(steps[:-1], X)
        else:
            return _inverse_all(steps[:-1], last.inverse_transform(X))

def _step_by_name(steps, name: Literal[str]):
    if static.len(steps) == 0:
        compile_error("Pipeline has no step with this name")
    elif steps[0][0] == name:
        return steps[0][1]
    else:
        return _step_by_name(steps[1:], name)

def _set_kw(est, keys, vals, i: int):
    from .model_selection._params import _set_param_rt, _pv
    if static.len(vals) > 0:
        if not _set_param_rt(est, keys[i], _pv(vals[0])):
            raise ValueError(f"Invalid parameter {keys[i]!r} for estimator Pipeline.")
        _set_kw(est, keys, vals[1:], i + 1)

def _nested_params(out: Dict[str, str], steps, deep: bool):
    from .base import get_params as _gp, _param_repr
    for st in steps:
        nm = st[0]
        e = st[1]
        if isinstance(e, str):
            out[nm] = e
        else:
            out[nm] = _param_repr(e)
            if deep and hasattr(e, "fit"):
                if hasattr(e, "steps"):
                    sub = Dict[str, str]()
                    _nested_params(sub, e.steps, deep)
                    for k, v in sub.items():
                        out[nm + "__" + k] = v
                else:
                    for k, v in _gp(e, False).items():
                        out[nm + "__" + k] = v

class _NamedSteps[S]:
    """Pipeline.named_steps. Step names are runtime strings, so lookup by
    name (ns["lr"]) is not available; use pipe[i] / keys()."""
    steps: S
    def __init__(self, steps: S):
        self.steps = steps
    def __getitem__(self, i: Literal[int]):
        return self.steps[i][1]
    def keys(self) -> List[str]:
        out = List[str]()
        for st in self.steps:
            out.append(st[0])
        return out
    def __len__(self) -> int:
        return static.len(self.steps)

class Pipeline[S]:
    def __repr__(self) -> str:
        return _estimator_repr(self, False)

    """Pipeline over a STATIC tuple of (name, estimator) pairs.

    Typhon resolves each step's type at compile time, so `steps` is a tuple
    (a Python list literal of pairs also works when all steps share a type)."""
    steps: S
    memory: Optional[str]  # not used (no transformer caching)
    verbose: bool

    def __init__(self, steps: S, *, memory: Optional[str] = None, verbose: bool = False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose

    def _sk_clone(self):
        return Pipeline(_clone_steps(self.steps), memory=self.memory, verbose=self.verbose)

    @property
    def named_steps(self):
        return _NamedSteps(self.steps)

    def get_params_names(self) -> List[str]:
        out = List[str]()
        for st in self.steps:
            out.append(st[0])
        return out

    def get_params(self, deep: bool = True) -> Dict[str, str]:
        """memory/verbose, each step name and (deep) step__param entries (reprs)."""
        out = Dict[str, str]()
        out["memory"] = "None" if self.memory is None else repr(self.memory)
        out["verbose"] = "True" if self.verbose else "False"
        _nested_params(out, self.steps, deep)
        return out

    def set_params(self, **params):
        """step__param=value (static names); a value for a bare step name
        is not supported (replacing a step changes the pipeline type)."""
        from .model_selection._params import _set_param_rt, _pv
        keys = params.__keys__()
        _set_kw(self, keys, params.args, 0)
        return self

    def slice(self, start: Literal[int], stop: Literal[int]):
        """pipe[start:stop] (static bounds)."""
        return Pipeline(self.steps[start:stop], memory=self.memory, verbose=self.verbose)

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    def __len__(self) -> int:
        return static.len(self.steps)

    def __getitem__(self, i: Literal[int]):
        return self.steps[i][1]

    def __static_slice__(self, has_start: Literal[bool], start: Literal[int],
                         has_stop: Literal[bool], stop: Literal[int],
                         has_step: Literal[bool], step: Literal[int]):
        """pipe[a:b] -> Pipeline of the selected steps (sklearn requires step 1)."""
        if has_step:
            if step != 1:
                compile_error("Pipeline slicing only supports a step of 1")
        n: Literal[int] = static.len(self.steps)
        a: Literal[int] = (start + n if start < 0 else start) if has_start else 0
        b: Literal[int] = (stop + n if stop < 0 else stop) if has_stop else n
        return Pipeline(self.steps[a:b], memory=self.memory, verbose=self.verbose)

    def head(self):
        """Equivalent of pipeline[:-1] (slices are not static in Typhon)."""
        return Pipeline(_drop_last(self.steps), memory=self.memory, verbose=self.verbose)

    def fit(self, X, y = None):
        Xt = _fit_transform_head(self.steps, X, y)
        if not _is_pass(self.steps[-1][1]):
            self.steps[-1][1].fit(Xt, y)
        return self

    def fit_transform(self, X, y = None):
        Xt = _fit_transform_head(self.steps, X, y)
        return _ft1(self.steps[-1][1], Xt, y)

    def inverse_transform(self, X):
        return _inverse_all(self.steps, X)

    def predict_log_proba(self, X):
        return self.steps[-1][1].predict_log_proba(_transform_head(self.steps, X))

    def score_samples(self, X):
        return self.steps[-1][1].score_samples(_transform_head(self.steps, X))

    @property
    def _estimator_type(self) -> str:
        last = self.steps[-1][1]
        if hasattr(last, "_estimator_type"):
            return last._estimator_type
        return ""

    def fit_predict(self, X, y = None):
        return self.fit(X, y).predict(X)

    def predict(self, X):
        return self.steps[-1][1].predict(_transform_head(self.steps, X))

    def predict_proba(self, X):
        return self.steps[-1][1].predict_proba(_transform_head(self.steps, X))

    def decision_function(self, X):
        return self.steps[-1][1].decision_function(_transform_head(self.steps, X))

    def transform(self, X):
        return _transform_all(self.steps, X)

    def score(self, X, y, sample_weight = None):
        return self.steps[-1][1].score(_transform_head(self.steps, X), y)

    @property
    def classes_(self):
        return self.steps[-1][1].classes_

def _named(ests):
    if static.len(ests) == 0:
        return ()
    else:
        return ((_name_of(ests[0]), ests[0]),) + _named(ests[1:])

def make_pipeline(*steps, memory: Optional[str] = None, verbose: bool = False):
    return Pipeline(_named(steps), memory=memory, verbose=verbose)

# ----------------------------------------------------------- FeatureUnion

def _hstack2(a, b):
    import numpy as np
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if static.len(A.shape) == 1:
        return _hstack2(A.reshape((A.shape[0], 1)), B)
    elif static.len(B.shape) == 1:
        return _hstack2(A, B.reshape((B.shape[0], 1)))
    else:
        return np.hstack((A, B))

def _union_list_ft(tl, X, y, w):
    import numpy as np
    out = np.empty((np.asarray(X).shape[0], 0))
    for item in tl:
        nm, est = item
        if isinstance(est, str):
            if est == "drop":
                continue
            out = _hstack2(out, np.asarray(X, dtype=float) * _w(w, nm))
        else:
            out = _hstack2(out, np.asarray(_ft1(est, X, y), dtype=float) * _w(w, nm))
    return out

def _union_list_t(tl, X, w):
    import numpy as np
    out = np.empty((np.asarray(X).shape[0], 0))
    for item in tl:
        nm, est = item
        if isinstance(est, str):
            if est == "drop":
                continue
            out = _hstack2(out, np.asarray(X, dtype=float) * _w(w, nm))
        else:
            out = _hstack2(out, np.asarray(est.transform(X), dtype=float) * _w(w, nm))
    return out

def _union_ft(tl, X, y, w):
    import numpy as np
    if isinstance(tl, List):
        return _union_list_ft(tl, X, y, w)
    elif static.len(tl) == 0:
        return np.empty((np.asarray(X).shape[0], 0))
    else:
        est = tl[0][1]
        rest = _union_ft(tl[1:], X, y, w)
        if isinstance(est, str):
            if est == "drop":
                return rest
            return _hstack2(np.asarray(X, dtype=float) * _w(w, tl[0][0]), rest)
        else:
            return _hstack2(np.asarray(_ft1(est, X, y), dtype=float) * _w(w, tl[0][0]), rest)

def _union_t(tl, X, w):
    import numpy as np
    if isinstance(tl, List):
        return _union_list_t(tl, X, w)
    elif static.len(tl) == 0:
        return np.empty((np.asarray(X).shape[0], 0))
    else:
        est = tl[0][1]
        rest = _union_t(tl[1:], X, w)
        if isinstance(est, str):
            if est == "drop":
                return rest
            return _hstack2(np.asarray(X, dtype=float) * _w(w, tl[0][0]), rest)
        else:
            return _hstack2(np.asarray(est.transform(X), dtype=float) * _w(w, tl[0][0]), rest)

def _w(w, name: str) -> float:
    if w is None:
        return 1.0
    elif name in w:
        return float(w[name])
    else:
        return 1.0

class FeatureUnion[T, W = NoneType]:
    def __repr__(self) -> str:
        return _estimator_repr(self, False)

    """Concatenates transformer outputs column-wise (dense float output).
    transformer_list: static tuple of (name, transformer|'drop'|'passthrough')."""
    transformer_list: T
    transformer_weights: W
    n_jobs: Optional[int]
    verbose: bool

    def __init__(self, transformer_list: T, *, n_jobs: Optional[int] = None,
                 transformer_weights: W = None, verbose: bool = False,
                 verbose_feature_names_out: bool = True):
        self.transformer_list = transformer_list
        self.transformer_weights = transformer_weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> Dict[str, str]:
        out = Dict[str, str]()
        out["n_jobs"] = "None" if self.n_jobs is None else str(self.n_jobs)
        out["verbose"] = "True" if self.verbose else "False"
        _nested_params(out, self.transformer_list, deep)
        return out

    def set_params(self, **params):
        keys = params.__keys__()
        _set_kw(self, keys, params.args, 0)
        return self

    def _sk_clone(self):
        return FeatureUnion(_clone_steps(self.transformer_list), n_jobs=self.n_jobs,
                            transformer_weights=self.transformer_weights, verbose=self.verbose)

    @property
    def named_transformers(self):
        return _NamedSteps(self.transformer_list)

    def fit(self, X, y = None):
        _union_ft(self.transformer_list, X, y, self.transformer_weights)
        return self

    def fit_transform(self, X, y = None):
        return _union_ft(self.transformer_list, X, y, self.transformer_weights)

    def transform(self, X):
        return _union_t(self.transformer_list, X, self.transformer_weights)

def make_union(*transformers, n_jobs: Optional[int] = None, verbose: bool = False):
    return FeatureUnion(_named(transformers), n_jobs=n_jobs, verbose=verbose)
