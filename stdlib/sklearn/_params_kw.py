# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Keyword -> runtime-key dispatch shared by set_params of meta estimators.

import internal.static as static

def set_kw(est, keys, vals, i: int):
    from .model_selection._params import _set_param_rt, _pv
    if static.len(vals) > 0:
        if not _set_param_rt(est, keys[i], _pv(vals[0])):
            raise ValueError(f"Invalid parameter {keys[i]!r} for estimator {est.__class__.__name__}.")
        set_kw(est, keys, vals[1:], i + 1)
