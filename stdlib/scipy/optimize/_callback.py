# Ported from scipy 1.18.1 (BSD-3-Clause)

# Callback plumbing shared by the minimizers (scipy _wrap_callback +
# _call_callback_maybe_halt): a callback taking a single keyword argument
# `intermediate_result` receives an OptimizeResult, anything else receives
# a copy of x. Raising StopIteration halts the solver (status 99).

import numpy as np
from ._common import OptimizeResult

def _cb_result(x: np.ndarray[float, 1], fun: float):
    return OptimizeResult(x.copy(), fun, np.empty(0, float))

def _cb_halt(callback, x: np.ndarray[float, 1], fun: float) -> bool:
    if callback is None:
        return False
    else:
        try:
            r = _cb_result(x, fun)
            if static.function.can_call(callback, intermediate_result=r):
                callback(intermediate_result=r)
            else:
                callback(x.copy())
        except StopIteration:
            return True
        return False

def _cb_halt_r(callback, r) -> bool:
    # variant with a caller-built intermediate result (trust-constr style
    # callbacks may take (x, result))
    if callback is None:
        return False
    else:
        try:
            if static.function.can_call(callback, intermediate_result=r):
                callback(intermediate_result=r)
            elif static.function.can_call(callback, r.x, r):
                callback(r.x.copy(), r)
            else:
                callback(r.x.copy())
        except StopIteration:
            return True
        return False

_STOP_MSG = '`callback` raised `StopIteration`.'
