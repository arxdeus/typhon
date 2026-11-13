
from .ndarray import ndarray
from .routines import asarray, empty, expand_dims
from .ufunc import UnaryUFunc, UnaryUFunc2, BinaryUFunc

import util
import internal.static as static

def apply_along_axis(func1d, axis: int, arr, *args, **kwargs):
    a = asarray(arr)
    axis = util.normalize_axis_index(axis, a.ndim)
    n = a.shape[axis]
    s = a.strides[axis]

    first_slice = ndarray((n,), (s,), a.data)
    f0 = asarray(func1d(first_slice, *args, **kwargs))

    a_iter_space = util.tuple_delete(a.shape, axis)
    new_shape = util.tuple_insert_tuple(a_iter_space, axis, f0.shape)
    ans = empty(new_shape, f0.dtype)
    first = True

    for idx0 in util.multirange(a_iter_space):
        if first:
            fx = f0
            first = False
        else:
            idx = util.tuple_insert(idx0, axis, 0)
            next_slice = ndarray((n,), (s,), a._ptr(idx))
            fx = asarray(func1d(next_slice, *args, **kwargs))

            if fx.shape != f0.shape:
                raise ValueError("function is returning arrays of different shape")

        if fx.ndim > 0:
            for idx1 in util.multirange(fx.shape):
                idx2 = util.tuple_insert_tuple(idx0, axis, idx1)
                p = fx._ptr(idx1)
                q = ans._ptr(idx2)
                q[0] = p[0]
        else:
            ans._ptr(idx0)[0] = fx.item()

    return ans

def apply_over_axes(func, a, axes):
    a = asarray(a)
    dtype = func(a, 0).dtype

    if static.len(dtype) >= 0:
        val = a.astype(dtype)
        if asarray(axes).ndim == 0:
            ax = (axes,)
        else:
            ax = axes

        for axis in ax:
            if axis < 0:
                axis += val.ndim
            b = func(val, axis)

            if isinstance(b, ndarray):
                if b.ndim == val.ndim:
                    val = b
                elif b.ndim == val.ndim - 1:
                    val = expand_dims(b, axis)
                else:
                    compile_error("function is not returning an array of the correct shape")
            else:
                compile_error("function is not returning ndarray")

        return val

def frompyfunc(func, nin: Literal[int], nout: Literal[int], identity):
    if nin == 1 and nout == 1:
        return UnaryUFunc(func, func.__name__ + ' (vectorized)')
    elif nin == 1 and nout == 2:
        return UnaryUFunc2(func, func.__name__ + ' (vectorized)')
    elif nin == 2 and nout == 1:
        return BinaryUFunc(func, func.__name__ + ' (vectorized)', identity)
    else:
        compile_error("given combination of 'nin' and 'nout' is not supported")

def vectorize(pyfunc):
    import internal.static as S
    nout: Literal[int] = 1
    nin: Literal[int] = static.len(static.function.args(pyfunc))
    return frompyfunc(pyfunc, nin=nin, nout=nout, identity=None)

def _piecewise_apply(f, e, dtype: type, args, kw):
    if hasattr(f, "__call__"):
        return util.cast(f(e, *args, **kw), dtype)
    else:
        return util.cast(f, dtype)

def piecewise(x, condlist, funclist, *args, **kw):
    x = asarray(x)
    ans = empty(x.shape, x.dtype)

    n_condlist = len(condlist)
    n_funclist = len(funclist)

    if n_condlist != n_funclist and n_condlist + 1 != n_funclist:
        raise ValueError(f"with {n_condlist} conditions, either {n_condlist} or {n_condlist + 1} functions are expected")

    for cond in condlist:
        acond = asarray(cond)
        if acond.ndim != x.ndim:
            compile_error("condition dimension does not match input domain dimension")
        if acond.shape != x.shape:
            raise ValueError("condition shape does not match input domain shape")

    for idx in util.multirange(x.shape):
        e = x._ptr(idx)[0]
        # numpy assigns in condlist order, so the last true condition wins;
        # the default function (n+1-th) applies only where no condition holds.
        i = -1
        k = 0
        for cond in condlist:
            if asarray(cond)._ptr(idx)[0]:
                i = k
            k += 1

        if i == -1:
            if n_condlist + 1 == n_funclist:
                i = n_funclist - 1
            else:
                ans._ptr(idx)[0] = util.cast(0, x.dtype)
                continue

        if isinstance(funclist, Tuple):
            # heterogeneous funclist (scalars mixed with callables)
            for j in static.range(static.len(funclist)):
                if j == i:
                    e = _piecewise_apply(funclist[j], e, x.dtype, args, kw)
        else:
            e = _piecewise_apply(funclist[i], e, x.dtype, args, kw)

        ans._ptr(idx)[0] = e

    return ans
