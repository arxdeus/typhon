# Ported from einops 0.8.2 (MIT)
#
# einops for torch Tensors. Patterns are parsed at runtime, so only tensor
# types with a runtime shape (torch.Tensor) are supported: numpy ndarrays in
# Typhon carry ndim in their static type, which a runtime pattern cannot set.

from ._recipe import EinopsError, make_plan, Plan
from torch._tensor import Tensor

def _apply_plan(x: Tensor, plan: Plan, op: str) -> Tensor:
    t = x
    if plan.need_init:
        t = t.reshape(plan.init_shape)
    if len(plan.perm) > 0:
        t = t.permute(plan.perm)
    if len(plan.reduced) > 0:
        if op == "sum":
            t = t.sum(plan.reduced)
        elif op == "mean":
            t = t.mean(plan.reduced)
        elif op == "max":
            t = t.amax(plan.reduced)
        elif op == "min":
            t = t.amin(plan.reduced)
        elif op == "prod":
            for i in range(len(plan.reduced) - 1, -1, -1):
                t = t.prod(plan.reduced[i])
        elif op == "any":
            for i in range(len(plan.reduced) - 1, -1, -1):
                t = t.any(plan.reduced[i])
        elif op == "all":
            for i in range(len(plan.reduced) - 1, -1, -1):
                t = t.all(plan.reduced[i])
        else:
            raise EinopsError(f"Unknown reduction {op}")
    if len(plan.added) > 0:
        src = t._shape
        shp = List[int]()
        full = List[int]()
        k = 0
        j = 0
        for i in range(plan.n_after_add):
            if j < len(plan.added) and plan.added[j][0] == i:
                shp.append(1)
                full.append(plan.added[j][1])
                j += 1
            else:
                shp.append(src[k])
                full.append(src[k])
                k += 1
        t = t.reshape(shp).expand(full)
    if plan.need_final:
        t = t.reshape(plan.final_shape)
    return t

def _apply(x, pattern: str, op: str, axes_lengths: Dict[str, int]) -> Tensor:
    if not isinstance(x, Tensor):
        if isinstance(x, List[Tensor]):
            from torch import stack
            return _apply(stack(x, 0), pattern, op, axes_lengths)
        compile_error("einops: only torch.Tensor inputs are supported in Typhon")
    plan = make_plan(pattern, op, axes_lengths, x._shape)
    return _apply_plan(x, plan, op)

def _lengths(kw) -> Dict[str, int]:
    d = Dict[str, int]()
    if static.len(kw.args) == 0:
        return d
    else:
        keys = kw.__keys__()
        for i in static.range(static.len(kw.args)):
            d[keys[i]] = int(kw.args[i])
    return d

def rearrange(tensor, pattern: str, **axes_lengths) -> Tensor:
    return _apply(tensor, pattern, "rearrange", _lengths(axes_lengths))

def reduce(tensor, pattern: str, reduction: str, **axes_lengths) -> Tensor:
    return _apply(tensor, pattern, reduction, _lengths(axes_lengths))

def repeat(tensor, pattern: str, **axes_lengths) -> Tensor:
    return _apply(tensor, pattern, "repeat", _lengths(axes_lengths))

def parse_shape(x: Tensor, pattern: str) -> Dict[str, int]:
    names = [n for n in pattern.split(" ") if n != ""]
    if len(names) != x.ndim:
        raise EinopsError(f"Can't parse shape with different number of dimensions: {pattern} {x._shape}")
    d = Dict[str, int]()
    for i in range(len(names)):
        if names[i] != "_":
            d[names[i]] = x._shape[i]
    return d
