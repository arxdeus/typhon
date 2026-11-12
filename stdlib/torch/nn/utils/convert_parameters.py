# Ported from torch 2.15.0 (BSD-3-Clause)

from torch._tensor import Tensor, no_grad

def parameters_to_vector(parameters) -> Tensor:
    from torch import cat
    return cat([p.reshape([-1]) for p in parameters], 0)

def vector_to_parameters(vec: Tensor, parameters):
    off = 0
    with no_grad():
        for p in parameters:
            n = p.numel()
            p.copy_(vec.narrow(0, off, n).view_as(p))
            off += n
