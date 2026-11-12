# Ported from torch 2.15.0 (BSD-3-Clause)
#
# torch.nn.utils.parametrizations.weight_norm. State-dict keys match torch:
# `parametrizations.<name>.original0` (g) and `.original1` (v).

from torch._tensor import Tensor, no_grad
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleDict
from torch.nn.utils.parametrize import ParametrizationList, _find_param

def _norm_except_dim(v: Tensor, dim: int) -> Tensor:
    nd = v.dim()
    if dim == -1:
        return v.norm(2)
    d = dim + nd if dim < 0 else dim
    dims = [i for i in range(nd) if i != d]
    return (v * v).sum(dims, keepdim=True).sqrt()

class _WeightNorm(Module):
    dim: int
    g: Tensor

    def __init__(self, dim: int, g: Tensor):
        super().__init__()
        self.dim = dim
        self.g = g

    def forward(self, input: Tensor) -> Tensor:
        # torch._weight_norm(v, g, dim) = v * (g / ||v||)
        return input * (self.g / _norm_except_dim(input, self.dim))

    def _get_name(self) -> str:
        return "_WeightNorm"

def weight_norm(module, name: str = "weight", dim: Optional[int] = 0):
    m: Module = module
    d = -1 if dim is None else dim.__val__()
    idx = _find_param(m, name)
    if idx < 0:
        raise ValueError(f"Module has no parameter '{name}'")
    w = m._params[idx].__val__()
    with no_grad():
        g = _norm_except_dim(w.detach(), d).clone()
        v = w.detach().clone()
    g.requires_grad = w.requires_grad
    v.requires_grad = w.requires_grad
    m._pnames.pop(idx)
    m._params.pop(idx)
    fn = _WeightNorm(d, g)
    plist = ParametrizationList(fn, v, False)
    # torch names: original0 = g, original1 = v (registered in that order)
    plist._pnames = ["original0", "original1"]
    plist._params = [Optional(g), Optional(v)]
    pd: ModuleDict
    if m._has_child("parametrizations"):
        pd = ModuleDict()
        for k, c in m._child("parametrizations").named_children():
            pd[k] = c
        for i in range(len(m._mnames)):
            if m._mnames[i] == "parametrizations":
                m._mods[i] = pd
    else:
        pd = ModuleDict()
        m.add_module("parametrizations", pd)
    pd[name] = plist
    m._ptz_names.append(name)
    m._ptz_fns.append(fn)
    m._ptz_orig.append(v)
    m._ptz_cache.append(w)
    w.grad = None
    m._refresh_parametrizations()
    return module
