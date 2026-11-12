# Ported from torch 2.15.0 (BSD-3-Clause)
#
# torch.nn.utils.parametrize for Typhon.
#
# register_parametrization(module, "weight", p) moves the parameter to
# `module.parametrizations.weight.original` (state_dict key
# `parametrizations.weight.original`, parametrization params under
# `parametrizations.weight.0.*`) and recomputes `module.weight = p(original)`
# on every call of `module(...)`. The module's typed `weight` field keeps
# pointing to the same Tensor object, whose contents are refreshed in place
# (with grad_fn connecting it to `original`). Single parametrization per
# tensor (chaining is applied by composing inside one Module).

from torch._tensor import Tensor, no_grad
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleDict

class ParametrizationList(Module):
    original: Tensor
    unsafe: bool

    def __init__(self, parametrization: Module, original: Tensor, unsafe: bool = False):
        super().__init__()
        self.unsafe = unsafe
        self.original = original
        self.add_module("0", parametrization)
        self.register_parameter("original", original)

    def forward(self, input: Tensor) -> Tensor:
        x = self.original
        for m in self.children():
            x = m(x)
        return x

    def right_inverse(self, value: Tensor):
        with no_grad():
            self.original.copy_(value)

    def _get_name(self) -> str:
        return "ParametrizationList"

def _find_param(module: Module, name: str) -> int:
    for i in range(len(module._pnames)):
        if module._pnames[i] == name and module._params[i] is not None:
            return i
    return -1

def is_parametrized(module: Module, tensor_name: Optional[str] = None) -> bool:
    if tensor_name is None:
        return len(module._ptz_names) > 0
    return tensor_name in module._ptz_names

def register_parametrization(module: Module, tensor_name: str, parametrization, unsafe: bool = False):
    pm: Module = parametrization
    if tensor_name in module._ptz_names:
        raise NotImplementedError("chaining several parametrizations on the same tensor is not supported in Typhon")
    idx = _find_param(module, tensor_name)
    if idx < 0:
        raise ValueError(f"Module '{module._get_name()}' has no parameter or buffer with name '{tensor_name}'")
    w = module._params[idx]
    # initial value of `original`
    orig_val: Tensor
    with no_grad():
        if hasattr(parametrization, "right_inverse"):
            orig_val = parametrization.right_inverse(w.detach()).clone()
        else:
            orig_val = w.detach().clone()
        if not unsafe:
            y = pm(orig_val)
            if y._shape != w._shape:
                raise ValueError(f"Registering a parametrization may not change the shape of the tensor, unless `unsafe` flag is enabled.")
            if y.dtype != w.dtype:
                raise ValueError("Registering a parametrization may not change the dtype of the tensor, unless `unsafe` flag is enabled.")
    original = orig_val.detach()
    original.requires_grad = w.requires_grad
    # unregister the plain parameter
    module._pnames.pop(idx)
    module._params.pop(idx)
    plist = ParametrizationList(pm, original, unsafe)
    pd: ModuleDict
    if module._has_child("parametrizations"):
        c = module._child("parametrizations")
        pd = ModuleDict()
        # rebuild with existing entries (ModuleDict is the stored child)
        for k, v in c.named_children():
            pd[k] = v
        for i in range(len(module._mnames)):
            if module._mnames[i] == "parametrizations":
                module._mods[i] = pd
    else:
        pd = ModuleDict()
        module.add_module("parametrizations", pd)
    pd[tensor_name] = plist
    module._ptz_names.append(tensor_name)
    module._ptz_fns.append(pm)
    module._ptz_orig.append(original)
    module._ptz_cache.append(w)
    # w is now a non-leaf computed tensor; refresh its value
    w.grad = None
    module._refresh_parametrizations()
    return module

def remove_parametrizations(module: Module, tensor_name: str, leave_parametrized: bool = True):
    if tensor_name not in module._ptz_names:
        raise ValueError(f"Module {module._get_name()} does not have a parametrization on {tensor_name}")
    i = module._ptz_names.index(tensor_name)
    w = module._ptz_cache[i]
    orig = module._ptz_orig[i]
    with no_grad():
        if leave_parametrized:
            v = module._ptz_fns[i](orig).detach()
        else:
            v = orig.detach()
        w._st = v._st
        w._off = v._off
        w._shape = v._shape
        w._strides = v._strides
        w._dt = v._dt
    w.grad_fn = None
    w.requires_grad = orig.requires_grad
    module._ptz_names.pop(i)
    module._ptz_fns.pop(i)
    module._ptz_orig.pop(i)
    module._ptz_cache.pop(i)
    module.register_parameter(tensor_name, w)
    pd = module._child("parametrizations")
    for j in range(len(pd._mnames)):
        if pd._mnames[j] == tensor_name:
            pd._mnames.pop(j)
            pd._mods.pop(j)
            break
    if len(pd._mnames) == 0:
        for j in range(len(module._mnames)):
            if module._mnames[j] == "parametrizations":
                module._mnames.pop(j)
                module._mods.pop(j)
                break
    return module

class cached:
    _c: int
    def __init__(self):
        self._c = 0
    def __enter__(self):
        pass
    def __exit__(self):
        pass
