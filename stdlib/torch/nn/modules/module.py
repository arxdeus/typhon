# Ported from torch 2.15.0 (BSD-3-Clause)
#
# torch.nn.Module for Typhon.
#
# Design: parameters, buffers and submodules live in ordered registries
# (name lists + object lists) on the base class, so traversal
# (named_parameters, state_dict, ...) works for any subclass. Subclasses keep
# typed fields that alias the same Tensor / Module objects (Tensors are
# reference types), e.g. `self.weight = self.register_parameter("weight", w)`.
# All in-place updates (optimizers, load_state_dict, to(dtype)) mutate the
# Tensor object itself, so the aliases stay valid.
#
# Every subclass must declare at least one field (Typhon auto-deduces fields of
# field-less classes, which breaks inheritance). forward has the single
# signature `forward(self, input: Tensor) -> Tensor`; virtual dispatch through
# `Module`-typed references works.

from torch._tensor import Tensor, _DType

class Module:
    training: bool
    _pnames: List[str]
    _params: List[Optional[Tensor]]
    _bnames: List[str]
    _buffers: List[Optional[Tensor]]
    _bpersist: List[bool]
    _mnames: List[str]
    _mods: List[Optional[Module]]
    # parametrizations: name -> index into _mods of the ParametrizationList
    _ptz_names: List[str]
    _ptz_fns: List[Module]
    _ptz_orig: List[Tensor]
    _ptz_cache: List[Tensor]
    _autoreg_done: bool

    def __init__(self):
        self._autoreg_done = False
        self.training = True
        self._pnames = []
        self._params = []
        self._bnames = []
        self._buffers = []
        self._bpersist = []
        self._mnames = []
        self._mods = []
        self._ptz_names = []
        self._ptz_fns = []
        self._ptz_orig = []
        self._ptz_cache = []

    # ------------------------------------------------------------ forward
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError(f'Module [{self._get_name()}] is missing the required "forward" function')

    def __call__(self, *inputs):
        # one generic entry for `model(x)` and multi-input forwards
        # (`model(x1, x2, x3, x4)`); an overloaded __call__ would bind `self`
        # to the base class and lose the subclass forward.
        self._autoreg_call()
        self._refresh_parametrizations()
        return self.forward(*inputs)

    def _autoreg_call(self):
        # hot path: fields are complete once the module is called
        if not self._autoreg_done:
            self._autoreg()
            self._autoreg_done = True

    def _autoreg(self):
        # torch registers submodules in Module.__setattr__. Plain attribute
        # assignment in a user subclass (`self.block = nn.Sequential(...)`)
        # bypasses add_module here, so the Module-typed fields of the static
        # type of `self` are registered lazily under their attribute names
        # (unless the same object is already registered). Recurses through the
        # concrete field types. Order: field order of the class, not assignment
        # order (affects only parameters() order, not state_dict keys).
        for k, v in static.vars(self):
            if isinstance(v, Module):
                if v.__raw__() != cobj():
                    known = False
                    for m in self._mods:
                        if m is not None and m.__val__() is v:
                            known = True
                    if not known:
                        self.add_module(k, v)
                    v._autoreg()

    def _get_name(self) -> str:
        n = self.__class__.__name__
        i = n.find("[")
        return n[:i] if i >= 0 else n

    def extra_repr(self) -> str:
        return ""

    # ------------------------------------------------------------ registration
    def register_parameter(self, name: str, param: Optional[Tensor]) -> Optional[Tensor]:
        if "." in name:
            raise KeyError('parameter name can\'t contain "."')
        if name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        if param is not None:
            p = param
            if not p.is_leaf:
                raise ValueError(f"Cannot assign non-leaf Tensor to parameter '{name}'.")
        for i in range(len(self._pnames)):
            if self._pnames[i] == name:
                self._params[i] = param
                return param
        self._pnames.append(name)
        self._params.append(param)
        return param

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> Optional[Tensor]:
        if "." in name:
            raise KeyError('buffer name can\'t contain "."')
        for i in range(len(self._bnames)):
            if self._bnames[i] == name:
                self._buffers[i] = tensor
                self._bpersist[i] = persistent
                return tensor
        self._bnames.append(name)
        self._buffers.append(tensor)
        self._bpersist.append(persistent)
        return tensor

    def add_module(self, name: str, module: Optional[Module]):
        if "." in name:
            raise KeyError(f'module name can\'t contain ".", got: {name}')
        if name == "":
            raise KeyError('module name can\'t be empty string ""')
        for i in range(len(self._mnames)):
            if self._mnames[i] == name:
                self._mods[i] = module
                return
        self._mnames.append(name)
        self._mods.append(module)

    def register_module(self, name: str, module: Optional[Module]):
        self.add_module(name, module)

    def get_submodule(self, target: str) -> Module:
        self._autoreg()
        if target == "":
            return self
        mod: Module = self
        for item in target.split("."):
            found = False
            for i in range(len(mod._mnames)):
                if mod._mnames[i] == item and mod._mods[i] is not None:
                    mod = mod._mods[i].__val__()
                    found = True
                    break
            if not found:
                raise AttributeError(mod._get_name() + " has no attribute `" + item + "`")
        return mod

    def get_parameter(self, target: str) -> Tensor:
        idx = target.rfind(".")
        mod = self.get_submodule(target[:idx] if idx >= 0 else "")
        name = target[idx + 1:]
        for i in range(len(mod._pnames)):
            if mod._pnames[i] == name and mod._params[i] is not None:
                return mod._params[i].__val__()
        raise AttributeError("`" + name + "` is not a parameter")

    def _child(self, name: str) -> Module:
        for i in range(len(self._mnames)):
            if self._mnames[i] == name and self._mods[i] is not None:
                return self._mods[i].__val__()
        raise KeyError(name)

    def _has_child(self, name: str) -> bool:
        for i in range(len(self._mnames)):
            if self._mnames[i] == name and self._mods[i] is not None:
                return True
        return False

    # ------------------------------------------------------------ traversal
    def _named_members_params(self, prefix: str, recurse: bool, out: List[Tuple[str, Tensor]], seen: Set[int]):
        for i in range(len(self._pnames)):
            p = self._params[i]
            if p is not None:
                pp = p.__val__()
                k = id(pp)
                if k not in seen:
                    seen.add(k)
                    out.append((prefix + self._pnames[i], pp))
        if recurse:
            for i in range(len(self._mnames)):
                m = self._mods[i]
                if m is not None:
                    m._named_members_params(prefix + self._mnames[i] + ".", True, out, seen)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True) -> List[Tuple[str, Tensor]]:
        self._autoreg()
        out = List[Tuple[str, Tensor]]()
        seen = Set[int]()
        self._named_members_params(prefix + "." if prefix else "", recurse, out, seen)
        return out

    def parameters(self, recurse: bool = True) -> List[Tensor]:
        return [p for _, p in self.named_parameters(recurse=recurse)]

    def _named_members_buffers(self, prefix: str, recurse: bool, out: List[Tuple[str, Tensor]], seen: Set[int]):
        for i in range(len(self._bnames)):
            b = self._buffers[i]
            if b is not None:
                bb = b.__val__()
                k = id(bb)
                if k not in seen:
                    seen.add(k)
                    out.append((prefix + self._bnames[i], bb))
        if recurse:
            for i in range(len(self._mnames)):
                m = self._mods[i]
                if m is not None:
                    m._named_members_buffers(prefix + self._mnames[i] + ".", True, out, seen)

    def named_buffers(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True) -> List[Tuple[str, Tensor]]:
        self._autoreg()
        out = List[Tuple[str, Tensor]]()
        seen = Set[int]()
        self._named_members_buffers(prefix + "." if prefix else "", recurse, out, seen)
        return out

    def buffers(self, recurse: bool = True) -> List[Tensor]:
        return [b for _, b in self.named_buffers(recurse=recurse)]

    def named_children(self) -> List[Tuple[str, Module]]:
        self._autoreg()
        out = List[Tuple[str, Module]]()
        seen = Set[int]()
        for i in range(len(self._mnames)):
            m = self._mods[i]
            if m is not None:
                mm = m.__val__()
                if id(mm) not in seen:
                    seen.add(id(mm))
                    out.append((self._mnames[i], mm))
        return out

    def children(self) -> List[Module]:
        return [m for _, m in self.named_children()]

    def _nm(self, prefix: str, out: List[Tuple[str, Module]], seen: Set[int]):
        if id(self) in seen:
            return
        seen.add(id(self))
        me: Module = self
        out.append((prefix, me))
        for i in range(len(self._mnames)):
            m = self._mods[i]
            if m is not None:
                m._nm(prefix + ("." if prefix else "") + self._mnames[i], out, seen)

    def named_modules(self, prefix: str = "") -> List[Tuple[str, Module]]:
        self._autoreg()
        out = List[Tuple[str, Module]]()
        seen = Set[int]()
        self._nm(prefix, out, seen)
        return out

    def modules(self) -> List[Module]:
        return [m for _, m in self.named_modules()]

    # ------------------------------------------------------------ state
    def _save_to_state_dict(self, prefix: str, d: Dict[str, Tensor], keep_vars: bool):
        for i in range(len(self._pnames)):
            p = self._params[i]
            if p is not None:
                pv = p.__val__()
                d[prefix + self._pnames[i]] = pv if keep_vars else pv.detach()
        for i in range(len(self._bnames)):
            b = self._buffers[i]
            if b is not None and self._bpersist[i]:
                bv = b.__val__()
                d[prefix + self._bnames[i]] = bv if keep_vars else bv.detach()
        for i in range(len(self._mnames)):
            m = self._mods[i]
            if m is not None:
                m._save_to_state_dict(prefix + self._mnames[i] + ".", d, keep_vars)

    def state_dict(self, prefix: str = "", keep_vars: bool = False) -> Dict[str, Tensor]:
        self._autoreg()
        d = Dict[str, Tensor]()
        self._save_to_state_dict(prefix, d, keep_vars)
        return d

    def _load(self, prefix: str, sd: Dict[str, Tensor], missing: List[str], used: Set[str]):
        from torch._tensor import no_grad
        for i in range(len(self._pnames)):
            p = self._params[i]
            if p is not None:
                k = prefix + self._pnames[i]
                if k in sd:
                    used.add(k)
                    src = sd[k]
                    if src.shape != p.shape:
                        raise RuntimeError(f"size mismatch for {k}: copying a param with shape {src.shape} from checkpoint, the shape in current model is {p.shape}.")
                    with no_grad():
                        p.copy_(src)
                else:
                    missing.append(k)
        for i in range(len(self._bnames)):
            b = self._buffers[i]
            if b is not None and self._bpersist[i]:
                k = prefix + self._bnames[i]
                if k in sd:
                    used.add(k)
                    src = sd[k]
                    if src.shape != b.shape:
                        raise RuntimeError(f"size mismatch for {k}: copying a param with shape {src.shape} from checkpoint, the shape in current model is {b.shape}.")
                    with no_grad():
                        b.copy_(src)
                else:
                    missing.append(k)
        for i in range(len(self._mnames)):
            m = self._mods[i]
            if m is not None:
                m._load(prefix + self._mnames[i] + ".", sd, missing, used)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        self._autoreg()
        missing = List[str]()
        used = Set[str]()
        self._load("", state_dict, missing, used)
        unexpected = [k for k in state_dict.keys() if k not in used]
        if strict and (len(missing) > 0 or len(unexpected) > 0):
            msg = "Error(s) in loading state_dict for " + self._get_name() + ":"
            if len(unexpected) > 0:
                msg += " Unexpected key(s) in state_dict: " + ", ".join(['"' + k + '"' for k in unexpected]) + "."
            if len(missing) > 0:
                msg += " Missing key(s) in state_dict: " + ", ".join(['"' + k + '"' for k in missing]) + "."
            raise RuntimeError(msg)
        return _IncompatibleKeys(missing, unexpected)

    # ------------------------------------------------------------ modes
    def train(self, mode: bool = True):
        # torch returns self; here train() returns None because user subclasses
        # commonly override it without a return (`super().train(mode)`) and a
        # virtual override must keep the base return type. eval() returns self.
        self._autoreg()
        self.training = mode
        for m in self.children():
            m.train(mode)

    def eval(self):
        # register first: an overridden train() reaches Module.train through
        # super(), where `self` is statically Module and has no field info
        self._autoreg()
        self.train(False)
        return self

    def requires_grad_(self, requires_grad: bool = True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True):
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    g = p.grad
                    g.detach_()
                    g.zero_()

    def apply(self, fn):
        for m in self.children():
            m.apply(fn)
        fn(self)
        return self

    def _to_dtype(self, dtype: _DType):
        from torch._tensor import no_grad
        with no_grad():
            for t in self.parameters() + self.buffers():
                if t.is_floating_point() and t.dtype != dtype:
                    n = t.to(dtype)
                    t._st = n._st
                    t._off = n._off
                    t._strides = n._strides
                    t._dt = n._dt
                    if t.grad is not None:
                        t.grad = t.grad.to(dtype)
            for o in self._ptz_orig:
                if o.is_floating_point() and o.dtype != dtype:
                    n = o.to(dtype)
                    o._st = n._st
                    o._off = n._off
                    o._strides = n._strides
                    o._dt = n._dt

    def to(self, arg, dtype: Optional[_DType] = None):
        # One method (not overloads, which would bind `self` to Module and
        # lose the subclass type of `model = Net().to(device)`): `to(dtype)`
        # or `to(device[, dtype])` (devices are strings, CPU only).
        if isinstance(arg, _DType):
            for m in self.modules():
                m._to_dtype(arg)
        elif dtype is not None:
            for m in self.modules():
                m._to_dtype(dtype)
        return self

    def float(self):
        from torch import float32 as _f32
        self.to(_f32)
        return self

    def double(self):
        from torch import float64 as _f64
        self.to(_f64)
        return self

    def cpu(self):
        return self

    def type(self, dst_type: _DType):
        self.to(dst_type)
        return self

    # ------------------------------------------------------------ parametrize
    def _refresh_parametrizations(self):
        # Recompute parametrized tensors (torch recomputes on every attribute
        # access; here recomputation happens at each module call).
        for i in range(len(self._ptz_names)):
            new = self._ptz_fns[i](self._ptz_orig[i])
            dst = self._ptz_cache[i]
            dst._st = new._st
            dst._off = new._off
            dst._shape = new._shape
            dst._strides = new._strides
            dst._dt = new._dt
            dst.grad_fn = new.grad_fn
            dst.requires_grad = new.requires_grad

    # ------------------------------------------------------------ repr
    def __repr__(self) -> str:
        lines = List[str]()
        er = self.extra_repr()
        if er:
            lines = er.split("\n")
        child_lines = List[str]()
        for name, m in self.named_children():
            s = m.__repr__()
            s = s.replace("\n", "\n  ")
            child_lines.append("(" + name + "): " + s)
        lines = lines + child_lines
        main = self._get_name() + "("
        if lines:
            if len(lines) == 1 and len(child_lines) == 0:
                main += lines[0]
            else:
                main += "\n  " + "\n  ".join(lines) + "\n"
        main += ")"
        return main

    def __str__(self) -> str:
        return self.__repr__()

@tuple
class _IncompatibleKeys:
    missing_keys: List[str]
    unexpected_keys: List[str]
