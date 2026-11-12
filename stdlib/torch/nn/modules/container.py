# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Containers: Sequential, ModuleList, ModuleDict, ParameterList, Identity.

from torch._tensor import Tensor
from torch.nn.modules.module import Module

def _as_module(m) -> Module:
    return m

class Identity(Module):
    _id: int

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._id = 0

    def forward(self, input: Tensor) -> Tensor:
        return input

    def _get_name(self) -> str:
        return "Identity"

class Sequential(Module):
    """Sequential container. Subclassable: layers added later via add_module
    participate in forward in insertion order, like torch."""
    _seq: int

    def __init__(self, *args):
        super().__init__()
        self._seq = 0
        for i in static.range(static.len(args)):
            if isinstance(args[i], Dict[str, Module]):
                for k, v in args[i].items():
                    self.add_module(k, v)
            elif isinstance(args[i], List[Tuple[str, Module]]):
                for k, v in args[i]:
                    self.add_module(k, v)
            else:
                self.add_module(str(i), _as_module(args[i]))

    def forward(self, input: Tensor) -> Tensor:
        x = input
        for m in self._mods:
            if m is not None:
                x = m(x)
        return x

    def __len__(self) -> int:
        return len(self.children())

    def __getitem__(self, idx: int) -> Module:
        c = self.children()
        n = len(c)
        if idx < -n or idx >= n:
            raise IndexError(f"index {idx} is out of range")
        return c[idx + n if idx < 0 else idx]

    def __getitem__(self, s: slice):
        c = self.named_children()
        out = Sequential()
        for i in range(len(c))[s]:
            out.add_module(c[i][0], c[i][1])
        return out

    def __setitem__(self, idx: int, module: Module):
        n = len(self._mnames)
        i = idx + n if idx < 0 else idx
        self._mods[i] = module

    def __delitem__(self, idx: int):
        n = len(self._mnames)
        i = idx + n if idx < 0 else idx
        self._mnames.pop(i)
        self._mods.pop(i)

    def __iter__(self):
        for m in self.children():
            yield m

    def append(self, module: Module):
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module):
        n = len(self._mods)
        if index < 0:
            index += n
        mods = [m for m in self._mods]
        mods.insert(index, module)
        self._mnames = [str(i) for i in range(len(mods))]
        self._mods = mods
        return self

    def extend(self, sequential):
        for m in sequential:
            self.append(m)
        return self

    def pop(self, key: int) -> Module:
        v = self[key]
        del self[key]
        return v

    def _get_name(self) -> str:
        return "Sequential"

class ModuleList(Module):
    _ml: int

    def __init__(self, modules = None):
        super().__init__()
        self._ml = 0
        if modules is not None:
            for m in modules:
                self.append(m)

    def append(self, module: Module):
        self.add_module(str(len(self._mnames)), module)
        return self

    def extend(self, modules):
        for m in modules:
            self.append(m)
        return self

    def insert(self, index: int, module: Module):
        mods = [m for m in self._mods]
        mods.insert(index, module)
        self._mnames = [str(i) for i in range(len(mods))]
        self._mods = mods

    def __len__(self) -> int:
        return len(self._mods)

    def __getitem__(self, idx: int) -> Module:
        n = len(self._mods)
        if idx < -n or idx >= n:
            raise IndexError(f"index {idx} is out of range")
        m = self._mods[idx + n if idx < 0 else idx]
        return m

    def __setitem__(self, idx: int, module: Module):
        n = len(self._mods)
        self._mods[idx + n if idx < 0 else idx] = module

    def __iter__(self):
        for m in self._mods:
            if m is not None:
                yield m

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError("ModuleList has no forward")

    def _get_name(self) -> str:
        return "ModuleList"

class ModuleDict(Module):
    _md: int

    def __init__(self, modules = None):
        super().__init__()
        self._md = 0
        if modules is not None:
            if isinstance(modules, Dict):
                for k, v in modules.items():
                    self[k] = v
            else:
                for k, v in modules:
                    self[k] = v

    def __getitem__(self, key: str) -> Module:
        return self._child(key)

    def __setitem__(self, key: str, module: Module):
        self.add_module(key, module)

    def __delitem__(self, key: str):
        for i in range(len(self._mnames)):
            if self._mnames[i] == key:
                self._mnames.pop(i)
                self._mods.pop(i)
                return
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._mnames

    def __len__(self) -> int:
        return len(self._mnames)

    def __iter__(self):
        for k in self._mnames:
            yield k

    def keys(self) -> List[str]:
        return self._mnames.copy()

    def items(self) -> List[Tuple[str, Module]]:
        return self.named_children()

    def values(self) -> List[Module]:
        return self.children()

    def update(self, modules):
        for k, v in modules.items():
            self[k] = v

    def pop(self, key: str) -> Module:
        v = self[key]
        del self[key]
        return v

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError("ModuleDict has no forward")

    def _get_name(self) -> str:
        return "ModuleDict"

class ParameterList(Module):
    _pl: int

    def __init__(self, values = None):
        super().__init__()
        self._pl = 0
        if values is not None:
            for v in values:
                self.append(v)

    def append(self, value: Tensor):
        self.register_parameter(str(len(self._pnames)), value)
        return self

    def __len__(self) -> int:
        return len(self._pnames)

    def __getitem__(self, idx: int) -> Tensor:
        n = len(self._params)
        p = self._params[idx + n if idx < 0 else idx]
        return p

    def __iter__(self):
        for p in self._params:
            if p is not None:
                yield p

    def _get_name(self) -> str:
        return "ParameterList"
