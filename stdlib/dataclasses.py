# `@dataclass` itself is handled by the compiler: it recognizes the decorator
# by name and accepts the static flags init/repr/eq/order/hash/container.
# This module provides the runtime helpers that operate on such classes.
import internal.static as _static


class FrozenInstanceError(AttributeError):
    def __init__(self, message: str = ""):
        super().__init__(message)


class _MISSING_TYPE:
    pass


MISSING = _MISSING_TYPE()
KW_ONLY = _MISSING_TYPE()


@__attribute__
def dataclass():
    pass


def field(default=None, default_factory=None, init: bool = True,
          repr: bool = True, hash=None, compare: bool = True, kw_only=None):
    # Class-level defaults are evaluated once when the class is defined, so a
    # `default_factory` result is shared between instances (unlike CPython).
    if default_factory is not None:
        return default_factory()
    return default


class Field:
    name: str

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"Field(name={self.name!r})"


def fields(obj) -> List[Field]:
    return [Field(k) for k, _ in _static.vars(obj)]


def is_dataclass(obj) -> bool:
    return hasattr(obj, "__dict__")


def _asdict_value(v):
    if hasattr(v, "__dict__") and not isinstance(v, str):
        return asdict(v)
    else:
        return v


def asdict(obj):
    return {k: _asdict_value(v) for k, v in _static.vars(obj)}


def astuple(obj):
    return tuple(v for _, v in _static.vars(obj))


def replace(obj, **changes):
    r = type(obj).__new__()
    for k, v in _static.vars(obj):
        if hasattr(changes, k):
            setattr(r, k, getattr(changes, k))
        else:
            setattr(r, k, v)
    return r
