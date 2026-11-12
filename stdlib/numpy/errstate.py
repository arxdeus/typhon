# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# Floating-point error handling state: seterr / geterr / errstate /
# seterrcall / geterrcall. The state is tracked faithfully (modes are
# validated and returned like numpy), but the Typhon runtime does not
# inspect hardware FP flags, so array operations never warn or raise on
# divide/over/under/invalid conditions regardless of the mode.

_ERR_MODES = ("ignore", "warn", "raise", "call", "print", "log")

class _ErrState:
    divide: str
    over: str
    under: str
    invalid: str
    call: Optional[Function[[str, int], None]]

    def __init__(self):
        self.divide = "warn"
        self.over = "warn"
        self.under = "ignore"
        self.invalid = "warn"
        self.call = None

_state = _ErrState()

def _check_mode(m: str):
    if m not in _ERR_MODES:
        raise ValueError(f"invalid error mode {m!r}; allowed: {_ERR_MODES}")

def geterr() -> Dict[str, str]:
    return {"divide": _state.divide, "over": _state.over, "under": _state.under,
            "invalid": _state.invalid}

def seterr(all: Optional[str] = None, divide: Optional[str] = None, over: Optional[str] = None,
           under: Optional[str] = None, invalid: Optional[str] = None) -> Dict[str, str]:
    old = geterr()
    if all is not None:
        a = all.__val__()
        _check_mode(a)
        _state.divide = a
        _state.over = a
        _state.under = a
        _state.invalid = a
    if divide is not None:
        _check_mode(divide.__val__())
        _state.divide = divide.__val__()
    if over is not None:
        _check_mode(over.__val__())
        _state.over = over.__val__()
    if under is not None:
        _check_mode(under.__val__())
        _state.under = under.__val__()
    if invalid is not None:
        _check_mode(invalid.__val__())
        _state.invalid = invalid.__val__()
    return old

def _seterr_dict(d: Dict[str, str]):
    seterr(divide=d["divide"], over=d["over"], under=d["under"], invalid=d["invalid"])

def geterrcall():
    return _state.call

def seterrcall(func):
    old = _state.call
    if func is None:
        _state.call = None
    else:
        _state.call = func
    return old

class errstate:
    _all: Optional[str]
    _divide: Optional[str]
    _over: Optional[str]
    _under: Optional[str]
    _invalid: Optional[str]
    _old: Dict[str, str]

    def __init__(self, *, call = None, all: Optional[str] = None, divide: Optional[str] = None,
                 over: Optional[str] = None, under: Optional[str] = None,
                 invalid: Optional[str] = None):
        self._all = all
        self._divide = divide
        self._over = over
        self._under = under
        self._invalid = invalid
        self._old = Dict[str, str]()

    def __enter__(self):
        self._old = seterr(all=self._all, divide=self._divide, over=self._over,
                           under=self._under, invalid=self._invalid)
        return self

    def __exit__(self):
        _seterr_dict(self._old)
