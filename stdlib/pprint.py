# Ported from CPython 3.14 (PSF-2.0)
# Supports list, tuple, dict (sorted keys), set, and scalars (via repr). Long str/bytes
# wrapping and dataclasses/compact mode are unsupported.

def _fmt(obj, indent: int, width: int, column: int, allowance: int, sort_dicts: bool) -> str:
    if isinstance(obj, Dict) or isinstance(obj, Set):
        return _fmt_big_fit(obj, indent, width, column, allowance, sort_dicts)
    r = repr(obj)
    if len(r) <= width - column - allowance:
        return r
    return _fmt_big(obj, indent, width, column, allowance, sort_dicts)

def _fmt_big_fit(obj, indent: int, width: int, column: int, allowance: int, sort_dicts: bool) -> str:
    # dicts/sets: render multi-line form, then collapse if it fits on one line

    one = _oneline(obj, sort_dicts)
    if len(one) <= width - column - allowance:
        return one
    return _fmt_big(obj, indent, width, column, allowance, sort_dicts)

def _oneline(obj, sort_dicts: bool) -> str:
    if isinstance(obj, Dict):
        return _oneline_dict(obj, sort_dicts)
    return repr(sorted(obj)).replace("[", "{", 1)[:-1] + "}" if len(obj) > 0 else "set()"

def _oneline_dict(obj: Dict[K, V], sort_dicts: bool, K: type, V: type) -> str:
    keys = list(obj.keys())
    if sort_dicts:
        keys.sort()
    parts: List[str] = []
    for k in keys:
        parts.append(repr(k) + ": " + pformat(obj[k], 1, 1000000, sort_dicts))
    return "{" + ", ".join(parts) + "}"

def _fmt_big(obj, indent: int, width: int, column: int, allowance: int, sort_dicts: bool) -> str:
    if isinstance(obj, List):
        return _seq("[", "]", obj, indent, width, column, allowance, sort_dicts)
    elif isinstance(obj, Set):
        items = list(obj)
        items.sort()
        return _seq("{", "}", items, indent, width, column, allowance, sort_dicts)
    elif isinstance(obj, Dict):
        return _dict(obj, indent, width, column, allowance, sort_dicts)
    elif isinstance(obj, Tuple):
        return _tup(obj, indent, width, column, allowance, sort_dicts)
    else:
        return repr(obj)

def _dict(obj: Dict[K, V], indent: int, width: int, column: int, allowance: int, sort_dicts: bool, K: type, V: type) -> str:
    keys = list(obj.keys())
    if sort_dicts:
        keys.sort()
    out = "{"
    pad = " " * (column + indent)
    n = len(keys)
    for i in range(n):
        k = keys[i]
        if i > 0:
            out += ",\n" + pad
        else:
            out += " " * (indent - 1)
        kr = repr(k)
        out += kr + ": "
        col = column + indent + len(kr) + 2
        out += _fmt(obj[k], indent, width, col, allowance + 1 if i == n - 1 else 1, sort_dicts)
    return out + "}"

def _tup(obj, indent: int, width: int, column: int, allowance: int, sort_dicts: bool) -> str:
    out = "("
    pad = " " * (column + indent)
    n = len(obj)
    i = 0
    for item in obj:
        if i > 0:
            out += ",\n" + pad
        else:
            out += " " * (indent - 1)
        out += _fmt(item, indent, width, column + indent, allowance + 1 if i == n - 1 else 1, sort_dicts)
        i += 1
    if n == 1:
        out += ","
    return out + ")"

def _seq(open_: str, close: str, items: List[T], indent: int, width: int, column: int,
         allowance: int, sort_dicts: bool, T: type) -> str:
    out = open_
    pad = " " * (column + indent)
    n = len(items)
    for i in range(n):
        if i > 0:
            out += ",\n" + pad
        else:
            out += " " * (indent - 1)
        out += _fmt(items[i], indent, width, column + indent, allowance + 1 if i == n - 1 else 1, sort_dicts)
    return out + close

def pformat(obj, indent: int = 1, width: int = 80, sort_dicts: bool = True) -> str:
    return _fmt(obj, indent, width, 0, 0, sort_dicts)

def pprint(obj, indent: int = 1, width: int = 80, sort_dicts: bool = True):
    print(pformat(obj, indent, width, sort_dicts))

def saferepr(obj) -> str:
    return repr(obj)
