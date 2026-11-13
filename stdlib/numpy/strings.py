# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.strings: element-wise string operations on ndarray[str].
# Extra (non-array) arguments are scalars; binary string ops broadcast.

import numpy as np

def _sarr(a):
    if isinstance(a, str):
        return np.array([a]).reshape(())
    return np.asarray(a)

def _map(a, T: type, f, *args):
    x = _sarr(a)
    flat = x.ravel()
    out = np.empty(flat.shape, dtype=T)
    for i in range(flat.size):
        out[i] = f(flat[i], *args)
    return out.reshape(x.shape)

def _map2(a, b, T: type, f):
    x, y = np.broadcast_arrays(_sarr(a), _sarr(b))
    fx = x.ravel()
    fy = y.ravel()
    out = np.empty(fx.shape, dtype=T)
    for i in range(fx.size):
        out[i] = f(fx[i], fy[i])
    return out.reshape(x.shape)

def _norm_idx(n: int, start: int, end: Optional[int]):
    e = n if end is None else end
    if e > n:
        e = n
    elif e < 0:
        e += n
        if e < 0:
            e = 0
    s = start
    if s < 0:
        s += n
        if s < 0:
            s = 0
    return s, e

# ---- elementwise kernels (top-level functions so they can be passed) ----

def _k_upper(s: str): return s.upper()
def _k_lower(s: str): return s.lower()
def _k_swapcase(s: str): return s.swapcase()
def _k_capitalize(s: str): return s.capitalize()
def _k_title(s: str): return s.title()
def _k_len(s: str): return len(s)
def _k_isalpha(s: str): return s.isalpha()
def _k_isdigit(s: str): return s.isdigit()
def _k_isalnum(s: str): return s.isalnum()
def _k_isspace(s: str): return s.isspace()
def _k_islower(s: str): return s.islower()
def _k_isupper(s: str): return s.isupper()
def _k_istitle(s: str): return s.istitle()
def _k_isdecimal(s: str): return s.isdecimal()
def _k_isnumeric(s: str): return s.isnumeric()

def _k_strip(s: str, c: Optional[str]): return s.strip() if c is None else s.strip(c)
def _k_lstrip(s: str, c: Optional[str]): return s.lstrip() if c is None else s.lstrip(c)
def _k_rstrip(s: str, c: Optional[str]): return s.rstrip() if c is None else s.rstrip(c)

def _k_center(s: str, w: int, f: str): return s.center(w, f)
def _k_ljust(s: str, w: int, f: str): return s.ljust(w, f)
def _k_rjust(s: str, w: int, f: str): return s.rjust(w, f)
def _k_zfill(s: str, w: int): return s.zfill(w)
def _k_expandtabs(s: str, t: int): return s.expandtabs(t)
def _k_replace(s: str, o: str, n: str, c: int): return s.replace(o, n, c)
def _k_mul(s: str, n: int): return s * n if n > 0 else ""

def _k_find(s: str, sub: str, st: int, en: Optional[int]):
    a, b = _norm_idx(len(s), st, en)
    if a > len(s):
        return -1
    return s.find(sub, a, b)

def _k_rfind(s: str, sub: str, st: int, en: Optional[int]):
    a, b = _norm_idx(len(s), st, en)
    if a > len(s):
        return -1
    return s.rfind(sub, a, b)

def _k_index(s: str, sub: str, st: int, en: Optional[int]):
    r = _k_find(s, sub, st, en)
    if r < 0:
        raise ValueError("substring not found")
    return r

def _k_rindex(s: str, sub: str, st: int, en: Optional[int]):
    r = _k_rfind(s, sub, st, en)
    if r < 0:
        raise ValueError("substring not found")
    return r

def _k_count(s: str, sub: str, st: int, en: Optional[int]):
    a, b = _norm_idx(len(s), st, en)
    if a > len(s) or b < a:
        return 0
    return s[a:b].count(sub)

def _k_startswith(s: str, p: str, st: int, en: Optional[int]):
    a, b = _norm_idx(len(s), st, en)
    if a > len(s):
        return False
    return s[a:b].startswith(p) if b >= a else p == ""

def _k_endswith(s: str, p: str, st: int, en: Optional[int]):
    a, b = _norm_idx(len(s), st, en)
    if a > len(s):
        return False
    return s[a:b].endswith(p) if b >= a else p == ""

def _k_slice(s: str, st: Optional[int], sp: Optional[int], step: Optional[int]):
    return s[st:sp:step]

def _k_part0(s: str, sep: str):
    i = s.find(sep)
    return s if i < 0 else s[:i]
def _k_part1(s: str, sep: str):
    return sep if s.find(sep) >= 0 else ""
def _k_part2(s: str, sep: str):
    i = s.find(sep)
    return "" if i < 0 else s[i + len(sep):]
def _k_rpart0(s: str, sep: str):
    i = s.rfind(sep)
    return "" if i < 0 else s[:i]
def _k_rpart1(s: str, sep: str):
    return sep if s.rfind(sep) >= 0 else ""
def _k_rpart2(s: str, sep: str):
    i = s.rfind(sep)
    return s if i < 0 else s[i + len(sep):]

def _k_add(a: str, b: str): return a + b
def _k_eq(a: str, b: str): return a == b
def _k_ne(a: str, b: str): return a != b
def _k_lt(a: str, b: str): return a < b
def _k_le(a: str, b: str): return a <= b
def _k_gt(a: str, b: str): return a > b
def _k_ge(a: str, b: str): return a >= b

# ---- public API ----

def upper(a): return _map(a, str, _k_upper)
def lower(a): return _map(a, str, _k_lower)
def swapcase(a): return _map(a, str, _k_swapcase)
def capitalize(a): return _map(a, str, _k_capitalize)
def title(a): return _map(a, str, _k_title)
def str_len(a): return _map(a, int, _k_len)
def isalpha(a): return _map(a, bool, _k_isalpha)
def isdigit(a): return _map(a, bool, _k_isdigit)
def isalnum(a): return _map(a, bool, _k_isalnum)
def isspace(a): return _map(a, bool, _k_isspace)
def islower(a): return _map(a, bool, _k_islower)
def isupper(a): return _map(a, bool, _k_isupper)
def istitle(a): return _map(a, bool, _k_istitle)
def isdecimal(a): return _map(a, bool, _k_isdecimal)
def isnumeric(a): return _map(a, bool, _k_isnumeric)

def strip(a, chars: Optional[str] = None): return _map(a, str, _k_strip, chars)
def lstrip(a, chars: Optional[str] = None): return _map(a, str, _k_lstrip, chars)
def rstrip(a, chars: Optional[str] = None): return _map(a, str, _k_rstrip, chars)

def center(a, width: int, fillchar: str = ' '): return _map(a, str, _k_center, width, fillchar)
def ljust(a, width: int, fillchar: str = ' '): return _map(a, str, _k_ljust, width, fillchar)
def rjust(a, width: int, fillchar: str = ' '): return _map(a, str, _k_rjust, width, fillchar)
def zfill(a, width: int): return _map(a, str, _k_zfill, width)
def expandtabs(a, tabsize: int = 8): return _map(a, str, _k_expandtabs, tabsize)
def replace(a, old: str, new: str, count: int = -1): return _map(a, str, _k_replace, old, new, count)
def multiply(a, i: int): return _map(a, str, _k_mul, i)

def find(a, sub: str, start: int = 0, end: Optional[int] = None): return _map(a, int, _k_find, sub, start, end)
def rfind(a, sub: str, start: int = 0, end: Optional[int] = None): return _map(a, int, _k_rfind, sub, start, end)
def index(a, sub: str, start: int = 0, end: Optional[int] = None): return _map(a, int, _k_index, sub, start, end)
def rindex(a, sub: str, start: int = 0, end: Optional[int] = None): return _map(a, int, _k_rindex, sub, start, end)
def count(a, sub: str, start: int = 0, end: Optional[int] = None): return _map(a, int, _k_count, sub, start, end)
def startswith(a, prefix: str, start: int = 0, end: Optional[int] = None): return _map(a, bool, _k_startswith, prefix, start, end)
def endswith(a, suffix: str, start: int = 0, end: Optional[int] = None): return _map(a, bool, _k_endswith, suffix, start, end)

def slice(a, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None):
    # numpy.strings.slice(a, n) means a[:n]
    if stop is None and step is None and start is not None:
        return _map(a, str, _k_slice, None, start, None)
    return _map(a, str, _k_slice, start, stop, step)

def partition(a, sep: str):
    return (_map(a, str, _k_part0, sep), _map(a, str, _k_part1, sep), _map(a, str, _k_part2, sep))

def rpartition(a, sep: str):
    return (_map(a, str, _k_rpart0, sep), _map(a, str, _k_rpart1, sep), _map(a, str, _k_rpart2, sep))

def add(x1, x2): return _map2(x1, x2, str, _k_add)
def equal(x1, x2): return _map2(x1, x2, bool, _k_eq)
def not_equal(x1, x2): return _map2(x1, x2, bool, _k_ne)
def less(x1, x2): return _map2(x1, x2, bool, _k_lt)
def less_equal(x1, x2): return _map2(x1, x2, bool, _k_le)
def greater(x1, x2): return _map2(x1, x2, bool, _k_gt)
def greater_equal(x1, x2): return _map2(x1, x2, bool, _k_ge)

def _k_encode(s: str, enc: str): return s.encode()
def _k_decode(s: bytes, enc: str): return s.decode()

def encode(a, encoding: str = 'utf-8', errors: str = 'strict'):
    return _map(a, bytes, _k_encode, encoding)

def decode(a, encoding: str = 'utf-8', errors: str = 'strict'):
    x = np.asarray(a)
    flat = x.ravel()
    out = np.empty(flat.shape, dtype=str)
    for i in range(flat.size):
        out[i] = _k_decode(flat[i], encoding)
    return out.reshape(x.shape)

def _k_translate(s: str, table: Dict[int, str], deletechars: str):
    r = List[str]()
    for ch in s:
        if ch in deletechars:
            continue
        o = ord(ch)
        r.append(table[o] if o in table else ch)
    return ''.join(r)

def translate(a, table: Dict[int, str], deletechars: str = ''):
    return _map(a, str, _k_translate, table, deletechars)

def _k_mod(s: str, v): return s % v

def mod(a, values):
    x = _sarr(a)
    flat = x.ravel()
    out = np.empty(flat.shape, dtype=str)
    for i in range(flat.size):
        out[i] = _k_mod(flat[i], values)
    return out.reshape(x.shape)
