# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# String formatting for numpy.ma masked arrays (object-style printing with
# '--' for masked entries, as numpy's _insert_masked_print produces).

def _nest(el: List[str], shp: List[int], lvl: int, start: int, comma: bool, indent: int) -> str:
    if len(shp) == 0:
        return el[0]
    n = shp[lvl]
    if lvl == len(shp) - 1:
        sep = ", " if comma else " "
        return "[" + sep.join(el[start:start + n]) + "]"
    stride = 1
    for k in range(lvl + 1, len(shp)):
        stride *= shp[k]
    parts = List[str]()
    for i in range(n):
        parts.append(_nest(el, shp, lvl + 1, start + i * stride, comma, indent))
    joiner = ("," if comma else "") + "\n" * (len(shp) - 1 - lvl) + " " * (indent + lvl + 1)
    return "[" + joiner.join(parts) + "]"

def _dtype_name(T: type) -> str:
    if T is float:
        return "float64"
    elif T is float32:
        return "float32"
    elif T is int:
        return "int64"
    elif T is i32:
        return "int32"
    elif T is bool:
        return "bool"
    elif T is complex:
        return "complex128"
    else:
        return "object"

def _is_default_dtype(T: type) -> bool:
    return T is float or T is int or T is bool

def _elems(data, mask) -> List[str]:
    fd = data.ravel()
    fm = mask.ravel()
    out = List[str](fd.size)
    for i in range(fd.size):
        if fm[i]:
            out.append("--")
        else:
            out.append(str(fd[i]))
    return out

def _mask_elems(mask) -> List[str]:
    fm = mask.ravel()
    anyf = False
    for i in range(fm.size):
        if not fm[i]:
            anyf = True
    out = List[str](fm.size)
    for i in range(fm.size):
        if fm[i]:
            out.append(" True" if anyf else "True")
        else:
            out.append("False")
    return out

def ma_str(data, mask) -> str:
    return _nest(_elems(data, mask), list(data.shape), 0, 0, False, 0)

def ma_repr(data, mask, nomask: bool, fill_str: str) -> str:
    shp = list(data.shape)
    T = data.dtype
    allm = data.size > 0
    fm = mask.ravel()
    for i in range(fm.size):
        if not fm[i]:
            allm = False
    show_dtype = (not _is_default_dtype(T)) or allm
    if data.ndim <= 1:
        s = "masked_array(data=" + _nest(_elems(data, mask), shp, 0, 0, True, 18) + ",\n"
        s += " " * 13 + "mask=" + ("False" if nomask else _nest(_mask_elems(mask), shp, 0, 0, True, 18)) + ",\n"
        s += " " * 7 + "fill_value=" + fill_str
        if show_dtype:
            s += ",\n" + " " * 12 + "dtype=" + _dtype_name(T)
        return s + ")"
    s = "masked_array(\n  data=" + _nest(_elems(data, mask), shp, 0, 0, True, 7) + ",\n"
    s += "  mask=" + ("False" if nomask else _nest(_mask_elems(mask), shp, 0, 0, True, 7)) + ",\n"
    s += "  fill_value=" + fill_str
    if show_dtype:
        s += ",\n  dtype=" + _dtype_name(T)
    return s + ")"
