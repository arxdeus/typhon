# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# Scalar type hierarchy names (generic, number, integer, ...) and the dtype
# predicates issubdtype / can_cast / result_type / promote_types / isdtype.
# Concrete scalar types are the native types; the hierarchy classes are
# marker types used as the second argument of issubdtype / isdtype.

import internal.static as static
from .ndarray import ndarray

class generic:
    pass

class number(generic):
    pass

class integer(number):
    pass

class signedinteger(integer):
    pass

class unsignedinteger(integer):
    pass

class inexact(number):
    pass

class floating(inexact):
    pass

class complexfloating(inexact):
    pass

class flexible(generic):
    pass

class character(flexible):
    pass

# category of a concrete scalar type: 1 bool, 2 signed, 3 unsigned,
# 4 floating, 5 complex, 6 str, 0 other
def _cat(T: type) -> Literal[int]:
    if T is bool:
        return 1
    elif T is int or T is Int[8] or T is Int[16] or T is Int[32]:
        return 2
    elif T is UInt[8] or T is UInt[16] or T is UInt[32] or T is UInt[64]:
        return 3
    elif T is float or T is float32 or T is float16 or T is float128:
        return 4
    elif T is complex or T is complex64:
        return 5
    elif T is str:
        return 6
    else:
        return 0

# is category c a member of hierarchy class H?
def _member(c: Literal[int], H: type) -> Literal[bool]:
    if H is generic:
        return True
    elif H is number:
        return c >= 2 and c <= 5
    elif H is integer:
        return c == 2 or c == 3
    elif H is signedinteger:
        return c == 2
    elif H is unsignedinteger:
        return c == 3
    elif H is inexact:
        return c == 4 or c == 5
    elif H is floating:
        return c == 4
    elif H is complexfloating:
        return c == 5
    elif H is flexible or H is character:
        return c == 6
    else:
        return False

def _is_hier(H: type) -> Literal[bool]:
    return (H is generic or H is number or H is integer or H is signedinteger
            or H is unsignedinteger or H is inexact or H is floating
            or H is complexfloating or H is flexible or H is character)

def _arr_of(T: type):
    from .routines import zeros
    return zeros(1, T)

def promote_types(a: type, b: type):
    x = _arr_of(a) + _arr_of(b)
    return type(x[0])

def result_type(*args):
    if static.len(args) == 0:
        compile_error("at least one array or dtype is required")
    return _rt(*args)

def _arg_type(x):
    if isinstance(x, ndarray):
        return x.dtype
    elif isinstance(x, (int, float, bool, complex, str)):
        return type(x)
    else:
        return x

def _rt(first, *rest):
    T = _arg_type(first)
    if static.len(rest) == 0:
        return T
    else:
        return _rt2(T, *rest)

def _rt2(acc: type, nxt, *rest):
    U = promote_types(acc, _arg_type(nxt))
    if static.len(rest) == 0:
        return U
    else:
        return _rt2(U, *rest)

def _cast_level(casting: str) -> int:
    if casting == 'no' or casting == 'equiv':
        return 0
    elif casting == 'safe':
        return 1
    elif casting == 'same_kind':
        return 2
    elif casting == 'unsafe':
        return 3
    raise ValueError("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'")

def _kind_rank(c: int) -> int:
    # bool < unsigned < signed < float < complex
    if c == 1:
        return 0
    elif c == 3:
        return 1
    elif c == 2:
        return 2
    elif c == 4:
        return 3
    elif c == 5:
        return 4
    return 5

def can_cast(from_: type, to: type, casting: str = 'safe') -> bool:
    lvl = _cast_level(casting)
    if from_ is to:
        return True
    if lvl == 0:
        return False
    if lvl == 3:
        return True
    if promote_types(from_, to) is to:
        return True
    if lvl == 2:
        cf = _cat(from_)
        ct = _cat(to)
        if cf == 0 or ct == 0:
            return False
        return _kind_rank(cf) <= _kind_rank(ct)
    return False

def issubdtype(arg1: type, arg2: type) -> bool:
    if static.len(arg1.__name__) < 0:
        return False
    return _issub(arg1, arg2)

def _issub(a: type, b: type) -> Literal[bool]:
    if _is_hier(b):
        return _member(_cat(a), b)
    else:
        return a is b

def isdtype(dtype: type, kind) -> bool:
    if isinstance(kind, tuple):
        for k in kind:
            if isdtype(dtype, k):
                return True
        return False
    elif isinstance(kind, str):
        c = _cat(dtype)
        if kind == 'bool':
            return c == 1
        elif kind == 'signed integer':
            return c == 2
        elif kind == 'unsigned integer':
            return c == 3
        elif kind == 'integral':
            return c == 2 or c == 3
        elif kind == 'real floating':
            return c == 4
        elif kind == 'complex floating':
            return c == 5
        elif kind == 'numeric':
            return c >= 2 and c <= 5
        raise ValueError("kind argument is a string, but '" + kind + "' is not a known kind name.")
    else:
        return _issub(dtype, kind)

def min_scalar_type(a) -> str:
    # runtime values cannot select a static type, so the dtype name is returned
    if isinstance(a, bool):
        return 'bool'
    elif isinstance(a, int):
        if a >= 0:
            if a < 256:
                return 'uint8'
            elif a < 65536:
                return 'uint16'
            elif a < 4294967296:
                return 'uint32'
            return 'uint64'
        else:
            if a >= -128:
                return 'int8'
            elif a >= -32768:
                return 'int16'
            elif a >= -2147483648:
                return 'int32'
            return 'int64'
    elif isinstance(a, float):
        return 'float16'
    else:
        return type(a).__name__

def common_type(*arrays):
    return _ct(*arrays)

def _ct(first, *rest):
    T = _arg_type(first)
    U = float if _cat(T) != 5 and _cat(T) != 4 else T
    if _cat(T) == 4 and T is float16:
        U = float16
    if static.len(rest) == 0:
        return U
    else:
        return _ct2(U, *rest)

def _ct2(acc: type, nxt, *rest):
    T = _arg_type(nxt)
    U = promote_types(acc, T if (_cat(T) == 4 or _cat(T) == 5) else float)
    if static.len(rest) == 0:
        return U
    else:
        return _ct2(U, *rest)

def mintypecode(typechars, typeset = 'GDFgdf', default = 'd') -> str:
    best = ''
    rank = 100
    for c in typechars:
        k = typeset.find(c)
        if k >= 0 and k < rank:
            rank = k
            best = c
    if rank == 100:
        return default
    ch = best
    if 'F' in typechars and 'd' in typechars and ('F' in typeset or 'D' in typeset):
        return 'D'
    return ch

def typename(char: str) -> str:
    if char == '?': return 'bool'
    elif char == 'b': return 'signed char'
    elif char == 'B': return 'unsigned char'
    elif char == 'h': return 'short'
    elif char == 'H': return 'unsigned short'
    elif char == 'i': return 'integer'
    elif char == 'I': return 'unsigned integer'
    elif char == 'l': return 'long integer'
    elif char == 'L': return 'unsigned long integer'
    elif char == 'q': return 'long long integer'
    elif char == 'Q': return 'unsigned long long integer'
    elif char == 'f': return 'single precision'
    elif char == 'd': return 'double precision'
    elif char == 'g': return 'long precision'
    elif char == 'F': return 'complex single precision'
    elif char == 'D': return 'complex double precision'
    elif char == 'G': return 'complex long double precision'
    elif char == 'S': return 'string'
    elif char == 'U': return 'unicode'
    elif char == 'V': return 'void'
    elif char == 'O': return 'object'
    raise ValueError("unknown type character")

typecodes = {
    'Character': 'c', 'Integer': 'bhilqnp', 'UnsignedInteger': 'BHILQNP',
    'Float': 'efdg', 'Complex': 'FDG', 'AllInteger': 'bBhHiIlLqQnNpP',
    'AllFloat': 'efdgFDG', 'Datetime': 'Mm', 'All': '?bhilqnpBHILQNPefdgFDGSUVOMm'
}
little_endian = True
