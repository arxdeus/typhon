
@__generated__
@no_arg_reorder
def partial():  # internal
    pass

@tuple
class _CmpKey:
    obj: T
    cmp: F
    T: type
    F: type

    def __hash__(self) -> int:
        compile_error("unhashable type: '_CmpKey'")

    def __lt__(self, other):
        return self.cmp(self.obj, other.obj) < 0

    def __le__(self, other):
        return self.cmp(self.obj, other.obj) <= 0

    def __eq__(self, other):
        return self.cmp(self.obj, other.obj) == 0

    def __ne__(self, other):
        return self.cmp(self.obj, other.obj) != 0

    def __gt__(self, other):
        return self.cmp(self.obj, other.obj) > 0

    def __ge__(self, other):
        return self.cmp(self.obj, other.obj) >= 0


def cmp_to_key(mycmp):
    return lambda obj: _CmpKey(obj, mycmp)


def reduce(function, iterable, initial):
    acc = initial
    for x in iterable:
        acc = function(acc, x)
    return acc

@overload
def reduce(function, iterable):
    first = True
    acc = None
    for x in iterable:
        if first:
            acc = x
            first = False
        else:
            acc = function(acc, x)
    if first:
        raise TypeError("reduce() of empty iterable with no initial value")
    return acc

def lru_cache(maxsize: Optional[int] = 128, typed: bool = False):
    # Unbounded memoization keyed on the positional-argument tuple. The cache dict's
    # key/value types are inferred from its use inside the wrapper closure.
    def decorating_function(user_function):
        return _cache_wrapper(user_function)
    return decorating_function

def _cache_wrapper(user_function):
    d = {}
    def wrapper(*args):
        if args in d:
            return d[args]
        r = user_function(*args)
        d[args] = r
        return r
    return wrapper

def cache(user_function):
    return _cache_wrapper(user_function)
