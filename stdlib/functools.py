
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
