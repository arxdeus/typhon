#%% list_unbound,barebones
a = []
print(a, a.__class__.__name__)
#: [] List[NoneType]

d = {}
print(d, d.__class__.__name__)
#: {} OrderedDict[NoneType,NoneType]

s = set()
print(s, s.__class__.__name__)
#: set() Set[NoneType]

#%% star_err,barebones
a = (1, 2, 3)
z = *a #! unexpected star expression

#%% list,barebones
a = [4, 5, 6]
print a #: [4, 5, 6]
b = [1, 2, 3, *a]
print b #: [1, 2, 3, 4, 5, 6]

#%% set,barebones
gs = {1.12}
print gs #: {1.12}
fs = {1, 2, 3, 1, 2, 3}
gs.add(1.12)
gs.add(1.13)
print fs, gs #: {1, 2, 3} {1.13, 1.12}
print {*fs, 5, *fs} #: {1, 2, 3, 5}

#%% dict,barebones
gd = {1: 'jedan', 2: 'dva', 2: 'two', 3: 'tri'}
fd = {}
fd['jedan'] = 1
fd['dva'] = 2
print gd, fd #: {1: 'jedan', 2: 'two', 3: 'tri'} {'jedan': 1, 'dva': 2}

#%% dictionary_unpacking
# **mapping entries retain order, overwrite semantics, and evaluation count.
source = {'a': 1, 'b': 2, 'c': 3}
copied = {**source}
copied = {**copied, 'x': 99, **{'a': 42}}
assert copied == {'a': 42, 'b': 2, 'c': 3, 'x': 99}
assert source == {'a': 1, 'b': 2, 'c': 3}
assert list(copied.keys()) == ['a', 'b', 'c', 'x']
empty: dict[str, int] = {}
assert {**empty, **source, **empty} == source

def record(events, label, value):
    events.append(label)
    return value
events: list[int] = []
combined = {**record(events, 1, {'a': 1}),
            record(events, 2, 'b'): record(events, 3, 2),
            **record(events, 4, {'a': 3})}
assert combined == {'a': 3, 'b': 2} and events == [1, 2, 3, 4]
print('dictionary unpacking ok')  #: dictionary unpacking ok

#%% union_set_layout
from typing import Union

Choice = Union[int, float]
# The declared union determines storage even when all initial values are ints.
homogeneous: set[Choice] = {1, 2, 3}
homogeneous.add(4.5)
assert type(homogeneous) is set[Choice]
assert sorted(float(value) for value in homogeneous) == [1.0, 2.0, 3.0, 4.5]
mixed: set[Choice] = {1, 2.5, 3}
assert sorted(float(value) for value in mixed) == [1.0, 2.5, 3.0]
expanded: set[Choice] = {0, *[1, 2], *[3.5]}
assert sorted(float(value) for value in expanded) == [0.0, 1.0, 2.0, 3.5]
print('union set layout ok')  #: union set layout ok

#%% union_dict_layout
from typing import Union

Choice = Union[int, str]
Key = Union[int, float]
# Keys and values need their declared representation before the first insertion.
values: dict[str, Choice] = {'one': 1, 'two': 'second'}
assert type(values) is dict[str, Choice]
assert isinstance(values['one'], int) and values['one'] == 1
assert isinstance(values['two'], str) and values['two'] == 'second'
keys: dict[Key, str] = {1: 'first', 2.5: 'second'}
assert keys[1] == 'first' and keys[2.5] == 'second'
both: dict[Key, Choice] = {1: 'first', 2.5: 2}
assert both[1] == 'first' and both[2.5] == 2
expanded: dict[Key, Choice] = {**{1: 'first'}, **{2.5: 2}, 3: 'third'}
assert expanded[1] == 'first' and expanded[2.5] == 2 and expanded[3] == 'third'
empty: dict[Key, Choice] = {}
empty[1] = 'first'
empty[2.5] = 2
assert empty[1] == 'first' and empty[2.5] == 2
print('union dict layout ok')  #: union dict layout ok

#%% union_collection_common_type
from typing import Union, Optional

Choice = Union[int, str]
Key = Union[int, float]
# A union key must not disable lowest-common-type inference for values, or vice versa.
values: dict[Key, float] = {1: 1, 2.5: 2.5}
reversed_values: dict[Key, float] = {2.5: 2.5, 1: 1}
assert type(values) is dict[Key, float] and type(reversed_values) is type(values)
assert values[1] == 1.0 and values[2.5] == 2.5
keys: dict[float, Choice] = {1: 'first', 2.5: 2}
assert type(keys) is dict[float, Choice]
assert keys[1.0] == 'first' and keys[2.5] == 2
nullable: dict[Key, Optional[int]] = {1: None, 2.5: 2}
assert nullable[1] is None and nullable[2.5] == 2

# Conversely, lctyp must not promote members of an explicitly declared union.
Number = Union[int, float]
numbers: list[Number] = [1, 2.5]
assert isinstance(numbers[0], int) and isinstance(numbers[1], float)
number_set: set[Number] = {1, 2.5}
assert sum(isinstance(value, int) for value in number_set) == 1
number_dict: dict[str, Number] = {'one': 1, 'two': 2.5}
assert isinstance(number_dict['one'], int) and isinstance(number_dict['two'], float)
print('union collection common type ok')  #: union collection common type ok



#%% comprehension,barebones
l = [(i, j, f'i{i}/{j}')
     for i in range(50) if i % 2 == 0 if i % 3 == 0
     for j in range(2) if j == 1]
print l #: [(0, 1, 'i0/1'), (6, 1, 'i6/1'), (12, 1, 'i12/1'), (18, 1, 'i18/1'), (24, 1, 'i24/1'), (30, 1, 'i30/1'), (36, 1, 'i36/1'), (42, 1, 'i42/1'), (48, 1, 'i48/1')]

s = {i%3 for i in range(20)}
print s #: {0, 1, 2}

d = {i: j for i in range(10) if i < 1 for j in range(10)}
print d  #: {0: 9}

t = 's'
x = {t: lambda x: x * t for t in range(5)}
print(x[3](10))  #: 40
print(t)  #: s

#%% comprehension_opt,barebones
@extend
class List:
    def __init__(self, cap: int):
        print 'optimize', cap
        self._ptr = Ptr[T](cap)
        self._len = 0
        self._cap = cap
def foo():
    yield 0
    yield 1
    yield 2
print [i for i in range(3)] #: optimize 3
#: [0, 1, 2]
print [i for i in foo()] #: [0, 1, 2]
print [i for i in range(3) if i%2 == 0] #: [0, 2]
print [i + j for i in range(1) for j in range(1)] #: [0]
print {i for i in range(3)} #: {0, 1, 2}

#%% generator,barebones
z = 3
g = (e for e in range(20) if e % z == 1)
print str(g)[:13] #: <generator at
print list(g) #: [1, 4, 7, 10, 13, 16, 19]

g1 = (a for a in range(3))
print list(g1) #: [0, 1, 2]
g2 = (a for a in range(z + 1))
print list(g2) #: [0, 1, 2, 3]

def nest(z):
    g1 = (a for a in range(3))
    print list(g1) #: [0, 1, 2]
    g2 = (a for a in range(z + 1))
    print list(g2) #: [0, 1, 2, 3, 4]
nest(4)

#%% tuple_generator,barebones
a = (1, 2)
b = ('f', 'g')
print a, b #: (1, 2) ('f', 'g')
c = (*a, True, *b)
print c #: (1, 2, True, 'f', 'g')
print a + b + c #: (1, 2, 'f', 'g', 1, 2, True, 'f', 'g')
print () + (1, ) + ('a', 'b') #: (1, 'a', 'b')

t = tuple(i+1 for i in (1,2,3))
print t #: (2, 3, 4)
print tuple((j, i) for i, j in ((1, 'a'), (2, 'b'), (3, 'c')))
#: (('a', 1), ('b', 2), ('c', 3))

#%% empty_tuple,barebones
T = type(())  # only errors with empty tuple type
p = Ptr[T](cobj())
print p.__class__.__name__  #: Ptr[Tuple]

print [a for a in ()]  #: []

def foo(*args):
    return [a for a in args]
args, result = ((), [()])
print list(foo(*args))  #: []
print result  #: [()]

#%% collection_common_type,barebones
l = [1, 2, 3]
print(l, l.__class__.__name__)
#: [1, 2, 3] List[int]

l = [1.1, 2, 3]
print(l, l.__class__.__name__)
#: [1.1, 2.0, 3.0] List[float]

l = [1, 2, 3.3]
print(l, l.__class__.__name__)
#: [1.0, 2.0, 3.3] List[float]

l = [1, None]
print(l, l.__class__.__name__)
#: [1, None] List[Optional[int]]

l = [None, 2.2]
print(l, l.__class__.__name__)
#: [None, 2.2] List[Optional[float]]

class A:
    def __repr__(self): return 'A'
class B(A):
    def __repr__(self): return 'B'
class C(B):
    def __repr__(self): return 'C'
class D(A):
    def __repr__(self): return 'D'

l = [A(), B(), C(), D()]
print(l, l.__class__.__name__)
#: [A, B, C, D] List[A]

l = [D(), C(), B(), A()]
print(l, l.__class__.__name__)
#: [D, C, B, A] List[A]

l = [C(), B()]
print(l, l.__class__.__name__)
#: [C, B] List[B]

l = [C(), A(), B()]
print(l, l.__class__.__name__)
#: [C, A, B] List[A]

l = [None, *[1, 2], None]
print(l, l.__class__.__name__)
#: [None, 1, 2, None] List[Optional[int]]

# l = [C(), D(), B()] # does not work (correct behaviour)
# print(l, l.__class__.__name__)

d = {1: None, 2.2: 's'}
print(d, d.__class__.__name__)
#: {1.0: None, 2.2: 's'} OrderedDict[float,Optional[str]]

#%% comprehension_opt_clone
import sys
z = [i for i in sys.argv]


#%% union_infer_heterogeneous_literals
import json
from typing import Union, List

def stage():
    return {"n": 3, "loss": [0.1, 0.2], "acc": 1.0, "pred": [0, 1], "name": "x"}
d = stage()
assert type(d) is Dict[str, Union[int, float, List[float], List[int], str]]
print(float(d["acc"]), d["n"], d["loss"], d["name"])  #: 1.0 3 [0.1, 0.2] x
print(isinstance(d["pred"], List[int]), d["n"] == 3)  #: True True
print(json.dumps(d))  #: {"n": 3, "loss": [0.1, 0.2], "acc": 1.0, "pred": [0, 1], "name": "x"}
for k, v in d.items():
    if k == "acc":
        print(k, v)  #: acc 1.0

keys = {1: "a", "b": "c"}
assert type(keys) is Dict[Union[int, str], str]
print(keys[1], keys["b"])  #: a c

l = [1, "a", 2]
assert type(l) is List[Union[int, str]]
print(l)  #: [1, 'a', 2]
s = {1, "a"}
assert type(s) is Set[Union[int, str]]
print(len(s), 1 in s, "a" in s)  #: 2 True True

nested = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
assert type(nested) is List[Dict[str, Union[int, str]]]
print(nested[1]["b"])  #: y

# Conservative: implicit conversions keep working.
assert type([1, 2.0]) is List[float]
assert type([None, 1]) is List[Optional[int]]
assert type([1, None]) is List[Optional[int]]
class Base:
    pass
class Derived(Base):
    pass
assert type([Derived(), Base()]) is List[Base]
assert type({"a": 1, "b": 2.5}) is Dict[str, float]

# Annotation wins.
ann: Dict[str, Union[int, str, float]] = {"a": 1, "b": "x"}
assert type(ann) is Dict[str, Union[int, str, float]]
print('union infer ok')  #: union infer ok

#%% union_infer_declared_non_union_error
x: List[int] = [1, "a"]  #! 'str' does not match expected type 'int'

#%% union_infer_empty_literal_inserts
import json
from typing import Union, List

def sklearn_like(names):
    out = {}
    for k in names:
        out[k + "_scores"] = [float(i) for i in range(2)]
    out["acc"] = 0.5
    return out
d = sklearn_like(["a", "b"])
assert type(d) is Dict[str, Union[List[float], float]]
print(json.dumps(d))  #: {"a_scores": [0.0, 1.0], "b_scores": [0.0, 1.0], "acc": 0.5}
print(float(d["acc"]))  #: 0.5

def appends():
    a = []
    b = []
    a.append(1)
    b.append(2)
    a.append("z")
    k = {}
    k[1] = "x"
    k["y"] = "z"
    e = {}
    e.setdefault("a", 1)
    e.setdefault("b", "s")
    return a, b, k, e
a, b, k, e = appends()
assert type(a) is List[Union[int, str]]
assert type(b) is List[int]
assert type(k) is Dict[Union[int, str], str]
print(a, b, k, e)  #: [1, 'z'] [2] {1: 'x', 'y': 'z'} {'a': 1, 'b': 's'}

def conservative():
    f = []
    f.append(1)
    f.append(2.5)
    class Base:
        pass
    class Derived(Base):
        pass
    c = []
    c.append(Base())
    c.append(Derived())
    return f, c
f, c = conservative()
assert type(f) is List[float]
print(f, len(c))  #: [1.0, 2.5] 2

#%% union_infer_empty_literal_annotated_error
x: List[int] = []
x.append("s")  #! 'str' does not match expected type 'int'

#%% union_new_generic_optional_field
# Regression: Union._new of a generic class with an Optional field must not treat
# the value as a one-member Union (LLVM "extractvalue operand must be aggregate").
class P[S]:
    s: S
    m: Optional[str]
    def __init__(self, s: S, m: Optional[str] = None):
        self.s = s
        self.m = m
u: Union[P[int], P[str]] = P(1)
print(u.s)  #: 1
d = {"a": P(1), "b": P("x", "m")}
for key, v in d.items():
    print(key, v.s, v.m)
#: a 1 None
#: b x m

#%% union_builtin_conversions
u: List[Union[int, float, List[float], str]] = [3, 2.7, [1.0], "12", List[float]()]
print(int(u[0]), int(u[1]), int(u[3]))  #: 3 2 12
print(float(u[0]), float(u[1]), float(u[3]))  #: 3.0 2.7 12.0
print(str(u[0]), str(u[2]), str(u[3]))  #: 3 [1.0] 12
print(bool(u[0]), bool(u[2]), bool(u[4]))  #: True True False
d = {"n": 3, "acc": 0.0, "loss": [0.5]}
print(int(d["n"]), bool(d["acc"]), float(d["acc"]))  #: 3 False 0.0
try:
    int(u[2])
except TypeError:
    print('no int')  #: no int

#%% tuple_literal_common_type,barebones
t: Tuple[float, str] = (1, 'a')
print t  #: (1.0, 'a')
u: Tuple[Optional[float], float] = (None, 2.0)
print u  #: (None, 2.0)
print [(1, 'a'), (2.0, 'b')]  #: [(1.0, 'a'), (2.0, 'b')]
print [(None, 2.0), (0.5, None)]  #: [(None, 2.0), (0.5, None)]
def f(p: Tuple[float, Optional[int]]): return p
print f((1, None))  #: (1.0, None)

#%% optional_generator_next,barebones
class S:
    gen: Optional[Generator[int]]
    def __init__(self):
        self.gen = None
    def g(self, n: int):
        for i in range(n):
            yield i
    def step(self) -> bool:
        if self.gen is None:
            self.gen = self.g(2)
        try:
            next(self.gen)
            return True
        except StopIteration:
            return False
s = S()
print s.step(), s.step(), s.step()  #: True True False

#%% generator_over_heterogeneous_tuple,barebones
def emit(*args):
    return ' '.join(str(a) for a in args)
print emit(1, [2.0], 3.5)  #: 1 [2.0] 3.5
print list(str(a) for a in (1, 'x')), sum(float(a) for a in (1, 2.5))  #: ['1', 'x'] 3.5

#%% list_numeric_promotion,barebones
a = [1., 2.+1j]
print a  #: [(1+0j), (2+1j)]
b = [[1, 2], [3.5, 4]]
print b  #: [[1.0, 2.0], [3.5, 4.0]]
c = [[1., 2.+1j], [3., 4.]]
print c  #: [[(1+0j), (2+1j)], [(3+0j), (4+0j)]]

#%% dict_literal_generic_values,barebones
def mk(T: type):
    return T()
def fdl(x, T: type):
    return {'a': mk(T), 'b': x}, [[], [x]], {'k': [], 'j': [x]}
print fdl(1, int)  #: ({'a': 0, 'b': 1}, [[], [1]], {'k': [], 'j': [1]})

#%% list_optional_mixed_numeric,barebones
a = [None, 1, 2.5]
print(a, a.__class__.__name__)
#: [None, 1, 2.5] List[Optional[float]]
b = [1, None, 2]
print(b, b.__class__.__name__)
#: [1, None, 2] List[Optional[int]]

#%% list_optional_heterogeneous,barebones
x = [None, True, 1]
print(len(x), x[0] is None)
#: 3 True
