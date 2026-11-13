# Ported from pandas 3.1.1 (BSD-3-Clause)

# Index: an immutable labelled axis backed by a Column.

from .scalar import Scalar, K_FLOAT, K_INT, K_BOOL, K_STR, K_NONE, NA_STR, _isnan
from .core import Column, _dtype_name
import internal.static as static

class Index:
    _col: Column
    name: Optional[str]
    _is_range: bool
    _map: Optional[Dict[Scalar, int]]
    _lv: List[Column]                 # level columns of a MultiIndex (empty otherwise)
    _lnames: List[Optional[str]]

    def __init__(self, data, name: Optional[str] = None):
        if isinstance(data, Index):
            self._col = data._col.copy()
            self._is_range = data._is_range
        else:
            self._col = Column.of(data)
            self._is_range = False
        self.name = name
        self._map = None
        self._lv = List[Column]()
        self._lnames = List[Optional[str]]()

    @staticmethod
    def _wrap(col: Column, name: Optional[str] = None, is_range: bool = False) -> Index:
        idx = Index.__new__()
        idx._col = col
        idx.name = name
        idx._is_range = is_range
        idx._map = None
        idx._lv = List[Column]()
        idx._lnames = List[Optional[str]]()
        return idx

    @staticmethod
    def _multi(levels: List[Column], names: List[Optional[str]]) -> Index:
        # MultiIndex: per-level columns (one entry per row) plus a composite
        # label column used for lookups.
        n = len(levels[0])
        lab = List[str](n)
        for i in range(n):
            parts = List[str]()
            for c in levels:
                parts.append(str(c.get(i)))
            lab.append("\x1f".join(parts))
        idx = Index._wrap(Column.strs(lab), None, False)
        idx._lv = levels
        idx._lnames = names
        return idx

    @property
    def nlevels(self) -> int:
        return len(self._lv) if len(self._lv) > 0 else 1

    @property
    def names(self) -> List[Optional[str]]:
        if len(self._lv) > 0:
            return self._lnames
        return [self.name]

    def get_level_values(self, level: int) -> Index:
        if len(self._lv) == 0:
            return self.copy()
        return Index._wrap(self._lv[level].copy(), self._lnames[level], False)

    def to_tuples(self) -> List[List[Scalar]]:
        out = List[List[Scalar]]()
        for i in range(len(self)):
            if len(self._lv) == 0:
                out.append([self._col.get(i)])
            else:
                out.append([c.get(i) for c in self._lv])
        return out

    @staticmethod
    def range(n: int, name: Optional[str] = None) -> Index:
        return Index._wrap(Column.ints(list(range(n))), name, True)

    @staticmethod
    def strs(labels: List[str], name: Optional[str] = None) -> Index:
        return Index._wrap(Column.strs(labels), name, False)

    def __len__(self) -> int:
        return len(self._col)

    @property
    def size(self) -> int:
        return len(self._col)

    @property
    def dtype(self) -> str:
        return "int64" if self._is_range else self._col.dtype

    def __getitem__(self, k: int) -> Scalar:
        n = len(self)
        if k < 0:
            k += n
        if k < 0 or k >= n:
            raise IndexError(f"index {k} is out of bounds for axis 0 with size {n}")
        return self._col.get(k)

    def __getitem__(self, s: Slice) -> Index:
        start, stop, step, _ = s.adjust_indices(len(self))
        return self.take(list(range(start, stop, step)))

    def __iter__(self):
        for k in range(len(self)):
            yield self._col.get(k)

    def __contains__(self, key) -> bool:
        return self.get_loc_opt(Scalar.of(key)) >= 0

    def tolist(self) -> List[Scalar]:
        return self._col.scalars()

    def to_list(self) -> List[Scalar]:
        return self.tolist()

    def copy(self) -> Index:
        if len(self._lv) > 0:
            return Index._multi([c.copy() for c in self._lv], self._lnames.copy())
        return Index._wrap(self._col.copy(), self.name, self._is_range)

    def take(self, idx: List[int]) -> Index:
        if len(self._lv) > 0:
            return Index._multi([c.take(idx) for c in self._lv], self._lnames.copy())
        is_range = False
        if self._is_range:
            is_range = True
            for k in range(len(idx)):
                if idx[k] != k:
                    is_range = False
                    break
            if is_range and len(idx) > 0 and self._col.i[0] != 0:
                is_range = False
        return Index._wrap(self._col.take(idx), self.name, is_range)

    def _lookup(self) -> Dict[Scalar, int]:
        if self._map is None:
            m = Dict[Scalar, int]()
            for k in range(len(self)):
                key = self._col.get(k)
                if key not in m:
                    m[key] = k
            self._map = m
        return self._map

    def get_loc_opt(self, key: Scalar) -> int:
        m = self._lookup()
        return m.get(key, -1)

    def get_loc(self, key) -> int:
        k = self.get_loc_opt(Scalar.of(key))
        if k < 0:
            raise KeyError(str(Scalar.of(key)))
        return k

    def get_locs(self, key) -> List[int]:
        # all positions of a (possibly duplicated) label
        sk = Scalar.of(key)
        res = List[int]()
        for k in range(len(self)):
            if self._col.get(k) == sk:
                res.append(k)
        if len(res) == 0:
            raise KeyError(str(sk))
        return res

    def equals(self, other: Index) -> bool:
        if len(self) != len(other):
            return False
        for k in range(len(self)):
            if self._col.get(k) != other._col.get(k):
                if not (self._col.isna_at(k) and other._col.isna_at(k)):
                    return False
        return True

    def __eq__(self, other: Index) -> bool:
        return self.equals(other)

    def is_unique(self) -> bool:
        return len(self._lookup()) == len(self)

    def append(self, other: Index) -> Index:
        if len(self._lv) > 0 and len(other._lv) == len(self._lv):
            return Index._multi([self._lv[j].extend(other._lv[j]) for j in range(len(self._lv))], self._lnames.copy())
        return Index._wrap(self._col.extend(other._col), self.name if self.name == other.name else None)

    def rename(self, name: Optional[str]) -> Index:
        return Index._wrap(self._col.copy(), name, self._is_range)

    def _fmt(self, k: int) -> str:
        if len(self._lv) > 0:
            return "(" + ", ".join([repr(c.get(k)) for c in self._lv]) + ")"
        return str(self._col.get(k))

    def _labels_str(self) -> List[str]:
        return [self._fmt(k) for k in range(len(self))]

    def __repr__(self) -> str:
        if len(self._lv) > 0:
            items = ",\n            ".join([self._fmt(k) for k in range(len(self))])
            nm = ", ".join([("'" + x.__val__() + "'") if x is not None else "None" for x in self._lnames])
            return f"MultiIndex([{items}],\n           names=[{nm}])"
        if self._is_range:
            n = len(self)
            nm = f", name='{self.name}'" if self.name is not None else ""
            return f"RangeIndex(start=0, stop={n}, step=1{nm})"
        items = ", ".join([repr(self._col.get(k)) for k in range(len(self))])
        nm = f", name='{self.name}'" if self.name is not None else ""
        return f"Index([{items}], dtype='{self.dtype}'{nm})"

    def __str__(self) -> str:
        return self.__repr__()

def _names(names: Optional[List[str]], w: int) -> List[Optional[str]]:
    out = List[Optional[str]]()
    for j in range(w):
        if names is None:
            out.append(None)
        else:
            out.append(names.__val__()[j])
    return out

class MultiIndex:
    @staticmethod
    def from_tuples(tuples, names: Optional[List[str]] = None) -> Index:
        lv = List[Column]()
        for j in static.range(static.len(tuples[0])):
            lv.append(Column.of([t[j] for t in tuples]))
        nm = _names(names, len(lv))
        return Index._multi(lv, nm)

    @staticmethod
    def from_arrays(arrays, names: Optional[List[str]] = None) -> Index:
        lv = List[Column]()
        for a in arrays:
            lv.append(Column.of(a))
        nm = _names(names, len(lv))
        return Index._multi(lv, nm)

    @staticmethod
    def from_product(iterables, names: Optional[List[str]] = None) -> Index:
        cols = List[Column]()
        if isinstance(iterables, Tuple):
            for it in iterables:
                cols.append(Column.of(it))
        else:
            for it in iterables:
                cols.append(Column.of(it))
        total = 1
        for c in cols:
            total *= len(c)
        lv = List[Column]()
        rep = total
        for c in cols:
            rep //= len(c)
            outer = total // (rep * len(c))
            pos = List[int](total)
            for _ in range(outer):
                for v in range(len(c)):
                    for _ in range(rep):
                        pos.append(v)
            lv.append(c.take(pos))
        nm = _names(names, len(lv))
        return Index._multi(lv, nm)
