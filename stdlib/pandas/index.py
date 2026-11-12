# Ported from pandas 3.1.1 (BSD-3-Clause)

# Index: an immutable labelled axis backed by a Column.

from .scalar import Scalar, K_FLOAT, K_INT, K_BOOL, K_STR, K_NONE, NA_STR, _isnan
from .core import Column, _dtype_name

class Index:
    _col: Column
    name: Optional[str]
    _is_range: bool
    _map: Optional[Dict[Scalar, int]]

    def __init__(self, data, name: Optional[str] = None):
        if isinstance(data, Index):
            self._col = data._col.copy()
            self._is_range = data._is_range
        else:
            self._col = Column.of(data)
            self._is_range = False
        self.name = name
        self._map = None

    @staticmethod
    def _wrap(col: Column, name: Optional[str] = None, is_range: bool = False) -> Index:
        idx = Index.__new__()
        idx._col = col
        idx.name = name
        idx._is_range = is_range
        idx._map = None
        return idx

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
        return Index._wrap(self._col.copy(), self.name, self._is_range)

    def take(self, idx: List[int]) -> Index:
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
        return Index._wrap(self._col.extend(other._col), self.name if self.name == other.name else None)

    def rename(self, name: Optional[str]) -> Index:
        return Index._wrap(self._col.copy(), name, self._is_range)

    def _fmt(self, k: int) -> str:
        return str(self._col.get(k))

    def _labels_str(self) -> List[str]:
        return [self._fmt(k) for k in range(len(self))]

    def __repr__(self) -> str:
        if self._is_range:
            n = len(self)
            nm = f", name='{self.name}'" if self.name is not None else ""
            return f"RangeIndex(start=0, stop={n}, step=1{nm})"
        items = ", ".join([repr(self._col.get(k)) for k in range(len(self))])
        nm = f", name='{self.name}'" if self.name is not None else ""
        return f"Index([{items}], dtype='{self.dtype}'{nm})"

    def __str__(self) -> str:
        return self.__repr__()
