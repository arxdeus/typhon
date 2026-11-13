# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# DictVectorizer over a list of dicts (str -> number, or str -> str for
# one-hot features "f=v"). sparse=True returns a scipy.sparse CSR array.
# Typhon: `sparse` is a static parameter, so DictVectorizer(...) is a factory
# returning _DictVectorizer[sparse].

import numpy as np

class _DictVectorizer[SP: Literal[bool]]:
    separator: str
    sort: bool
    feature_names_: List[str]
    vocabulary_: Dict[str, int]

    def __init__(self, *, dtype: type = float, separator: str = "=", sort: bool = True):
        self.separator = separator
        self.sort = sort
        self.feature_names_ = []
        self.vocabulary_ = {}

    def _items(self, x) -> List[Tuple[str, float]]:
        out = List[Tuple[str, float]]()
        for f, v in x.items():
            if isinstance(v, str):
                out.append((f"{f}{self.separator}{v}", 1.0))
            elif isinstance(v, List[str]):
                for vv in v:
                    out.append((f"{f}{self.separator}{vv}", 1.0))
            elif isinstance(v, NoneType):
                out.append((str(f), 0.0))
            else:
                out.append((str(f), float(v)))
        return out

    def fit(self, X, y = None):
        names = List[str]()
        vocab = Dict[str, int]()
        for x in X:
            for name, _ in self._items(x):
                if name not in vocab:
                    vocab[name] = len(names)
                    names.append(name)
        if self.sort:
            names.sort()
            vocab = {f: i for i, f in enumerate(names)}
        self.feature_names_ = names
        self.vocabulary_ = vocab
        return self

    def _dense(self, X) -> np.ndarray[float, 2]:
        rows = [self._items(x) for x in X]
        if len(rows) == 0:
            raise ValueError("Sample sequence X is empty.")
        D = np.zeros((len(rows), len(self.feature_names_)))
        for i in range(len(rows)):
            for name, v in rows[i]:
                if name in self.vocabulary_:
                    D[i, self.vocabulary_[name]] += v
        return D

    def transform(self, X):
        import scipy.sparse as sp
        D = self._dense(X)
        if SP:
            return sp.csr_array(D)
        else:
            return D

    def transform_dense(self, X):
        return self._dense(X)

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features = None):
        return self.feature_names_.copy()

    def inverse_transform(self, X) -> List[Dict[str, float]]:
        A = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=float)
        out = List[Dict[str, float]]()
        for i in range(A.shape[0]):
            d = Dict[str, float]()
            for j in range(A.shape[1]):
                if A[i, j] != 0:
                    d[self.feature_names_[j]] = A[i, j]
            out.append(d)
        return out

    def restrict(self, support, indices: bool = False):
        s = np.asarray(support)
        keep = List[int]()
        if indices:
            keep = [int(v) for v in s]
        else:
            keep = [i for i in range(s.shape[0]) if s[i]]
        names = [self.feature_names_[i] for i in keep]
        if self.sort:
            names.sort()
        self.feature_names_ = names
        self.vocabulary_ = {f: i for i, f in enumerate(names)}
        return self

def DictVectorizer(*, dtype: type = float, separator: str = "=", sparse: Literal[bool] = True,
                   sort: bool = True):
    return _DictVectorizer[sparse](separator=separator, sort=sort)
