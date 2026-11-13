# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from ..base import estimator_repr as _estimator_repr
import numpy as np
from ..base import clone

class TransformedTargetRegressor[R, T = NoneType, F = NoneType, G = NoneType]:
    def __repr__(self) -> str:
        return _estimator_repr(self)

    """regressor fit on transformer(y) or func(y); predictions are mapped
    back with inverse_transform / inverse_func. y must be 1-d."""
    regressor: R
    transformer: T
    func: F
    inverse_func: G
    check_inverse: bool
    regressor_: R
    transformer_: T
    _y_ndim: int

    def __init__(self, regressor: R, *, transformer: T = None, func: F = None,
                 inverse_func: G = None, check_inverse: bool = True):
        if not isinstance(transformer, NoneType) and (not isinstance(func, NoneType) or not isinstance(inverse_func, NoneType)):
            raise ValueError("'transformer' and functions 'func'/'inverse_func' cannot both be set.")
        if isinstance(func, NoneType) != isinstance(inverse_func, NoneType):
            raise ValueError("When 'func' is provided, 'inverse_func' must also be provided. If 'inverse_func' is provided, 'func' must also be provided.")
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.regressor_ = clone(regressor)
        self.transformer_ = transformer
        self._y_ndim = 1

    _estimator_type: Literal[str] = "regressor"

    def _sk_clone(self):
        if isinstance(self.transformer, NoneType):
            return TransformedTargetRegressor(clone(self.regressor), func=self.func,
                                              inverse_func=self.inverse_func, check_inverse=self.check_inverse)
        else:
            return TransformedTargetRegressor(clone(self.regressor), transformer=clone(self.transformer),
                                              check_inverse=self.check_inverse)

    def _fwd(self, Y2):
        if not isinstance(self.transformer, NoneType):
            return np.asarray(self.transformer_.transform(Y2), dtype=float)
        elif not isinstance(self.func, NoneType):
            return np.asarray(self.func(Y2), dtype=float)
        else:
            return Y2

    def _inv(self, Y2):
        if not isinstance(self.transformer, NoneType):
            return np.asarray(self.transformer_.inverse_transform(Y2), dtype=float)
        elif not isinstance(self.inverse_func, NoneType):
            return np.asarray(self.inverse_func(Y2), dtype=float)
        else:
            return Y2

    def fit(self, X, y):
        ya = np.asarray(y, dtype=float)
        n = ya.shape[0]
        Y2 = ya.reshape((n, 1))
        if not isinstance(self.transformer, NoneType):
            self.transformer_ = clone(self.transformer)
            self.transformer_.fit(Y2)
        Yt = self._fwd(Y2)
        if self.check_inverse and (not isinstance(self.transformer, NoneType) or not isinstance(self.func, NoneType)):
            step = max(1, n // 10)
            idx = np.arange(0, n, step)
            back = self._inv(self._fwd(Y2[idx]))
            if not np.allclose(back, Y2[idx]):
                import warnings
                warnings.warn("The provided functions or transformer are not strictly inverse of each other. If you are sure you want to proceed regardless, set 'check_inverse=False'")
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, np.ascontiguousarray(Yt.reshape((n, Yt.size // n))[:, 0]))
        return self

    def predict(self, X):
        p = np.asarray(self.regressor_.predict(X), dtype=float)
        n = p.shape[0]
        return np.ascontiguousarray(self._inv(p.reshape((n, 1))).reshape((n, 1))[:, 0])

    def score(self, X, y, sample_weight = None) -> float:
        from ..metrics import r2_score
        return float(r2_score(y, self.predict(X)))
