# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

class FunctionTransformer[F, G]:
    """func / inverse_func are Typhon callables (functions or lambdas)."""
    func: F
    inverse_func: G
    validate: bool

    def __init__(self, func: F = None, inverse_func: G = None, *, validate: bool = False,
                 accept_sparse: bool = False, check_inverse: bool = True,
                 feature_names_out = None, kw_args = None, inv_kw_args = None):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if self.func is None:
            return X
        else:
            return self.func(X)

    def inverse_transform(self, X):
        if self.inverse_func is None:
            return X
        else:
            return self.inverse_func(X)

    def fit_transform(self, X, y = None):
        return self.transform(X)
