# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.frozen.FrozenEstimator: wraps a fitted estimator so that fit() is a
# no-op and all prediction methods delegate to the wrapped estimator.

class FrozenEstimator[E]:
    estimator: E

    def __init__(self, estimator: E):
        self.estimator = estimator

    def fit(self, X, y = None, **kwargs):
        return self

    def fit_transform(self, X, y = None, **kwargs):
        raise AttributeError("FrozenEstimator does not have fit_transform")

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict_log_proba(self, X):
        return self.estimator.predict_log_proba(X)

    def decision_function(self, X):
        return self.estimator.decision_function(X)

    def transform(self, X):
        return self.estimator.transform(X)

    def inverse_transform(self, X):
        return self.estimator.inverse_transform(X)

    def score(self, X, y, sample_weight = None):
        return self.estimator.score(X, y)

    @property
    def classes_(self):
        return self.estimator.classes_

    @property
    def n_features_in_(self):
        return self.estimator.n_features_in_

    def _sk_clone(self):
        return FrozenEstimator(self.estimator)

    def __repr__(self) -> str:
        return "FrozenEstimator(estimator=" + repr(self.estimator) + ")"
