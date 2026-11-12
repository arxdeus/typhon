# Ported from pyriemann 0.12 (BSD-3-Clause)

import numpy as np
from .geometry.covariance import covariances

class Covariances:
    estimator: str

    def __init__(self, estimator: str = "scm"):
        self.estimator = estimator

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return covariances(X, estimator=self.estimator)

    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)
