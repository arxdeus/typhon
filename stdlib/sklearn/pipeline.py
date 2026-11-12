# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import numpy as np
from .base import clone

def _name_of(est: T, T: type) -> str:
    return T.__name__.lower()

def _clone_steps(steps):
    if static.len(steps) == 0:
        return ()
    else:
        return ((steps[0][0], clone(steps[0][1])),) + _clone_steps(steps[1:])

def _fit_transform_head(steps, X, y):
    # fit_transform every step except the last; returns the transformed X
    if static.len(steps) <= 1:
        return X
    else:
        Xt = steps[0][1].fit_transform(X, y)
        return _fit_transform_head(steps[1:], Xt, y)

def _transform_head(steps, X):
    if static.len(steps) <= 1:
        return X
    else:
        return _transform_head(steps[1:], steps[0][1].transform(X))

def _transform_all(steps, X):
    if static.len(steps) == 0:
        return X
    else:
        return _transform_all(steps[1:], steps[0][1].transform(X))

class Pipeline[S]:
    """Pipeline over a STATIC tuple of (name, estimator) pairs.

    Typhon resolves each step's type at compile time, so `steps` is a tuple
    (a Python list literal of pairs also works when all steps share a type)."""
    steps: S
    memory: Optional[str]  # not used (no transformer caching)
    verbose: bool

    def __init__(self, steps: S, *, memory: Optional[str] = None, verbose: bool = False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose

    def _sk_clone(self):
        return Pipeline(_clone_steps(self.steps), memory=self.memory, verbose=self.verbose)

    @property
    def named_steps(self):
        return self.steps

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    def __len__(self) -> int:
        return static.len(self.steps)

    def __getitem__(self, i: Literal[int]):
        return self.steps[i][1]

    def fit(self, X, y = None):
        Xt = _fit_transform_head(self.steps, X, y)
        self.steps[-1][1].fit(Xt, y)
        return self

    def fit_transform(self, X, y = None):
        Xt = _fit_transform_head(self.steps, X, y)
        return self.steps[-1][1].fit_transform(Xt, y)

    def fit_predict(self, X, y = None):
        return self.fit(X, y).predict(X)

    def predict(self, X):
        return self.steps[-1][1].predict(_transform_head(self.steps, X))

    def predict_proba(self, X):
        return self.steps[-1][1].predict_proba(_transform_head(self.steps, X))

    def decision_function(self, X):
        return self.steps[-1][1].decision_function(_transform_head(self.steps, X))

    def transform(self, X):
        return _transform_all(self.steps, X)

    def score(self, X, y, sample_weight = None):
        return self.steps[-1][1].score(_transform_head(self.steps, X), y)

    @property
    def classes_(self):
        return self.steps[-1][1].classes_

def _named(ests):
    if static.len(ests) == 0:
        return ()
    else:
        return ((_name_of(ests[0]), ests[0]),) + _named(ests[1:])

def make_pipeline(*steps, memory: Optional[str] = None, verbose: bool = False):
    return Pipeline(_named(steps), memory=memory, verbose=verbose)
