# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

import math
import numpy as np

FLOAT_DTYPES = (float64, float32, float16)

def is_scalar_nan(x) -> bool:
    """True for a float NaN (not for integers/None/strings)."""
    if isinstance(x, float) or isinstance(x, float32) or isinstance(x, float16):
        return math.isnan(float(x))
    return False

def safe_mask(X, mask):
    """Mask usable to index X: boolean arrays stay boolean, integer
    index lists become index arrays."""
    m = np.asarray(mask)
    if m.dtype == bool:
        return m
    return m.astype(int)

class Memory:
    """joblib.Memory stand-in: caching is a no-op, calls run directly."""
    location: Optional[str]
    verbose: int

    def __init__(self, location: Optional[str] = None, verbose: int = 0):
        self.location = location
        self.verbose = verbose

    def cache(self, func):
        return func

    def clear(self, warn: bool = True):
        pass

def check_memory(memory = None):
    """Return a Memory-like object. None gives a no-op memory; str paths are
    accepted (no caching is performed)."""
    if memory is None:
        return Memory(None)
    elif isinstance(memory, str):
        return Memory(memory)
    elif isinstance(memory, Memory):
        return memory
    else:
        if hasattr(memory, "cache"):
            return memory
        raise ValueError("'memory' should be None, a string or have the same interface as joblib.Memory.")

def all_estimators(type_filter = None) -> List[str]:
    """Minimal registry: names of the public estimators ported to Typhon
    (classes are statically typed, so names rather than class objects)."""
    names = ["AdaBoostClassifier", "AdaBoostRegressor", "BaggingClassifier", "BaggingRegressor",
             "BernoulliNB", "DecisionTreeClassifier", "DecisionTreeRegressor", "ElasticNet",
             "ExtraTreesClassifier", "ExtraTreesRegressor", "GaussianNB", "KMeans",
             "KNeighborsClassifier", "KNeighborsRegressor", "Lasso", "LinearRegression",
             "LogisticRegression", "MultinomialNB", "PCA", "RandomForestClassifier",
             "RandomForestRegressor", "Ridge", "SVC", "SVR", "StandardScaler"]
    kinds = ["classifier", "classifier", "classifier", "regressor",
             "classifier", "classifier", "regressor", "regressor",
             "classifier", "regressor", "classifier", "cluster",
             "classifier", "regressor", "regressor", "regressor",
             "classifier", "classifier", "transformer", "classifier",
             "regressor", "regressor", "classifier", "regressor", "transformer"]
    out = List[str]()
    for i in range(len(names)):
        if type_filter is None or kinds[i] == str(type_filter):
            out.append(names[i])
    return out
