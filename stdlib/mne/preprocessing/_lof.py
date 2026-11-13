# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .._info import _picks_to_idx


def find_bad_channels_lof(raw, n_neighbors: int = 20, *, picks = None, metric: str = "euclidean",
                          threshold: float = 1.5, return_scores: Literal[bool] = False, verbose = None):
    """Bad channels via Local Outlier Factor over channel time courses."""
    pk = _picks_to_idx(raw.info, picks, "data", "bads")
    types = Set[str]()
    for p in pk:
        types.add(raw.info.ch_types[p])
    if len(types) != 1:
        raise ValueError(f"Need exactly one channel type in picks, got {sorted(list(types))}")
    data = raw.get_data(picks=pk)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric=metric)
    clf.fit_predict(data)
    scores = clf.negative_outlier_factor_
    bads = List[str]()
    for i in range(len(pk)):
        if abs(scores[i]) >= threshold:
            bads.append(raw.info.ch_names[pk[i]])
    if return_scores:
        return bads, scores
    return bads
