# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from ._samples_generator import (make_classification, make_multilabel_classification,
                                 make_hastie_10_2, make_regression, make_circles, make_moons,
                                 make_blobs, make_friedman1, make_friedman2, make_friedman3,
                                 make_low_rank_matrix, make_sparse_uncorrelated,
                                 make_sparse_coded_signal, make_spd_matrix,
                                 make_sparse_spd_matrix, make_swiss_roll, make_s_curve,
                                 make_gaussian_quantiles, make_biclusters, make_checkerboard)
from ._base import (load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes,
                    load_csv_data, load_linnerud)
from ._svmlight_format_io import load_svmlight_file, load_svmlight_files, dump_svmlight_file
from ._base import load_files

def _shuf(n: int, seed: int):
    from ..utils import _LegacyRS
    return _LegacyRS(seed).permutation(n)
