# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.preprocessing (dense ndarray input).

from ._data import StandardScaler, MinMaxScaler, _handle_zeros_in_scale
from ._data2 import (MaxAbsScaler, RobustScaler, Normalizer, normalize, Binarizer, binarize,
                     KernelCenterer, add_dummy_feature, scale, minmax_scale, maxabs_scale,
                     robust_scale)
from ._function_transformer import FunctionTransformer
from ._encoders import OneHotEncoder
from ._ordinal import OrdinalEncoder
from ._label import LabelEncoder, LabelBinarizer, label_binarize, MultiLabelBinarizer
from ._polynomial import PolynomialFeatures
from ._spline import SplineTransformer
from ._quantile import QuantileTransformer, quantile_transform, KBinsDiscretizer
from ._power import PowerTransformer, power_transform
from ._target_encoder import TargetEncoder
