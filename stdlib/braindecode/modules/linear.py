# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization
from .parametrization import MaxNorm, MaxNormParametrize

class MaxNormLinear(nn.Linear):
    _max_norm_val: float
    _eps: float
    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_norm_val: float = 2.0, eps: float = 1e-05):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self._max_norm_val = max_norm_val
        self._eps = eps
        register_parametrization(self, "weight", MaxNorm(self._max_norm_val, self._eps))
    def _get_name(self) -> str:
        return "MaxNormLinear"

class LinearWithConstraint(nn.Linear):
    max_norm: float
    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_norm: float = 1.0):
        super().__init__(in_features, out_features, bias)
        self.max_norm = max_norm
        register_parametrization(self, "weight", MaxNormParametrize(self.max_norm))
    def _get_name(self) -> str:
        return "LinearWithConstraint"
