# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch.nn as nn
from torch import Tensor

def glorot_weight_zero_bias(model):
    # BatchNorm weights to one and all biases to zero (name kept for
    # compatibility, as in braindecode).
    for module in model.modules():
        bn = "BatchNorm" in module._get_name()
        for i in range(len(module._pnames)):
            p = module._params[i]
            if p is not None:
                if module._pnames[i] == "bias":
                    nn.init.constant_(p, 0.0)
                elif bn and module._pnames[i] == "weight":
                    nn.init.constant_(p, 1.0)

def _param_by_name(module, name: str) -> Optional[Tensor]:
    for i in range(len(module._pnames)):
        if module._pnames[i] == name:
            return module._params[i]
    return None

def _weight_leaf(module, init_parametrized_original: bool) -> Optional[Tensor]:
    for i in range(len(module._ptz_names)):
        if module._ptz_names[i] == "weight":
            if init_parametrized_original:
                return module._ptz_orig[i]
            return module._ptz_cache[i]
    return _param_by_name(module, "weight")

def init_weights_trunc_normal_(module, linear_kinds: List[str], conv_kinds: List[str], std_linear: float,
                               std_conv: float, norm_kinds: List[str], init_parametrized_original: bool = False):
    # Typhon form of braindecode.functional.init_weights_ with trunc_normal_
    # rules, keyed by module class name (_get_name) instead of isinstance.
    n = module._get_name()
    std = -1.0
    if n in linear_kinds:
        std = std_linear
    elif n in conv_kinds:
        std = std_conv
    if std >= 0.0:
        w = _weight_leaf(module, init_parametrized_original)
        if w is not None:
            nn.init.trunc_normal_(w.__val__(), std=std)
        b = _param_by_name(module, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())
        return
    if n in norm_kinds:
        w = _param_by_name(module, "weight")
        if w is not None:
            nn.init.ones_(w.__val__())
        b = _param_by_name(module, "bias")
        if b is not None:
            nn.init.zeros_(b.__val__())
