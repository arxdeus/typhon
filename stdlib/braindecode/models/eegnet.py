# Ported from braindecode 2.0.0 (BSD-3-Clause)

from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from braindecode.functional import glorot_weight_zero_bias
from braindecode.models.base import EEGModuleMixin, _get_output_shape
from braindecode.modules import Conv2dWithConstraint, Ensure4d, LinearWithConstraint, SqueezeFinalOutput

class EEGNet(EEGModuleMixin, nn.Sequential):
    """EEGNet v4 from Lawhern et al (2018)."""
    final_conv_length: int
    pool_mode: str
    F1: int
    D: int
    F2: int
    kernel_length: int
    depthwise_kernel_length: int
    pool1_kernel_size: int
    pool2_kernel_size: int
    drop_prob: float
    batch_norm_momentum: float
    batch_norm_affine: bool
    batch_norm_eps: float
    conv_spatial_max_norm: float
    norm_rate: float

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 final_conv_length = "auto", pool_mode: str = "mean", F1: int = 8, D: int = 2,
                 F2: Optional[int] = None, kernel_length: int = 64, depthwise_kernel_length: int = 16,
                 pool1_kernel_size: int = 4, pool2_kernel_size: int = 8, conv_spatial_max_norm: float = 1.0,
                 activation = nn.ELU, batch_norm_momentum: float = 0.01, batch_norm_affine: bool = True,
                 batch_norm_eps: float = 0.001, drop_prob: float = 0.25, final_layer_with_constraint: bool = False,
                 norm_rate: float = 0.25, chs_info: Optional[List[Dict[str, str]]] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Sequential.__init__(self)
        auto = isinstance(final_conv_length, str)
        if auto:
            if final_conv_length != "auto":
                raise ValueError("final_conv_length must be 'auto' or an int")
            self.n_times
        self.final_conv_length = 0 if auto else int(final_conv_length)
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D if F2 is None else F2
        self.kernel_length = kernel_length
        self.depthwise_kernel_length = depthwise_kernel_length
        self.pool1_kernel_size = pool1_kernel_size
        self.pool2_kernel_size = pool2_kernel_size
        self.drop_prob = drop_prob
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_eps = batch_norm_eps
        self.conv_spatial_max_norm = conv_spatial_max_norm
        self.norm_rate = norm_rate
        self.mapping["conv_classifier.weight"] = "final_layer.conv_classifier.weight"
        self.mapping["conv_classifier.bias"] = "final_layer.conv_classifier.bias"
        if pool_mode != "max" and pool_mode != "mean":
            raise KeyError(pool_mode)
        FD = F1 * D
        self.add_module("ensuredims", Ensure4d())
        self.add_module("dimshuffle", Rearrange("batch ch t 1 -> batch 1 ch t"))
        self.add_module("conv_temporal", nn.Conv2d(1, F1, (1, kernel_length), bias=False, padding=(0, kernel_length // 2)))
        self.add_module("bnorm_temporal", nn.BatchNorm2d(F1, momentum=batch_norm_momentum, affine=batch_norm_affine, eps=batch_norm_eps))
        self.add_module("conv_spatial", Conv2dWithConstraint(in_channels=F1, out_channels=FD, kernel_size=(self.n_chans, 1),
                        max_norm=conv_spatial_max_norm, bias=False, groups=F1))
        self.add_module("bnorm_1", nn.BatchNorm2d(FD, momentum=batch_norm_momentum, affine=batch_norm_affine, eps=batch_norm_eps))
        self.add_module("elu_1", activation())
        self.add_module("pool_1", self._pool(pool1_kernel_size))
        self.add_module("drop_1", nn.Dropout(p=drop_prob))
        self.add_module("conv_separable_depth", nn.Conv2d(FD, FD, (1, depthwise_kernel_length), bias=False, groups=FD,
                        padding=(0, depthwise_kernel_length // 2)))
        self.add_module("conv_separable_point", nn.Conv2d(FD, self.F2, kernel_size=(1, 1), bias=False))
        self.add_module("bnorm_2", nn.BatchNorm2d(self.F2, momentum=batch_norm_momentum, affine=batch_norm_affine, eps=batch_norm_eps))
        self.add_module("elu_2", activation())
        self.add_module("pool_2", self._pool(pool2_kernel_size))
        self.add_module("drop_2", nn.Dropout(p=drop_prob))
        output_shape = _get_output_shape(self, self.n_chans, self.n_times)
        n_out_virtual_chans = output_shape[2]
        if auto:
            self.final_conv_length = output_shape[3]
        module = nn.Sequential()
        if not final_layer_with_constraint:
            module.add_module("conv_classifier", nn.Conv2d(self.F2, self.n_outputs, (n_out_virtual_chans, self.final_conv_length), bias=True))
            module.add_module("permute_back", Rearrange("batch x y z -> batch x z y"))
            module.add_module("squeeze", SqueezeFinalOutput())
        else:
            module.add_module("flatten", nn.Flatten())
            module.add_module("linearconstraint", LinearWithConstraint(in_features=self.F2 * self.final_conv_length,
                              out_features=self.n_outputs, max_norm=norm_rate))
        self.add_module("final_layer", module)
        glorot_weight_zero_bias(self)

    def _pool(self, k: int) -> nn.Module:
        if self.pool_mode == "max":
            return nn.MaxPool2d(kernel_size=(1, k))
        return nn.AvgPool2d(kernel_size=(1, k))

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "EEGNet"
