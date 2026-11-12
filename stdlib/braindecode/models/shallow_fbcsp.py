# Ported from braindecode 2.0.0 (BSD-3-Clause)

from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from braindecode.models.base import EEGModuleMixin, _get_output_shape
from braindecode.modules import CombinedConv, Ensure4d, Expression, SafeLog, Square, SqueezeFinalOutput
from einops.layers.torch import Rearrange

class ShallowFBCSPNet(EEGModuleMixin, nn.Sequential):
    """Shallow ConvNet model from Schirrmeister et al (2017).

    `conv_nonlin` / `activation_pool_nonlin` take a Module class (default
    Square / SafeLog). `final_conv_length` is "auto" or an int.
    """
    n_filters_time: int
    filter_time_length: int
    n_filters_spat: int
    pool_time_length: int
    pool_time_stride: int
    final_conv_length: int
    pool_mode: str
    split_first_layer: bool
    batch_norm: bool
    batch_norm_alpha: float
    drop_prob: float

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 n_filters_time: int = 40, filter_time_length: int = 25, n_filters_spat: int = 40,
                 pool_time_length: int = 75, pool_time_stride: int = 15, final_conv_length = "auto",
                 conv_nonlin = Square, pool_mode: str = "mean", activation_pool_nonlin = SafeLog,
                 split_first_layer: bool = True, batch_norm: bool = True, batch_norm_alpha: float = 0.1,
                 drop_prob: float = 0.5, chs_info: Optional[List[Dict[str, str]]] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Sequential.__init__(self)
        auto = isinstance(final_conv_length, str)
        if auto:
            if final_conv_length != "auto":
                raise ValueError("final_conv_length must be 'auto' or an int")
            self.n_times
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = 0 if auto else int(final_conv_length)
        self.pool_mode = pool_mode
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        self.mapping["conv_classifier.weight"] = "final_layer.conv_classifier.weight"
        self.mapping["conv_classifier.bias"] = "final_layer.conv_classifier.bias"
        if split_first_layer:
            self.mapping["conv_time.weight"] = "conv_time_spat.conv_time.weight"
            self.mapping["conv_spat.weight"] = "conv_time_spat.conv_spat.weight"
            self.mapping["conv_time.bias"] = "conv_time_spat.conv_time.bias"
            self.mapping["conv_spat.bias"] = "conv_time_spat.conv_spat.bias"
        if pool_mode != "max" and pool_mode != "mean":
            raise KeyError(pool_mode)

        self.add_module("ensuredims", Ensure4d())
        n_filters_conv = 0
        if split_first_layer:
            self.add_module("dimshuffle", Rearrange("batch C T 1 -> batch 1 T C"))
            self.add_module("conv_time_spat", CombinedConv(in_chans=self.n_chans, n_filters_time=n_filters_time,
                            n_filters_spat=n_filters_spat, filter_time_length=filter_time_length,
                            bias_time=True, bias_spat=not batch_norm))
            n_filters_conv = n_filters_spat
        else:
            self.add_module("conv_time", nn.Conv2d(self.n_chans, n_filters_time, (filter_time_length, 1),
                                                   stride=1, bias=not batch_norm))
            n_filters_conv = n_filters_time
        if batch_norm:
            self.add_module("bnorm", nn.BatchNorm2d(n_filters_conv, momentum=batch_norm_alpha, affine=True))
        self.add_module("conv_nonlin_exp", conv_nonlin())
        if pool_mode == "max":
            self.add_module("pool", nn.MaxPool2d(kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)))
        else:
            self.add_module("pool", nn.AvgPool2d(kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)))
        self.add_module("pool_nonlin_exp", activation_pool_nonlin())
        self.add_module("drop", nn.Dropout(p=drop_prob))
        self.eval()
        if auto:
            self.final_conv_length = _get_output_shape(self, self.n_chans, self.n_times)[2]
        module = nn.Sequential()
        module.add_module("conv_classifier", nn.Conv2d(n_filters_conv, self.n_outputs, (self.final_conv_length, 1), bias=True))
        module.add_module("squeeze", SqueezeFinalOutput())
        self.add_module("final_layer", module)
        self._init_parameters()
        self.train()

    def _init_parameters(self):
        if self.split_first_layer:
            init.xavier_uniform_(self.get_parameter("conv_time_spat.conv_time.weight"), gain=1.0)
            init.constant_(self.get_parameter("conv_time_spat.conv_time.bias"), 0.0)
            init.xavier_uniform_(self.get_parameter("conv_time_spat.conv_spat.weight"), gain=1.0)
            if not self.batch_norm:
                init.constant_(self.get_parameter("conv_time_spat.conv_spat.bias"), 0.0)
        else:
            init.xavier_uniform_(self.get_parameter("conv_time.weight"), gain=1.0)
            if not self.batch_norm:
                init.constant_(self.get_parameter("conv_time.bias"), 0.0)
        if self.batch_norm:
            init.constant_(self.get_parameter("bnorm.weight"), 1.0)
            init.constant_(self.get_parameter("bnorm.bias"), 0.0)
        init.xavier_uniform_(self.get_parameter("final_layer.conv_classifier.weight"), gain=1.0)
        init.constant_(self.get_parameter("final_layer.conv_classifier.bias"), 0.0)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "ShallowFBCSPNet"
