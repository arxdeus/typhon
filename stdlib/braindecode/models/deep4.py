# Ported from braindecode 2.0.0 (BSD-3-Clause)

from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin, _get_output_shape, _pyround
from braindecode.modules import AvgPool2dWithConv, CombinedConv, Ensure4d, SqueezeFinalOutput

class Deep4Net(EEGModuleMixin, nn.Sequential):
    """Deep ConvNet model from Schirrmeister et al (2017)."""
    final_conv_length: int
    _auto_final: bool
    n_filters_time: int
    n_filters_spat: int
    filter_time_length: int
    pool_time_length: int
    pool_time_stride: int
    n_filters_2: int
    filter_length_2: int
    n_filters_3: int
    filter_length_3: int
    n_filters_4: int
    filter_length_4: int
    first_pool_mode: str
    later_pool_mode: str
    drop_prob: float
    split_first_layer: bool
    batch_norm: bool
    batch_norm_alpha: float
    stride_before_pool: bool

    def __init__(self, n_chans: Optional[int] = None, n_outputs: Optional[int] = None, n_times: Optional[int] = None,
                 final_conv_length = "auto", n_filters_time: int = 25, n_filters_spat: int = 25,
                 filter_time_length: int = 10, pool_time_length: int = 3, pool_time_stride: int = 3,
                 n_filters_2: int = 50, filter_length_2: int = 10, n_filters_3: int = 100, filter_length_3: int = 10,
                 n_filters_4: int = 200, filter_length_4: int = 10, activation_first_conv_nonlin = nn.ELU,
                 first_pool_mode: str = "max", first_pool_nonlin = nn.Identity, activation_later_conv_nonlin = nn.ELU,
                 later_pool_mode: str = "max", later_pool_nonlin = nn.Identity, drop_prob: float = 0.5,
                 split_first_layer: bool = True, batch_norm: bool = True, batch_norm_alpha: float = 0.1,
                 stride_before_pool: bool = False, chs_info: Optional[List[Dict[str, str]]] = None,
                 input_window_seconds: Optional[float] = None, sfreq: Optional[float] = None):
        EEGModuleMixin.__init__(self, n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                                input_window_seconds=input_window_seconds, sfreq=sfreq)
        nn.Sequential.__init__(self)
        auto = isinstance(final_conv_length, str)
        fcl = 0
        if isinstance(final_conv_length, str):
            if final_conv_length != "auto":
                raise ValueError("final_conv_length must be 'auto' or an int")
            self.n_times
        else:
            fcl = int(final_conv_length)
        self._auto_final = auto
        self.final_conv_length = fcl
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_pool_mode = first_pool_mode
        self.later_pool_mode = later_pool_mode
        self.drop_prob = drop_prob
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        min_n_times = self._get_min_n_times()
        resolved: Optional[int] = self._n_times
        if resolved is None and self._input_window_seconds is not None and self._sfreq is not None:
            resolved = _pyround(self._input_window_seconds * self._sfreq)
        if resolved is not None and resolved >= min_n_times:
            min_n_times = self._get_min_n_times(exact=True)
        if resolved is not None and resolved < min_n_times:
            sf = float(resolved) / float(min_n_times)
            print(f"UserWarning: n_times ({resolved}) is smaller than the minimum required ({min_n_times}) for the current model parameters configuration. Adjusting parameters to ensure compatibility.")
            self.filter_time_length = max(1, int(self.filter_time_length * sf))
            self.pool_time_length = max(1, int(self.pool_time_length * sf))
            self.pool_time_stride = max(1, int(self.pool_time_stride * sf))
            self.filter_length_2 = max(1, int(self.filter_length_2 * sf))
            self.filter_length_3 = max(1, int(self.filter_length_3 * sf))
            self.filter_length_4 = max(1, int(self.filter_length_4 * sf))

        self.mapping["conv_classifier.weight"] = "final_layer.conv_classifier.weight"
        self.mapping["conv_classifier.bias"] = "final_layer.conv_classifier.bias"
        if split_first_layer:
            self.mapping["conv_time.weight"] = "conv_time_spat.conv_time.weight"
            self.mapping["conv_spat.weight"] = "conv_time_spat.conv_spat.weight"
            self.mapping["conv_time.bias"] = "conv_time_spat.conv_time.bias"
            self.mapping["conv_spat.bias"] = "conv_time_spat.conv_spat.bias"
        conv_stride = 1
        pool_stride = self.pool_time_stride
        if stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        for pm in (first_pool_mode, later_pool_mode):
            if pm != "max" and pm != "mean":
                raise KeyError(pm)

        self.add_module("ensuredims", Ensure4d())
        n_filters_conv = 0
        if split_first_layer:
            self.add_module("dimshuffle", Rearrange("batch C T 1 -> batch 1 T C"))
            self.add_module("conv_time_spat", CombinedConv(in_chans=self.n_chans, n_filters_time=n_filters_time,
                            n_filters_spat=n_filters_spat, filter_time_length=self.filter_time_length,
                            bias_time=True, bias_spat=not batch_norm, stride=(conv_stride, 1)))
            n_filters_conv = n_filters_spat
        else:
            self.add_module("conv_time", nn.Conv2d(self.n_chans, n_filters_time, (self.filter_time_length, 1),
                                                   stride=(conv_stride, 1), bias=not batch_norm))
            n_filters_conv = n_filters_time
        if batch_norm:
            self.add_module("bnorm", nn.BatchNorm2d(n_filters_conv, momentum=batch_norm_alpha, affine=True, eps=1e-05))
        self.add_module("conv_nonlin", activation_first_conv_nonlin())
        self.add_module("pool", self._pool(first_pool_mode, pool_stride))
        self.add_module("pool_nonlin", first_pool_nonlin())

        blocks = [(n_filters_conv, n_filters_2, self.filter_length_2),
                  (n_filters_2, n_filters_3, self.filter_length_3),
                  (n_filters_3, n_filters_4, self.filter_length_4)]
        for bi in range(3):
            nb, nf, fl = blocks[bi]
            suffix = f"_{bi + 2}"
            self.add_module("drop" + suffix, nn.Dropout(p=drop_prob))
            self.add_module("conv" + suffix, nn.Conv2d(nb, nf, (fl, 1), stride=(conv_stride, 1), bias=not batch_norm))
            if batch_norm:
                self.add_module("bnorm" + suffix, nn.BatchNorm2d(nf, momentum=batch_norm_alpha, affine=True, eps=1e-05))
            self.add_module("nonlin" + suffix, activation_later_conv_nonlin())
            self.add_module("pool" + suffix, self._pool(later_pool_mode, pool_stride))
            self.add_module("pool_nonlin" + suffix, later_pool_nonlin())
        self.eval()
        if auto:
            self.final_conv_length = _get_output_shape(self, self.n_chans, self.n_times)[2]
        module = nn.Sequential()
        module.add_module("conv_classifier", nn.Conv2d(n_filters_4, self.n_outputs, (self.final_conv_length, 1), bias=True))
        module.add_module("squeeze", SqueezeFinalOutput())
        self.add_module("final_layer", module)
        self._init_parameters()
        self.train()

    def _pool(self, mode: str, pool_stride: int) -> nn.Module:
        if mode == "max":
            return nn.MaxPool2d(kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1))
        return AvgPool2dWithConv(kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1))

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
        for b in range(2, 5):
            init.xavier_uniform_(self.get_parameter(f"conv_{b}.weight"), gain=1.0)
            if not self.batch_norm:
                init.constant_(self.get_parameter(f"conv_{b}.bias"), 0.0)
            else:
                init.constant_(self.get_parameter(f"bnorm_{b}.weight"), 1.0)
                init.constant_(self.get_parameter(f"bnorm_{b}.bias"), 0.0)
        init.xavier_uniform_(self.get_parameter("final_layer.conv_classifier.weight"), gain=1.0)
        init.constant_(self.get_parameter("final_layer.conv_classifier.bias"), 0.0)

    def _get_min_n_times(self, exact: bool = False) -> int:
        min_len = 1
        if exact and not self._auto_final:
            min_len = self.final_conv_length
        bp = [(self.filter_length_4, self.pool_time_length, self.pool_time_stride),
              (self.filter_length_3, self.pool_time_length, self.pool_time_stride),
              (self.filter_length_2, self.pool_time_length, self.pool_time_stride),
              (self.filter_time_length, self.pool_time_length, self.pool_time_stride)]
        for fl, pl, ps in bp:
            if exact and self.stride_before_pool:
                min_len = ps * (min_len + pl - 2) + fl
                continue
            min_len = ps * (min_len - 1) + pl
            min_len = min_len + fl - 1
        return min_len

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True, assign: bool = False):
        return nn.Module.load_state_dict(self, self._map_state_dict(state_dict), strict, assign)

    def _get_name(self) -> str:
        return "Deep4Net"
