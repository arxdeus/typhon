# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# EEGModuleMixin: signal-related parameter resolution shared by all models.
# Usage: class M(EEGModuleMixin, nn.Sequential) and call
# `EEGModuleMixin.__init__(self, ...)` then `nn.Sequential.__init__(self)`
# (Typhon has no cooperative super() across two bases). chs_info is a list of
# channel dicts (Dict[str, str]-like); only its length is used here.

import torch
from torch import Tensor

class EEGModuleMixin:
    _n_outputs: Optional[int]
    _n_chans: Optional[int]
    _chs_info: Optional[List[Dict[str, str]]]
    _n_times: Optional[int]
    _input_window_seconds: Optional[float]
    _sfreq: Optional[float]
    mapping: Dict[str, str]

    def __init__(
        self,
        n_outputs: Optional[int] = None,
        n_chans: Optional[int] = None,
        chs_info: Optional[List[Dict[str, str]]] = None,
        n_times: Optional[int] = None,
        input_window_seconds: Optional[float] = None,
        sfreq: Optional[float] = None,
    ):
        if n_chans is not None and chs_info is not None and len(chs_info) != n_chans:
            raise ValueError(f"n_chans={n_chans} different from chs_info={chs_info} length")
        if (n_times is not None and input_window_seconds is not None and sfreq is not None
                and n_times != _pyround(input_window_seconds * sfreq)):
            raise ValueError(
                f"n_times={n_times} different from input_window_seconds={input_window_seconds} * sfreq={sfreq}")
        self._input_window_seconds = input_window_seconds
        self._chs_info = chs_info
        self._n_outputs = n_outputs
        self._n_chans = n_chans
        self._n_times = n_times
        self._sfreq = sfreq
        self.mapping = {}

    @property
    def n_outputs(self) -> int:
        if self._n_outputs is None:
            raise ValueError("n_outputs not specified.")
        return self._n_outputs

    @property
    def n_chans(self) -> int:
        if self._n_chans is None and self._chs_info is not None:
            return len(self._chs_info)
        elif self._n_chans is None:
            raise ValueError("n_chans could not be inferred. Either specify n_chans or chs_info.")
        return self._n_chans

    @property
    def chs_info(self) -> List[Dict[str, str]]:
        if self._chs_info is None:
            raise ValueError("chs_info not specified.")
        return self._chs_info

    @property
    def n_times(self) -> int:
        if self._n_times is None and self._input_window_seconds is not None and self._sfreq is not None:
            return _pyround(self._input_window_seconds * self._sfreq)
        elif self._n_times is None:
            raise ValueError("n_times could not be inferred. "
                             "Either specify n_times or input_window_seconds and sfreq.")
        return self._n_times

    @property
    def input_window_seconds(self) -> float:
        if self._input_window_seconds is None and self._n_times is not None and self._sfreq is not None:
            return float(self._n_times) / self._sfreq
        elif self._input_window_seconds is None:
            raise ValueError("input_window_seconds could not be inferred. "
                             "Either specify input_window_seconds or n_times and sfreq.")
        return self._input_window_seconds

    @property
    def sfreq(self) -> float:
        if self._sfreq is None and self._input_window_seconds is not None and self._n_times is not None:
            return float(self._n_times) / self._input_window_seconds
        elif self._sfreq is None:
            raise ValueError("sfreq could not be inferred. "
                             "Either specify sfreq or input_window_seconds and n_times.")
        return self._sfreq

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (1, self.n_chans, self.n_times)

    def _map_state_dict(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        new = Dict[str, Tensor]()
        for k, v in state_dict.items():
            if k in self.mapping:
                new[self.mapping[k]] = v
            else:
                new[k] = v
        return new

def _pyround(v: float) -> int:
    # Python round(): banker's rounding on exact halves.
    f = v.__floor__()
    d = v - f
    r = f
    if d > 0.5:
        r = f + 1.0
    elif d == 0.5:
        if int(f) % 2 != 0:
            r = f + 1.0
    return int(r)

def _get_output_shape(model, n_chans: int, n_times: int) -> List[int]:
    # Probe with zeros of shape (1, n_chans, n_times) in eval mode, restoring
    # each submodule's training flag afterwards.
    mods = model.modules()
    modes = [m.training for m in mods]
    model.eval()
    out = List[int]()
    with torch.no_grad():
        x = torch.zeros(1, n_chans, n_times)
        y = model.forward(x)
        out = [int(s) for s in y.shape]
    for i in range(len(mods)):
        mods[i].training = modes[i]
    return out
