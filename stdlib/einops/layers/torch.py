# Ported from einops 0.8.2 (MIT)
#
# einops.layers.torch: Rearrange / Reduce as torch.nn.Module.

from torch._tensor import Tensor
from torch.nn.modules.module import Module
from einops import _apply, _lengths

class Rearrange(Module):
    pattern: str
    _axes_lengths: Dict[str, int]

    def __init__(self, pattern: str, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self._axes_lengths = _lengths(axes_lengths)

    def forward(self, input: Tensor) -> Tensor:
        return _apply(input, self.pattern, "rearrange", self._axes_lengths)

    def _get_name(self) -> str:
        return "Rearrange"

    def __repr__(self) -> str:
        s = f"Rearrange('{self.pattern}'"
        for k, v in self._axes_lengths.items():
            s += f", {k}={v}"
        return s + ")"

class Reduce(Module):
    pattern: str
    reduction: str
    _axes_lengths: Dict[str, int]

    def __init__(self, pattern: str, reduction: str, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self._axes_lengths = _lengths(axes_lengths)

    def forward(self, input: Tensor) -> Tensor:
        return _apply(input, self.pattern, self.reduction, self._axes_lengths)

    def _get_name(self) -> str:
        return "Reduce"

    def __repr__(self) -> str:
        s = f"Reduce('{self.pattern}', '{self.reduction}'"
        for k, v in self._axes_lengths.items():
            s += f", {k}={v}"
        return s + ")"
