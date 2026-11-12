# Ported from braindecode 2.0.0 (BSD-3-Clause)

from torch import Tensor
import torch.nn as nn

class Expression(nn.Module):
    # expression_fn: any callable Tensor -> Tensor (function or lambda).
    expression_fn: Function[[Tensor], Tensor]
    def __init__(self, expression_fn: Function[[Tensor], Tensor]):
        super().__init__()
        self.expression_fn = expression_fn
    def forward(self, x: Tensor) -> Tensor:
        return self.expression_fn(x)
    def _get_name(self) -> str:
        return "Expression"
