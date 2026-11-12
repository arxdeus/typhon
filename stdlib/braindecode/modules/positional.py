# Ported from braindecode 2.0.0 (BSD-3-Clause)

from torch import Tensor
import torch.nn as nn
from braindecode.functional import sinusoidal_positional_encoding

class SinusoidalPositionalEncoding(nn.Module):
    dropout: nn.Dropout
    pe: Tensor
    def __init__(self, max_len: int, dim: int, drop_prob: float = 0.0, persistent: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.add_module("dropout", self.dropout)
        self.pe = sinusoidal_positional_encoding(max_len, dim).unsqueeze(0)
        self.register_buffer("pe", self.pe, persistent=persistent)
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])
    def _get_name(self) -> str:
        return "SinusoidalPositionalEncoding"
