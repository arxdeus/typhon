# Ported from braindecode 2.0.0 (BSD-3-Clause)

from torch import Tensor
import torch.nn as nn

class FeedForwardBlock(nn.Sequential):
    gated: bool
    def __init__(self, emb_size: int, expansion: int, drop_p: float, activation = nn.GELU,
                 hidden_features: Optional[int] = None, bias: bool = True, output_drop_p: float = 0.0):
        super().__init__()
        self.gated = False
        hf = expansion * emb_size if hidden_features is None else hidden_features
        self.add_module("0", nn.Linear(emb_size, hf, bias=bias))
        self.add_module("1", activation())
        self.add_module("2", nn.Dropout(drop_p))
        self.add_module("3", nn.Linear(hf, emb_size, bias=bias))
        if output_drop_p:
            self.add_module("4", nn.Dropout(output_drop_p))
    def _get_name(self) -> str:
        return "FeedForwardBlock"

import torch

class InceptionBlock(nn.Module):
    branches: nn.ModuleList
    def __init__(self, branches: List[nn.Module]):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.add_module("branches", self.branches)
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([b(x) for b in self.branches.children()], 1)
    def _get_name(self) -> str:
        return "InceptionBlock"
