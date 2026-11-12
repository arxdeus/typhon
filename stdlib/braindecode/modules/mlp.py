# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch.nn as nn

class MLP(nn.Sequential):
    """braindecode.modules.MLP: Linear/activation(/LayerNorm) stack, last
    activation dropped, final Dropout."""
    in_features: int
    out_features: int
    hidden_features: List[int]

    def __init__(self, in_features: int, hidden_features: Optional[List[int]] = None,
                 out_features: Optional[int] = None, activation = nn.GELU, drop: float = 0.0, normalize: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features
        hf = [in_features, in_features]
        if hidden_features is not None and len(hidden_features) > 0:
            hf = hidden_features
        self.hidden_features = hf
        dims = [in_features] + hf + [self.out_features]
        mods = List[nn.Module]()
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(activation())
                if normalize:
                    mods.append(nn.LayerNorm(dims[i + 1]))
        mods.append(nn.Dropout(drop))
        for i in range(len(mods)):
            self.add_module(str(i), mods[i])

    def _get_name(self) -> str:
        return "MLP"
