# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn.functional as F
from .functions import dwt_max_level

# float32 Daubechies decomposition filters (pywt order), as produced by
# braindecode.functional.daubechies_filters. The reference computes them from
# polynomial roots with numpy; the tabulated values are its float32 output.
_DB = {
    1: [[0.7071067690849304, 0.7071067690849304], [-0.7071067690849304, 0.7071067690849304]],
    2: [[-0.12940952181816101, 0.2241438627243042, 0.8365163207054138, 0.4829629063606262],
        [-0.4829629063606262, 0.8365163207054138, -0.2241438627243042, -0.12940952181816101]],
    4: [[-0.010597402229905128, 0.03288301080465317, 0.03084138222038746, -0.18703481554985046,
         -0.02798376977443695, 0.6308807730674744, 0.7148465514183044, 0.23037780821323395],
        [-0.23037780821323395, 0.7148465514183044, -0.6308807730674744, -0.02798376977443695,
         0.18703481554985046, 0.03084138222038746, -0.03288301080465317, -0.010597402229905128]],
}

def daubechies_filters(n_vanishing: int) -> Tensor:
    if n_vanishing not in _DB:
        raise NotImplementedError(f"daubechies_filters: only db1/db2/db4 are tabulated, got db{n_vanishing}")
    f = _DB[n_vanishing]
    return torch.tensor(f[0] + f[1], dtype=torch.float32).reshape(2, -1)

def wavelet_decomposition(x: Tensor, filters: Tensor, n_levels: Optional[int] = None) -> Tensor:
    fl = filters.shape[-1]
    nl = dwt_max_level(x.shape[-1], fl) if n_levels is None else n_levels.__val__()
    lead = list(x.shape[:-1])
    approx = x.reshape(-1, x.shape[-1])
    w = filters.flip(-1).unsqueeze(1).to(approx.dtype)
    details = List[Tensor]()
    for _ in range(nl):
        pad = (2 * fl - 3) // 2
        p = F.pad(approx.unsqueeze(1), [pad, pad + approx.shape[-1] % 2], mode="circular")
        u = F.conv1d(p, w, stride=2).unbind(1)
        approx = u[0]
        details.append(u[1])
    details.reverse()
    return torch.cat([approx] + details, dim=-1).reshape(lead + [-1])
