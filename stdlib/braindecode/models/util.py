# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch.nn as nn

def _bn_off_if_batch_one(model, batch: int) -> List[nn.Module]:
    # Temporarily set BatchNorm layers to eval for batch size one (training).
    out = List[nn.Module]()
    if model.training and batch == 1:
        for m in model.modules():
            if m.training and "BatchNorm" in m._get_name():
                m.training = False
                out.append(m)
    return out

def _bn_restore(mods: List[nn.Module]):
    for m in mods:
        m.training = True
