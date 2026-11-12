# Ported from braindecode 2.0.0 (BSD-3-Clause)

import torch
from torch import Tensor
import torch.nn.functional as F

def _check_on_non_divisible(on_non_divisible: str):
    if on_non_divisible not in ("pad", "crop", "error"):
        raise ValueError(f"on_non_divisible must be 'pad', 'crop', or 'error', got {on_non_divisible!r}.")

def patch_count(n_times: int, patch_size: int, on_non_divisible: str = "pad", step: int = 0) -> int:
    if step <= 0:
        step = patch_size
    if n_times < 1:
        raise ValueError(f"n_times must be positive, got {n_times}.")
    if n_times < patch_size:
        if on_non_divisible == "pad":
            return 1
        if on_non_divisible == "crop":
            raise ValueError(f"Need at least one full patch of {patch_size} samples, got {n_times}.")
    elif (n_times - patch_size) % step == 0:
        return (n_times - patch_size) // step + 1
    elif on_non_divisible != "error":
        return (n_times - patch_size) // step + 1 + (1 if on_non_divisible == "pad" else 0)
    raise ValueError(f"Time dimension ({n_times}) is not divisible by patch_size ({patch_size}).")

def pad_or_crop_to_patches(x: Tensor, patch_size: int, on_non_divisible: str = "pad", step: int = 0,
                           min_patches: int = 1, center: bool = False) -> Tensor:
    n = patch_count(x.shape[-1], patch_size, on_non_divisible, step)
    st = step if step > 0 else patch_size
    left = 0
    if n < min_patches:
        left = (min_patches - n) // 2 * st if center else 0
        n = min_patches
    target = patch_size + (n - 1) * st
    if target > x.shape[-1]:
        return F.pad(x, [left, target - x.shape[-1] - left])
    return x[..., :target]
