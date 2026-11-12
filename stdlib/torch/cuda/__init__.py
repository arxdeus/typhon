# Ported from torch 2.15.0 (BSD-3-Clause)
# CPU-only build: CUDA is never available.

def is_available() -> bool:
    return False

def device_count() -> int:
    return 0

def manual_seed(seed: int):
    pass

def manual_seed_all(seed: int):
    pass
