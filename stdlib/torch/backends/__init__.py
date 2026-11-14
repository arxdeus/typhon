# Ported from torch 2.15.0 (BSD-3-Clause)
# CPU-only build: cuDNN is never available. `torch.backends.cudnn` is an object
# (not a module) so user code can assign its flags (`cudnn.deterministic = True`).

class _CudnnModule:
    enabled: bool
    benchmark: bool
    deterministic: bool
    allow_tf32: bool

    def __init__(self):
        self.enabled = True
        self.benchmark = False
        self.deterministic = False
        self.allow_tf32 = True

    def is_available(self) -> bool:
        return False

    def version(self):
        return None

cudnn = _CudnnModule()
