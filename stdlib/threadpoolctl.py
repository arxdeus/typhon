# Ported from threadpoolctl 3.x (BSD-3-Clause)
# API shim: Typhon manages its own threads, so limits are accepted and ignored.

class threadpool_limits:
    limits: int

    def __init__(self, limits: int = 0, user_api: str = ""):
        self.limits = limits

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def unregister(self):
        pass

    def restore_original_limits(self):
        pass

def threadpool_info() -> List[str]:
    return List[str]()
