
def _or(v, default):
    """Optional[T] or default -> T (Typhon does not narrow `x if x is not None`)."""
    if v is None:
        return default
    elif isinstance(v, Optional):
        return v.__val__()
    else:
        return v
