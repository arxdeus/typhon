class ABCMeta:
    pass


class ABC:
    pass


def abstractmethod(funcobj):
    """Mark a method abstract (identity: abstract-instantiation checks are not enforced)."""
    return funcobj


@__attribute__
def abstractproperty():
    pass


def get_cache_token() -> int:
    return 0
