#
# Thin re-export of numpy's legacy RandomState (stdlib/numpy/random/legacy.py).
# braindecode reaches it through sklearn.utils.check_random_state; with
# random_state=None it uses numpy's global legacy stream (np.random.seed).

from numpy.random.legacy import RandomState, check_random_state, _rand


def _global() -> RandomState:
    return _rand


def seed(s: int):
    """np.random.seed equivalent (kept for braindecode.util.set_random_seeds)."""
    _rand.seed(s)
