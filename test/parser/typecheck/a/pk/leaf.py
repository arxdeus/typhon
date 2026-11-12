from . import rel
from .rel import r
def g():
    return 'pk.leaf.g ' + rel.r() + ' ' + r()
