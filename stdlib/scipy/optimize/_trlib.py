# Ported from scipy 1.18.1 _trlib (trlib, MIT License, Copyright (c) 2016--2017 Felix Lenders)

import math
import numpy as np

class _TrlibState:
    n: int

    def __init__(self, n: int):
        self.n = n

def _trlib_solve(m, trust_radius: float):
    raise NotImplementedError('trust-krylov')
    return np.zeros(0), False
