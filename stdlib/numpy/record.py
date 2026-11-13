# Ported from numpy 2.5.4 (BSD-3-Clause)
#
# numpy.record / recarray: column-oriented record array with attribute access
# via field(name).

import numpy as np

class record:
    names: List[str]
    values: List[float]

    def __init__(self, names: List[str], values: List[float]):
        self.names = names
        self.values = values

    def __getitem__(self, k: str):
        for i in range(len(self.names)):
            if self.names[i] == k:
                return self.values[i]
        raise KeyError(k)

class recarray:
    names: List[str]
    cols: List[np.ndarray]

    def __init__(self, names: List[str], cols: List[np.ndarray]):
        self.names = names
        self.cols = cols

    def field(self, name: str):
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.cols[i]
        raise AttributeError(name)

    def __len__(self):
        return len(self.cols[0]) if len(self.cols) else 0

    def __getitem__(self, i: int):
        vals = List[float]()
        for c in self.cols:
            vals.append(float(c[i]))
        return record(self.names, vals)
