# generates expected values for test_final3.py with real numpy
import numpy as np
m = np.ma.masked_array([1., 2, 3, 4], mask=[0, 1, 0, 0])
print(np.ma.append(m, [5., 6.]).filled(-1).tolist(), np.ma.append(m, [5., 6.]).mask.tolist())
print(np.ma.in1d(m, [1., 4.]).filled(False).tolist() if hasattr(np.ma, 'in1d') else '')
print(np.ma.round_(np.ma.masked_array([1.26, 2.5], mask=[0, 1]), 1).filled(0).tolist())
