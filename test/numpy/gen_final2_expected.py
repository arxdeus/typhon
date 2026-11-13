# generates expected values for test_final2.py with real numpy
import numpy as np, io
m = np.asmatrix([[1., 2], [3, 4]])
print((m * m).tolist(), (m ** 2).tolist(), m.T.tolist(), m.I.tolist())
x = np.array(['a1', 'b22'])
print(np.strings.encode(x).tolist(), np.strings.translate(x, {ord('a'): 'Z'}).tolist() if False else '')
