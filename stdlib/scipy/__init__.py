
# Native Codon implementation of a SciPy subset. Submodules (scipy.linalg,
# scipy.signal, scipy.stats, scipy.special, scipy.fft) are imported on demand,
# as in CPython, so `import scipy` alone stays cheap to compile.

__version__: Literal[str] = "1.18.1"
