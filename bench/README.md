# Typhon vs Codon vs Cython vs CPython

`compare.py` compares five backends on the pyperformance-style programs from
`../../bench/codon/`, measuring both runtime and compile time.

| backend | how |
|---|---|
| CPython 3.14 | `python src/<name>.py args` (venv python) |
| Cython (plain) | `cythonize -3` of the unmodified-semantics `.py` (pure Python mode, no type declarations), clang `-O3`, run with `python -c "import sys; sys.argv=[...]; import <name>"` |
| Cython (typed) | hand-typed `cython_typed/<name>.pyx` (separate agent, see its README), clang `-O3 -ffp-contract=off`, same driver |
| Codon (C++) | `build/codon build -release -exe`, then run the binary (`CODON_PATH=stdlib`) |
| Typhon (Rust) | `typhon/target/release/typhon build -release -exe`, then run the binary |

```sh
python3 -m venv $JCODE_SCRATCH_DIR/cyenv   # built with python3.14
$JCODE_SCRATCH_DIR/cyenv/bin/pip install cython setuptools
python3 compare.py check            # correctness, tiny args
python3 compare.py measure          # -> results/<date>.json + .md (resumable)
python3 compare.py report results/<date>.json
```

## Sources (`src/`)

These are copies of `bench/codon/*.py`. The `.codon` variants are not used because
they rely on `@par`/GPU, which would be unfair to Cython. Changes from the originals:

* Problem sizes are parameterized only where they were hard-coded:
  `mandelbrot` N = argv[1], `spectral_norm` loops = argv[1], `float` POINTS = argv[1].
  The defaults are unchanged.
* `go.py`: `if __name__ == "__main__"` became `if True` so that the Cython
  import driver runs it.
* **C++ Codon cannot compile the original `.py` files** for binary_trees, chaos and go
  (it fails to type the untyped recursive class fields). I added only the
  annotations and tweaks that the repository's own `.codon` variants use: class field
  annotations, `length = 0.`, `float(white_dead)`, `maxscore = -1.`,
  `__rmul__` written as a method, chaos `transform_point` without the never-used
  `trafo` argument, and go `class Square[Board]` (PEP 695 generics). The algorithms
  are unchanged and CPython output is identical.
* Cython rejects PEP 695 class generics, so plain Cython compiles `src/go_cy.py`,
  which differs from `go.py` only in those annotations.

Sizes were chosen so that CPython takes about 6-19 s per run: binary_trees 17,
chaos (fixed, 2048² and 1e6 iters), fannkuch 10, float 1e7, go (fixed),
mandelbrot 512, nbody 3e6, primes 30000, set_partition 14, spectral_norm 40 loops.

## NumPy set (`src_np/`, `compare.py --set numpy check|measure`)

Seven programs. Each `.py` is used unchanged by every backend: Codon/Typhon build it,
CPython+NumPy runs it, and plain Cython cythonizes it (NumPy calls still go through Python).
There are no typed Cython variants. BLAS-heavy: np_matmul (2048², 60 reps),
np_solve_chol (solve + cholesky 2000, 40 reps), np_eigh_svd (1000, 25 reps).
Per-call overhead: np_small_linalg (5e5 iterations of 4x4 matmul/inv plus 8x8 solve/dot).
From npbench: np_arc_distance (ufunc chain, 1e7 x 15), np_jacobi_2d (700², 2000 steps),
np_covariance (4000x5000). Each size is chosen so CPython+NumPy takes about 2-5 s.
Results go to `results/<date>-numpy.{json,md}` (override the prefix with `BENCH_TAG`).
On macOS, Codon is built with `-linker-flags "-framework Accelerate"`, because its NumPy
calls ILP64 Accelerate symbols and does not link the framework itself. Typhon links Accelerate itself.
CPython uses NumPy 2.5.4 wheels, which also use Accelerate, so every backend uses the same BLAS.
The float tolerance is 1e-5 relative.

## Latest results

`results/2026-11-12-v3-summary.md` (typhon 27acf1fc): scalar Typhon/Codon geomean
0.979 (23.2x vs CPython, Codon 22.7x), NumPy set 1.013 (np_arc_distance 1.07, was 1.32).
Full tables in `results/2026-11-12-v3{,-numpy}.md`, memory in
`results/memory-2026-11-12-v3-final.md, memory-2026-11-12-v3final-numpy.md` (`memory.py --set numpy` for the NumPy set).

## Method

* Timings: each program's internal timer (its last printed line) is the primary
  metric, and the region it covers is identical across backends. Total process
  wall time is also recorded.
* Strictly serial runs. For every benchmark the backend order rotates, and runs
  are interleaved round-robin (one run per backend per round). Codon, Typhon and
  typed Cython get 1 warm-up and 5 measured runs. CPython and plain Cython get
  3 runs. The report gives the median and the min.
* Compile time is the median of 3 builds: `build -release -exe` for
  Codon/Typhon, and `setup.py build_ext` for Cython (cythonize + clang + link).
* `uptime` is recorded before and after each benchmark. The machine is shared and
  loaded (load avg ~7-17 from unrelated processes), so expect noise of roughly 10-20%.
* Correctness: `check` compares outputs (minus the timing line) to CPython,
  with float tolerance 1e-6 relative. chaos is compared via the PPM sha256.

## Memory harness (`memory.py`)

`python3 bench/memory.py --typhon-dir DIR --label L [--cpython] [names]` builds
every benchmark plus `empty`/`hello` with Codon and with `DIR/typhon`, retargets
Typhon exes to `@rpath/libtyphonrt.dylib` with rpath `DIR` (so a snapshot dylib is
really used; verify with `DYLD_PRINT_LIBRARIES=1`), and reports median-of-3 peak
RSS and peak memory footprint from `/usr/bin/time -l`, serial, round-robin.
Output: `results/memory-<date>-<label>.{json,md}`.

Baseline findings (pre-gcfix snapshot): startup overhead is not worse than Codon
(hello 6.8 vs 7.2 MiB RSS, footprint 2.4 MiB both, 1 thread each, no omp threads).
Outliers are heap-policy related: chaos 143 vs 57 MiB, binary_trees 57 vs 46,
spectral_norm 15.5 vs 13.1. libtyphonrt.dylib 3.77 MB vs 2.2 MB, but the extra
~1.5 MB of text is mapped lazily and does not show in RSS/footprint.
