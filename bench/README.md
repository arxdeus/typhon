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
