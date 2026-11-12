# Typed Cython benchmark variants

Idiomatic typed Cython ports of `bench/codon/*.py`. Each keeps the same
algorithm, output and timing boundaries as the original. They are
single-threaded, with no prange or OpenMP. Every `.pyx` has
`language_level=3`, `boundscheck=False` and `wraparound=False`, plus
`cdivision=True` where it is safe.

## Build / run

```sh
python3 -m venv $JCODE_SCRATCH_DIR/cyenv_typed
$JCODE_SCRATCH_DIR/cyenv_typed/bin/pip install cython setuptools
PY=$JCODE_SCRATCH_DIR/cyenv_typed/bin/python
$PY build.py $JCODE_SCRATCH_DIR/cyt            # all, or: build.py DIR nbody go
$PY run.py $JCODE_SCRATCH_DIR/cyt nbody 50000000 # run.py BUILD_DIR NAME [argv...]
```

`build.py` runs `cythonize` and compiles with `-O3 -ffp-contract=off`. The flag
stops clang from fusing float ops, so results stay bit-identical to CPython.

**Entry point:** every module runs its benchmark as module-level code when it
is imported, just like the `.py` scripts. `run.py` sets
`sys.argv = [name, *args]` and imports the module. The last printed line is
the elapsed time, measured around the same region as the original.

Env overrides exist only for quick correctness runs. When unset, the
defaults match the originals: `FLOAT_POINTS`, `MANDEL_N`, `SPECTRAL_LOOPS`,
`CHAOS_ITERS`/`CHAOS_SIZE`, `GO_GAMES`/`GO_ROUNDS`/`GO_DEBUG`.

## What was typed

| bench | typing |
|---|---|
| binary_trees | `cdef class Node` with typed `Node` children. `make_tree`/`check_tree`/`make_check` are `cdef`, with a `long long` checksum. The chunk generator is kept, and `sum(map(...))` became a typed loop. |
| chaos | `cdef class GVector` (double x,y,z, cdef methods), `Spline` (C double knot array, cdef de Boor `call`/`GetIndex`), `Chaosgame` (typed fields, cdef `transform_point`/`truncate`). Image rows are `bytearray`. Python `random` is kept so the sequence is identical. **Partly dynamic**: allocating a GVector per step and Python RNG calls dominate. |
| fannkuch | Entirely C: `int*` permutation buffer, C `fact`, cdef `perm`, typed flip loop. |
| float | `cdef class Point` with double fields, cdef `normalize`/`maximize`, libc `sin/cos/sqrt`. The list of 10M objects stays (it is the algorithm). |
| go | `cdef class` Square/EmptySet/ZobristHash/Board/UCTNode with typed int/long/bint fields. EmptySet uses C `int[81]` arrays, the zobrist strings are `long long[3]`, and all methods are cdef with libc `sqrt/log`. **Partly dynamic**: neighbour lists, the hash `set`, UCT child lists and `random` stay Python. |
| mandelbrot | C `int*` pixel buffer and an explicit double re/im loop with `hypot`. CPython's complex multiply (clang, arm64) uses FMA, so `fma()` is used to reproduce it bit-for-bit. Without it, 4 of 40k pixels differ at N=200. |
| nbody | C `struct Body[5]` (pos, vel, mass) in the original order, nogil `advance` with `pow(d2,-1.5)`. |
| primes | Fully C (`long long`, `bint` cdef). |
| set_partition | cdef `conforms` with typed `list`/`Py_ssize_t`, generator with typed locals. **Inherently dynamic**: it is recursive list slicing and concatenation, so the speedup is small. |
| spectral_norm | `double*` vectors, nogil cdef matrix-vector kernels, inline `eval_A`. The list-of-floats `eval_times_u` plumbing became C loops. Same math and order. |

## Correctness (small args, CPython 3.9 vs typed build, timing line ignored)

binary_trees 10, fannkuch 7, primes 3000, set_partition 10, float 20000 pts,
mandelbrot N=200, spectral_norm 2 loops, chaos 20000 iters at 256²
(PPM byte-identical), go 20 games x 5 rounds (chosen moves identical):
all exact. nbody 1000: the first energy is exact, and the second differs in
the last printed digit (1 ulp, from accumulated `pow` rounding).
