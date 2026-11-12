# Typhon benchmark comparison (2026-11-12T21:01:17)

Machine: Apple M5 Pro, 15 logical CPUs, 24 GiB, macOS 26.4.1. Python 3.14.7, Cython 3.3.1, Apple clang version 21.0.0 (clang-2100.1.1.101), codon 0.20.6, typhon 0.1.0, git a5b41837.

Times are the program's internal timer (same region for every backend). Median of 5 runs after 1 warm-up for Codon/Typhon/typed Cython, median of 3 for CPython/plain Cython. Machine was shared and loaded, see load averages below.

## Runtime, median seconds (min in parentheses)

| benchmark | args | CPython 3.14 | Cython (plain .py) | Cython (typed .pyx) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|---|---|
| binary_trees | 17 | 7.813 (7.601) | 4.210 (4.184) | 1.012 (0.974) | 0.323 (0.318) | 0.392 (0.380) |
| chaos | {out} | 8.898 (8.790) | 9.230 (9.032) | 1.625 (1.619) | 0.866 (0.853) | 0.743 (0.720) |
| float | 10000000 | 7.368 (7.336) | 6.013 (5.966) | 0.571 (0.564) | 0.387 (0.381) | 0.438 (0.411) |
| set_partition | 14 | 6.508 (6.397) | 4.389 (4.383) | 2.659 (2.653) | 0.991 (0.982) | 0.789 (0.762) |

## Speedup vs CPython (median / median)

| benchmark | Cython (plain .py) | Cython (typed .pyx) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|
| binary_trees | 1.9x | 7.7x | 24.2x | 19.9x |
| chaos | 1.0x | 5.5x | 10.3x | 12.0x |
| float | 1.2x | 12.9x | 19.0x | 16.8x |
| set_partition | 1.5x | 2.4x | 6.6x | 8.2x |
| **geomean** | **1.3x** | **6.0x** | **13.3x** | **13.5x** |

## Typhon / Codon ratio (>1 = Typhon slower)

| benchmark | runtime (internal) | process wall | compile time |
|---|---|---|---|
| binary_trees | 1.214 | 1.182 | 0.983 |
| chaos | 0.858 | 0.844 | 1.038 |
| float | 1.131 | 1.087 | 0.963 |
| set_partition | 0.796 | 0.789 | 0.997 |
| **geomean** | **0.984** | **0.962** | **0.995** |

## Compile time, median of 3 builds (s)

Codon/Typhon: `build -release -exe`. Cython: cythonize + clang -O3 + link (setup.py build_ext).

| benchmark | Cython (plain .py) | Cython (typed .pyx) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|
| binary_trees | 1.40 | 1.36 | 3.24 | 3.18 |
| chaos | 2.44 | 1.92 | 5.31 | 5.51 |
| float | 1.09 | 1.18 | 3.49 | 3.36 |
| set_partition | 1.26 | 1.17 | 3.74 | 3.72 |
| **geomean** | **1.47** | **1.38** | **3.87** | **3.85** |

## Process wall time, median (s)

| benchmark | CPython 3.14 | Cython (plain .py) | Cython (typed .pyx) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|---|
| binary_trees | 7.843 | 4.249 | 1.042 | 0.337 | 0.399 |
| chaos | 8.936 | 9.264 | 1.659 | 0.888 | 0.749 |
| float | 7.400 | 6.048 | 0.599 | 0.416 | 0.452 |
| set_partition | 6.533 | 4.416 | 2.685 | 1.007 | 0.794 |

## Output match vs CPython and load

| benchmark | mismatching backends | uptime before | uptime after |
|---|---|---|---|
| binary_trees | none | 6.91 6.70 6.28 | 6.08 6.48 6.23 |
| chaos | cython_typed, codon, typhon | 6.08 6.48 6.23 | 6.06 6.33 6.20 |
| float | none | 6.06 6.33 6.20 | 5.89 6.21 6.16 |
| set_partition | none | 5.89 6.21 6.16 | 5.35 5.97 6.07 |
