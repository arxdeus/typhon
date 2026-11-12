# Typhon benchmark comparison (2026-11-12T21:09:18)

Machine: Apple M5 Pro, 15 logical CPUs, 24 GiB, macOS 26.4.1. Python 3.14.7, Cython 3.3.1, Apple clang version 21.0.0 (clang-2100.1.1.101), codon 0.20.6, typhon 0.1.0, git a5b41837.

Times are the program's internal timer (same region for every backend). Median of 5 runs after 1 warm-up for Codon/Typhon/typed Cython, median of 3 for CPython/plain Cython. Machine was shared and loaded, see load averages below.

## Runtime, median seconds (min in parentheses)

| benchmark | args | CPython 3.14 | Cython (plain .py) | Cython (typed .pyx) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|---|---|
| binary_trees | 17 | 7.991 (7.672) | 4.383 (4.382) | 1.054 (1.020) | 0.327 (0.326) | 0.398 (0.387) |
| chaos | {out} | 8.631 (8.590) | 8.754 (8.725) | 1.546 (1.520) | 0.716 (0.705) | 0.608 (0.603) |
| float | 10000000 | 6.709 (6.692) | 5.332 (5.328) | 0.544 (0.539) | 0.459 (0.440) | 0.559 (0.531) |
| set_partition | 14 | 6.191 (6.158) | 4.120 (4.090) | 2.679 (2.647) | 0.974 (0.964) | 0.772 (0.761) |

## Speedup vs CPython (median / median)

| benchmark | Cython (plain .py) | Cython (typed .pyx) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|
| binary_trees | 1.8x | 7.6x | 24.5x | 20.1x |
| chaos | 1.0x | 5.6x | 12.0x | 14.2x |
| float | 1.3x | 12.3x | 14.6x | 12.0x |
| set_partition | 1.5x | 2.3x | 6.4x | 8.0x |
| **geomean** | **1.4x** | **5.9x** | **12.9x** | **12.9x** |

## Typhon / Codon ratio (>1 = Typhon slower)

| benchmark | runtime (internal) | process wall | compile time |
|---|---|---|---|
| binary_trees | 1.217 | 1.175 | 0.940 |
| chaos | 0.849 | 0.841 | 0.967 |
| float | 1.219 | 1.188 | 0.974 |
| set_partition | 0.793 | 0.788 | 1.004 |
| **geomean** | **0.999** | **0.980** | **0.971** |

## Compile time, median of 3 builds (s)

Codon/Typhon: `build -release -exe`. Cython: cythonize + clang -O3 + link (setup.py build_ext).

| benchmark | Cython (plain .py) | Cython (typed .pyx) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|
| binary_trees | 1.46 | 1.42 | 3.48 | 3.27 |
| chaos | 2.40 | 1.94 | 5.29 | 5.12 |
| float | 1.01 | 1.01 | 3.15 | 3.07 |
| set_partition | 1.08 | 1.02 | 3.25 | 3.26 |
| **geomean** | **1.40** | **1.30** | **3.70** | **3.60** |

## Process wall time, median (s)

| benchmark | CPython 3.14 | Cython (plain .py) | Cython (typed .pyx) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|---|
| binary_trees | 8.024 | 4.417 | 1.084 | 0.343 | 0.403 |
| chaos | 8.659 | 8.830 | 1.572 | 0.730 | 0.613 |
| float | 6.734 | 5.358 | 0.569 | 0.479 | 0.569 |
| set_partition | 6.217 | 4.143 | 2.703 | 0.986 | 0.777 |

## Output match vs CPython and load

| benchmark | mismatching backends | uptime before | uptime after |
|---|---|---|---|
| binary_trees | none | 5.74 5.90 6.03 | 6.10 5.97 6.04 |
| chaos | cython_typed, codon, typhon | 6.10 5.97 6.04 | 2.93 5.04 5.67 |
| float | none | 2.93 5.04 5.67 | 2.02 4.28 5.33 |
| set_partition | none | 2.02 4.28 5.33 | 2.13 3.77 5.03 |
