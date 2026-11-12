# Typhon benchmark comparison (2026-11-12T21:15:18)

Machine: Apple M5 Pro, 15 logical CPUs, 24 GiB, macOS 26.4.1. Python 3.14.7, Cython 3.3.1, Apple clang version 21.0.0 (clang-2100.1.1.101), codon 0.20.6, typhon 0.1.0, git 2bd60e40.

Times are the program's internal timer (same region for every backend). Median of 5 runs after 1 warm-up for Codon/Typhon/typed Cython, median of 3 for CPython/plain Cython. Machine was shared and loaded, see load averages below.

## Runtime, median seconds (min in parentheses)

| benchmark | args | CPython 3.14 | Cython (plain .py) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|---|
| np_arc_distance | 10000000 15 | 3.118 (3.111) | 3.150 (3.140) | 2.160 (2.149) | 2.887 (2.875) |

## Speedup vs CPython (median / median)

| benchmark | Cython (plain .py) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|
| np_arc_distance | 1.0x | 1.4x | 1.1x |
| **geomean** | **1.0x** | **1.4x** | **1.1x** |

## Typhon / Codon ratio (>1 = Typhon slower)

| benchmark | runtime (internal) | process wall | compile time |
|---|---|---|---|
| np_arc_distance | 1.336 | 1.329 | 0.934 |
| **geomean** | **1.336** | **1.329** | **0.934** |

## Compile time, median of 3 builds (s)

Codon/Typhon: `build -release -exe`. Cython: cythonize + clang -O3 + link (setup.py build_ext).

| benchmark | Cython (plain .py) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|
| np_arc_distance | 0.85 | 6.17 | 5.77 |
| **geomean** | **0.85** | **6.17** | **5.77** |

## Process wall time, median (s)

| benchmark | CPython 3.14 | Cython (plain .py) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|
| np_arc_distance | 3.187 | 3.221 | 2.182 | 2.899 |

## Output match vs CPython and load

| benchmark | mismatching backends | uptime before | uptime after |
|---|---|---|---|
| np_arc_distance | none | 2.13 3.77 5.03 | 1.86 3.26 4.71 |
