# Typhon benchmark comparison (2026-11-12T21:07:34)

Machine: Apple M5 Pro, 15 logical CPUs, 24 GiB, macOS 26.4.1. Python 3.14.7, Cython 3.3.1, Apple clang version 21.0.0 (clang-2100.1.1.101), codon 0.20.6, typhon 0.1.0, git a5b41837.

Times are the program's internal timer (same region for every backend). Median of 5 runs after 1 warm-up for Codon/Typhon/typed Cython, median of 3 for CPython/plain Cython. Machine was shared and loaded, see load averages below.

## Runtime, median seconds (min in parentheses)

| benchmark | args | CPython 3.14 | Cython (plain .py) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|---|
| np_arc_distance | 10000000 15 | 3.821 (3.566) | 3.731 (3.637) | 2.611 (2.527) | 3.435 (3.350) |

## Speedup vs CPython (median / median)

| benchmark | Cython (plain .py) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|
| np_arc_distance | 1.0x | 1.5x | 1.1x |
| **geomean** | **1.0x** | **1.5x** | **1.1x** |

## Typhon / Codon ratio (>1 = Typhon slower)

| benchmark | runtime (internal) | process wall | compile time |
|---|---|---|---|
| np_arc_distance | 1.315 | 1.300 | 0.948 |
| **geomean** | **1.315** | **1.300** | **0.948** |

## Compile time, median of 3 builds (s)

Codon/Typhon: `build -release -exe`. Cython: cythonize + clang -O3 + link (setup.py build_ext).

| benchmark | Cython (plain .py) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|
| np_arc_distance | 0.89 | 6.69 | 6.34 |
| **geomean** | **0.89** | **6.69** | **6.34** |

## Process wall time, median (s)

| benchmark | CPython 3.14 | Cython (plain .py) | Codon (C++) | Typhon (Rust) |
|---|---|---|---|---|
| np_arc_distance | 3.969 | 4.171 | 2.654 | 3.451 |

## Output match vs CPython and load

| benchmark | mismatching backends | uptime before | uptime after |
|---|---|---|---|
| np_arc_distance | none | 5.35 5.97 6.07 | 5.74 5.90 6.03 |
