//! Rust implementation of the OpenMP runtime ABI (`__kmpc_*`, `omp_*`) that
//! Codon-compiled code and `stdlib/openmp.codon` call. Replaces the GC-patched
//! LLVM `libomp` that the C++ build links (omp lane: see ../../DESIGN.md).
