# Robot-BCI patient fine-tuning in Typhon

`examples/Robot-BCI-Movement-GUI/src/models/models_finetuning.py` (user code) runs in
Typhon as `models_finetuning_typhon.py`: a copy next to the original that differs only on
the lines listed in its header (8 `.numpy()` call sites given an explicit dtype/rank and an
optional `models_dir=` for on-device use). `test/robot/check_typhon_copies.py` checks that
nothing else changed.

It builds with the release compiler (f6b624c2 or later) and needs no other changes: the
default `target_channels` and no `TYPHON_PATH`, from a driver next to `src/`. The compiler
features this relies on are:
- implicit `Optional` for `x: list = None`
- imports resolved from the main script's directory
- int to float widening of function locals (`best_val_f1 = 0`, later assigned a float;
  `correct_val += t.item()`)

## Parity (S025_test / S050_test `combined_record.edf`, torch 2.15.0, mne 1.13.2, sklearn 1.9.1)

- Data pipeline (`prepare_patient_loaders_from_edf`): bit-exact. The epoch tensors, labels
  and all 4 masked sub-loaders hash identically (`test_finetune_data.py`). Fixes needed:
  `Raw.filter` filters each `skip_by_annotation` segment ("EDGE boundary") separately, the
  float32 `sum` uses ATen's cascade order, and `var`/`std` use float64 Welford.
- RNG streams: identical to Python for `seed_everything`, Linear init, DataLoader shuffle,
  `rand`, `randn_like`, `randint`, `np.random.beta`, `randperm` and Dropout
  (`test_finetune_rng.py`).
- Training (`train_final_model`, default epochs): the full S025 and S050 logs are identical
  to torch at the same `torch.get_num_threads()` (1 and 5 checked), and identical across
  Typhon worker counts 1/2/4/5. Kernels that follow ATen exactly: NNPACK ft8x8 conv
  (N >= 16) and Slow2d conv (N < 16), BatchNorm train and eval, ELU (with ATen's
  parallel_for chunking), weighted cross_entropy (fma accumulation), last-dim softmax,
  clip_grad_norm_ (ATen norm kernel), Adam.

## Threads

Two separate settings:
- `torch.set_num_threads` / `torch.get_num_threads()` (default `OMP_NUM_THREADS`, else the
  performance-core count, as torch). This only changes results where torch's own results
  depend on it (channels-last BatchNorm buffers, ELU chunking), so a run matches torch at
  the same value.
- Worker threads for Typhon's kernels (native NNPACK conv, ELU, BatchNorm per-channel
  loops). These never change results. `TYPHON_NUM_WORKERS` sets them. Default:
  `min(torch.get_num_threads(), 4)` on desktop (a 5th worker only adds spin time); on
  Android/iOS the big + mid core count (iOS `hw.perflevel0.physicalcpu`; Android the CPUs
  whose `cpuinfo_max_freq` is above the slowest cluster). Single-input-channel conv layers
  always use 1 worker.

## Time (S025 / S050, Apple M-series host, e0d0d386)

Interleaved runs, 3 repetitions each, medians; load 3.2-7.1. wall / user / sys seconds.
Every Typhon run's log is identical to the torch run at the same thread count (60/60).

| run | S025 | S050 |
|---|---|---|
| torch, 1 thread | 59.1 / 57.8 / 0.6 | 34.0 / 32.8 / 0.5 |
| Typhon, OMP=1, 1 worker | 56.6 / 56.1 / 0.4 | 31.4 / 31.1 / 0.2 |
| Typhon, OMP=1, 2 workers | 37.3 / 59.4 / 3.1 | 20.8 / 32.8 / 1.8 |
| Typhon, OMP=1, 4 workers | 35.3 / 72.9 / 15.9 | 19.7 / 40.2 / 9.0 |
| torch, 5 threads (default) | 110.0 / 132.9 / 182.7 | 62.6 / 74.1 / 102.0 |
| Typhon, default (OMP=5, 4 workers) | 35.7 / 73.8 / 16.0 | 19.9 / 40.8 / 9.0 |
| Typhon, OMP=5, 1 worker | 56.7 / 56.2 / 0.4 | 31.0 / 30.7 / 0.2 |

At 1 worker Typhon uses less CPU than torch at 1 thread; by default it finishes about 3x
sooner than default torch with a quarter of its CPU time.

## Pixel 9a (device)

Android 16, f1d63717, `finetune_harness` over adb, battery 33-34 C. Android defaults:
`torch.get_num_threads()` = online CPUs (8), workers = big + mid cores (4). The matching torch
reference is torch at 8 threads (the host library is also exact against it: S050 0/27).

| run | wall / user / sys s | lines differing from torch (same thread count) |
|---|---|---|
| S050 default (OMP 8, 4 workers) | 82.3 / 296.2 / 5.3 | 5 / 27 |
| S050 OMP 1, 1 worker | 45.3 / 43.2 / 1.9 | 4 / 27 |
| S025 default | 141.5 / 514.2 / 8.5 | 1 / 51 |
| torch host, 8 threads | S050 82.7 / 93.2 / 231.6, S025 145.9 / 169.3 / 416.3 | |

All differences are in the 4th decimal of a loss or one F1 value. Early-stop epochs and the
best F1 are the same. The device result does not depend on the thread count. Causes found:
bionic `logf`/`cosf`/`sinf` rounding (changes `randn` values), plus a non-elementwise
difference that remains when both platforms use identical libm results, not the BLAS `sgemm`
(see below). See
`examples/robot_bci/README.md`.

Follow-up (robot-device-parity):
- sgemm is not the cause. Linking the same naive sequential-fma `sgemm` into both the host and the
  device library (all `cblas_sgemm` calls), the device still differs from the host in 3/27 (OMP 1)
  and 2/27 (OMP 8) lines. The real platform difference is elsewhere in the step (libm and the
  remaining float paths).
- torch itself is platform-dependent. torch 2.15 linux-aarch64 (podman VM, glibc 2.43, bundled
  OpenBLAS 0.3.34 neoversen1, oneDNN 3.13 + ACL) differs from torch-macOS in S050 3/27 (8 threads)
  and 7/27 (1 thread) lines. Its own 1-vs-8-thread logs differ in 5/27 lines (S050) and 2/51 (S025).
  Conv1d forward and weight-grad bits differ between the two torch builds even with mkldnn disabled.
  Linear, mm, ELU, exp and log give the same bits on both.
- glibc 2.43 and bionic give identical bits for `logf`/`sinf`/`cosf`/`expf`/`sqrtf` (FNV hash
  over every 7th float32 bit pattern), so `randn` agrees on both. `expm1f`/`tanhf`/`log1pf`
  differ (Typhon's ELU uses SLEEF for these, libm only on n % 4 tails).
- Device vs torch-linux-aarch64: S050 4/27 (OMP 1 vs 1 thread), 6/27 (default vs 8); S025
  0/51 vs 1 thread, 1/51 vs 8 threads. The device does not match any torch build exactly,
  and no torch reference is thread-count-independent, while Typhon is by design. All
  differences stay in the 4th decimal of a loss or one F1 value; early stop and best F1 are the same.

Divergence bisect (robot-device-bisect, device run 2026-11-15, battery 31.6-32.9 C). A Typhon-built
probe hashes (FNV-1a over float32 bit patterns) every intermediate of one training step: randn,
randn_like, rand masks, augment, per-module forward, loss, every parameter grad, one Adam step,
BN buffers. It also runs each module in isolation on `rand` inputs (no libm) plus elementwise
exp/expm1/log/tanh/log1p/ELU/softmax/CE.
- Native libraries (as shipped): the first differing tensor is `randn` (libm logf/sinf/cosf). In
  isolation, conv weight grads (sgemm), Linear forward (sgemm) and exp/expm1/log/tanh/log1p (libm)
  differ. Softmax, CE, BatchNorm, ELU's SLEEF part, pooling and dropout masks are identical.
- With the same naive sequential-fma `sgemm` and the same float libm (`(float)<double fn>`,
  including Apple's `__sincosf_stret`, which LLVM emits for sin+cos pairs on macOS) linked into
  both binaries, every hash of the step (93 lines), the isolated modules and the Adam step is
  identical on host and device at OMP 1 / 1 worker, OMP 8 and OMP 8 / 2 workers.
So the only sources of divergence are native BLAS and libm rounding. No flag or code-path
difference in our code (runtime C kernels: SLEEF `-ffp-contract=off`, NNPACK; Typhon codegen;
stdlib) changes bits between android-arm64 and macOS. The only platform checks in stdlib/torch
pick thread and worker counts, and results do not depend on those.

Policy: each platform uses its native libraries (Apple libm + Accelerate on macOS/iOS, bionic libm
+ OpenBLAS on Android). Inference labels and command sequences are identical. Fine-tune logs agree
with torch-mac except at the float32-ulp level (last printed digit of a loss or one F1 value).
Not yet measured on the device (the device went offline): final-weight max abs/rel diff, val-loss
diff at full precision, and test-window label agreement of the fine-tuned model.

Fine-tune magnitude, device vs torch-mac (S050, OMP 1 / 1 worker, native libraries as shipped,
device run 2026-11-15, battery 35.7-35.9 C, 58 s wall). Weights were compared on the device against
the torch-mac 1-thread state_dict. Val loss is printed with repr() in a scratch copy. The host
Typhon build is bit-exact against the same reference: 0 weight diffs, same val losses, same probs.
- Final weights: 3374087 of 4249442 values differ. Max abs diff 1.21e-4 (motor_classifier.0.bias).
  Max rel diff 4.7, on near-zero values (fwd_model.spatial_conv.0.bias). Abs diff stays <= 1.3e-4 everywhere.
- Val loss per epoch (22 epochs): max abs diff 2.29e-5 (rel 1.8e-5). Early stop is the same (epoch 12,
  best F1 0.5790).
- Fine-tuned model on the 1466 S050_test windows: labels 1466/1466 identical, max |dprob| 3.5e-5.
Not measured: default thread count, S025, torch linux-aarch64.

Device runs 2026-11-15 (same tool, native libraries as shipped, battery 33-36 C):
- S050, OMP 8 (default 4 workers on the big+mid cores): 78.9 s wall / 274.6 s user. With 1 worker:
  88.5 s / 213.9 s. The OMP 1 run above took 58 s wall, faster than OMP 8 (see the profile below).
- Worker invariance (S050, OMP 8, default workers vs TYPHON_NUM_WORKERS=1): the final-weight hash,
  all val losses (repr) and the test probs are bit-identical.
- S050 OMP 8 vs torch-mac 8 threads: 3365827 of 4249442 weights differ, max abs 1.02e-4
  (motor_classifier.0.bias). Val loss max abs 1.37e-5. Same early stop (epoch 12, best F1 0.5790).
  Labels 1466/1466, max |dprob| 2.6e-5.
- S025 OMP 1 vs torch-mac 1 thread: 3570429 weights differ, max abs 1.87e-4
  (left_model.spatial_conv.0.bias). Val loss max abs 1.59e-5 over 46 epochs. Same early stop as
  torch (epoch 36, best F1 0.3597). Labels 1465/1466, max |dprob| 2.0e-5. The one mismatch,
  window 1268, is a near-tie between classes 3 and 4: torch 0.253497 / 0.253503,
  device 0.253501 / 0.253500.

Profile (simpleperf, cpu-cycles:u, first 20 s of S050 at OMP 8). All compute hot spots are native
vector kernels:
| share | symbol | what |
|---|---|---|
| 45.0% | inner_thread | OpenBLAS level-3 thread driver (driver/level3/level3_thread.c): sync and spin-wait between BLAS threads |
| 14.0% + 3.9% | nnp_c4gemm / nnp_s4c2gemm `__neon` | NNPACK FFT conv GEMM, NEON |
| 5.5% + 3.9% | nnp_ifft8x8 / nnp_fft8x8 `__psimd` | NNPACK transforms, psimd (NEON vectors) |
| 4.2% | Sleef_expm1f4_u10advsimd | ELU, SLEEF AdvSIMD |
| 3.4% + 1.7% + 1.1% | thread_main, typhon_omp barrier/take | NNPACK pthreadpool and Typhon workers, idle/sync |
| 2.6% | sgemm_kernel | OpenBLAS NEON sgemm micro-kernel |
No scalar fallback is hot: the largest Typhon-code addresses are each below 0.4%. The hot spot is
OpenBLAS's threaded small-GEMM synchronisation. Real sgemm work is 2.6%, but the threads spinning
around it take 45%, and that is why OMP 8 is slower than OMP 1 on wall time. torch uses the same
OpenBLAS threading, and ATen gives the thread count to BLAS. Changing it would change neither
results nor bits, only speed: a possible follow-up is fewer BLAS threads for small GEMMs.

OpenBLAS small-GEMM threading (Android only). Root cause of the slow OMP 8 run: OpenBLAS 0.3.32 on
generic ARMV8 (interface/gemm.c get_gemm_optimal_nthreads) threads every GEMM with M*N*K > 65536*4
over omp_get_max_threads() = 8, little cores included. driver/level3/level3_thread.c then
spin-waits on the slowest thread. Device check before the fix (S050, scratch build that sets
BLAS threads via omp_set_num_threads, battery 33-35 C):

| OMP | BLAS threads | wall | user |
|---|---|---|---|
| 8 | 8 (as shipped) | 77.6 s | 190 s |
| 8 | 4 | 73.0 s | 165 s |
| 8 | 1 | 68.8 s | 155 s |

The weight hash, val losses and probs are identical in all three rows, and equal to the runs
above. The fix is crates/runtime/build.rs: `GEMM_MULTITHREAD_THRESHOLD=1024` for the
aarch64-android OpenBLAS build, so GEMMs up to 2^26 M*N*K run on the caller thread. Apple and
Accelerate are not touched.

Device re-measure, not yet measured on device (the Pixel was unplugged):
- [ ] Rebuild with `tools/build-runtime.sh aarch64-linux-android` and rebuild the probe or library.
- [ ] S050 OMP 8 (default workers) and OMP 1 (1 worker): wall/CPU, battery temperature.
- [ ] Bits: WHASH -0x4420ac99c30d7d12 (OMP 8), val-loss md5 268d858e..., probs md5 12cf0c40...
  must be unchanged. Also OMP 8 with TYPHON_NUM_WORKERS=1.
- [ ] simpleperf: inner_thread should drop from 45%.
- [ ] Separate check: with OMP 1 but the default workers, the run took 90 s vs 58 s with
  1 worker. The NNPACK and typhon-omp pools also lose time on this SoC (cores 0-3 are 1.95 GHz).
  Measure workers pinned to cores 4-7 (`taskset f0`) before changing `_fast_cores`.

### Float libm on host vs device (native libraries)

Host and device each use their native libm (Apple libm on macOS/iOS, bionic on Android). Neither is correctly rounded, and they differ from each other, so the last digits of Val Loss can differ on the device. Exhaustive check over all 2^32 float inputs, with CORE-MATH (correctly rounded) as the reference. NaN==NaN counts as equal. Apple is `libsystem_m` in-process. Bionic is modelled by host builds of its upstream sources: ARM optimized-routines for logf/sinf/cosf/expf, FreeBSD msun for expm1f/tanhf/log1pf, built with `-O2 -ffp-contract=off`. The device was not used, so device-build details such as FMA contraction may move the bionic counts a little.

| fn | Apple != CR | Apple max ulp | bionic != CR | bionic max ulp | Apple != bionic | max ulp |
|---|---|---|---|---|---|---|
| logf | 13540 (0.0003%) | 1 | 416908 (0.0097%) | 1 | 420256 (0.0098%) | 1 |
| sinf | 346516414 (8.07%) | 1 | 29362804 (0.68%) | 1 | 335149902 (7.80%) | 1 |
| cosf | 295141874 (6.87%) | 1 | 28209648 (0.66%) | 1 | 282252698 (6.57%) | 1 |
| expf | 2041856 (0.048%) | 1 | 170646 (0.0040%) | 1 | 2048808 (0.048%) | 1 |
| expm1f | 2978469 (0.069%) | 1 | 12920593 (0.30%) | 1 | 14215778 (0.33%) | 1 |
| tanhf | 4421346 (0.10%) | 1 | 70733480 (1.65%) | 2 | 70968362 (1.65%) | 2 |
| log1pf | 2237332 (0.052%) | 1 | 11588705 (0.27%) | 1 | 13571531 (0.32%) | 1 |

Both libraries are within 1 ulp of the correctly rounded result (bionic tanhf: 2 ulp), and they are at most 2 ulp apart. None of the open candidates matches Apple bit for bit over all inputs: CORE-MATH, ARM optimized-routines, musl, FreeBSD msun, and SLEEF u10/u35 all differ on 13540 or more inputs for logf, and on 2.8e8 or more for sinf/cosf. So a device/host difference of 1-2 ulp in these functions is expected, and the results are not bit-identical across platforms.
