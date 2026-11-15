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
difference that remains when both platforms use identical libm results, most likely
OpenBLAS `sgemm` on Android versus Accelerate in the host torch reference. See
`examples/robot_bci/README.md`.
