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

## Time (S025 / S050, Apple M-series host, interleaved runs, load 3.8-6.9)

wall / user / sys seconds; every Typhon row's log is identical to the torch row above it.

| run | S025 | S050 |
|---|---|---|
| torch, 1 thread | 60.4 / 59.2 / 0.6 | 33.9 / 32.8 / 0.5 |
| Typhon, OMP=1, 1 worker | 64.7 / 64.3 / 0.3 | 34.1 / 33.9 / 0.2 |
| Typhon, OMP=1, 2 workers | 40.0 / 64.6 / 3.0 | 22.3 / 35.8 / 1.8 |
| Typhon, OMP=1, 4 workers | 36.7 / 78.5 / 15.6 | 17.8 / 39.6 / 6.8 |
| torch, 5 threads (default) | 110.8 / 134.0 / 183.6 | 56.9 / 69.7 / 95.3 |
| Typhon, default (OMP=5, 4 workers) | 35.9 / 77.3 / 15.3 | 19.7 / 42.0 / 8.5 |
| Typhon, OMP=5, 1 worker | 61.7 / 61.4 / 0.3 | 33.9 / 33.5 / 0.3 |
