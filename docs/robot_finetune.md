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
- Training (`train_final_model`, default epochs): all 4 submodel results (Accuracy / F1)
  match exactly. Every final-classifier epoch line matches, and the same early stop happens
  at the same epoch. The saved checkpoint loads in real torch and has the same 165 keys.
  The only differences are in the 4th decimal: 2 lines on S025 and 6 lines on S050, all
  ±0.0001 in loss, plus one F1 at S050 epoch 4 (0.0747 vs 0.0762). The first divergence
  is 1 ulp in the first forward conv layer (batched depthwise Conv1d). Weights end up
  within 2.3e-4.
- CPU time of the full run (user+sys), Typhon vs torch:
  - S025: Typhon 27.6 s on release f6b624c2 (an earlier build: 41.6 s). Torch at 1 thread:
    98.2 s. Torch at default threads: 282 s (106 s wall).
  - S050: Typhon 22.3 s. Torch at default threads: 140 s.
