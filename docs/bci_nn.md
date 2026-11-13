# Neural nets for BCI through Typhon (EEGNet, ShallowFBCSPNet)

Question: can the BCI mobile app run *and train* its braindecode models natively through
Typhon's `torch`/`braindecode` port instead of exporting to ONNX?

Answer: yes for EEGNet and ShallowFBCSPNet at the notebook's size (n_chans=6, n_outputs=3,
n_times=500, float32, CrossEntropyLoss + Adam). Inference matches PyTorch to float32 rounding,
training runs on-device (no ONNX training graph needed), and speed is on par with
single-threaded PyTorch CPU.

## Parity (same weights, loaded from a PyTorch `state_dict` saved as safetensors)

| model | forward 1 trial, max abs diff | forward batch 64, max abs diff |
|---|---|---|
| EEGNet | 3.7e-9 | 2.5e-8 |
| ShallowFBCSPNet | 1.7e-6 | 2.9e-6 |

Reference: braindecode 2.0.0, torch 2.15.0 CPU. Weights move with
`m.load_state_dict(safetensors.torch.load_file(path))` on both sides (`torch.save`/`load` in
Typhon also uses the safetensors container, not pickle).

## Timing (Apple M-series host, ms; Typhon single-threaded, `-release`)

| | Typhon | PyTorch 1 thread | PyTorch default threads (wall) |
|---|---|---|---|
| EEGNet forward, 1 trial | 0.34 (0.21 via `-lib`) | 0.21 | 0.76 |
| EEGNet forward, batch 64 | 11.7 | 51.0 | 474 |
| EEGNet train step, batch 64 (Adam) | 49 | 71 | 469 |
| EEGNet 5 epochs x 240 trials | 890 | 1425 | 8898 |
| Shallow forward, 1 trial | 0.98 | 0.44 | 0.27 |
| Shallow forward, batch 64 | 19.1 | 17.0 | 4.2 |
| Shallow train step, batch 64 | 49 | 36 | 10.7 |
| Shallow 5 epochs x 240 trials | 995 | 683 | 226 |

PyTorch's multi-threaded EEGNet path is pathological on this machine (thread oversubscription
on the 1-wide depthwise convs). Android arm64 emulator (API 35, host CPU): EEGNet predict 1 trial
0.53 ms, 5 epochs 1.53 s.

Hot spots: convolution is already im2col + BLAS sgemm (`stdlib/torch/_conv.py`, Accelerate on
macOS). What remains on small inputs (ShallowFBCSPNet 1 trial, 2.2x slower than torch) is
im2col copying, `memmove` and GC marking of many small temporary tensors. Fixes, if needed: a
direct (non-im2col) kernel for `(1,k)` temporal and `(C,1)` spatial convs, fusing BatchNorm into
the preceding conv at eval time, and reusing autograd buffers. Not done here (not worth a
rewrite at these sizes).

## Mobile library

`test/bci/lib/eegnet_lib.py` exports C functions (handle-based, float64 input `(n,6,500)`):
`create(seed)`, `load_weights(h, path)`, `save_weights(h, path)`, `set_lr(h, lr)`,
`predict(h, x, n, out_probs)` (softmax, n x 3), `train_step(h, x, y, n) -> loss`.

```sh
typhon build -release -lib -o libeegnet.dylib test/bci/lib/eegnet_lib.py
typhon build -release -lib -target android-arm64 -o libeegnet.so test/bci/lib/eegnet_lib.py
```

Checks: `test/bci/lib/check_eegnet_lib_ctypes.py` (host, ctypes) and
`test/bci/lib/mobile/driver_eegnet.c` (Android, dlopen). Both give probability max abs diff
6.0e-8 vs PyTorch and identical training losses (1.169309 -> 1.104428 over 5 epochs).
Library size: 6.4 MB (macOS), 8.1 MB (Android, unstripped).

## Limits

- Single-threaded; no GPU/NNAPI/CoreML delegates.
- Models must be in the Typhon braindecode port (most of braindecode 2.0.0 is; see
  `test/braindecode`). Arbitrary PyTorch models are not loadable as graphs; you port code.
- Training results equal PyTorch only up to float32 reduction order (no bitwise determinism).
- Compile time ~40 s per library build.

## Recommendation vs ONNX Runtime (see `onnx.md`)

- Inference only, many model types, or need of NNAPI/CoreML acceleration: ONNX Runtime.
- On-device training/fine-tuning (per-user calibration) of EEGNet/ShallowFBCSPNet: Typhon.
  ONNX Runtime training needs separate training artifacts and a much larger runtime; here the
  same Python model code trains on the phone with PyTorch-equal results and comparable speed,
  in one ~8 MB `.so` that can also hold the rest of the pipeline (MNE filtering, sklearn).
