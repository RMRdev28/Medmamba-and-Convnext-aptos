# Running on a rented GPU (RunPod / Vast.ai / Lambda)

Why bother vs Kaggle:
- **Build `mamba-ssm` once** → the fast CUDA selective scan instead of the slow
  pure-PyTorch fallback (~37 s/iter measured). Biggest single speedup.
- **More VRAM → higher resolution.** DR grade depends on tiny lesions
  (microaneurysms are a few pixels), so 512px meaningfully helps QWK + accuracy.
- Root + persistent env, no 12h/weekly session cap.

Caveats worth knowing:
- Higher resolution improves **accuracy/QWK**, not generalisation directly —
  cross-dataset transfer is driven by Ben Graham + CLAHE + augmentation.
- Don't upscale past the source image detail; resize **down** to a common size.
- Rented boxes **bill idle time** — stop the instance when not training.
- APTOS (~10 GB) isn't pre-mounted; `setup_gpu.sh` pulls it via the Kaggle API.

## Quick start

```bash
# on the rented box, in the repo dir:
export DATA_DIR=/workspace/data
chmod +x setup_gpu.sh && ./setup_gpu.sh          # deps + kernel + data

python - <<'PY'
from src.config import cfg, rented_gpu_preset
from src.train import main
cfg = rented_gpu_preset(cfg, image_size=512, batch_size=24, data_root="/workspace/data")
cfg.epochs = 25
cfg.train_folds = [0, 1, 2, 3, 4]   # full 5-fold CV ensemble for best QWK
main(cfg)
PY
```

## Resolution × GPU cheat-sheet

Approximate — depends on batch, AMP (on by default), and `mamba_variant`.
Start one notch below OOM and raise `batch_size` until it fits.

| GPU | VRAM | Practical res | Suggested batch @512 |
|---|---|---|---|
| T4 (Kaggle) | 16 GB | 384 | 16 @384 (512 won't fit) |
| RTX 4090 | 24 GB | 512 | 16–20 |
| A6000 / L40S | 48 GB | 512–640 | 32–48 |
| A100 | 80 GB | 640–768 | 48+ |

If you hit OOM: lower `batch_size`, or `image_size` (512→448→384), or keep
`mamba_variant="tiny"`.

## Getting the best of all three (QWK, accuracy, generalisation)

- **QWK**: the training loop already selects the checkpoint on val QWK and fits
  QWK-optimal thresholds. Averaging raw scores across the 5 folds before
  thresholding gives the biggest QWK bump.
- **Accuracy**: now printed every epoch next to QWK; ensembling helps it too.
- **Generalisation**: after training, run `src.infer.evaluate_external(...)` on a
  *different* dataset (IDRiD/Messidor/DDR). Watch the gap between **zero-shot
  QWK** (APTOS thresholds) and **re-fit QWK** — a large gap means the model
  transfers features but needs recalibration; a low zero-shot QWK overall points
  at the random Mamba branch overfitting (raise `drop_path_rate`, try
  `fusion="gated"`, or revisit pretraining the SSM branch).

## Recommended workflow

1. **Debug on Kaggle (free)** at 384px, 1 fold, 2–3 epochs — shake out bugs.
2. **Train on the rented GPU** at 512px, 5 folds, full epochs.
3. Stop the instance. Download `best_fold*.pt` before terminating.
