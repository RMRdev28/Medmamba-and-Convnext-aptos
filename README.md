# Hybrid MedMamba × ConvNeXt for Diabetic Retinopathy

A generalisable diabetic-retinopathy (DR) grading model that **trains on APTOS
2019** and is designed to **transfer to other fundus datasets** (Messidor,
IDRiD, DDR, EyePACS) without retraining. Built to run on a **Kaggle T4 (16 GB)**.

## Idea in one picture

```
             ┌─> ConvNeXt-tiny  (timm, ImageNet-pretrained) ─> feat_A (768)
 fundus img ─┤
             └─> MedMamba / VSS backbone (2D selective scan) ─> feat_B (768)
                          fuse (concat+MLP | gated) ─> 1 scalar (DR grade 0..4)
```

Three design choices drive cross-dataset generalisation:

1. **Preprocessing = domain normaliser.** Circle-crop → **Ben Graham** weighted
   Gaussian-blur subtraction → **CLAHE**. This cancels the lighting/colour
   differences between cameras that otherwise wreck transfer. (`src/preprocessing.py`)
2. **Hybrid backbone.** ConvNeXt captures local lesion detail; the MedMamba VSS
   branch (4-direction 2D selective scan) captures global retinal context. Both
   keep ImageNet-pretrained / well-initialised weights. (`src/models/`)
3. **Ordinal regression + QWK thresholds.** One scalar output, SmoothL1 loss,
   then thresholds fit to maximise Quadratic Weighted Kappa. Respects grade
   ordering and calibrates cleanly to new datasets. (`src/losses.py`)

## Layout

```
src/
  config.py         # ALL hyperparameters (the CFG dataclass)
  preprocessing.py  # circle crop, Ben Graham, CLAHE
  dataset.py        # APTOSDataset, albumentations, stratified folds, cache
  losses.py         # SmoothL1/MSE + OptimizedRounder (QWK thresholds)
  models/
    vss.py          # SS2D + VSSBlock (CUDA kernel OR pure-PyTorch fallback)
    medmamba.py     # 4-stage VSS pyramid backbone
    hybrid.py       # dual-branch fusion + regression head
  engine.py         # train/val loop, AMP, EMA, cosine LR, QWK eval
  train.py          # k-fold entry point
  infer.py          # cross-dataset (zero-shot) evaluation + TTA
  smoke_test.py     # CPU shape/gradient check (no GPU/data needed)
```

## Selective-scan kernel

`src/models/vss.py` uses `mamba_ssm`'s CUDA `selective_scan_fn` when available
(fast) and otherwise a **correct pure-PyTorch reference** (slower). On Kaggle,
enable the fast path:

```bash
pip install causal-conv1d==1.4.0 mamba-ssm==2.2.2
```

If that build fails, everything still runs — set `cfg.force_pytorch_scan=True`
to force the fallback, or just leave it (auto-detected). For real training you
want the CUDA kernel; the fallback is best for the 224px / small-batch setting.

## Quick start (Kaggle)

See `notebooks/kaggle_train.md` for copy-paste cells. In short:

1. New notebook, GPU **T4 ×1**, internet **ON**.
2. Add data: `aptos2019-blindness-detection`.
3. Upload this `src/` folder (as a Kaggle *Dataset* utility, or `git clone`).
4. `pip install -r requirements.txt` (+ optional mamba-ssm).
5. `python -m src.train`
6. Evaluate transfer with `src.infer.evaluate_external(...)`.

## Local sanity check

```powershell
python -m src.smoke_test   # builds the model, runs fwd+bwd on CPU, checks QWK
```

## Key knobs (`src/config.py`)

| Setting | Default | Note |
|---|---|---|
| `image_size` | 384 | 224 for fast iteration |
| `batch_size` | 16 | fits T4 at 384 + AMP |
| `mamba_variant` | tiny | `small` = deeper stage 3 |
| `fusion` | concat_mlp | or `gated` |
| `backbone_lr` / `lr` | 4e-5 / 2e-4 | discriminative LR |
| `tta` | True | hflip+vflip at eval |
| `n_folds` / `train_folds` | 5 / [0] | add folds to ensemble |
