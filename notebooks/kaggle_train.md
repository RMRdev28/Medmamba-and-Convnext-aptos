# Kaggle notebook — copy/paste cells

Settings: **Accelerator = GPU T4 ×1**, **Internet = ON**.
Add input dataset: `aptos2019-blindness-detection`.

Get the `src/` package into the notebook by ONE of:
- Upload the whole repo folder as a Kaggle **Dataset** and add it as input, or
- `git clone` your repo, or
- Create a Kaggle *utility script* dataset from `src/`.

Then adjust `REPO` below to wherever `src/` lives.

---

### Cell 1 — install deps

```python
!pip install -q timm==0.9.16 albumentations==1.4.0 opencv-python-headless
# Optional fast Mamba kernel (skip if it fails — pure-PyTorch fallback kicks in):
!pip install -q causal-conv1d==1.4.0 mamba-ssm==2.2.2 || echo "no CUDA scan; using fallback"
```

### Cell 2 — put the repo on the path

```python
import sys, os
REPO = "/kaggle/input/medmamba-src"     # <-- folder that CONTAINS the `src/` package
sys.path.insert(0, REPO)
os.chdir(REPO)                           # so `python -m src.train` works too
```

### Cell 3 — configure

```python
from src.config import cfg
cfg.image_size = 384
cfg.batch_size = 16
cfg.epochs = 20
cfg.train_folds = [0]        # add [0,1,2,3,4] to build a full CV ensemble
cfg.tta = True
# cfg.force_pytorch_scan = True   # uncomment if mamba-ssm failed to import
print(cfg)
```

### Cell 4 — (optional) precompute preprocessing cache for faster epochs

```python
from src.dataset import precompute_cache
precompute_cache(cfg)        # writes Ben-Graham/CLAHE PNGs to cfg.cache_dir
cfg.use_cache = True
```

### Cell 5 — train

```python
from src.train import main
main(cfg)                    # saves /kaggle/working/best_fold0.pt (+ QWK thresholds)
```

### Cell 6 — zero-shot cross-dataset evaluation

Point at any other DR set you added as input (folder of images + a labels csv).

```python
from src.infer import evaluate_external
res = evaluate_external(
    ckpt="/kaggle/working/best_fold0.pt",
    csv="/kaggle/input/idrid-grading/labels.csv",
    image_dir="/kaggle/input/idrid-grading/images",
    id_col="Image name", target_col="Retinopathy grade", ext=".jpg",
)
# prints QWK with APTOS thresholds (true transfer) and with re-fit thresholds
```

---

### Tips
- OOM on T4? Drop `cfg.batch_size` to 12/8, or `cfg.image_size` to 288/224.
- No CUDA scan → training is slow at 384; use 224 + `mamba_variant="tiny"`.
- Ensemble folds: run each fold, average the raw scores from `predict_scores`
  before applying `OptimizedRounder` for the best QWK.
