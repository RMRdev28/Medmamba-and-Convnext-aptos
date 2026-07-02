"""Cross-dataset evaluation - the whole point of the project.

Load a checkpoint trained on APTOS and evaluate its QWK / accuracy on ANOTHER
dataset (Messidor, IDRiD, DDR, EyePACS...) *without* retraining. Point it at a
folder of images + a csv with columns [id_col, target_col]. The saved APTOS
thresholds are reused, and we also report QWK with thresholds re-fit on the new
set (an upper bound on transfer given perfect calibration).

Usage (Kaggle):
    from src.infer import evaluate_external
    evaluate_external(
        ckpt="/kaggle/working/best_fold0.pt",
        csv="/kaggle/input/idrid/labels.csv",
        image_dir="/kaggle/input/idrid/images",
        id_col="image", target_col="grade", ext=".jpg")
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import CFG
from .dataset import APTOSDataset
from .engine import predict_scores
from .losses import OptimizedRounder, quadratic_weighted_kappa
from .models import build_model
from sklearn.metrics import accuracy_score


def load_checkpoint(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = CFG(**{k: v for k, v in ckpt["cfg"].items() if k in CFG().__dict__})
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt["thresholds"]


def evaluate_external(ckpt, csv, image_dir, id_col, target_col,
                      ext=".png", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, thresholds = load_checkpoint(ckpt, device)

    # override I/O columns for this dataset
    cfg.id_col, cfg.target_col, cfg.image_ext = id_col, target_col, ext
    cfg.use_cache = False

    df = pd.read_csv(csv)
    ds = APTOSDataset(df, cfg, train=False, image_dir=image_dir)
    ld = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)

    scores, targets = predict_scores(model, ld, device, tta=cfg.tta)

    # 1) transfer with APTOS thresholds (true zero-shot generalisation)
    rounder = OptimizedRounder(cfg.num_classes)
    rounder.coef_ = list(thresholds)
    preds_transfer = rounder.predict(scores)
    qwk_transfer = quadratic_weighted_kappa(targets, preds_transfer)
    acc_transfer = accuracy_score(targets, preds_transfer)

    # 2) re-fit thresholds on this dataset (calibration upper bound)
    refit = OptimizedRounder(cfg.num_classes).fit(scores, targets)
    preds_refit = refit.predict(scores)
    qwk_refit = quadratic_weighted_kappa(targets, preds_refit)

    print(f"[{csv}]  n={len(df)}")
    print(f"  zero-shot (APTOS thresholds): QWK={qwk_transfer:.4f}  acc={acc_transfer:.4f}")
    print(f"  re-fit thresholds           : QWK={qwk_refit:.4f}")
    return {"qwk_transfer": qwk_transfer, "acc_transfer": acc_transfer,
            "qwk_refit": qwk_refit, "scores": scores, "targets": targets}
