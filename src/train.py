"""Entry point: run k-fold training on APTOS.

Kaggle:  !python -m src.train      (after `%cd` into the repo)
Local:   python -m src.train
"""
import numpy as np
import torch

from .config import cfg
from .dataset import make_folds
from .engine import train_one_fold
from .utils import seed_everything


def main(cfg=cfg):
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  image_size={cfg.image_size}  batch={cfg.batch_size}")

    df = make_folds(cfg)
    scores = {}
    for fold in cfg.train_folds:
        qwk, path = train_one_fold(df, cfg, fold, device)
        scores[fold] = qwk
    print("\n=== summary ===")
    for f, q in scores.items():
        print(f"  fold {f}: QWK={q:.4f}")
    if scores:
        print(f"  mean QWK={np.mean(list(scores.values())):.4f}")


if __name__ == "__main__":
    main()
