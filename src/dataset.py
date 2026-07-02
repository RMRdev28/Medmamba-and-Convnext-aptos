"""APTOS dataset + augmentations + stratified fold splitting.

Targets are kept as float grades (0..4) because we train an ordinal regressor.
Preprocessing (Ben Graham/CLAHE/crop) is applied here; if `cfg.use_cache` is
set we read precomputed .png files from `cfg.cache_dir` instead (much faster).
"""
import os

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from .preprocessing import preprocess_image, read_rgb


def make_folds(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg.train_csv)
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for f, (_, val_idx) in enumerate(skf.split(df, df[cfg.target_col])):
        df.loc[val_idx, "fold"] = f
    return df


def _coarse_dropout(cfg, p=0.3):
    """CoarseDropout that works on both albumentations 1.x and 2.x APIs."""
    h = cfg.image_size // 12
    try:  # albumentations >= 2.0
        return A.CoarseDropout(
            num_holes_range=(1, 8), hole_height_range=(h // 2, h),
            hole_width_range=(h // 2, h), fill=0, p=p)
    except TypeError:  # albumentations 1.x
        return A.CoarseDropout(
            max_holes=8, max_height=h, max_width=h, fill_value=0, p=p)


def build_transforms(cfg, train: bool):
    """Augmentations applied *after* Ben Graham/CLAHE. Geometry + photometric
    jitter that mimics cross-camera variation, plus dropout for regularisation.
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                               rotate_limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12,
                                 val_shift_limit=8, p=0.4),
            A.OneOf([A.MotionBlur(blur_limit=3), A.GaussianBlur(blur_limit=3)], p=0.2),
            _coarse_dropout(cfg, p=0.3),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=cfg.mean, std=cfg.std),
        ToTensorV2(),
    ])


class APTOSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, train: bool, image_dir: str = None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.tf = build_transforms(cfg, train)
        self.image_dir = image_dir or cfg.train_dir

    def __len__(self):
        return len(self.df)

    def _load(self, image_id: str) -> np.ndarray:
        if self.cfg.use_cache:
            p = os.path.join(self.cfg.cache_dir, image_id + ".png")
            if os.path.exists(p):
                return read_rgb(p)
        p = os.path.join(self.image_dir, image_id + self.cfg.image_ext)
        img = read_rgb(p)
        return preprocess_image(img, self.cfg)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self._load(row[self.cfg.id_col])
        img = self.tf(image=img)["image"]
        if self.cfg.target_col in self.df.columns:
            y = np.float32(row[self.cfg.target_col])
        else:
            y = np.float32(-1)
        return img, y


def precompute_cache(cfg, df: pd.DataFrame = None, image_dir: str = None):
    """Run the (slow) Ben Graham/CLAHE pipeline once and save PNGs to cache_dir.

    Call this in a setup cell, then set cfg.use_cache = True for fast epochs.
    """
    from tqdm import tqdm
    os.makedirs(cfg.cache_dir, exist_ok=True)
    if df is None:
        df = pd.read_csv(cfg.train_csv)
    image_dir = image_dir or cfg.train_dir
    for _, row in tqdm(df.iterrows(), total=len(df), desc="cache"):
        iid = row[cfg.id_col]
        out = os.path.join(cfg.cache_dir, iid + ".png")
        if os.path.exists(out):
            continue
        img = read_rgb(os.path.join(image_dir, iid + cfg.image_ext))
        img = preprocess_image(img, cfg)
        cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
