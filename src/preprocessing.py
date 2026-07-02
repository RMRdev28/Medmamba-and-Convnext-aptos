"""Fundus preprocessing: retina crop -> Ben Graham -> CLAHE.

These steps are the single most important factor for *cross-dataset*
generalisation: APTOS, EyePACS, Messidor, IDRiD and DDR are shot with
different cameras / lighting, and Ben Graham normalisation + CLAHE remove
most of that domain shift by locally equalising illumination and colour.

All functions take/return uint8 BGR or RGB numpy arrays (documented per fn).
"""
import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Retina cropping
# ----------------------------------------------------------------------------
def crop_black_borders(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Remove the black frame around a fundus photo. Expects RGB/BGR uint8."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.ix_(mask.any(1), mask.any(0))
    if img.ndim == 3:
        return img[coords[0].ravel()][:, coords[1].ravel(), :]
    return img[coords]


def circle_crop(img: np.ndarray) -> np.ndarray:
    """Crop to the largest centred circle (the retina), zeroing the corners.

    Expects RGB uint8, returns a square RGB image.
    """
    img = crop_black_borders(img)
    h, w = img.shape[:2]
    # pad to square
    size = max(h, w)
    top = (size - h) // 2
    left = (size - w) // 2
    square = cv2.copyMakeBorder(
        img, top, size - h - top, left, size - w - left,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )
    r = size // 2
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (r, r), int(r * 0.98), 1, thickness=-1)
    return square * mask[..., None]


# ----------------------------------------------------------------------------
# Ben Graham weighted-blur enhancement
# ----------------------------------------------------------------------------
def ben_graham(img: np.ndarray, size: int, sigma_frac: float = 30.0) -> np.ndarray:
    """Ben Graham's Kaggle-2015-winning enhancement.

    out = 4*img - 4*GaussianBlur(img) + 128
    Highlights vessels/microaneurysms and cancels large illumination gradients,
    which is exactly what varies across datasets. Expects RGB uint8, returns RGB
    uint8 resized to `size`.
    """
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    sigma = size / sigma_frac
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# CLAHE
# ----------------------------------------------------------------------------
def apply_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel (LAB). Expects RGB uint8, returns RGB uint8."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


# ----------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------
def preprocess_image(img_rgb: np.ndarray, cfg) -> np.ndarray:
    """Run the configured pipeline on an RGB uint8 image, return RGB uint8 [size,size,3]."""
    if cfg.circle_crop:
        img_rgb = circle_crop(img_rgb)

    if cfg.ben_graham:
        img_rgb = ben_graham(img_rgb, cfg.image_size)
    else:
        img_rgb = cv2.resize(img_rgb, (cfg.image_size, cfg.image_size),
                             interpolation=cv2.INTER_AREA)

    if cfg.clahe:
        img_rgb = apply_clahe(img_rgb)

    return img_rgb


def read_rgb(path: str) -> np.ndarray:
    """Read an image file as RGB uint8 (cv2 reads BGR)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
