"""Central configuration for the hybrid MedMamba x ConvNeXt DR project.

Everything tunable lives here so the Kaggle notebook only imports `CFG`.
Ordinal regression (single scalar) is used, so `num_classes` refers to the
number of DR grades (0..4) only for the threshold optimiser / metrics.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class CFG:
    # --- data -------------------------------------------------------------
    # APTOS 2019 competition data (add via "Add Input" on Kaggle).
    data_root: str = "/kaggle/input/aptos2019-blindness-detection"
    train_csv: str = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_dir: str = "/kaggle/input/aptos2019-blindness-detection/train_images"
    image_ext: str = ".png"
    id_col: str = "id_code"
    target_col: str = "diagnosis"
    num_classes: int = 5

    # --- preprocessing ----------------------------------------------------
    image_size: int = 384          # T4-friendly with AMP + bs 16
    ben_graham: bool = True        # weighted Gaussian-blur subtraction
    clahe: bool = True             # CLAHE on the green/L channel
    circle_crop: bool = True       # crop retina to a centred circle
    sigma_scale: float = 10.0      # Ben Graham blur sigma = size/ sigma_scale? no -> fixed
    cache_dir: str = "/kaggle/working/cache"  # precomputed images (optional)
    use_cache: bool = False        # set True after running precompute_cache()

    # ImageNet stats (both branches are ImageNet-pretrained)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # --- model ------------------------------------------------------------
    convnext_name: str = "convnext_tiny.fb_in22k_ft_in1k"  # timm
    mamba_variant: str = "tiny"    # tiny|small - VSS branch depth/width
    force_pytorch_scan: bool = False  # True = never use mamba_ssm CUDA kernel
    fusion: str = "concat_mlp"     # concat_mlp | gated
    drop_path_rate: float = 0.2
    head_dropout: float = 0.4

    # --- training ---------------------------------------------------------
    task: str = "regression"       # single-scalar ordinal regression
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 2e-4               # head / fusion
    backbone_lr: float = 4e-5      # PRETRAINED ConvNeXt branch (low LR)
    mamba_lr: float = 2e-4         # SSM branch: use ~head LR while random-init;
                                   # drop to ~4e-5 if you later load pretrained VMamba weights
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    label_smooth_reg: float = 0.0  # optional target jitter
    loss: str = "smooth_l1"        # smooth_l1 | mse
    amp: bool = True
    grad_clip: float = 5.0
    ema: bool = True
    ema_decay: float = 0.999

    # --- cross validation -------------------------------------------------
    n_folds: int = 5
    train_folds: List[int] = field(default_factory=lambda: [0])  # which folds to run
    seed: int = 42

    # --- inference / TTA --------------------------------------------------
    tta: bool = True               # hflip + vflip TTA at eval
    output_dir: str = "/kaggle/working"


def rented_gpu_preset(cfg: "CFG", image_size: int = 512, batch_size: int = 24,
                      data_root: str = "/workspace/data") -> "CFG":
    """Overrides for a rented GPU (24 GB+): higher resolution, real CUDA scan,
    local data paths. Tune batch_size to your card (see RENTED_GPU.md).

        from src.config import cfg, rented_gpu_preset
        cfg = rented_gpu_preset(cfg, image_size=512, batch_size=24)
    """
    cfg.image_size = image_size
    cfg.batch_size = batch_size
    cfg.force_pytorch_scan = False      # build mamba-ssm -> use fast CUDA kernel
    cfg.num_workers = 8                 # rented boxes usually have more CPU
    cfg.data_root = data_root
    cfg.train_csv = f"{data_root}/train.csv"
    cfg.train_dir = f"{data_root}/train_images"
    cfg.cache_dir = f"{data_root}/cache"
    cfg.output_dir = "/workspace/out"
    return cfg


cfg = CFG()
