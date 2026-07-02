"""Ready-to-run 5-fold training for a rented A100 80GB (RunPod).

Prereqs (once):
    export DATA_DIR=/workspace/data
    ./setup_gpu.sh          # installs deps + mamba-ssm kernel + downloads APTOS

Run:
    python train_a100.py

Everything (data, cache, checkpoints) lives under /workspace so it survives a
pod stop when a network volume is attached. STOP the pod when this finishes.
"""
from src.config import cfg, rented_gpu_preset
from src.train import main

# 512px is the DR sweet spot; A100 80GB fits a comfortable batch at that size.
cfg = rented_gpu_preset(cfg, image_size=512, batch_size=32,
                        data_root="/workspace/data")

cfg.epochs = 25
cfg.train_folds = [0, 1, 2, 3, 4]   # full CV ensemble -> best QWK + accuracy
cfg.tta = True

# Regularise the random-init Mamba branch a bit harder to protect transfer:
cfg.drop_path_rate = 0.3
# cfg.fusion = "gated"              # uncomment to let the model down-weight a noisy branch

if __name__ == "__main__":
    # One-time (optional) speed-up: precompute Ben Graham/CLAHE to the volume,
    # then reuse across all 5 folds instead of recomputing every epoch.
    from src.dataset import precompute_cache
    precompute_cache(cfg)
    cfg.use_cache = True

    main(cfg)
    print("\nDone. Checkpoints in /workspace/out/best_fold*.pt")
    print("Next: evaluate cross-dataset transfer with src.infer.evaluate_external(...)")
    print("Then STOP the pod to stop billing.")
