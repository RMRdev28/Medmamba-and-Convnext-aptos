#!/usr/bin/env bash
# One-shot setup for a rented Linux GPU box (RunPod / Vast.ai / Lambda).
# Installs deps, builds the fast Mamba CUDA kernel, and pulls APTOS via Kaggle.
#
#   chmod +x setup_gpu.sh
#   ./setup_gpu.sh
#
# Assumes: an NVIDIA GPU + recent PyTorch base image (most rented images have
# torch preinstalled). If torch is missing, install the CUDA build that matches
# the box FIRST (https://pytorch.org), then re-run this.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/data}"

echo "==> [1/5] Python / CUDA sanity"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo "==> [2/5] Core Python deps"
pip install -q -U pip
pip install -q timm==0.9.16 albumentations==1.4.0 opencv-python-headless \
    pandas scikit-learn scipy tqdm kaggle

echo "==> [3/5] Fast selective-scan CUDA kernels (causal-conv1d + mamba-ssm)"
# Prefer the pinned versions (prebuilt wheels exist for torch 2.1-2.4 / cu12x /
# py3.10-3.11). If the template ships a different torch, fall back to latest,
# then to a source build. --no-build-isolation compiles against the box's torch.
install_mamba() {
    pip install -q --no-build-isolation causal-conv1d==1.4.0 mamba-ssm==2.2.2 && return 0
    echo "   pinned versions unavailable for this torch; trying latest..."
    pip install -q --no-build-isolation causal-conv1d mamba-ssm && return 0
    return 1
}
if install_mamba && python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null; then
    echo "CUDA selective_scan OK -> keep cfg.force_pytorch_scan=False"
else
    echo "!! mamba-ssm not available. Training still works via the pure-PyTorch"
    echo "!! fallback (set cfg.force_pytorch_scan=True), just slower. Most likely"
    echo "!! cause: no nvcc (use a CUDA 'devel' template) or torch/cuda mismatch."
fi

echo "==> [4/5] APTOS 2019 data"
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "!! ~/.kaggle/kaggle.json not found."
    echo "!! Create an API token at kaggle.com -> Account -> Create New Token,"
    echo "!! upload kaggle.json to ~/.kaggle/, then: chmod 600 ~/.kaggle/kaggle.json"
    echo "!! Skipping data download."
else
    mkdir -p "$DATA_DIR"
    chmod 600 "$HOME/.kaggle/kaggle.json"
    # NOTE: you must have joined the competition on the website first.
    kaggle competitions download -c aptos2019-blindness-detection -p "$DATA_DIR"
    (cd "$DATA_DIR" && unzip -o -q aptos2019-blindness-detection.zip && rm -f aptos2019-blindness-detection.zip)
    echo "data at $DATA_DIR:"; ls "$DATA_DIR"
fi

echo "==> [5/5] Done. Next:"
cat <<'EOF'

  python - <<'PY'
  from src.config import cfg, rented_gpu_preset
  from src.train import main
  cfg = rented_gpu_preset(cfg, image_size=512, batch_size=24, data_root="/workspace/data")
  main(cfg)
  PY

  # Remember: STOP the instance when training finishes (rented GPUs bill idle time).
EOF
