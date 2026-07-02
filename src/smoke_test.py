"""Fast CPU sanity check: build the hybrid model and run a forward+backward pass
with the pure-PyTorch selective scan. Catches shape bugs without a GPU/dataset.

    python -m src.smoke_test
"""
import torch

from .config import CFG
from .losses import OptimizedRounder, quadratic_weighted_kappa
from .models import build_model


def main():
    cfg = CFG()
    cfg.image_size = 64          # tiny so CPU + ref-scan is fast
    cfg.mamba_variant = "tiny"
    cfg.force_pytorch_scan = True
    cfg.convnext_name = "convnext_atto"  # smallest timm convnext for the test
    cfg.drop_path_rate = 0.0

    model = build_model(cfg)
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    y = torch.tensor([1.0, 3.0])

    out = model(x)
    assert out.shape == (2,), out.shape
    loss = torch.nn.functional.smooth_l1_loss(out, y)
    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"forward out={out.detach().numpy()}  loss={loss.item():.4f}  params_with_grad={n_grad}")

    # threshold optimiser sanity
    scores = torch.linspace(0, 4, 50).numpy() + torch.randn(50).numpy() * 0.2
    targets = scores.round().clip(0, 4).astype(int)
    r = OptimizedRounder().fit(scores, targets)
    qwk = quadratic_weighted_kappa(targets, r.predict(scores))
    print(f"OptimizedRounder QWK on synthetic={qwk:.4f}  thresholds={r.coef_}")
    print("OK")


if __name__ == "__main__":
    main()
