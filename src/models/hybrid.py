"""Dual-branch hybrid: ImageNet-pretrained ConvNeXt + MedMamba VSS backbone.

    img ─┬─> ConvNeXt-tiny (timm, pretrained) ──> feat_A (768)
         └─> MedMamba/VSS backbone           ──> feat_B (768)
                    fuse (concat+MLP or gated) ──> 1 scalar (DR grade 0..4)

Single-scalar output because we train ordinal regression + QWK thresholds.
"""
import timm
import torch
import torch.nn as nn

from .medmamba import MedMambaBackbone


class GatedFusion(nn.Module):
    """Learn a per-dim gate that mixes the two branch features."""

    def __init__(self, dim_a, dim_b, out_dim):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, out_dim)
        self.proj_b = nn.Linear(dim_b, out_dim)
        self.gate = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())

    def forward(self, a, b):
        a, b = self.proj_a(a), self.proj_b(b)
        g = self.gate(torch.cat([a, b], dim=1))
        return g * a + (1 - g) * b


class HybridDRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # ConvNeXt branch (num_classes=0 -> pooled feature vector)
        self.convnext = timm.create_model(
            cfg.convnext_name, pretrained=True, num_classes=0,
            drop_path_rate=cfg.drop_path_rate)
        dim_a = self.convnext.num_features

        # MedMamba / VSS branch
        self.mamba = MedMambaBackbone(
            variant=cfg.mamba_variant, drop_path_rate=cfg.drop_path_rate,
            force_ref=cfg.force_pytorch_scan)
        dim_b = self.mamba.num_features

        if cfg.fusion == "gated":
            self.fusion = GatedFusion(dim_a, dim_b, 512)
            fused_dim = 512
        else:  # concat_mlp
            self.fusion = None
            fused_dim = dim_a + dim_b

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(fused_dim, 256), nn.GELU(),
            nn.Dropout(cfg.head_dropout * 0.5),
            nn.Linear(256, 1),
        )

    def head_parameters(self):
        params = list(self.head.parameters())
        if self.fusion is not None:
            params += list(self.fusion.parameters())
        return params

    def param_groups(self, cfg):
        """Discriminative LR groups. ConvNeXt is pretrained (low LR); the Mamba
        branch is random-init here so it gets a much higher LR, like the head.
        """
        return [
            {"params": list(self.convnext.parameters()), "lr": cfg.backbone_lr},
            {"params": list(self.mamba.parameters()), "lr": cfg.mamba_lr},
            {"params": self.head_parameters(), "lr": cfg.lr},
        ]

    def forward(self, x):
        fa = self.convnext(x)          # (B, dim_a)
        fb = self.mamba(x)             # (B, dim_b)
        if self.fusion is not None:
            f = self.fusion(fa, fb)
        else:
            f = torch.cat([fa, fb], dim=1)
        return self.head(f).squeeze(1)  # (B,) raw regression score


def build_model(cfg):
    return HybridDRModel(cfg)
