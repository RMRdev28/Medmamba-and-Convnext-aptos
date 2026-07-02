"""A compact VMamba/MedMamba-style hierarchical backbone (the SSM branch).

4-stage pyramid of VSS blocks. Returns a global-pooled feature vector so it can
be fused with the ConvNeXt branch. Kept intentionally small ("tiny"/"small") so
two backbones fit on a 16 GB T4 at 384px with AMP.
"""
import torch
import torch.nn as nn

from .vss import VSSBlock

_VARIANTS = {
    #            dims                depths
    "tiny":  ([96, 192, 384, 768], [2, 2, 4, 2]),
    "small": ([96, 192, 384, 768], [2, 2, 8, 2]),
}


class PatchEmbed(nn.Module):
    """Stride-4 conv stem -> (B, H/4, W/4, C) in channels-last layout."""

    def __init__(self, in_ch=3, dim=96):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim // 2), nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)  # (B,H,W,C)


class Downsample(nn.Module):
    """Halve spatial, double channels. channels-last in/out."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.reduce = nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.norm(x).permute(0, 3, 1, 2)
        return self.reduce(x).permute(0, 2, 3, 1)


class MedMambaBackbone(nn.Module):
    def __init__(self, variant="tiny", d_state=16, drop_path_rate=0.2, force_ref=False):
        super().__init__()
        dims, depths = _VARIANTS[variant]
        self.num_features = dims[-1]
        self.patch_embed = PatchEmbed(3, dims[0])

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        cur = 0
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            blocks = nn.ModuleList([
                VSSBlock(dim, d_state=d_state, drop_path=dpr[cur + j], force_ref=force_ref)
                for j in range(depth)
            ])
            self.stages.append(blocks)
            cur += depth
            if i < len(dims) - 1:
                self.downs.append(Downsample(dim, dims[i + 1]))

        self.norm = nn.LayerNorm(dims[-1])

    def forward(self, x):
        x = self.patch_embed(x)                 # (B,H,W,C)
        for i, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        x = self.norm(x)                        # (B,H,W,C)
        return x.mean(dim=(1, 2))               # (B, C) global average pool
