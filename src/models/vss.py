"""Visual State-Space (VSS) building blocks - the "MedMamba" branch.

SS2D performs a 2D selective scan in 4 directions (VMamba-style). The heavy
inner op is the selective scan; we use the CUDA kernel from `mamba_ssm` when it
imports, otherwise a correct pure-PyTorch reference (slower). This keeps the
project runnable on any Kaggle image even when the kernel fails to build.

The VSS block mirrors MedMamba's SS-Conv-SSM idea: an SS2D path for global
context plus a depthwise-conv path for local detail (microaneurysms/exudates),
each added residually. Good inductive bias for fundus lesions.

Direction handling follows VMamba: the 4 scan directions are folded into the
channel dimension as `K` groups, so B/C are grouped tensors of shape
(batch, K, d_state, L) - a layout supported by both `selective_scan_fn` and the
reference implementation below.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- optional fast kernel ----------------------------------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
    _HAS_CUDA_SCAN = True
except Exception:  # pragma: no cover - depends on Kaggle env
    _cuda_scan = None
    _HAS_CUDA_SCAN = False


def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    """Pure-PyTorch selective scan with grouped B/C (matches mamba_ssm).

    u, delta : (b, d, l)
    A        : (d, n)
    B, C     : (b, g, n, l)   where g divides d  (here g = K directions)
    D        : (d,)
    delta_bias : (d,)
    returns  : (b, d, l)
    """
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    A = A.float()
    B, C = B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    b, d, l = u.shape
    n = A.shape[1]
    g = B.shape[1]
    per_group = d // g
    # expand grouped B/C to per-channel: (b, d, n, l)
    B = B.repeat_interleave(per_group, dim=1)
    C = C.repeat_interleave(per_group, dim=1)

    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))        # (b,d,l,n)
    deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)        # (b,d,l,n)

    x = u.new_zeros((b, d, n))
    ys = []
    for i in range(l):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]                   # (b,d,n)
        ys.append(torch.einsum("bdn,bdn->bd", x, C[:, :, :, i]))
    y = torch.stack(ys, dim=2)                                       # (b,d,l)
    if D is not None:
        y = y + u * D[None, :, None].float()
    return y.to(dtype_in)


def _scan(u, delta, A, B, C, D, delta_bias, force_ref=False):
    if _HAS_CUDA_SCAN and not force_ref and u.is_cuda:
        return _cuda_scan(u, delta, A, B, C, D, z=None,
                          delta_bias=delta_bias, delta_softplus=True,
                          return_last_state=False)
    return selective_scan_ref(u, delta, A, B, C, D, delta_bias, delta_softplus=True)


class SS2D(nn.Module):
    """2D selective scan over 4 directional flattenings of an H x W map."""

    K = 4

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2.0,
                 dt_rank="auto", force_ref=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.force_ref = force_ref
        K = self.K

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv // 2, groups=self.d_inner, bias=True)
        self.act = nn.SiLU()

        # per-direction x -> (dt_rank, B, C) and dt_rank -> d_inner projections
        self.x_proj_weight = nn.Parameter(torch.randn(
            K, self.dt_rank + 2 * d_state, self.d_inner) * (self.d_inner ** -0.5))
        dt_w, dt_b = zip(*[self._dt_init(self.dt_rank, self.d_inner) for _ in range(K)])
        self.dt_proj_weight = nn.Parameter(torch.stack(dt_w, 0))      # (K,d_inner,dt_rank)
        self.dt_proj_bias = nn.Parameter(torch.stack(dt_b, 0))        # (K,d_inner)

        self.A_logs = nn.Parameter(self._A_init(d_state, self.d_inner, K))  # (K*d_inner,n)
        self.Ds = nn.Parameter(torch.ones(K * self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    @staticmethod
    def _dt_init(dt_rank, d_inner):
        w = torch.empty(d_inner, dt_rank)
        nn.init.uniform_(w, -dt_rank ** -0.5, dt_rank ** -0.5)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        return w, inv_dt

    @staticmethod
    def _A_init(d_state, d_inner, K):
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        return torch.log(A).repeat(K, 1).contiguous()  # (K*d_inner, n)

    def _four_way(self, x):
        """x: (B,C,H,W) -> (B, K, C, L): [H-major, W-major, +both flips]."""
        B, C, H, W = x.shape
        hw = x.reshape(B, C, -1)
        wh = x.transpose(2, 3).reshape(B, C, -1)
        xs = torch.stack([hw, wh], dim=1)                        # (B,2,C,L)
        return torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)  # (B,4,C,L)

    def _merge(self, ys, H, W):
        """ys: (B,K,C,L) -> (B, L, C) recombined into H-major order."""
        B, K, C, L = ys.shape
        y1, y2, y3, y4 = ys[:, 0], ys[:, 1], ys[:, 2], ys[:, 3]
        y3 = torch.flip(y3, dims=[-1])
        y4 = torch.flip(y4, dims=[-1])
        y2 = y2.reshape(B, C, W, H).transpose(2, 3).reshape(B, C, L)  # W- -> H-major
        y4 = y4.reshape(B, C, W, H).transpose(2, 3).reshape(B, C, L)
        return (y1 + y2 + y3 + y4).transpose(1, 2)                # (B,L,C)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        L = H * W
        K, di, n = self.K, self.d_inner, self.d_state

        xz = self.in_proj(x)                                     # (B,H,W,2*di)
        xc, z = xz.chunk(2, dim=-1)
        xc = xc.permute(0, 3, 1, 2).contiguous()                # (B,di,H,W)
        xc = self.act(self.conv2d(xc))

        xs = self._four_way(xc)                                 # (B,K,di,L)
        x_dbl = torch.einsum("bkcl,krc->bkrl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, n, n], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_proj_weight)

        # fold K into channels -> grouped scan
        u = xs.reshape(B, K * di, L)
        delta = dts.reshape(B, K * di, L)
        Bs = Bs.contiguous().view(B, K, n, L)                   # grouped
        Cs = Cs.contiguous().view(B, K, n, L)
        A = -torch.exp(self.A_logs.float())                    # (K*di, n)
        D = self.Ds.float()                                    # (K*di,)
        dt_bias = self.dt_proj_bias.reshape(K * di).float()    # (K*di,)

        y = _scan(u, delta, A, Bs, Cs, D, dt_bias, force_ref=self.force_ref)
        ys = y.view(B, K, di, L)
        out = self._merge(ys, H, W)                            # (B,L,di)
        out = self.out_norm(out)
        out = out * F.silu(z.reshape(B, L, di))
        out = self.out_proj(out)
        return out.view(B, H, W, C)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep * mask.floor()


class VSSBlock(nn.Module):
    """SS2D global path + depthwise-conv local path, both residual (SS-Conv-SSM)."""

    def __init__(self, dim, d_state=16, drop_path=0.0, force_ref=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state, force_ref=force_ref)
        self.drop_path = DropPath(drop_path)
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, x):  # x: (B,H,W,C)
        x = x + self.drop_path(self.ss2d(self.norm(x)))
        c = x.permute(0, 3, 1, 2)
        c = self.conv_branch(c).permute(0, 2, 3, 1)
        return x + self.drop_path(c)
