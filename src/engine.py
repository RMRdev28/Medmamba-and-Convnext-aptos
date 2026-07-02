"""Train / validate loops for one fold, with AMP, EMA, cosine LR and QWK thresholds."""
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from .dataset import APTOSDataset
from .losses import OptimizedRounder, get_loss, quadratic_weighted_kappa
from .utils import AvgMeter, ModelEMA


def make_optimizer(model, cfg):
    """Discriminative LR: low for the pretrained ConvNeXt, high for the random
    Mamba branch and the fusion/head (see HybridDRModel.param_groups)."""
    return torch.optim.AdamW(model.param_groups(cfg), weight_decay=cfg.weight_decay)


def cosine_lr(optimizer, base_lrs, step, total_steps, warmup_steps):
    if step < warmup_steps:
        scale = (step + 1) / max(warmup_steps, 1)
    else:
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        scale = 0.5 * (1 + math.cos(math.pi * prog))
    for pg, base in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base * scale


@torch.no_grad()
def predict_scores(model, loader, device, tta=False):
    model.eval()
    scores, targets = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(imgs)
            if tta:
                out = out + model(torch.flip(imgs, dims=[3]))   # hflip
                out = out + model(torch.flip(imgs, dims=[2]))   # vflip
                out = out / 3.0
        scores.append(out.float().cpu())
        targets.append(ys)
    return torch.cat(scores).numpy(), torch.cat(targets).numpy()


def train_one_fold(df, cfg, fold, device="cuda"):
    from .models import build_model

    tr_df = df[df.fold != fold]
    va_df = df[df.fold == fold]
    train_ds = APTOSDataset(tr_df, cfg, train=True)
    valid_ds = APTOSDataset(va_df, cfg, train=False)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_ld = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    criterion = get_loss(cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema else None

    total_steps = cfg.epochs * len(train_ld)
    warmup_steps = cfg.warmup_epochs * len(train_ld)
    best_qwk, best_path = -1.0, os.path.join(cfg.output_dir, f"best_fold{fold}.pt")
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        loss_m = AvgMeter()
        pbar = tqdm(train_ld, desc=f"fold{fold} ep{epoch+1}/{cfg.epochs}")
        for imgs, ys in pbar:
            imgs = imgs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            if cfg.label_smooth_reg > 0:
                ys = ys + torch.randn_like(ys) * cfg.label_smooth_reg
            cosine_lr(optimizer, base_lrs, step, total_steps, warmup_steps)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.amp):
                out = model(imgs)
                loss = criterion(out, ys)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)

            loss_m.update(loss.item(), imgs.size(0))
            pbar.set_postfix(loss=f"{loss_m.avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            step += 1

        # --- validate (use EMA weights) ---
        eval_model = ema.ema if ema else model
        scores, targets = predict_scores(eval_model, valid_ld, device, tta=cfg.tta)
        rounder = OptimizedRounder(cfg.num_classes).fit(scores, targets)
        preds = rounder.predict(scores)
        qwk = quadratic_weighted_kappa(targets, preds)
        acc = accuracy_score(targets, preds)
        print(f"  fold{fold} ep{epoch+1}: val QWK={qwk:.4f}  acc={acc:.4f}  "
              f"thresholds={np.round(rounder.coef_,3)}")

        # QWK is the competition metric -> use it for model selection,
        # but accuracy is tracked/saved too since you care about all three.
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save({
                "model": eval_model.state_dict(),
                "thresholds": rounder.coef_,
                "qwk": qwk, "acc": acc, "cfg": cfg.__dict__, "fold": fold,
            }, best_path)
            print(f"  saved {best_path} (QWK={qwk:.4f}  acc={acc:.4f})")

    print(f"fold{fold} best QWK = {best_qwk:.4f}")
    return best_qwk, best_path
