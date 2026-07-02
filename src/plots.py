"""Training-curve plots for overfitting diagnosis.

Two panels per fold:
  1. train loss vs val loss  -> the classic overfitting check. If train loss
     keeps falling while val loss flattens/rises, the model is overfitting.
  2. val QWK & val accuracy   -> the metrics you actually care about.

Called automatically each epoch from engine.py, and runnable standalone to
re-draw from a saved history CSV:

    python -m src.plots /workspace/out/history_fold0.csv
"""
import matplotlib
matplotlib.use("Agg")               # headless: save PNGs, no display needed
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history, out_png, fold=0):
    """history: list[dict] or a DataFrame with columns
    epoch, train_loss, val_loss, val_qwk, val_acc."""
    df = pd.DataFrame(history)
    if len(df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(df.epoch, df.train_loss, marker="o", label="train loss")
    ax1.plot(df.epoch, df.val_loss, marker="o", label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title(f"Fold {fold} — loss (overfitting check)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(df.epoch, df.val_qwk, marker="o", color="green", label="val QWK")
    ax2.plot(df.epoch, df.val_acc, marker="o", color="orange", label="val accuracy")
    best = df.loc[df.val_qwk.idxmax()]
    ax2.scatter([best.epoch], [best.val_qwk], s=120, facecolors="none",
                edgecolors="red", zorder=5, label=f"best QWK={best.val_qwk:.3f}@ep{int(best.epoch)}")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score")
    ax2.set_title(f"Fold {fold} — val QWK & accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    import sys
    csv = sys.argv[1]
    out = csv.replace(".csv", ".png")
    fold = 0
    df = pd.read_csv(csv)
    plot_history(df, out, fold=fold)
    print("saved", out)
