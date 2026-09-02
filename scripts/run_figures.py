"""Result figures from the per-window dump written by run_eval.py.

Produces:
  figures/results_mae.png    per-subject and per-activity MAE vs published baselines
  figures/tracking.png       best/worst subject HR tracks with low-confidence shading
  figures/calibration.png    error vs confidence and the risk-coverage curve

Usage: python scripts/run_figures.py [--npz results/windows.npz] [--out figures]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppg_hr.data import ACTIVITIES, SHIFT_S, SUBJECTS  # noqa: E402

C_OURS = "#4c72b0"
C_SPAMA = "#dd8452"
C_CNN = "#55a868"
CONF_LOW = 0.15  # tracking-figure shading; ~bottom third of the confidence range

# Reiss et al. (2019), LOSO on PPG-DaLiA. Overall MAE and Table 11 per activity.
BASE_OVERALL = {"SpaMaPlus": 11.06, "CNN ensemble": 7.65}
BASE_ACTIVITY = {  # activity code -> (SpaMaPlus, CNN ensemble)
    1: (4.27, 4.93), 2: (25.42, 16.98), 3: (21.48, 12.16), 4: (11.97, 12.48),
    5: (6.24, 4.96), 6: (7.33, 5.22), 7: (18.16, 9.21), 8: (4.91, 3.84),
}


def fig_results(d, out):
    err = np.abs(d["est"] - d["gt"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)

    maes = [err[d["subject"] == s].mean() for s in SUBJECTS]
    ax1.bar(range(len(SUBJECTS)), maes, color=C_OURS, width=0.65)
    ax1.axhline(np.mean(maes), color=C_OURS, lw=1, ls=(0, (4, 3)),
                label=f"this work, mean {np.mean(maes):.2f}")
    ax1.axhline(BASE_OVERALL["SpaMaPlus"], color=C_SPAMA, lw=1.2,
                label="SpaMaPlus 11.06 (best classical)")
    ax1.axhline(BASE_OVERALL["CNN ensemble"], color=C_CNN, lw=1.2,
                label="CNN ensemble 7.65")
    ax1.set_xticks(range(len(SUBJECTS)), SUBJECTS, fontsize=8)
    ax1.set_ylabel("MAE [bpm]")
    ax1.set_title("Per-subject MAE", loc="left", fontsize=10)
    ax1.legend(fontsize=8)

    codes = sorted(BASE_ACTIVITY)
    ours = [err[d["activity"] == c].mean() for c in codes]
    x = np.arange(len(codes))
    for off, vals, color, label in [
        (-0.27, ours, C_OURS, "this work"),
        (0.0, [BASE_ACTIVITY[c][0] for c in codes], C_SPAMA, "SpaMaPlus"),
        (0.27, [BASE_ACTIVITY[c][1] for c in codes], C_CNN, "CNN ensemble"),
    ]:
        ax2.bar(x + off, vals, width=0.25, color=color, label=label)
    ax2.set_xticks(x, [ACTIVITIES[c] for c in codes], fontsize=8, rotation=20)
    ax2.set_ylabel("MAE [bpm]")
    ax2.set_title("Per-activity MAE (transitions excluded)", loc="left", fontsize=10)
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "results_mae.png", dpi=150)


def fig_tracking(d, out):
    err_by_s = {s: np.abs(d["est"] - d["gt"])[d["subject"] == s].mean() for s in SUBJECTS}
    best = min(err_by_s, key=err_by_s.get)
    worst = max(err_by_s, key=err_by_s.get)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), constrained_layout=True)
    for ax, s, tag in [(axes[0], best, "best"), (axes[1], worst, "worst")]:
        m = d["subject"] == s
        t = np.arange(int(m.sum())) * SHIFT_S / 60.0
        low = d["conf"][m] < CONF_LOW
        ax.fill_between(t, 0, 1, where=low, transform=ax.get_xaxis_transform(),
                        color="0.88", label=f"confidence < {CONF_LOW}")
        ax.plot(t, d["gt"][m], color="0.3", lw=1.0, label="ground truth (ECG)")
        ax.plot(t, d["est"][m], color=C_OURS, lw=0.8, label="estimate")
        ax.set_title(f"{s} ({tag} subject, MAE {err_by_s[s]:.2f} bpm)",
                     loc="left", fontsize=10)
        ax.set_ylabel("HR [bpm]")
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("time [min]")
    fig.savefig(out / "tracking.png", dpi=150)


def fig_calibration(d, out):
    err = np.abs(d["est"] - d["gt"])
    conf = d["conf"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    edges = np.quantile(conf, np.linspace(0, 1, 11))
    centers, mae_bin = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < edges[-1] else conf <= hi)
        centers.append(conf[m].mean())
        mae_bin.append(err[m].mean())
    ax1.plot(centers, mae_bin, color=C_OURS, lw=2, marker="o", ms=5)
    ax1.set_xlabel("confidence (decile mean)")
    ax1.set_ylabel("MAE [bpm]")
    ax1.set_title("Error vs confidence", loc="left", fontsize=10)

    order = np.argsort(-conf)
    cum_mae = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    cov = np.arange(1, len(err) + 1) / len(err)
    k0 = int(0.01 * len(err))  # drop the unstable tiny-sample start of the curve
    ax2.plot(cov[k0:] * 100, cum_mae[k0:], color=C_OURS, lw=2)
    for c in (0.5, 0.8):
        k = int(c * len(err)) - 1
        ax2.annotate(f"{cum_mae[k]:.1f} bpm at {int(c * 100)}%",
                     xy=(c * 100, cum_mae[k]), xytext=(c * 100 - 2, cum_mae[k] + 1.2),
                     fontsize=8, ha="right",
                     arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.8})
    ax2.set_xlabel("coverage: most-confident windows kept [%]")
    ax2.set_ylabel("MAE of kept windows [bpm]")
    ax2.set_title("Risk-coverage", loc="left", fontsize=10)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "calibration.png", dpi=150)
    print(f"coverage 50%: MAE {cum_mae[int(0.5 * len(err)) - 1]:.2f} bpm; "
          f"80%: {cum_mae[int(0.8 * len(err)) - 1]:.2f}; 100%: {cum_mae[-1]:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="results/windows.npz")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    d = dict(np.load(args.npz, allow_pickle=False))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_results(d, out)
    fig_tracking(d, out)
    fig_calibration(d, out)
    print(f"wrote figures to {out}/")


if __name__ == "__main__":
    main()
