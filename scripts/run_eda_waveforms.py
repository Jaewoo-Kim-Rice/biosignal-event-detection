"""Time-domain look at PPG and accelerometer waveforms per activity.

For a few representative activities, extracts a short segment from the middle
of that activity's longest contiguous block and plots the band-passed PPG
(what the estimation pipeline actually sees) next to the raw, detrended
3-axis accelerometer traces, with the ECG-derived mean HR annotated.

Usage: python scripts/run_eda_waveforms.py <data_dir> [--subject S1] [--out figures]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppg_hr.data import ACTIVITIES, FS_ACC, FS_PPG, SHIFT_S, WIN_S, load_subject  # noqa: E402
from ppg_hr.spectral import bandpass  # noqa: E402

SHOW_ACTIVITIES = (1, 7, 2, 4)  # sitting, walking, stairs, cycling
SEG_S = 20
ACC_COLORS = {"x": "#4c72b0", "y": "#dd8452", "z": "#55a868"}


def longest_run_center(activity, fs_activity, code):
    """Center time [s] of the longest contiguous run of an activity code."""
    mask = np.concatenate([[0], (activity.astype(int) == code).astype(int), [0]])
    edges = np.flatnonzero(np.diff(mask))
    if edges.size == 0:
        raise ValueError(f"activity code {code} ({ACTIVITIES[code]}) not present")
    starts, ends = edges[::2], edges[1::2]
    k = np.argmax(ends - starts)
    return (starts[k] + ends[k]) / 2 / fs_activity


def segment_mean_hr(hr, t0):
    """Mean ground-truth HR of the windows overlapping [t0, t0 + SEG_S]."""
    i0 = max(0, int((t0 - WIN_S) // SHIFT_S) + 1)
    i1 = min(len(hr), int((t0 + SEG_S) // SHIFT_S) + 1)
    if i1 <= i0:
        raise ValueError(f"no HR windows overlap segment at t0={t0:.0f}s")
    return float(np.mean(hr[i0:i1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", help="path to extracted PPG_FieldStudy directory")
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    rec = load_subject(args.data_dir, args.subject)
    ppg_f = bandpass(rec["ppg"], FS_PPG)

    fig, axes = plt.subplots(
        len(SHOW_ACTIVITIES), 2, figsize=(13, 2.2 * len(SHOW_ACTIVITIES)),
        sharex=True, constrained_layout=True,
    )
    for row, code in enumerate(SHOW_ACTIVITIES):
        tc = longest_run_center(rec["activity"], rec["fs_activity"], code)
        t0 = tc - SEG_S / 2
        hr = segment_mean_hr(rec["hr"], t0)

        sl_p = slice(int(t0 * FS_PPG), int((t0 + SEG_S) * FS_PPG))
        t_p = np.arange(sl_p.stop - sl_p.start) / FS_PPG
        axes[row, 0].plot(t_p, ppg_f[sl_p], color="#4c72b0", lw=0.8)
        axes[row, 0].set_title(
            f"{ACTIVITIES[code]}  (GT HR {hr:.0f} bpm ≈ {60 / hr:.2f} s/beat)",
            loc="left", fontsize=10,
        )
        axes[row, 0].set_ylabel("PPG [a.u.]")

        sl_a = slice(int(t0 * FS_ACC), int((t0 + SEG_S) * FS_ACC))
        t_a = np.arange(sl_a.stop - sl_a.start) / FS_ACC
        seg_a = rec["acc"][sl_a]
        seg_a = seg_a - seg_a.mean(axis=0)  # remove gravity offset per axis
        for j, name in enumerate(ACC_COLORS):
            axes[row, 1].plot(t_a, seg_a[:, j], color=ACC_COLORS[name],
                              lw=0.7, label=name)
        axes[row, 1].set_ylabel("accel [a.u.]")
        axes[row, 1].set_title(f"accelerometer, t0 = {t0 / 60:.1f} min",
                               loc="left", fontsize=10)

    axes[0, 1].legend(loc="upper right", ncols=3, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time in segment [s]")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eda_{args.subject}_waveforms.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
