"""Exploratory look at one PPG-DaLiA subject.

Produces a three-panel figure sharing a common time axis:
  1. PPG spectrogram (y-axis in bpm) with ECG-derived ground-truth HR overlaid
  2. accelerometer-magnitude spectrogram (same y-axis)
  3. activity protocol track

The point of the figure: wherever the accelerometer shows spectral power, the
same ridges appear in the PPG spectrogram and compete with the true HR ridge.

Usage: python scripts/run_eda.py <data_dir> [--subject S1] [--out figures]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppg_hr.data import (  # noqa: E402
    ACTIVITIES, FS_ACC, FS_PPG, SHIFT_S, load_subject, n_windows, window_slice,
)
from ppg_hr.spectral import bandpass  # noqa: E402

NFFT = 4096
BPM_LIM = (30, 210)


def windowed_spectrogram(x, fs, n_win):
    """Magnitude spectra of the 8 s / 2 s windows, each column peak-normalized."""
    win_len = int(8 * fs)
    taper = get_window("hann", win_len)
    freqs = np.fft.rfftfreq(NFFT, d=1 / fs)
    spec = np.empty((len(freqs), n_win))
    for i in range(n_win):
        seg = x[window_slice(i, fs)]
        if len(seg) != win_len:
            raise ValueError(f"window {i}: got {len(seg)} samples, expected {win_len}")
        seg = (seg - seg.mean()) * taper
        mag = np.abs(np.fft.rfft(seg, n=NFFT))
        peak = mag.max()
        if peak == 0:
            raise ValueError(f"window {i}: all-zero segment")
        spec[:, i] = mag / peak
    return freqs, spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", help="path to extracted PPG_FieldStudy directory")
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    rec = load_subject(args.data_dir, args.subject)
    ppg, acc, hr = rec["ppg"], rec["acc"], rec["hr"]
    n_win = min(n_windows(len(ppg)), len(hr))
    t_win = np.arange(n_win) * SHIFT_S / 60.0  # window start time, minutes

    print(f"{args.subject}: {len(ppg) / FS_PPG / 60:.1f} min, {n_win} windows, "
          f"HR {hr.min():.0f}-{hr.max():.0f} bpm (mean {hr.mean():.0f})")
    print(f"activity sampling rate inferred: {rec['fs_activity']:.2f} Hz")

    ppg_f = bandpass(ppg, FS_PPG)
    acc_mag = np.linalg.norm(bandpass(acc, FS_ACC), axis=1)

    f_ppg, spec_ppg = windowed_spectrogram(ppg_f, FS_PPG, n_win)
    f_acc, spec_acc = windowed_spectrogram(acc_mag, FS_ACC, n_win)

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1]}, constrained_layout=True,
    )
    for ax, freqs, spec, title in [
        (axes[0], f_ppg, spec_ppg, "Wrist PPG spectrogram"),
        (axes[1], f_acc, spec_acc, "Wrist accelerometer-magnitude spectrogram"),
    ]:
        bpm = freqs * 60.0
        sel = (bpm >= BPM_LIM[0]) & (bpm <= BPM_LIM[1])
        ax.imshow(
            spec[sel], origin="lower", aspect="auto", cmap="magma",
            extent=[t_win[0], t_win[-1], bpm[sel][0], bpm[sel][-1]],
            vmin=0, vmax=1,
        )
        ax.set_ylabel("frequency [bpm]")
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].plot(t_win, hr[:n_win], color="deepskyblue", lw=0.7, ls=(0, (4, 3)),
                 label="ground-truth HR (chest ECG)")
    axes[0].legend(loc="upper right", framealpha=0.9)

    act = rec["activity"]
    t_act = np.arange(len(act)) / rec["fs_activity"] / 60.0
    axes[2].step(t_act, act, where="post", color="0.4", lw=1)
    axes[2].set_yticks(sorted(ACTIVITIES))
    axes[2].set_yticklabels([ACTIVITIES[c] for c in sorted(ACTIVITIES)], fontsize=7)
    axes[2].set_ylabel("activity")
    axes[2].set_xlabel("time [min]")
    axes[2].set_xlim(t_win[0], t_win[-1])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eda_{args.subject}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
