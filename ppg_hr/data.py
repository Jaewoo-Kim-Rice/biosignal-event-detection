"""Loading and windowing utilities for the PPG-DaLiA dataset.

Dataset: Reiss et al. (2019), "Deep PPG: Large-Scale Heart Rate Estimation
with Convolutional Neural Networks", Sensors 19(14). Distributed via the UCI
Machine Learning Repository (CC BY 4.0). One pickle file per subject holds
synchronized wrist (Empatica E4) and chest (RespiBAN) recordings plus
ground-truth heart rate computed from chest ECG in 8 s windows shifted by 2 s.

This module deliberately fails loudly on any deviation from the documented
file structure rather than skipping or defaulting.
"""

import pickle
from pathlib import Path

import numpy as np

FS_PPG = 64   # wrist PPG ("BVP" channel), Hz
FS_ACC = 32   # wrist 3-axis accelerometer, Hz
WIN_S = 8     # evaluation window length, s (protocol of Reiss et al. 2019)
SHIFT_S = 2   # evaluation window shift, s

SUBJECTS = tuple(f"S{i}" for i in range(1, 16))

# Activity codes from the dataset README.
ACTIVITIES = {
    0: "transition",
    1: "sitting",
    2: "stairs",
    3: "table soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "walking",
    8: "working",
}


def load_subject(data_dir, subject):
    """Load one subject's recording.

    Parameters
    ----------
    data_dir : path to the extracted PPG_FieldStudy directory
    subject : e.g. "S1"

    Returns
    -------
    dict with keys:
        ppg        (n,)   wrist PPG at 64 Hz
        acc        (m, 3) wrist accelerometer at 32 Hz
        hr         (k,)   ground-truth HR [bpm], one value per 8 s / 2 s window
        activity   (a,)   activity code time series (rate inferred, see fs_activity)
        fs_activity float  inferred activity sampling rate [Hz]
    """
    if subject not in SUBJECTS:
        raise ValueError(f"unknown subject {subject!r}; expected one of {SUBJECTS}")
    path = Path(data_dir) / subject / f"{subject}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; is data_dir the PPG_FieldStudy root?")

    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    missing = {"signal", "label", "activity"} - set(raw)
    if missing:
        raise KeyError(f"{subject}: pickle missing keys {sorted(missing)}; has {sorted(raw)}")

    wrist = raw["signal"]["wrist"]
    ppg = np.asarray(wrist["BVP"], dtype=float).squeeze()
    acc = np.asarray(wrist["ACC"], dtype=float)
    hr = np.asarray(raw["label"], dtype=float).squeeze()
    activity = np.asarray(raw["activity"], dtype=float).squeeze()

    if ppg.ndim != 1:
        raise ValueError(f"{subject}: BVP shape {ppg.shape}, expected 1-D")
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"{subject}: ACC shape {acc.shape}, expected (m, 3)")

    duration_s = len(ppg) / FS_PPG
    acc_duration_s = len(acc) / FS_ACC
    if abs(duration_s - acc_duration_s) > 1.0:
        raise ValueError(
            f"{subject}: PPG duration {duration_s:.1f}s vs ACC {acc_duration_s:.1f}s"
        )

    n_expected = n_windows(len(ppg))
    if abs(len(hr) - n_expected) > 2:
        raise ValueError(
            f"{subject}: {len(hr)} HR labels but {n_expected} windows expected "
            f"from {duration_s:.1f}s of PPG"
        )

    return {
        "ppg": ppg,
        "acc": acc,
        "hr": hr,
        "activity": activity,
        "fs_activity": len(activity) / duration_s,
    }


def n_windows(n_samples, fs=FS_PPG):
    """Number of complete 8 s / 2 s windows in a signal of n_samples at fs."""
    return int((n_samples / fs - WIN_S) // SHIFT_S) + 1


def window_slice(i, fs):
    """Sample slice of window i (covers seconds [2i, 2i + 8)) at rate fs."""
    start = int(round(i * SHIFT_S * fs))
    return slice(start, start + int(round(WIN_S * fs)))


def window_activity(activity, fs_activity, i):
    """Majority activity code within window i."""
    seg = activity[window_slice(i, fs_activity)]
    if seg.size == 0:
        raise ValueError(f"window {i} beyond activity track")
    codes, counts = np.unique(seg.astype(int), return_counts=True)
    return int(codes[np.argmax(counts)])
