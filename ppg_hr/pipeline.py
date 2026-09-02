"""Wiener-style motion suppression and Viterbi heart-rate tracking.

The estimation follows the WFPV family (Temko 2017): the accelerometer
spectrum serves as a per-window noise estimate that attenuates the PPG
spectrum, and dynamic programming finds the HR trajectory that jointly
maximizes spectral evidence and physiological continuity. This
implementation runs offline (the full recording is decoded at once), so the
path uses future context; causal variants exist in the literature.
"""

import numpy as np

from .data import FS_ACC, FS_PPG, n_windows
from .spectral import acc_power, band_grid, bandpass, ppg_power

ALPHA = 4.0        # noise weight in the Wiener gain; tuned on S1-S3 only
SIGMA_BPM = 2.5    # HR change scale [bpm] per 2 s step in the transition prior
CONF_BPM = 2.0     # half-width [bpm] of the confidence integration window
EPS = 1e-12


def wiener_clean(p_ppg, p_acc, alpha=ALPHA):
    """Per-window posterior-like spectrum after motion suppression.

    Both spectra are normalized to unit power in the HR band; the gain
    S/(S + alpha*N) suppresses frequencies where the accelerometer carries
    power. Rows are renormalized to sum to 1.
    """
    s = p_ppg / p_ppg.sum(axis=1, keepdims=True)
    n = p_acc / p_acc.sum(axis=1, keepdims=True)
    cleaned = s * (s / (s + alpha * n + EPS))
    total = cleaned.sum(axis=1, keepdims=True)
    if np.any(total == 0):
        raise ValueError("window with zero power after Wiener gain")
    return cleaned / total


def viterbi_track(prob, bpm, sigma_bpm=SIGMA_BPM):
    """Maximum a posteriori HR path through the window/frequency plane.

    prob : (n_win, n_bins) rows summing to 1 (data term)
    Returns integer bin indices of the best path, (n_win,).
    """
    log_p = np.log(prob + EPS)
    d = bpm[:, None] - bpm[None, :]
    log_trans = -(d ** 2) / (2.0 * sigma_bpm ** 2)  # from-state i -> to-state j

    n_win, n_bins = log_p.shape
    ptr = np.empty((n_win, n_bins), dtype=np.int32)
    score = log_p[0].copy()
    cols = np.arange(n_bins)
    for t in range(1, n_win):
        cand = score[:, None] + log_trans
        ptr[t] = np.argmax(cand, axis=0)
        score = cand[ptr[t], cols] + log_p[t]

    path = np.empty(n_win, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(n_win - 1, 0, -1):
        path[t - 1] = ptr[t, path[t]]
    return path


def refine_peak(prob_row, k, bpm):
    """Parabolic sub-bin interpolation of the path bin k."""
    if k == 0 or k == len(bpm) - 1:
        return bpm[k]
    y0, y1, y2 = np.log(prob_row[k - 1 : k + 2] + EPS)
    denom = y0 - 2.0 * y1 + y2
    if denom >= 0:  # not a local maximum in log domain; keep the bin center
        return bpm[k]
    shift = 0.5 * (y0 - y2) / denom
    return bpm[k] + np.clip(shift, -1.0, 1.0) * (bpm[1] - bpm[0])


def confidence(prob, path, bpm, half_width_bpm=CONF_BPM):
    """Fraction of in-band power within +/- half_width_bpm of the path."""
    conf = np.empty(len(path))
    for t, k in enumerate(path):
        sel = np.abs(bpm - bpm[k]) <= half_width_bpm
        conf[t] = prob[t, sel].sum()
    return conf


def hr_from_spectra(p_ppg, p_acc, alpha=ALPHA, sigma_bpm=SIGMA_BPM):
    """Estimate HR and confidence from precomputed search-band spectra."""
    bpm, _ = band_grid()
    prob = wiener_clean(p_ppg, p_acc, alpha=alpha)
    path = viterbi_track(prob, bpm, sigma_bpm=sigma_bpm)
    hr_est = np.array([refine_peak(prob[t], k, bpm) for t, k in enumerate(path)])
    return hr_est, confidence(prob, path, bpm)


def subject_spectra(rec):
    """Search-band power spectra of one loaded subject record."""
    n_win = n_windows(len(rec["ppg"]))
    if n_win != len(rec["hr"]):
        raise ValueError(f"{n_win} windows vs {len(rec['hr'])} labels")
    p_ppg = ppg_power(bandpass(rec["ppg"], FS_PPG), n_win)
    p_acc = acc_power(bandpass(rec["acc"], FS_ACC), n_win)
    return p_ppg, p_acc, n_win


def estimate_hr(rec, alpha=ALPHA, sigma_bpm=SIGMA_BPM):
    """Full pipeline for one loaded subject record.

    Returns dict: hr_est (n_win,), conf (n_win,), n_win.
    """
    p_ppg, p_acc, n_win = subject_spectra(rec)
    hr_est, conf = hr_from_spectra(p_ppg, p_acc, alpha, sigma_bpm)
    return {"hr_est": hr_est, "conf": conf, "n_win": n_win}
