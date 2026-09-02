"""Windowed power spectra of PPG and accelerometer on a shared frequency grid.

Both sensors are zero-padded to the same frequency spacing (0.9375 bpm), so
the accelerometer noise spectrum can index directly into the PPG spectrum.
The zero-padded FFT stands in for the phase-vocoder refinement of Temko
(2017): with a 0.94 bpm grid plus parabolic peak interpolation, the 8 s
window's native 7.5 bpm bin resolution is no longer the accuracy floor.
"""

import numpy as np
from scipy.signal import butter, get_window, sosfiltfilt

from .data import FS_ACC, FS_PPG, WIN_S, window_slice

BAND_HZ = (0.4, 4.0)        # analysis band-pass: 24-240 bpm
SEARCH_BPM = (40.0, 210.0)  # HR search band; dataset ground truth never goes below 41.7
NFFT_PPG = 4096             # 64 Hz -> df = 0.015625 Hz = 0.9375 bpm
NFFT_ACC = 2048             # 32 Hz -> same df, grids align exactly

assert FS_PPG / NFFT_PPG == FS_ACC / NFFT_ACC, "frequency grids must align"


def bandpass(x, fs, lo=BAND_HZ[0], hi=BAND_HZ[1], order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def band_grid():
    """HR search-band frequency grid: (bpm values, absolute rfft bin indices)."""
    bpm = np.fft.rfftfreq(NFFT_PPG, d=1 / FS_PPG) * 60.0
    idx = np.flatnonzero((bpm >= SEARCH_BPM[0]) & (bpm <= SEARCH_BPM[1]))
    return bpm[idx], idx


def power_spectra(x, fs, nfft, n_win):
    """Power spectra of the 8 s / 2 s windows, restricted to the search band.

    Returns (n_win, n_band). Raises on short or all-zero windows.
    """
    win_len = int(WIN_S * fs)
    taper = get_window("hann", win_len)
    _, idx = band_grid()  # same integer indices on both grids (equal df)
    if idx[-1] >= nfft // 2 + 1:
        raise ValueError(f"HR band exceeds Nyquist range of nfft={nfft} at fs={fs}")
    out = np.empty((n_win, len(idx)))
    for i in range(n_win):
        seg = x[window_slice(i, fs)]
        if len(seg) != win_len:
            raise ValueError(f"window {i}: {len(seg)} samples, expected {win_len}")
        spec = np.abs(np.fft.rfft((seg - seg.mean()) * taper, n=nfft)) ** 2
        out[i] = spec[idx]
        if out[i].sum() == 0:
            raise ValueError(f"window {i}: zero power in HR band")
    return out


def ppg_power(ppg_filtered, n_win):
    return power_spectra(ppg_filtered, FS_PPG, NFFT_PPG, n_win)


def acc_power(acc_filtered, n_win):
    """Per-axis power spectra averaged over the 3 axes.

    Averaging per-axis spectra (rather than taking the vector magnitude
    first) avoids the rectification harmonics a nonlinear norm introduces.
    """
    per_axis = [
        power_spectra(acc_filtered[:, j], FS_ACC, NFFT_ACC, n_win) for j in range(3)
    ]
    return np.mean(per_axis, axis=0)
