"""Hand-crafted integrity features: spectral, spatial, and replay-similarity."""
import numpy as np
from scipy.signal import welch, fftconvolve
from data import FS

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
FEATURE_NAMES = [f"logP_{b}" for b in BANDS] + ["spectral_slope", "mean_abs_chan_corr", "log_max_ptp", "max_bank_similarity"]


def _z(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)


def max_bank_similarity(win, bank, ref_chs, max_lag=16):
    """Highest normalised cross-correlation (within +/- max_lag samples) between this window and any
    enrolment window of the same user. Near 1.0 => suspiciously similar to something already seen => replay."""
    n = win.shape[-1]
    zw, zb = _z(win[ref_chs]), _z(bank[:, ref_chs, :])
    acc = 0
    for c in range(len(ref_chs)):
        conv = fftconvolve(zb[:, c, :], zw[c, ::-1][None, :], mode="full", axes=-1) / n
        acc = acc + conv[:, n - 1 - max_lag : n + max_lag]
    return float(np.max(acc / len(ref_chs)))


def extract(win, bank, fs=FS, n_ref=8):
    f, P = welch(win, fs=fs, nperseg=fs, axis=-1)
    Pm = P.mean(0)
    feats = [np.log10(Pm[(f >= lo) & (f < hi)].mean() + 1e-20) for lo, hi in BANDS.values()]
    m = (f >= 2) & (f <= 30)
    slope = np.polyfit(np.log10(f[m]), np.log10(Pm[m] + 1e-20), 1)[0]
    C = np.nan_to_num(np.corrcoef(win))
    corr = np.abs(C[~np.eye(len(C), dtype=bool)]).mean()
    ptp = np.log10(np.ptp(win, axis=-1).max() + 1e-12)
    sim = max_bank_similarity(win, bank, list(range(min(n_ref, win.shape[0]))))
    return np.array(feats + [slope, corr, ptp, sim])
