"""Attack and artifact simulators. All functions take one window (n_ch, n) and return a new one."""
import numpy as np
from data import FS


def replay(win, rng, snr_db=(5, 20), max_shift=8):
    """Replay a previously recorded window with a small time shift and channel noise."""
    y = np.roll(win, int(rng.integers(-max_shift, max_shift + 1)), axis=-1)
    snr = rng.uniform(*snr_db)
    noise = rng.standard_normal(win.shape) * np.sqrt(np.mean(win ** 2) / 10 ** (snr / 10))
    return y + noise


def phase_surrogate(win, rng, shared_phase=False):
    """Forgery by phase randomisation: keeps each channel's amplitude spectrum.
    shared_phase=False -> independent phases per channel (destroys inter-channel structure; naive forger).
    shared_phase=True  -> same random phase on all channels (keeps cross-spectra; much harder to detect)."""
    F = np.fft.rfft(win, axis=-1)
    shape = (1, F.shape[-1]) if shared_phase else F.shape
    ph = rng.uniform(0, 2 * np.pi, shape)
    ph[..., 0] = 0
    # ADD the random phase to the original phase (F * e^{i*ph}); with shared_phase=True the cross-spectrum
    # F_i * conj(F_j) is therefore unchanged, so the zero-lag inter-channel correlation is preserved.
    return np.fft.irfft(F * np.exp(1j * ph), n=win.shape[-1], axis=-1)


def add_blink(win, rng, frontal_idx, fs=FS, amp=6.0):
    """Benign artifact: eye-blink-like Gaussian bump on frontal channels."""
    n = win.shape[-1]
    t = np.arange(n) / fs
    centre = rng.uniform(0.3, t[-1] - 0.3)
    bump = amp * np.exp(-((t - centre) ** 2) / (2 * 0.08 ** 2))
    y = win.copy()
    for i in frontal_idx:
        y[i] = y[i] + bump * win[i].std() * rng.uniform(0.7, 1.3)
    return y