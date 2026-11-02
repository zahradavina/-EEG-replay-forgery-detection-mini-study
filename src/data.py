"""Data loading: PhysioNet EEG Motor Movement/Imagery (eegmmidb) + a synthetic fallback."""
import numpy as np

FS = 128  # Hz, after resampling


def to_windows(data, n):
    """(n_ch, n_samples) -> (n_win, n_ch, n) non-overlapping windows."""
    n_win = data.shape[1] // n
    x = data[:, : n_win * n].reshape(data.shape[0], n_win, n)
    return x.transpose(1, 0, 2)


def load_subject(subject, runs=(3, 7, 11), fs=FS, win_s=2.0):
    """Load one eegmmidb subject. Downloads from PhysioNet on first use (needs internet)."""
    import mne
    from mne.datasets import eegbci

    paths = eegbci.load_data(subject, list(runs))
    raws = [mne.io.read_raw_edf(p, preload=True, verbose="ERROR") for p in paths]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.filter(1.0, 40.0, verbose="ERROR")      # bandpass
    raw.notch_filter(50.0, verbose="ERROR")     # no-op after 40 Hz low-pass; kept as a reminder (PLN = 50 Hz)
    raw.resample(fs, verbose="ERROR")
    data = raw.get_data()
    data = data / data.std()                    # per-subject scale normalisation
    names = [c.strip(".").upper() for c in raw.ch_names]
    return to_windows(data, int(win_s * fs)), names


def synthetic_subject(rng, n_ch=16, n_win=60, fs=FS, win_s=2.0):
    """Pink-noise + alpha-rhythm fake EEG, only for smoke-testing the pipeline."""
    n_tot = n_win * int(win_s * fs)
    f = np.fft.rfftfreq(n_tot, 1 / fs)
    amp = np.zeros_like(f)
    amp[1:] = 1 / np.sqrt(f[1:])
    spec = (rng.standard_normal((n_ch, len(f))) + 1j * rng.standard_normal((n_ch, len(f)))) * amp
    src = np.fft.irfft(spec, n=n_tot, axis=-1)
    t = np.arange(n_tot) / fs
    f0 = rng.uniform(8, 12)
    phase = np.cumsum(rng.standard_normal(n_tot)) * 0.02
    alpha = np.sin(2 * np.pi * f0 * t + phase) * src.std() * rng.uniform(0.5, 1.5)
    mix = np.eye(n_ch) + 0.4 * rng.standard_normal((n_ch, n_ch))
    data = mix @ (src + alpha[None, :])
    data = data / data.std()
    names = ["FP1", "FP2"] + [f"CH{i}" for i in range(n_ch - 2)]
    return to_windows(data, int(win_s * fs)), names
