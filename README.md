# EEG replay / forgery detection (mini-study)

A small, reproducible study of **signal-level integrity verification for wearable EEG**: can a simple detector tell a genuine EEG window from a replayed or forged one, *without* flagging ordinary artifacts such as eye blinks? All attacks are **simulated**; this is a preliminary study, not a security evaluation of a real device.

## Data
PhysioNet *EEG Motor Movement/Imagery Dataset* (eegmmidb): 109 volunteers, 64 channels, 160 Hz, recorded with BCI2000. This study uses subjects 1-20, runs 3, 7 and 11, downloaded automatically from PhysioNet through MNE-Python, band-pass filtered 1-40 Hz, resampled to 128 Hz and cut into non-overlapping 2 s windows. License: Open Data Commons Attribution License v1.0.

## Threat model (simplified)
Per subject, the first half of the recording is the *enrolment bank*; the second half is the *test stream*.
- **Replay**: a bank window re-injected with a small time shift and added noise (SNR 5-20 dB).
- **Forgery, independent phase**: phase-randomised surrogate of a genuine window (amplitude spectrum kept, inter-channel structure destroyed).
- **Forgery, shared phase**: the same random phase is added to every channel, so the cross-spectrum (and hence zero-lag inter-channel correlation) is preserved.
- **Benign artifact** (negative class): blink-like bump on frontal channels.

## Method
Features: log band power (delta-beta), spectral slope, mean absolute inter-channel correlation, log max peak-to-peak amplitude, and the maximum normalised cross-correlation between a window and the user's enrolment bank. Classifier: random forest. Evaluation: **subject-wise** GroupKFold, PR-AUC, and detection rate at a 5% false-alarm rate (threshold set on all negatives, artifacts included). Three seeds (each reseeds attack generation and the classifier); the reported std is across seeds only, not across subjects.

## Run
```bash
pip install -r requirements.txt
python run.py --synthetic                                  # smoke test (fake data, numbers are meaningless)
python run.py --subjects 1 20                              # real data, 3 seeds, mean +- std
python run.py --subjects 1 20 --drop max_bank_similarity   # ablation: full vs. feature removed
python diagnose.py --subjects 1 5                          # how much each feature changes under forgery
```
Results, including per-seed numbers, are written to `results.json`.

## Results (20 subjects, 3 seeds, 5% false-alarm budget, mean ± std across seeds)

| | All features | Without enrolment-similarity feature |
|---|---|---|
| PR-AUC (all attacks pooled) | 0.827 ± 0.017 | 0.586 ± 0.023 |
| Replay detected | 96.6% ± 0.9 | 14.4% ± 0.6 |
| Forgery, independent phase detected | 99.8% ± 0.4 | 99.2% ± 1.0 |
| Forgery, shared phase detected | 10.4% ± 4.1 | 5.8% ± 2.2 |
| Genuine windows flagged | 5.0% ± 0.1 | 4.9% ± 0.3 |
| Blink artifacts flagged | 4.6% ± 0.9 | 5.4% ± 0.8 |

Pooled metrics (PR-AUC, "all attacks") average over three very different attack types and should be read per attack type.

**Observations (within this simulation)**
1. Replay is detected almost only through similarity to the user's enrolment recordings; the spectral and spatial features alone detect 14.4% of replays.
2. Independent-phase forgery destroys inter-channel structure and is detected almost perfectly.
3. Shared-phase forgery is detected at 10.4%, close to the 5% a random detector would reach at this operating point. This is expected: the features are second-order statistics, which this forgery preserves by construction. The result shows that second-order features are insufficient against such an attacker, not that this forgery is undetectable in general. Higher-order or temporal features were not tested.

## Limitations
- Attacks are simulated, not captured from a real wearable pipeline; no transmission-layer effects.
- Replays are drawn from the same enrolment bank used for comparison. An attacker who records the victim elsewhere would have no counterpart in the bank, so replay detection is likely overestimated.
- The enrolment-similarity feature requires storing raw enrolment recordings on the device, which conflicts with privacy and memory goals of wearable deployment.
- The shared-phase forgery starts from a genuine window of the victim, which is a strong assumption about the attacker.
- Artifacts are synthetic blinks only (no EMG, motion, electrode pop).
- Non-adaptive attacker: the attacker does not know the detector. Adaptive attacks are future work.
- Classical features and a random forest only; no embedded-deployment measurements (latency, memory, energy).
- Single dataset, 20 subjects, 3 seeds; no cross-dataset test and no confidence intervals across subjects.

## References
- Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C28G6P
- Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z
- Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004). BCI2000: A general-purpose brain-computer interface (BCI) system. IEEE Transactions on Biomedical Engineering, 51(6), 1034-1043.
