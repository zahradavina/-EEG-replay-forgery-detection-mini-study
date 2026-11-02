"""Dataset construction + subject-wise cross-validation."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
import attacks
from features import extract


def build_dataset(subject_windows, frontal_idx, rng, artifact_frac=0.2, attack_rate=0.3):
    """For each subject: first half of windows = enrolment bank, second half = test stream.
    Test stream items are genuine, benign artifact (label 0), or attack (label 1)."""
    X, y, groups, kinds = [], [], [], []
    for sid, w in subject_windows.items():
        half = len(w) // 2
        bank, test = w[:half], w[half:]
        for win in test:
            if rng.random() < artifact_frac:
                x, kind = attacks.add_blink(win, rng, frontal_idx), "artifact"
            else:
                x, kind = win, "genuine"
            X.append(extract(x, bank)); y.append(0); groups.append(sid); kinds.append(kind)
            if rng.random() < attack_rate:
                kind = rng.choice(["replay", "forgery_naive", "forgery_shared_phase"])
                if kind == "replay":
                    a = attacks.replay(bank[rng.integers(len(bank))], rng)
                elif kind == "forgery_naive":
                    a = attacks.phase_surrogate(win, rng, shared_phase=False)
                else:
                    a = attacks.phase_surrogate(win, rng, shared_phase=True)
                X.append(extract(a, bank)); y.append(1); groups.append(sid); kinds.append(str(kind))
    return np.array(X), np.array(y), np.array(groups), np.array(kinds)


def evaluate(X, y, groups, kinds, n_splits=5, fpr=0.05, seed=0):
    scores = np.zeros(len(y))
    cv = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    for tr, te in cv.split(X, y, groups):   # subject-wise: no subject appears in both train and test
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)
        scores[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    thr = float(np.quantile(scores[y == 0], 1 - fpr))   # threshold at fixed false-alarm rate on all negatives
    res = {"pr_auc": float(average_precision_score(y, scores)),
           "positive_rate": float(y.mean()),
           "threshold_at_fpr": thr, "fpr_target": fpr,
           "detection_rate_all_attacks": float((scores[y == 1] > thr).mean())}
    for k in sorted(set(kinds)):
        key = ("false_alarm_" if k in ("genuine", "artifact") else "detection_") + k
        res[key] = float((scores[kinds == k] > thr).mean())
    return res, scores
