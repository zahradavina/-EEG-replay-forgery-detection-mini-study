"""Usage:
  python run.py --synthetic                                   # smoke test, no download
  python run.py --subjects 1 20                               # real eegmmidb, 3 seeds (default)
  python run.py --subjects 1 20 --seeds 5                     # more seeds
  python run.py --subjects 1 20 --drop max_bank_similarity    # ablation: full vs. feature removed
"""
import argparse, json, sys
import numpy as np
sys.path.insert(0, "src")
from data import load_subject, synthetic_subject
from evaluate import build_dataset, evaluate
from features import FEATURE_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--subjects", type=int, nargs=2, default=[1, 20])
ap.add_argument("--seeds", type=int, default=3, help="number of repeats (each reseeds attack generation and the classifier)")
ap.add_argument("--drop", nargs="+", default=[], choices=FEATURE_NAMES, help="ablation: features to remove")
a = ap.parse_args()

# ---- load data once ----
subj = {}
if a.synthetic:
    rng0 = np.random.default_rng(0)
    for s in range(12):
        subj[s], names = synthetic_subject(rng0)
else:
    for s in range(a.subjects[0], a.subjects[1] + 1):
        subj[s], names = load_subject(s)
        print(f"loaded subject {s}: {subj[s].shape}", flush=True)
frontal = [i for i, n in enumerate(names) if n.startswith(("FP", "AF"))] or [0, 1]
keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in a.drop]

# ---- repeat over seeds ----
full_runs, drop_runs = [], []
for seed in range(a.seeds):
    X, y, groups, kinds = build_dataset(subj, frontal, np.random.default_rng(seed))
    r, _ = evaluate(X, y, groups, kinds, seed=seed)
    full_runs.append(r)
    if a.drop:
        r2, _ = evaluate(X[:, keep], y, groups, kinds, seed=seed)
        drop_runs.append(r2)
    print(f"seed {seed}: windows={len(y)} attacks={int(y.sum())} PR-AUC={r['pr_auc']:.3f}", flush=True)


def summarize(runs):
    keys = [k for k in runs[0] if k not in ("threshold_at_fpr", "fpr_target") and all(k in r for r in runs)]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in runs])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


full = summarize(full_runs)
drop = summarize(drop_runs) if a.drop else None
fmt = lambda d: f"{d['mean']:.3f} ± {d['std']:.3f}"
head = f"{'metric':36s}{'full':>18s}" + (f"{'without ' + '+'.join(a.drop):>34s}" if a.drop else "")
print(f"\n{a.seeds} seed(s), mean ± std (std across seeds only; subjects are pooled)\n" + head)
for k in full:
    print(f"{k:36s}{fmt(full[k]):>18s}" + (f"{fmt(drop[k]):>34s}" if a.drop and k in drop else ""))

json.dump({"args": vars(a), "full": full, "dropped": drop, "per_seed_full": full_runs, "per_seed_dropped": drop_runs},
          open("results.json", "w"), indent=2)
print("\nsaved results.json")
