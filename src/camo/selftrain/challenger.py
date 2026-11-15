"""Challenger training (CatBoost on the tabular node representation) with optional class-balanced
loss, and the promotion rule: challenger replaces the incumbent only if it beats it on the labelled
probe (the only recent labels a practitioner has)."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score


def fit_challenger(xtab: np.ndarray, y_lab: np.ndarray, idx: np.ndarray, seed: int, balanced: bool,
                   iterations: int = 300, sample_weight: np.ndarray | None = None) -> CatBoostClassifier:
    clf = CatBoostClassifier(iterations=iterations, depth=6, learning_rate=0.08, loss_function="Logloss",
                             auto_class_weights="Balanced" if balanced else None, random_seed=seed,
                             verbose=False, thread_count=-1)
    clf.fit(xtab[idx], y_lab, sample_weight=sample_weight)
    return clf


def promote(inc_scores: np.ndarray, chal_scores: np.ndarray, probe_idx: np.ndarray, y: np.ndarray,
            min_win_prob: float = 0.0, n_boot: int = 1000, seed: int = 0) -> dict:
    """Promote only if the challenger beats the deployed model on the probe AND does so robustly:
    in at least `min_win_prob` of paired bootstrap resamples of the probe nodes (decisions D17).
    With ~100 labels per window a plain `chal_ap > inc_ap` promotes on noise; min_win_prob=0 is the
    old rule."""
    yp = y[probe_idx]
    if yp.sum() == 0 or yp.sum() == len(yp):
        return {"inc_probe_ap": np.nan, "chal_probe_ap": np.nan, "win_prob": np.nan, "promoted": False}
    a, b = average_precision_score(yp, inc_scores[probe_idx]), average_precision_score(yp, chal_scores[probe_idx])
    rng = np.random.default_rng(seed)
    si, sc = inc_scores[probe_idx], chal_scores[probe_idx]
    wins = valid = 0
    for _ in range(n_boot):
        k = rng.integers(0, len(yp), len(yp))
        if 0 < yp[k].sum() < len(k):
            valid += 1
            wins += average_precision_score(yp[k], sc[k]) > average_precision_score(yp[k], si[k])
    win_prob = wins / valid if valid else np.nan
    ok = bool(b > a and (min_win_prob <= 0 or (valid and win_prob >= min_win_prob)))
    return {"inc_probe_ap": float(a), "chal_probe_ap": float(b), "win_prob": float(win_prob), "promoted": ok}
