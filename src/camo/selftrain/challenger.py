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


def promote(inc_scores: np.ndarray, chal_scores: np.ndarray, probe_idx: np.ndarray, y: np.ndarray) -> dict:
    yp = y[probe_idx]
    if yp.sum() == 0 or yp.sum() == len(yp):
        return {"inc_probe_ap": np.nan, "chal_probe_ap": np.nan, "promoted": False}
    a, b = average_precision_score(yp, inc_scores[probe_idx]), average_precision_score(yp, chal_scores[probe_idx])
    return {"inc_probe_ap": float(a), "chal_probe_ap": float(b), "promoted": bool(b > a)}
