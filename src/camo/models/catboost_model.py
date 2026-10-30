"""CatBoost on node features + per-relation 1-hop neighbour aggregates (weighted mean of neighbour
features and weighted degree). Aggregates are computed on the *masked* graph passed in, so they
respect the temporal cutoff like every other model."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier


def neighbour_aggregates(x: np.ndarray, edges: dict[str, np.ndarray],
                         weights: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, list[str]]:
    n, d = x.shape
    cols, names = [], []
    for r, e in edges.items():
        w = np.ones(e.shape[1]) if weights is None or r not in weights else np.asarray(weights[r], float)
        src = np.concatenate([e[0], e[1]]); dst = np.concatenate([e[1], e[0]]); ww = np.concatenate([w, w])
        den = np.bincount(dst, weights=ww, minlength=n)
        agg = np.zeros((n, d))
        for j in range(d):
            agg[:, j] = np.bincount(dst, weights=ww * x[src, j], minlength=n)
        agg = agg / np.maximum(den, 1e-9)[:, None]
        agg[den == 0] = 0.0
        cols.append(agg); cols.append(den[:, None])
        names += [f"{r}_nbmean_{j}" for j in range(d)] + [f"{r}_wdeg"]
    return np.hstack(cols), names


def make_table(x: np.ndarray, edges, weights=None) -> np.ndarray:
    agg, _ = neighbour_aggregates(x, edges, weights)
    return np.hstack([x, agg])


def train_catboost(xtab: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray | None,
                   seed: int, iterations: int = 600) -> CatBoostClassifier:
    clf = CatBoostClassifier(iterations=iterations, depth=6, learning_rate=0.05, loss_function="Logloss",
                             auto_class_weights="Balanced", random_seed=seed, verbose=False,
                             eval_metric="PRAUC", od_type="Iter", od_wait=50, thread_count=-1)
    if val_idx is not None and len(val_idx) and y[val_idx].sum() > 0:
        clf.fit(xtab[train_idx], y[train_idx], eval_set=(xtab[val_idx], y[val_idx]), use_best_model=True)
    else:
        clf.fit(xtab[train_idx], y[train_idx])
    return clf
