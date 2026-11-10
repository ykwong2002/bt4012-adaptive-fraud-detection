"""Soft / hard rewiring statistics for the tau sweep (label-based columns are EDA-only diagnostics)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def oracle_scores(edges: dict[str, np.ndarray], y: np.ndarray) -> dict[str, np.ndarray]:
    """Upper bound for any rewiring: 1 for same-label edges, 0 for cross-label. Diagnostic only."""
    return {r: (y[e[0]] == y[e[1]]).astype(np.float32) for r, e in edges.items()}


def noisy_oracle_scores(edges: dict[str, np.ndarray], y: np.ndarray, auc: float, seed: int = 0) -> dict[str, np.ndarray]:
    """Simulated scorer with a chosen same- vs cross-label AUC. Equal-variance binormal model:
    z = d * same + N(0, 1) with d = sqrt(2) * Phi^-1(auc), squashed to (0, 1). Diagnostic only."""
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    d = np.sqrt(2) * norm.ppf(auc)
    out = {}
    for r, e in edges.items():
        z = d * (y[e[0]] == y[e[1]]) + rng.standard_normal(e.shape[1]) - d / 2
        out[r] = (1 / (1 + np.exp(-z))).astype(np.float32)
    return out


def edge_label_auc(edges: dict[str, np.ndarray], scores: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    """Per-relation AUC of the edge score for same- vs cross-label edges. Diagnostic only."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for r, e in edges.items():
        same = y[e[0]] == y[e[1]]
        out[r] = float(roc_auc_score(same, scores[r])) if 0 < same.sum() < len(same) else float("nan")
    return out


def tau_sweep_table(edges: dict[str, np.ndarray], scores: dict[str, np.ndarray], y: np.ndarray,
                    taus: list[float]) -> pd.DataFrame:
    rows = []
    for r, e in edges.items():
        s = scores[r]; hetero = y[e[0]] != y[e[1]]; touches_fraud = (y[e[0]] == 1) | (y[e[1]] == 1)
        rows.append({"relation": r, "tau": "soft", "kept_frac": 1.0, "mean_weight": float(s.mean()),
                     "mean_weight_hetero": float(s[hetero].mean()) if hetero.any() else np.nan,
                     "mean_weight_homo": float(s[~hetero].mean()) if (~hetero).any() else np.nan,
                     "mean_weight_touches_fraud": float(s[touches_fraud].mean()) if touches_fraud.any() else np.nan})
        for t in taus:
            keep = s >= t
            rows.append({"relation": r, "tau": t, "kept_frac": float(keep.mean()),
                         "hetero_frac_dropped": float(hetero[~keep].mean()) if (~keep).any() else np.nan,
                         "hetero_frac_kept": float(hetero[keep].mean()) if keep.any() else np.nan,
                         "fraud_isolated_frac": float(np.isin(np.where(y == 1)[0], np.unique(e[:, keep]), invert=True).mean())})
    return pd.DataFrame(rows)
