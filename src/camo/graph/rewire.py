"""Soft / hard rewiring statistics for the tau sweep (label-based columns are EDA-only diagnostics)."""
from __future__ import annotations

import numpy as np
import pandas as pd


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
