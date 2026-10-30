from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_k(y: np.ndarray, s: np.ndarray, k: float = 0.1) -> float:
    """Recall of fraud among the top-k fraction of scores."""
    n = max(1, int(round(k * len(y))))
    top = np.argsort(-s, kind="stable")[:n]
    return float(y[top].sum() / max(1, y.sum()))


def evaluate(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return {"auc_pr": np.nan, "auroc": np.nan, "recall@5%": np.nan, "recall@10%": np.nan, "n": len(y), "n_pos": int(y.sum())}
    return {"auc_pr": float(average_precision_score(y, s)), "auroc": float(roc_auc_score(y, s)),
            "recall@5%": recall_at_k(y, s, 0.05), "recall@10%": recall_at_k(y, s, 0.10),
            "n": int(len(y)), "n_pos": int(y.sum())}


def summarise(rows: list[dict], by: list[str]) -> pd.DataFrame:
    """mean ± std over seeds for each metric, grouped by `by`."""
    df = pd.DataFrame(rows)
    metrics = [c for c in ["auc_pr", "auroc", "recall@5%", "recall@10%"] if c in df]
    g = df.groupby(by)[metrics]
    out = g.mean().round(4).astype(str) + " ± " + g.std().fillna(0).round(4).astype(str)
    out["n_seeds"] = g.size()
    return out.reset_index()
