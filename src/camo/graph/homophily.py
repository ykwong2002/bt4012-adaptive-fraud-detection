"""Homophily diagnostics (uses labels: EDA only, never inside the method)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def node_homophily(edges: np.ndarray, y: np.ndarray, n: int,
                   weights: np.ndarray | None = None) -> np.ndarray:
    """Per-node (weighted) fraction of neighbours with the same label. NaN for isolated nodes."""
    u, v = edges
    w = np.ones(u.size) if weights is None else np.asarray(weights, float)
    same = (y[u] == y[v]).astype(float)
    num = np.bincount(u, weights=w * same, minlength=n) + np.bincount(v, weights=w * same, minlength=n)
    den = np.bincount(u, weights=w, minlength=n) + np.bincount(v, weights=w, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def homophily_table(edges: dict[str, np.ndarray], y: np.ndarray, n: int,
                    weights: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Mean node homophily per relation, split by class, plus edge homophily and degree."""
    rows = []
    for r, e in edges.items():
        w = None if weights is None else weights.get(r)
        h = node_homophily(e, y, n, w)
        deg = np.bincount(e[0], minlength=n) + np.bincount(e[1], minlength=n)
        rows.append({"relation": r, "edges": int(e.shape[1]),
                     "edge_homophily": float((y[e[0]] == y[e[1]]).mean()) if e.shape[1] else np.nan,
                     "node_homophily_fraud": float(np.nanmean(h[y == 1])),
                     "node_homophily_legit": float(np.nanmean(h[y == 0])),
                     "frac_fraud_edges_to_legit": float((y[e[0]] != y[e[1]]).sum() / max(1, ((y[e[0]] == 1) | (y[e[1]] == 1)).sum())),
                     "mean_degree_fraud": float(deg[y == 1].mean()), "mean_degree_legit": float(deg[y == 0].mean()),
                     "isolated_frac": float((deg == 0).mean())})
    return pd.DataFrame(rows)


def text_homophily(edges: np.ndarray, scores: np.ndarray, n: int, dates=None) -> np.ndarray:
    """h_text(v) = mean edge-consistency score over v's neighbours. Label-free.
    If `dates` is given, only neighbours dated <= date(v) count (the PIT monitor, decisions D5)."""
    u, v = edges
    s = np.asarray(scores, float)
    if dates is None:
        num = np.bincount(u, weights=s, minlength=n) + np.bincount(v, weights=s, minlength=n)
        den = np.bincount(u, minlength=n) + np.bincount(v, minlength=n).astype(float)
    else:
        # for endpoint a, neighbour b counts iff date(b) <= date(a)
        ok_u = dates[v] <= dates[u]   # v is a valid neighbour of u
        ok_v = dates[u] <= dates[v]   # u is a valid neighbour of v
        num = np.bincount(u[ok_u], weights=s[ok_u], minlength=n) + np.bincount(v[ok_v], weights=s[ok_v], minlength=n)
        den = np.bincount(u[ok_u], minlength=n) + np.bincount(v[ok_v], minlength=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)
