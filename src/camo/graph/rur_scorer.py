"""R-U-R-anchored edge scorer for the graph-only track (decisions D15). Label-free.

R-U-R (same user) is the one relation camouflage does not break: 93 % of a fraud review's R-U-R
neighbours are fraud. So we describe each review by its *user profile*, the mean of its own and its
R-U-R neighbours' (quantile-normalised) features, i.e. how this account behaves across its reviews,
and score an R-S-R / R-T-R edge by how close the two profiles are:

    score(u, v) = exp( -||p_u - p_v|| / s ),   s = median distance over edges among fit nodes

R-U-R edges themselves keep weight 1. Time-respecting: an edge is scored when it appears (the later
arrival of its two endpoints), with profiles built only from R-U-R neighbours visible by then, so no
score depends on a review that has not arrived. The quantile transform is fitted on `fit_idx` only.
Selected among a handful of variants on training-wave edges only (edge AUC ~0.67 there vs 0.52 for
the link-prediction scorer and 0.54 for cosine); test labels were never looked at.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import QuantileTransformer


def _profiles(z: np.ndarray, rur: np.ndarray, visible: np.ndarray) -> np.ndarray:
    n = z.shape[0]
    keep = visible[rur[0]] & visible[rur[1]]
    e = rur[:, keep]
    a = sp.coo_matrix((np.ones(e.shape[1], np.float32), (e[0], e[1])), shape=(n, n))
    a = (a + a.T + sp.eye(n, dtype=np.float32)).tocsr()
    return (np.asarray(a @ z) / np.asarray(a.sum(1))).astype(np.float32)


def rur_profile_scores(x: np.ndarray, edges: dict[str, np.ndarray], fit_idx: np.ndarray,
                       arrival: np.ndarray | None = None, anchor: str = "rur",
                       chunk: int = 500_000) -> dict[str, np.ndarray]:
    """Edge scores in (0, 1] per relation. `arrival`: integer arrival stage per node (waves or
    temporal windows); None scores everything on the full graph (stratified ablation only)."""
    n = x.shape[0]
    qt = QuantileTransformer(n_quantiles=min(1000, len(fit_idx)), output_distribution="normal", random_state=0)
    z = qt.fit(x[fit_idx]).transform(x).astype(np.float32)
    arrival = np.zeros(n, np.int64) if arrival is None else np.asarray(arrival)
    fit = np.zeros(n, bool); fit[fit_idx] = True
    rur = edges[anchor]
    dist = {r: np.zeros(e.shape[1], np.float32) for r, e in edges.items() if r != anchor}
    for t in np.unique(arrival):
        p = _profiles(z, rur, arrival <= t)
        for r, e in edges.items():
            if r == anchor:
                continue
            idx = np.where(np.maximum(arrival[e[0]], arrival[e[1]]) == t)[0]
            for i in range(0, idx.size, chunk):
                j = idx[i:i + chunk]
                dist[r][j] = np.linalg.norm(p[e[0, j]] - p[e[1, j]], axis=1)
    fit_d = np.concatenate([d[fit[edges[r][0]] & fit[edges[r][1]]] for r, d in dist.items()])
    scale = float(np.median(fit_d)) if fit_d.size else 1.0
    out = {r: np.exp(-d / max(scale, 1e-9)).astype(np.float32) for r, d in dist.items()}
    out[anchor] = np.ones(rur.shape[1], np.float32)
    return out
