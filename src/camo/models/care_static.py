"""CARE-static: the neighbour-filtering idea of CARE-GNN (Dou et al. 2020) with a fixed keep-ratio
per relation instead of the RL-tuned threshold. A label-aware scorer (logistic regression on node
features, trained on *train labels only*) gives each node a score s; for every node and relation we
keep the rho fraction of neighbours with the smallest |s_u - s_v|. The filtered graph is then fed to
WeightedSAGE, so the only difference from the baseline is the neighbour filter."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def label_aware_scores(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, seed: int) -> np.ndarray:
    sc = StandardScaler().fit(x[train_idx])
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(x[train_idx]), y[train_idx])
    return clf.predict_proba(sc.transform(x))[:, 1]


def care_filter(edges: np.ndarray, s: np.ndarray, rho: float) -> np.ndarray:
    """Per node keep the ceil(rho*deg) most label-similar neighbours (undirected: an edge survives if
    either endpoint keeps it, as in CARE-GNN's per-centre-node selection)."""
    if edges.shape[1] == 0:
        return edges
    e = np.hstack([edges, edges[::-1]])
    centre, nb = e[0], e[1]
    dist = np.abs(s[centre] - s[nb])
    order = np.lexsort((dist, centre))
    centre_o, nb_o, dist_o = centre[order], nb[order], dist[order]
    first = np.r_[True, centre_o[1:] != centre_o[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(centre_o)), 0))
    rank = np.arange(len(centre_o)) - start
    deg = np.bincount(centre_o)
    keep = rank < np.ceil(rho * deg[centre_o])
    kept = np.vstack([np.minimum(centre_o[keep], nb_o[keep]), np.maximum(centre_o[keep], nb_o[keep])])
    kept = np.unique(kept, axis=1)
    return kept.astype(np.int64)


def care_filter_all(edges: dict[str, np.ndarray], s: np.ndarray, rho: dict[str, float] | float) -> dict[str, np.ndarray]:
    return {r: care_filter(e, s, rho[r] if isinstance(rho, dict) else rho) for r, e in edges.items()}
