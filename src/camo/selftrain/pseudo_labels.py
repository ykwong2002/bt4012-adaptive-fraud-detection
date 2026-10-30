"""Pseudo-label sources for the label-scarce response (decisions D6).

  * incumbent score s(v)
  * community propensity c(v): mean incumbent score of v's Louvain community (excluding v), with
    Louvain run on R-U-R ∪ scorer-weighted R-S-R (plain R-S-R returns product×star blocks)
  * autoencoder anomaly rank a(v)
  fused f(v) = w_s s + w_c c + w_a a   (weights in configs/selftrain.yaml)
Prior estimate for the positive share of the unlabelled pool: training base rate, and a
Saerens-Latinne-Decaestecker EM estimate from the incumbent's (calibrated) scores.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.stats import rankdata


def community_labels(n: int, edges: dict[str, np.ndarray], weights: dict[str, np.ndarray] | None,
                     relations: list[str], seed: int, max_edges: int = 500_000, resolution: float = 1.0) -> np.ndarray:
    """Louvain community id per node. `rsr_weighted` keeps the top-weighted R-S-R edges only."""
    G = nx.Graph(); G.add_nodes_from(range(n))
    for r in relations:
        base = r.replace("_weighted", "")
        e = edges[base]
        w = None if weights is None or base not in weights else weights[base]
        if e.shape[1] > max_edges:
            if w is not None:
                keep = np.argsort(-w)[:max_edges]
            else:
                keep = np.random.default_rng(seed).choice(e.shape[1], max_edges, replace=False)
            e, w = e[:, keep], None if w is None else w[keep]
        ww = np.ones(e.shape[1]) if w is None else w
        G.add_weighted_edges_from(zip(e[0].tolist(), e[1].tolist(), ww.tolist()))
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=resolution)
    lab = np.zeros(n, dtype=np.int64)
    for k, c in enumerate(comms):
        lab[list(c)] = k
    return lab


def community_propensity(score: np.ndarray, comm: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Mean score of the other pool members in the same community (leave-one-out); singletons get
    their own score."""
    s = np.zeros_like(score); s[pool] = score[pool]
    tot = np.bincount(comm, weights=s, minlength=comm.max() + 1)
    cnt = np.bincount(comm[pool], minlength=comm.max() + 1)
    out = score.copy()
    c = comm[pool]
    loo_cnt = cnt[c] - 1
    out[pool] = np.where(loo_cnt > 0, (tot[c] - score[pool]) / np.maximum(loo_cnt, 1), score[pool])
    return out


def fuse(score: np.ndarray, comm_prop: np.ndarray, anomaly: np.ndarray, pool: np.ndarray,
         w: tuple[float, float, float] = (0.5, 0.3, 0.2)) -> np.ndarray:
    a = np.zeros_like(score); a[pool] = rankdata(anomaly[pool]) / len(pool)
    return w[0] * score + w[1] * comm_prop + w[2] * a


def em_prior(scores: np.ndarray, train_prior: float, iters: int = 100, tol: float = 1e-6) -> float:
    """Saerens et al. (2002) EM for label shift: re-estimates the positive prior in a new sample from
    posteriors calibrated on the training prior."""
    p = scores.clip(1e-6, 1 - 1e-6); pi_tr = train_prior; pi = train_prior
    for _ in range(iters):
        num = p * pi / pi_tr; den = num + (1 - p) * (1 - pi) / (1 - pi_tr)
        post = num / den
        new = float(post.mean())
        if abs(new - pi) < tol:
            break
        pi = new
    return float(np.clip(pi, 1e-4, 1 - 1e-4))


def select_pseudo(fused: np.ndarray, pool: np.ndarray, q: float, per_class: bool, prior: float | None):
    """Return (idx, labels) of pseudo-labelled pool nodes.
    naive: the most confident q-fraction of the pool regardless of class (confidence = |f-0.5|)
    per_class: the most confident q-fraction among predicted positives AND among predicted negatives
    prior: if given, the positive share of the pseudo set is forced to `prior` (prior matching)."""
    f = fused[pool]
    n_take = max(2, int(round((1 - q) * len(pool))))  # q is the confidence quantile -> keep the top (1-q)
    if prior is not None:
        n_pos = max(1, int(round(prior * n_take))); n_neg = n_take - n_pos
        order = np.argsort(-f)
        pos = order[:n_pos]; neg = order[::-1][:n_neg]
    elif per_class:
        pred = f >= 0.5
        pos_c = np.where(pred)[0]; neg_c = np.where(~pred)[0]
        n_pos = max(1, int(round((1 - q) * len(pos_c)))) if len(pos_c) else 0
        n_neg = max(1, int(round((1 - q) * len(neg_c)))) if len(neg_c) else 0
        pos = pos_c[np.argsort(-f[pos_c])[:n_pos]] if n_pos else np.array([], int)
        neg = neg_c[np.argsort(f[neg_c])[:n_neg]] if n_neg else np.array([], int)
    else:
        conf = np.abs(f - 0.5)
        top = np.argsort(-conf)[:n_take]
        pos = top[f[top] >= 0.5]; neg = top[f[top] < 0.5]
    idx = np.concatenate([pool[pos], pool[neg]])
    lab = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
    return idx, lab
