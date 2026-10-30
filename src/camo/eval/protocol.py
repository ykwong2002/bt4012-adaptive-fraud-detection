"""Evaluation protocol shared by scripts 06-09: which nodes/edges are visible when.

Temporal (graphs with dates): incumbent trains on the first k windows with the graph masked at the
end of those windows; the val window selects hyper-parameters / early stopping; each test window t is
scored with the graph masked at the end of t (later windows do not exist yet; nodes between train
end and t are visible but unlabelled). Stratified (no dates, .mat only): random split, flagged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from camo.eval.splits import equal_count_windows, stratified_split, temporal_plan


@dataclass
class Fold:
    name: str                       # "train", "val" or "test_w<k>"
    eval_idx: np.ndarray            # nodes scored/evaluated in this fold
    edges: dict[str, np.ndarray]    # edges visible in this fold
    weights: dict[str, np.ndarray] | None
    cutoff: object = None


def _mask(edges, weights, dates, cutoff):
    if dates is None:
        return edges, weights
    ok = dates <= cutoff
    out_e, out_w = {}, {} if weights is not None else None
    for r, e in edges.items():
        keep = ok[e[0]] & ok[e[1]]
        out_e[r] = e[:, keep]
        if weights is not None:
            out_w[r] = weights[r][keep]
    return out_e, out_w


def apply_rewiring(edges, weights, mode: str | None, tau: float = 0.0):
    """soft: keep weights; hard: drop edges with weight < tau, then unweighted; None: raw graph."""
    if weights is None or mode is None:
        return edges, None
    if mode == "soft":
        return edges, weights
    if mode == "softrank":   # per-relation percentile rank of the score: same ordering, full [0,1] spread
        from scipy.stats import rankdata
        return edges, {r: (rankdata(w) / max(1, len(w))).astype(np.float32) if len(w) else w for r, w in weights.items()}
    if mode == "hard":
        return {r: e[:, weights[r] >= tau] for r, e in edges.items()}, None
    raise ValueError(mode)


def build_folds(g, cfg: dict, seed: int, weights=None, mode=None, tau=0.0, smoke=False):
    """Returns (protocol_name, train_idx, folds) with folds = [train, val, test_w...]."""
    edges, weights = apply_rewiring(g.edges, weights, mode, tau)
    if g.dates is None:
        tr, va, te = stratified_split(g.y, seed)
        f = lambda name, idx: Fold(name, idx, edges, weights)  # noqa: E731
        return "stratified", tr, [f("train", tr), f("val", va), f("test", te)]
    w = equal_count_windows(g.dates, cfg["windows"]["n_windows"])
    plan = temporal_plan(w, cfg["windows"]["n_train_windows"], cfg["windows"]["n_val_windows"])
    folds = []
    tr = np.where(np.isin(w, plan["train"]))[0]
    cut = g.dates[tr].max()
    e, ww = _mask(edges, weights, g.dates, cut)
    folds.append(Fold("train", tr, e, ww, cut))
    va = np.where(np.isin(w, plan["val"]))[0]
    cut = g.dates[va].max()
    e, ww = _mask(edges, weights, g.dates, cut)
    folds.append(Fold("val", va, e, ww, cut))
    for k in plan["test"]:
        te = np.where(w == k)[0]
        cut = g.dates[te].max()
        e, ww = _mask(edges, weights, g.dates, cut)
        folds.append(Fold(f"test_w{k}", te, e, ww, cut))
    return "temporal", tr, folds
