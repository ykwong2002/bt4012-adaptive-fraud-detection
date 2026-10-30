"""Temporal windows, graph masking (the main leak guard) and the stratified fallback.

Rules (AGENTS.md):
  * any reported number uses temporal windows + mask_edges(cutoff); nodes dated after the cutoff and
    their edges do not exist for the model at that time;
  * `stratified_split` is only for the early bias-ablation on the .mat graph (no dates) and must be
    reported as such.
"""
from __future__ import annotations

import numpy as np


def equal_count_windows(dates: np.ndarray, n_windows: int) -> np.ndarray:
    """Window id (0..n_windows-1) per node, equal counts, no date split across windows."""
    order = np.argsort(dates, kind="stable")
    ranks = np.empty(len(dates), dtype=np.int64)
    ranks[order] = np.arange(len(dates))
    w = (ranks * n_windows) // len(dates)
    # make windows date-consistent: a calendar day belongs to the window of its first review
    w_day = {}
    for i in order:
        w_day.setdefault(dates[i], w[i])
    return np.array([w_day[d] for d in dates], dtype=np.int64)


def window_bounds(dates: np.ndarray, windows: np.ndarray) -> list[tuple]:
    return [(int(k), dates[windows == k].min(), dates[windows == k].max(), int((windows == k).sum()))
            for k in np.unique(windows)]


def mask_edges(edges: dict[str, np.ndarray], dates: np.ndarray, cutoff) -> dict[str, np.ndarray]:
    """Keep only edges whose two endpoints are both dated <= cutoff."""
    ok = dates <= cutoff
    return {r: e[:, ok[e[0]] & ok[e[1]]] for r, e in edges.items()}


def visible_nodes(dates: np.ndarray, cutoff) -> np.ndarray:
    return np.where(dates <= cutoff)[0]


def temporal_plan(windows: np.ndarray, n_train: int, n_val: int) -> dict:
    """Which windows the incumbent trains on, tunes on, and is evaluated on."""
    ks = np.unique(windows)
    return {"train": ks[:n_train].tolist(), "val": ks[n_train:n_train + n_val].tolist(),
            "test": ks[n_train + n_val:].tolist()}


def stratified_split(y: np.ndarray, seed: int, train: float = 0.6, val: float = 0.2):
    """Random stratified split for the no-dates .mat graph. NOT for reported temporal results."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        a, b = int(train * len(idx)), int((train + val) * len(idx))
        tr.append(idx[:a]); va.append(idx[a:b]); te.append(idx[b:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)), np.sort(np.concatenate(te))
