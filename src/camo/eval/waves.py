"""Synthetic arrival waves for graphs without dates (the graph-only track, decisions D13).

The public .mat graph has no timestamps, so the temporal protocol cannot run on it directly. We give
every node a random *arrival wave* (stratified by label, re-drawn per seed) and encode the wave as a
pseudo-date. The temporal machinery then applies unchanged: the incumbent trains on the first waves
with every later node and its edges masked out, the next wave tunes, and each test wave is scored on
the graph as it stands at the end of that wave. Clean waves are i.i.d. by construction, so any
degradation across waves comes from what is injected into them (camouflage), not from natural drift.
"""
from __future__ import annotations

import numpy as np

from camo.data.load_mat import Graph

EPOCH = np.datetime64("2000-01-01", "D")


def assign_waves(y: np.ndarray, n_windows: int, seed: int) -> np.ndarray:
    """Wave id (0..n_windows-1) per node: within each class a random permutation is cut into
    n_windows near-equal parts, so every wave has the same fraud rate up to rounding."""
    rng = np.random.default_rng(seed)
    w = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for k, part in enumerate(np.array_split(idx, n_windows)):
            w[part] = k
    return w


def wave_dates(windows: np.ndarray) -> np.ndarray:
    return EPOCH + np.asarray(windows, dtype=np.int64).astype("timedelta64[D]")


def with_waves(g: Graph, windows: np.ndarray) -> Graph:
    """Copy of g whose dates are the arrival waves (any real dates are replaced)."""
    meta = {**g.meta, "protocol": "waves"}
    return Graph(g.x, g.y, g.edges, wave_dates(windows), meta, g.node_table)
