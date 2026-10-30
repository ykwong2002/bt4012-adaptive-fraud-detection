"""Population Stability Index with bins fixed on the reference distribution's quantiles."""
from __future__ import annotations

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    reference = np.asarray(reference, float); current = np.asarray(current, float)
    reference = reference[~np.isnan(reference)]; current = current[~np.isnan(current)]
    if reference.size == 0 or current.size == 0:
        return np.nan
    qs = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if qs.size < 3:                      # near-constant feature
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    r = np.histogram(reference, qs)[0] / reference.size
    c = np.histogram(current, qs)[0] / current.size
    r = np.clip(r, eps, None); c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_table(ref: "np.ndarray", cur: "np.ndarray", names: list[str], bins: int = 10) -> dict[str, float]:
    return {n: psi(ref[:, j], cur[:, j], bins) for j, n in enumerate(names)}
