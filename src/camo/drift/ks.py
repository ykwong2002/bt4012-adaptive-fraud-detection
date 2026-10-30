from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic and p-value (NaNs dropped)."""
    a = np.asarray(reference, float); b = np.asarray(current, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    res = ks_2samp(a, b)
    return float(res.statistic), float(res.pvalue)
