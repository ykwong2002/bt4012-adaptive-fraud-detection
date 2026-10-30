"""Label-free drift monitors per temporal window and the trigger rule (decisions D3, D5).

Signals per window t vs the reference (training windows):
  * PSI / KS on each point-in-time behavioural feature (mean and max over features)
  * PSI / KS on each LLM review field
  * PSI / KS on text-homophily h_text computed on the RAW graph with neighbours dated <= the node
  * convergence: KS distance between refined-graph and raw-graph h_text within window t
Labelled comparator (from scripts/06 results): incumbent AUC-PR per window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from camo.drift.ks import ks
from camo.drift.psi import psi
from camo.graph.homophily import text_homophily


def h_text_raw_and_refined(edges, scores, n, dates, tau: float):
    """Both h_text variants from the same edge scores (raw = all dated neighbours; refined = kept
    neighbours after hard pruning at tau, so that refined != raw by construction only via pruning)."""
    rows_raw, rows_ref = np.zeros(n), np.zeros(n)
    cnt_raw, cnt_ref = np.zeros(n), np.zeros(n)
    for r, e in edges.items():
        if e.shape[1] == 0:
            continue
        s = scores[r]
        h = text_homophily(e, s, n, dates)
        m = ~np.isnan(h); rows_raw[m] += h[m]; cnt_raw[m] += 1
        keep = s >= tau
        h2 = text_homophily(e[:, keep], s[keep], n, dates)
        m2 = ~np.isnan(h2); rows_ref[m2] += h2[m2]; cnt_ref[m2] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt_raw > 0, rows_raw / cnt_raw, np.nan), np.where(cnt_ref > 0, rows_ref / cnt_ref, np.nan)


def window_signals(windows: np.ndarray, ref_windows: list[int], blocks: dict[str, tuple[np.ndarray, list[str]]],
                   h_raw: np.ndarray | None, h_ref: np.ndarray | None, bins: int = 10) -> pd.DataFrame:
    """blocks: name -> (matrix n x d, column names). One row per (window, signal)."""
    ref = np.isin(windows, ref_windows)
    rows = []
    for t in np.unique(windows):
        if t in ref_windows:
            continue
        cur = windows == t
        for name, (X, cols) in blocks.items():
            ps = [psi(X[ref, j], X[cur, j], bins) for j in range(X.shape[1])]
            kp = [ks(X[ref, j], X[cur, j])[1] for j in range(X.shape[1])]
            rows.append({"window": int(t), "signal": name, "psi_mean": float(np.nanmean(ps)), "psi_max": float(np.nanmax(ps)),
                         "ks_p_min": float(np.nanmin(kp)), "top_feature": cols[int(np.nanargmax(ps))]})
        if h_raw is not None:
            stat, pv = ks(h_raw[ref], h_raw[cur])
            rows.append({"window": int(t), "signal": "h_text_raw", "psi_mean": psi(h_raw[ref], h_raw[cur], bins),
                         "psi_max": np.nan, "ks_p_min": pv, "ks_stat": stat,
                         "mean_ref": float(np.nanmean(h_raw[ref])), "mean_cur": float(np.nanmean(h_raw[cur]))})
        if h_raw is not None and h_ref is not None:
            stat, pv = ks(h_ref[cur], h_raw[cur])
            rows.append({"window": int(t), "signal": "convergence_refined_vs_raw", "psi_mean": np.nan, "psi_max": np.nan,
                         "ks_p_min": pv, "ks_stat": stat, "mean_ref": float(np.nanmean(h_ref[cur])), "mean_cur": float(np.nanmean(h_raw[cur]))})
    return pd.DataFrame(rows)


def trigger(flags: np.ndarray, k: int) -> np.ndarray:
    """Alarm at window i if the last k windows (including i) were all flagged."""
    out = np.zeros(len(flags), dtype=bool)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run >= k
    return out


def detection_latency(alarm: np.ndarray, degraded: np.ndarray) -> dict:
    """Windows between the first true degradation and the first alarm (NaN if never alarmed;
    negative = alarm before degradation = early warning or false alarm)."""
    d = np.where(degraded)[0]; a = np.where(alarm)[0]
    if len(d) == 0:
        return {"first_degraded": None, "first_alarm": int(a[0]) if len(a) else None, "latency": np.nan,
                "false_alarms": int(alarm.sum())}
    first_d = int(d[0]); first_a = int(a[0]) if len(a) else None
    return {"first_degraded": first_d, "first_alarm": first_a,
            "latency": (first_a - first_d) if first_a is not None else np.nan,
            "false_alarms": int(alarm[:first_d].sum())}
