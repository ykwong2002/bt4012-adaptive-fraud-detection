"""Label-free drift monitors per temporal window and the trigger rule (decisions D3, D5, D8).

Signals per window t vs the reference (training windows), all as effect sizes, never p-values:
  * PSI and KS statistic on each point-in-time behavioural feature (mean PSI, max KS over features),
    excluding features that grow with time by construction (counts since the start of the data)
  * the same on each LLM review field
  * PSI / KS on text-homophily h_text computed on the RAW graph with neighbours dated <= the node
  * convergence (diagnostic): KS distance between refined-graph and raw-graph h_text within window t

Thresholds are calibrated, not fixed (review issue #10). With ~5.7k reviews per window a KS p-value is
below 0.01 for any difference at all, so p-values flag every window. Instead each training window is
compared with the other training windows (leave-one-window-out): that is how much the data moves when
nothing is wrong. A window is flagged when its statistic exceeds the largest such value times
`null_margin` (and at least the configured floor). Labelled comparator (from scripts/06): incumbent
AUC-PR per window.
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


def block_stats(Xr: np.ndarray, Xc: np.ndarray, cols: list[str], bins: int = 10) -> dict:
    ps = [psi(Xr[:, j], Xc[:, j], bins) for j in range(Xr.shape[1])]
    kst = [ks(Xr[:, j], Xc[:, j])[0] for j in range(Xr.shape[1])]
    return {"psi_mean": float(np.nanmean(ps)), "psi_max": float(np.nanmax(ps)), "ks_max": float(np.nanmax(kst)),
            "top_feature": cols[int(np.nanargmax(ps))]}


def h_stats(hr: np.ndarray, hc: np.ndarray, bins: int = 10) -> dict:
    return {"psi_mean": psi(hr, hc, bins), "psi_max": np.nan, "ks_max": ks(hr, hc)[0],
            "mean_ref": float(np.nanmean(hr)), "mean_cur": float(np.nanmean(hc))}


def _stats(ref: np.ndarray, cur: np.ndarray, blocks, h_raw, bins) -> list[dict]:
    out = [{"signal": name, **block_stats(X[ref], X[cur], cols, bins)} for name, (X, cols) in blocks.items()]
    if h_raw is not None:
        out.append({"signal": "h_text_raw", **h_stats(h_raw[ref], h_raw[cur], bins)})
    return out


def window_signals(windows: np.ndarray, ref_windows: list[int], blocks: dict[str, tuple[np.ndarray, list[str]]],
                   h_raw: np.ndarray | None, h_ref: np.ndarray | None, bins: int = 10) -> pd.DataFrame:
    """blocks: name -> (matrix n x d, column names). One row per (window, signal).
    Rows with null=True are the leave-one-window-out comparisons among the reference windows, used to
    calibrate thresholds; null=False rows are the monitored windows."""
    rows = []
    for t in np.unique(windows):
        cur = windows == t
        if t in ref_windows:
            others = [k for k in ref_windows if k != t]
            if not others:
                continue
            ref = np.isin(windows, others)
            rows += [{"window": int(t), "null": True, **r} for r in _stats(ref, cur, blocks, h_raw, bins)]
            continue
        ref = np.isin(windows, ref_windows)
        rows += [{"window": int(t), "null": False, **r} for r in _stats(ref, cur, blocks, h_raw, bins)]
        if h_raw is not None and h_ref is not None:
            rows.append({"window": int(t), "null": False, "signal": "convergence_refined_vs_raw", "psi_mean": np.nan,
                         "psi_max": np.nan, "ks_max": ks(h_ref[cur], h_raw[cur])[0],
                         "mean_ref": float(np.nanmean(h_ref[cur])), "mean_cur": float(np.nanmean(h_raw[cur]))})
    return pd.DataFrame(rows)


def calibrate(sig: pd.DataFrame, margin: float, min_psi: float, min_ks: float) -> pd.DataFrame:
    """Per-signal thresholds from the null (leave-one-reference-window-out) rows."""
    null = sig[sig["null"]]
    rows = []
    for name in sig.loc[~sig["null"], "signal"].unique():
        n = null[null["signal"] == name]
        psi_n, ks_n = n["psi_mean"].max(), n["ks_max"].max()
        rows.append({"signal": name, "n_null": len(n), "null_psi_max": psi_n, "null_ks_max": ks_n,
                     "psi_thr": max(min_psi, margin * psi_n) if len(n) and np.isfinite(psi_n) else np.nan,
                     "ks_thr": max(min_ks, margin * ks_n) if len(n) and np.isfinite(ks_n) else np.nan})
    return pd.DataFrame(rows)


def flag_windows(sig: pd.DataFrame, thresholds: pd.DataFrame, alarm_signals: list[str], k: int) -> pd.DataFrame:
    """One row per monitored window: per-signal stats of the alarm signals, flag and k-consecutive alarm.
    A signal flags when its PSI or KS statistic exceeds its calibrated threshold."""
    cur = sig[~sig["null"]].merge(thresholds, on="signal", how="left")
    cur["flag"] = (cur["psi_mean"] > cur["psi_thr"]) | (cur["ks_max"] > cur["ks_thr"])
    use = [s for s in alarm_signals if s in set(cur["signal"])]
    m = cur[cur["signal"].isin(use)]
    per = m.groupby("window").agg(flag=("flag", "any"), psi_mean=("psi_mean", "max"), ks_max=("ks_max", "max"),
                                  psi_thr=("psi_thr", "max"), ks_thr=("ks_thr", "max")).reset_index().sort_values("window")
    per["monitor"] = "+".join(use)
    per["alarm"] = trigger(per["flag"].to_numpy(), k)
    return per


def monitor_blocks(g, dcfg: dict, feature_names: list[str], llm_fields=None, llm_names=None) -> dict:
    """Signal blocks for the monitor, with trend features (counts that grow with time) removed."""
    excl = set(dcfg.get("exclude_features", []))
    keep = [j for j, n in enumerate(feature_names) if n not in excl]
    blocks = {"features_pit": (g.x[:, keep], [feature_names[j] for j in keep])}
    if llm_fields is not None:
        blocks["llm_fields"] = (llm_fields, llm_names)
    return blocks


def run_monitor(g, windows: np.ndarray, ref_windows: list[int], dcfg: dict, feature_names: list[str],
                edge_scores: dict | None = None, llm_fields=None, llm_names=None, tau: float = 0.5):
    """Everything the scripts need: (signals table, thresholds, per-window flags/alarms)."""
    blocks = monitor_blocks(g, dcfg, feature_names, llm_fields, llm_names)
    h_raw = h_ref = None
    if edge_scores is not None:
        h_raw, h_ref = h_text_raw_and_refined(g.edges, edge_scores, g.n, g.dates, tau)
    sig = window_signals(windows, ref_windows, blocks, h_raw, h_ref, dcfg["psi_bins"])
    thr = calibrate(sig, dcfg["null_margin"], dcfg["min_psi"], dcfg["min_ks"])
    wanted = [s for s in dcfg["alarm_signals"] if s in set(sig["signal"])] or ["features_pit"]
    flags = flag_windows(sig, thr, wanted, dcfg["k_consecutive"])
    return sig, thr, flags, h_raw, h_ref


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
