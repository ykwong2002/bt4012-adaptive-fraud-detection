"""Paired uncertainty for the graph-only results (decisions D17).

Every comparison is paired on the seed (same arrival waves, same initialisation): per seed we average
the difference over the evaluated waves, then report the mean over seeds, a 95 % t-interval across
seeds (n-1 degrees of freedom; wide with 3 seeds, which is the honest statement) and the share of
individual (seed, wave) pairs in which the difference is positive.

  * rewiring:  AUC-PR(refined graph, scorer s) - AUC-PR(raw graph), per model, under waves
  * loop:      deployed - control and deployed - incumbent, per self-training variant, on the waves
               after the first retrain (attack scenario, scripts/11 on .mat)
  * selftrain: the same two differences for the text track (scripts/08 on a dated graph, raw and
               refined), on the windows after the first retrain

  python scripts/12_stats.py --tag mat [--level 0.2] [--noise 0.1]
  python scripts/12_stats.py --tag raw
Writes outputs/tables/stats_rewiring_<tag>[_waves].csv, stats_loop_<tag>_lvl<level>[_noise<noise>].csv
(the suffix matches the loop run's tables from scripts/11) and stats_selftrain_<tag>_<graph>.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from _common import ROOT
from scipy import stats

TAB = ROOT / "outputs" / "tables"


def paired_summary(pairs: pd.DataFrame, by: list[str], col: str = "diff") -> pd.DataFrame:
    """pairs: one row per (by..., seed, wave) with a `diff` column."""
    rows = []
    for key, sub in pairs.groupby(by):
        per_seed = sub.groupby("seed")[col].mean()
        n = len(per_seed); m = float(per_seed.mean())
        sd = float(per_seed.std(ddof=1)) if n > 1 else np.nan
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "n_seeds": n, "mean_diff": m,
                     "ci95_low": m - half, "ci95_high": m + half, "seed_sd": sd,
                     "share_pairs_positive": float((sub[col] > 0).mean()), "n_pairs": len(sub)})
    return pd.DataFrame(rows)


def rewiring(tag: str) -> pd.DataFrame | None:
    """Two layouts: the .mat scorer sweep under waves (results_<tag>_waves_<scorer>_softrank.csv vs
    results_<tag>_waves_raw_graph.csv, one row per scorer) and the text track on a dated graph
    (results_<tag>_refined_<mode>.csv vs results_<tag>_raw_graph.csv, one row per rewiring mode)."""
    if (TAB / f"results_{tag}_waves_raw_graph.csv").exists():
        base_p, files, label = TAB / f"results_{tag}_waves_raw_graph.csv", sorted(TAB.glob(f"results_{tag}_waves_*_softrank.csv")), "weights"
    else:
        base_p, files, label = TAB / f"results_{tag}_raw_graph.csv", sorted(TAB.glob(f"results_{tag}_refined_*.csv")), "graph"
    files = [f for f in files if not f.name.endswith("_summary.csv")]
    if not base_p.exists() or not files:
        return None
    base = pd.read_csv(base_p)
    base = base[base["fold"].str.startswith("test_w")][["model", "seed", "fold", "auc_pr"]]
    out = []
    for f in files:
        d = pd.read_csv(f)
        d = d[d["fold"].str.startswith("test_w")]
        scorer = str(d[label].iloc[0]).removeprefix(f"{tag}_")
        j = d.merge(base, on=["model", "seed", "fold"], suffixes=("", "_raw"))
        j["diff"] = j["auc_pr"] - j["auc_pr_raw"]; j["scorer"] = scorer
        out.append(j[["model", "scorer", "seed", "fold", "diff"]])
    return paired_summary(pd.concat(out), ["model", "scorer"])


def selftrain(tag: str, graph: str) -> pd.DataFrame | None:
    """Text track (scripts/08 on a dated graph): deployed - control and deployed - incumbent per
    variant on the windows after the first retrain (control_ap present)."""
    p = TAB / f"selftrain_windows_{tag}_{graph}.csv"
    if not p.exists():
        return None
    w = pd.read_csv(p)
    w = w[w["control_ap"].notna()]
    if w.empty:
        return None
    out = []
    for ref in ("control", "incumbent"):
        d = w[["variant", "seed", "window"]].copy()
        d["diff"] = w["deployed_ap"] - w[f"{ref}_ap"]; d["vs"] = ref
        out.append(d)
    return paired_summary(pd.concat(out), ["vs", "variant"])


def loop(tag: str, suffix: str) -> pd.DataFrame | None:
    p = TAB / f"graph_loop_windows_{tag}_{suffix}.csv"
    if not p.exists():
        return None
    w = pd.read_csv(p)
    w = w[(w["scenario"] == "attack") & w["control_ap"].notna()]      # waves after the first retrain
    out = []
    for ref in ("control", "incumbent"):
        d = w[["variant", "seed", "window"]].copy()
        d["diff"] = w["deployed_ap"] - w[f"{ref}_ap"]; d["vs"] = ref
        out.append(d)
    return paired_summary(pd.concat(out), ["vs", "variant"])


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tag", default="mat"); p.add_argument("--level", default="0.2")
    p.add_argument("--noise", default="0.1", help="feature noise of the loop run (scripts/11 --noise)")
    a = p.parse_args()
    r = rewiring(a.tag)
    if r is not None:
        suffix = "_waves" if (TAB / f"results_{a.tag}_waves_raw_graph.csv").exists() else ""
        r.to_csv(TAB / f"stats_rewiring_{a.tag}{suffix}.csv", index=False)
        print(r.round(4).to_string(index=False))
    for graph in ("raw_graph", "refined_soft"):
        st = selftrain(a.tag, graph)
        if st is not None:
            st.to_csv(TAB / f"stats_selftrain_{a.tag}_{graph}.csv", index=False)
            print(f"-- self-training on {graph}"); print(st.round(4).to_string(index=False))
    suffix = f"lvl{a.level}{'' if float(a.noise) == 0.1 else f'_noise{a.noise}'}"
    lp = loop(a.tag, suffix)
    if lp is not None:
        lp.to_csv(TAB / f"stats_loop_{a.tag}_{suffix}.csv", index=False)
        print(lp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
