"""EDA: class balance, degree and homophily per relation, rebuilt-vs-published table, reviews and
fraud per temporal window, PIT feature distributions by class. Writes outputs/tables/eda_*.csv and
outputs/figures/eda_*.png."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ROOT, load_config, parser, processed_path, resolve_source
from camo.data.load_mat import load_graph
from camo.eval.runlog import RunLog
from camo.eval.splits import equal_count_windows, window_bounds
from camo.graph.homophily import homophily_table, node_homophily

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"


def main():
    args = parser(__doc__).parse_args()
    cfg = load_config("base")
    source = resolve_source(args.source)
    tag = f"{source}{'_smoke' if args.smoke else ''}"
    log = RunLog(f"03_eda_{tag}", cfg)
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    g = load_graph(processed_path(source, args.smoke))
    n = g.n

    # 1. class balance + degrees + homophily per relation
    ht = homophily_table(g.edges, g.y, n)
    ht.insert(0, "graph", tag)
    ht.to_csv(TAB / f"eda_homophily_{tag}.csv", index=False)
    print(ht.round(4).to_string(index=False))
    summary = {"graph": tag, "n_nodes": n, "n_fraud": int(g.y.sum()), "fraud_rate": float(g.y.mean()),
               "n_features": g.x.shape[1], **{f"edges_{r}": m for r, m in g.edge_count().items()}}
    pd.DataFrame([summary]).to_csv(TAB / f"eda_summary_{tag}.csv", index=False)

    # 2. node-homophily distributions by class (the camouflage diagnostic)
    fig, axes = plt.subplots(1, len(g.edges), figsize=(4 * len(g.edges), 3.2), sharey=True)
    for ax, (r, e) in zip(np.atleast_1d(axes), g.edges.items()):
        h = node_homophily(e, g.y, n)
        for c, name in [(0, "legit"), (1, "fraud")]:
            vals = h[(g.y == c) & ~np.isnan(h)]
            ax.hist(vals, bins=20, range=(0, 1), density=True, alpha=0.6, label=f"{name} (n={vals.size})")
        ax.set_title(f"{r.upper()}: node homophily"); ax.set_xlabel("fraction of same-label neighbours")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"eda_homophily_{tag}.png", dpi=150); plt.close(fig)

    # 3. degree distributions (log-log CCDF) per relation by class
    fig, axes = plt.subplots(1, len(g.edges), figsize=(4 * len(g.edges), 3.2))
    for ax, (r, e) in zip(np.atleast_1d(axes), g.edges.items()):
        deg = np.bincount(e[0], minlength=n) + np.bincount(e[1], minlength=n)
        for c, name in [(0, "legit"), (1, "fraud")]:
            d = np.sort(deg[g.y == c])
            ccdf = 1 - np.arange(d.size) / d.size
            ax.loglog(d + 1, ccdf, label=name)
        ax.set_title(f"{r.upper()} degree CCDF"); ax.set_xlabel("degree + 1"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"eda_degree_{tag}.png", dpi=150); plt.close(fig)

    # 4. temporal: reviews and fraud per equal-count window (rebuilt graph only)
    if g.dates is not None:
        w = equal_count_windows(g.dates, cfg["windows"]["n_windows"])
        wb = pd.DataFrame(window_bounds(g.dates, w), columns=["window", "start", "end", "n"])
        wb["n_fraud"] = [int(g.y[w == k].sum()) for k in wb["window"]]
        wb["fraud_rate"] = wb["n_fraud"] / wb["n"]
        wb.to_csv(TAB / f"eda_windows_{tag}.csv", index=False)
        print(wb.to_string(index=False))
        months = pd.to_datetime(g.dates).to_period("M")
        vol = pd.DataFrame({"m": months, "y": g.y}).groupby("m")["y"].agg(["size", "sum"])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(vol.index.to_timestamp(), vol["size"], width=25, label="reviews")
        ax.bar(vol.index.to_timestamp(), vol["sum"], width=25, label="fraud")
        for _, row in wb.iterrows():
            ax.axvline(pd.Timestamp(row["start"]), color="k", lw=0.5, ls="--")
        ax.set_title("Reviews per month with equal-count window boundaries"); ax.legend()
        fig.tight_layout(); fig.savefig(FIG / f"eda_volume_{tag}.png", dpi=150); plt.close(fig)
        log.note(windows=wb.to_dict("records"))

    # 5. feature distributions by class
    names = g.meta.get("feature_names") or [f"f{j}" for j in range(g.x.shape[1])]
    if g.node_table is not None and g.x.shape[1] > 0:
        from camo.data.features_pit import FEATURES
        names = FEATURES if g.x.shape[1] == len(FEATURES) else names
    if g.x.shape[1] > 0:
        rows = []
        for j, nm in enumerate(names):
            a, b = g.x[g.y == 0, j], g.x[g.y == 1, j]
            rows.append({"feature": nm, "mean_legit": a.mean(), "mean_fraud": b.mean(),
                         "std_legit": a.std(), "std_fraud": b.std(),
                         "cohen_d": (b.mean() - a.mean()) / (np.sqrt((a.var() + b.var()) / 2) + 1e-9)})
        fd = pd.DataFrame(rows).sort_values("cohen_d", key=np.abs, ascending=False)
        fd.to_csv(TAB / f"eda_features_{tag}.csv", index=False)
        print(fd.head(12).round(3).to_string(index=False))

    log.note(**summary); log.finish()
    print("wrote", TAB, FIG)


if __name__ == "__main__":
    main()
