"""Edge-scorer diagnostics (decisions D15): for every edge-score file of a source, the same- vs
cross-label AUC on all edges (global), on edges touching a fraud node (fraud-centred: fraud-fraud vs
fraud-legit) and on edges touching a legit node (legit-centred), plus the mean per-relation rank weight
of fraud-fraud, fraud-legit and legit-legit edges. Labels are used for diagnosis only.

Global AUC is dominated by the ~77 % legit-legit edges, so a scorer can look good globally while
treating a fraud node's fraud and legit neighbours alike; the fraud-centred AUC is what decides whether
a fraud node's neighbourhood gets less camouflaged.

  python scripts/05c_edge_diagnostics.py --source mat
Writes outputs/tables/edge_auc_centred_<source>.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import ROOT, load_config, parser, processed_path, resolve_source
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from camo.data.load_mat import load_graph

TAB = ROOT / "outputs" / "tables"


def centred_rows(name: str, edges: dict, scores: dict, y: np.ndarray) -> list[dict]:
    rows = []
    for r, e in edges.items():
        s = scores[r]
        same = y[e[0]] == y[e[1]]
        fr = (y[e[0]] == 1) | (y[e[1]] == 1)
        lg = (y[e[0]] == 0) | (y[e[1]] == 0)
        ff = (y[e[0]] == 1) & (y[e[1]] == 1)

        def auc(m, same=same, s=s):
            return float(roc_auc_score(same[m], s[m])) if 0 < same[m].sum() < m.sum() else np.nan

        rank = rankdata(s) / len(s)
        rows.append({"scorer": name, "relation": r, "edges": int(e.shape[1]),
                     "auc_global": auc(np.ones(len(s), bool)), "auc_fraud_centred": auc(fr), "auc_legit_centred": auc(lg),
                     "rank_weight_ff": float(rank[ff].mean()), "rank_weight_fl": float(rank[fr & ~ff].mean()),
                     "rank_weight_ll": float(rank[~fr].mean())})
    return rows


def main():
    args = parser(__doc__).parse_args()
    cfg = load_config("base")
    source = resolve_source(args.source)
    tag = f"{source}{'_smoke' if args.smoke else ''}"
    g = load_graph(processed_path(source, args.smoke))
    proc = ROOT / cfg["paths"]["processed_dir"]
    rows = []
    # edge_scores_<tag>.npz is the distilled text scorer (scripts/05); edge_scores_<tag>_<name>.npz are
    # the graph-only scorers (scripts/05b)
    for f in sorted(proc.glob(f"edge_scores_{tag}.npz")) + sorted(proc.glob(f"edge_scores_{tag}_*.npz")):
        z = np.load(f)
        if not all(r in z.files and len(z[r]) == g.edges[r].shape[1] for r in g.edges):
            continue
        name = "text_distilled" if f.stem == f"edge_scores_{tag}" else f.stem.removeprefix(f"edge_scores_{tag}_")
        rows += centred_rows(name, g.edges, {r: z[r] for r in g.edges}, g.y)
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"edge_auc_centred_{tag}.csv", index=False)
    print(df[df["relation"] != "rur"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
