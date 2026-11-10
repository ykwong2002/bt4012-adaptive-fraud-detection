"""Graph-only edge scores (no text): label-free feature-consistency scorer, cosine baseline, or the
label oracle and its noisy version with a chosen edge AUC (diagnostics only). Writes
data/processed/edge_scores_<tag>_<scorer>.npz for scripts/06 (--weights), the tau sweep and the
weighted homophily table."""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import ROOT, load_config, parser, processed_path, resolve_source

from camo.data.load_mat import load_graph
from camo.eval.runlog import RunLog
from camo.eval.splits import equal_count_windows, temporal_plan
from camo.graph.feature_scorer import FeatureEdgeScorer, cosine_scores
from camo.graph.homophily import homophily_table
from camo.graph.rewire import edge_label_auc, noisy_oracle_scores, oracle_scores, tau_sweep_table

TAB = ROOT / "outputs" / "tables"


def fit_edges(g, cfg):
    """Edges the scorer may learn from: all of them on .mat, only training-period ones when dated."""
    if g.dates is None:
        return g.edges
    w = equal_count_windows(g.dates, cfg["windows"]["n_windows"])
    plan = temporal_plan(w, cfg["windows"]["n_train_windows"], cfg["windows"]["n_val_windows"])
    ok = np.isin(w, plan["train"])
    return {r: e[:, ok[e[0]] & ok[e[1]]] for r, e in g.edges.items()}


def main():
    p = parser(__doc__)
    p.add_argument("--scorer", choices=["feature", "cosine", "oracle", "noisy_oracle"], default="feature")
    p.add_argument("--edge-auc", type=float, default=0.7, help="target same- vs cross-label AUC for noisy_oracle")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg, rcfg = load_config("base"), load_config("reconstruction")
    source = resolve_source(args.source)
    g = load_graph(processed_path(source, args.smoke))
    name = f"noisy{args.edge_auc}" if args.scorer == "noisy_oracle" else args.scorer
    tag = f"{source}{'_smoke' if args.smoke else ''}_{name}"
    log = RunLog(f"05b_feature_edges_{tag}", {**rcfg, **vars(args)}, seed=args.seed)
    TAB.mkdir(parents=True, exist_ok=True)

    if args.scorer == "oracle":
        scores = oracle_scores(g.edges, g.y)
    elif args.scorer == "noisy_oracle":
        scores = noisy_oracle_scores(g.edges, g.y, args.edge_auc, args.seed)
    elif args.scorer == "cosine":
        scores = cosine_scores(g.x, g.edges)
    else:
        fcfg = dict(rcfg["feature_scorer"])
        if args.smoke:
            fcfg.update(steps=300, batch=512)
        scorer = FeatureEdgeScorer(seed=args.seed, **fcfg).fit(g.x, fit_edges(g, cfg))
        pd.DataFrame(scorer.history).to_csv(TAB / f"feature_scorer_history_{tag}.csv", index=False)
        print(f"link-prediction val AUC (label-free model selection): {scorer.val_auc:.4f}")
        log.note(val_auc=scorer.val_auc, steps_run=scorer.history[-1]["step"] if scorer.history else 0)
        scores = scorer.score(g.x, g.edges)

    out = ROOT / cfg["paths"]["processed_dir"] / f"edge_scores_{tag}.npz"
    np.savez_compressed(out, **scores)

    # Labels below are for diagnostics only; they never feed back into the scores.
    sweep = tau_sweep_table(g.edges, scores, g.y, rcfg["rewire"]["tau_grid"])
    sweep.to_csv(TAB / f"tau_sweep_{tag}.csv", index=False)
    homo = homophily_table(g.edges, g.y, g.n, scores)
    homo.insert(0, "scorer", name)
    auc = edge_label_auc(g.edges, scores, g.y)
    homo["edge_auc_same_vs_cross"] = homo["relation"].map(auc)
    homo.to_csv(TAB / f"homophily_weighted_{tag}.csv", index=False)
    print(homo.round(3).to_string(index=False))
    soft = sweep[sweep["tau"] == "soft"]
    for row in soft.itertuples():
        ok = row.mean_weight_hetero < row.mean_weight_homo
        print(f"direction check {row.relation}: cross-label {row.mean_weight_hetero:.3f} vs same-label "
              f"{row.mean_weight_homo:.3f} -> {'OK' if ok else 'WRONG DIRECTION'}")
    log.note(edge_auc=auc, direction_ok={r.relation: bool(r.mean_weight_hetero < r.mean_weight_homo) for r in soft.itertuples()},
             out=str(out))
    log.finish()


if __name__ == "__main__":
    main()
