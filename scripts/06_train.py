"""Train and evaluate the four models (WeightedSAGE, CARE-static, CatBoost, AE) on the raw or
refined graph under the temporal protocol. Writes outputs/tables/results_<tag>.csv (+ _summary).
--waves: synthetic arrival waves instead of dates (the graph-only track on .mat, decisions D13)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from _common import ROOT, load_config, parser, processed_path, resolve_source

from camo.data.load_mat import load_graph
from camo.eval.metrics import evaluate, summarise
from camo.eval.protocol import build_folds
from camo.eval.runlog import RunLog
from camo.eval.waves import assign_waves, with_waves
from camo.models.runner import run_model, standardise

TAB = ROOT / "outputs" / "tables"


def load_weights(path, g):
    if path is None:
        return None
    z = np.load(path)
    return {r: z[r] for r in g.edges}


def main():
    p = parser(__doc__)
    p.add_argument("--weights", default=None, help="npz of edge scores per relation (from scripts/05)")
    p.add_argument("--mode", choices=["soft", "softrank", "hard"], default=None)
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--models", default="wsage,care,catboost,ae")
    p.add_argument("--rho", type=float, default=0.5, help="CARE-static keep ratio")
    p.add_argument("--tag", default=None)
    p.add_argument("--waves", action="store_true", help="random arrival waves (re-drawn per seed) instead of dates")
    p.add_argument("--seeds", default=None, help="comma list, overrides the config")
    args = p.parse_args()
    cfg = load_config("base")
    source = resolve_source(args.source)
    g = load_graph(processed_path(source, args.smoke))
    weights = load_weights(args.weights, g)
    graph_name = "raw_graph" if weights is None else f"refined_{args.mode}{f'_tau{args.tau}' if args.mode == 'hard' else ''}"
    weights_name = "none" if args.weights is None else Path(args.weights).stem.removeprefix("edge_scores_")
    tag = args.tag or f"{source}{'_smoke' if args.smoke else ''}{'_waves' if args.waves else ''}_{graph_name}"
    log = RunLog(f"06_train_{tag}", {**cfg, **vars(args)})
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = cfg["smoke"]["seeds"] if args.smoke else cfg["seeds"]
    if g.x.shape[1] == 0:
        raise SystemExit("graph has no node features; build with features first")

    rows = []
    for seed in seeds:
        gs, wv = g, None
        if args.waves:
            wv = assign_waves(g.y, cfg["windows"]["n_windows"], seed); gs = with_waves(g, wv)
        protocol, train_idx, folds = build_folds(gs, cfg, seed, weights, args.mode, args.tau, args.smoke, windows=wv)
        x = standardise(gs.x, train_idx)
        for name in args.models.split(","):
            scores = run_model(name, gs, x, train_idx, folds, seed, args.smoke, args.rho)
            for f in folds[1:]:
                m = evaluate(g.y[f.eval_idx], scores[f.name][f.eval_idx])
                rows.append({"model": name, "graph": graph_name, "weights": weights_name, "protocol": protocol, "seed": seed,
                             "fold": f.name, "cutoff": str(f.cutoff), **m})
                print(f"seed {seed} {name:8s} {f.name:9s} auc_pr={m['auc_pr']:.4f} auroc={m['auroc']:.4f} n={m['n']} pos={m['n_pos']}")
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"results_{tag}.csv", index=False)
    summ = summarise(rows, ["model", "graph", "weights", "protocol", "fold"])
    summ.to_csv(TAB / f"results_{tag}_summary.csv", index=False)
    print(summ.to_string(index=False))
    log.note(protocol=protocol, n_rows=len(df)); log.finish()


if __name__ == "__main__":
    main()
