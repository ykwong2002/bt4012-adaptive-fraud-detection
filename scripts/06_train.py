"""Train and evaluate the four models (WeightedSAGE, CARE-static, CatBoost, AE) on the raw or
refined graph under the temporal protocol. Writes outputs/tables/results_<tag>.csv (+ _summary)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ROOT, load_config, parser, processed_path, resolve_source
from camo.data.load_mat import load_graph
from camo.eval.metrics import evaluate, summarise
from camo.eval.protocol import build_folds
from camo.eval.runlog import RunLog
from camo.models.autoencoder import ae_anomaly_scores
from camo.models.care_static import care_filter_all, label_aware_scores
from camo.models.catboost_model import make_table, train_catboost
from camo.models.wsage import RelGraph, predict_wsage, train_wsage

TAB = ROOT / "outputs" / "tables"


def load_weights(path, g):
    if path is None:
        return None
    z = np.load(path)
    return {r: z[r] for r in g.edges}


def standardise(x, train_idx):
    mu, sd = x[train_idx].mean(0), x[train_idx].std(0) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def run_model(name, g, x, train_idx, folds, seed, smoke, rho):
    """Returns {fold_name: scores over all nodes}."""
    y = g.y
    tr, va, tests = folds[0], folds[1], folds[2:]
    epochs = 30 if smoke else 150
    out = {}
    if name in ("wsage", "care"):
        if name == "care":
            s = label_aware_scores(x, y, train_idx, seed)
            fe = lambda f: care_filter_all(f.edges, s, rho)   # noqa: E731
            fw = lambda f: None                                # noqa: E731  (filter is unweighted)
        else:
            fe = lambda f: f.edges   # noqa: E731
            fw = lambda f: f.weights  # noqa: E731
        gtr = RelGraph(g.n, fe(tr), fw(tr)); gva = RelGraph(g.n, fe(va), fw(va))
        model = train_wsage(x, y, gtr, train_idx, va.eval_idx, gva, seed, epochs=epochs,
                            k_eval=50 if not smoke else 20)
        out["val"] = predict_wsage(model, x, gva, seed)
        for f in tests:
            out[f.name] = predict_wsage(model, x, RelGraph(g.n, fe(f), fw(f)), seed)
    elif name == "catboost":
        xtr = make_table(x, tr.edges, tr.weights); xva = make_table(x, va.edges, va.weights)
        xt = np.where(np.isin(np.arange(g.n), train_idx)[:, None], xtr, xva)  # train rows from train graph
        clf = train_catboost(xt, y, train_idx, va.eval_idx, seed, iterations=100 if smoke else 600)
        out["val"] = clf.predict_proba(xva)[:, 1]
        for f in tests:
            out[f.name] = clf.predict_proba(make_table(x, f.edges, f.weights))[:, 1]
    elif name == "ae":
        s = ae_anomaly_scores(x, train_idx, seed, epochs=50 if smoke else 200)
        out["val"] = s
        for f in tests:
            out[f.name] = s
    return out


def main():
    p = parser(__doc__)
    p.add_argument("--weights", default=None, help="npz of edge scores per relation (from scripts/05)")
    p.add_argument("--mode", choices=["soft", "softrank", "hard"], default=None)
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--models", default="wsage,care,catboost,ae")
    p.add_argument("--rho", type=float, default=0.5, help="CARE-static keep ratio")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    cfg = load_config("base")
    source = resolve_source(args.source)
    g = load_graph(processed_path(source, args.smoke))
    weights = load_weights(args.weights, g)
    graph_name = "raw_graph" if weights is None else f"refined_{args.mode}{f'_tau{args.tau}' if args.mode == 'hard' else ''}"
    tag = args.tag or f"{source}{'_smoke' if args.smoke else ''}_{graph_name}"
    log = RunLog(f"06_train_{tag}", {**cfg, **vars(args)})
    seeds = cfg["smoke"]["seeds"] if args.smoke else cfg["seeds"]
    if g.x.shape[1] == 0:
        raise SystemExit("graph has no node features; build with features first")

    rows = []
    for seed in seeds:
        protocol, train_idx, folds = build_folds(g, cfg, seed, weights, args.mode, args.tau, args.smoke)
        x = standardise(g.x, train_idx)
        for name in args.models.split(","):
            scores = run_model(name, g, x, train_idx, folds, seed, args.smoke, args.rho)
            for f in folds[1:]:
                m = evaluate(g.y[f.eval_idx], scores[f.name][f.eval_idx])
                rows.append({"model": name, "graph": graph_name, "protocol": protocol, "seed": seed,
                             "fold": f.name, "cutoff": str(f.cutoff), **m})
                print(f"seed {seed} {name:8s} {f.name:9s} auc_pr={m['auc_pr']:.4f} auroc={m['auroc']:.4f} n={m['n']} pos={m['n_pos']}")
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"results_{tag}.csv", index=False)
    summ = summarise(rows, ["model", "graph", "protocol", "fold"])
    summ.to_csv(TAB / f"results_{tag}_summary.csv", index=False)
    print(summ.to_string(index=False))
    log.note(protocol=protocol, n_rows=len(df)); log.finish()


if __name__ == "__main__":
    main()
