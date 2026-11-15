"""Graph-only adversarial experiment (decisions D13): works on any graph, including the public .mat.

Per seed, nodes get random arrival waves (camo.eval.waves). For every injection level, camouflaged
fraud nodes are appended to each test wave (camo.adversarial.graph_inject): template fraud features,
linked to a hijacked legitimate user's reviews (R-U-R) and to legitimate slices of product blocks
(R-S-R, R-T-R). Training waves are identical at every level, so a change in test performance is caused
by the injection alone. Reported per wave: AUC-PR on all nodes, AUC-PR on clean nodes only, and recall
of the injected nodes in the top 10 %.

  uv run python scripts/09b_graph_adversarial.py --source mat [--levels 0,0.05,0.1,0.2] [--models wsage,care,catboost]

Edge-weighted (rewired) graphs plug in through --scorer once a label-free scorer that can score the
injected edges is on main (feat/edge-scorer); until then only the raw graph and the comparators run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import ROOT, load_config, parser, processed_path, resolve_source

from camo.adversarial.graph_inject import inject_camouflage
from camo.data.load_mat import load_graph
from camo.eval.metrics import evaluate
from camo.eval.protocol import build_folds
from camo.eval.runlog import RunLog
from camo.eval.splits import temporal_plan
from camo.eval.waves import assign_waves
from camo.models.runner import run_model, standardise

TAB = ROOT / "outputs" / "tables"


def main():
    p = parser(__doc__)
    p.add_argument("--levels", default="0,0.05,0.1,0.2")
    p.add_argument("--models", default="wsage,care,catboost")
    p.add_argument("--rho", type=float, default=0.5, help="CARE-static keep ratio")
    p.add_argument("--noise", type=float, default=0.1, help="feature noise, in training standard deviations")
    p.add_argument("--seeds", default=None, help="comma list; default: first 3 config seeds (1 for --smoke)")
    args = p.parse_args()
    cfg = load_config("base")
    source = resolve_source(args.source)
    tag = f"{source}{'_smoke' if args.smoke else ''}"
    g0 = load_graph(processed_path(source, args.smoke))
    if g0.x.shape[1] == 0:
        raise SystemExit("graph has no node features; build with features first")
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else cfg["smoke"]["seeds"] if args.smoke else cfg["seeds"][:3])
    levels = [float(v) for v in args.levels.split(",")]
    log = RunLog(f"09b_graph_adversarial_{tag}", {**cfg, **vars(args)})
    nw = cfg["windows"]["n_windows"]
    rows = []
    for seed in seeds:
        w0 = assign_waves(g0.y, nw, seed)
        plan = temporal_plan(w0, cfg["windows"]["n_train_windows"], cfg["windows"]["n_val_windows"])
        train_ref = None
        for level in levels:
            g, w, inj = inject_camouflage(g0, w0, plan["train"], plan["test"], level, seed, args.noise)
            protocol, train_idx, folds = build_folds(g, cfg, seed, windows=w)
            # the injection must leave training untouched: same nodes, same visible edges
            sig = (tuple(train_idx), tuple(int(e.shape[1]) for e in folds[0].edges.values()))
            train_ref = train_ref or sig
            assert sig == train_ref, "training set or training graph changed under injection"
            x = standardise(g.x, train_idx)
            for name in args.models.split(","):
                scores = run_model(name, g, x, train_idx, folds, seed, args.smoke, args.rho)
                for f in folds[2:]:
                    s, idx = scores[f.name], f.eval_idx
                    clean = idx[~inj[idx]]
                    m_all, m_clean = evaluate(g.y[idx], s[idx]), evaluate(g.y[clean], s[clean])
                    ninj = int(inj[idx].sum())
                    top = idx[np.argsort(-s[idx])[: max(1, len(idx) // 10)]]
                    rows.append({"level": level, "seed": seed, "model": name, "protocol": protocol,
                                 "window": int(f.name.split("_w")[1]), "n": len(idx), "n_injected": ninj,
                                 "auc_pr": m_all["auc_pr"], "auroc": m_all["auroc"],
                                 "auc_pr_clean": m_clean["auc_pr"],
                                 "injected_recall@10%": float(inj[top].sum() / ninj) if ninj else np.nan})
                done = [r for r in rows if r["level"] == level and r["seed"] == seed and r["model"] == name]
                print(f"seed {seed} level {level:<5} {name:8s} injected={int(inj.sum()):5d} "
                      f"auc_pr={np.nanmean([r['auc_pr'] for r in done]):.4f} "
                      f"inj_recall={np.nanmean([r['injected_recall@10%'] for r in done]):.4f}")
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"graph_adversarial_{tag}.csv", index=False)
    summ = (df.groupby(["level", "model"])
              # AUC-PR on all nodes rises with the injected fraud's share of the wave, so it is not
              # comparable across levels; compare auc_pr_clean, auroc and injected_recall instead
              .agg(auc_pr=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"), auroc=("auroc", "mean"),
                   auc_pr_clean=("auc_pr_clean", "mean"),
                   injected_recall=("injected_recall@10%", "mean"), n_seeds=("seed", "nunique"))
              .reset_index())
    summ.to_csv(TAB / f"graph_adversarial_summary_{tag}.csv", index=False)
    print(summ.round(4).to_string(index=False))
    log.note(n_rows=len(df)); log.finish()


if __name__ == "__main__":
    main()
