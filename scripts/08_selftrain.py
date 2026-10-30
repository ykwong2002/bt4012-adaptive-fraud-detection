"""Label-scarce response + normal-class bias audit (professor's comment #2).

After the incumbent is trained on the labelled training windows, the later windows form an unlabelled
pool (plus a small labelled probe of `probe_labels_per_window` nodes, the only recent labels a
practitioner has). Pseudo-labels are fused from the incumbent score, Louvain community propensity on
R-U-R ∪ scorer-weighted R-S-R, and AE anomaly rank; a challenger is self-trained for several rounds
under each variant in configs/selftrain.yaml (naive / per-class thresholds / prior matching /
balanced loss / all). Every round is audited for collapse toward the normal class, and the challenger
is promoted only if it beats the incumbent on the probe. Brackets: keep-incumbent (lower) and
retrain-on-all-pool-labels (upper, oracle). Works on the temporal rebuilt graph and, flagged as
`stratified`, on the no-dates .mat graph.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from _common import ROOT, load_config, parser, processed_path, resolve_source
from camo.data.load_mat import load_graph
from camo.eval.protocol import build_folds
from camo.eval.runlog import RunLog
from camo.models.autoencoder import ae_anomaly_scores
from camo.models.catboost_model import make_table, train_catboost
from camo.selftrain.bias_audit import audit_round, collapse_flag
from camo.selftrain.challenger import fit_challenger, promote
from camo.selftrain.pseudo_labels import community_labels, community_propensity, em_prior, fuse, select_pseudo

TAB = ROOT / "outputs" / "tables"


def ap(y, s, idx):
    return float(average_precision_score(y[idx], s[idx])) if 0 < y[idx].sum() < len(idx) else np.nan


def main():
    p = parser(__doc__)
    p.add_argument("--weights", default=None); p.add_argument("--variants", default=None)
    p.add_argument("--seeds", default=None); p.add_argument("--tag", default=None)
    args = p.parse_args()
    cfg, scfg = load_config("base"), load_config("selftrain")
    source = resolve_source(args.source)
    g = load_graph(processed_path(source, args.smoke))
    weights = None
    if args.weights:
        z = np.load(args.weights); weights = {r: z[r] for r in g.edges}
    graph_name = "raw_graph" if weights is None else "refined_soft"
    tag = args.tag or f"{source}{'_smoke' if args.smoke else ''}_{graph_name}"
    log = RunLog(f"08_selftrain_{tag}", {**scfg, **vars(args)})
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (cfg["smoke"]["seeds"] if args.smoke else cfg["seeds"])
    variants = {k: v for k, v in scfg["variants"].items() if args.variants is None or k in args.variants.split(",")}
    iters = 50 if args.smoke else 300
    rows, brackets = [], []
    for seed in seeds:
        protocol, train_idx, folds = build_folds(g, cfg, seed, weights, "soft" if weights is not None else None)
        pool = np.concatenate([f.eval_idx for f in folds[1:]])
        rng = np.random.default_rng(seed)
        per_w = 20 if args.smoke else scfg["probe_labels_per_window"]
        probe = np.concatenate([rng.choice(f.eval_idx, min(per_w, len(f.eval_idx)), replace=False) for f in folds[1:]])
        eval_idx = np.setdiff1d(pool, probe); unl = np.setdiff1d(pool, probe)
        mu, sd = g.x[train_idx].mean(0), g.x[train_idx].std(0) + 1e-6
        x = ((g.x - mu) / sd).astype(np.float32)
        xt_tr = make_table(x, folds[0].edges, folds[0].weights); xt_pool = make_table(x, folds[-1].edges, folds[-1].weights)
        is_tr = np.isin(np.arange(g.n), train_idx)[:, None]
        xtab = np.where(is_tr, xt_tr, xt_pool)

        inc = train_catboost(xtab, g.y, train_idx, None, seed, iterations=iters)
        inc_score = inc.predict_proba(xtab)[:, 1]
        train_prior = float(g.y[train_idx].mean())
        # scores of a class-balanced model are calibrated to their own mean on train, not to the base rate
        prior_em = em_prior(inc_score[pool], float(inc_score[train_idx].mean()))
        prior_used = train_prior if scfg.get("prior_source", "train") == "train" else prior_em
        true_prior = float(g.y[pool].mean())
        anom = ae_anomaly_scores(x, train_idx, seed, hidden=scfg["pseudo"]["ae_hidden"], epochs=50 if args.smoke else 200)
        comm = community_labels(g.n, folds[-1].edges, folds[-1].weights, scfg["pseudo"]["community_relations"], seed,
                                resolution=scfg["pseudo"]["louvain_resolution"])
        upper = fit_challenger(xtab, np.concatenate([g.y[train_idx], g.y[unl]]), np.concatenate([train_idx, unl]), seed, True, iters)
        up_ap = ap(g.y, upper.predict_proba(xtab)[:, 1], eval_idx); low_ap = ap(g.y, inc_score, eval_idx)
        brackets.append({"seed": seed, "protocol": protocol, "graph": graph_name, "lower_keep_incumbent_ap": low_ap,
                         "upper_all_labels_ap": up_ap, "train_prior": train_prior, "em_prior": prior_em, "true_pool_prior": true_prior,
                         "n_pool": len(pool), "n_probe": len(probe), "n_communities": int(comm.max() + 1)})
        print(f"seed {seed} {protocol}: incumbent eval AP={low_ap:.4f} upper={up_ap:.4f} prior train={train_prior:.3f} em={prior_em:.3f} true={true_prior:.3f}")

        for vname, v in variants.items():
            score = inc_score.copy(); hist = []
            for rd in range(scfg["rounds"]):
                cp = community_propensity(score, comm, pool); f = fuse(score, cp, anom, pool)
                prior = prior_used if v["prior_match"] else None
                pidx, plab = select_pseudo(f, unl, scfg["confidence_quantile"], v["per_class_threshold"], prior)
                aud = audit_round(pidx, plab, comm, prior_used, probe, g.y, score, k_frac=train_prior)
                chal = fit_challenger(xtab, np.concatenate([g.y[train_idx], plab]), np.concatenate([train_idx, pidx]), seed, v["balanced_loss"], iters)
                score = chal.predict_proba(xtab)[:, 1]
                pr = promote(inc_score, score, probe, g.y)
                hist.append(aud)
                rows.append({"variant": vname, "seed": seed, "round": rd, "protocol": protocol, "graph": graph_name,
                             **aud, **pr, "chal_eval_ap": ap(g.y, score, eval_idx), "inc_eval_ap": low_ap, "upper_ap": up_ap})
            rows[-1]["collapse"] = collapse_flag(hist)
            chosen = rows[-1]["chal_eval_ap"] if rows[-1]["promoted"] else low_ap
            rows[-1]["recovery_gap"] = up_ap - chosen
            print(f"  {vname:13s} final: pseudo_pos={rows[-1]['pseudo_pos_share']:.3f} probe_rec={rows[-1]['probe_minority_recall']:.3f} "
                  f"chal_eval_ap={rows[-1]['chal_eval_ap']:.4f} promoted={rows[-1]['promoted']} collapse={rows[-1]['collapse']}")
    df = pd.DataFrame(rows); TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"selftrain_rounds_{tag}.csv", index=False)
    pd.DataFrame(brackets).to_csv(TAB / f"selftrain_brackets_{tag}.csv", index=False)
    last = df[df["round"] == df["round"].max()]
    summ = last.groupby("variant").agg(pseudo_pos_share=("pseudo_pos_share", "mean"), prior=("prior", "mean"),
                                       probe_minority_recall=("probe_minority_recall", "mean"), chal_eval_ap=("chal_eval_ap", "mean"),
                                       chal_eval_ap_std=("chal_eval_ap", "std"), inc_eval_ap=("inc_eval_ap", "mean"),
                                       upper_ap=("upper_ap", "mean"), promoted_rate=("promoted", "mean"),
                                       collapse_rate=("collapse", "mean"), recovery_gap=("recovery_gap", "mean")).reset_index()
    summ.to_csv(TAB / f"selftrain_summary_{tag}.csv", index=False)
    print(summ.round(4).to_string(index=False))
    log.note(n_rows=len(df)); log.finish()


if __name__ == "__main__":
    main()
