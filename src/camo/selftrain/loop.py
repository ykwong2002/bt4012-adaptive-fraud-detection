"""The per-seed self-training loop used by scripts/08 and the recovery part of scripts/09."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from camo.eval.protocol import build_folds
from camo.models.autoencoder import ae_anomaly_scores
from camo.models.catboost_model import make_table, train_catboost
from camo.selftrain.bias_audit import audit_round, collapse_flag
from camo.selftrain.challenger import fit_challenger, promote
from camo.selftrain.pseudo_labels import (
    community_labels,
    community_propensity,
    em_prior,
    fuse,
    select_pseudo,
)


def ap(y, s, idx):
    return float(average_precision_score(y[idx], s[idx])) if 0 < y[idx].sum() < len(idx) else np.nan


def run_selftrain(g, cfg, scfg, seed, weights, variants, iters, probe_per_window, ae_epochs=200, graph_name="raw_graph"):
    protocol, train_idx, folds = build_folds(g, cfg, seed, weights, "soft" if weights is not None else None)
    pool = np.concatenate([f.eval_idx for f in folds[1:]])
    rng = np.random.default_rng(seed)
    probe = np.concatenate([rng.choice(f.eval_idx, min(probe_per_window, len(f.eval_idx)), replace=False) for f in folds[1:]])
    eval_idx = np.setdiff1d(pool, probe); unl = eval_idx
    mu, sd = g.x[train_idx].mean(0), g.x[train_idx].std(0) + 1e-6
    x = ((g.x - mu) / sd).astype(np.float32)
    xt_tr = make_table(x, folds[0].edges, folds[0].weights); xt_pool = make_table(x, folds[-1].edges, folds[-1].weights)
    xtab = np.where(np.isin(np.arange(g.n), train_idx)[:, None], xt_tr, xt_pool)

    inc = train_catboost(xtab, g.y, train_idx, None, seed, iterations=iters)
    inc_score = inc.predict_proba(xtab)[:, 1]
    train_prior = float(g.y[train_idx].mean())
    prior_em = em_prior(inc_score[pool], float(inc_score[train_idx].mean()))
    prior_used = train_prior if scfg.get("prior_source", "train") == "train" else prior_em
    anom = ae_anomaly_scores(x, train_idx, seed, hidden=scfg["pseudo"]["ae_hidden"], epochs=ae_epochs)
    comm = community_labels(g.n, folds[-1].edges, folds[-1].weights, scfg["pseudo"]["community_relations"], seed,
                            resolution=scfg["pseudo"]["louvain_resolution"])
    upper = fit_challenger(xtab, np.concatenate([g.y[train_idx], g.y[unl]]), np.concatenate([train_idx, unl]), seed, True, iters)
    up_ap, low_ap = ap(g.y, upper.predict_proba(xtab)[:, 1], eval_idx), ap(g.y, inc_score, eval_idx)
    bracket = {"seed": seed, "protocol": protocol, "graph": graph_name, "lower_keep_incumbent_ap": low_ap,
               "upper_all_labels_ap": up_ap, "train_prior": train_prior, "em_prior": prior_em,
               "true_pool_prior": float(g.y[pool].mean()), "n_pool": len(pool), "n_probe": len(probe),
               "n_communities": int(comm.max() + 1)}
    rows = []
    for vname, v in variants.items():
        score = inc_score.copy(); hist = []
        for rd in range(scfg["rounds"]):
            cp = community_propensity(score, comm, pool); f = fuse(score, cp, anom, pool)
            prior = prior_used if v["prior_match"] else None
            pidx, plab = select_pseudo(f, unl, scfg["confidence_quantile"], v["per_class_threshold"], prior)
            aud = audit_round(pidx, plab, comm, prior_used, probe, g.y, score, k_frac=train_prior)
            chal = fit_challenger(xtab, np.concatenate([g.y[train_idx], plab]), np.concatenate([train_idx, pidx]), seed, v["balanced_loss"], iters)
            score = chal.predict_proba(xtab)[:, 1]
            pr = promote(inc_score, score, probe, g.y); hist.append(aud)
            rows.append({"variant": vname, "seed": seed, "round": rd, "protocol": protocol, "graph": graph_name,
                         **aud, **pr, "chal_eval_ap": ap(g.y, score, eval_idx), "inc_eval_ap": low_ap, "upper_ap": up_ap})
        rows[-1]["collapse"] = collapse_flag(hist)
        chosen = rows[-1]["chal_eval_ap"] if rows[-1]["promoted"] else low_ap
        rows[-1]["recovery_gap"] = up_ap - chosen
    return rows, bracket
