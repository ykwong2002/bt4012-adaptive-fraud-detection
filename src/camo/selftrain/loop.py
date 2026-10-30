"""Self-training loops used by scripts/08 and the recovery part of scripts/09.

Two entry points:
  * run_triggered (temporal graphs, the reported setting, review issue #9): walk the post-training
    windows in time order. The deployed model scores window t on the graph masked at the end of t. When
    the label-free monitor raises an alarm at t, a challenger is self-trained using only what exists at
    the end of t (graph, communities and pseudo-labels from windows <= t) and promoted on the probe
    labels gathered so far; it is first used on window t+1. This yields detection-to-recovery numbers.
  * run_selftrain (no dates, the .mat graph only): the static ablation, reported as `stratified`.

Fairness of the comparison (review issue #8): the incumbent, the challengers and the no-pseudo-label
control all use the same learner and settings (selftrain.challenger.fit_challenger). The control is
retrained at the same moment on the training labels only, so `pseudo_label_gain` isolates what the
pseudo-labels add. The EM prior estimate accounts for the balanced class weights (issue #7).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from camo.eval.protocol import _mask, apply_rewiring, build_folds
from camo.eval.splits import equal_count_windows, temporal_plan
from camo.models.autoencoder import ae_anomaly_scores
from camo.models.catboost_model import make_table
from camo.selftrain.bias_audit import audit_round, collapse_flag
from camo.selftrain.challenger import fit_challenger, promote
from camo.selftrain.pseudo_labels import (
    community_labels,
    community_propensity,
    em_prior,
    fuse,
    select_pseudo,
)

# fit_challenger(balanced=True) reweights the classes to 50/50, so its posteriors are calibrated to a
# prior of 0.5, not to the training base rate. EM must start from the prior the scores are calibrated to.
BALANCED_MODEL_PRIOR = 0.5


def ap(y, s, idx):
    return float(average_precision_score(y[idx], s[idx])) if len(idx) and 0 < y[idx].sum() < len(idx) else np.nan


def self_train_variant(xtab, y, train_idx, pool, unl, probe, comm, anom, start_score, prior_used, v, scfg, seed, iters):
    """Rounds of pseudo-labelling + retraining for one variant. Returns (model, score, audits)."""
    score, hist, model = start_score.copy(), [], None
    train_prior = float(y[train_idx].mean())
    for _ in range(scfg["rounds"]):
        cp = community_propensity(score, comm, pool); f = fuse(score, cp, anom, pool)
        pidx, plab = select_pseudo(f, unl, scfg["confidence_quantile"], v["per_class_threshold"],
                                   prior_used if v["prior_match"] else None)
        hist.append(audit_round(pidx, plab, comm, prior_used, probe, y, score, k_frac=train_prior))
        model = fit_challenger(xtab, np.concatenate([y[train_idx], plab]), np.concatenate([train_idx, pidx]),
                               seed, v["balanced_loss"], iters)
        score = model.predict_proba(xtab)[:, 1]
    return model, score, hist


def run_selftrain(g, cfg, scfg, seed, weights, variants, iters, probe_per_window, ae_epochs=200, graph_name="raw_graph"):
    """Static ablation for the no-dates .mat graph (stratified split). Temporal graphs must use
    run_triggered: pooling every later window at once would let the future leak into the past."""
    if g.dates is not None:
        raise ValueError("graph has dates: use run_triggered (time-respecting) instead of run_selftrain")
    protocol, train_idx, folds = build_folds(g, cfg, seed, weights, "soft" if weights is not None else None)
    pool = np.concatenate([f.eval_idx for f in folds[1:]])
    rng = np.random.default_rng(seed)
    probe = np.concatenate([rng.choice(f.eval_idx, min(probe_per_window, len(f.eval_idx)), replace=False) for f in folds[1:]])
    unl = np.setdiff1d(pool, probe); eval_idx = unl
    mu, sd = g.x[train_idx].mean(0), g.x[train_idx].std(0) + 1e-6
    x = ((g.x - mu) / sd).astype(np.float32)
    xtab = make_table(x, folds[0].edges, folds[0].weights)

    inc = fit_challenger(xtab, g.y[train_idx], train_idx, seed, True, iters)
    inc_score = inc.predict_proba(xtab)[:, 1]
    train_prior = float(g.y[train_idx].mean())
    prior_em = em_prior(inc_score[pool], BALANCED_MODEL_PRIOR)
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
        control = inc if v["balanced_loss"] else fit_challenger(xtab, g.y[train_idx], train_idx, seed, False, iters)
        control_ap = ap(g.y, control.predict_proba(xtab)[:, 1], eval_idx)
        score, hist = inc_score.copy(), []
        for rd in range(scfg["rounds"]):
            _, score, h = self_train_variant(xtab, g.y, train_idx, pool, unl, probe, comm, anom, score, prior_used,
                                             v, {**scfg, "rounds": 1}, seed, iters)
            hist += h
            pr = promote(inc_score, score, probe, g.y)
            rows.append({"variant": vname, "seed": seed, "round": rd, "protocol": protocol, "graph": graph_name,
                         **h[0], **pr, "chal_eval_ap": ap(g.y, score, eval_idx), "inc_eval_ap": low_ap,
                         "control_eval_ap": control_ap, "upper_ap": up_ap})
        rows[-1]["collapse"] = collapse_flag(hist)
        rows[-1]["pseudo_label_gain"] = rows[-1]["chal_eval_ap"] - control_ap
        chosen = rows[-1]["chal_eval_ap"] if rows[-1]["promoted"] else low_ap
        rows[-1]["recovery_gap"] = up_ap - chosen
    return rows, bracket


def run_triggered(g, cfg, scfg, seed, weights, variants, iters, probe_per_window, alarms: dict[int, bool],
                  windows: np.ndarray | None = None, ae_epochs=200, graph_name="raw_graph"):
    """Time-respecting, alarm-triggered self-training. `alarms`: window -> alarm from the label-free
    monitor. Returns (per-window rows, per-round audit rows, per-variant summaries)."""
    if g.dates is None:
        raise ValueError("run_triggered needs dated nodes")
    w = windows if windows is not None else equal_count_windows(g.dates, cfg["windows"]["n_windows"])
    plan = temporal_plan(w, cfg["windows"]["n_train_windows"], cfg["windows"]["n_val_windows"])
    post = plan["val"] + plan["test"]
    train_idx = np.where(np.isin(w, plan["train"]))[0]
    y = g.y
    mu, sd = g.x[train_idx].mean(0), g.x[train_idx].std(0) + 1e-6
    x = ((g.x - mu) / sd).astype(np.float32)
    edges, ew = apply_rewiring(g.edges, weights, "soft" if weights is not None else None)

    def table_at(cut):
        e, ww = _mask(edges, ew, g.dates, cut)
        return make_table(x, e, ww), e, ww

    rng = np.random.default_rng(seed)
    probe_of = {k: rng.choice(np.where(w == k)[0], min(probe_per_window, int((w == k).sum())), replace=False) for k in post}
    xt_tr, _, _ = table_at(g.dates[train_idx].max())
    inc = fit_challenger(xt_tr, y[train_idx], train_idx, seed, True, iters)
    train_prior = float(y[train_idx].mean())
    anom = ae_anomaly_scores(x, train_idx, seed, hidden=scfg["pseudo"]["ae_hidden"], epochs=ae_epochs)
    tables = {k: table_at(g.dates[w == k].max()) for k in post}

    win_rows, round_rows, summaries = [], [], []
    for vname, v in variants.items():
        deployed, deployed_tag, shadow_control, shadow_upper, any_collapse = inc, "incumbent", None, None, False
        retrains = []
        for i, t in enumerate(post):
            xt_t, e_t, ww_t = tables[t]
            eval_t = np.setdiff1d(np.where(w == t)[0], probe_of[t])
            s_dep = deployed.predict_proba(xt_t)[:, 1]
            row = {"variant": vname, "seed": seed, "graph": graph_name, "window": int(t), "alarm": bool(alarms.get(int(t), False)),
                   "deployed": deployed_tag,
                   "deployed_ap": ap(y, s_dep, eval_t), "incumbent_ap": ap(y, inc.predict_proba(xt_t)[:, 1], eval_t),
                   "control_ap": ap(y, shadow_control.predict_proba(xt_t)[:, 1], eval_t) if shadow_control else np.nan,
                   "upper_ap": ap(y, shadow_upper.predict_proba(xt_t)[:, 1], eval_t) if shadow_upper else np.nan,
                   "retrained": False, "promoted": False, "n_eval": len(eval_t), "n_pos": int(y[eval_t].sum())}
            if row["alarm"] and i < len(post) - 1:      # retrain at the end of t, used from t+1
                done = post[: i + 1]
                pool = np.where(np.isin(w, done))[0]
                probe = np.concatenate([probe_of[k] for k in done])
                unl = np.setdiff1d(pool, probe)
                comm = community_labels(g.n, e_t, ww_t, scfg["pseudo"]["community_relations"], seed,
                                        resolution=scfg["pseudo"]["louvain_resolution"])
                prior_em = em_prior(s_dep[pool], BALANCED_MODEL_PRIOR)
                prior_used = train_prior if scfg.get("prior_source", "train") == "train" else prior_em
                chal, s_chal, hist = self_train_variant(xt_t, y, train_idx, pool, unl, probe, comm, anom, s_dep,
                                                        prior_used, v, scfg, seed, iters)
                shadow_control = fit_challenger(xt_t, y[train_idx], train_idx, seed, v["balanced_loss"], iters)
                shadow_upper = fit_challenger(xt_t, np.concatenate([y[train_idx], y[unl]]), np.concatenate([train_idx, unl]),
                                              seed, True, iters)
                pr = promote(s_dep, s_chal, probe, y)
                collapse = collapse_flag(hist); any_collapse |= collapse
                for rd, a in enumerate(hist):
                    round_rows.append({"variant": vname, "seed": seed, "graph": graph_name, "window": int(t), "round": rd, **a})
                row.update({"retrained": True, "promoted": pr["promoted"], "deployed_probe_ap": pr["inc_probe_ap"],
                            "chal_probe_ap": pr["chal_probe_ap"], "em_prior": prior_em, "true_pool_prior": float(y[pool].mean()),
                            "pseudo_pos_share": hist[-1]["pseudo_pos_share"], "collapse": collapse,
                            "n_pool": len(pool), "n_probe": len(probe)})
                retrains.append(int(t))
                if pr["promoted"]:
                    deployed, deployed_tag = chal, f"challenger@w{t}"
            win_rows.append(row)

        vr = [r for r in win_rows if r["variant"] == vname]
        after = [r for r in vr if retrains and r["window"] > retrains[0]]
        promoted_after = [r for r in after if r["deployed"] != "incumbent"]
        mean = lambda rs, k: float(np.nanmean([r[k] for r in rs])) if rs else np.nan
        summaries.append({"variant": vname, "seed": seed, "graph": graph_name,
                          "first_alarm": next((r["window"] for r in vr if r["alarm"]), None),
                          "n_retrains": len(retrains), "n_promoted": int(sum(r["promoted"] for r in vr)),
                          "deployed_ap_after": mean(after, "deployed_ap"), "incumbent_ap_after": mean(after, "incumbent_ap"),
                          "control_ap_after": mean(after, "control_ap"), "upper_ap_after": mean(after, "upper_ap"),
                          "recovery_gap": mean(after, "upper_ap") - mean(after, "deployed_ap") if after else np.nan,
                          "pseudo_label_gain": mean(promoted_after, "deployed_ap") - mean(promoted_after, "control_ap")
                          if promoted_after else np.nan,
                          "collapse_any": any_collapse})
    return win_rows, round_rows, summaries
