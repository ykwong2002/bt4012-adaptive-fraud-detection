"""Train one of the four models on a fold list and score every fold (shared by scripts 06 and 09b)."""
from __future__ import annotations

import numpy as np

from camo.models.autoencoder import ae_anomaly_scores
from camo.models.care_static import care_filter_all, label_aware_scores
from camo.models.catboost_model import make_table, train_catboost
from camo.models.wsage import RelGraph, predict_wsage, train_wsage


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
            fe = lambda f: care_filter_all(f.edges, s, rho)
            fw = lambda f: None
        else:
            fe = lambda f: f.edges
            fw = lambda f: f.weights
        gtr = RelGraph(g.n, fe(tr), fw(tr)); gva = RelGraph(g.n, fe(va), fw(va))
        model = train_wsage(x, y, gtr, train_idx, va.eval_idx, gva, seed, epochs=epochs,
                            k_eval=50 if not smoke else 20)
        out["val"] = predict_wsage(model, x, gva, seed)
        for f in tests:
            out[f.name] = predict_wsage(model, x, RelGraph(g.n, fe(f), fw(f)), seed)
    elif name == "catboost":
        xtr = make_table(x, tr.edges, tr.weights, tr.n_visible); xva = make_table(x, va.edges, va.weights, va.n_visible)
        xt = np.where(np.isin(np.arange(g.n), train_idx)[:, None], xtr, xva)  # train rows from train graph
        clf = train_catboost(xt, y, train_idx, va.eval_idx, seed, iterations=100 if smoke else 600)
        out["val"] = clf.predict_proba(xva)[:, 1]
        for f in tests:
            out[f.name] = clf.predict_proba(make_table(x, f.edges, f.weights, f.n_visible))[:, 1]
    elif name == "ae":
        s = ae_anomaly_scores(x, train_idx, seed, epochs=50 if smoke else 200)
        out["val"] = s
        for f in tests:
            out[f.name] = s
    else:
        raise ValueError(name)
    return out
