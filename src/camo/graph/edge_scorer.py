"""Distilled edge-consistency scorer (decisions D4).

Pair features from local, label-free signals:  [ |e_u - e_v|, e_u * e_v, cos(e_u, e_v),
|f_u - f_v|, (f_u + f_v)/2, relation one-hot ]  where e = sentence embedding, f = LLM review fields.
Target: the LLM's pairwise relative_fit on the sampled pairs. Model: ridge regression (small,
regularised, applies to millions of edges in chunks). Ablations: cosine-only, fields-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

RELS = ["rur", "rsr", "rtr"]


def pair_features(u: np.ndarray, v: np.ndarray, rel: np.ndarray, emb: np.ndarray, fields: np.ndarray,
                  subset: str = "all") -> np.ndarray:
    eu, ev = emb[u], emb[v]
    cos = np.einsum("ij,ij->i", eu, ev)[:, None]
    onehot = np.stack([(rel == r).astype(np.float32) for r in RELS], 1)
    fu, fv = fields[u], fields[v]
    fdiff, fmean = np.abs(fu - fv), (fu + fv) / 2
    if subset == "cos":
        return np.hstack([cos, onehot, cos * onehot])
    if subset == "fields":
        return np.hstack([fdiff, fmean, onehot])
    return np.hstack([np.abs(eu - ev), eu * ev, cos, fdiff, fmean, onehot, cos * onehot])


class EdgeScorer:
    def __init__(self, subset: str = "all", alpha: float = 1.0):
        self.subset, self.alpha = subset, alpha
        self.sc, self.model = StandardScaler(), Ridge(alpha=alpha)

    def fit(self, u, v, rel, y, emb, fields):
        X = pair_features(u, v, rel, emb, fields, self.subset)
        self.model.fit(self.sc.fit_transform(X), y)
        return self

    def predict(self, u, v, rel, emb, fields, chunk: int = 200_000) -> np.ndarray:
        out = np.empty(len(u), dtype=np.float32)
        for i in range(0, len(u), chunk):
            X = pair_features(u[i:i + chunk], v[i:i + chunk], rel[i:i + chunk], emb, fields, self.subset)
            out[i:i + chunk] = self.model.predict(self.sc.transform(X))
        return np.clip(out, 0.0, 1.0)


def fit_and_evaluate(pairs: pd.DataFrame, emb: np.ndarray, fields: np.ndarray, seed: int,
                     alpha: float = 1.0, test_frac: float = 0.2) -> tuple[dict[str, EdgeScorer], pd.DataFrame]:
    """pairs: u_node, v_node, relation, relative_fit (swap==0, ok rows). Returns scorers per subset
    and a metrics table (R2, Spearman on held-out pairs, per subset and per relation)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs)); n_te = int(test_frac * len(pairs))
    te, tr = idx[:n_te], idx[n_te:]
    u, v = pairs["u_node"].to_numpy(), pairs["v_node"].to_numpy()
    rel, y = pairs["relation"].to_numpy(), pairs["relative_fit"].to_numpy(float)
    scorers, rows = {}, []
    for subset in ["all", "cos", "fields"]:
        s = EdgeScorer(subset, alpha).fit(u[tr], v[tr], rel[tr], y[tr], emb, fields)
        scorers[subset] = s
        pred = s.predict(u[te], v[te], rel[te], emb, fields)
        rows.append({"subset": subset, "relation": "all", "n_test": len(te), "r2": r2_score(y[te], pred),
                     "spearman": spearmanr(y[te], pred).statistic if len(te) > 2 else np.nan})
        for r in RELS:
            m = rel[te] == r
            if m.sum() > 2:
                rows.append({"subset": subset, "relation": r, "n_test": int(m.sum()), "r2": r2_score(y[te][m], pred[m]),
                             "spearman": spearmanr(y[te][m], pred[m]).statistic})
    return scorers, pd.DataFrame(rows)


def retest_agreement(cache_df: pd.DataFrame) -> dict:
    """Agreement between original and order-swapped LLM calls: the ceiling for any distilled scorer."""
    ok = cache_df[cache_df["status"] == "ok"]
    a = ok[ok["swap"] == 0].set_index(["relation", "u", "v"])["relative_fit"]
    b = ok[ok["swap"] == 1].set_index(["relation", "u", "v"])["relative_fit"]
    j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    if len(j) < 3:
        return {"n": len(j), "pearson": np.nan, "spearman": np.nan, "mae": np.nan}
    return {"n": len(j), "pearson": float(np.corrcoef(j["a"], j["b"])[0, 1]),
            "spearman": float(spearmanr(j["a"], j["b"]).statistic), "mae": float((j["a"] - j["b"]).abs().mean())}
