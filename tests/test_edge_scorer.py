import numpy as np
import pandas as pd

from camo.graph.edge_scorer import fit_and_evaluate, pair_features, retest_agreement


def test_pair_features_shapes():
    emb = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)
    fields = np.random.default_rng(1).random((10, 4)).astype(np.float32)
    u, v, rel = np.array([0, 1]), np.array([2, 3]), np.array(["rur", "rsr"])
    assert pair_features(u, v, rel, emb, fields).shape == (2, 8 + 8 + 1 + 4 + 4 + 3 + 3)
    assert pair_features(u, v, rel, emb, fields, "cos").shape == (2, 7)


def test_scorer_recovers_linear_target():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 8)).astype(np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    fields = rng.random((200, 3)).astype(np.float32)
    u, v = rng.integers(0, 200, 400), rng.integers(0, 200, 400)
    rel = rng.choice(["rur", "rsr", "rtr"], 400)
    cos = np.einsum("ij,ij->i", emb[u], emb[v])
    y = np.clip(0.5 + 0.4 * cos - 0.3 * np.abs(fields[u, 0] - fields[v, 0]), 0, 1)
    pairs = pd.DataFrame({"u_node": u, "v_node": v, "relation": rel, "relative_fit": y})
    scorers, m = fit_and_evaluate(pairs, emb, fields, seed=0)
    assert m[(m.subset == "all") & (m.relation == "all")]["r2"].iloc[0] > 0.8
    pred = scorers["all"].predict(u, v, rel, emb, fields)
    assert pred.min() >= 0 and pred.max() <= 1


def test_retest_agreement():
    df = pd.DataFrame({"relation": ["rur"] * 6, "u": [1, 2, 5, 1, 2, 5], "v": [3, 4, 6, 3, 4, 6],
                       "swap": [0, 0, 0, 1, 1, 1], "status": ["ok"] * 6,
                       "relative_fit": [0.2, 0.8, 0.5, 0.25, 0.75, 0.55]})
    r = retest_agreement(df)
    assert r["n"] == 3 and r["mae"] < 0.1 and r["spearman"] > 0.99
