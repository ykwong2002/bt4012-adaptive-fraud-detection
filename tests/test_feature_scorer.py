import numpy as np

from camo.graph.feature_scorer import FeatureEdgeScorer, cosine_scores
from camo.graph.rewire import oracle_scores


def _two_cluster_graph(seed=0, n=400, d=8, n_cross=60):
    """Edges mostly join nodes of the same feature cluster; a few cross edges play camouflage."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 2, n)
    x = (rng.normal(size=(n, d)) + 3.0 * c[:, None]).astype(np.float32)
    same = [rng.choice(np.where(c == k)[0], size=(2, 800)) for k in (0, 1)]
    cross = np.vstack([rng.choice(np.where(c == 0)[0], n_cross), rng.choice(np.where(c == 1)[0], n_cross)])
    e = np.hstack(same + [cross])
    e = e[:, e[0] != e[1]]
    return x, c, {"rur": np.sort(e, axis=0)}


def test_feature_scorer_downweights_cross_cluster_edges():
    x, c, edges = _two_cluster_graph()
    sc = FeatureEdgeScorer(steps=300, batch=256, eval_every=50, seed=0).fit(x, edges)
    s = sc.score(x, edges)["rur"]
    e = edges["rur"]
    cross = c[e[0]] != c[e[1]]
    assert sc.val_auc > 0.6  # random negatives share a cluster half the time, so the ceiling is ~0.75
    assert s.min() > 0 and s.max() < 1 and s.shape == (e.shape[1],)
    assert s[cross].mean() < s[~cross].mean()


def test_cosine_scores_range():
    x, _, edges = _two_cluster_graph()
    s = cosine_scores(x, edges)["rur"]
    assert s.min() >= 0 and s.max() <= 1


def test_oracle_scores():
    y = np.array([0, 0, 1, 1])
    e = {"rsr": np.array([[0, 0, 2], [1, 2, 3]])}
    assert oracle_scores(e, y)["rsr"].tolist() == [1.0, 0.0, 1.0]
