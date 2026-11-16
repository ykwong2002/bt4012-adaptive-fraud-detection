"""R-U-R profile scorer (decisions D15): label-free, time-respecting, separates profiles."""
import numpy as np

from camo.graph.rur_scorer import rur_profile_scores


def _graph(seed=0, n=600):
    """Two behaviour types; R-U-R links reviews of the same type (users), R-S-R mixes both types."""
    rng = np.random.default_rng(seed)
    t = (rng.random(n) < 0.3).astype(int)
    x = rng.standard_normal((n, 5)).astype(np.float32) + 1.5 * t[:, None]
    users = rng.integers(0, 150, n) * 2 + t           # a user only writes one type of review
    order = np.argsort(users, kind="stable")
    same = users[order][1:] == users[order][:-1]
    rur = np.vstack([order[:-1][same], order[1:][same]])
    rur = np.vstack([rur.min(0), rur.max(0)])
    u, v = rng.integers(0, n, 3000), rng.integers(0, n, 3000)
    rsr = np.unique(np.vstack([np.minimum(u, v), np.maximum(u, v)]), axis=1)
    rsr = rsr[:, rsr[0] < rsr[1]]
    return x, t, {"rur": rur, "rsr": rsr}


def test_scores_separate_and_are_bounded():
    from sklearn.metrics import roc_auc_score

    x, t, e = _graph()
    s = rur_profile_scores(x, e, np.arange(len(t)))
    assert (s["rur"] == 1).all()
    assert ((s["rsr"] > 0) & (s["rsr"] <= 1)).all()
    same = t[e["rsr"][0]] == t[e["rsr"][1]]
    assert roc_auc_score(same, s["rsr"]) > 0.75


def test_label_free():
    """Scores depend on features and structure only: shuffling which nodes are 'fraud' changes nothing
    because labels are never an input."""
    import inspect

    from camo.graph import rur_scorer
    assert "y" not in inspect.signature(rur_scorer.rur_profile_scores).parameters


def test_time_respecting():
    """Changing anything about nodes that arrive later never changes the score of an earlier edge."""
    x, t, e = _graph(seed=1)
    n = len(t)
    arrival = np.random.default_rng(2).integers(0, 4, n)
    fit = np.where(arrival == 0)[0]
    a = rur_profile_scores(x, e, fit, arrival)
    x2 = x.copy()
    x2[arrival == 3] += 10.0                               # rewrite the latest stage's features
    b = rur_profile_scores(x2, e, fit, arrival)
    early = np.maximum(arrival[e["rsr"][0]], arrival[e["rsr"][1]]) < 3
    np.testing.assert_allclose(a["rsr"][early], b["rsr"][early], rtol=1e-6)
    assert not np.allclose(a["rsr"][~early], b["rsr"][~early])
