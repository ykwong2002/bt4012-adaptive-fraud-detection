import numpy as np

from camo.selftrain.bias_audit import collapse_flag
from camo.selftrain.pseudo_labels import (
    community_labels,
    community_propensity,
    em_prior,
    select_pseudo,
)


def test_select_pseudo_naive_vs_per_class():
    rng = np.random.default_rng(0)
    # scores: many confident negatives, few mildly-positive -> naive picks almost only negatives
    f = np.concatenate([rng.uniform(0.0, 0.1, 900), rng.uniform(0.55, 0.7, 100)])
    pool = np.arange(1000)
    idx, lab = select_pseudo(f, pool, 0.9, per_class=False, prior=None)
    assert lab.mean() < 0.05
    idx, lab = select_pseudo(f, pool, 0.9, per_class=True, prior=None)
    assert 0.05 < lab.mean() < 0.2
    idx, lab = select_pseudo(f, pool, 0.9, per_class=False, prior=0.3)
    assert abs(lab.mean() - 0.3) < 0.02


def test_em_prior_recovers_shift():
    rng = np.random.default_rng(0)
    # calibrated posteriors under train prior 0.1; new sample has prior 0.4
    y = rng.random(20000) < 0.4
    s = np.where(y, rng.beta(5, 2, 20000), rng.beta(2, 8, 20000))
    # make scores "calibrated at prior 0.1" by Bayes-shifting them from 0.4 to 0.1
    s_tr = (s * 0.1 / 0.4) / (s * 0.1 / 0.4 + (1 - s) * 0.9 / 0.6)
    est = em_prior(s_tr, 0.1)
    assert abs(est - 0.4) < 0.08


def test_community_propensity_leave_one_out():
    score = np.array([1.0, 0.0, 0.5, 0.9]); comm = np.array([0, 0, 0, 1]); pool = np.arange(4)
    cp = community_propensity(score, comm, pool)
    assert np.isclose(cp[0], 0.25) and np.isclose(cp[1], 0.75) and np.isclose(cp[3], 0.9)


def test_community_labels_runs():
    edges = {"rur": np.array([[0, 1, 3], [1, 2, 4]]), "rsr": np.array([[0, 3], [2, 5]])}
    lab = community_labels(6, edges, {"rsr": np.array([0.9, 0.1])}, ["rur", "rsr_weighted"], seed=0)
    assert lab.shape == (6,) and lab[0] == lab[1] == lab[2]


def test_collapse_flag():
    rounds = [{"pseudo_pos_share": 0.12, "prior": 0.15, "probe_minority_recall": 0.6},
              {"pseudo_pos_share": 0.02, "prior": 0.15, "probe_minority_recall": 0.2}]
    assert collapse_flag(rounds)
    assert not collapse_flag([rounds[0], rounds[0]])


def test_em_prior_with_balanced_weights():
    """Review issue #7: a class-balanced model is calibrated to prior 0.5, so EM must start there."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n, prior = 40000, 0.12
    y = rng.random(n) < prior
    x = np.where(y, rng.normal(1.5, 1, n), rng.normal(0, 1, n))[:, None]
    clf = LogisticRegression(class_weight="balanced").fit(x[: n // 2], y[: n // 2])
    s = clf.predict_proba(x[n // 2:])[:, 1]
    assert abs(em_prior(s, 0.5) - prior) < 0.03
    assert em_prior(s, float(s.mean())) > prior + 0.1      # the old call: badly biased upwards


def _dated_graph(raw_mini):
    from camo.data.features_pit import compute_pit_features
    from camo.data.graph_build import build_graph

    return build_graph(raw_mini, 60, features=compute_pit_features(raw_mini))


def test_triggered_loop_never_sees_the_future(raw_mini):
    """Review issue #9: the challenger built at an alarm in window t may only use data up to t.
    Scrambling the features and labels of every later node must not change it."""
    from camo import load_config
    from camo.eval.splits import equal_count_windows
    from camo.selftrain.loop import run_triggered

    cfg, scfg = load_config("base"), {**load_config("selftrain"), "rounds": 2}
    v = {"all": scfg["variants"]["all"]}
    g = _dated_graph(raw_mini)
    w = equal_count_windows(g.dates, cfg["windows"]["n_windows"])
    t = 5
    kw = dict(iters=20, probe_per_window=10, alarms={t: True}, windows=w, ae_epochs=5)
    rows1, _, _ = run_triggered(g, cfg, scfg, 0, None, v, **kw)

    g2 = g.subgraph(np.arange(g.n))
    future = w > t
    g2.x = g2.x.copy(); g2.x[future] = np.random.default_rng(1).normal(size=g2.x[future].shape).astype(np.float32)
    g2.y = g2.y.copy(); g2.y[future] = 1 - g2.y[future]
    rows2, _, _ = run_triggered(g2, cfg, scfg, 0, None, v, **kw)

    r1 = next(r for r in rows1 if r["window"] == t); r2 = next(r for r in rows2 if r["window"] == t)
    assert r1["retrained"] and r2["retrained"]
    for k in ("chal_probe_ap", "deployed_probe_ap", "pseudo_pos_share", "promoted", "n_pool"):
        assert r1[k] == r2[k], k
    # windows before the alarm are scored by the incumbent; nothing is retrained before t
    assert all(not r["retrained"] for r in rows1 if r["window"] < t)
    assert all(r["deployed"] == "incumbent" for r in rows1 if r["window"] <= t)


def test_triggered_loop_labels_the_model_actually_deployed(raw_mini):
    """A retrain that is not promoted must leave the previously deployed model in place."""
    from camo import load_config
    from camo.eval.splits import equal_count_windows
    from camo.selftrain.loop import run_triggered

    cfg, scfg = load_config("base"), {**load_config("selftrain"), "rounds": 1}
    g = _dated_graph(raw_mini)
    w = equal_count_windows(g.dates, cfg["windows"]["n_windows"])
    rows, _, _ = run_triggered(g, cfg, scfg, 0, None, {"all": scfg["variants"]["all"]}, iters=20,
                               probe_per_window=10, alarms={3: True, 4: True, 5: True}, windows=w, ae_epochs=5)
    current = "incumbent"
    for r in sorted(rows, key=lambda r: r["window"]):
        assert r["deployed"] == current
        if r["promoted"]:
            current = f"challenger@w{r['window']}"


def test_static_ablation_runs_with_control_and_refuses_dates(raw_mini):
    """The .mat (no dates) path: fair incumbent, a same-settings control, and no temporal graphs."""
    import pytest

    from camo import load_config
    from camo.selftrain.loop import run_selftrain

    cfg, scfg = load_config("base"), {**load_config("selftrain"), "rounds": 2}
    g = _dated_graph(raw_mini)
    variants = {k: scfg["variants"][k] for k in ("naive", "all")}
    with pytest.raises(ValueError, match="run_triggered"):
        run_selftrain(g, cfg, scfg, 0, None, variants, 20, 10, ae_epochs=5)
    g.dates = None
    rows, bracket = run_selftrain(g, cfg, scfg, 0, None, variants, 20, 10, ae_epochs=5)
    last = [r for r in rows if r["round"] == 1]
    assert {r["variant"] for r in last} == {"naive", "all"}
    assert all("control_eval_ap" in r and "pseudo_label_gain" in r for r in last)
    assert bracket["protocol"] == "stratified" and 0 < bracket["em_prior"] < 1
