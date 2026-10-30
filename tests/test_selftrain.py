import numpy as np

from camo.selftrain.bias_audit import collapse_flag
from camo.selftrain.pseudo_labels import community_labels, community_propensity, em_prior, select_pseudo


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
