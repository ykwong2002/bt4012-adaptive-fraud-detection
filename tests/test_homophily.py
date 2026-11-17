import numpy as np

from camo.graph.homophily import homophily_table, node_homophily, text_homophily

# 0-1 same label, 0-2 cross-label (camouflage), 2-3 fraud-fraud; node 4 is isolated
Y = np.array([0, 0, 1, 1, 0])
E = np.array([[0, 0, 2], [1, 2, 3]])
S = np.array([0.2, 0.8, 0.5])


def test_node_homophily():
    h = node_homophily(E, Y, 5)
    assert np.allclose(h[:4], [0.5, 1.0, 0.5, 1.0]) and np.isnan(h[4])


def test_node_homophily_weighted():
    h = node_homophily(E, Y, 5, weights=np.array([1.0, 3.0, 1.0]))
    assert np.isclose(h[0], 0.25)       # the down-weighted same-label edge counts less


def test_homophily_table():
    t = homophily_table({"rsr": E}, Y, 5).iloc[0]
    assert np.isclose(t["edge_homophily"], 2 / 3)
    assert np.isclose(t["frac_fraud_edges_to_legit"], 0.5)
    assert np.isclose(t["isolated_frac"], 0.2)


def test_text_homophily_undated():
    h = text_homophily(E, S, 5)
    assert np.allclose(h[:4], [0.5, 0.2, 0.65, 0.5]) and np.isnan(h[4])


def test_text_homophily_only_counts_earlier_neighbours():
    dates = np.array([1, 2, 0, 3, 5])
    h = text_homophily(E, S, 5, dates)
    assert np.isclose(h[0], 0.8)        # neighbour 1 is later, so only neighbour 2 counts
    assert np.isclose(h[1], 0.2)
    assert np.isnan(h[2])               # both neighbours are later
    assert np.isclose(h[3], 0.5)


def test_text_homophily_no_future_leak():
    rng = np.random.default_rng(0)
    n, m = 200, 1500
    e = rng.integers(0, n, size=(2, m)); e = e[:, e[0] != e[1]]
    s = rng.random(e.shape[1]); dates = rng.integers(0, 50, n)
    before = text_homophily(e, s, n, dates)
    # a later review linked to existing ones must not change any existing review's value
    new = np.vstack([rng.integers(0, n, 30), np.full(30, n)])
    after = text_homophily(np.hstack([e, new]), np.r_[s, rng.random(30)], n + 1, np.r_[dates, 99])
    assert np.allclose(before, after[:n], equal_nan=True)
