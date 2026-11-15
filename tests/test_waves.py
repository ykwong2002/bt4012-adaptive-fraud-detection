"""Graph-only track (decisions D13): arrival waves and graph-level camouflage injection."""
import numpy as np

from camo.adversarial.graph_inject import inject_camouflage
from camo.data.load_mat import Graph
from camo.eval.protocol import build_folds
from camo.eval.splits import temporal_plan
from camo.eval.waves import assign_waves, with_waves

CFG = {"windows": {"n_windows": 8, "n_train_windows": 3, "n_val_windows": 1}}


def random_graph(n=800, seed=0, fraud=0.15):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud).astype(np.int64)
    x = rng.standard_normal((n, 6)).astype(np.float32)
    edges = {}
    for r, m in [("rur", 400), ("rsr", 3000), ("rtr", 1500)]:
        u, v = rng.integers(0, n, m), rng.integers(0, n, m)
        e = np.unique(np.vstack([np.minimum(u, v), np.maximum(u, v)]), axis=1)
        edges[r] = e[:, e[0] < e[1]]
    return Graph(x, y, edges, None, {"source": "test"})


def test_waves_are_stratified_and_equal_sized():
    g = random_graph()
    w = assign_waves(g.y, 8, seed=1)
    sizes = np.bincount(w, minlength=8)
    assert sizes.max() - sizes.min() <= 2
    pos = np.bincount(w[g.y == 1], minlength=8)
    assert pos.max() - pos.min() <= 1
    assert not np.array_equal(w, assign_waves(g.y, 8, seed=2))     # re-drawn per seed


def test_wave_protocol_masks_later_waves():
    g = random_graph()
    w = assign_waves(g.y, 8, seed=0)
    protocol, train_idx, folds = build_folds(with_waves(g, w), CFG, 0, windows=w)
    assert protocol == "waves"
    assert set(np.unique(w[train_idx])) == {0, 1, 2}
    for e in folds[0].edges.values():                               # train graph: train waves only
        assert (w[e] <= 2).all()
    for f in folds[2:]:                                              # test wave k sees waves <= k only
        k = int(f.name.split("_w")[1])
        assert (w[f.eval_idx] == k).all()
        for e in f.edges.values():
            assert (w[e] <= k).all()


def test_injection_leaves_training_untouched_and_links_no_future():
    g = random_graph()
    w = assign_waves(g.y, 8, seed=0)
    plan = temporal_plan(w, 3, 1)
    clean, _, _ = inject_camouflage(g, w, plan["train"], plan["test"], 0.0, seed=0)
    gi, wi, inj = inject_camouflage(g, w, plan["train"], plan["test"], 0.2, seed=0)
    for k in plan["test"]:
        assert inj[wi == k].sum() == round(0.2 * (w == k).sum())
    assert (wi[inj] >= min(plan["test"])).all() and gi.y[inj].all()
    np.testing.assert_array_equal(gi.x[: g.n], g.x)
    _, tr0, f0 = build_folds(clean, CFG, 0, windows=w)
    _, tr1, f1 = build_folds(gi, CFG, 0, windows=wi)
    np.testing.assert_array_equal(tr0, tr1)
    for r in g.edges:
        np.testing.assert_array_equal(f0[0].edges[r], f1[0].edges[r])
    for r, e in gi.edges.items():
        assert (e[0] < e[1]).all()
        new = e[:, e.shape[1] - (e[1] >= g.n).sum():]
        assert (new[1] >= g.n).all() and (new[0] < g.n).all()      # injected nodes link only to clean nodes
        assert (gi.y[new[0]] == 0).all()                            # ... that are legitimate (camouflage)
        assert (wi[new[0]] <= wi[new[1]]).all()                     # ... and have already arrived


def test_injection_is_reproducible():
    g = random_graph()
    w = assign_waves(g.y, 8, seed=0)
    plan = temporal_plan(w, 3, 1)
    a = inject_camouflage(g, w, plan["train"], plan["test"], 0.1, seed=3)
    b = inject_camouflage(g, w, plan["train"], plan["test"], 0.1, seed=3)
    np.testing.assert_array_equal(a[0].x, b[0].x)
    for r in g.edges:
        np.testing.assert_array_equal(a[0].edges[r], b[0].edges[r])


def test_degree_features_are_comparable_across_graph_states():
    """D14: rescaled degree has the same mean on the sparse training graph as on the full graph;
    raw degree would differ by the visible share (about 3/8 here)."""
    from camo.models.catboost_model import DEG_SCALE, neighbour_aggregates

    g = random_graph(n=4000, seed=1)
    w = assign_waves(g.y, 8, seed=0)
    _, tr, folds = build_folds(with_waves(g, w), CFG, 0, windows=w)
    first, last = folds[0], folds[-1]
    assert first.n_visible == len(tr) and last.n_visible == g.n
    d = g.x.shape[1]
    col = d  # first relation's degree column sits right after its d neighbour means
    a, names = neighbour_aggregates(g.x, first.edges, None, first.n_visible)
    b, _ = neighbour_aggregates(g.x, last.edges, None, last.n_visible)
    assert names[col].endswith(f"wdeg_per{DEG_SCALE}")
    ratio = a[tr, col].mean() / b[tr, col].mean()
    assert 0.85 < ratio < 1.15
    raw_a = a[tr, col] * first.n_visible / DEG_SCALE
    raw_b = b[tr, col] * last.n_visible / DEG_SCALE
    assert raw_a.mean() / raw_b.mean() < 0.5
