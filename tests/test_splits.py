import numpy as np

from camo.data.graph_build import build_graph
from camo.eval.splits import equal_count_windows, mask_edges, stratified_split, temporal_plan


def test_windows_equal_count_and_monotone(raw_mini):
    g = build_graph(raw_mini, 60)
    w = equal_count_windows(g.dates, 4)
    assert set(np.unique(w)) <= {0, 1, 2, 3}
    for k in range(3):   # all dates in window k <= all dates in window k+1
        assert g.dates[w == k].max() <= g.dates[w == k + 1].min()
    assert np.diff(w[np.argsort(g.dates, kind="stable")]).min() >= 0


def test_mask_edges_no_edge_crosses_cutoff(raw_mini):
    g = build_graph(raw_mini, 60)
    cutoff = np.sort(g.dates)[g.n // 2]
    m = mask_edges(g.edges, g.dates, cutoff)
    for r, e in m.items():
        assert (g.dates[e[0]] <= cutoff).all() and (g.dates[e[1]] <= cutoff).all(), r
        assert e.shape[1] <= g.edges[r].shape[1]


def test_temporal_plan_and_stratified():
    w = np.repeat(np.arange(8), 10)
    plan = temporal_plan(w, 3, 1)
    assert plan == {"train": [0, 1, 2], "val": [3], "test": [4, 5, 6, 7]}
    y = np.r_[np.zeros(85, int), np.ones(15, int)]
    tr, va, te = stratified_split(y, 0)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(va) + len(te) == 100
    assert y[tr].mean() > 0 and y[te].mean() > 0
