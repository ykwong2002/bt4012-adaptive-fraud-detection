import numpy as np
import torch

from camo.models.care_static import care_filter
from camo.models.wsage import RelGraph, WeightedSAGE, sample_neighbours


def test_sample_neighbours_caps_indegree():
    dst = torch.tensor([0] * 10 + [1] * 3 + [2] * 1)
    src = torch.arange(14); w = torch.ones(14)
    s, d, ww = sample_neighbours(src, dst, w, 4, torch.Generator().manual_seed(0))
    counts = torch.bincount(d, minlength=3)
    assert counts.tolist() == [4, 3, 1]
    assert set(zip(s.tolist(), d.tolist())) <= set(zip(src.tolist(), dst.tolist()))


def test_weighted_mean_aggregation_uses_weights():
    # node 0 has neighbours 1 (weight 1) and 2 (weight 0): aggregation must equal h[1]
    edges = {"r": np.array([[0, 0], [1, 2]])}
    g = RelGraph(3, edges, {"r": np.array([1.0, 0.0])})
    model = WeightedSAGE(2, ["r"], hidden=2, n_layers=1, dropout=0.0)
    layer = model.layers[0]
    h = torch.tensor([[0.0, 0.0], [1.0, 2.0], [5.0, 5.0]])
    rel = g.sampled(0, torch.Generator().manual_seed(0))
    out = layer(h, rel)
    expected = layer.lin_self(h[0]) + layer.lin_rel["r"](h[1])
    assert torch.allclose(out[0], expected, atol=1e-6)


def test_care_filter_keeps_most_similar():
    # centre 0 with neighbours 1,2,3; scores make 1 closest, 3 farthest
    edges = np.array([[0, 0, 0], [1, 2, 3]])
    s = np.array([0.5, 0.52, 0.7, 0.95])
    kept = care_filter(edges, s, rho=0.34)
    pairs = set(map(tuple, kept.T.tolist()))
    assert (0, 1) in pairs
    # neighbours 2 and 3 have degree 1 so they keep their own edge to 0 (ceil(0.34*1)=1)
    assert len(pairs) == 3
    kept2 = care_filter(edges, s, rho=1.0)
    assert kept2.shape[1] == 3
