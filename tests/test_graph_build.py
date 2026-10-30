import numpy as np
import pandas as pd

from camo.data.graph_build import build_graph, build_relations, filter_products


def test_filter_drops_big_products(raw_mini):
    kept = filter_products(raw_mini, max_reviews_per_product=60)
    assert kept.groupby("product_id").size().max() <= 60
    assert len(kept) < len(raw_mini)


def test_relations_match_bruteforce(raw_mini):
    df = filter_products(raw_mini, 60).sort_values(["date", "review_id"]).reset_index(drop=True)
    e = build_relations(df)
    for r, key in {"rur": lambda a, b: a.user_id == b.user_id,
                   "rsr": lambda a, b: a.product_id == b.product_id and round(a.rating) == round(b.rating),
                   "rtr": lambda a, b: a.product_id == b.product_id and a.date.to_period("M") == b.date.to_period("M")}.items():
        brute = {(i, j) for i in range(len(df)) for j in range(i + 1, len(df)) if key(df.iloc[i], df.iloc[j])}
        got = set(map(tuple, e[r].T.tolist()))
        assert got == brute, r
        assert (e[r][0] < e[r][1]).all()


def test_build_graph_shapes(raw_mini):
    g = build_graph(raw_mini, 60)
    assert g.y.shape[0] == g.n and g.dates.shape[0] == g.n
    assert set(g.edges) == {"rur", "rsr", "rtr"}
    assert np.all(np.diff(g.dates.astype(np.int64)) >= 0)   # nodes ordered by date
    assert g.node_table is not None and "text" in g.node_table


def test_features_align_after_filter(raw_mini):
    from camo.data.features_pit import compute_pit_features

    feats = compute_pit_features(raw_mini)
    g = build_graph(raw_mini, 60, features=feats)
    assert g.x.shape[0] == g.y.shape[0] == len(g.node_table)
    assert g.n < len(raw_mini)
    # rating_extreme (column 9) must agree with the node table's own rating
    ext = g.node_table["rating"].isin([1, 5]).astype(int).to_numpy()
    assert np.array_equal(g.x[:, 9].astype(int), ext)
