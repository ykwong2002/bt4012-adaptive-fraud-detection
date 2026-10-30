"""Rebuild the YelpChi review graph from raw metadata.

Steps (GraphConsis / CARE-GNN conventions, see docs/decisions.md D1):
  1. drop products with more than `max_reviews_per_product` reviews (default 800)
  2. R-U-R: reviews by the same user
     R-S-R: reviews of the same product with the same star rating
     R-T-R: reviews of the same product posted in the same calendar month
Edges are undirected, stored once with u < v, node ids are 0..n-1 in the filtered table's order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from camo.data.load_mat import Graph


def _pairs_within_groups(keys: pd.DataFrame | pd.Series, node_ids: np.ndarray) -> np.ndarray:
    """All unordered pairs of node ids sharing the same key. Vectorised per group size."""
    df = pd.DataFrame({"node": node_ids})
    df["g"] = pd.factorize(pd.util.hash_pandas_object(keys, index=False))[0]
    df = df.sort_values(["g", "node"], kind="stable")
    sizes = df.groupby("g").size()
    out = []
    nodes = df["node"].to_numpy()
    starts = np.r_[0, np.cumsum(sizes.to_numpy())[:-1]]
    for s, k in zip(starts, sizes.to_numpy()):
        if k < 2:
            continue
        grp = nodes[s:s + k]
        iu, ju = np.triu_indices(k, k=1)
        out.append(np.vstack([grp[iu], grp[ju]]))
    if not out:
        return np.zeros((2, 0), dtype=np.int64)
    e = np.hstack(out).astype(np.int64)
    return e[:, np.lexsort((e[1], e[0]))]


def filter_products(df: pd.DataFrame, max_reviews_per_product: int = 800) -> pd.DataFrame:
    """Drop products with more than N reviews. Keeps the original index so callers can align features."""
    counts = df.groupby("product_id")["review_id"].transform("size")
    return df[counts <= max_reviews_per_product]


def build_relations(df: pd.DataFrame) -> dict[str, np.ndarray]:
    node = np.arange(len(df))
    month = df["date"].dt.to_period("M").astype(str)
    return {
        "rur": _pairs_within_groups(df[["user_id"]], node),
        "rsr": _pairs_within_groups(pd.DataFrame({"p": df["product_id"], "s": df["rating"].round()}), node),
        "rtr": _pairs_within_groups(pd.DataFrame({"p": df["product_id"], "m": month}), node),
    }


def build_graph(raw: pd.DataFrame, max_reviews_per_product: int = 800,
                features: pd.DataFrame | None = None) -> Graph:
    """`features` must be a DataFrame indexed like `raw` (e.g. from compute_pit_features(raw))."""
    df = filter_products(raw, max_reviews_per_product)
    df = df.sort_values(["date", "review_id"], kind="stable")
    orig_index = df.index
    df = df.reset_index(drop=True)
    df["node"] = np.arange(len(df))
    edges = build_relations(df)
    if features is not None:
        x = features.loc[orig_index].to_numpy()
    else:
        x = np.zeros((len(df), 0), dtype=np.float32)
    meta = {"source": "raw", "n_products": int(df["product_id"].nunique()),
            "n_users": int(df["user_id"].nunique()), "max_reviews_per_product": max_reviews_per_product}
    return Graph(x.astype(np.float32), df["label"].to_numpy().astype(np.int64), edges,
                 df["date"].to_numpy().astype("datetime64[D]"), meta, df)


def compare_to_published(g: Graph, published: dict, tolerance: dict) -> pd.DataFrame:
    """Rebuilt-vs-published table used in the EDA and the report (pre-registered tolerances)."""
    rows = [("nodes", g.n, published["n_nodes"], tolerance["nodes"]),
            ("products", g.meta.get("n_products", np.nan), published["n_products"], 0.0),
            ("fraud_rate", float(g.y.mean()), published["fraud_rate"], tolerance["nodes"])]
    for r, m in g.edge_count().items():
        rows.append((f"edges_{r}", m, published["edges"][r], tolerance["edges"]))
    out = pd.DataFrame(rows, columns=["quantity", "rebuilt", "published", "tolerance"])
    out["rel_diff"] = (out["rebuilt"] - out["published"]) / out["published"]
    out["within_tol"] = out["rel_diff"].abs() <= out["tolerance"]
    return out
