"""Loader for the public CARE-GNN / DGL YelpChi.mat graph (no text, no dates, no ids)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

RELATIONS = ("rur", "rtr", "rsr")


@dataclass
class Graph:
    """A review graph. Edges are undirected and stored once with u < v."""

    x: np.ndarray                       # (n, d) float32 node features
    y: np.ndarray                       # (n,) int, 1 = fraud
    edges: dict[str, np.ndarray]        # relation -> (2, m) int64, u < v
    dates: np.ndarray | None = None     # (n,) datetime64[D] if known
    meta: dict = field(default_factory=dict)
    node_table: object = None           # pandas.DataFrame for rebuilt graphs (ids, rating, text)

    @property
    def n(self) -> int:
        return self.x.shape[0]

    def subgraph(self, keep: np.ndarray) -> Graph:
        """Induced subgraph on node indices `keep` (sorted), relabelled 0..k-1."""
        keep = np.asarray(keep)
        remap = -np.ones(self.n, dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        edges = {}
        for r, e in self.edges.items():
            m = (remap[e[0]] >= 0) & (remap[e[1]] >= 0)
            edges[r] = remap[e[:, m]]
        nt = self.node_table.iloc[keep].reset_index(drop=True) if self.node_table is not None else None
        return Graph(self.x[keep], self.y[keep], edges,
                     None if self.dates is None else self.dates[keep], dict(self.meta), nt)

    def edge_count(self) -> dict[str, int]:
        return {r: int(e.shape[1]) for r, e in self.edges.items()}


def _upper_edges(a: sp.spmatrix) -> np.ndarray:
    a = sp.triu(a.tocoo(), k=1).tocoo()
    return np.vstack([a.row, a.col]).astype(np.int64)


def load_mat(path: str | Path) -> Graph:
    m = sio.loadmat(str(path))
    x = m["features"]
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    y = np.asarray(m["label"]).ravel().astype(np.int64)
    edges = {r: _upper_edges(m[f"net_{r}"]) for r in RELATIONS}
    return Graph(x.astype(np.float32), y, edges, None, {"source": "mat", "path": str(path)})


def save_graph(g: Graph, path: str | Path) -> None:
    arrays = {"x": g.x, "y": g.y}
    for r, e in g.edges.items():
        arrays[f"edges_{r}"] = e
    if g.dates is not None:
        arrays["dates"] = g.dates.astype("datetime64[D]").astype(np.int64)
    np.savez_compressed(str(path), **arrays)
    if g.node_table is not None:
        g.node_table.to_parquet(str(path).replace(".npz", "_nodes.parquet"), index=False)


def load_graph(path: str | Path) -> Graph:
    z = np.load(str(path))
    edges = {k[6:]: z[k] for k in z.files if k.startswith("edges_")}
    dates = z["dates"].astype("datetime64[D]") if "dates" in z.files else None
    nt = None
    p = Path(str(path).replace(".npz", "_nodes.parquet"))
    if p.exists():
        import pandas as pd

        nt = pd.read_parquet(p)
    return Graph(z["x"], z["y"], edges, dates, {"source": "npz", "path": str(path)}, nt)
