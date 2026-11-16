"""Graph-level camouflage injection for the graph-only track (decisions D13).

scripts/09 injects *reviews* into the raw metadata and rebuilds the graph, which needs the raw release.
This module injects *nodes* straight into a graph with arrival waves (camo.eval.waves), so the
adversarial experiment also runs on the public .mat graph.

Each injected node is a camouflaged fraud review:
  * features: a randomly chosen training-wave fraud node's features plus Gaussian noise (noise x the
    training standard deviation), i.e. the attacker reuses behaviour that worked before;
  * degrees: in R-S-R / R-T-R up to the template's degree, so those degree distributions are
    preserved; in R-U-R the node takes the host user's whole review history (see next point), so its
    R-U-R degree is the host's, not the template's;
  * R-U-R (camouflage via a hijacked account): the reviews of one legitimate "host" user, i.e. a
    visible legitimate node and its visible R-U-R neighbours;
  * R-S-R / R-T-R (camouflage via a busy product): legitimate members of one visible node's
    neighbourhood in that relation (a product x star / product x month block), up to the degree.
Neighbours are only nodes that have arrived by the injected node's wave, so nothing links to the
future. Training waves are never touched. Injected nodes are not linked to each other (no campaign
structure); that is the hardest case for homophily-based detection and is stated as such.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from camo.data.load_mat import Graph
from camo.eval.waves import wave_dates


def _adjacency(e: np.ndarray, n: int) -> sp.csr_matrix:
    a = sp.coo_matrix((np.ones(e.shape[1], dtype=np.int8), (e[0], e[1])), shape=(n, n))
    return (a + a.T).tocsr()


def inject_camouflage(g: Graph, windows: np.ndarray, train_windows, target_windows, level: float,
                      seed: int, noise: float = 0.1) -> tuple[Graph, np.ndarray, np.ndarray]:
    """Append level x |wave| injected fraud nodes to every target wave.

    Returns (augmented graph with wave dates, wave id per node, injected mask). Clean nodes keep
    their ids 0..n-1, so every clean-node result is directly comparable with the clean run."""
    rng = np.random.default_rng(seed)
    n, y, x = g.n, g.y, g.x
    tr = np.isin(windows, list(train_windows))
    templates = np.where(tr & (y == 1))[0]
    if level <= 0 or templates.size == 0:
        return Graph(x, y, g.edges, wave_dates(windows), {**g.meta, "protocol": "waves"}), windows, np.zeros(n, bool)
    sd = x[tr].std(0)
    adj = {r: _adjacency(e, n) for r, e in g.edges.items()}
    deg = {r: np.diff(a.indptr) for r, a in adj.items()}
    new_x, new_w, new_e = [], [], {r: [] for r in g.edges}
    nid = n
    for k in target_windows:
        n_k = int(round(level * (windows == k).sum()))
        visible_legit = (windows <= k) & (y == 0)
        hosts = np.where(visible_legit)[0]
        if n_k == 0 or hosts.size == 0:
            continue
        for _ in range(n_k):
            t = rng.choice(templates)
            new_x.append(x[t] + noise * sd * rng.standard_normal(x.shape[1]))
            new_w.append(k)
            for r, a in adj.items():
                d = int(deg[r][t])
                if d == 0:
                    continue
                h = rng.choice(hosts)
                nb = a.indices[a.indptr[h]:a.indptr[h + 1]]
                nb = nb[visible_legit[nb]]
                if r == "rur":          # the hijacked user's other reviews, all of them
                    cand = np.r_[h, nb]
                else:                   # a legitimate slice of one product block, up to the degree
                    cand = np.r_[h, rng.permutation(nb)[: max(0, d - 1)]]
                cand = np.unique(cand)
                new_e[r].append(np.vstack([cand, np.full(cand.size, nid)]))
            nid += 1
    if not new_x:
        return Graph(x, y, g.edges, wave_dates(windows), {**g.meta, "protocol": "waves"}), windows, np.zeros(n, bool)
    m = len(new_x)
    x_aug = np.vstack([x, np.asarray(new_x, dtype=np.float32)]).astype(np.float32)
    y_aug = np.r_[y, np.ones(m, dtype=y.dtype)]
    w_aug = np.r_[windows, np.asarray(new_w, dtype=np.int64)]
    edges = {}
    for r, e in g.edges.items():
        extra = np.hstack(new_e[r]).astype(np.int64) if new_e[r] else np.zeros((2, 0), np.int64)
        edges[r] = np.hstack([e, extra])          # u (clean, < n) < v (injected, >= n): u < v holds
    inj = np.r_[np.zeros(n, bool), np.ones(m, bool)]
    meta = {**g.meta, "protocol": "waves", "n_injected": m}
    return Graph(x_aug, y_aug, edges, wave_dates(w_aug), meta), w_aug, inj
