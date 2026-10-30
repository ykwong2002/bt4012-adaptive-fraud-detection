"""WeightedSAGE: GraphSAGE (Hamilton et al. 2017) mean aggregator rewritten with edge weights and
multiple relations, plus fixed-size neighbour sampling done by us (no pyg-lib / torch-sparse needed).

h_v' = W_self h_v + sum_r W_r * ( sum_{u in N_r(v)} w_uv h_u / sum_{u in N_r(v)} w_uv )

Stock PyG SAGEConv ignores edge weights, which is exactly what soft rewiring needs, hence this layer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def directed(edges: np.ndarray, weights: np.ndarray | None):
    """Undirected (2, m) with u<v  ->  both directions (2, 2m) and matching weights."""
    e = np.hstack([edges, edges[::-1]])
    w = np.ones(e.shape[1], dtype=np.float32) if weights is None else np.concatenate([weights, weights]).astype(np.float32)
    return e, w


def sample_neighbours(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, k: int, gen: torch.Generator):
    """Keep at most k incoming edges per destination node, uniformly at random."""
    if k <= 0 or src.numel() == 0:
        return src, dst, w
    perm = torch.randperm(src.numel(), generator=gen)
    src, dst, w = src[perm], dst[perm], w[perm]
    order = torch.argsort(dst, stable=True)
    src, dst, w = src[order], dst[order], w[order]
    # rank within each dst group
    first = torch.zeros_like(dst, dtype=torch.bool); first[0] = True; first[1:] = dst[1:] != dst[:-1]
    start = torch.cummax(torch.where(first, torch.arange(dst.numel()), torch.zeros_like(dst)), 0).values
    rank = torch.arange(dst.numel()) - start
    keep = rank < k
    return src[keep], dst[keep], w[keep]


class RelGraph:
    """Tensors for one graph state: per relation src/dst/weight (both directions)."""

    def __init__(self, n: int, edges: dict[str, np.ndarray], weights: dict[str, np.ndarray] | None = None):
        self.n = n
        self.rel = {}
        for r, e in edges.items():
            ed, wd = directed(e, None if weights is None else weights.get(r))
            self.rel[r] = (torch.from_numpy(ed[0]).long(), torch.from_numpy(ed[1]).long(), torch.from_numpy(wd))

    def sampled(self, k: int, gen: torch.Generator) -> dict[str, tuple]:
        return {r: sample_neighbours(s, d, w, k, gen) for r, (s, d, w) in self.rel.items()}


class WeightedSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, relations: list[str]):
        super().__init__()
        self.relations = relations
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_rel = nn.ModuleDict({r: nn.Linear(in_dim, out_dim, bias=False) for r in relations})

    def forward(self, h: torch.Tensor, rel: dict[str, tuple]) -> torch.Tensor:
        out = self.lin_self(h)
        n = h.shape[0]
        for r in self.relations:
            src, dst, w = rel[r]
            if src.numel() == 0:
                continue
            msg = h[src] * w.unsqueeze(1)
            agg = torch.zeros(n, h.shape[1], dtype=h.dtype).index_add_(0, dst, msg)
            den = torch.zeros(n, dtype=h.dtype).index_add_(0, dst, w).clamp_min(1e-9).unsqueeze(1)
            out = out + self.lin_rel[r](agg / den)
        return out


class WeightedSAGE(nn.Module):
    def __init__(self, in_dim: int, relations: list[str], hidden: int = 64, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList([WeightedSAGELayer(dims[i], dims[i + 1], relations) for i in range(n_layers)])
        self.out = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, rel: dict[str, tuple]) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, rel))
            h = F.dropout(h, self.dropout, self.training)
        return self.out(h).squeeze(-1)


def train_wsage(x: np.ndarray, y: np.ndarray, graph_train: RelGraph, train_idx: np.ndarray,
                val_idx: np.ndarray | None, graph_val: RelGraph | None, seed: int,
                hidden: int = 64, epochs: int = 150, lr: float = 5e-3, k_sample: int = 20,
                k_eval: int = 100, patience: int = 20, verbose: bool = False) -> WeightedSAGE:
    """Full-batch training with per-epoch neighbour sampling; early stopping on val AUC-PR.
    Train labels come only from train_idx; the val graph may contain more (unlabelled) nodes."""
    from sklearn.metrics import average_precision_score

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    xt = torch.from_numpy(x).float(); yt = torch.from_numpy(y).float()
    model = WeightedSAGE(x.shape[1], list(graph_train.rel), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    pos_weight = torch.tensor([(len(train_idx) - y[train_idx].sum()) / max(1, y[train_idx].sum())]).float()
    tr = torch.from_numpy(train_idx).long()
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits = model(xt, graph_train.sampled(k_sample, gen))
        loss = F.binary_cross_entropy_with_logits(logits[tr], yt[tr], pos_weight=pos_weight)
        loss.backward(); opt.step()
        if val_idx is not None and len(val_idx) and y[val_idx].sum() > 0:
            s = predict_wsage(model, x, graph_val or graph_train, seed, k_eval)
            ap = average_precision_score(y[val_idx], s[val_idx])
            if ap > best:
                best, bad, best_state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
            if verbose and ep % 10 == 0:
                print(f"ep {ep} loss {loss.item():.4f} val_ap {ap:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_wsage(model: WeightedSAGE, x: np.ndarray, graph: RelGraph, seed: int, k_eval: int = 100) -> np.ndarray:
    model.eval()
    gen = torch.Generator().manual_seed(seed + 12345)
    logits = model(torch.from_numpy(x).float(), graph.sampled(k_eval, gen))
    return torch.sigmoid(logits).numpy()
