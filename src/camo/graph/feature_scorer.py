"""Label-free feature-consistency edge scorer for the graph-only track (decisions D11).

An MLP encodes each node from its own features; a per-relation diagonal bilinear decoder scores an
edge as sigmoid(z_u' diag(d_r) z_v + b_r). It is trained to tell observed edges from random node
pairs, so an edge scores low when its endpoints' features do not look like those of a typical edge
of that relation. The encoder deliberately sees no neighbours: a GNN encoder would pull a
camouflaged node towards its neighbours before its edges are scored.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn


class FeatureLinkModel(nn.Module):
    def __init__(self, d: int, relations: list[str], hidden: int = 64, emb: int = 32, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, emb))
        self.diag = nn.ParameterDict({r: nn.Parameter(torch.ones(emb)) for r in relations})
        self.bias = nn.ParameterDict({r: nn.Parameter(torch.zeros(1)) for r in relations})

    def logits(self, z: torch.Tensor, u: torch.Tensor, v: torch.Tensor, r: str) -> torch.Tensor:
        return (z[u] * self.diag[r] * z[v]).sum(1) + self.bias[r]


class FeatureEdgeScorer:
    def __init__(self, hidden: int = 64, emb: int = 32, lr: float = 3e-3, batch: int = 4096,
                 steps: int = 3000, eval_every: int = 100, patience: int = 8, val_frac: float = 0.05,
                 seed: int = 0):
        self.hidden, self.emb, self.lr, self.batch = hidden, emb, lr, batch
        self.steps, self.eval_every, self.patience, self.val_frac, self.seed = steps, eval_every, patience, val_frac, seed
        self.history: list[dict] = []

    def fit(self, x: np.ndarray, edges: dict[str, np.ndarray]) -> FeatureEdgeScorer:
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        n = x.shape[0]
        self.rels = [r for r, e in edges.items() if e.shape[1] > 0]
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        xt = torch.from_numpy(((x - self.mu) / self.sd).astype(np.float32))
        self.model = FeatureLinkModel(x.shape[1], self.rels, self.hidden, self.emb)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        train, val = {}, {}
        for r in self.rels:
            e = edges[r][:, rng.permutation(edges[r].shape[1])]
            k = max(1, int(self.val_frac * e.shape[1]))
            val[r], train[r] = e[:, :k], e[:, k:]
        val_neg = {r: rng.integers(0, n, size=(2, val[r].shape[1])) for r in self.rels}

        best, best_state, bad = -1.0, None, 0
        for step in range(1, self.steps + 1):
            self.model.train(); opt.zero_grad()
            z = self.model.enc(xt)
            loss = 0.0
            for r in self.rels:
                pos = train[r][:, rng.integers(0, train[r].shape[1], self.batch)]
                neg = rng.integers(0, n, size=(2, self.batch))
                lp = self.model.logits(z, torch.from_numpy(pos[0]), torch.from_numpy(pos[1]), r)
                ln = self.model.logits(z, torch.from_numpy(neg[0]), torch.from_numpy(neg[1]), r)
                loss = loss + F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) \
                    + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
            loss.backward(); opt.step()

            if step % self.eval_every == 0:
                aucs = self._val_auc(xt, val, val_neg)
                mean_auc = float(np.mean(list(aucs.values())))
                self.history.append({"step": step, "loss": loss.item(), "val_auc": mean_auc, **{f"val_auc_{r}": a for r, a in aucs.items()}})
                if mean_auc > best:
                    best, bad = mean_auc, 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.val_auc = best
        return self

    @torch.no_grad()
    def _val_auc(self, xt, val, val_neg) -> dict[str, float]:
        self.model.eval()
        z = self.model.enc(xt)
        out = {}
        for r in self.rels:
            p = self.model.logits(z, torch.from_numpy(val[r][0]), torch.from_numpy(val[r][1]), r).numpy()
            q = self.model.logits(z, torch.from_numpy(val_neg[r][0]), torch.from_numpy(val_neg[r][1]), r).numpy()
            out[r] = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q])
        return out

    @torch.no_grad()
    def score(self, x: np.ndarray, edges: dict[str, np.ndarray], chunk: int = 500_000) -> dict[str, np.ndarray]:
        """Edge plausibility in (0, 1) for every edge, aligned with `edges`."""
        self.model.eval()
        z = self.model.enc(torch.from_numpy(((x - self.mu) / self.sd).astype(np.float32)))
        out = {}
        for r, e in edges.items():
            s = np.empty(e.shape[1], dtype=np.float32)
            for i in range(0, e.shape[1], chunk):
                u, v = torch.from_numpy(e[0, i:i + chunk]), torch.from_numpy(e[1, i:i + chunk])
                s[i:i + chunk] = torch.sigmoid(self.model.logits(z, u, v, r)).numpy() if r in self.rels else 1.0
            out[r] = s
        return out


def cosine_scores(x: np.ndarray, edges: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Baseline: cosine similarity of standardised features, mapped to [0, 1]."""
    xs = (x - x.mean(0)) / (x.std(0) + 1e-6)
    xs /= np.linalg.norm(xs, axis=1, keepdims=True) + 1e-9
    return {r: ((np.einsum("ij,ij->i", xs[e[0]], xs[e[1]]) + 1) / 2).astype(np.float32) for r, e in edges.items()}
