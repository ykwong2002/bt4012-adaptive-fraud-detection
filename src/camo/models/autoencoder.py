"""Feature autoencoder; reconstruction error is an unsupervised anomaly score (used as a baseline and
as one of the pseudo-label sources)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class AE(nn.Module):
    def __init__(self, d: int, hidden: int = 16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden))
        self.dec = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def ae_anomaly_scores(x: np.ndarray, fit_idx: np.ndarray, seed: int, hidden: int = 16,
                      epochs: int = 200, lr: float = 1e-3) -> np.ndarray:
    """Fit on fit_idx (unlabelled), return per-node reconstruction error for all nodes."""
    torch.manual_seed(seed)
    mu, sd = x[fit_idx].mean(0), x[fit_idx].std(0) + 1e-6
    xt = torch.from_numpy((x - mu) / sd).float()
    model = AE(x.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    fit = torch.from_numpy(fit_idx).long()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(xt[fit]) - xt[fit]) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return ((model(xt) - xt) ** 2).mean(1).numpy()
