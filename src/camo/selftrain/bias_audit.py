"""Per-round audit of normal-class bias in self-training (professor's comment #2)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def audit_round(pseudo_idx: np.ndarray, pseudo_lab: np.ndarray, comm: np.ndarray, prior: float,
                probe_idx: np.ndarray, y: np.ndarray, score: np.ndarray, k_frac: float) -> dict:
    """pseudo-set positive share vs prior, dispersion of per-community positive shares, the score
    threshold implied by the pseudo positives, and minority recall / AUC-PR on the labelled probe."""
    pos_share = float(pseudo_lab.mean()) if len(pseudo_lab) else np.nan
    cp = {}
    for c in np.unique(comm[pseudo_idx]):
        m = comm[pseudo_idx] == c
        if m.sum() >= 5:
            cp[c] = pseudo_lab[m].mean()
    pos_scores = score[pseudo_idx[pseudo_lab == 1]]
    thr = float(pos_scores.min()) if len(pos_scores) else np.nan
    # probe: minority recall when flagging the top k_frac of probe nodes
    sp, yp = score[probe_idx], y[probe_idx]
    k = max(1, int(round(k_frac * len(probe_idx))))
    top = np.argsort(-sp)[:k]
    rec = float(yp[top].sum() / max(1, yp.sum()))
    ap = float(average_precision_score(yp, sp)) if yp.sum() > 0 and yp.sum() < len(yp) else np.nan
    return {"n_pseudo": int(len(pseudo_idx)), "pseudo_pos_share": pos_share, "prior": prior,
            "pos_share_minus_prior": pos_share - prior, "comm_pos_share_std": float(np.std(list(cp.values()))) if cp else np.nan,
            "frac_comms_all_negative": float(np.mean([v == 0 for v in cp.values()])) if cp else np.nan,
            "pos_threshold": thr, "probe_minority_recall": rec, "probe_auc_pr": ap}


def collapse_flag(rounds: list[dict], share_tol: float = 0.5, recall_drop: float = 0.3) -> bool:
    """True if the pseudo-set positive share fell below (1-share_tol)*prior or probe minority recall
    dropped by more than recall_drop relative to round 0 — the signature of normal-class collapse."""
    if not rounds:
        return False
    r0, rl = rounds[0], rounds[-1]
    share_bad = rl["pseudo_pos_share"] < (1 - share_tol) * rl["prior"]
    rec_bad = (r0["probe_minority_recall"] > 0) and (rl["probe_minority_recall"] < (1 - recall_drop) * r0["probe_minority_recall"])
    return bool(share_bad or rec_bad)
