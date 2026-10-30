"""Stage 2: pairwise relative-fit labels on a sample of edges (training period only), per relation,
with a test-retest subset in swapped order. -> data/llm/pair_scores.parquet keyed by
(relation, u, v, swap) where u, v are review_ids."""
from __future__ import annotations

import numpy as np
import pandas as pd

from camo.llm.batch_runner import make_request
from camo.llm.prompts import PAIR_SCHEMA, pair_system, pair_user_message


def sample_pairs(edges: dict[str, np.ndarray], eligible: np.ndarray, emb: np.ndarray,
                 n_per_relation: int, seed: int, n_bins: int = 5) -> pd.DataFrame:
    """Edges with both endpoints eligible (training period), stratified over embedding-cosine
    quantile bins so the distilled scorer sees the whole similarity range."""
    rng = np.random.default_rng(seed)
    rows = []
    for r, e in edges.items():
        ok = eligible[e[0]] & eligible[e[1]]
        e = e[:, ok]
        if e.shape[1] == 0:
            continue
        cos = np.einsum("ij,ij->i", emb[e[0]], emb[e[1]])
        bins = np.digitize(cos, np.quantile(cos, np.linspace(0, 1, n_bins + 1)[1:-1]))
        per_bin = max(1, n_per_relation // n_bins)
        for b in range(n_bins):
            idx = np.where(bins == b)[0]
            take = rng.choice(idx, min(per_bin, idx.size), replace=False) if idx.size else []
            for i in take:
                rows.append({"relation": r, "u_node": int(e[0, i]), "v_node": int(e[1, i]), "cos": float(cos[i]), "bin": b})
    return pd.DataFrame(rows)


def build_pair_requests(pairs: pd.DataFrame, node_table: pd.DataFrame, model: str,
                        n_retest: int = 0, seed: int = 0) -> list[dict]:
    txt = node_table["text"].astype(str).to_numpy(); rid = node_table["review_id"].to_numpy()
    reqs = []
    for p in pairs.itertuples(index=False):
        cid = f"pair-{p.relation}-{rid[p.u_node]}-{rid[p.v_node]}-0"
        reqs.append(make_request(cid, model, pair_system(p.relation), pair_user_message(txt[p.u_node], txt[p.v_node]), PAIR_SCHEMA))
    if n_retest:
        rng = np.random.default_rng(seed)
        for i in rng.choice(len(pairs), min(n_retest, len(pairs)), replace=False):
            p = pairs.iloc[i]
            cid = f"pair-{p.relation}-{rid[p.u_node]}-{rid[p.v_node]}-1"
            reqs.append(make_request(cid, model, pair_system(p.relation), pair_user_message(txt[p.v_node], txt[p.u_node]), PAIR_SCHEMA))
    return reqs


def row_from_result(custom_id: str, data: dict | None, status: str, model: str) -> dict:
    _, rel, u, v, swap = custom_id.split("-")
    d = data or {}
    return {"relation": rel, "u": int(u), "v": int(v), "swap": int(swap), "status": status, "model": model,
            "relative_fit": d.get("relative_fit", np.nan), "consistency": d.get("consistency", np.nan),
            "reason": d.get("reason", "")}
