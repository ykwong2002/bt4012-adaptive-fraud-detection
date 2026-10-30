"""Camouflage injection for the adversarial experiment (proposal 3d).

Injected reviews are fraud nodes attached to *legitimate* users (so R-U-R links them to genuine
reviews, the camouflage) and to real products, dated inside the test windows, with text that is
either a perturbed real legitimate review from the training period (default) or an LLM-generated
review (texts supplied from data/llm/injected_texts.parquet). Train windows are never touched, so
any change in performance is caused by the injection alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENERIC = ["amazing", "great", "best", "wonderful", "excellent", "perfect", "fantastic", "awesome", "lovely", "superb"]


def perturb_text(text: str, rng: np.random.Generator, frac: float = 0.15) -> str:
    words = str(text).split()
    if not words:
        return "great place, highly recommend!"
    k = max(1, int(frac * len(words)))
    for i in rng.choice(len(words), min(k, len(words)), replace=False):
        words[i] = rng.choice(GENERIC)
    out = " ".join(words)
    return out + ("!" if rng.random() < 0.5 else "")


def inject_reviews(raw: pd.DataFrame, train_end, windows: list[tuple], level: float, seed: int,
                   mode: str = "perturbed", llm_texts: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """windows: list of (start_date, end_date, n_reviews) for the test windows. level = injected
    reviews as a fraction of each window's size. Returns (augmented raw, injected-rows info)."""
    rng = np.random.default_rng(seed)
    train = raw[raw["date"] <= train_end]
    fraud_users = set(train.loc[train["label"] == 1, "user_id"])
    legit_users = np.array(sorted(set(train.loc[train["label"] == 0, "user_id"]) - fraud_users))
    legit_texts = train.loc[train["label"] == 0, "text"].dropna().astype(str).to_numpy()
    legit_src_ids = train.loc[train["label"] == 0, "review_id"].to_numpy()[train.loc[train["label"] == 0, "text"].notna().to_numpy()]
    prods, pcount = np.unique(raw["product_id"], return_counts=True)
    next_id = int(raw["review_id"].max()) + 1
    rows = []
    llm_iter = iter(llm_texts["text"].tolist()) if (mode == "llm" and llm_texts is not None) else None
    for (start, end, n_w) in windows:
        n_inj = int(round(level * n_w))
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        for _ in range(n_inj):
            j = rng.integers(len(legit_texts))
            if llm_iter is not None:
                text = next(llm_iter, None) or perturb_text(legit_texts[j], rng)
            else:
                text = perturb_text(legit_texts[j], rng)
            rows.append({"user_id": legit_users[rng.integers(len(legit_users))],
                         "product_id": prods[rng.choice(len(prods), p=pcount / pcount.sum())],
                         "rating": float(rng.choice([4.0, 5.0, 5.0])),
                         "date": pd.Timestamp(start) + pd.Timedelta(days=int(rng.integers(0, max(1, days + 1)))),
                         "label": 1, "review_id": next_id, "text": text, "injected": 1,
                         "source_review_id": int(legit_src_ids[j])})
            next_id += 1
    inj = pd.DataFrame(rows)
    aug = raw.copy(); aug["injected"] = 0; aug["source_review_id"] = -1
    aug = pd.concat([aug, inj], ignore_index=True) if len(inj) else aug
    return aug, inj
