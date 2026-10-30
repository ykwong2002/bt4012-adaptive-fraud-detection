"""Point-in-time (PIT) behavioural features for the rebuilt graph.

Every feature of review v uses only reviews dated <= date(v) (same-day reviews count as "at time t";
dates have day granularity). Adding reviews dated after v never changes v's features — tested in
tests/test_features_pit.py. Inspired by Rayana & Akoglu (2015) Table 2 but recomputed PIT and
deliberately limited to ~12 features (docs/decisions.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "u_n_prev", "u_mean_rating_prev", "u_rating_dev_prev", "u_days_since_prev", "u_sameday_n",
    "u_first_review", "p_n_prev", "p_mean_rating_prev", "rating_dev_p", "rating_extreme",
    "p_rate_30d", "text_len_log", "caps_ratio", "excl_count",
]


def _expanding_prev_mean(df: pd.DataFrame, key: str, col: str, fill: pd.Series) -> pd.Series:
    """Mean of `col` over the key's earlier rows (strictly earlier in sort order)."""
    g = df.groupby(key, sort=False)[col]
    csum = g.cumsum() - df[col]
    cnt = g.cumcount()
    out = csum / cnt.replace(0, np.nan)
    return out.fillna(fill)


def compute_pit_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs columns user_id, product_id, rating, date (datetime64), review_id, optional text.
    Returns a DataFrame aligned to df.index with FEATURES columns."""
    d = df[["user_id", "product_id", "rating", "date", "review_id"]].copy()
    d["_pos"] = np.arange(len(d))
    d = d.sort_values(["date", "review_id"], kind="stable")

    # global running mean rating (PIT), as a neutral fill for first-time users/products
    glob_prev = (d["rating"].cumsum() - d["rating"]) / pd.Series(np.arange(len(d)), index=d.index).replace(0, np.nan)
    glob_prev = glob_prev.fillna(d["rating"].iloc[0])

    f = pd.DataFrame(index=d.index)
    f["u_n_prev"] = d.groupby("user_id", sort=False).cumcount()
    f["u_mean_rating_prev"] = _expanding_prev_mean(d, "user_id", "rating", glob_prev)
    f["u_rating_dev_prev"] = (d["rating"] - f["u_mean_rating_prev"]).abs()
    prev_date = d.groupby("user_id", sort=False)["date"].shift(1)
    f["u_days_since_prev"] = (d["date"] - prev_date).dt.days.fillna(3650).clip(upper=3650)
    f["u_sameday_n"] = d.groupby(["user_id", "date"], sort=False)["rating"].transform("size")
    f["u_first_review"] = (f["u_n_prev"] == 0).astype(int)

    f["p_n_prev"] = d.groupby("product_id", sort=False).cumcount()
    f["p_mean_rating_prev"] = _expanding_prev_mean(d, "product_id", "rating", glob_prev)
    f["rating_dev_p"] = d["rating"] - f["p_mean_rating_prev"]
    f["rating_extreme"] = d["rating"].isin([1, 5]).astype(int)

    # product reviews in the previous 30 days (inclusive of same day)
    rate = np.zeros(len(d))
    for _, idx in d.groupby("product_id", sort=False).indices.items():
        days = d["date"].to_numpy()[idx].astype("datetime64[D]").astype(np.int64)
        lo = np.searchsorted(days, days - 30, side="left")
        hi = np.searchsorted(days, days, side="right")
        rate[idx] = hi - lo - 1
    f["p_rate_30d"] = rate

    if "text" in df.columns:
        t = df.loc[d.index, "text"].fillna("").astype(str)
        words = t.str.split().str.len()
        letters = t.str.count(r"[A-Za-z]").replace(0, np.nan)
        f["text_len_log"] = np.log1p(words)
        f["caps_ratio"] = (t.str.count(r"[A-Z]") / letters).fillna(0.0)
        f["excl_count"] = t.str.count("!")
    else:
        f["text_len_log"] = 0.0
        f["caps_ratio"] = 0.0
        f["excl_count"] = 0.0

    return f.loc[df.index, FEATURES].astype(np.float32)
