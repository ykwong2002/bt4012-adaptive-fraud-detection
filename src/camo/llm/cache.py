"""Parquet caches for LLM outputs (committed; contain no review text).
review_fields.parquet: one row per review_id. pair_scores.parquet: one row per (relation, u, v, swap)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetCache:
    def __init__(self, path: str | Path, key: list[str]):
        self.path, self.key = Path(path), key
        self.df = pd.read_parquet(self.path) if self.path.exists() else pd.DataFrame(columns=key)

    def have(self) -> set:
        return set(map(tuple, self.df[self.key].itertuples(index=False, name=None))) if len(self.df) else set()

    def missing(self, keys: list[tuple]) -> list[tuple]:
        h = self.have()
        return [k for k in keys if tuple(k) not in h]

    def add(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        new = pd.DataFrame(rows)
        self.df = pd.concat([self.df, new], ignore_index=True).drop_duplicates(self.key, keep="last")
        return len(new)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(self.path, index=False)
        return self.path
