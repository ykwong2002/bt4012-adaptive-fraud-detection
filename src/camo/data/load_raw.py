"""Loader for the raw YelpChi release (Rayana & Akoglu 2015): metadata + review text.

Expected files (plain or .gz), tab-separated, one review per line:
    metadata       user_id  product_id  rating  label  date
    reviewContent  user_id  product_id  date    text
`label` in the release is -1 for filtered ("fake") and 1 for recommended; we map to 1 = fraud, 0 = genuine.
Text is joined on (user_id, product_id, date); if that key is not unique we fall back to line order.
If the shipped files differ, change this file and data/README.md together.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

META_COLS = ["user_id", "product_id", "rating", "label_raw", "date"]
TEXT_COLS = ["user_id", "product_id", "date", "text"]


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" \
        else open(path, encoding="utf-8", errors="replace")


def _find(raw_dir: Path, stem: str) -> Path:
    for cand in (raw_dir / stem, raw_dir / f"{stem}.gz", raw_dir / f"{stem}.txt"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{stem} not found in {raw_dir}; see data/README.md")


def load_raw(raw_dir: str | Path, with_text: bool = True) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    with _open(_find(raw_dir, "metadata")) as f:
        meta = pd.read_csv(f, sep="\t", header=None, names=META_COLS, dtype=str)
    meta["rating"] = meta["rating"].astype(float)
    meta["date"] = pd.to_datetime(meta["date"]).dt.normalize()
    lab = meta["label_raw"].astype(int)
    meta["label"] = (lab == -1).astype(int) if (lab == -1).any() else (lab == 1).astype(int)
    meta = meta.drop(columns="label_raw")
    meta["review_id"] = range(len(meta))

    if with_text:
        with _open(_find(raw_dir, "reviewContent")) as f:
            txt = pd.read_csv(f, sep="\t", header=None, names=TEXT_COLS, dtype=str,
                              quoting=3, on_bad_lines="warn")
        txt["date"] = pd.to_datetime(txt["date"]).dt.normalize()
        key = ["user_id", "product_id", "date"]
        if len(txt) == len(meta) and not txt.duplicated(key).any() and not meta.duplicated(key).any():
            meta = meta.merge(txt, on=key, how="left")
        else:
            # line-order fallback, documented in data/README.md
            meta["text"] = txt["text"].values[: len(meta)] if len(txt) >= len(meta) else None
    return meta
