"""Loader for the raw YelpChi release (Rayana & Akoglu 2015): metadata + review text.

Expected files (plain or .gz), tab-separated, one review per line:
    metadata       user_id  product_id  rating  label  date
    reviewContent  user_id  product_id  date    text
`label` in the release is -1 for filtered ("fake") and 1 for recommended; we map to 1 = fraud, 0 = genuine.

Text join (review issue #4): reviewContent lines are split on their first three tabs only, so tabs inside
the review text stay in the text; a line that cannot be parsed raises with its line number instead of
being dropped. Text is matched to metadata row-by-row when both files list the same (user, product, date)
in the same order, otherwise on (user, product, date, k-th occurrence of that key). Any metadata row left
without text raises. The join statistics are attached as `df.attrs["text_join"]` for the EDA / run log.
If the shipped files differ, change this file and data/README.md together.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
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


def read_review_content(path: Path) -> pd.DataFrame:
    """Parse reviewContent strictly: `user_id \t product_id \t date \t text`, text may contain tabs."""
    rows = []
    with _open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t", 3)
            if len(parts) < 3:
                raise ValueError(f"{path.name} line {i}: expected user, product, date, text; got {line[:80]!r}")
            rows.append((parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else ""))
    txt = pd.DataFrame(rows, columns=TEXT_COLS)
    txt["date"] = pd.to_datetime(txt["date"]).dt.normalize()
    return txt


def join_text(meta: pd.DataFrame, txt: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Text per metadata row (aligned to meta.index) and join statistics. Raises if any row is unmatched."""
    key = ["user_id", "product_id", "date"]
    if len(txt) == len(meta) and (txt[key].to_numpy() == meta[key].to_numpy()).all():
        text = pd.Series(txt["text"].to_numpy(), index=meta.index)
        method = "row_order_verified"
    else:
        m = meta[key].copy(); m["_k"] = m.groupby(key).cumcount(); m["_row"] = np.arange(len(m))
        t = txt.copy(); t["_k"] = t.groupby(key).cumcount()
        j = m.merge(t, on=key + ["_k"], how="left").sort_values("_row")
        text = pd.Series(j["text"].to_numpy(), index=meta.index)
        method = "key_occurrence"
    missing = int(text.isna().sum())
    stats = {"method": method, "n_meta": len(meta), "n_text_lines": len(txt), "n_matched": len(meta) - missing,
             "n_unmatched": missing, "n_empty_text": int((text.fillna("").str.strip() == "").sum())}
    if missing:
        raise ValueError(f"{missing} of {len(meta)} reviews have no text after the join ({stats}); "
                         "check the reviewContent layout before going on")
    return text, stats


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
        meta["text"], stats = join_text(meta, read_review_content(_find(raw_dir, "reviewContent")))
        meta.attrs["text_join"] = stats
    return meta
