"""Stage 1: one structured call per review -> data/llm/review_fields.parquet (keyed by review_id)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from camo.llm.batch_runner import make_request
from camo.llm.prompts import REVIEW_FIELDS, REVIEW_SCHEMA, REVIEW_SYSTEM, review_user_message


def build_review_requests(node_table: pd.DataFrame, review_ids, model: str) -> list[dict]:
    txt = node_table.set_index("review_id")["text"]
    return [make_request(f"rev-{rid}", model, REVIEW_SYSTEM, review_user_message(str(txt.loc[rid])), REVIEW_SCHEMA)
            for rid in review_ids]


def row_from_result(custom_id: str, data: dict | None, status: str, model: str) -> dict:
    rid = int(custom_id.split("-", 1)[1])
    row = {"review_id": rid, "status": status, "model": model}
    for f in REVIEW_FIELDS:
        row[f] = (data or {}).get(f, np.nan)
    return row


def fields_matrix(node_table: pd.DataFrame, cache_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Per-node LLM field matrix aligned to node_table rows; unscored rows get the column median and
    an `llm_unscored` indicator (so a refusal never silently becomes a zero)."""
    m = node_table[["review_id"]].merge(cache_df[["review_id", "status"] + REVIEW_FIELDS], on="review_id", how="left")
    ok = (m["status"] == "ok").to_numpy()
    cols = []
    for f in REVIEW_FIELDS:
        v = pd.to_numeric(m[f], errors="coerce").astype(float)
        v = v.where(pd.Series(ok, index=v.index), np.nan)
        cols.append(v.fillna(v.median() if ok.any() else 0.0).to_numpy())
    x = np.column_stack(cols + [(~ok).astype(float)]).astype(np.float32)
    return x, REVIEW_FIELDS + ["llm_unscored"]
