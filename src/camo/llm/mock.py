"""Offline stand-ins for the LLM stages so the whole pipeline runs without an API key (smoke / CI).
They are crude text heuristics and are NEVER used for reported results."""
from __future__ import annotations

import re

import numpy as np

GENERIC = {"amazing", "best", "great", "love", "recommend", "wonderful", "excellent", "perfect", "fantastic", "awesome"}


def mock_review_fields(text: str) -> dict:
    t = str(text); words = t.lower().split(); n = max(1, len(words))
    gen = sum(w.strip("!.,") in GENERIC for w in words) / n
    nums = len(re.findall(r"\d", t)); caps = sum(w.isupper() and len(w) > 1 for w in words)
    return {"specificity": float(min(1, 0.2 + 0.05 * nums + 0.01 * n)), "experiential_evidence": float(min(1, 0.02 * n)),
            "generic_praise": float(min(1, 3 * gen)), "promotional_tone": float(min(1, 2 * gen)),
            "first_person_detail": float(min(1, 0.3 * t.lower().count(" i ") + 0.1 * t.lower().count(" we "))),
            "named_concretes": int(nums), "exclamation_heavy": t.count("!") >= 2, "all_caps_words": caps >= 1,
            "template_like": gen > 0.3}


MOCK_NUMERIC = ["specificity", "experiential_evidence", "generic_praise", "promotional_tone", "first_person_detail"]


def mock_pair_score(text_a: str, text_b: str, relation: str, rng: np.random.Generator) -> dict:
    """Same target as prompt v2: similarity of the *kind* of evidence (1 = same kind)."""
    fa, fb = mock_review_fields(text_a), mock_review_fields(text_b)
    gap = np.mean([abs(fa[k] - fb[k]) for k in MOCK_NUMERIC])
    fit = float(np.clip(1 - 2 * gap + rng.normal(0, 0.05), 0, 1))
    return {"relative_fit": fit, "consistency": fit, "reason": "mock"}
