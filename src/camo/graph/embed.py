"""Local sentence embeddings (MiniLM) for every review, cached as .npy (shared as a release asset)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def embed_texts(texts: list[str], model_name: str, cache: Path | None = None, batch_size: int = 128) -> np.ndarray:
    if cache is not None and Path(cache).exists():
        e = np.load(cache)
        if e.shape[0] == len(texts):
            return e
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    e = model.encode([t if isinstance(t, str) else "" for t in texts], batch_size=batch_size,
                     normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, e)
    return e


def mock_embeddings(texts: list[str], dim: int = 32, seed: int = 0) -> np.ndarray:
    """Deterministic bag-of-hashed-words embedding for offline smoke runs (no model download)."""
    rng = np.random.default_rng(seed)
    table = rng.normal(size=(4096, dim)).astype(np.float32)
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in str(t).lower().split():
            out[i] += table[hash(w) % 4096]
    nrm = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out / nrm
