"""Generate a small synthetic dataset in the raw YelpChi file layout (metadata + reviewContent).
Deterministic. Used by tests and by --smoke runs when the real raw text is absent.
Run: uv run python tests/fixtures/make_raw_mini.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "raw_mini"
WORDS_GOOD = ["the", "staff", "were", "friendly", "our", "server", "remembered", "the", "lamb", "shank", "came", "with", "roasted", "garlic", "mash", "and", "the", "wait", "was", "twenty", "minutes", "on", "a", "friday"]
WORDS_FAKE = ["amazing", "best", "ever", "great", "place", "love", "it", "highly", "recommend", "wonderful", "excellent", "perfect", "fantastic", "must", "visit"]


def main(n_users=60, n_products=8, n_reviews=400, seed=0, big_product_reviews=120):
    rng = np.random.default_rng(seed)
    users = [f"u{i}" for i in range(n_users)]
    prods = [f"p{i}" for i in range(n_products)]
    # one product gets far more reviews so the >N filter has something to drop
    pw = np.ones(n_products); pw[0] = 6
    pw /= pw.sum()
    rows, texts = [], []
    base = np.datetime64("2010-01-01")
    for i in range(n_reviews):
        u = users[rng.integers(n_users)]
        p = prods[rng.choice(n_products, p=pw)]
        fraud = rng.random() < 0.15
        rating = int(rng.choice([4, 5, 5, 5]) if fraud else rng.choice([1, 2, 3, 4, 5]))
        date = base + np.timedelta64(int(rng.integers(0, 730)), "D")
        label = -1 if fraud else 1
        n_w = int(rng.integers(8, 30))
        vocab = WORDS_FAKE if fraud else WORDS_GOOD
        text = " ".join(rng.choice(vocab, n_w)) + ("!" * int(rng.integers(0, 3)) if fraud else ".")
        rows.append((u, p, rating, label, str(date)))
        texts.append((u, p, str(date), text))
    # make sure product 0 exceeds the fixture threshold
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "metadata", "w") as f:
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)
    with open(OUT / "reviewContent", "w") as f:
        f.writelines("\t".join(t) + "\n" for t in texts)
    print(f"wrote {len(rows)} reviews to {OUT}")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
