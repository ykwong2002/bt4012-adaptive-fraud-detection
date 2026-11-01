# Data

Two YelpChi versions are used. Nothing under `data/raw/` is committed.

## 1. Graph file (public, auto-downloaded)

`data/raw/yelpchi_mat/YelpChi.mat` from the CARE-GNN repo (`data/YelpChi.zip`,
sha256 `3a31296b951e6c8158dddb783ff2c62709fa987d106bbe21e59f8e119053cec3`).
Run `make data` (or `uv run python scripts/01_download.py`).

Keys: `features` (45,954 × 32 sparse), `label` (1 × 45,954), `net_rur`, `net_rtr`, `net_rsr`
(45,954 × 45,954 sparse, symmetric), `homo` (union). No text, dates, ids or index→review mapping.
Published undirected edge counts: R-U-R 49,315 · R-T-R 573,616 · R-S-R 3,402,743.

## 2. Raw reviews with text and dates (by email from the dataset author)

Rayana & Akoglu, KDD 2015. Request from Shebuti Rayana (srayana@cs.stonybrook.edu), see the ODDS
page https://odds.cs.stonybrook.edu/yelpchi-dataset/. Expected: 67,395 reviews, 38,063 users,
201 hotels/restaurants, 13.23 % filtered ("fake").

Place the files here (names as shipped; the loader also accepts gzipped files):

```
data/raw/yelpchi_raw/metadata        # one line per review: user_id  product_id  rating  label  date
data/raw/yelpchi_raw/reviewContent   # one line per review: user_id  product_id  date  review_text
```

The loader splits each reviewContent line on its first three tabs only (review text may contain tabs),
matches text to metadata row by row when both files list the same (user, product, date) in the same
order, otherwise by (user, product, date, k-th occurrence), and **raises** if any line is malformed or any
review is left without text. The join statistics are printed by `scripts/02_build_graph.py` and saved in
its `run.json`.

`label` is `-1` (filtered / fake) or `1` (recommended / genuine) in the Rayana release; the loader maps
filtered → 1, genuine → 0 so that **1 = fraud** everywhere in this repo. If the shipped files use a
different layout, fix `src/camo/data/load_raw.py` and this README in the same PR.

The graph we train on is **rebuilt** from these files (`scripts/02_build_graph.py --source raw`):
drop products with more than 800 reviews (this is what produces the 45,954-node graph used by
CARE-GNN, per the GraphConsis paper), then

| relation | edge between two reviews when |
|---|---|
| R-U-R | same user |
| R-S-R | same product and same star rating |
| R-T-R | same product and same calendar month |

`scripts/03_eda.py` prints the rebuilt-vs-published comparison table.

## 3. LLM caches (committed)

`data/llm/review_fields.parquet` and `data/llm/pair_scores.parquet` hold Claude outputs keyed by
review id / edge id. They contain no review text. Never re-score what is cached.
