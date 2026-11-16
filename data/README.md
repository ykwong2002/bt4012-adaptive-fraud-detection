# Data

Two kinds of input are supported: the public YelpChi graph file (no text, no dates) and a Rayana-format
raw release with review text and dates. The YelpChi raw release never arrived; since 2026-11-16 the raw
track runs on **YelpZip** (same authors, same labels, same format; decisions D19). Nothing under
`data/raw/` is committed.

## 1. Graph file (public, auto-downloaded)

`data/raw/yelpchi_mat/YelpChi.mat` from the CARE-GNN repo (`data/YelpChi.zip`,
sha256 `3a31296b951e6c8158dddb783ff2c62709fa987d106bbe21e59f8e119053cec3`).
Run `make data` (or `uv run python scripts/01_download.py`).

Keys: `features` (45,954 × 32 sparse), `label` (1 × 45,954), `net_rur`, `net_rtr`, `net_rsr`
(45,954 × 45,954 sparse, symmetric), `homo` (union). No text, dates, ids or index→review mapping.
Published undirected edge counts: R-U-R 49,315 · R-T-R 573,616 · R-S-R 3,402,743.

## 2. Raw reviews with text and dates (Rayana & Akoglu, KDD 2015)

Three releases share one format (`metadata` + `reviewContent`, tab-separated, one review per line):
YelpChi (67,395 reviews, Chicago hotels + restaurants), YelpNYC (359,052) and YelpZip (608,598
reviews, 260,277 users, 5,044 restaurants, 13.2 % filtered, 2004-10 → 2015-01). The ODDS pages
(https://odds.cs.stonybrook.edu/yelpchi-dataset/ and `.../yelpzip-dataset/`) ask for an email request;
the YelpChi request (Nov 2026) was never answered. **YelpZip** is also mirrored publicly by projects
that cite the release (Google Drive folder `16A3lqy53BdCr-K70NnOgb205SruxIcub`, files `metadata`,
`reviewContent`, `reviewGraph`, `readmeYelp.txt`); that copy is what we use. Cite Rayana & Akoglu 2015
and the ODDS page, as `readmeYelp.txt` asks.

Place the files here (names as shipped; the loader also accepts gzipped files):

```
data/raw/yelpzip_raw/metadata        # one line per review: user_id  product_id  rating  label  date
data/raw/yelpzip_raw/reviewContent   # one line per review: user_id  product_id  date  review_text
```

(`paths.raw_dir` in `configs/base.yaml`; a YelpChi release would go in the same place with
`graph.raw_release: yelpchi`, which re-enables the rebuilt-vs-published check.)

**YelpChi-sized slice (D19).** YelpZip is too large for the R-S-R relation (one restaurant has 7,378
reviews). `scripts/02_build_graph.py --source raw` therefore keeps, after the 800-review cap, whole
restaurants from the largest down until `graph.raw_max_reviews` (48,000) reviews: 71 restaurants,
47,532 reviews, 35,812 users, 10.2 % fraud, 2005-03 → 2015-01, edges R-U-R 25,119 · R-T-R 290,324 ·
R-S-R 5,260,757. Point-in-time features are computed on the whole release before slicing, so a review's
user and product history is complete. `--max-reviews 0` keeps everything.

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
