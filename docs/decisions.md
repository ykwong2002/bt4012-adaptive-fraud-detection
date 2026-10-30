# Design decisions log

Each entry: what we decided, why, what we rejected. Add to the top.

## D6 2026-10-30 Louvain runs on R-U-R ∪ scorer-weighted R-S-R, not on plain R-S-R
Plain R-S-R (same product & star) is a union of near-cliques, so Louvain just returns product×star
blocks and "community propensity" collapses to a per-product mean, smearing normal-class bias.

## D5 2026-10-30 Drift signal is text-homophily on the *raw* graph with date-respecting neighbours
On the refined graph the score distribution is stable by construction (we reweighted by it). On the
raw graph, a review's neighbourhood must only include neighbours dated ≤ the review, otherwise R-U-R
links to a user's future reviews leak the future into the monitor.

## D4 2026-10-30 Edge scoring is distilled, not one LLM call per edge
3.4M R-S-R edges make per-edge LLM calls unaffordable. We score each review once (structured fields),
embed each review locally, label ~15–20k sampled edges pairwise with a per-relation prompt, and fit a
small scorer that we apply to every edge. The scorer is supervised on LLM text judgements only, never
on fraud labels. Ablation: distilled scorer vs raw embedding cosine vs field-difference only.

## D3 2026-10-30 Equal-count temporal windows
YelpChi volume is late-skewed; calendar windows would starve early windows of fraud cases.
Windows of ~5.7k reviews (~800 fraud each) keep per-window AUC-PR meaningful.

## D2 2026-10-30 CARE-static instead of a full CARE-GNN re-implementation
The RL threshold search is a comparator, not our contribution, and would cost days. We keep the core
idea (label-aware neighbour filtering with a fixed threshold) and quote published CARE-GNN numbers.

## D1 2026-10-30 Rebuild the graph from raw metadata; never align by index
There is no public mapping from `.mat` node index to a raw review. The `.mat` graph is the raw set
with products > 800 reviews removed, so rebuilding from metadata with the published relation
definitions is reproducible and verifiable against the published counts. Pre-registered tolerance:
nodes ±2 %, products = 182, edges ±10 % per relation; residual reported, not hidden.

## Pivot (if raw text has not arrived by 8 Nov)
Graph-only track on `.mat`: the edge scorer becomes a label-free feature-consistency model (graph
autoencoder link plausibility on the 32 features); injection waves replace temporal windows; the
self-training bias audit is unchanged. Text code stays ready so arrival later is one command.
