# Design decisions log

Each entry: what we decided, why, what we rejected. Add to the top.

## D10 2026-10-30 Adversarial runs keep the clean graph's windows and respect the product cap
Injected reviews only go to products with room under the 800-review filter (otherwise the filter
deletes the whole product, training reviews included), and windows reuse the clean graph's date
boundaries (recomputing equal-count windows on the bigger graph moved the training cutoff). Script 09
asserts the training set and its features are identical at every injection level.

## D9 2026-10-30 Self-training is alarm-triggered, time-respecting, and compared against a control
On temporal graphs the post-training windows are walked in order; at an alarm in window t the
challenger is built only from what exists at the end of t (graph, communities, pseudo-labels, probe
labels) and first used on t+1. Rejected: pooling all later windows at once, which let later reviews
leak into earlier ones and never connected the monitor to retraining. The incumbent, challengers and a
no-pseudo-label control share one learner and settings, so `pseudo_label_gain` measures what the
pseudo-labels add (before, the incumbent had a lower learning rate and the challenger always won).
EM label-shift estimates start from prior 0.5 because the models use balanced class weights; EM stays
a diagnostic and the training base rate remains the default prior.

## D8 2026-10-30 Drift thresholds are calibrated effect sizes, not p-values
With ~5.7k reviews per window, KS p < 0.01 flags every window. Signals use PSI and the KS statistic;
thresholds are 1.25 x the largest value seen when each training window is compared with the other
training windows (with floors PSI 0.1, KS 0.05). Features that grow with time by construction
(u_n_prev, p_n_prev, p_rate_30d, u_first_review, u_days_since_prev) are excluded from the monitor.

## D7 2026-10-30 Pair prompt v2: score "same kind of evidence", never penalise similarity
Prompt v1 gave low fit to templated, near-duplicate or coordinated pairs. Those are mostly fraud-fraud
edges, the few same-label edges a fraud review has on R-S-R / R-T-R, so v1 would have removed the edges
we want to keep. v2 scores whether both texts carry the same kind of evidence (both detailed first-hand,
or both generic/promotional); camouflage edges join opposite kinds and score low. Pair judgements are
cached with `prompt_version`; scripts/05 uses only the current version and prints a direction check
(cross-label edges must get lower weights than same-label edges, labels used for this check only).

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
