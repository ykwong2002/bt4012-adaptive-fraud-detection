# Design decisions log

Each entry: what we decided, why, what we rejected. Add to the top.

## D15 2026-11-15 R-U-R user-profile scorer, and why global edge AUC is the wrong yardstick
Idea: R-U-R is the one relation camouflage does not break (93 % fraud homophily), so describe each
review by its user profile (mean of its own and its R-U-R neighbours' quantile-normalised features)
and score an R-S-R / R-T-R edge by profile distance (`camo.graph.rur_scorer`, `05b --scorer
rur_profile`). Label-free and time-respecting: each edge is scored when it appears, from R-U-R
neighbours visible by then; the transform and scale are fitted on training-wave nodes. Chosen among
six variants by edge AUC on training-wave edges only (test labels never used).
Result (.mat, waves, 3 seeds): global edge AUC 0.686, close to the noisy oracle's 0.7, but
WeightedSAGE +0.010 and CatBoost -0.012, where the simulated 0.7 scorer gives +0.05. Diagnosis
(`05c_edge_diagnostics.py`, `edge_auc_centred_mat.csv`): the 0.686 comes from legit-legit edges
(legit-centred AUC 0.69, ~77 % of edges); on edges touching a fraud node it is at chance (0.48-0.51),
because most fraud reviews have no R-U-R neighbours, so their profiles are unsmoothed and far from
everything; fraud-fraud and fraud-legit edges get the same low weight (rank 0.34 vs 0.36). Cosine and
the link-prediction scorer are the reverse: 0.52-0.54 globally but 0.61-0.67 fraud-centred.
Consequences: (1) we report fraud-centred and legit-centred AUC next to global AUC for every scorer;
(2) the noisy-oracle curve (D12) assumes errors independent of the node, which real scorers violate,
so its x-axis is an optimistic guide, not a prediction; (3) dividing the distance by its expected
noise (sqrt(1/(k_u+1) + 1/(k_v+1))) balances the two sides at ~0.58-0.60 but does not beat cosine
around fraud nodes, so we stop the scorer search here. On .mat, no label-free scorer we tried
reaches the ~0.7 fraud-centred quality where repair starts to pay.

## D14 2026-11-15 Degree features are rescaled by the number of visible nodes
Under the wave protocol (D13) an edge is visible only once both endpoints have arrived, so the
training graph holds about (3/8)^2 = 14 % of the edges and the graph keeps growing through the test
waves. CatBoost's per-relation weighted degree learned on the sparse training graph therefore meant
something different on every later wave (on training nodes, mean R-S-R degree 12.0 in the training
graph vs 32.2 in the full graph). Degree is now divided by the number of nodes visible at the cutoff
(reported per 10,000 visible nodes; `catboost_model.neighbour_aggregates`, `Fold.n_visible`). Under
random arrival each neighbour is visible with probability n_visible / n, so this is an unbiased
estimate of the full-graph degree rate (32.1 vs 32.2 after rescaling). Neighbour means need no
correction. Check on .mat (waves, 2 seeds, CatBoost): test AUC-PR 0.844 -> 0.850, and the decline
across test waves (0.849 -> 0.841) disappears (0.853 -> 0.850). The same rescaling applies to dated
graphs, where it removes the same train/val mismatch; undated (stratified) runs are unchanged up to a
constant. Rejected: dropping degree (it carries signal) and full-graph transductive training (it
would let the incumbent see edges to nodes that have not arrived yet).

## D13 2026-11-10 Graph-only protocol: random arrival waves plus graph-level camouflage injection
The .mat graph has no dates, so `build_folds` fell back to a stratified split, which we may not report,
and script 09 needs the raw metadata to inject reviews. On the graph-only track each node instead gets
a random arrival wave (stratified by label, re-drawn per seed; `camo.eval.waves`) encoded as a
pseudo-date. The existing temporal code then runs unchanged (protocol name "waves"): train on waves
0-2 with later nodes and edges masked, tune on wave 3, score each test wave on the graph as it stands at
its end. Clean waves are i.i.d., so this is a null-drift baseline. Drift comes from
`camo.adversarial.graph_inject`: camouflaged fraud nodes (a training fraud node's features plus noise
and degrees) are appended to the test waves, linked to a hijacked legitimate user's reviews (R-U-R)
and to legitimate slices of one product block (R-S-R, R-T-R), only to nodes that have already arrived.
Script 09b asserts the training nodes and training graph are identical at every level and reports per
wave AUC-PR (all nodes and clean nodes) and injected-node recall@10%. AUC-PR on all nodes rises with
the injected share of fraud, so across levels only clean-node AUC-PR, AUROC and injected recall are
compared.
Rejected: keeping the stratified split (not reportable), and injecting edges without nodes (no
injected fraud to detect). Limitations, stated in the report: the arrival order is synthetic, so this
measures robustness to camouflage, not real temporal drift; .mat features are full-data aggregates;
injected nodes are not linked to each other (no campaign structure).

## D12 2026-11-10 Scorer-quality curve with a noisy oracle; compare scorers by rank, not raw score
The perfect oracle reaches AUC-PR ~1.0 because pruning every cross-label edge encodes the labels in
the graph, so it bounds nothing useful. Instead we simulate scorers of known quality (edge AUC for
same- vs cross-label edges, binormal noise) and retrain at each level, then place the real scorers on
the same axis. All scorers use `softrank` weights: the feature scorer's raw outputs sit in a narrow
band, so plain soft weights would hide its ordering. Limitation: simulated errors are random, real
ones are not, so the curve is optimistic.

## D11 2026-11-09 Graph-only edge scorer: MLP encoder link model, not a GNN autoencoder
Implements the pivot's "feature-consistency model" (scripts/05b). Each node is encoded from its own
32 features only; a per-relation bilinear decoder is trained to separate observed edges from random
pairs, and model selection uses held-out link AUC, so no labels are used. Rejected: a GCN encoder,
which mixes in neighbour features and so hides the camouflage the score is meant to expose. A label
oracle (keep same-label edges only) is run as the ceiling for any rewiring, and a feature-cosine
scorer as the trivial baseline.

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
