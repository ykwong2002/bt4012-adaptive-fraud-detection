# AGENTS.md — shared context for every coding agent on this repo

This file is read by Claude Code (via CLAUDE.md), Codex, Cursor and any other agent a team member
uses. Keep it short and current. If you change how the project works, change this file in the same PR.

## What this project is

NUS BT4012 Fraud Analytics, AY2026/27 Sem 1, **Group 05**: Lin Yushan Edison, Melody Goh Xueni,
Rohan Jaggi, Wong Yeow Kang. Due **20 Nov 2026 23:59**: `report05.pdf` (8–12 pages), `presentation05.pptx`,
`video05.mp4` (≤15 min), and this repo (README must reproduce results). Proposal: `docs/proposal.pdf`.
Guidelines + rubric: `docs/guidelines.pdf`.

**Setting.** Fraudsters camouflage by linking to legitimate accounts, so GNN neighbourhood averaging hides
them. The proposal planned to repair the graph with LLM text evidence; the raw text is no longer
obtainable, so the project runs on the public `YelpChi.mat` graph only (**D18**). The claim has two parts:

1. **When does structure repair pay?** A noisy-oracle edge scorer sets the bar (gain from edge AUC
   ~0.6–0.7; WeightedSAGE matches raw-graph CatBoost at ~0.85). The right yardstick is *fraud-centred*
   edge AUC, not global (D15). None of our label-free scorers (feature, cosine, R-U-R profile) clears
   the bar on `.mat`: a negative result with a usable threshold.
2. **Can a label-free monitor + label-scarce retraining respond to camouflage without collapsing to the
   normal class?** 0 false alarms on clean waves; 20 % injection alarmed one wave late; per_class /
   prior_match self-training recover +0.03 AUC-PR over a no-pseudo-label control; naive and
   balanced-loss self-training collapse toward legit (professor Q2). The monitor misses 5–10 % injection.

**We are graded on cleverness and justification, not on AUC.** Nothing off-the-shelf end to end; every
component is ours or a clearly acknowledged re-implementation of a comparator.

## Professor's comments on the proposal (verbatim; each must be answered in the report)

1. "Please provide more detail on EDA and data preparation, including how the raw review text and
   timestamps are aligned with the YelpChi graph."
2. "One question I have is whether confidence-based self-training could reinforce a bias towards the
   normal class, given the class imbalance. How would you check whether this happens and address it
   if it does?"
3. "In the final report, please provide sufficient detail and justification for the graph
   reconstruction and drift detection. Also, please make sure to address the practical implications,
   limitations, and reproducibility of your results."

## The headline experiment (must exist even if everything else slips)

On `.mat` under random arrival waves (D13; train 0–2, val 3, test 4–7, 5 seeds): raw graph vs
scorer-reweighted graph (softrank weights), WeightedSAGE + CatBoost, per-wave AUC-PR, with the
noisy-oracle curve as the reference and paired 95 % intervals (D17). Second pillar: the adaptive loop
(`11_graph_loop.py`, D16). Everything else supports these.

## Data (read `data/README.md` before touching data)

- **Used:** the graph file `YelpChi.mat` (CARE-GNN / DGL, `make data`): 45,954 review nodes, 32 features,
  labels (14.5 % fraud), relations R-U-R / R-S-R / R-T-R. No text, no dates, no ids, no index→review map.
- **Not obtainable:** the raw text (Rayana & Akoglu 2015). The text pipeline (`load_raw`, `04`, `05`,
  `07`–`09`) stays in the repo and runs on the synthetic fixture in `tests/fixtures/raw_mini`; it produces
  **no reported number**.
- `.mat` features are full-data aggregates; arrival waves are random, so they mimic arrival order without
  claiming real time. Say this in the report wherever a wave result appears.
- Raw data lives in `data/raw/` (git-ignored).

## Non-negotiables

- Every reported model number uses **arrival waves** (`--waves`, `src/camo/eval/waves.py`): an edge is
  visible only once both endpoints have arrived, degree features are scaled by visible nodes (D14), and
  waves are re-drawn per seed. Stratified `.mat` splits appear only as a labelled comparison.
- Edge scorers and the drift monitor never see fraud labels (labels are used for diagnosis only, e.g.
  `05c_edge_diagnostics.py`). Scorers are fit on training waves only.
- Every number and figure in the report is produced by a script in `scripts/` and lands in `outputs/`.
- Each script writes `outputs/runs/<name>/run.json` (git SHA, config hash, seed, wall time).
- Report uncertainty as paired seed-level 95 % intervals (`12_stats.py`), not seed std.
- External code consulted or adapted (CARE-GNN, DGL loader, Nash-Detect features) goes in
  `docs/acknowledgements.md`. Generative-AI use is acknowledged in the report per module rules.
- `ANTHROPIC_API_KEY` lives only in `.env`. No LLM batch is submitted without the owner's go-ahead;
  run `scripts/04_llm_score.py --dry-run` first and quote the cost.
- Never commit `data/raw/`, `.env`, or `outputs/runs/`.

## Repo map

```
src/camo/data        load_mat.py (Graph)  ·  text track: load_raw.py graph_build.py features_pit.py
src/camo/eval        waves.py (arrival waves) protocol.py (folds, n_visible) splits.py metrics.py runlog.py
src/camo/graph       feature_scorer.py rur_scorer.py edge_scorer.py rewire.py homophily.py embed.py
src/camo/models      wsage.py care_static.py catboost_model.py autoencoder.py runner.py
src/camo/drift       psi.py ks.py monitor.py (calibrated PSI/KS, D8)
src/camo/selftrain   loop.py (alarm-triggered retraining, D9) pseudo_labels.py bias_audit.py challenger.py
src/camo/adversarial graph_inject.py (graph-level camouflage, D13)  ·  text track: inject.py
src/camo/llm         text track only
scripts/             graph-only: 01 02 03 05b 05c 06 09b 10 11 12  ·  text track: 04 05 07 08 09
configs/             base.yaml reconstruction.yaml drift.yaml selftrain.yaml
tests/               pytest; fixtures/raw_mini is a synthetic raw-format sample
docs/                proposal, guidelines, decisions.md (D1–D18), report_outline.md, data_sources.md, acknowledgements.md
```

## How to work

GitHub: https://github.com/ykwong2002/bt4012-adaptive-fraud-detection (public; clone, branch, PR to `main`).

```
make setup        # uv sync (+dev)
make test         # pytest
make data         # download .mat (hash checked)
make graph-all    # every reported number and figure (graph-only track)
make all SMOKE=1  # every phase on a 2k-node subsample
```

- Python 3.11 via `uv`; run things as `uv run python scripts/…` or `uv run pytest`.
- Branch `feat/<area>`, PR to `main`, small commits under **your own GitHub identity** (the teaching
  team may inspect contribution history).
- Notebooks are stripped of outputs on commit (`nbstripout`); put real logic in `src/camo`, not notebooks.
- Heavy sweeps run on Colab via `notebooks/colab_runner.ipynb` using the same pinned `requirements.txt`.

## Ownership (edit as you like)

| Area | Owner |
|---|---|
| Data, arrival waves, EDA (report §2, §3.1) | TBD |
| Edge scorers + scorer curve (§3.2, §4.1) | TBD |
| Models + drift monitor (§3.3, §3.4) | TBD |
| Self-training, bias audit, camouflage injection (§3.5, §4.2) | TBD |

## Status / next steps (keep to ~5 lines, update in each PR)

- 2026-11-15 (feat/graph-waves) **experiments frozen; graph-only track final (D18).** Docs restated around the two-part claim; `make graph-all` reproduces everything.
- Results (waves, test AUC-PR): CatBoost 0.850, WeightedSAGE 0.698, CARE-static 0.683, AE 0.210 (5 seeds). Scorer curve: repair pays from edge AUC ~0.6–0.7; no label-free scorer gives a gain whose interval excludes 0 for both models (D15, D17).
- Loop at 20 % injection (D16, D17): 0 false alarms, alarm at wave 5; incumbent 0.862, control 0.902, per_class 0.932 (+0.030 [0.023, 0.037]), oracle 0.973; naive / balanced_loss collapse to 1.9 % / 6.0 % pseudo-positives.
- Open, optional: `09b_graph_adversarial.py` not yet run on `.mat`; the committed stratified `selftrain_*_mat` tables predate D9 (do not report them).
- 16–20 Nov: report, slides, video (`docs/report_outline.md`).
