# AGENTS.md — shared context for every coding agent on this repo

This file is read by Claude Code (via CLAUDE.md), Codex, Cursor and any other agent a team member
uses. Keep it short and current. If you change how the project works, change this file in the same PR.

## What this project is

NUS BT4012 Fraud Analytics, AY2026/27 Sem 1, **Group 05**: Lin Yushan Edison, Melody Goh Xueni,
Rohan Jaggi, Wong Yeow Kang. Due **20 Nov 2026 23:59**: `report05.pdf` (8–12 pages), `presentation05.pptx`,
`video05.mp4` (≤15 min), and this repo (README must reproduce results). Proposal: `docs/proposal.pdf`.
Guidelines + rubric: `docs/guidelines.pdf`.

**Setting.** Fraudsters camouflage by linking to legitimate accounts, so GNN neighbourhood averaging hides
them. The proposal planned to repair the graph with LLM text evidence. The YelpChi raw text never
arrived (**D18**); the full text track therefore ran on a YelpChi-sized slice of **YelpZip** (same
authors, labels and format, with text and real dates, **D19**, results **D20**), which is the headline
dataset of the report. The public `YelpChi.mat` graph under arrival waves (D11–D17) is the supporting
track. The claim, in four parts (numbers in `docs/decisions.md` D20 and D16/D17):

1. **Reconstruction pays a little, and we can say why.** On YelpZip the LLM-distilled scorer gives
   CatBoost +0.005 AUC-PR (interval excludes 0), nothing for the GNNs. Its fraud-centred edge AUC is
   0.50, below the ~0.6 bar the `.mat` noisy-oracle curve sets; the right yardstick is fraud-centred,
   not global, edge AUC (D15). Two datasets, one explanation.
2. **The label-free text-consistency monitor works on real drift.** It flags every degraded test window
   with 0 false alarms and latency 0 while the feature monitor flags none, and survives the same-window
   control. On `.mat` it alarms one wave after a 20 % injection and misses 5–10 %.
3. **Confidence-based self-training does collapse toward the normal class, and quotas fix it**
   (professor Q2): naive pseudo-labels end at 0.07 % (YelpZip) / 1.9 % (`.mat`) positives, per_class and
   prior_match hold the prior. On `.mat` the non-collapsing variants recover +0.03 over the control; on
   YelpZip's natural drift there is no headroom (oracle +0.018) and the promotion guard blocks ~90 % of
   challengers, which is the guard working.
4. **Camouflage that copies legitimate text and users defeats the label-free loop.** Injected recall
   ≈ 0, the monitor cannot separate the attack from natural drift, nothing is promoted; the oracle with
   the new labels reaches 0.78, so the labels an analyst provides must feed training, not only promotion.

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
- **Text track (D19):** YelpZip slice in `data/raw/yelpzip_raw/` (`--source raw`, tag `raw`): 71 restaurants,
  47,532 reviews, 10.2 % fraud, real dates 2005–2015, strictly temporal windows. The YelpChi raw text
  (Rayana & Akoglu 2015) itself was never obtainable. Never mix `.mat` and `raw` numbers in one table.
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
- Anthropic credentials live only in `.env` (`ANTHROPIC_API_KEY`) or in the owner's `ant auth login`
  profile; never in the repo. No LLM batch is submitted without the owner's go-ahead;
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

- 2026-11-18 (feat/report-tables) report support, no new experiments: `fig_scorer_bar_mat_waves.png` (D21, §4.1), `variables_raw.csv` (`03b_variables.py`, the rubric's variable table; PIT features plus the 9 LLM fields' effect sizes), `business_raw.csv` / `business_mat_lvl0.2.csv` (`14_business.py`, §4.5 operating point), tests for `homophily.py` incl. the dated-neighbour no-leak rule, `make lint` clean. 58 tests. Case study of the LLM pair judgements (`05d_pair_case_study.py`, D22): the judgement itself is at chance around fraud; detailed fakes read genuine; text separates single reviews (AUC 0.65) but not edges (§3.2, §5).

- 2026-11-16 (feat/yelpzip-text-track, D19–D20, PR #5) **text track complete on the YelpZip slice; experiments frozen; report framed on YelpZip** (`docs/report_outline.md`). `make raw-all` reproduces it from the committed LLM caches (`data/llm/*.parquet`; no API call needed). Headline tables: `stats_rewiring_raw.csv`, `drift_summary_raw*.csv`, `stats_selftrain_raw_*.csv`, `selftrain_rounds_raw_*.csv`, `adversarial_summary_raw.csv`, `adversarial_recovery_summary_raw.csv`. Never compare AUC-PR across injection levels (prevalence artefact; use AUROC and injected recall). Pending from Edison: 5-seed `.mat` tables and Gemma pair scores as a second-judge check.
- 2026-11-16 (fix/report-prep, after merging #3 and closing #2 as superseded): review findings every report writer must respect — (1) at noise 0.1 the injected nodes are near-copies of training fraud, so the **incumbent's AUC-PR does not drop under attack** (0.85–0.86 attacked vs 0.83–0.84 clean waves): the loop shows detect → retrain → promote → audit, not recovery from damage; a harder run (`11_graph_loop.py --noise 1.0`, outputs suffixed `_noise1.0`) is pending; (2) decompose the gain: incumbent 0.862 → control 0.902 → per_class 0.932 → oracle 0.973; (3) the plain feature monitor is at least as sensitive as the structural signal on this attack (10 %: 58 % vs 17 % flagged); (4) "alarm one wave late" = `k_consecutive: 2`; (5) keep D15's "optimistic guide" caveat next to the scorer curve. New figure `fig_graph_loop_rounds_*.png` (Q2).
- 2026-11-15 (feat/graph-waves) **experiments frozen; graph-only track final (D18).** Docs restated around the two-part claim; `make graph-all` reproduces everything.
- Results (waves, test AUC-PR): CatBoost 0.850, WeightedSAGE 0.698, CARE-static 0.683, AE 0.210 (5 seeds). Scorer curve: repair pays from edge AUC ~0.6–0.7; no label-free scorer gives a gain whose interval excludes 0 for both models (D15, D17).
- Loop at 20 % injection (D16, D17): 0 false alarms, alarm at wave 5; incumbent 0.862, control 0.902, per_class 0.932 (+0.030 [0.023, 0.037]), oracle 0.973; naive / balanced_loss collapse to 1.9 % / 6.0 % pseudo-positives.
- Open, optional: `09b_graph_adversarial.py` not yet run on `.mat`; the committed stratified `selftrain_*_mat` tables predate D9 (do not report them).
- 16–20 Nov: report, slides, video (`docs/report_outline.md`).
