# AGENTS.md — shared context for every coding agent on this repo

This file is read by Claude Code (via CLAUDE.md), Codex, Cursor and any other agent a team member
uses. Keep it short and current. If you change how the project works, change this file in the same PR.

## What this project is

NUS BT4012 Fraud Analytics, AY2026/27 Sem 1, **Group 05**: Lin Yushan Edison, Melody Goh Xueni,
Rohan Jaggi, Wong Yeow Kang. Due **20 Nov 2026 23:59**: `report05.pdf` (8–12 pages), `presentation05.pptx`,
`video05.mp4` (≤15 min), and this repo (README must reproduce results). Proposal: `docs/proposal.pdf`.
Guidelines + rubric: `docs/guidelines.pdf`.

**Claim.** Fraudsters camouflage by linking to legitimate accounts so GNN neighbourhood averaging hides
them. If camouflage attacks graph structure, the defence should repair the structure: we use LLM-derived
text evidence to reweight or remove edges the text does not support (label-free), and because the
rewiring is label-free it doubles as a drift monitor. On trigger, a challenger is retrained with
pseudo-labels and promoted only if it beats the incumbent on recent held-out data.

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

Raw graph vs text-refined graph (soft weights), WeightedSAGE + CatBoost, strictly temporal split,
per-window AUC-PR, plus the per-relation homophily diagnostic. Everything else supports this.

## Data (read `data/README.md` before touching data)

- Two YelpChi versions. The **graph file** `YelpChi.mat` (CARE-GNN / DGL): 45,954 review nodes,
  32 features, labels, relations R-U-R / R-S-R / R-T-R. No text, no dates, no ids, no index→review map.
- The **raw text** (Rayana & Akoglu 2015): 67,395 reviews with `(user_id, product_id, rating, label, date)`
  and review text. Obtained by email from the author; **we rebuild the graph from it** (drop products
  with >800 reviews → 45,954 nodes) rather than align by index. See `docs/decisions.md`.
- `.mat` features are full-data aggregates: **never use them under a temporal split.**
- Raw data lives in `data/raw/` (git-ignored). LLM scoring caches in `data/llm/*.parquet` are
  **committed** (no text inside). Never re-score what is cached.

## Non-negotiables

- Temporal splits only for any reported model number; graph masking removes nodes dated after the
  cutoff (`src/camo/eval/splits.py`). Stratified splits are allowed only for the early bias-ablation
  on the `.mat` graph and must be labelled as such.
- Graph reconstruction never sees fraud labels.
- Every number and figure in the report is produced by a script in `scripts/` and lands in `outputs/`.
- Each script writes `outputs/runs/<name>/run.json` (git SHA, config hash, seed, wall time).
- External code consulted or adapted (CARE-GNN, DGL loader, Nash-Detect features) goes in
  `docs/acknowledgements.md`. Generative-AI use is acknowledged in the report per module rules.
- `ANTHROPIC_API_KEY` lives only in `.env`. No LLM batch is submitted without the owner's go-ahead;
  run `scripts/04_llm_score.py --dry-run` first and quote the cost.
- Never commit `data/raw/`, `.env`, or `outputs/runs/`.

## Repo map

```
src/camo/data       load_mat.py load_raw.py graph_build.py features_pit.py
src/camo/llm        prompts.py score_reviews.py score_pairs.py cache.py
src/camo/graph      homophily.py edge_scorer.py rewire.py
src/camo/models     wsage.py care_static.py catboost_model.py autoencoder.py
src/camo/drift      psi.py ks.py monitor.py
src/camo/selftrain  pseudo_labels.py bias_audit.py challenger.py
src/camo/adversarial inject.py
src/camo/eval       splits.py metrics.py runlog.py
scripts/            01_download … 10_figures (numbered, idempotent)
configs/            base.yaml reconstruction.yaml drift.yaml selftrain.yaml
tests/              pytest; fixtures/raw_mini is a synthetic raw-format sample
docs/               proposal, guidelines, decisions.md, data_sources.md, acknowledgements.md, report_outline.md
```

## How to work

```
make setup        # uv sync (+dev)
make test         # pytest
make data         # download .mat; verify raw files if present
make eda          # scripts/03_eda.py
make train        # scripts/06_train.py
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
| Data, rebuild, EDA (report §2, §3.1) | TBD |
| LLM scoring + graph reconstruction (§3.2) | TBD |
| Models + drift detection (§3.3, §4.1) | TBD |
| Self-training, bias audit, adversarial (§3.4, §4.2) | TBD |

## Status / next steps (keep to ~5 lines, update in each PR)

- 2026-10-30: full pipeline (scripts 01–10) runs end to end on the synthetic fixture in `--smoke --mock` mode; `.mat` baselines running; raw text requested from the dataset author (nothing text-dependent can be reported until it lands).
- Pivot date **8 Nov**: if raw text has not arrived, switch to the graph-only track (see `docs/decisions.md`).
- Experiment freeze **15 Nov**; 16–20 Nov report, slides, video.
