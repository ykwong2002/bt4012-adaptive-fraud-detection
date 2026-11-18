# Adaptive Fraud Detection: Camouflage-Aware Graph Reconstruction under Concept Drift

NUS BT4012 Fraud Analytics, AY2026/27 Sem 1, **Group 05**
Lin Yushan Edison · Melody Goh Xueni · Rohan Jaggi · Wong Yeow Kang

Fake reviews hide in a review graph: most of their neighbours are genuine, so a graph model that
averages over neighbours makes them look genuine (camouflage). On real review text and dates we test
whether an LLM can repair the graph, whether text can warn of drift without labels, whether
label-scarce self-training collapses toward the genuine class, and what a deliberate camouflage attack
defeats.

**Data.** Headline: a 71-restaurant slice of YelpZip (Rayana & Akoglu 2015; 47,532 reviews, 10.2 %
filtered as fraud) with real text and dates, split into eight equal-count time windows. Supporting:
the public YelpChi graph (`YelpChi.mat`, CARE-GNN), which has no text or dates, under random arrival
waves. The proposal's YelpChi raw text release could not be obtained, so the text track runs on its
sibling release YelpZip (`docs/decisions.md` D18-D20).

**Findings** (report Section 6.1; numbers in `outputs/tables/`):

1. Re-weighting edges with a distilled LLM scorer gives CatBoost a small gain that holds across seeds
   (+0.005 AUC-PR) and none for the GNNs, because the scorer is at chance around fraud reviews
   (fraud-centred edge AUC 0.44-0.50); simulated scorers on `YelpChi.mat` put the bar at about 0.6.
2. A label-free text-consistency monitor alarms in all four degraded test windows (from window 5 once
   graph growth is controlled for), but the one healthy window we could check was also flagged, and a
   monitor on the LLM fields alone would have fired too (D23).
3. Naive confidence-based self-training collapses toward the genuine class (0.07 % fraud pseudo-labels
   against a 10.5 % prior); fixing the fraud share at the prior prevents it. On YelpZip no variant beats
   a no-pseudo-label control, since even true labels add only +0.018 AUC-PR.
4. Injected camouflage (edited genuine text posted from established genuine accounts) defeats every
   label-free component; a model trained with the injected reviews' labels catches it.

Repo: https://github.com/ykwong2002/bt4012-adaptive-fraud-detection

Proposal: `docs/proposal.pdf`. Design log: `docs/decisions.md`. Agent/team context: `AGENTS.md`.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once
make setup                                           # uv sync + requirements.txt export
make data                                            # downloads the public YelpChi.mat
make test
```

On a corporate network with TLS interception, prefix uv calls with `UV_SYSTEM_CERTS=1`.
No API key is needed for any reported result.

## Reproduce

Two tracks. **YelpZip text track** (headline, decisions D19–D20): `make raw-all` runs every step below
in order from the committed LLM caches (`data/llm/*.parquet`; no API call is made unless the cache is
incomplete), about 3 h on a laptop CPU. Needs the raw release under `data/raw/yelpzip_raw/`
(`data/README.md`).

| step | command | output |
|---|---|---|
| build the YelpZip slice (strict text join, point-in-time features, 48k-review slice) | `uv run python scripts/02_build_graph.py --source raw` | `data/processed/graph_raw.npz` |
| EDA | `uv run python scripts/03_eda.py --source raw` | `eda_*_raw.csv`, `outputs/figures/eda_*_raw.png` |
| variable summary: type, meaning, use, leakage control, legit-vs-fraud effect size | `uv run python scripts/03b_variables.py` | `variables_raw.csv` |
| LLM review fields + pair judgements (cached; `--dry-run` quotes the cost of anything missing) | `uv run python scripts/04_llm_score.py --source raw --collect` | `data/llm/*.parquet` |
| distilled edge scorer + diagnostics (D4, D15) | `uv run python scripts/05_distil_edges.py --source raw`, `05c_edge_diagnostics.py --source raw` | `distil_metrics_raw.csv`, `edge_auc_centred_raw.csv`, `data/processed/edge_scores_raw.npz` |
| why the text scorer is at chance around fraud: LLM pair judgements by pair type (D22; example texts go to git-ignored `data/processed/`) | `uv run python scripts/05d_pair_case_study.py` | `pair_case_study_*_raw.csv` |
| 4 models × 5 seeds, raw graph and soft / softrank / hard refined graphs | `uv run python scripts/06_train.py --source raw [--weights data/processed/edge_scores_raw.npz --mode soft\|softrank\|hard --tau 0.5]` | `results_raw_*.csv` |
| drift monitor (+ same-window control) | `uv run python scripts/07_drift.py --source raw [--same-window]` | `drift_*_raw*.csv` |
| alarm-triggered self-training + bias audit, raw and refined graph | `uv run python scripts/08_selftrain.py --source raw [--weights data/processed/edge_scores_raw.npz]` | `selftrain_*_raw_*.csv` |
| camouflage injection with recovery | `uv run python scripts/09_adversarial.py --source raw --recover` | `adversarial_*_raw.csv` |
| paired intervals; figures | `uv run python scripts/12_stats.py --tag raw`, `10_figures.py --tag raw`, `13_process_figure.py` | `stats_*_raw*.csv`, `outputs/figures/fig_*_raw*.png`, `fig_process.png` |
| operating point per window: reviews flagged, fraud caught, reviews checked per fraud found | `uv run python scripts/14_business.py` | `business_raw.csv` |

**YelpChi graph-only track** (supporting, D11–D17): `make graph-all` runs every step below in order
(several hours on a laptop CPU; the scorer curve and the loop dominate). Each step can also be run on
its own:

| step | command | output |
|---|---|---|
| download + build the `.mat` graph | `make data` then `uv run python scripts/02_build_graph.py --source mat` | `data/processed/graph_mat.npz` |
| EDA: homophily by relation and class, degrees | `uv run python scripts/03_eda.py --source mat` | `outputs/tables/eda_*_mat.csv`, `outputs/figures/eda_*_mat.png` |
| baselines under arrival waves, 4 models × 5 seeds (D13, D14) | `uv run python scripts/06_train.py --source mat --waves` | `results_mat_waves_raw_graph*.csv` |
| scorer curve: noisy oracle, cosine, feature, R-U-R profile (D12, D15) | `make graph-scorers` | `results_mat_waves_*_softrank*.csv`, `scorer_curve_mat_waves.csv` |
| edge diagnostics: global vs fraud-centred AUC (D15) | `uv run python scripts/05c_edge_diagnostics.py --source mat` | `edge_auc_centred_mat.csv` |
| adaptive loop: monitor, alarm, self-training, bias audit (D16); paired intervals (D17); operating point | `make graph-loop` | `graph_loop_*_mat_lvl*.csv`, `stats_rewiring_mat_waves.csv`, `stats_loop_mat_lvl0.2.csv`, `business_mat_lvl0.2.csv` |
| all report figures, incl. the scorer bar: gain vs fraud-centred edge AUC (D21) | `uv run python scripts/10_figures.py` | `outputs/figures/fig_*_mat*.png`, `scorer_bar_mat_waves.csv` |

Tables land in `outputs/tables/`. Every script writes `outputs/runs/<name>/run.json` with the git SHA,
config hash, seed and wall time. `make all SMOKE=1` runs the pipeline on a 2k-node subsample in minutes.

Not part of the reported results: `scripts/09b_graph_adversarial.py` (static injection into test waves,
`make graph-adversarial`). `uv run python scripts/04_llm_score.py --mock` exercises the text track's LLM step on the synthetic
fixture without the raw files or an API key.

## Where each report table and figure comes from

| report item | output | script |
|---|---|---|
| Table 1 (dataset summary) | `eda_summary_raw.csv`, `eda_summary_mat.csv` | `03_eda.py` |
| Table 2 (raw vs refined graph) | `results_raw_*_summary.csv`, `stats_rewiring_raw.csv` | `06_train.py`, `12_stats.py` |
| Table 3 (edge scorer quality) | `edge_auc_centred_raw.csv` | `05c_edge_diagnostics.py` |
| Section 5.1 pair analysis | `distil_metrics_raw.csv`, `pair_case_study_*_raw.csv` | `05_distil_edges.py`, `05d_pair_case_study.py` |
| Section 5.2 drift numbers | `drift_*_raw.csv`, `drift_*_raw_samewin.csv` | `07_drift.py` |
| Table 4 (self-training bias audit) | `selftrain_*_raw_raw_graph.csv`, `stats_selftrain_raw_raw_graph.csv` | `08_selftrain.py`, `12_stats.py` |
| Section 5.4 injection and recovery | `adversarial_*_raw.csv` | `09_adversarial.py` |
| Section 5.5 operating point | `business_raw.csv` | `14_business.py` |
| Table A1 (scorer curve) | `stats_rewiring_mat_waves.csv`, `scorer_curve_mat_waves.csv` | `make graph-scorers`, `12_stats.py` |
| Table A2 (loop on `YelpChi.mat`) | `graph_loop_*_mat_lvl0.2.csv`, `stats_loop_mat_lvl0.2.csv` | `11_graph_loop.py`, `12_stats.py` |
| Appendix B (degradation label, AUROC check) | `results_raw_raw_graph_summary.csv` | `06_train.py` |
| Table D1 (variables) | `eda_features_raw.csv`, `variables_raw.csv` | `03_eda.py`, `03b_variables.py` |
| Table D2 (time windows) | `eda_windows_raw.csv` | `03_eda.py` |
| Figure 1 (camouflage illustration) | drawn diagram, not generated | - |
| Figure 2 (homophily) | `outputs/figures/eda_homophily_raw.png` | `03_eda.py` |
| Figure 3 (system architecture) | drawn diagram, not generated | - |
| Figure 4 (AUC-PR per window) | `outputs/figures/fig_headline_raw.png` | `10_figures.py` |
| Figure 5 (drift monitor) | `outputs/figures/fig_drift_raw.png` | `10_figures.py` |
| Figure 6 (injection) | `outputs/figures/fig_adversarial_raw.png` | `10_figures.py` |
| Figure 7 (deployment loop) | drawn diagram, not generated (an earlier version is `outputs/figures/fig_process.png`) | `13_process_figure.py` for the earlier version |
| Figure A1 (scorer curve) | `outputs/figures/fig_scorer_bar_mat_waves.png` | `10_figures.py` |
| Figure A2 (bias audit per round) | `outputs/figures/fig_graph_loop_rounds_mat_lvl0.2.png` | `10_figures.py` |

Tables are in `outputs/tables/` unless a path is given.

## Layout

See `AGENTS.md` → Repo map.
