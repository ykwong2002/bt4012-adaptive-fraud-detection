# Adaptive Fraud Detection: Camouflage-Aware Graph Reconstruction under Concept Drift

NUS BT4012 Fraud Analytics, AY2026/27 Sem 1, **Group 05**
Lin Yushan Edison · Melody Goh Xueni · Rohan Jaggi · Wong Yeow Kang

Fraudsters camouflage by linking to legitimate accounts so that GNN neighbourhood averaging hides
them. We repair the graph instead of patching the classifier: an LLM scores review text once, a
distilled scorer reweights every edge the text does not support (label-free), the same signal is used
as an early-warning drift monitor, and on trigger a challenger is retrained from pseudo-labels with an
explicit audit for normal-class bias. Dataset: YelpChi (graph version from CARE-GNN; raw text and
dates from Rayana & Akoglu 2015, rebuilt into the same graph).

Proposal: `docs/proposal.pdf`. Design log: `docs/decisions.md`. Agent/team context: `AGENTS.md`.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once
make setup                                           # uv sync + requirements.txt export
cp .env.example .env                                 # add ANTHROPIC_API_KEY (only needed for scripts/04)
make data                                            # downloads the public YelpChi.mat
make test
```

On a corporate network with TLS interception, prefix uv calls with `UV_SYSTEM_CERTS=1`.

Raw review text is obtained by email from the dataset author; see `data/README.md`.

## Reproduce

| step | command | output |
|---|---|---|
| build graph (rebuilt from raw, or `.mat`) | `uv run python scripts/02_build_graph.py --source raw` | `data/processed/graph_raw.npz` |
| EDA + published-vs-rebuilt table | `uv run python scripts/03_eda.py` | `outputs/tables/eda_*.csv`, `outputs/figures/eda_*.png` |
| LLM scoring (dry run first) | `uv run python scripts/04_llm_score.py --dry-run` | cost estimate; then `data/llm/*.parquet` |
| distil edge scorer + rewire | `uv run python scripts/05_distil_edges.py` | `data/processed/edge_scores.npz` |
| train raw vs refined, 5 seeds | `uv run python scripts/06_train.py` | `outputs/tables/results_*.csv` |
| drift monitors | `uv run python scripts/07_drift.py` | `outputs/tables/drift_*.csv` |
| self-training + bias audit | `uv run python scripts/08_selftrain.py` | `outputs/tables/selftrain_*.csv` |
| adversarial injection | `uv run python scripts/09_adversarial.py` | `outputs/tables/adversarial_*.csv` |
| all report figures | `uv run python scripts/10_figures.py` | `outputs/figures/` |

`make all SMOKE=1` runs every step on a 2k-node subsample (minutes on a laptop). Every script writes
`outputs/runs/<name>/run.json` with the git SHA, config hash, seed and wall time.

## Layout

See `AGENTS.md` → Repo map.
