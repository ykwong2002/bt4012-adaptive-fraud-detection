# Adaptive Fraud Detection: Camouflage-Aware Graph Reconstruction under Concept Drift

NUS BT4012 Fraud Analytics, AY2026/27 Sem 1, **Group 05**
Lin Yushan Edison · Melody Goh Xueni · Rohan Jaggi · Wong Yeow Kang

Fraudsters camouflage by linking to legitimate accounts so that GNN neighbourhood averaging hides
them. On the public YelpChi graph (`YelpChi.mat`, CARE-GNN), with nodes arriving in random waves and
fraud nodes injected with camouflaging links, we ask two questions:

1. **How good must a label-free edge scorer be before repairing the graph pays?** A noisy-oracle curve
   sets the bar (gains from edge AUC ~0.6–0.7), fraud-centred edge AUC is the right yardstick, and none
   of our label-free scorers clears it on this graph.
2. **Can a label-free drift monitor and label-scarce self-training respond to camouflage without
   collapsing toward the normal class?** At 20 % injection: no false alarms on clean waves, an alarm one
   wave after the attack, and +0.03 AUC-PR over a retrained no-pseudo-label control for the
   class-balanced variants, while naive self-training's pseudo-labels collapse toward the legit class.

The proposal planned LLM text evidence from the raw reviews (Rayana & Akoglu 2015); that release is no
longer obtainable, so every reported result is graph-only (`docs/decisions.md` D18). The text pipeline
is kept and runs on a synthetic fixture.

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

`make graph-all` runs every step below in order (several hours on a laptop CPU; the scorer curve and
the loop dominate). Each step can also be run on its own:

| step | command | output |
|---|---|---|
| download + build the `.mat` graph | `make data` then `uv run python scripts/02_build_graph.py --source mat` | `data/processed/graph_mat.npz` |
| EDA: homophily by relation and class, degrees | `uv run python scripts/03_eda.py --source mat` | `outputs/tables/eda_*_mat.csv`, `outputs/figures/eda_*_mat.png` |
| baselines under arrival waves, 4 models × 5 seeds (D13, D14) | `uv run python scripts/06_train.py --source mat --waves` | `results_mat_waves_raw_graph*.csv` |
| scorer curve: noisy oracle, cosine, feature, R-U-R profile (D12, D15) | `make graph-scorers` | `results_mat_waves_*_softrank*.csv`, `scorer_curve_mat_waves.csv` |
| edge diagnostics: global vs fraud-centred AUC (D15) | `uv run python scripts/05c_edge_diagnostics.py --source mat` | `edge_auc_centred_mat.csv` |
| adaptive loop: monitor, alarm, self-training, bias audit (D16); paired intervals (D17) | `make graph-loop` | `graph_loop_*_mat_lvl*.csv`, `stats_rewiring_mat_waves.csv`, `stats_loop_mat_lvl0.2.csv` |
| all report figures | `uv run python scripts/10_figures.py` | `outputs/figures/fig_*_mat*.png` |

Tables land in `outputs/tables/`. Every script writes `outputs/runs/<name>/run.json` with the git SHA,
config hash, seed and wall time. `make all SMOKE=1` runs the pipeline on a 2k-node subsample in minutes.

Not part of the reported results: `scripts/09b_graph_adversarial.py` (static injection into test waves,
`make graph-adversarial`) and the text track (`scripts/04`, `05`, `07`–`09`, `--source raw`), which
needs the unavailable raw files; `--smoke --mock` exercises it on the fixture.

## Layout

See `AGENTS.md` → Repo map.
