# Report outline → scripts and figures (report05.pdf, 8–12 pages)

| § | Content | Produced by | Artefact |
|---|---|---|---|
| 1 Problem | camouflage attacks homophily; drift confirmed late; repair structure not classifier | – | `docs/proposal.pdf` |
| 2 Data | two YelpChi versions; rebuild from raw; **rebuilt-vs-published table**; variable table (PIT features + LLM fields); fraud per window | `02_build_graph.py`, `03_eda.py` | `eda_rebuilt_vs_published.csv`, `eda_windows_raw.csv`, `eda_features_raw.csv` |
| 2 EDA | homophily by relation and class (camouflage diagnostic); degree CCDFs | `03_eda.py` | `eda_homophily_raw.png`, `eda_degree_raw.png` |
| 3.1 Pre-processing | >800 filter, relation builder, point-in-time features (no-leak test), temporal windows + graph masking | `src/camo/data`, `src/camo/eval/splits.py`, `tests/` | `docs/decisions.md` D1, D3 |
| 3.2 Reconstruction | per-review structured scoring, per-relation pairwise prompts, distilled ridge scorer, soft/hard rewiring; ablations vs cosine / fields; LLM test-retest ceiling; cost | `04_llm_score.py`, `05_distil_edges.py` | `distil_metrics_raw.csv`, `fig_distil_raw.png`, `tau_sweep_raw.csv`, `fig_tau_sweep_raw.png`, `outputs/runs/04_*/run.json` (cost) |
| 3.3 Models | WeightedSAGE (ours), CARE-static (comparator), CatBoost, AE; why each | `src/camo/models` | – |
| 3.4 Drift | PSI/KS effect sizes on features, LLM fields, raw-graph text-homophily with dated neighbours; thresholds calibrated across training windows (D8); trigger rule; latency | `07_drift.py` | `drift_monitor_raw.csv`, `drift_thresholds_raw.csv`, `fig_drift_raw.png` |
| 3.5 Label-scarce response | alarm-triggered, time-respecting retraining (D9); pseudo-labels (community propensity on R-U-R ∪ weighted R-S-R, AE rank), self-training variants, **bias audit** (prof. Q2), promotion rule; deployed vs incumbent vs no-pseudo-label control vs oracle per window | `08_selftrain.py` | `selftrain_windows_raw_*.csv`, `selftrain_summary_raw_*.csv`, `fig_selftrain_raw_*.png` |
| 4.1 Headline | raw vs refined per window, 4 models, 5 seeds | `06_train.py` (raw, soft, hard τ) | `results_raw_*_summary.csv`, `fig_headline_raw.png` |
| 4.1b Headline, graph-only | raw vs rewired on `.mat`, arrival waves (D13) | `06_train.py --source mat --waves` | `results_mat_waves_*_summary.csv`, `fig_headline_mat_waves.png` |
| 3.2b Edge scorers, graph-only | feature, cosine, R-U-R profile vs noisy-oracle curve; global vs fraud-centred edge AUC (D12, D15) | `make graph-scorers`, `05c_edge_diagnostics.py` | `scorer_curve_mat_waves.csv`, `edge_auc_centred_mat.csv`, `fig_scorer_curve_mat_waves.png` |
| 4.2b Adversarial, graph-only | graph-level camouflage injection into test waves; AUC-PR (all / clean), injected recall | `09b_graph_adversarial.py --source mat` | `graph_adversarial_summary_mat.csv`, `fig_graph_adversarial_mat.png` |
| 4.2 Adversarial | injection levels, degradation raw vs refined, alarm latency, recovery | `09_adversarial.py` | `adversarial_summary_raw.csv`, `fig_adversarial_raw.png` |
| 4.3 Business integration | monitor → trigger → challenger → promotion loop; label budget of 100/window; cost per graph | – | from run logs |
| 5 Limitations | scoring errors propagate (τ sweep); unspecific sincere reviewers (manual review of top false positives); label is as-of-crawl; injection text style; natural drift may be weak | – | – |
| 5 Reproducibility | README reproduce table; `run.json` per script (git SHA, config hash, seed, wall time); committed LLM caches | all | `outputs/runs/*/run.json` |

Answers to the professor's comments are in §2 (alignment), §3.5 + §4 (self-training bias) and §3.2/§3.4/§5 (justification, implications, limitations, reproducibility).
