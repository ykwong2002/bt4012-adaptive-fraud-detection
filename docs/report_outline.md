# Report outline → scripts and figures (report05.pdf, 8–12 pages)

Graph-only track (D18). All model numbers: `.mat` under arrival waves (D13), test waves 4–7, paired
seed-level 95 % intervals (D17). Table files are in `outputs/tables/`, figures in `outputs/figures/`.

| § | Content | Produced by | Artefact |
|---|---|---|---|
| 1 Problem | camouflage lowers fraud homophily so neighbourhood averaging hides fraud; the two questions (repair bar; adaptive loop without normal-class collapse) | – | `docs/proposal.pdf` |
| 2 Data | `YelpChi.mat`: 45,954 reviews, 32 features, 14.5 % fraud, 3 relations; why the raw text was planned (D1) and why it is unavailable (D18); features are full-data aggregates | `01_download.py`, `02_build_graph.py --source mat` | `data/README.md` |
| 2 EDA | homophily by relation and class (R-U-R 93 % fraud homophily vs R-S-R / R-T-R ~18–20 %: the camouflage diagnostic); degree CCDFs | `03_eda.py --source mat` | `eda_homophily_mat.png`, `eda_degree_mat.png` |
| 3.1 Protocol | arrival waves as the temporal substitute (stratified, re-drawn per seed; train 0–2 / val 3 / test 4–7); edge masking; degree rescaling by visible nodes (D13, D14); what random waves cannot show (no real drift) | `src/camo/eval/waves.py`, `protocol.py`, `tests/test_waves.py` | D13, D14 |
| 3.2 Edge scorers | label-free scorers: feature link model (D11), cosine, R-U-R user profile (D15); noisy-oracle reference (D12); softrank reweighting; fraud-centred vs global edge AUC | `05b_feature_edges.py`, `05c_edge_diagnostics.py` | `edge_auc_centred_mat.csv` |
| 3.3 Models | WeightedSAGE (ours), CARE-static (comparator, D2), CatBoost on neighbour aggregates, AE; why each | `src/camo/models` | – |
| 3.4 Drift monitor | PSI/KS effect sizes on features and same-wave edge consistency; thresholds calibrated leave-one-training-wave-out (D8); why same-wave neighbours (D16); trigger rule | `src/camo/drift`, `11_graph_loop.py` | D8, D16 |
| 3.5 Label-scarce response | camouflage injection (template fraud + hijacked legit links, D13); alarm-triggered, time-respecting retraining (D9); pseudo-label variants naive / per_class / prior_match / balanced_loss / all; **bias audit** (prof. Q2); bootstrap promotion rule (D17) | `src/camo/selftrain`, `src/camo/adversarial/graph_inject.py` | D9, D13, D17 |
| 4.1 Baselines + repair | 4 models raw graph under waves vs stratified (waves cost CatBoost 0.04, GNNs 0.10); scorer curve: gain vs edge AUC, real scorers placed on it, intervals | `06_train.py --source mat --waves`, `make graph-scorers`, `12_stats.py` | `results_mat_waves_raw_graph_summary.csv`, `fig_headline_mat_waves.png`, `scorer_curve_mat_waves.csv`, `fig_scorer_curve_mat_waves.png`, `stats_rewiring_mat_waves.csv` |
| 4.2 Adaptive loop | false alarms (clean), alarm latency and sensitivity (5 / 10 / 20 %); AUC-PR per wave: incumbent / control / variants / oracle; recall and precision at threshold; promotion win probability | `make graph-loop` | `graph_loop_monitor_summary_mat_lvl*.csv`, `graph_loop_summary_mat_lvl0.2.csv`, `stats_loop_mat_lvl0.2.csv`, `fig_graph_loop_mat_lvl0.2.png` |
| 4.3 Bias audit | pseudo-positive share per round vs 14.5 % prior (naive 1.9 %, balanced_loss 6.0 %, per_class 21 %, prior_match 14.5 %); EM prior estimate vs true 25 % pool prior under attack | `11_graph_loop.py` | `graph_loop_rounds_mat_lvl0.2.csv` |
| 4.4 Business integration | monitor → alarm → challenger → promotion; 100 probe labels per wave; what an analyst sees; operating threshold | – | from run logs |
| 5 Limitations | random waves are not real time; features are full-data aggregates; injection is our attack model; monitor blind below ~10 % injection; 3 seeds → wide intervals; no text, so the repair bar is untested with the planned scorer | – | – |
| 5 Reproducibility | README reproduce table, `make graph-all`; `run.json` per script (git SHA, config hash, seed, wall time); tests | all | `outputs/runs/*/run.json` |

## Professor's comments → where they are answered

1. *EDA, data preparation, alignment of raw text and timestamps with the graph.* §2 + §3.1: we planned
   to rebuild the graph from the raw metadata rather than align by index (D1); the raw release became
   unobtainable (D18), so alignment is moot and arrival waves replace timestamps, with their limits stated.
2. *Does confidence-based self-training reinforce normal-class bias; how to check and fix?* §3.5 + §4.3:
   yes, it does: naive and balanced_loss pseudo-labels shrink to 1.9 % / 6.0 % positives in every seed;
   per-class quotas and prior matching hold the positive share and give the best recovery.
3. *Justification of reconstruction and drift detection; practical implications, limitations,
   reproducibility.* §3.2 + §4.1 (the noisy-oracle bar justifies when reconstruction is worth it), §3.4
   (calibrated, same-wave monitor; 0 false alarms), §4.4, §5.
