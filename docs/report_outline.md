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
| 4.2 Adaptive loop | false alarms (clean), alarm latency and sensitivity (5 / 10 / 20 %); AUC-PR per wave: incumbent / control / variants / oracle; recall and precision at threshold; promotion win probability. **Say plainly:** (i) at noise 0.1 the incumbent's AUC-PR does *not* drop on attacked waves (0.85–0.86 vs 0.83–0.84 clean) because injected nodes are near-copies of training fraud — the loop shows detect → retrain → promote → audit, not recovery from damage (the `--noise 1.0` run, if it degrades the incumbent, is the one to headline); (ii) decompose the gain: incumbent 0.862 → control 0.902 → per_class 0.932 → oracle 0.973, i.e. most of it is retraining on the denser graph; (iii) "one wave late" is the 2-consecutive-flags rule | `make graph-loop`, `make graph-loop-hard` | `graph_loop_monitor_summary_mat_lvl*.csv`, `graph_loop_summary_mat_lvl0.2.csv`, `stats_loop_mat_lvl0.2.csv`, `fig_graph_loop_mat_lvl0.2.png` (+ `_noise1.0` variants if run) |
| 4.3 Bias audit | pseudo-positive share per round vs 14.5 % prior (naive 1.8 %, balanced_loss 5 %, per_class 21 %, prior_match 14.5 %); collapse rule (share < 50 % of prior or probe minority recall −30 %); operating-point metrics; EM prior estimate (0.09–0.15) vs true 25 % pool prior under attack | `11_graph_loop.py` | `graph_loop_rounds_mat_lvl0.2.csv`, `fig_graph_loop_rounds_mat_lvl0.2.png` |
| 4.4 Business integration | monitor → alarm → challenger → promotion; 100 probe labels per wave; what an analyst sees; operating threshold; what the scorer bar means for anyone buying text evidence (≥ ~0.7 fraud-centred edge AUC before repair pays) | – | from run logs |
| 5 Limitations | random waves are not real time; features are full-data aggregates; injection is our attack model and does not hurt the incumbent at noise 0.1; **a plain feature monitor is at least as sensitive as the structural signal on this attack** (10 %: 58 % vs 17 % of attacked waves flagged) because the injection shifts features — the structural signal would matter for attacks that change links but not features (untested); no clean-scenario control, so "retraining helps as the graph grows" is not separated from "retraining helps under attack"; the noisy-oracle curve assumes node-independent errors (optimistic); monitor blind below ~10 % injection; 3 seeds → wide intervals; no text, so the repair bar is untested with the planned scorer | – | – |
| 5 Reproducibility | README reproduce table, `make graph-all`; `run.json` per script (git SHA, config hash, seed, wall time); tests | all | `outputs/runs/*/run.json` |

## Professor's comments → where they are answered

1. *EDA, data preparation, alignment of raw text and timestamps with the graph.* §2 + §3.1. Do not
   answer "moot"; describe the alignment design that was built and tested, then why it could not run:
   (a) there is no public `.mat`-index → review mapping, so we never align by index (D1); (b) the graph
   is rebuilt from the raw metadata by dropping products > 800 reviews and applying the published relation
   definitions (R-U-R same user; R-S-R product × star; R-T-R product × month), checked against the
   published counts with a pre-registered tolerance (`02_build_graph.py`, `graph_build.py`); (c) text is
   joined on (user, product, date, k-th occurrence) and the loader raises on any unmatched review
   (`load_raw.py`); (d) features are recomputed point-in-time so nothing after a review's date leaks into
   it (`features_pit.py`, `tests/test_features_pit.py`); (e) equal-count temporal windows (D3). All of this
   runs end to end on the synthetic fixture (`tests/fixtures/raw_mini`, `make all SMOKE=1 --mock`). The
   release never arrived (D18), so arrival waves replace timestamps and the `.mat` features (full-data
   aggregates) are used as given — both stated as limitations. EDA to show: class balance, the homophily
   table and histograms by relation and class (the camouflage evidence), degree CCDFs (48 % of reviews
   have no R-U-R neighbour; fraud R-U-R degree 0.29 vs 2.46), and the variable table: 32 unnamed
   behavioural features (15 review + 9 user + 8 product per Rayana & Akoglu Table 2; column order not
   public, see `docs/data_sources.md`) with the top-10 by |Cohen's d| from `eda_features_mat.csv`.
2. *Does confidence-based self-training reinforce normal-class bias; how to check and fix?* §3.5 + §4.3:
   yes, it does: naive and balanced_loss pseudo-labels shrink to 1.9 % / 6.0 % positives in every seed;
   per-class quotas and prior matching hold the positive share and give the best recovery.
3. *Justification of reconstruction and drift detection; practical implications, limitations,
   reproducibility.* §3.2 + §4.1 (the noisy-oracle bar justifies when reconstruction is worth it), §3.4
   (calibrated, same-wave monitor; 0 false alarms), §4.4, §5.
