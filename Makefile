.PHONY: setup test lint data eda llm-score distil feature-edges train drift selftrain adversarial graph-adversarial graph-scorers rur-scorer graph-loop graph-loop-hard graph-all raw-all figures all

UV ?= uv
PY  = $(UV) run python
SMOKE ?= 0
SMOKEFLAG = $(if $(filter 1,$(SMOKE)),--smoke,)

setup:
	$(UV) sync --extra dev
	$(UV) export --no-dev --no-hashes -o requirements.txt

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src scripts tests

data:
	$(PY) scripts/01_download.py

eda:
	$(PY) scripts/02_build_graph.py $(SMOKEFLAG)
	$(PY) scripts/03_eda.py $(SMOKEFLAG)

llm-score:
	$(PY) scripts/04_llm_score.py --dry-run $(SMOKEFLAG)

distil:
	$(PY) scripts/05_distil_edges.py $(SMOKEFLAG)

feature-edges:
	$(PY) scripts/05b_feature_edges.py --scorer feature $(SMOKEFLAG)

train:
	$(PY) scripts/06_train.py $(SMOKEFLAG)

drift:
	$(PY) scripts/07_drift.py $(SMOKEFLAG)

selftrain:
	$(PY) scripts/08_selftrain.py $(SMOKEFLAG)

adversarial:
	$(PY) scripts/09_adversarial.py $(SMOKEFLAG)

graph-adversarial:                 # graph-only track (D13): arrival waves + node injection, runs on .mat
	$(PY) scripts/06_train.py --source mat --waves $(SMOKEFLAG)
	$(PY) scripts/09b_graph_adversarial.py --source mat $(SMOKEFLAG)

figures:
	$(PY) scripts/10_figures.py

all: data eda train drift selftrain figures

# graph-only scorer-quality curve under arrival waves (D12, D13): noisy oracle, cosine, feature scorer
NOISY ?= 0.5 0.6 0.7 0.8 0.9
CURVE_SEEDS ?= 0,1,2
comma := ,
graph-scorers:
	$(PY) scripts/05b_feature_edges.py --source mat --scorer cosine
	for a in $(NOISY); do $(PY) scripts/05b_feature_edges.py --source mat --scorer noisy_oracle --edge-auc $$a; done
	for s in $(subst $(comma), ,$(CURVE_SEEDS)); do $(PY) scripts/05b_feature_edges.py --source mat --scorer feature --waves-seed $$s; done
	for w in cosine $(addprefix noisy,$(NOISY)); do $(PY) scripts/06_train.py --source mat --waves --mode softrank --models wsage,catboost --seeds $(CURVE_SEEDS) --weights data/processed/edge_scores_mat_$$w.npz --tag mat_waves_$${w}_softrank; done
	$(PY) scripts/06_train.py --source mat --waves --mode softrank --models wsage,catboost --seeds $(CURVE_SEEDS) --weights 'data/processed/edge_scores_mat_feature_w{seed}.npz' --tag mat_waves_feature_softrank
	$(MAKE) rur-scorer PY="$(PY)" CURVE_SEEDS=$(CURVE_SEEDS)

# R-U-R user-profile scorer (D15), fit per seed on that seed's training waves
rur-scorer:
	for s in $(subst $(comma), ,$(CURVE_SEEDS)); do $(PY) scripts/05b_feature_edges.py --source mat --scorer rur_profile --waves-seed $$s; done
	$(PY) scripts/06_train.py --source mat --waves --mode softrank --models wsage,catboost --seeds $(CURVE_SEEDS) --weights 'data/processed/edge_scores_mat_rur_profile_w{seed}.npz' --tag mat_waves_rur_profile_softrank

# end-to-end adaptive loop on the graph-only track (D16): monitor sensitivity at 5/10 %, full loop at 20 %
graph-loop:
	for l in 0.05 0.1; do $(PY) scripts/11_graph_loop.py --source mat --monitor-only --scenarios attack --level $$l; done
	$(PY) scripts/11_graph_loop.py --source mat --level 0.2
	$(PY) scripts/12_stats.py --tag mat

# harder attack: injected features further from the fraud templates (noise in training sd). At the
# default 0.1 the incumbent is not hurt by the injection; this run checks whether it can be.
HARD_NOISE ?= 1.0
graph-loop-hard:
	$(PY) scripts/11_graph_loop.py --source mat --level 0.2 --noise $(HARD_NOISE)
	$(PY) scripts/12_stats.py --tag mat --level 0.2 --noise $(HARD_NOISE)
	$(PY) scripts/10_figures.py

# text track on the raw release (YelpZip slice, D19): every reported `raw` number and figure, given
# complete LLM caches in data/llm/ (04 --submit was run and all batches have ended)
raw-all:
	$(PY) scripts/02_build_graph.py --source raw
	$(PY) scripts/03_eda.py --source raw
	$(PY) scripts/04_llm_score.py --source raw --collect
	$(PY) scripts/05_distil_edges.py --source raw
	$(PY) scripts/06_train.py --source raw
	for m in soft softrank; do $(PY) scripts/06_train.py --source raw --weights data/processed/edge_scores_raw.npz --mode $$m; done
	$(PY) scripts/06_train.py --source raw --weights data/processed/edge_scores_raw.npz --mode hard --tau 0.5
	$(PY) scripts/07_drift.py --source raw
	$(PY) scripts/08_selftrain.py --source raw
	$(PY) scripts/08_selftrain.py --source raw --weights data/processed/edge_scores_raw.npz
	$(PY) scripts/09_adversarial.py --source raw --recover
	$(PY) scripts/10_figures.py

# every reported number and figure (graph-only track, D18), from a fresh clone
graph-all: data
	$(PY) scripts/02_build_graph.py --source mat
	$(PY) scripts/03_eda.py --source mat
	$(PY) scripts/06_train.py --source mat --waves
	$(MAKE) graph-scorers PY="$(PY)"
	$(PY) scripts/05c_edge_diagnostics.py --source mat
	$(MAKE) graph-loop PY="$(PY)"
	$(PY) scripts/10_figures.py
