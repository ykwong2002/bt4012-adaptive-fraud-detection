.PHONY: setup test lint data eda llm-score distil train drift selftrain adversarial figures all

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

train:
	$(PY) scripts/06_train.py $(SMOKEFLAG)

drift:
	$(PY) scripts/07_drift.py $(SMOKEFLAG)

selftrain:
	$(PY) scripts/08_selftrain.py $(SMOKEFLAG)

adversarial:
	$(PY) scripts/09_adversarial.py $(SMOKEFLAG)

figures:
	$(PY) scripts/10_figures.py

all: data eda train drift selftrain figures
