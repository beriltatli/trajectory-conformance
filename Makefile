PY := .venv/bin/python

.PHONY: env smoke demo multiday-data folds check reproduce survey sample ingest stub verify track figures harness references conformance drift sim-data sim

# Pinned environment: Python 3.12 (pyproject.toml), exact versions in requirements.txt.
env:
	python3.12 -m venv .venv
	$(PY) -m pip install -r requirements.txt

# The whole pipeline on synthetic traffic (configs/smoke.yaml), no OpenSky data; outputs in
# build/smoke/, never in data/, results/ or figures/. About 3 min on 8 cores.
smoke:
	rm -rf build/smoke
	TC_CONFIG=configs/smoke.yaml TC_OUT=build/smoke $(PY) -m scripts.reproduce --synthetic
	TC_CONFIG=configs/smoke_multiday.yaml TC_OUT=build/smoke/multiday $(PY) -m scripts.synth
	TC_CONFIG=configs/smoke_multiday.yaml TC_OUT=build/smoke/multiday $(PY) -m scripts.run ingest
	TC_OUT=build/smoke/multiday $(PY) -m scripts.folds run --config configs/smoke_multiday.yaml --out build/smoke/multiday
	TC_OUT=build/smoke/multiday $(PY) -m scripts.folds aggregate --config configs/smoke_multiday.yaml --out build/smoke/multiday
	TC_CONFIG=configs/smoke_multiday.yaml TC_OUT=build/smoke/multiday $(PY) -m scripts.figures rolling

# Rolling evaluation over six Mondays (configs/multiday.yaml); everything OpenSky-derived stays
# in data/multiday/, the aggregate goes to results/phase6_rolling.json.
multiday-data:
	TC_CONFIG=configs/multiday.yaml TC_OUT=data/multiday $(PY) -m scripts.fetch sample
	TC_CONFIG=configs/multiday.yaml TC_OUT=data/multiday $(PY) -m scripts.run ingest

folds:
	$(PY) -m scripts.folds run
	$(PY) -m scripts.folds aggregate
	$(PY) -m scripts.figures rolling

check:
	$(PY) -m pytest -q

survey:
	$(PY) -m scripts.fetch survey

sample:
	$(PY) -m scripts.fetch sample

ingest:
	$(PY) -m scripts.run ingest

stub:
	$(PY) -m scripts.evaluate stub

harness:
	$(PY) -m scripts.evaluate harness

references:
	$(PY) -m scripts.evaluate references

conformance:
	$(PY) -m scripts.evaluate conformance
	$(PY) -m scripts.evaluate single-count
	$(PY) -m scripts.evaluate streaming

drift:
	$(PY) -m scripts.evaluate drift

reproduce:
	$(PY) -m scripts.reproduce

verify:
	$(PY) -m scripts.track verify

track:
	$(PY) -m scripts.track real
	$(PY) -m scripts.track streaming

figures:
	$(PY) -m scripts.figures phase2
	$(PY) -m scripts.figures phase4
	$(PY) -m scripts.figures phase5
	$(PY) -m scripts.figures phase6

# Public replay on synthetic traffic (configs/demo.yaml) -> sim/demo/, which GitHub Pages serves.
DEMO := TC_CONFIG=configs/demo.yaml TC_OUT=build/demo
demo:
	rm -rf build/demo sim/demo
	$(DEMO) $(PY) -m scripts.synth
	$(DEMO) $(PY) -m scripts.run ingest
	$(DEMO) $(PY) -m scripts.evaluate references
	$(DEMO) $(PY) -m scripts.evaluate conformance --only imm_conformance fixed_threshold
	$(DEMO) $(PY) -m scripts.sim export
	mkdir -p sim/demo && cp build/demo/data/sim/scene.json build/demo/data/sim/scene.bin sim/demo/

# 3D replay (sim/): data export, then a local-only server (the export is OpenSky data; DATA.md).
sim-data:
	$(PY) -m scripts.sim basemap
	$(PY) -m scripts.sim export

sim:
	@echo "http://127.0.0.1:8765/sim/"
	$(PY) -m http.server 8765 --bind 127.0.0.1
