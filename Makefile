PY := .venv/bin/python

.PHONY: check reproduce survey sample ingest stub verify track figures harness references conformance drift sim-data sim

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

# 3D replay (sim/): data export, then a local-only server (the export is OpenSky data; DATA.md).
sim-data:
	$(PY) -m scripts.sim basemap
	$(PY) -m scripts.sim export

sim:
	@echo "http://127.0.0.1:8765/sim/"
	$(PY) -m http.server 8765 --bind 127.0.0.1
