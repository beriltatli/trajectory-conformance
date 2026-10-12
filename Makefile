PY := .venv/bin/python

.PHONY: check reproduce survey sample ingest stub verify track figures harness

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

reproduce:
	$(PY) -m scripts.reproduce

verify:
	$(PY) -m scripts.track verify

track:
	$(PY) -m scripts.track real
	$(PY) -m scripts.track streaming

figures:
	$(PY) -m scripts.figures phase2
