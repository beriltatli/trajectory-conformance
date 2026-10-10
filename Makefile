PY := .venv/bin/python

.PHONY: check reproduce survey sample ingest stub

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

reproduce:
	$(PY) -m scripts.reproduce
