"""One-command reproduction: every step from download to results, then the test suite.

    python -m scripts.reproduce              OpenSky sample -> data/, results/, figures/
    python -m scripts.reproduce --synthetic  synthetic traffic (scripts.synth) in place of the
                                             download; run it with TC_CONFIG and TC_OUT set
                                             (`make smoke`) so the committed results stay as
                                             they are

The survey runs first because it needs the global hour on disk; `sample` then reuses that
tar and, with keep_raw: false, deletes each tar after hashing and filtering it.
"""

import argparse
import os
import subprocess
import sys
import time

DOWNLOAD = [
    ["-m", "scripts.fetch", "survey"],
    ["-m", "scripts.fetch", "sample"],
]
PIPELINE = [
    ["-m", "scripts.run", "ingest"],
    ["-m", "scripts.evaluate", "stub"],
    ["-m", "scripts.track", "verify"],
    ["-m", "scripts.track", "real"],
    ["-m", "scripts.track", "streaming"],
    ["-m", "scripts.evaluate", "harness"],
    ["-m", "scripts.evaluate", "references"],
    ["-m", "scripts.evaluate", "conformance"],
    ["-m", "scripts.evaluate", "single-count"],
    ["-m", "scripts.evaluate", "streaming"],
    ["-m", "scripts.evaluate", "drift"],
    ["-m", "scripts.figures", "phase2"],
    ["-m", "scripts.figures", "phase4"],
    ["-m", "scripts.figures", "phase5"],
]
TESTS = [["-m", "pytest", "-q"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic and not os.environ.get("TC_OUT"):
        raise SystemExit("--synthetic writes over data/ and results/ unless TC_OUT is set; use `make smoke`")
    # The tests read config.yaml and the committed results, so they are not run on the synthetic set.
    steps = [["-m", "scripts.synth"], *PIPELINE] if args.synthetic else [*DOWNLOAD, *PIPELINE, *TESTS]
    for step in steps:
        print("==>", " ".join(step), flush=True)
        started = time.perf_counter()
        subprocess.run([sys.executable, *step], check=True)
        print(f"    {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
