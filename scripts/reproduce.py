"""One-command reproduction: every step from download to results, then the test suite.

The survey runs first because it needs the global hour on disk; `sample` then reuses that
tar and, with keep_raw: false, deletes each tar after hashing and filtering it.
"""

import subprocess
import sys

STEPS = [
    ["-m", "scripts.fetch", "survey"],
    ["-m", "scripts.fetch", "sample"],
    ["-m", "scripts.run", "ingest"],
    ["-m", "scripts.evaluate", "stub"],
    ["-m", "scripts.track", "verify"],
    ["-m", "scripts.track", "real"],
    ["-m", "scripts.track", "streaming"],
    ["-m", "scripts.figures", "phase2"],
    ["-m", "pytest", "-q"],
]


def main() -> None:
    for step in STEPS:
        print("==>", " ".join(step), flush=True)
        subprocess.run([sys.executable, *step], check=True)


if __name__ == "__main__":
    main()
