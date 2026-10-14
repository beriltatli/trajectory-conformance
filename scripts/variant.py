"""Phase 6 variants: the phase 4 suite rerun with one change, under its own output root.

    python -m scripts.variant qr        per-phase R and q from results/phase6_qr_fit.json
                                        (python -m scripts.track qr-fit): filter statistics,
                                        consistency and the full suite
    python -m scripts.variant subflows  references split into flows (configs/subflows.yaml):
                                        references, the IMM monitor and the fixed distance

Each run goes to data/variants/<name>/ (or under TC_OUT) with a config that extends the current
one, so the smoke run tests the same path on synthetic traffic. The aggregates are copied to
results/ under their phase 6 names; nothing per flight leaves data/.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from ingest.config import REPO, load_config, repo_path

STEPS = {
    "qr": [["scripts.track", "real"], ["scripts.track", "consistency"],
           ["scripts.evaluate", "references"], ["scripts.evaluate", "conformance"]],
    "subflows": [["scripts.evaluate", "references"],
                 ["scripts.evaluate", "conformance", "--only", "imm_conformance", "fixed_threshold"]],
}
# results/ of the variant run -> results/ of this one.
COPIES = {
    "qr": {"phase6_consistency.json": "phase6_consistency_fitted.json",
           "phase2_real_filter.json": "phase6_real_filter_fitted.json",
           **{f"phase4_{d}.json": f"phase6_qr_{d}.json" for d in ("imm_conformance", "cv_kalman", "eta_deviation")}},
    "subflows": {f"phase4_{d}.json": f"phase6_subflows_{d}.json" for d in ("references", "imm_conformance", "fixed_threshold")},
}
# Keys that differ between runs of the same computation (machine load), not results.
VOLATILE = ("peak_rss_mb_main_process", "peak_rss_mb_largest_worker")


def overlay(name: str) -> dict:
    if name == "qr":
        fit = json.loads(repo_path("results/phase6_qr_fit.json").read_text())
        return {"imm": {"phase_scale": fit["phase_scale"]}}
    sub = yaml.safe_load((REPO / "configs" / "subflows.yaml").read_text())
    sub.pop("extends", None)
    return sub


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE}
    return value


def run(name: str, workers: int) -> None:
    cfg = load_config()
    out = repo_path(f"data/variants/{name}")
    out.mkdir(parents=True, exist_ok=True)
    base = Path(os.environ.get("TC_CONFIG") or REPO / "config.yaml")
    variant_cfg = out / "config.yaml"
    variant_cfg.write_text(yaml.safe_dump({
        "extends": str(base if base.is_absolute() else REPO / base),
        **overlay(name),
        # The variant's own data/ holds its references and events, not the measurements.
        "data": {"measurements_path": str(repo_path(cfg["data"]["measurements_path"]))},
    }, sort_keys=False))
    env = {**os.environ, "TC_CONFIG": str(variant_cfg), "TC_OUT": str(out)}
    for module, *args in STEPS[name]:
        print(f"==> {name}: {module} {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, "-m", module, *args, "--workers", str(workers)], env=env, check=True, cwd=REPO)
    for src, dst in COPIES[name].items():
        shutil.copyfile(out / "results" / src, repo_path(f"results/{dst}"))
    if name == "qr":
        # The fixed distance uses no filter: the fitted R and q must leave it unchanged.
        same = _strip(json.loads((out / "results" / "phase4_fixed_threshold.json").read_text())) == \
            _strip(json.loads(repo_path("results/phase4_fixed_threshold.json").read_text()))
        print("fixed distance unchanged by the fitted filter:", same)
        if not same:
            raise SystemExit("the fixed distance changed with the filter: the variant is not isolated")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(STEPS))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run(args.name, args.workers)


if __name__ == "__main__":
    main()
