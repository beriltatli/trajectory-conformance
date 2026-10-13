"""Phase 2 entry points: the IMM's verification checks.

    python -m scripts.track verify   synthetic NIS consistency and mode switching delays
                                     -> results/phase2_filter_checks.json
    python -m scripts.track real     IMM over every kept flight of the sample
                                     -> results/phase2_real_filter.json,
                                        results/phase2_measurement_noise.json
    python -m scripts.track streaming
                                     online track bank vs whole-flight runs on the first rows
                                     of the sample -> results/phase2_streaming_equivalence.json
    python -m scripts.track consistency
                                     normalised innovation per component and phase
                                     -> results/phase6_consistency.json
    python -m scripts.track qr-fit   per-phase R and Q scales fitted on the tune day
                                     -> results/phase6_qr_fit.json, configs/qr_fitted.yaml
"""

import argparse
import json

import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial

from evaluate.consistency import Consistency, flight_e2
from evaluate.filter_checks import mode_delays, nis_consistency
from evaluate.qr_fit import fit as qr_fit
from evaluate.filter_real import RealFilterStats, flight_summary, streaming_equivalence
from ingest.config import load_config, repo_path
from ingest.pipeline import json_ready, peak_rss_mb
from evaluate.flights import iter_flights
from track.imm import ImmConfig

# Flights handed to the worker pool at a time; bounds how many closed flights wait in memory.
BATCH_FLIGHTS = 24
# First rows of the measurement file used for the streaming equivalence check.
STREAMING_ROWS = 50_000


def verify(cfg: dict) -> dict:
    imm = ImmConfig.from_config(cfg["imm"])
    checks = cfg["imm"]["checks"]
    return json_ready({
        "seed": cfg["seed"],
        "nis_consistency": nis_consistency(imm, cfg["imm"]["q"]["cv"], cfg["seed"], checks["nis_runs"], checks["nis_duration_s"]),
        "mode_delays": mode_delays(imm, cfg["seed"] + 1, checks["delay_runs"]),
        # CV's missing acceleration mixed into CA/TURN as zero with zero variance; same seed.
        "mode_delays_zero_padded": mode_delays(replace(imm, mix_own_acceleration=False), cfg["seed"] + 1, checks["delay_runs"]),
    })


def real(cfg: dict, workers: int) -> tuple[dict, dict]:
    imm = ImmConfig.from_config(cfg["imm"])
    ing = cfg["ingest"]
    stats = RealFilterStats([m.name for m in imm.models])
    tune_days = set(cfg["evaluation"]["tune_days"])
    summarise = partial(flight_summary, cfg=imm, window_s=cfg["phase"]["window_s"],
                        level_rate_mps=cfg["phase"]["level_rate_mps"], burn_in=cfg["imm"]["checks"]["burn_in_updates"])
    skipped = 0
    batch = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for flight in iter_flights(cfg):
            t = flight["lastposupdate"].to_numpy(float)
            if len(flight) < ing["min_flight_updates"] or t[-1] - t[0] < ing["min_flight_duration_s"]:
                skipped += 1
                continue
            batch.append(flight)
            if len(batch) == BATCH_FLIGHTS:
                for s in pool.map(summarise, batch):
                    stats.add(s, tune_days)
                batch = []
        for s in pool.map(summarise, batch):
            stats.add(s, tune_days)
    summary = stats.summary()
    summary["flights_skipped_short"] = skipped
    summary["tune_days"] = sorted(tune_days)
    summary["peak_rss_mb_main_process"] = peak_rss_mb()
    summary["peak_rss_mb_largest_worker"] = peak_rss_mb(children=True)
    summary["workers"] = workers
    noise = stats.measurement_noise()
    noise["days"] = sorted(tune_days)
    noise["configured_sd"] = cfg["imm"]["measurement_sd"]
    return json_ready(summary), json_ready(noise)


def consistency(cfg: dict, workers: int) -> dict:
    imm = ImmConfig.from_config(cfg["imm"])
    ing = cfg["ingest"]
    stats = Consistency()
    fn = partial(flight_e2, cfg=imm, window_s=cfg["phase"]["window_s"], level_rate_mps=cfg["phase"]["level_rate_mps"],
                 burn_in=cfg["imm"]["checks"]["burn_in_updates"])
    batch = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for flight in iter_flights(cfg):
            t = flight["lastposupdate"].to_numpy(float)
            if len(flight) < ing["min_flight_updates"] or t[-1] - t[0] < ing["min_flight_duration_s"]:
                continue
            batch.append(flight)
            if len(batch) == BATCH_FLIGHTS:
                for item in pool.map(fn, batch):
                    stats.add(item)
                batch = []
        for item in pool.map(fn, batch):
            stats.add(item)
    out = stats.summary()
    out["measurement_sd"] = cfg["imm"]["measurement_sd"]
    out["q"] = cfg["imm"]["q"]
    return json_ready(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["verify", "real", "streaming", "consistency", "qr-fit"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cfg = load_config()
    if args.command == "qr-fit":
        import yaml

        result = json_ready(qr_fit(cfg, args.workers, iterations=8, sample_flights=600))
        repo_path("results/phase6_qr_fit.json").write_text(json.dumps(result, indent=2))
        header = ("# Written by `python -m scripts.track qr-fit` (evaluate/qr_fit.py) from the tune day only:\n"
                  "# config.yaml with per-phase R and Q scales. Run the pipeline with TC_CONFIG=configs/qr_fitted.yaml.\n")
        repo_path("configs/qr_fitted.yaml").write_text(header + yaml.safe_dump({"extends": "../config.yaml", "imm": {"phase_scale": result["phase_scale"]}}, sort_keys=False))
        return
    if args.command == "consistency":
        result = consistency(cfg, args.workers)
        repo_path("results/phase6_consistency.json").write_text(json.dumps(result, indent=2))
        for day, phases in result["days"].items():
            for phase, comps in phases.items():
                print(day, f"{phase:8s}", "  ".join(f"{c}:{v['median_e2']:.3f}/{v['share_above_95']:.2f}" for c, v in comps.items()))
        return
    if args.command == "verify":
        result = verify(cfg)
        repo_path("results/phase2_filter_checks.json").write_text(json.dumps(result, indent=2))
        for name, f in result["nis_consistency"]["filters"].items():
            print(name, {k: v for k, v in f.items() if k != "run_average"})
        for variant in ("mode_delays", "mode_delays_zero_padded"):
            for name, d in result[variant].items():
                print(variant, name, {k: v for k, v in d.items() if k not in ("mean_mode_prob", "start_s", "end_s", "runs")})
    elif args.command == "streaming":
        source = pq.ParquetFile(repo_path(cfg["data"]["measurements_path"]))
        rows = next(source.iter_batches(batch_size=STREAMING_ROWS)).to_pandas()
        result = json_ready(streaming_equivalence(ImmConfig.from_config(cfg["imm"]), rows, 2_500, cfg["ingest"]["flight_split_gap_s"]))
        repo_path("results/phase2_streaming_equivalence.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    else:
        summary, noise = real(cfg, args.workers)
        repo_path("results/phase2_real_filter.json").write_text(json.dumps(summary, indent=2))
        repo_path("results/phase2_measurement_noise.json").write_text(json.dumps(noise, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k not in ("nis", "nis_over_dof_edges")}, indent=2))
        for day, phases in summary["nis"]["nis"].items():
            for phase, row in phases.items():
                print(day, phase, {k: v for k, v in row.items() if k != "nis_over_dof_histogram"})
        print(json.dumps(noise, indent=2))


if __name__ == "__main__":
    main()
