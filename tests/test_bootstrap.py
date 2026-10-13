"""Flight-level bootstrap of the suite's rates (evaluate/bootstrap.py)."""

import numpy as np
import pytest

from evaluate.bootstrap import DayCounts, bootstrap, metrics


def day(rng, flights=200, detect_p=0.3, fa_per_flight=1.0, thresholds=5):
    seconds = np.zeros((flights, 4))
    seconds[:, :3] = 1200.0
    fa = np.zeros((flights, 4), np.int64)
    fa[:, :3] = rng.poisson(fa_per_flight, (flights, 3))
    sweep = np.repeat(fa[:, None, :], thresholds, axis=1) * np.arange(thresholds, 0, -1)[None, :, None]
    event_flight = np.repeat(np.arange(flights), 3)
    phase = np.tile([0, 1, 2], flights)
    detected = rng.random(len(event_flight)) < detect_p
    sweep_detected = np.repeat(detected[:, None], thresholds, axis=1)
    return DayCounts(np.arange(flights).astype(str), seconds, fa, sweep, event_flight, phase, detected, sweep_detected)


def test_unit_weights_give_the_plain_rates():
    d = day(np.random.default_rng(0))
    m = metrics([d], [np.ones(200)], budget=2.0)
    assert m["fa_rate"][0] == pytest.approx(d.fa[:, 0].sum() / (200 * 1200 / 3600))
    assert m["detection"][3] == pytest.approx(d.detected.mean())


def test_interval_covers_the_estimate_and_shrinks_with_more_flights():
    rng = np.random.default_rng(1)
    small = bootstrap({"a": [day(rng, 100)]}, 2.0, 300, seed=0)["detectors"]["a"]["detection"]["all"]
    large = bootstrap({"a": [day(rng, 1600)]}, 2.0, 300, seed=0)["detectors"]["a"]["detection"]["all"]
    for r in (small, large):
        assert r["ci95"][0] <= r["estimate"] <= r["ci95"][1]
    assert np.diff(large["ci95"])[0] < 0.5 * np.diff(small["ci95"])[0]


def test_paired_difference_of_a_detector_with_itself_is_exactly_zero():
    d = day(np.random.default_rng(2))
    out = bootstrap({"a": [d], "b": [d]}, 2.0, 100, seed=0, pairs=[("a", "b")])
    diff = out["differences"]["a - b"]["detection"]["all"]
    assert diff["estimate"] == 0 and diff["ci95"] == [0.0, 0.0]


def test_refuses_detectors_run_on_different_flights():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        bootstrap({"a": [day(rng, 100)], "b": [day(rng, 101)]}, 2.0, 10, seed=0)
