import numpy as np
import pandas as pd

import conform.fusion as fusion
from conform.fusion import DistanceTails, FusedConformance
from conform.score import fit_tail
from ingest.config import load_config
from reference.build import split_subflows

CFG = load_config()
REF_CFG = CFG["reference"]


def tail(scale: float):
    rng = np.random.default_rng(0)
    return fit_tail(np.abs(rng.normal(0.0, scale, 20_000)), REF_CFG["tail_points"], REF_CFG["tail_fraction"])


def test_a_tune_flight_is_scored_with_the_distance_tail_of_the_other_fold():
    tails = DistanceTails(tail(1.0), (tail(2.0), tail(3.0)), {"a": 0, "b": 1})
    assert tails.for_flight("a") is tails.folds[1]
    assert tails.for_flight("b") is tails.folds[0]
    assert tails.for_flight("test") is tails.all


def test_fused_score_is_the_larger_of_the_imm_score_and_the_offset_distance_rarity(monkeypatch):
    tails = DistanceTails(tail(1000.0), (tail(1000.0), tail(1000.0)), {})
    monkeypatch.setattr(fusion, "load_tails", lambda path: tails)
    detector = FusedConformance(CFG["imm"], REF_CFG, "unused", offset=0.5)
    imm = np.array([3.0, 0.1, np.nan, 0.2, np.nan])
    distance = np.array([10.0, 3000.0, 3000.0, np.nan, np.nan])
    monkeypatch.setattr(detector._imm, "scores", lambda flight: imm)
    monkeypatch.setattr(detector._distance, "scores", lambda flight: distance)

    got = detector.scores(pd.DataFrame({"flight_id": ["f"] * 5}))

    rarity = tails.all.score(np.nan_to_num(distance)) + 0.5
    assert got[0] == 3.0  # the IMM score wins
    assert got[1] == rarity[1] and got[1] > 2.0  # a 3 sd distance wins
    assert got[2] == rarity[2]  # no IMM score: the distance alone
    assert got[3] == 0.2  # no distance: the IMM score alone
    assert np.isnan(got[4])


def airway(offset_m: float, n: int, rng) -> list[np.ndarray]:
    x = np.linspace(0.0, 200_000.0, 20)
    return [np.column_stack([x, offset_m + rng.normal(0.0, 300.0, len(x))]) for _ in range(n)]


def test_a_pair_holding_two_airways_is_split_and_flights_are_assigned_from_their_start():
    rng = np.random.default_rng(1)
    shapes = airway(0.0, 6, rng) + airway(20_000.0, 6, rng)
    fids = [f"t{i}" for i in range(12)]
    od = {f: "A>B" for f in fids} | {"test_north": "A>B", "test_south": "A>B"}
    prefixes = {f: s[:4] for f, s in zip(fids, shapes)}
    prefixes["test_south"] = airway(0.0, 1, rng)[0][:4]
    # Starts on the northern airway, then turns away: only the start decides the flow.
    prefixes["test_north"] = airway(20_000.0, 1, rng)[0][:4]
    cfg = {**REF_CFG, "subflows": {"enabled": True, "min_gain": 0.3, "assign_s": 300}}

    members, out_od, report = split_subflows({"A>B": fids}, od, dict(zip(fids, shapes)), prefixes, cfg)

    assert report["pairs_split"] == 1 and report["tune_members_assigned_to_other_flow"] == 0
    assert sorted(members) == ["A>B#0", "A>B#1"]
    flows = {out_od[f] for f in fids[:6]}, {out_od[f] for f in fids[6:]}
    assert len(flows[0]) == 1 and len(flows[1]) == 1 and flows[0] != flows[1]
    assert out_od["test_south"] == out_od["t0"] and out_od["test_north"] == out_od["t6"]


def test_a_pair_with_one_airway_is_not_split():
    rng = np.random.default_rng(2)
    shapes = airway(0.0, 12, rng)
    fids = [f"t{i}" for i in range(12)]
    od = {f: "A>B" for f in fids}
    cfg = {**REF_CFG, "subflows": {"enabled": True, "min_gain": 0.3, "assign_s": 300}}

    members, out_od, report = split_subflows({"A>B": fids}, od, dict(zip(fids, shapes)), {}, cfg)

    assert report["pairs_split"] == 0 and members == {"A>B": fids} and out_od == od
