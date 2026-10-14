import numpy as np
import pandas as pd

from ingest.flights import FlightAccumulator, FlightSegmenter


def measurements(icao24: str, t: list[float], callsign: list[str | None]) -> pd.DataFrame:
    return pd.DataFrame({"icao24": icao24, "lastposupdate": np.asarray(t, float), "callsign": callsign})


def test_gap_longer_than_split_starts_new_flight():
    seg = FlightSegmenter(600)
    out = seg.push(measurements("a", [0, 10, 20, 700, 710], ["X1"] * 5))
    assert out["flight_id"].tolist() == ["a_0"] * 3 + ["a_700"] * 2
    assert seg.splits == {"first_seen": 1, "gap": 1, "callsign": 0}


def test_callsign_change_splits_but_missing_callsign_does_not():
    seg = FlightSegmenter(600)
    out = seg.push(measurements("a", [0, 10, 20, 30], ["X1", None, "X1", "Y2"]))
    assert out["flight_id"].tolist() == ["a_0", "a_0", "a_0", "a_30"]


def test_state_carries_across_pushes():
    seg = FlightSegmenter(600)
    first = seg.push(measurements("a", [0, 10], ["X1", "X1"]))
    second = seg.push(measurements("a", [20, 900], [None, "X1"]))
    assert first["flight_id"].tolist() == ["a_0", "a_0"]
    assert second["flight_id"].tolist() == ["a_0", "a_900"]


def test_a_previous_flights_callsign_does_not_split_the_next_one():
    # Flight X1, a week's gap, then a flight whose first update has no callsign and whose
    # callsign Y2 arrives in the next push. Found by the rolling evaluation: the stale X1
    # was compared with Y2 across the push boundary and cut the new flight in two.
    t = [0, 10, 604_800, 605_330, 605_340]
    callsigns = ["X1", "X1", None, "Y2", "Y2"]
    seg = FlightSegmenter(600)
    split = [*seg.push(measurements("a", t[:3], callsigns[:3]))["flight_id"],
             *seg.push(measurements("a", t[3:], callsigns[3:]))["flight_id"]]
    whole = FlightSegmenter(600).push(measurements("a", t, callsigns))["flight_id"].tolist()
    assert split == whole == ["a_0", "a_0"] + ["a_604800"] * 3


def test_a_callsign_change_across_pushes_still_splits():
    seg = FlightSegmenter(600)
    seg.push(measurements("a", [0, 10], ["X1", None]))
    assert seg.push(measurements("a", [20, 30], [None, "Y2"]))["flight_id"].tolist() == ["a_0", "a_30"]


def test_accumulator_releases_flight_only_after_it_closes():
    acc = FlightAccumulator(600)
    m1 = measurements("a", [0, 10], ["X", "X"]).assign(flight_id="a_0")
    m2 = measurements("b", [500], ["Y"]).assign(flight_id="b_500")
    m3 = measurements("b", [700], ["Y"]).assign(flight_id="b_500")
    assert list(acc.push(m1)) == []
    assert list(acc.push(m2)) == []
    released = list(acc.push(m3))
    assert [f["flight_id"].iloc[0] for f in released] == ["a_0"]
    assert acc.open_flights == 1
    assert [f["flight_id"].iloc[0] for f in acc.flush()] == ["b_500"]
