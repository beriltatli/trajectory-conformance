import numpy as np
import pandas as pd
import pytest

from ingest.clean import RepeatAudit, StaleRowFilter, consecutive_lastposupdate_repeats
from tests.synthetic import snapshot_track, state_rows


def run(rows: pd.DataFrame, chunk: int, window: float = 30.0) -> tuple[pd.DataFrame, StaleRowFilter]:
    cleaner = StaleRowFilter(window)
    parts = [cleaner.push(rows.iloc[i:i + chunk]) for i in range(0, len(rows), chunk)]
    parts.append(cleaner.flush())
    return pd.concat([p for p in parts if len(p)], ignore_index=True), cleaner


def test_stale_repeats_are_removed_and_counted():
    rows = snapshot_track("aaa001", 1000, 30, stale_every=3)
    out, cleaner = run(rows, chunk=len(rows))
    assert cleaner.counts.stale_repeat == 9
    assert len(out) == 21
    assert consecutive_lastposupdate_repeats(out) == 0
    assert out["lastposupdate"].is_monotonic_increasing


def test_measurement_time_is_lastposupdate_not_snapshot_time():
    rows = state_rows("aaa002", [1000, 1010], [990.5, 1009.6], [50.0, 50.0], [8.0, 8.1])
    out, _ = run(rows, chunk=10)
    assert out["lastposupdate"].tolist() == [990.5, 1009.6]


def test_exact_duplicates_onground_and_null_positions():
    base = snapshot_track("aaa003", 1000, 10)
    dup = base.iloc[[2, 5]]
    ground = snapshot_track("aaa004", 1000, 4).assign(onground=True)
    nopos = snapshot_track("aaa005", 1000, 3).assign(lat=np.nan)
    rows = pd.concat([base, dup, ground, nopos]).sort_values("time", kind="stable")
    out, cleaner = run(rows, chunk=7)
    c = cleaner.counts
    assert (c.exact_duplicate, c.onground, c.null_position) == (2, 4, 3)
    assert set(out["icao24"]) == {"aaa003"}
    assert len(out) == 10


def test_out_of_order_rows_within_window_are_reordered():
    rows = snapshot_track("aaa006", 1000, 12)
    shuffled = rows.iloc[[0, 1, 3, 2, 4, 6, 5, 7, 8, 9, 11, 10]]
    out, cleaner = run(shuffled, chunk=2)
    assert cleaner.counts.late == 0
    assert out["lastposupdate"].is_monotonic_increasing
    assert len(out) == 12


def test_row_later_than_window_is_dropped_as_late():
    rows = snapshot_track("aaa007", 1000, 12)
    late = rows.iloc[[3]]
    rest = rows.drop(rows.index[3])
    out, cleaner = run(pd.concat([rest, late]), chunk=1, window=20.0)
    assert cleaner.counts.late == 1
    assert len(out) == 11
    assert consecutive_lastposupdate_repeats(out) == 0


@pytest.mark.parametrize("chunk", [1, 3, 17, 1000])
def test_output_does_not_depend_on_chunking(chunk):
    rows = pd.concat([
        snapshot_track("bbb001", 1000, 40, stale_every=4),
        snapshot_track("bbb002", 1005, 35, stale_every=5, lat0=49.0),
        snapshot_track("bbb003", 1100, 20, lat0=51.0),
    ]).sort_values(["time", "icao24"], kind="stable").reset_index(drop=True)
    reference, ref_cleaner = run(rows, chunk=len(rows))
    out, cleaner = run(rows, chunk=chunk)
    key = ["icao24", "lastposupdate"]
    pd.testing.assert_frame_equal(
        out.sort_values(key).reset_index(drop=True)[key + ["lat", "lon"]],
        reference.sort_values(key).reset_index(drop=True)[key + ["lat", "lon"]],
    )
    assert cleaner.counts.as_dict() == ref_cleaner.counts.as_dict()


def test_repeat_audit_flags_a_repeated_position_across_chunks():
    audit = RepeatAudit()
    audit.push(pd.DataFrame({"icao24": ["x", "x"], "lastposupdate": [1.0, 2.0]}))
    audit.push(pd.DataFrame({"icao24": ["x"], "lastposupdate": [2.0]}))
    assert audit.violations == 1


def test_contact_before_separates_silence_from_undecoded_positions():
    times = np.arange(1000, 1100, 10)
    lpu = np.array([999.6, 999.6, 999.6, 999.6, 999.6, 999.6, 999.6, 999.6, 999.6, 1089.6])
    heard = np.array([999.7, 1009.8, 1019.8, 1029.8, 1039.8, 1049.8, 1059.8, 1069.8, 1079.8, 1089.8])
    rows = state_rows("ccc001", times, lpu, np.full(10, 50.0), np.r_[np.full(9, 8.0), 8.5], lastcontact=heard)
    out, _ = run(rows, chunk=3)
    # The aircraft kept talking through the 90 s gap: last contact before the new fix is 1079.8.
    assert out["contact_before"].iloc[-1] == pytest.approx(1079.8)
