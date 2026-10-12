import numpy as np

from alerts.manage import hold, manage, persist, presented
from evaluate.rates import alert_starts

T = 10.0 * np.arange(12)


def test_persistence_needs_consecutive_exceedances():
    e = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0], bool)
    assert persist(e, 2).tolist() == [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    assert persist(e, 1).tolist() == e.tolist()


def test_hold_merges_recrossings_within_holdoff_and_stays_causal():
    a = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], bool)
    held = hold(a, T, 30.0)
    # Re-crossing 30 s after the last active row continues the alert; 70 s later is a new one.
    assert alert_starts(held).tolist() == [1, 11]
    # The alert closes 30 s after its last active row without looking at later rows.
    assert held[:9].tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 0]
    assert hold(a[:9], T[:9], 30.0).tolist() == held[:9].tolist()


def test_one_slow_drift_hovering_at_threshold_is_one_alert():
    e = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0], bool)
    raw = alert_starts(e)
    managed = alert_starts(manage(e, T, {"persist_updates": 2, "holdoff_s": 60.0}))
    assert len(raw) == 3 and managed.tolist() == [2]


def test_an_alert_left_open_is_presented_again_every_represent_s():
    t = 10.0 * np.arange(100)
    active = np.zeros(100, bool)
    active[5:75] = True  # open 700 s
    active[90:92] = True
    assert presented(active, t, 300.0).tolist() == [5, 35, 65, 90]
