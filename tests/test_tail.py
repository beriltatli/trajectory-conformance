import numpy as np
import pytest
from scipy.stats import chi2

from conform.tail import chi2_tail_log10


@pytest.mark.parametrize("dof", [1, 2, 3, 4, 5, 6])
def test_tail_matches_scipy_where_scipy_is_finite_and_stays_finite_and_monotone_beyond(dof):
    x = np.array([0.5, 5.0, 50.0, 500.0, 1400.0])
    d = np.full(len(x), dof)
    assert np.allclose(chi2_tail_log10(x, d), -chi2.logsf(x, dof) / np.log(10.0), rtol=1e-12)
    far = np.geomspace(1300.0, 1e7, 200)
    score = chi2_tail_log10(far, np.full(len(far), dof))
    assert np.all(np.isfinite(score)) and np.all(np.diff(score) > 0)


def test_asymptotic_branch_agrees_with_exact_tail_at_the_switch():
    # Even dof: Q(a, y) is exactly e^-y * sum_{k<a} y^k / k!, so the far tail has a closed form.
    y = 900.0
    exact = (y - np.log(1.0 + y + y**2 / 2.0)) / np.log(10.0)
    assert chi2_tail_log10(np.array([2 * y]), np.array([6.0]))[0] == pytest.approx(exact, rel=1e-12)
