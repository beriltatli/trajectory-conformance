import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2


def chi2_tail_log10(nis: np.ndarray, dof: np.ndarray) -> np.ndarray:
    """-log10 P(chi2_dof > nis): one threshold is one tail probability whatever the dof.

    scipy's logsf underflows to -inf above NIS ~1500, which would put every large innovation at
    the same infinite score and leave a floor of alerts no threshold can remove. There the
    asymptotic series of the upper incomplete gamma function is used instead,
    Q(a, y) ~ y^(a-1) e^-y / Gamma(a) * (1 + (a-1)/y + (a-1)(a-2)/y^2), a = dof/2, y = nis/2;
    for dof <= 6 and y > 700 the omitted terms are below 1e-7 relative.
    """
    nis = np.asarray(nis, float)
    dof = np.asarray(dof, float)
    out = -chi2.logsf(nis, dof)
    deep = ~np.isfinite(out)
    if deep.any():
        a = 0.5 * dof[deep]
        y = 0.5 * nis[deep]
        series = 1.0 + (a - 1.0) / y + (a - 1.0) * (a - 2.0) / y**2
        out[deep] = y - (a - 1.0) * np.log(y) + gammaln(a) - np.log(series)
    return out / np.log(10.0)
