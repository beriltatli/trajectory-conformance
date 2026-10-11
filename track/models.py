"""Filter models of 04_genovese_2001_imm.pdf (CV, CA, TURN), on a common 9-element state.

State order is Genovese's CA/TURN state, X = [x, x', x'', y, y', y'', z, z', z''], with x east
and y north of the flight's tangent frame and z barometric altitude.

Genovese's CV model has 6 states; the paper does not say how a 6-state CV is mixed with 9-state
CA and TURN models. Here CV is carried on the 9-element state with its acceleration rows and
columns of Phi and Q set to zero. CV's acceleration is therefore not an estimate, and mixing it
into CA or TURN as "zero with zero variance" locks those models out: with q_CA entering only the
acceleration (p. 617), a model whose mixed acceleration variance is ~0 gets no gain on
acceleration and can never learn a turn (results/phase2_filter_checks.json, `zero_padded`).
The mixing step (track/imm.py) instead fills CV's missing acceleration with the receiving
model's own acceleration estimate and covariance.
"""

from dataclasses import dataclass

import numpy as np

AXES = 3
STATE_DIM = 9
POS = np.array([0, 3, 6])
VEL = np.array([1, 4, 7])
ACC = np.array([2, 5, 8])

# Below this omega*dt the TURN transition is replaced by its limit, the CA transition; the
# closed form divides by omega^2.
_SMALL_TURN = 1e-6


def _block_diag(block: np.ndarray) -> np.ndarray:
    out = np.zeros((STATE_DIM, STATE_DIM))
    for a in range(AXES):
        out[3 * a:3 * a + 3, 3 * a:3 * a + 3] = block
    return out


def cv_transition(dt: float) -> np.ndarray:
    # phi_CV (p. 617), padded with a zero acceleration row and column.
    return _block_diag(np.array([[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))


def cv_noise(q: float, dt: float) -> np.ndarray:
    # q_CV = q dt [[dt^2/3, dt/2], [dt/2, 1]] (p. 617).
    return _block_diag(q * dt * np.array([[dt**2 / 3.0, dt / 2.0, 0.0], [dt / 2.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))


def ca_transition(dt: float) -> np.ndarray:
    # phi_CA (p. 617).
    return _block_diag(np.array([[1.0, dt, dt**2 / 2.0], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]))


def ca_noise(q: float, dt: float) -> np.ndarray:
    # q_CA = q dt diag(0, 0, 1) (p. 617); TURN uses the same plant noise matrix (p. 618).
    return _block_diag(q * dt * np.diag([0.0, 0.0, 1.0]))


def turn_rate(state: np.ndarray) -> float:
    """omega = |A| / |V| from the filtered state (p. 617)."""
    speed = float(np.linalg.norm(state[VEL]))
    if speed < 1.0:
        return 0.0
    return float(np.linalg.norm(state[ACC])) / speed


def turn_transition(dt: float, omega: float) -> np.ndarray:
    # phi_TURN (p. 617). The paper writes the interval as delta-tau here and delta-t for CV and
    # CA; both are the time from the track's valid time to the measurement time.
    wt = omega * dt
    if wt < _SMALL_TURN:
        return ca_transition(dt)
    s, c = np.sin(wt), np.cos(wt)
    return _block_diag(np.array([
        [1.0, s / omega, (1.0 - c) / omega**2],
        [0.0, c, s / omega],
        [0.0, -omega * s, c],
    ]))


def constrain_turn(state: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constant-speed pseudo-measurement update, V . A = 0 (p. 618).

    The paper selects R^c "to achieve a gain of 0.5"; the scalar gain v K^c equals
    vPv^T / (vPv^T + R^c), so R^c = vPv^T.
    """
    speed = float(np.linalg.norm(state[VEL]))
    if speed < 1.0:
        return state, cov
    v = np.zeros(STATE_DIM)
    v[ACC] = state[VEL] / speed
    pv = cov @ v
    vpv = float(v @ pv)
    if vpv <= 0.0:
        return state, cov
    gain = pv / (2.0 * vpv)
    reduce = np.eye(STATE_DIM) - np.outer(gain, v)
    return reduce @ state, reduce @ cov


@dataclass(frozen=True, slots=True)
class FilterModel:
    name: str
    q: float

    def transition(self, dt: float, state: np.ndarray) -> np.ndarray:
        if self.name == "cv":
            return cv_transition(dt)
        if self.name == "ca":
            return ca_transition(dt)
        return turn_transition(dt, turn_rate(state))

    def noise(self, dt: float) -> np.ndarray:
        return cv_noise(self.q, dt) if self.name == "cv" else ca_noise(self.q, dt)

    @property
    def constrained(self) -> bool:
        return self.name == "turn"

    @property
    def has_acceleration(self) -> bool:
        return self.name != "cv"
