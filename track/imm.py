"""Interacting multiple model filter, following 04_genovese_2001_imm.pdf pp. 615-616.

One update, in the paper's order: state interaction (mixing), the TURN constraint after
interaction, per-model Kalman prediction and update, the TURN constraint after the measurement
update, model probability update, state estimate combination.

Departures from the paper, each forced by something the paper does not cover:

- Update intervals here are irregular (10 s snapshots, coverage holes up to 600 s). The paper's
  switching matrix Pi is per filter cycle; it is defined here per nominal cycle `cycle_s` and
  raised to the power dt / cycle_s, so two 10 s steps and one 20 s step mix modes alike.
- The paper's likelihood text says S^j is "computed using the predicted covariance matrix" but
  the symbol printed is the mixed covariance P^0j. S^j here uses the predicted covariance
  Phi P^0j Phi^T + Q, as in the Kalman equations on p. 615 where S = H P~ H^T + R.
- The mixing normalizer is printed as psi-bar^j in its definition and as c-bar^j in the text and
  in the probability update; they are the same vector (the predicted model probabilities).
- Likelihoods are combined in log space; exp(-NIS/2) underflows for a gross outlier.
- The covariance update uses the Joseph form (I-KH) P~ (I-KH)^T + K R K^T, equal to the paper's
  (I-KH) P~ (p. 615) in exact arithmetic. After a coverage hole of several minutes the CA
  model's predicted position variance reaches ~1e12 m^2, and (I-KH) P~ then cancels to
  negative variances (seen on the sample after a 549 s gap).
- The position's timestamp (lastposupdate, the network's receive time) has its own error.
  A time error d moves the reported position by v*d along the velocity, so the position block
  of R is sd_pos^2 I + sd_time^2 v v^T: along-track innovations on the sample have a robust sd
  ~3x the cross-track one (results/phase2_measurement_noise.json), which an isotropic R
  cannot express.
- Measurements may lack some components (null velocity or altitude); H and R are cut to the
  rows present, and every model sees the same rows, so the likelihoods stay comparable.
- Gross data errors (altitude jumps of tens of km, vertical rates of hundreds of m/s within one
  10 s step) are rejected by a kinematic gate before the update; see `Gate`.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from track.models import ACC, POS, STATE_DIM, VEL, FilterModel, constrain_turn

# Measurement order: east, north, baro altitude, v_east, v_north, vertical rate.
MEAS_DIM = 6
_H_FULL = np.zeros((MEAS_DIM, STATE_DIM))
_H_FULL[[0, 1, 2], POS] = 1.0
_H_FULL[[3, 4, 5], VEL] = 1.0
# Components gated together: a horizontal position is accepted or rejected as a pair.
_GATE_GROUPS = ((0, 1), (2,), (3, 4), (5,))


class Switching:
    """The switching matrix Pi per nominal cycle, and its fractional powers."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, float)
        eigval, eigvec = np.linalg.eig(self.matrix)
        if np.any(np.abs(eigval.imag) > 1e-12) or np.any(eigval.real <= 0.0):
            raise ValueError("switching matrix needs real positive eigenvalues for fractional powers")
        self._eigval = eigval.real
        self._eigvec = eigvec.real
        self._eigvec_inv = np.linalg.inv(eigvec.real)

    def power(self, cycles: float) -> np.ndarray:
        powered = (self._eigvec * self._eigval**cycles) @ self._eigvec_inv
        powered = np.clip(powered, 0.0, None)
        return powered / powered.sum(axis=1, keepdims=True)

    def stationary(self) -> np.ndarray:
        eigval, eigvec = np.linalg.eig(self.matrix.T)
        p = np.abs(eigvec[:, np.argmin(np.abs(eigval - 1.0))].real)
        return p / p.sum()


@dataclass(frozen=True, slots=True)
class Gate:
    """Rejects a measurement component that no aircraft motion could produce.

    A component is rejected when, under every model, its innovation exceeds `sigma` innovation
    standard deviations plus the error an unmodelled acceleration of `horizontal_accel_mps2`
    (or `vertical_accel_mps2`) would cause over dt. The envelope grows with dt^2, so a
    measurement after a long coverage hole is never rejected for having moved far. A rejected
    component is treated as missing and counted. A track whose state went wrong would reject
    every correct measurement after it, so after `reset_after` consecutive rejections of the same
    component group the track restarts from the current measurement.
    """

    sigma: float
    horizontal_accel_mps2: float
    vertical_accel_mps2: float
    reset_after: int

    def envelope(self, dt: float) -> np.ndarray:
        a_h, a_v = self.horizontal_accel_mps2, self.vertical_accel_mps2
        return np.array([0.5 * a_h * dt**2, 0.5 * a_h * dt**2, 0.5 * a_v * dt**2, a_h * dt, a_h * dt, a_v * dt])


@dataclass(frozen=True, slots=True)
class ImmConfig:
    models: tuple[FilterModel, ...]
    switching: Switching
    cycle_s: float
    measurement_sd: np.ndarray
    timestamp_sd_s: float
    init_acceleration_sd: float
    init_velocity_sd_missing: float
    gate: Gate | None
    mix_own_acceleration: bool = True

    @classmethod
    def from_config(cls, imm_cfg: dict) -> "ImmConfig":
        sd = imm_cfg["measurement_sd"]
        gate = imm_cfg.get("gate")
        return cls(
            models=tuple(FilterModel(name, float(imm_cfg["q"][name])) for name in imm_cfg["models"]),
            switching=Switching(imm_cfg["switching"]),
            cycle_s=float(imm_cfg["cycle_s"]),
            measurement_sd=np.array([
                sd["position_m"], sd["position_m"], sd["altitude_m"],
                sd["velocity_mps"], sd["velocity_mps"], sd["vertical_rate_mps"],
            ], float),
            timestamp_sd_s=float(sd["timestamp_s"]),
            init_acceleration_sd=float(imm_cfg["init"]["acceleration_sd_mps2"]),
            init_velocity_sd_missing=float(imm_cfg["init"]["velocity_sd_missing_mps"]),
            gate=Gate(**gate) if gate else None,
        )

    def only(self, name: str) -> "ImmConfig":
        """A single Kalman filter with one of the models; the IMM with N = 1 is exactly that."""
        model = next(m for m in self.models if m.name == name)
        return replace(self, models=(model,), switching=Switching(np.array([[1.0]])))


@dataclass(slots=True)
class ImmStep:
    """What one update produced.

    `nis` is the NIS of the model with the highest predicted probability (chosen before the
    measurement is seen), against its own innovation covariance. `mixture_nis` uses the
    moment-matched mixture of all models' predictions; that covariance carries the spread of
    improbable wide models, so it runs below chi-square in steady flight.
    """

    t: float
    state: np.ndarray
    cov: np.ndarray
    mode_prob: np.ndarray
    predicted_prob: np.ndarray
    nis: float
    mixture_nis: float
    dof: int
    model_nis: np.ndarray
    innovation: np.ndarray
    innovation_sd: np.ndarray
    rejected: np.ndarray
    restarted: bool


@dataclass(slots=True)
class ImmTrack:
    cfg: ImmConfig
    t: float = np.nan
    states: list[np.ndarray] = field(default_factory=list)
    covs: list[np.ndarray] = field(default_factory=list)
    mode_prob: np.ndarray = field(default_factory=lambda: np.empty(0))
    rejected_run: np.ndarray = field(default_factory=lambda: np.zeros(len(_GATE_GROUPS), np.int64))

    @property
    def started(self) -> bool:
        return bool(self.states)

    def start(self, t: float, z: np.ndarray, restarted: bool = False) -> ImmStep:
        """Initialise every model from one measurement; Genovese leaves this open (p. 616)."""
        sd = self.cfg.measurement_sd
        state = np.zeros(STATE_DIM)
        var = np.zeros(STATE_DIM)
        state[POS] = np.nan_to_num(z[:3])
        var[POS] = np.where(np.isnan(z[:3]), 1e6**2, sd[:3] ** 2)
        state[VEL] = np.nan_to_num(z[3:])
        var[VEL] = np.where(np.isnan(z[3:]), self.cfg.init_velocity_sd_missing**2, sd[3:] ** 2)
        var[ACC] = self.cfg.init_acceleration_sd**2
        cov = np.diag(var)
        self.t = t
        self.states = [state.copy() for _ in self.cfg.models]
        self.covs = [cov.copy() for _ in self.cfg.models]
        self.mode_prob = self.cfg.switching.stationary()
        self.rejected_run = np.zeros(len(_GATE_GROUPS), np.int64)
        n = len(self.cfg.models)
        missing = np.full(MEAS_DIM, np.nan)
        return ImmStep(t, state, cov, self.mode_prob.copy(), self.mode_prob.copy(), np.nan, np.nan, 0,
                       np.full(n, np.nan), missing, missing.copy(), np.zeros(MEAS_DIM, bool), restarted)

    def measurement_cov(self, z: np.ndarray) -> np.ndarray:
        r = np.diag(self.cfg.measurement_sd**2)
        velocity = z[3:6].copy()
        known = ~np.isnan(velocity)
        combined = sum(p * x for p, x in zip(self.mode_prob, self.states))
        velocity[~known] = combined[VEL][~known]
        r[:3, :3] += self.cfg.timestamp_sd_s**2 * np.outer(velocity, velocity)
        return r

    def update(self, t: float, z: np.ndarray) -> ImmStep:
        if not self.started:
            return self.start(t, z)
        dt = t - self.t
        if dt <= 0.0:
            raise ValueError(f"measurement time {t} not after track time {self.t}")
        n = len(self.cfg.models)
        r_full = self.measurement_cov(z)

        # State interaction (p. 616).
        pi = self.cfg.switching.power(dt / self.cfg.cycle_s)
        c_bar = self.mode_prob @ pi
        mix = pi * self.mode_prob[:, None] / c_bar[None, :]
        predicted_states, predicted_covs = [], []
        for j, model in enumerate(self.cfg.models):
            sources = [self._as_seen_by(i, j) for i in range(n)]
            x0 = sum(mix[i, j] * sources[i][0] for i in range(n))
            p0 = sum(mix[i, j] * (sources[i][1] + np.outer(sources[i][0] - x0, sources[i][0] - x0)) for i in range(n))
            if model.constrained:
                x0, p0 = constrain_turn(x0, p0)
            phi = model.transition(dt, x0)
            predicted_states.append(phi @ x0)
            predicted_covs.append(phi @ p0 @ phi.T + model.noise(dt))

        rejected = self._gate(z, dt, r_full, predicted_states, predicted_covs)
        group_rejected = np.array([rejected[g[0]] for g in _GATE_GROUPS])
        self.rejected_run = np.where(group_rejected, self.rejected_run + 1, 0)
        if self.cfg.gate is not None and np.any(self.rejected_run >= self.cfg.gate.reset_after):
            return self.start(t, z, restarted=True)
        rows = np.flatnonzero(~np.isnan(z) & ~rejected)
        if len(rows) == 0:
            return self._coast(t, c_bar, predicted_states, predicted_covs, rejected)
        h = _H_FULL[rows]
        r = r_full[np.ix_(rows, rows)]
        zr = z[rows]

        # Mode-matched filtering (p. 615) and likelihoods (p. 616).
        log_like = np.empty(n)
        model_nis = np.empty(n)
        predicted_z, predicted_s = [], []
        states, covs = [], []
        for j, model in enumerate(self.cfg.models):
            x_pred, p_pred = predicted_states[j], predicted_covs[j]
            s = h @ p_pred @ h.T + r
            s_inv = np.linalg.inv(s)
            innovation = zr - h @ x_pred
            gain = p_pred @ h.T @ s_inv
            x_upd = x_pred + gain @ innovation
            reduce = np.eye(STATE_DIM) - gain @ h
            p_upd = reduce @ p_pred @ reduce.T + gain @ r @ gain.T
            p_upd = 0.5 * (p_upd + p_upd.T)
            if model.constrained:
                x_upd, p_upd = constrain_turn(x_upd, p_upd)
            model_nis[j] = float(innovation @ s_inv @ innovation)
            _, logdet = np.linalg.slogdet(2.0 * np.pi * s)
            log_like[j] = -0.5 * (logdet + model_nis[j])
            predicted_z.append(h @ x_pred)
            predicted_s.append(s)
            states.append(x_upd)
            covs.append(p_upd)

        # Model probability update (p. 616): mu^j = Lambda^j c-bar^j / c.
        log_post = log_like + np.log(np.maximum(c_bar, 1e-300))
        post = np.exp(log_post - log_post.max())
        mode_prob = post / post.sum()

        # State estimate combination (p. 616).
        state = sum(mode_prob[i] * states[i] for i in range(n))
        cov = sum(mode_prob[i] * (covs[i] + np.outer(states[i] - state, states[i] - state)) for i in range(n))

        z_mix = sum(c_bar[j] * predicted_z[j] for j in range(n))
        s_mix = sum(c_bar[j] * (predicted_s[j] + np.outer(predicted_z[j] - z_mix, predicted_z[j] - z_mix)) for j in range(n))
        resid = zr - z_mix
        mixture_nis = float(resid @ np.linalg.solve(s_mix, resid))
        expected = int(np.argmax(c_bar))
        innovation = np.full(MEAS_DIM, np.nan)
        innovation_sd = np.full(MEAS_DIM, np.nan)
        innovation[rows] = zr - predicted_z[expected]
        innovation_sd[rows] = np.sqrt(np.diag(predicted_s[expected]))

        self.t = t
        self.states, self.covs, self.mode_prob = states, covs, mode_prob
        return ImmStep(t, state, cov, mode_prob.copy(), c_bar, float(model_nis[expected]), mixture_nis, len(rows),
                       model_nis, innovation, innovation_sd, rejected, False)

    def _as_seen_by(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Model i's estimate as mixed into model j. A model without acceleration (CV) has no
        estimate of it to contribute, so j's own acceleration estimate stands in, uncorrelated
        with i's position and velocity (track/models.py explains why not zero)."""
        if self.cfg.models[i].has_acceleration or not self.cfg.models[j].has_acceleration or not self.cfg.mix_own_acceleration:
            return self.states[i], self.covs[i]
        x = self.states[i].copy()
        x[ACC] = self.states[j][ACC]
        p = self.covs[i].copy()
        p[ACC, :] = 0.0
        p[:, ACC] = 0.0
        p[np.ix_(ACC, ACC)] = self.covs[j][np.ix_(ACC, ACC)]
        return x, p

    def _coast(self, t: float, c_bar: np.ndarray, states: list[np.ndarray], covs: list[np.ndarray], rejected: np.ndarray) -> ImmStep:
        n = len(states)
        state = sum(c_bar[i] * states[i] for i in range(n))
        cov = sum(c_bar[i] * (covs[i] + np.outer(states[i] - state, states[i] - state)) for i in range(n))
        self.t = t
        self.states, self.covs, self.mode_prob = states, covs, c_bar
        missing = np.full(MEAS_DIM, np.nan)
        return ImmStep(t, state, cov, c_bar.copy(), c_bar.copy(), np.nan, np.nan, 0, np.full(n, np.nan), missing, missing.copy(), rejected, False)

    def _gate(self, z: np.ndarray, dt: float, r: np.ndarray, states: list[np.ndarray], covs: list[np.ndarray]) -> np.ndarray:
        rejected = np.zeros(MEAS_DIM, bool)
        gate = self.cfg.gate
        if gate is None:
            return rejected
        present = ~np.isnan(z)
        envelope = gate.envelope(dt)
        outside = np.ones(MEAS_DIM, bool)
        for x, p in zip(states, covs):
            sd = np.sqrt(np.diag(_H_FULL @ p @ _H_FULL.T) + np.diag(r))
            with np.errstate(invalid="ignore"):
                outside &= np.abs(z - _H_FULL @ x) > gate.sigma * sd + envelope
        for group in _GATE_GROUPS:
            idx = list(group)
            if np.any(outside[idx] & present[idx]):
                rejected[idx] = present[idx]
        return rejected


def run_track(cfg: ImmConfig, t: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    """Run one flight through a fresh track; z is (N, 6) with NaN for missing components."""
    track = ImmTrack(cfg)
    n = len(t)
    m = len(cfg.models)
    out = {
        "state": np.empty((n, STATE_DIM)),
        "pos_var": np.empty((n, 3)),
        "mode_prob": np.empty((n, m)),
        "predicted_prob": np.empty((n, m)),
        "nis": np.empty(n),
        "mixture_nis": np.empty(n),
        "dof": np.empty(n, np.int64),
        "model_nis": np.empty((n, m)),
        "innovation": np.empty((n, MEAS_DIM)),
        "innovation_sd": np.empty((n, MEAS_DIM)),
        "rejected": np.empty((n, MEAS_DIM), bool),
        "restarted": np.empty(n, bool),
    }
    for k in range(n):
        step = track.update(float(t[k]), z[k])
        out["state"][k] = step.state
        out["pos_var"][k] = np.diag(step.cov)[POS]
        out["mode_prob"][k] = step.mode_prob
        out["predicted_prob"][k] = step.predicted_prob
        out["nis"][k] = step.nis
        out["mixture_nis"][k] = step.mixture_nis
        out["dof"][k] = step.dof
        out["model_nis"][k] = step.model_nis
        out["innovation"][k] = step.innovation
        out["innovation_sd"][k] = step.innovation_sd
        out["rejected"][k] = step.rejected
        out["restarted"][k] = step.restarted
    return out
