# Notes

Design decisions per phase, with the reference each rests on. Numbers come from `results/`.

## Phase 2 — IMM tracking filter

Source: `refs/04_genovese_2001_imm.pdf` (Genovese 2001, JHU APL Tech. Digest 22(4)). Method 5 of
its Table 1, the CV-CA-TURN IMM, with its 3x3 switching matrix. Code: `track/models.py`,
`track/imm.py`.

### What the paper gives, and where it is used

| paper | here |
|---|---|
| Kalman predict/update, p. 615 | `ImmTrack.update`, per model |
| state interaction (mixing), p. 616 | `ImmTrack.update`, first block |
| likelihood and model probability update, p. 616 | `ImmTrack.update`, log space |
| state estimate combination, p. 616 | `ImmTrack.update` |
| CV, CA, TURN transition and plant noise, p. 617 | `track/models.py` |
| TURN constant-speed pseudo-measurement, p. 618 | `constrain_turn`, applied after mixing and after the update |
| Table 1 switching matrix | `config.yaml` `imm.switching` |

### Paper inconsistencies

No page of the paper was unreadable, and no equation was garbled in the rendered PDF. The text
layer is garbled, so every equation was read from the page images. The equations have four
problems:

1. The mixing normalizer is defined as psi-bar^j, but the text and the probability update call
   it c-bar^j. Both denote the predicted model probabilities, so they are treated as one vector.
2. On the likelihood (p. 616), the text says S^j uses "the predicted covariance matrix", but the
   printed symbol is P-hat^0j, the mixed covariance. The predicted covariance is used, as in the
   p. 615 Kalman equations.
3. phi_TURN writes the interval as delta-tau, while CV and CA write delta-t. They are the same
   interval.
4. R^c is "selected ... to achieve a gain of 0.5". This is read as R^c = v P v^T, which makes
   the scalar gain exactly 0.5.

### Additions the paper does not cover

- **Mixing a 6-state CV into 9-state CA/TURN.** The paper does not say how. If CV's missing
  acceleration is entered as "zero with zero variance", the CA and TURN models are locked out.
  q_CA enters only the acceleration (p. 617), so their mixed acceleration variance stays near 0
  and they get no gain on acceleration. In 100 synthetic runs with the same seed, that
  zero-padded variant missed the 1.5 deg/s turn in 26 runs and the 3 deg/s turn in 22
  (`results/phase2_filter_checks.json`, `mode_delays_zero_padded`). Here CV's acceleration is
  instead filled with the receiving model's own estimate when mixing. With that fix every
  scenario was detected in 100 of 100 runs.
- **Irregular intervals.** Pi is defined per nominal 10 s cycle and raised to dt/10 through its
  eigendecomposition. Genovese's matrix has real positive eigenvalues (1, 0.83, 0.57).
- **Joseph-form covariance update.** It equals (I-KH)P~ in exact arithmetic. The paper's form
  produced negative variances after a 549 s coverage hole, where the CA position variance
  reaches ~1e12 m^2.
- **Measurement model.** The measurement is z = [east, north, baro altitude, v_east, v_north,
  vertical rate].
  - Velocity is rotated into the flight's tangent frame (`ingest/frame.py`). Without the
    rotation, meridian convergence puts the north component off by more than 5 m/s at
    51.2 N 11.5 E seen from 50 N 8 E (`tests/test_imm.py`).
  - The position block of R adds sd_time^2 v v^T for the error in lastposupdate. On steady
    flight, the robust sd of the innovation is 11.0 m along-track against 3.7 m cross-track.
- **Kinematic gate.** It rejects a component that no motion within 1 g horizontal or 0.5 g
  vertical, plus 6 sigma, could produce. After 3 consecutive rejections of the same component
  the track restarts.
- **Determinism.** The forward geodetic transforms use scalar libm, so online (chunked) and
  offline (whole-flight) runs see bit-identical measurements.

### Parameters

- q values: cv 0.05 m^2/s^3, ca 0.5 m^2/s^5, turn 0.05 m^2/s^5. They are chosen for transport
  aircraft, not taken from Genovese's Table 1, whose values are for military targets pulling
  5.6 g.
- Measurement sd: these are set on the tune day only (2022-06-20) and bounded by the robust sd of
  steady-flight innovations (`results/phase2_measurement_noise.json`):
  - position 3.5 m;
  - baro altitude 2.2 m (the 25 ft quantization);
  - velocity 0.5 m/s;
  - vertical rate 0.3 m/s;
  - timestamp 0.05 s.

### Verification results

**NIS consistency, synthetic.** Truth is drawn from the CV model with the filter's own q and R;
there are 100 runs of 1800 s each.

| filter | mean NIS | 95% band | above chi2(6) 95% |
|---|---:|---:|---:|
| single CV Kalman | 5.995 | [5.947, 6.053] | 4.9% |
| IMM, NIS of the expected model | 5.995 | [5.947, 6.053] | 4.9% |
| IMM, moment-matched mixture NIS | 3.09 | [5.947, 6.053] | 0.6% |

The mixture's covariance carries improbable wide models, so it is conservative by
construction. The expected-model NIS is the one used from here on.

**NIS on the real sample.** The second day is reported without having been tuned on.

| | climb | cruise | descent |
|---|---:|---:|---:|
| median NIS/dof (chi2: 0.89), 06-20 | 1.19 | 0.33 | 0.94 |
| above chi2 95% (chi2: 5%), 06-20 | 34.5% | 18.2% | 28.4% |
| median NIS/dof, 06-27 | 1.06 | 0.27 | 0.94 |
| above chi2 95%, 06-27 | 30.8% | 12.6% | 26.1% |

Real innovations have a core narrower than chi-square and a far heavier tail, and both depend
on the flight phase. A Gaussian R cannot match both. A raw NIS threshold read off chi-square
tables would therefore alert on 13-31% of updates; thresholds have to be set empirically per
phase.

**Mode switching delays, synthetic.** These are 100 runs per scenario. "Leave CV" is the first
update with P(CV) < 0.5 after maneuver start; "back to CV" is the first update with
P(CV) > 0.5 after maneuver end.

| scenario | detected | leave CV median / max (s) | back to CV median / max (s) | false switches in straight flight |
|---|---:|---:|---:|---:|
| 1.5 deg/s turn, 230 m/s | 100/100 | 7.9 / 13.3 | 11.6 / 17.6 | 0 |
| 3 deg/s turn, 150 m/s | 100/100 | 7.5 / 22.0 | 11.3 / 25.8 | 0 |
| climb onset to 10 m/s | 100/100 | 8.6 / 15.4 | 12.7 / 18.8 | 0 |
| 0.5 m/s^2 speed change | 100/100 | 14.3 / 27.3 | 10.4 / 18.5 | 0 |

**Mode switching on real data.** Onsets are read off the reported track and vertical rate.

- 1,261 turn onsets: 1,247 switched within 60 s. Of those, 69.6% switched at the same update the
  heading first showed the turn and 29.8% one update later.
- 2,185 vertical onsets: 1,909 switched within 60 s. Of those, 80.2% switched at the same update
  and 16.2% one update later.

**Streaming equivalence.** On the first 50,000 rows (596 flights), the online bank in 2,500-row
chunks was compared with whole-flight runs. The largest absolute difference in state, NIS and
mode probability was exactly 0.

**Gate and memory.** Over 1,231,395 updates:
- 750 horizontal positions and 1,795 baro altitudes were rejected;
- 106 tracks restarted;
- peak RSS was 362 MB in the main process and 163 MB in the largest worker.
