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

## Phase 3 — injection harness and evaluation suite

Built before the conformance detector. Every definition below was fixed before any detector
other than the baselines had been run (`config.yaml` `injection`, `evaluation`; code in
`evaluate/inject.py`, `evaluate/suite.py`, `evaluate/harness.py`).

### What is injected

Each kept flight gets one injection of each kind, on its own copy:
- the phase is drawn uniformly from the phases the flight can host;
- the onset is drawn uniformly within that phase, at least 300 s after the first fix and 300 s
  before the last;
- the magnitude and sign are drawn uniformly.

The draw is seeded by the flight key, so it doesn't depend on processing order. The
perturbation is applied to the real rows, so real noise, gaps and manoeuvres remain.

| kind | what changes | magnitudes | source |
|---|---|---|---|
| lateral | heading offset, turned into at 1.5 deg/s and held; each later displacement step of the real flight rotated by it | 2, 5, 10, 20, 30 deg | 01 sec. 5.3.1 (10 deg flight test), Fig. 5.10 (1.5 deg/s) |
| vertical | vertical rate offset, ramped in over 10 s; altitude kept on the 25 ft grid | 250, 500, 1000, 2000 ft/min | 03 sec. 5.3.1 (MONA unauthorised level change, wrong level) |
| longitudinal | speed offset ramped in at 0.5 m/s^2; the aircraft stays on the real path, ahead or behind | 10, 20, 40 kt | 03 glossary 124 (+-10 kt speed tolerance) |

### Definitions

- **Violation:** the injected deviation reaches
  - 1 NM cross-track or along-track (RNAV 1 / RNP 1, 03 Table 2 and sec. 3.3.1), or
  - 300 ft vertical (03 sec. 5.1.6).

  The deviation is measured against the same flight without the injection.
- **Detection:** an alert starts within 600 s of onset, at a row where the same detector on the
  uninjected flight was not exceeding the same threshold. Alerts the flight would have raised
  anyway are not credited.
- **False alert:** an alert on an uninjected flight. Real flights contain genuine deviations, so
  this is an upper bound on the false-alert rate.
- **Lead time:** violation time minus first detection time (03 sec. 5.1.5, "advanced delay
  time"). It is measured to the violation, not from onset, because the onset of a slow
  deviation is ill-defined (01 sec. 3.4.2).
- **Gradual event:** at least 120 s from onset to violation.
- **Alerts per event:** attributable alert starts within the 600 s window, averaged over
  detected events.
- **Operating points.** Both are chosen per phase on 2022-06-20 only and reported on
  2022-06-27:
  - *budget:* the lowest threshold whose alerts per flight-hour are within the budget
    (2.0, the brief's figure);
  - *fixed detection:* the highest threshold whose detection reaches 90%.

  Rows are thresholded by the phase the online system observes. Rows of unknown phase get the
  strictest phase threshold.
- **Phase of an event:** the phase at onset on the uninjected flight.

### Not used: accuracy, precision, recall

There is no natural count of negatives: every update of every uninjected flight is a negative,
so accuracy is dominated by them and reads ~100% for the never-alert baseline. Precision
depends on how many events were injected per flight-hour, which is a harness choice, not a
property of the airspace. Recall without a time window and without the alert rate beside it
says nothing an operator can use. The suite instead reports:
- alerts per flight-hour of uninjected traffic, per phase;
- detection within a window, per phase, kind and magnitude;
- lead time to violation;
- alerts per event.

### Results on the baselines

`make harness` (`results/phase3_never_alert.json`, `results/phase3_cv_kalman.json`). 3,406
flights and 9,036 injected events on 2022-06-20 choose the operating points; 3,370 flights and
9,006 events on 2022-06-27 are reported below. The single CV Kalman scores each update by
-log10 of its chi-square tail probability.

| detector | operating point | phase | false alerts / flight-hour | detection | median lead, gradual (s) | alerts per detected event |
|---|---|---|---:|---:|---:|---:|
| never alert | either | climb, cruise, descent | 0 | 0 | — | — |
| CV Kalman | budget | climb | 1.67 | 0.092 | -10.9 | 1.21 |
| CV Kalman | budget | cruise | 0.92 | 0.122 | -60.7 | 1.21 |
| CV Kalman | budget | descent | 1.40 | 0.122 | 10.0 | 1.41 |
| CV Kalman | fixed detection (unreachable, lowest threshold) | climb | 33.55 | 0.668 | 159.9 | 2.23 |
| CV Kalman | fixed detection (unreachable, lowest threshold) | cruise | 16.73 | 0.758 | 190.0 | 1.80 |
| CV Kalman | fixed detection (unreachable, lowest threshold) | descent | 36.04 | 0.660 | 170.2 | 2.17 |

- **Budget point.** The thresholds chosen on the tune day are tail probabilities of 10^-891
  (climb, descent) and 10^-794 (cruise). Only innovations of thousands of sigma are rare enough
  to stay within 2 alerts per flight-hour, because turns alone produce them in a CV filter. At
  that point, the lateral detection rates are:
  - 20 deg: 0.31 climb, 0.49 cruise, 0.21 descent;
  - 30 deg: 0.71 climb, 0.78 cruise, 0.43 descent;
  - 10 deg and below: at most 0.03.

  Vertical detection is at most 0.04 at any magnitude, and longitudinal is at most 0.22. The
  median lead is negative in climb and cruise: the alert, when it comes, comes after the
  violation.
- **Fixed detection is unreachable.** Even the loosest threshold, a 10% tail probability,
  stays below 90% in every phase. Detection there barely depends on magnitude. A 2 deg lateral
  offset is detected 0.42 / 0.66 / 0.42 (climb / cruise / descent), close to the 0.70 / 0.80 /
  0.51 of a 30 deg one. At a threshold crossed by a few percent of all updates, any
  perturbation reshuffles which rows cross it, and a newly crossing row inside the window counts
  as attributable. Detection read without the alert rate beside it would call this a working
  detector.
- **Unknown phase.** Rows with no phase label (the first 30 s of a track and after gaps) raised
  164 alerts at the budget point and 2,038 at the loosest threshold, in 39.9 flight-hours. They
  are thresholded at the strictest phase threshold, as defined above, and are not in any
  phase's rate.
- **Run.** It took 155 s with 6 workers. Peak RSS was 345 MB in the main process and 170 MB in
  the largest worker.

### Defect found while running the suite

`scipy.stats.chi2.logsf` returns -inf above NIS ~1500. Every larger innovation then got the same
infinite score, so no finite threshold could remove it. On the tune day the CV Kalman's alert rate flattened at
7.4 / 3.3 / 5.3 per flight-hour (climb / cruise / descent) above threshold 500, and the budget
looked unreachable. `conform/tail.py` switches to the asymptotic series of the incomplete gamma
function where scipy underflows. `tests/test_tail.py` checks it against scipy where scipy is
finite, and against the closed form for even dof. The threshold grid was widened to 10^6, and
the results above are from the corrected run.
