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
  - *budget:* of the thresholds whose alerts per flight-hour are within the budget (2.0, the
    brief's figure), the one with the highest detection (see "Operating-point rule" in phase 4
    for why this replaced "the lowest threshold within budget");
  - *fixed detection:* the highest threshold whose detection reaches 90%; where none does, the
    threshold of highest detection, reported as unreachable.

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
| CV Kalman | fixed detection (unreachable, highest detection) | climb | 33.83 | 0.650 | 155.3 | 2.22 |
| CV Kalman | fixed detection (unreachable, highest detection) | cruise | 16.87 | 0.748 | 190.0 | 1.76 |
| CV Kalman | fixed detection (unreachable, highest detection) | descent | 32.08 | 0.676 | 189.7 | 2.16 |

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
- **Fixed detection is unreachable.** No threshold reaches 90% in any phase; the best are a
  10% tail probability in climb and cruise and 1% in descent. Detection there barely depends on magnitude. A 2 deg lateral
  offset is detected 0.42 / 0.66 / 0.50 (climb / cruise / descent), close to the 0.70 / 0.79 /
  0.55 of a 30 deg one. At a threshold crossed by a few percent of all updates, any
  perturbation reshuffles which rows cross it, and a newly crossing row inside the window counts
  as attributable. Detection read without the alert rate beside it would call this a working
  detector.
- **Unknown phase.** Rows with no phase label (the first 30 s of a track and after gaps) raised
  164 alerts at the budget point and 1,862 at the fixed-detection point, in 39.9 flight-hours. They
  are thresholded at the strictest phase threshold, as defined above, and are not in any
  phase's rate.
- **Run.** Peak RSS was 404 MB in the main process and 184 MB in the largest worker (6 workers).

### Defect found while running the suite

`scipy.stats.chi2.logsf` returns -inf above NIS ~1500. Every larger innovation then got the same
infinite score, so no finite threshold could remove it. On the tune day the CV Kalman's alert
rate flattened at 7.4 / 3.3 / 5.3 per flight-hour (climb / cruise / descent) above threshold
500, and the budget looked unreachable. `conform/tail.py` switches to the asymptotic series of the incomplete gamma
function where scipy underflows. `tests/test_tail.py` checks it against scipy where scipy is
finite, and against the closed form for even dof. The threshold grid was widened to 10^6, and
the results above are from the corrected run.

## Phase 4 — reference paths, conformance scoring, alert management

Sources: `01_reynolds_hansman_conformance_fault_detection.pdf` (conformance residuals, eq.
3.2-3.7; heading-based detection, sec. 5.3.1), `02_reynolds_hansman_structure_intent_conformance.pdf`
(standard flows and their ingress and egress points, sec. 4.1; conformance basis, sec. 5.1).
Code: `reference/`, `conform/features.py`, `conform/score.py`, `conform/monitor.py`,
`alerts/manage.py`, `baselines/fixed_threshold.py`, `baselines/eta_deviation.py`.

### Origin-destination pairs and references

Most flights in the region are overflights, and flight plans are not in the data. A flight's key
is how it enters and leaves the region (`reference/od.py`):
- an airport, if the endpoint is below 1,500 m. Airports are clusters of tune-day endpoints,
  linked at 5 km;
- a boundary gate, a 50 km slot of the side it crosses within 30 km of;
- none, if the flight starts or ends mid-air inside the region.

The key is taken from the whole uninjected flight, standing in for the flight plan, which is
known before departure. Conformance is measured against these observed norms, not filed
intent.

`results/phase4_references.json`:

| | tune day (06-20) | test day (06-27) |
|---|---:|---:|
| flights | 3,406 | 3,370 |
| with a key | 3,021 | 2,837 |
| no key (an endpoint mid-air) | 385 | 533 |
| with a reference | 1,416 (41.6%) | 1,096 (32.5%) |

- The tune day has 1,253 pairs. 152 have at least 5 flights and get a reference. The other
  1,101 pairs, holding 1,605 tune-day flights, are excluded.
- References have 5 to 37 flights (median 7) and are 7 to 743 km long (median 359 km).
- Every phase 4 result below is on the flights with a reference, for all five detectors alike.

**Median path.** Member flights are resampled at 100 common fractions of their own length in
the region-centre frame. The point-by-point median is then spaced every 2 km.

**Norms.** Members are run through the IMM and followed along the path
(`conform/features.py`). This is the same code the monitor runs. At each station:
- the median and robust sd of five quantities: cross-track, track angle error, altitude,
  vertical rate and ground speed;
- the sd is pooled over +-1 station and shrunk toward the path-wide sd with n0 = 5 flights;
- expected time along the path is the median station-to-station time;
- the ETA sd is sqrt(floor^2 + (rel x expected)^2), both terms fitted on the members.

Median station sds across references are 4.2 km cross-track, 3.4 deg track angle, 546 m
altitude, 0.38 m/s vertical rate and 8.4 m/s ground speed. Gates 50 km wide put several real
flows into one pair, which is why the cross-track spread is kilometres, not hundreds of metres.

### Score

Six residuals are computed: the five quantities above minus their station median, over the
station sd, plus the ETA error over its sd. This is the vector conformance residual of 01 eq.
3.7, normalised as in eq. 3.4 by observed conforming behaviour.

Their tails differ by orders of magnitude on the tune-day members:

| residual | p99 | p99.9 |
|---|---:|---:|
| ground speed | 6.3 | 16.6 |
| altitude | 87.9 | 1,465 |

Each residual is therefore mapped to -log10 of its empirical exceedance probability, with an
exponential tail beyond the sample. The score is the largest of the six. Two checks:
- **Streaming.** Monitor scores on 300 test-day flights (44,932 rows), pushed in random chunks
  of 1-40 rows, equal whole-flight runs exactly: the worst gap is 0 for both score and
  residuals (`results/phase4_streaming_equivalence.json`).
- **Synthetic.** On a synthetic reference, a 10 deg turn-off scores above 6 within 30 s, long
  before it reaches 1 NM (`tests/test_conformance.py`).

### Alert management, the same for every detector

`alerts/manage.py`, `config.yaml` `alerts`:
- **Persistence:** 2 consecutive exceedances.
- **Hold-off:** an alert stays open until 300 s pass without an exceedance.
- **Re-presentation:** an alert still open is presented again every 300 s and counted again.

These were fixed from the phase 2 mode-switch delays and the 10 s cadence. The never-alert
baseline needs none.

### Operating-point rule, and two defects in counting

Running managed detectors exposed two holes in the phase 3 definitions. Both were fixed before
the numbers below, and the phase 3 baselines were re-run under the corrected rule; only their
fixed-detection rows moved.

1. **The budget rule picked always-on detectors.** "The lowest threshold within budget" took
   100 m for the fixed distance in cruise and descent, and 1 s for the ETA deviation. At those
   thresholds every flight is in alert from its first rows, which carry no phase, so the phase
   rates looked near zero and detection was 0.
   - Neither alert rate nor detection is monotone in the threshold once alerts are held open.
   - The budget point is now, of the thresholds within budget on the tune day, the one with
     the highest tune-day detection.
2. **An alert held open for a whole flight counted once.** With only hold-off, the IMM
   conformance detector met the budget on the tune day while spending 51-64% of monitored time
   in alert on the test day. That left no room to detect anything.
   - Re-presentation, above, counts such an alert every 300 s.
   - Every report now also gives `share_of_time_in_alert`.

### Results: alert rate against detection rate, per phase, five detectors

`results/phase4_<detector>.json`, `figures/phase4_alert_vs_detection.png`. 1,416 tune-day
flights (3,612 events) choose the thresholds. The table is 1,096 test-day flights (2,871
events) at the budget operating point.

| detector | phase | false alerts / flight-hour | detection | time in alert | median lead, gradual (s) | alerts per detected event |
|---|---|---:|---:|---:|---:|---:|
| never alert | climb / cruise / descent | 0 / 0 / 0 | 0 / 0 / 0 | 0 | — | — |
| fixed distance | climb | 2.42 | 0.184 | 0.153 | -79.9 | 1.36 |
| fixed distance | cruise | 2.33 | 0.169 | 0.164 | -179.3 | 1.42 |
| fixed distance | descent | 2.53 | 0.163 | 0.176 | -90.0 | 1.34 |
| single CV Kalman | climb | 1.54 | 0.076 | 0.100 | -145.2 | 1.08 |
| single CV Kalman | cruise | 1.27 | 0.144 | 0.067 | -54.9 | 1.20 |
| single CV Kalman | descent | 2.14 | 0.115 | 0.148 | -79.8 | 1.14 |
| ETA deviation | climb | 3.47 | 0.111 | 0.242 | -120.2 | 1.05 |
| ETA deviation | cruise | 5.38 | 0.099 | 0.370 | -115.1 | 1.04 |
| ETA deviation | descent | 3.72 | 0.126 | 0.281 | -109.4 | 1.18 |
| IMM conformance | climb | 4.02 | 0.166 | 0.307 | -59.3 | 1.28 |
| IMM conformance | cruise | 5.02 | 0.262 | 0.355 | 39.7 | 1.56 |
| IMM conformance | descent | 3.43 | 0.149 | 0.250 | -120.0 | 1.26 |

Detection by magnitude at the budget point, IMM conformance against fixed distance (climb /
cruise / descent):

| injection | IMM conformance | fixed distance |
|---|---|---|
| lateral 10 deg | 0.22 / 0.30 / 0.08 | 0.67 / 0.61 / 0.32 |
| lateral 30 deg | 0.49 / 0.50 / 0.51 | 0.72 / 0.75 / 0.77 |
| vertical 2000 ft/min | 0.08 / 0.45 / 0.11 | 0.00 / 0.01 / 0.09 |
| longitudinal 40 kt | 0.20 / 0.28 / 0.25 | 0.06 / 0.03 / 0.06 |

### Reading the results

- **Every success criterion is missed, by every detector, in every phase.** The fixed-detection
  point is unreachable everywhere: no threshold reaches 90% on the tune day. The best
  detection anywhere on the test-day sweep is 0.32 (CV Kalman, cruise, at 7 alerts per
  flight-hour).
- **The IMM conformance detector does not clearly beat the fixed distance threshold.**
  - It is the only one to detect vertical deviations in cruise (0.45 at 2,000 ft/min). It is
    the best on longitudinal ones (40 kt: 0.20-0.28; ETA deviation 0.14-0.22). It is the only
    one with a positive median lead (40 s, cruise).
  - It loses on lateral, the dimension the fixed distance threshold measures directly.
  - It overshoots the budget on the test day (3.4-5.0 against 2.0). The fixed distance
    overshoots only slightly (2.3-2.5).

  By the brief's own rule, the complexity has not yet earned its place.
- **Tune-to-test gap.** At the budget point every reference-based detector exceeds the budget
  on the test day. A tune-day flight is scored against a reference it helped build, and the
  tails are calibrated on the same flights. So the tune day sees fewer surprises than the test
  day, and the thresholds come out too low. This is the main methodological weakness of this
  phase; cross-fitting the references (build from half the tune day, choose thresholds on the
  other half) would remove it.
- **Why detection is capped.** Real flights depart from the norms of a 50 km-gate pair far more
  than the injected deviations do: 1% of normal fixes have a track error beyond 10 sd. An
  injected event is also credited only if it starts an alert on a flight that is not already in
  one, and at these thresholds flights are in alert 25-36% of the time.
- **Coverage.** Only 32.5% of test-day flights have a reference.
