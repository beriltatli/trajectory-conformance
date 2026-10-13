/# Notes

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

Their tails differ by orders of magnitude (held-out tune-day residuals, see cross-fitting below):

| residual | p99 | p99.9 |
|---|---:|---:|
| ground speed | 9.8 | 38.4 |
| altitude | 677 | 4,681 |

Each residual is therefore mapped to -log10 of its empirical exceedance probability, with an
exponential tail beyond the sample. The score is the largest of the six. Two checks:
- **Streaming.** Monitor scores on 300 test-day flights (44,932 rows), pushed in random chunks
  of 1-40 rows, equal whole-flight runs exactly: the worst gap is 0 for both score and
  residuals (`results/phase4_streaming_equivalence.json`).
- **Synthetic.** On a synthetic reference, a 10 deg turn-off scores above 6 within 30 s, long
  before it reaches 1 NM (`tests/test_conformance.py`).

### Cross-fitting (after Gate 4)

The first run scored every tune-day flight against a reference it was a member of, and fitted
the tails on those same residuals. The thresholds chosen on them were too low for flights the
references had never seen: the IMM monitor ran at 3.4-5.0 false alerts per flight-hour on the
test day against a budget of 2. `reference/build.py` now:
- scores each tune-day flight against its pair's reference rebuilt without it (leave one out;
  a held-out reference has 4-36 flights, median 9);
- fits the tails on those held-out residuals in two folds of 708 flights each, and scores a
  flight with the tails of the fold it is not in;
- gives test-day flights the reference built from all members and the tails of both folds.

A half split of the flights was rejected: 41 of the 152 pairs have 5 flights and would be left
with 2 or 3. Build time is 145 s (was ~40 s).

How much the reuse had hidden, p99 of |residual| in sd (`results/phase4_references.json`,
`residual_quantiles`):

| residual | in sample | held out |
|---|---:|---:|
| cross-track | 14.0 | 22.8 |
| track error | 10.6 | 16.7 |
| altitude | 87.9 | 677 |
| vertical rate | 32.4 | 44.7 |
| ground speed | 6.3 | 9.8 |
| ETA | 16.4 | 27.6 |

Tail resolution (`calibration`). 241,151 held-out fixes resolve exceedances down to 1 in
241,151, a score of 5.38; above it the score is extrapolated. Fixes 10 s apart are far from
independent, so the 1,416 flights behind them are the better count. The two folds' tails agree
within 10% at a score of 2 for cross-track, track error, altitude and vertical rate, but not for
ground speed (7.8 against 14.5 sd) or ETA (14.3 against 67.1 sd), whose tails a few flights
decide. Above a score of 3 they disagree several-fold (cross-track at 4: 1,860 against 412 sd).
The budget thresholds sit at scores of 1.85-2.0.

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
flights (3,612 events) choose the thresholds, cross-fitted as above. The table is 1,096 test-day
flights (2,871 events) at the budget operating point.

| detector | phase | false alerts / flight-hour | detection | time in alert | median lead, gradual (s) | alerts per detected event |
|---|---|---:|---:|---:|---:|---:|
| never alert | climb / cruise / descent | 0 / 0 / 0 | 0 / 0 / 0 | 0 | — | — |
| fixed distance | climb | 2.20 | 0.184 | 0.132 | -115.0 | 1.35 |
| fixed distance | cruise | 1.70 | 0.172 | 0.123 | -189.7 | 1.36 |
| fixed distance | descent | 2.21 | 0.147 | 0.152 | -139.8 | 1.36 |
| single CV Kalman | climb | 1.54 | 0.076 | 0.100 | -145.2 | 1.08 |
| single CV Kalman | cruise | 1.27 | 0.144 | 0.067 | -54.9 | 1.20 |
| single CV Kalman | descent | 2.14 | 0.115 | 0.148 | -79.8 | 1.14 |
| ETA deviation | climb | 2.42 | 0.096 | 0.173 | -139.9 | 1.06 |
| ETA deviation | cruise | 4.85 | 0.084 | 0.326 | -50.1 | 1.04 |
| ETA deviation | descent | 3.03 | 0.140 | 0.238 | -155.1 | 1.08 |
| IMM conformance | climb | 2.55 | 0.122 | 0.213 | -120.2 | 1.24 |
| IMM conformance | cruise | 2.79 | 0.233 | 0.185 | -74.7 | 1.63 |
| IMM conformance | descent | 2.15 | 0.103 | 0.152 | -189.8 | 1.38 |

The single CV Kalman uses no reference and did not change. Before cross-fitting the IMM monitor
read 4.02 / 5.02 / 3.43 false alerts per flight-hour and detected 0.166 / 0.262 / 0.149; its
thresholds (scores 2.0 / 1.85 / 2.0) did not move, because a score means the same rarity under
either calibration, but test-day flights now score lower against the wider held-out tails.

Detection by magnitude at the budget point (climb / cruise / descent):

| injection | IMM conformance | fixed distance | ETA deviation |
|---|---|---|---|
| lateral 10 deg | 0.13 / 0.14 / 0.05 | 0.59 / 0.58 / 0.35 | 0.06 / 0.09 / 0.08 |
| lateral 30 deg | 0.38 / 0.49 / 0.31 | 0.74 / 0.79 / 0.77 | 0.19 / 0.21 / 0.21 |
| vertical 2000 ft/min | 0.10 / 0.52 / 0.09 | 0.02 / 0.03 / 0.00 | 0.06 / 0.01 / 0.13 |
| longitudinal 40 kt | 0.11 / 0.17 / 0.09 | 0.02 / 0.02 / 0.08 | 0.11 / 0.21 / 0.25 |

### Counting a held alert once, against re-presentation every 300 s

Gate 4 asked for the comparison (`results/phase4_imm_conformance_single_count.json`). The IMM
monitor with alerts never re-presented, each setting at its own tune-day budget point, test day:

| phase | counting | threshold | false alerts / flight-hour | time in alert | events starting inside an open alert | detection | median lead, gradual (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| climb | every 300 s | 2.00 | 2.55 | 0.213 | 0.291 | 0.122 | -120.2 |
| climb | once | 1.47 | 1.98 | 0.425 | 0.535 | 0.114 | 0.1 |
| cruise | every 300 s | 1.85 | 2.79 | 0.185 | 0.177 | 0.233 | -74.7 |
| cruise | once | 1.58 | 1.35 | 0.364 | 0.370 | 0.264 | 25.2 |
| descent | every 300 s | 2.00 | 2.15 | 0.152 | 0.211 | 0.103 | -189.8 |
| descent | once | 1.36 | 2.73 | 0.406 | 0.446 | 0.170 | -20.1 |

Counted once, the budget admits lower thresholds, and flights spend about twice as long in
alert (36-43% of monitored time) while the reported rate falls (2.10 per flight-hour over all
phases, against 2.77). 37-54% of injected events then start inside an alert already open and
cannot be credited. Detection does not collapse: the lower threshold catches more of the events
that start outside an alert, so it comes out similar (climb) or higher (cruise, descent). The
detector is blind on about half the events, and the alert rate stops measuring the time
controllers spend on alerts. Re-presentation stays.

### Reading the results

- **Every success criterion is missed, by every detector, in every phase.** The fixed-detection
  point is unreachable everywhere: no threshold reaches 90% on the tune day. The best
  detection anywhere on the test-day sweep is 0.32 (CV Kalman, cruise, at 6.9 alerts per
  flight-hour); the IMM monitor's best is 0.27 (cruise, at 5.9).
- **The IMM conformance detector does not beat the fixed distance threshold.**
  - It is the only one to detect vertical deviations in cruise (0.52 at 2,000 ft/min).
  - On longitudinal deviations the ETA deviation baseline now does as well or better (40 kt:
    0.11-0.25 against 0.09-0.17). Before cross-fitting the IMM monitor led there; part of that
    lead came from the in-sample references.
  - It loses on lateral, the dimension the fixed distance threshold measures directly.
  - No detector has a positive median lead for gradual events.
  - Every reference-based detector except the fixed distance in cruise still exceeds the budget
    on the test day (IMM 2.15-2.79, ETA 2.42-4.85, fixed distance 2.20-2.21 in climb and
    descent). The overshoot is now a week of change between the two days, not reuse of flights.

  By the brief's own rule, the complexity has not earned its place.
- **Why detection is capped.** Real flights depart from the norms of a 50 km-gate pair far more
  than the injected deviations do: 1% of held-out fixes have a cross-track residual beyond
  22.8 sd. An injected event is also credited only if it starts an alert on a flight that is not
  already in one, and at the budget point 18-29% of the IMM monitor's events start inside one.
- **Coverage.** Only 32.5% of test-day flights have a reference.

## Phase 5 — drift: per-flight adaptation and what it absorbs

Code: `drift/adapt.py` (offset and guard), `drift/calibrate.py` (tails per setting),
`drift/detector.py`, `evaluate/drift.py`; `config.yaml` `drift`. Run: `make drift`.

### What adapts, and why at the flight level

Drift here is a flight's slow departure from the norms of its pair. Within a pair keyed by
50 km gates, a flight can hold its own flow, flight level or speed for its whole life; the
phase 4 monitor scores that as a deviation from the first row to the last. The adaptation
subtracts a running estimate of each residual component's offset. It is an exponentially
weighted mean with time constant tau, from rows before the current one only. The price is the
one the brief names: a deviation that grows more slowly than the adaptation is learnt as offset
and never alerted on, which is how an uncorrected inertial drift (01 sec. 1.3.2, Fig. 1.5)
would pass.

Drift between days, the week between tune and test day, is not adapted here. It is what still
puts most detectors over the budget on the test day.

**Guard** (`drift/adapt.py`):
- Freeze: a row whose adapted residual scores above 2.0 under the unadapted tails (rarer than 1
  in 100 held-out fixes) does not update the offset.
- Cap: the offset is clipped at 2 sd per component. 03 sec. 5.1.8 takes mu +- 1.96 sigma of
  conforming flights as the conformance range, so a flight may learn an offset inside the range
  normal traffic occupies, not beyond it.

**Settings**: tau = 60, 120, 300, 600, 1800 s, each unguarded and guarded, plus the phase 4
monitor (tau = inf). All were fixed in `config.yaml` before any run. Adaptation changes the
residual distribution, so each setting has its tails refitted on the held-out tune-day residuals
in the same two folds as phase 4 (`results/phase5_tails.json`).

### Two measurements

The first plan read each setting at its own budget point. That cannot show what adaptation
removes. The score is a rarity, refitted per setting, so a threshold near 2 flags about 1% of
fixes whatever the adaptation does to their size; only how the exceedances cluster in time can
change. Both sides are therefore also measured directly, on one fixed scale: the unadapted tails
and the unadapted budget thresholds (2.00 / 1.85 / 2.00), without alert management
(`results/phase5_drift_absorption.json`).
- **Speed side.** The share of monitored time on uninjected test-day flights whose adapted
  residuals still exceed the threshold: how much of a flight's own off-norm behaviour is learnt
  away.
- **Absorption side.** The injected events that the unadapted residuals show before violation
  (an exceeding row where the uninjected flight does not exceed). Of these, the share that the
  adapted residuals no longer show before violation.

Same 1,096 test-day flights and 2,871 injected events as phase 4.

### Results

The trade-off curve is `figures/phase5_drift_tradeoff.png`. The unadapted monitor shows only a
few slow (gradual) events before violation: 19 of 209 in climb, 64 of 537 in cruise and 13 of
183 in descent. The absorbed shares rest on those counts.

| setting | off-norm time, climb / cruise / descent | slow events hidden, climb / cruise / descent | abrupt events hidden, cruise |
|---|---|---|---:|
| no adaptation | 0.127 / 0.091 / 0.089 | 0 / 0 / 0 | 0 |
| unguarded, 60 s | 0.070 / 0.026 / 0.063 | 0.58 / 0.53 / 0.54 | 0.14 |
| unguarded, 300 s | 0.106 / 0.043 / 0.080 | 0.68 / 0.39 / 0.54 | 0.10 |
| unguarded, 1800 s | 0.110 / 0.067 / 0.085 | 0.42 / 0.19 / 0.54 | 0.06 |
| guarded, 60 s | 0.121 / 0.067 / 0.082 | 0.21 / 0.13 / 0.23 | 0.07 |
| guarded, 300 s | 0.122 / 0.068 / 0.083 | 0.21 / 0.11 / 0.23 | 0.04 |
| guarded, 1800 s | 0.124 / 0.071 / 0.084 | 0.16 / 0.09 / 0.23 | 0.03 |

At each setting's own budget point, with alert management (`results/phase5_drift_tradeoff.json`),
test day:

| setting | false alerts / flight-hour, climb / cruise / descent | detection, climb / cruise / descent | time in alert, cruise |
|---|---|---|---:|
| no adaptation | 2.55 / 2.79 / 2.15 | 0.122 / 0.233 / 0.103 | 0.185 |
| unguarded, 60 s | 2.81 / 2.24 / 2.20 | 0.102 / 0.205 / 0.097 | 0.158 |
| unguarded, 300 s | 3.16 / 1.99 / 2.65 | 0.105 / 0.212 / 0.113 | 0.139 |
| unguarded, 600 s | 3.13 / 1.88 / 2.64 | 0.108 / 0.208 / 0.112 | 0.136 |
| unguarded, 1800 s | 2.81 / 2.79 / 2.48 | 0.130 / 0.236 / 0.113 | 0.188 |
| guarded, 60 s | 2.49 / 2.71 / 2.78 | 0.130 / 0.236 / 0.103 | 0.180 |
| guarded, 300 s | 2.48 / 2.67 / 2.75 | 0.123 / 0.236 / 0.104 | 0.178 |
| guarded, 1800 s | 2.52 / 2.62 / 2.05 | 0.116 / 0.232 / 0.103 | 0.172 |

### Reading the results

- **The trade-off is real on the fixed scale.** Unguarded adaptation at 60 s removes 71% of
  cruise off-norm time (0.091 to 0.026). It also hides 53% of the slow events the unadapted
  monitor saw, and 14% of the abrupt ones. Slowing it to 1800 s keeps 74% of the off-norm time
  (0.067) and still hides 19% of the slow events.
- **The guard bounds the absorption, and the cap decides it.** Every guarded setting lands
  within half a percentage point of off-norm time of every other. Once the offset reaches 2 sd,
  tau no longer matters. Guarded settings hide 9-13% of the cruise slow events, against 19-53%
  unguarded, for a smaller gain: off-norm time falls to 0.067-0.071 instead of 0.026-0.067.
- **At the budget point nothing improves beyond noise.** Detection at each setting's own budget
  point stays within 0.035 of the unadapted monitor in every phase. False alerts move by up
  to 0.9 per flight-hour, in both directions. In cruise, unguarded 300 and 600 s land within the
  2 per flight-hour budget (1.99 and 1.88), at detection 0.212 and 0.208 against 0.233. They do
  so by hiding 39-41% of the slow events, and they put climb at 3.1-3.2 per flight-hour.
- **At an equal alert rate nothing improves either.** The best test-day detection among
  thresholds within 2 per flight-hour on the test sweep, read off the test day and so not an
  operating point (`sweep_detection_within_budget_on_test_day`), climb / cruise / descent:

  | setting | detection |
  |---|---|
  | no adaptation | 0.066 / 0.215 / 0.073 |
  | guarded, any tau | 0.064 / 0.210-0.213 / 0.067-0.069 |
  | unguarded, any tau | 0.044-0.069 / 0.170-0.212 / 0.046-0.092 |

  Adaptation removes off-norm time from normal flights. It also removes part of the injected
  deviations, so the two do not separate better.
- **The counts are small.** 13-64 slow events are seen unadapted per phase, so one event in
  descent moves the absorbed share by 0.08.
- **What is not studied.** Adaptation across flights or days, which the remaining tune-to-test
  overshoot calls for, and guard parameters other than the configured 2.0 and 2 sd.

## 3D replay (sim/)

A three.js replay of the test day, for looking at what the numbers above are made of.
`make sim-data` writes `sim/assets/basemap.json` (Natural Earth 1:10m v5.1.2, public domain,
clipped to the region +- 1 deg) and `data/sim/scene.json` + `scene.bin` (58 MB, about 70 s on
6 workers); `make sim` serves the repository on 127.0.0.1:8765, open `/sim/`. three.js 0.170.0
loads from cdn.jsdelivr.net.

- **Same code as the evaluation.** `scripts/sim.py` runs the IMM conformance monitor and the
  fixed distance baseline through `alerts/manage.py` at their phase 4 budget thresholds, and
  draws the injected events with `evaluate/inject.py` and the suite's seed. Detection is
  credited with the suite's `_attributable`. Checked: the export reproduces the test-day
  detection rates of both detectors in every phase exactly (IMM 0.122 / 0.233 / 0.103, fixed
  distance 0.184 / 0.172 / 0.147, 656 / 1,650 / 565 events).
- **What is shown.** All 3,844 test-day tracks; the 1,096 with a reference are coloured by score
  over their phase's threshold, with the rarest component, both detectors' open and presented
  alerts, and the corridor (median path at median altitude, +-2 robust sd cross-track). Each of
  the 2,871 injected events can be replayed against the uninjected flight, with onset,
  violation and first-alert times.
- **Causal display.** Positions are interpolated between fixes; score, phase and alert state are
  those of the last fix, as the monitor had them. In a coverage gap the panel says how old that
  fix is.
- **Approximate.** The rolling "alerts per flight-hour" counts every minute a reference flight is
  in the air, not the suite's monitored seconds per phase; it is a guide, not a reported rate.
- **Not published.** The export holds OpenSky data, so it stays in `data/` and the replay is
  served on localhost only (DATA.md, sec. 3.iii). Flights are numbered T0001... in start order;
  no icao24 or callsign is exported.

## Phase 6 — what limits detection

`python -m scripts.evaluate breakdown` (`evaluate/breakdown.py`, `results/phase6_detectability.json`,
`figures/phase6_detectability.png`). Each of the 2,871 test-day events of phase 4 is put in
one class for the IMM monitor and for the fixed distance, at their budget points. The
breakdown's detected share equals the suite's detection rate in every phase for both detectors
(IMM 0.122 / 0.233 / 0.103, fixed distance 0.184 / 0.172 / 0.147), so it is the same
computation, not a re-implementation.

| class | meaning |
|---|---|
| detected | the suite's rule |
| onset in alert | the uninjected flight is in alert for the whole 600 s window; nothing can be attributed |
| below threshold | no row where the injected flight exceeds the operating threshold and the uninjected one does not |
| not persistent / masked | attributable exceedances exist, but never two in a row, or every alert start is inside the flight's own alert |

Shares over all events (IMM / fixed distance):

| kind | detected | onset in alert | below threshold | not persistent + masked |
|---|---|---|---|---|
| lateral | 0.20 / 0.46 | 0.11 / 0.06 | 0.63 / 0.45 | 0.06 / 0.02 |
| vertical | 0.26 / 0.02 | 0.10 / 0.07 | 0.61 / 0.90 | 0.03 / 0.01 |
| longitudinal | 0.09 / 0.03 | 0.11 / 0.07 | 0.73 / 0.88 | 0.08 / 0.02 |

- **The update rate is not the limit.** 2 of 2,871 events (0.07%) reach 1 NM or 300 ft within
  two fixes of onset, the only ones no alert with 2-update persistence could precede. Events
  that never violate within the horizon are 15% of 10 kt longitudinal and up to 43% of 2 deg
  lateral in descent (the flight lands first).
- **The corridor is.** For both detectors most missed events are below threshold: the
  deviation never becomes rarer, against the reference norms, than the 2 per flight-hour budget
  allows. The largest change an injection makes in its own channels, over the residual at the
  IMM threshold (both in station sd), has a median below 1 for every kind, magnitude and phase
  except lateral 30 deg in cruise (1.45) and vertical 1,000 and 2,000 ft/min in cruise (1.01,
  1.76). The norms are wide because a 50 km gate pair mixes flows (phase 4: median station sd
  4.2 km cross-track), and the tails are heavy: at the thresholds (scores 1.85-2.0) a residual must reach
  17.6-22.8 sd cross-track, 13.2-16.6 sd track error or 33-45 sd vertical rate.
- **Alert logic costs little.** Persistence and hold-off lose 3-8% of events; 10-11% start
  while the uninjected flight is in alert for the whole window (IMM).
- **Where the IMM monitor is weak, and where it is not.** On lateral offsets of 10 deg or more
  the fixed distance detects 0.54-0.78 against the monitor's 0.12-0.43; on vertical rate offsets
  the monitor detects 0.22-0.33 and the fixed distance 0.00-0.03. The monitor flags lateral
  events first through track error (184 events) more than cross-track (47), and vertical ones
  through vertical rate (297 of 318). The two detectors fail on different events, which is what
  phase 6's fused detector tests.
