# Learnings

Gate questions, answers, and corrections. One section per gate.

## Gate 1 — data, stale rows, trivial baselines

GO was given without answers to either question. The questions and the reference answers are
recorded so they can be answered later; nothing below is the user's answer.

**Q1.** The filter treats two rows with the same `lastposupdate` as one measurement and stamps it
with `lastposupdate`, not `time`. Take a row whose position is 200 s old (the 99th percentile of
position age in this sample). What happens to the innovation and the velocity estimate if that
row is stamped with `time`?

*Reference answer.* The filter predicts the aircraft 200 s further along its track (~50 km at
250 m/s) and then receives a position that sits where the aircraft was 200 s earlier. The
innovation is roughly speed x age, tens of kilometres, against an innovation covariance built
for ~10 s of prediction, so the NIS is enormous: a false anomaly that is purely a timestamp
error. If the update is accepted, the gain pulls position back and the velocity estimate
collapses toward zero (the position "did not move" over the stamped interval); the next fresh
position then looks like a jump forward, a second false anomaly. Stamped with `lastposupdate`,
the same row is either an exact repeat (dropped as stale) or a genuine old position placed at
the right time, and the prediction interval is the true elapsed time.

**Q2.** A flight has a 90 s gap between updates. Why is that never an anomaly by itself, and why
are intervals over 60 s left out of the flight-hours denominator of the alert rate? What would
counting them do to a detector's reported false-alert rate?

*Reference answer.* A gap is a property of the receivers (76% of the >60 s gaps in this sample
were silent: no message of any kind reached the network), not of the aircraft. The filter
propagates by the true elapsed time, so the covariance grows over the gap and the first
position after it is judged against that wider uncertainty; whether it conforms is decided by
where the aircraft is, not by the silence. Inside the gap nobody could have alerted, so those
seconds are not monitored time. Counting them would inflate the denominator and lower the
reported false alerts per flight-hour without the detector doing anything better; it would
also make a region with poor coverage look quieter than a well-covered one.

## Gate 2 — IMM

Questions, answers and corrections are recorded here after the gate discussion.

GO was given without answers. Reference answers below; nothing here is the user's answer.

**Q1.** In cruise the median NIS per component is 0.27 of its chi-square expectation, yet 12.6%
of updates exceed the chi-square 95% line. What does that say about the innovation
distribution, and why is "inflate R until 5% exceed" the wrong fix for an anomaly detector?

*Reference answer.* The distribution is leptokurtic: a core much narrower than the Gaussian the
filter assumes, plus a tail far heavier than it allows (unmodelled small manoeuvres, timing
jitter, message errors). No single Gaussian scale fits both: shrinking R fixes the median and
worsens the tail, inflating R until 5% exceed makes the covariance several times too wide for
the bulk of updates. For detection that is the worst of both: the filter becomes sluggish and
the statistic loses sensitivity for every genuine deviation that sits in the core range, while
the tail events it was inflated for are still not anomalies of the kind sought. The tail has to
be handled by thresholds set empirically per phase and by persistence (alert management), not
by mis-stating the measurement noise.

**Q2.** With CV's missing acceleration zero-filled, the turn model missed a quarter of synthetic
turns although it is the right model. Given that Genovese's q_CA enters only the acceleration
term, why can a model whose mixed acceleration variance is ~0 never learn the turn?

*Reference answer.* The measurement sees position and velocity, not acceleration. The update
moves the acceleration estimate only through its covariance with position and velocity,
K_acc = P(acc, pos/vel) S^-1. That cross-covariance is created by propagating the acceleration
variance at the start of the step through Phi (position gains a*dt^2/2, velocity a*dt). With
the mixed acceleration variance ~0 there is nothing to propagate; q_CA's noise is added to the
acceleration at the end of the step, uncorrelated with the predicted position and velocity.
So the gain on acceleration is zero, the CA/TURN prediction equals a noise-free CV prediction
(worse than CV itself), its likelihood loses, and the next mixing step, dominated by CV,
zeroes the acceleration again. It is a lock-out, not a slow start.

## Gate 3 — injection harness and evaluation suite

GO was given without answers. Reference answers below; nothing here is the user's answer.

**Q1.** An injected event counts as detected only if the alert starts at a row where the same
detector on the uninjected flight was not exceeding the same threshold. What would the CV
Kalman's detection rate measure without that rule, at the loosest threshold (17-36 false
alerts per flight-hour)?

*Reference answer.* Mostly its false-alert rate. Every flight raises alerts of its own
(turns, noisy updates), and a 600 s window at 17-36 alerts per flight-hour very likely contains
one of them, injection or not. Without the rule, those alerts would be credited as detections,
and the loosest threshold would look like a near-perfect detector. The rule removes alerts the
flight would have raised anyway, but not all coincidence. At a threshold crossed by a few
percent of updates, the perturbation shifts which rows cross it, so some new crossings appear
that the deviation did not cause. That is why a 2 deg offset (0.42-0.66) is detected almost as
often as a 30 deg one (0.51-0.80) there, and why a detection rate is never read without the
alert rate beside it.

**Q2.** Lead time is measured to the violation (1 NM, 300 ft), not from the onset. At the
budget point the CV Kalman's median lead for gradual events is -61 s in cruise. What does a
negative lead mean operationally, and why would "time from onset" hide it?

*Reference answer.* Negative lead means the alert came after the aircraft had already left
the 1 NM / 300 ft containment. It confirms a violation instead of warning of one, and gives no
time for a controller to intervene. Time from onset is always positive and small for large
deviations, so it would look good. It also depends on when the deviation "started", which is
ill-defined for a slow drift (01 sec. 3.4.2). Measured to the violation, the same alert scores
by whether it was early enough to matter. A detector that is fast to fire but only on large,
already-violating deviations shows up as negative lead.

## Gate 4 — reference paths, conformance scoring, alert management

Answered. The answers below are condensed and translated from Turkish; the corrections follow
each one. The numbers come from the cross-fitted rerun the second answer led to.

**Q1.** Why must an alert that stays open for the whole flight be counted again every 300 s,
and why did counting it once make the budget point pick a detector that is quiet but blind?

*Answer.* An alert costs the controller in proportion to how long it is open, not how often it
starts; a single alert left on all flight occupies more attention than five short ones. Worse,
while it is open the detector cannot report anything new on that flight: a real event that
starts later is lost inside the old alert, so for that flight the detector is blind. The budget
counts alerts per flight-hour; if a held alert counts once, settings whose alerts stay open and
merge look cheap, and the optimisation picks exactly those. Re-presenting every 300 s turns
duration into count, so a stuck setting pays for its time and is pushed out of the budget.
Check it by comparing single counting with 300 s counting at the chosen threshold: total time
in alert, and how many injected events start inside an alert that is already open.

*Correction.* Right in substance. Three refinements:
- Two separate defects were at work, and each needed its own fix. The always-on detectors
  (fixed distance at 100 m, ETA at 1 s, detection 0) were picked by the old rule "lowest
  threshold within budget", not by optimising: that rule is fixed by taking the highest
  detection within budget. Counting a held alert once is what let the IMM monitor meet the
  budget on the tune day while in alert 51-64% of the test day; that is fixed by
  re-presentation. Neither fix covers the other's case.
- A re-presentation is never credited as a detection where the uninjected flight is already in
  alert: attribution requires the clean flight not to be exceeding at that row. So
  re-presentation adds cost without adding lucky detections; it cannot make a stuck detector
  look sharper.
- The check you proposed was run (`results/phase4_imm_conformance_single_count.json`, NOTES.md
  phase 4). Counted once, the budget admits lower thresholds (scores 1.36-1.58 against
  1.85-2.0); flights spend 36-43% of monitored time in alert against 15-21%, and 37-54% of
  injected events start inside an alert already open, against 18-29%. The reported rate falls
  (2.10 per flight-hour over all phases, against 2.77). Detection, however, does not collapse:
  0.114 / 0.264 / 0.170 against 0.122 / 0.233 / 0.103 (climb / cruise / descent), because the
  lower threshold catches more of the events that start outside an alert. So "blind" holds for
  about half the events, not for the detector as a whole; what breaks is the alert rate as a
  measure of the controller's load, which is the reason re-presentation stays.

**Q2.** Each component is mapped to its empirical tail probability on the tune day's reference
flights. What is the risk?

*Answer.* The same flights are used twice: for the tail probabilities and for choosing the
thresholds, so the thresholds fit what happened to be quiet in that data and the false-alert
rate on new flights is higher than reported, like seeing the exam before taking it. Second,
the far tail: a low budget puts the threshold at an extreme quantile, say 1 in 10,000, and
below 1/N an empirical probability is either zero or set by the chance position of a few
samples, so the scale is least reliable exactly where the decision is made. Fix: split the
flights, calibrate on one part, choose thresholds and report false alerts on the other, and
state the resolution limit of the tail.

*Correction.* The mechanism is right; the data place the damage elsewhere.
- The larger reuse was the references, not the tails. Each tune-day flight was one of the 5-7
  members of its own reference: it had pulled the median toward itself and widened the spread
  it was measured against. Against references rebuilt without it, the p99 residuals grow by a
  factor 1.4-7.7: cross-track 14.0 to 22.8 sd, track error 10.6 to 16.7, altitude 87.9 to
  677, vertical rate 32.4 to 44.7, ground speed 6.3 to 9.8, ETA 16.4 to 27.6
  (`results/phase4_references.json`, `residual_quantiles`).
- The reported false-alert rates were never in-sample: they are on the test day, which nothing
  was fitted on. The optimism showed as the tune-to-test overshoot (IMM monitor 3.4-5.0 per
  flight-hour against a budget of 2), not as a misreported rate.
- The thresholds were not in the far tail. The budget point sits at a score of about 2, an
  exceedance of 1 in 100 fixes: persistence, hold-off and the per-hour budget mean a fix need
  not be 1-in-10,000 rare. 241,151 held-out fixes resolve scores up to 5.38. But fixes at 10 s
  are far from independent, and the folds show where the resolution limit bites: at score 2
  the two folds agree within 10% for cross-track, track error, altitude and vertical rate, but
  not for ground speed (7.8 against 14.5 sd) or ETA (14.3 against 67.1 sd), whose tails a few
  flights decide; above score 3 they disagree several-fold (cross-track at score 4: 1,860
  against 412 sd). That is now in `results/phase4_references.json`, `calibration`.
- A half split of the flights would leave 41 of the 152 pairs, those with 5 flights, with 2 or
  3, below the 5 a reference needs. References are therefore left one out (every tune-day
  flight is scored against its pair rebuilt without it). Tails are split as you proposed, in
  two folds: a flight is scored with the tails of the fold it is not in. Thresholds are chosen
  on the tune day; false alerts are reported on the test day. After cross-fitting the IMM
  monitor's test-day rate is 2.15-2.79 per flight-hour (was 3.4-5.0) at unchanged thresholds,
  and its detection fell to 0.103-0.233 (was 0.149-0.262): part of the earlier detection was
  the same optimism. The remaining overshoot is the week between the two days.

## Gate 5 — drift: per-flight adaptation

Answered. Condensed and translated from Turkish; corrections follow each answer.

**Q1.** Adaptation removes a flight's offset and narrows the residuals markedly. Why does it
still not lower the false-alert rate at its own budget point, and what has the rarity score
to do with it?

*Answer.* The score measures rank among normal flights, not size: "what share of normal data
is larger than this". The threshold is chosen so that a set share of normal data exceeds it, so
the false-alert rate at the budget point is the budget by definition; however much adaptation
narrows the residuals, the threshold is re-chosen on the same narrowed distribution. Halve every
grade in a class and the bottom 10% are the same students. The benefit of adaptation, if any,
shows in detection at the same false-alert rate: it helps only if it narrows normal residuals
more than anomalous ones.

*Correction.* The mechanism and the conclusion are right. Three corrections:
- The rate is not the budget by definition. Refitted tails fix the share of *fixes* a score
  threshold flags. Alerts per flight-hour count something else: exceedances after persistence,
  hold-off and re-presentation, so they depend on how exceedances cluster in time. Adaptation
  changes that (long episodes become shorter ones), which is why the settings' rates at their
  budget points range from 1.88 to 3.16 on the test day. And the test day is not where the
  threshold was chosen.
- The grades analogy fits a uniform rescale; adaptation is not one. It subtracts each flight's
  own offset, which reorders fixes across flights: a flight that holds a persistent offset drops
  in rank. Any benefit would have to come from that reordering.
- The comparison you asked for. At the budget points the settings sit at different alert rates,
  so a fair reading is at equal rate: the best test-day detection among thresholds within 2 per
  flight-hour on the test sweep (`results/phase5_drift_tradeoff.json`; read off the test day,
  so not deployable). No adaptation 0.066 / 0.215 / 0.073 (climb / cruise / descent); guarded,
  any tau, 0.064 / 0.210-0.213 / 0.067-0.069; unguarded 0.044-0.069 / 0.170-0.212 /
  0.046-0.092. Adaptation does not separate better. Unguarded, it shrinks the anomalies too: it
  hides 6-14% of the abrupt cruise events and 19-53% of the slow ones that the unadapted monitor
  shows.

**Q2.** How far behind a ramp of constant slope does an unguarded adaptation with tau = 300 s
run? Below what slope does a deviation never cross the threshold? Why does the cap, not tau,
decide how much a guarded setting absorbs?

*Answer.* The offset is a first-order (exponentially weighted) filter. Behind a ramp of slope s
it settles at a lag of s x tau, so the residual never exceeds s x tau however long the ramp runs.
With the threshold at residual level theta, an alert needs s x tau > theta, so every ramp
slower than s* = theta / tau is absorbed forever; with theta = 300 m and tau = 300 s, 1 m/s,
which is 3.6 km an hour unalerted. The guard caps the offset at C: the filter follows with lag
tau until the offset reaches C, then every further metre goes into the residual, which grows
without bound and crosses the threshold. The total absorbed is at most C; tau sets only how
fast C is reached, and a slow ramp is detected after about (C + theta) / s, in which tau does
not appear.

*Correction.* The derivation is right. Four refinements:
- Discrete form. The offset updates once per 10 s fix and the residual of a fix uses the offset
  from before it, so the steady residual is s x dt / (1 - exp(-dt / tau)), about
  s x (tau + dt / 2): 305 s x s at tau = 300 s, not 300.
- theta is not in the config. The threshold is a score: 2.0 for the unguarded 300 s setting in
  cruise. In residual units it is a different multiple of the station spread for each
  component: 14.6 sd cross-track, 13.2 sd track error, 309 sd altitude, 51 sd vertical rate,
  4.6 sd ground speed, 11.8 sd ETA. In physical units it then varies with the station. Over
  cruise-like stations (p10-p90), the cross-track threshold is 26-219 km, so s* at tau = 300 s
  is 88-729 m/s. That is faster than any aircraft drifts, so every lateral ramp is absorbed in
  that channel. The unadapted monitor would not see a slow lateral ramp there within the 600 s
  horizon either. The slow deviations it does see, it sees at stations with tight norms and
  through rate components.
- Rate components see a step, not a ramp. A heading offset is a constant track error, and a
  vertical rate offset is a constant vertical rate residual. The adaptation decays a step h as
  exp(-t / tau). The step is visible for about tau x ln(h / theta) if h > theta, and never if
  h < theta. On those channels tau sets a window in time, not a limit on slope.
- The cap argument holds, and the data show how small C is here. C = 2 sd. The guarded 300 s
  setting's cruise thresholds are 6.4 sd ground speed, 15.9 sd cross-track and 326 sd
  altitude, so the cap can absorb 31%, 13% and under 1% of them. That is why every guarded setting sits within half a
  percentage point of off-norm time of every other, and close to no adaptation. The 9-23% of
  slow events it still hides are events just above the threshold, tipped below it by up to
  2 sd. The freeze also acts before the cap: once the adapted residual scores above 2.0 the
  offset stops, so a ramp with s x tau above that level is never learnt at all.
