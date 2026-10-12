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
