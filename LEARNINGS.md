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
