# Data

## Source

OpenSky Network state-vector samples, `https://s3.opensky-network.org/data-samples/states/`
(landing page `https://opensky-network.org/datasets/states/`). One file per hour, one day per
week; the most recent day published is 2022-06-27 (bucket listing, 2026-10-11). The bucket's
`README.txt` and `LICENSE.txt` are copied verbatim to `refs/opensky_states_README.txt` and
`refs/opensky_states_LICENSE.txt` (sha256 identical to the bucket copies, checked 2026-10-11).

Format used: Avro inside an uncompressed tar, deflate codec inside the Avro. The README notes
CSV and JSON are much larger once decompressed; Avro is streamed straight out of the tar
without extraction (`ingest/avro_stream.py`).

### Routes not used

- **REST API.** Not used as the data source. The API documentation
  (`https://openskynetwork.github.io/opensky-api/rest.html`, read 2026-10-11) states that for
  anonymous users "Only the most recent state vectors are available - the `time` parameter is
  ignored", authenticated users get "State vectors up to 1 hour in the past", and "The tracks
  endpoint is purely experimental". A multi-day historical study cannot be built on that.
- **Trino historical database.** Requires an approved OpenSky account; not assumed. Optional
  upgrade later.

## Terms of use (refs/opensky_states_LICENSE.txt)

- **Permitted use (sec. 1):** a limited, non-exclusive, non-transferable licence to copy, modify
  and use the data *solely* for non-profit research, non-profit education, commercial internal
  testing and evaluation, or government purposes. This project is non-profit education.
- **No redistribution (sec. 3.iii):** the data may not be distributed or made available to anyone
  outside one's institute; collaborators must request access themselves. Hence raw and derived
  data stay out of git (`data/` is ignored) and the project stays local.
- **Privacy (sec. 3.i):** in any publication or disclosure, personally identifiable information
  and aircraft identifiers must be anonymised. REPORT.md therefore must not print icao24 or
  callsigns of individual flights.
- **Citation (sec. 4.i):** any publication must cite "The OpenSky Network,
  http://www.opensky-network.org" or Schafer et al., "Bringing up OpenSky: A large-scale ADS-B
  sensor network for research", IPSN 2014 (`refs/05_schafer_2014_opensky.pdf`), and a copy or
  link must be sent to OpenSky.
- **Retention (sec. 4.ii):** copies are to be deleted on completion of the research, allowing a
  reasonable period for reproducibility.
- **No warranty (sec. 5.iv):** data is provided as is.
- The README states that the user must agree to these terms; downloading the samples does not
  require an account.

## What the files actually contain (measured)

- Snapshots every **10 s**, not the one vector per second the README's column description gives:
  in `states_2022-06-20-14`, all 360 distinct `time` values are multiples of 10 and consecutive
  values differ by exactly 10 s; about 9,900 aircraft per snapshot worldwide
  (`results/phase1_region_survey.json`).
- Rows are in non-decreasing `time` order within a file (0 decreases in 3,564,526 rows).
- Altitudes are in metres; velocity in m/s; vertical rate in m/s; heading in degrees true.

## Sample

Two Mondays, the same six UTC hours on each (12:00-17:59), so scheduled flights repeat one week
apart: 2022-06-20 and 2022-06-27. Files, sizes and sha256 are listed below and pinned in
`config.yaml` (`data.files`); `scripts/fetch.py` refuses a download whose hash differs.

| file | bytes | rows (global) | rows in region | sha256 | downloaded |
|---|---:|---:|---:|---|---|
| `states_2022-06-20-12.avro.tar` | 184,207,360 | 2,991,818 | 131,543 | `1484dad89700f04ed7e8bec14e845d724d5cb2ed9c6aaba4f29e601ca6d8a322` | 2026-10-11 |
| `states_2022-06-20-13.avro.tar` | 208,455,680 | 3,372,576 | 131,186 | `7c6a170a89f2c2a4025dedff42d4c0fd9c36f3d5d9ee7431955500268e703e58` | 2026-10-11 |
| `states_2022-06-20-14.avro.tar` | 221,306,880 | 3,564,526 | 130,216 | `9f39c603f35f53a8d5c937d505d87e14002f5ec6ea00027d859cf9bd1efc57c7` | 2026-10-11 |
| `states_2022-06-20-15.avro.tar` | 226,764,800 | 3,650,760 | 121,802 | `27c0908061e4a2f989942e443c43c56e53cb045de8a643cd77c64e7cdc2a4f98` | 2026-10-11 |
| `states_2022-06-20-16.avro.tar` | 225,761,280 | 3,627,825 | 117,760 | `dda6353655fc347163ddef7666669ed5b462e3338bb39e96764f94ff164abeb4` | 2026-10-11 |
| `states_2022-06-20-17.avro.tar` | 220,897,280 | 3,540,952 | 111,486 | `378029a316289fb2e828c636d6a0f762ac63a3fad85057e4029e145f05acc778` | 2026-10-11 |
| `states_2022-06-27-12.avro.tar` | 177,080,320 | 2,858,582 | 121,861 | `a7313af2d7a6dd75329695417ddfc4d2dd915f9c9771451616d89eaed89133f9` | 2026-10-11 |
| `states_2022-06-27-13.avro.tar` | 200,181,760 | 3,220,440 | 118,142 | `bcd10a07c40d3dccde80f1756148261cdacc76d97d5839bd00cf1ed29960c24b` | 2026-10-11 |
| `states_2022-06-27-14.avro.tar` | 213,975,040 | 3,438,067 | 120,548 | `bdba8ee42bc99d8da8cf05fe946451962841a596b92e9d073b41b7b5fa02e651` | 2026-10-11 |
| `states_2022-06-27-15.avro.tar` | 214,517,760 | 3,445,353 | 118,684 | `65e34cd8c314a2904c2c3d2b70e95e1e5e1ac03d363d728e533e3b042b0443c2` | 2026-10-11 |
| `states_2022-06-27-16.avro.tar` | 213,913,600 | 3,429,896 | 120,717 | `d9b4e3d93fccd240a2842ad212e47f8f0f8dd64cd54a1f5b93f01fe895786b96` | 2026-10-11 |
| `states_2022-06-27-17.avro.tar` | 211,066,880 | 3,377,222 | 110,888 | `40d4c9ab3c80e3e732ddf84cf23c9f4802a3ff6097d1c28d06dfac974ae09244` | 2026-10-11 |

Raw tars are deleted after hashing and filtering (`keep_raw: false`) because the disk had
~4-6 GiB free during setup; re-running `make sample` re-downloads and re-verifies them.

## Region

Bounding box `frankfurt`: lat 47.5-52.5 N, lon 4.8-12.3 E (297,942 km2). Contains EDDF, EDDM,
EDDL, EDDK and EDDS. Chosen from a survey of nine hub-centred boxes of equal angular size over one
global hour, `states_2022-06-20-14` (`results/phase1_region_survey.json`, `make survey`):

| region | mean airborne | per 10k km2 | stale/late rows | fresh contact | time covered | intervals > 15 s | updates < 1500 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| london | 338 | 11.7 | 9.8% | 89.4% | 94.3% | 1.6% | 36,100 |
| amsterdam | 222 | 7.8 | 4.8% | 94.6% | 96.8% | 0.7% | 16,592 |
| frankfurt | 319 | 10.7 | 3.7% | 96.2% | 96.8% | 0.8% | 19,640 |
| paris | 286 | 9.4 | 6.0% | 93.3% | 96.4% | 1.0% | 14,095 |
| zurich | 311 | 9.9 | 4.9% | 94.9% | 96.6% | 0.8% | 13,468 |
| madrid | 110 | 3.1 | 7.5% | 90.0% | 92.7% | 2.2% | 1,614 |
| new_york | 624 | 17.8 | 14.3% | 82.9% | 90.1% | 2.5% | 88,640 |
| chicago | 348 | 10.1 | 13.1% | 83.8% | 91.3% | 2.5% | 30,179 |
| atlanta | 400 | 10.4 | 18.6% | 76.6% | 84.8% | 4.3% | 21,100 |

Among the European boxes with at least 280 aircraft airborne (London, Frankfurt, Paris, Zurich),
Frankfurt is best or tied-best on every coverage column, at within 6% of London's density.
Amsterdam matches Frankfurt's coverage but carries 30% less traffic. New York is twice as dense but loses 10% of flight time to
coverage holes and has 4x Frankfurt's stale-row rate. London has more low-altitude updates
(useful for origin/destination detection) but 2.7x the stale rate. Coverage holes are a
property of the receivers, not the aircraft; the density/coverage trade is the region choice.

Rows with no position at all cannot be attributed to a region and are dropped by the bbox
filter at fetch time; they are not counted in the region's cleaning statistics.
