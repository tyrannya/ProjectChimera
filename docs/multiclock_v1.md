# multiclock_v1 — the causal source every Chimera clock is cut from

Engineering, not a research checkpoint. Nothing here selects a model, a feature,
a threshold, a horizon or a target, and nothing here is evidence about alpha.

## 1. Why this exists

`v4` through `P5` all observed one clock. Changing that is the next research
question, and a question about clocks cannot be asked from an hourly file: 5m
bars cannot be recovered from 1h bars, and a 1m specialist has nothing to read.
So the programme needs a minute-resolution source — and it needs one whose
construction cannot quietly leak the future, because at a 1m decision cadence
the ways to do that multiply.

## 2. What was acquired, and from where

**Binance's own published spot archive**, the same exchange, asset and
instrument the 1h programme ran on:

```
data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip
```

65 monthly objects, `2020-01` through `2025-05`. The canonical hostname for the
archive is `https://data.binance.vision`; in an environment where that name is
not resolvable the same bucket is addressed through its S3 origin
(`https://s3-ap-northeast-1.amazonaws.com/data.binance.vision`), which is the
same publisher serving the same objects. The manifest records both, so a reader
never has to infer which endpoint served the bytes.

**Identity is established by Binance's own digest, not by the hostname.** Every
object has a `.zip.CHECKSUM` companion holding the SHA-256 the exchange
computed. `tools.acquire_multiclock_source` refuses any object whose bytes do
not hash to that value, and the manifest records the digest of every archive and
of the CSV member inside it. That is what makes an endpoint substitution a
transport detail rather than a provenance change.

Deliberately **not** done: no other exchange, no other pair, no perpetual-futures
klines, no REST backfill, no third-party mirror. A different venue would have
changed two things at once and made every P6 number uninterpretable.

### 2.1 The unit change that would have destroyed the history silently

Binance switched the `open_time` column of these archives from **milliseconds**
to **microseconds** with the `2025-01` file. Reading the whole history as
milliseconds places every 2025 candle in the year 51726; reading it as
microseconds places every 2020 candle in 1970. Either mistake yields a frame
that sorts, de-duplicates and resamples without complaint.

The unit is therefore derived per file from the magnitude of the values — the
two eras are three orders of magnitude apart, so the threshold sits in an empty
band — and recorded per file in the manifest. 60 of the 65 months are `ms` and 5
are `us`.

## 3. The research-visible boundary

```
2025-05-19T08:00:00+00:00
```

This is the first instant of `P4-HOLD`, the region `[45802, 48211)` that
`data/research/p4_holdout_ledger.json` retired unread with `checkpoint: null`.
It is available to nobody, so acquisition stops before it: the committed source
ends at `2025-05-19T07:59:00+00:00` and holds 2,827,755 minutes.

Styx (`2025-08-27T23:00:00+00:00`) is three months further on and is never
approached. **A change of clock is not a reason to manufacture a new pristine
holdout, and this generation manufactures none.**

The boundary is enforced on *constituent minutes*, not on bar-open timestamps.
A 1h bar opening at `07:00` needs the minute `07:59`, which is legal; a 1h bar
opening at `08:00` would need minutes the boundary forbids and therefore does
not exist rather than existing in a shortened form.

## 4. How a derived bar is defined

For a timeframe of `N` minutes, the bar opening at `t` is:

| field | value |
| --- | --- |
| `open` | the `open` of the 1m candle at `t` |
| `high` | the maximum `high` over the `N` constituents |
| `low` | the minimum `low` over the `N` constituents |
| `close` | the `close` of the 1m candle at `t + (N-1)min` |
| `volume` | the sum of the `N` constituent volumes |

subject to four rules, each of which is checked rather than assumed:

1. **Strict UTC boundaries.** `t` is a multiple of `N` minutes measured from the
   epoch. `nn.multiclock.resample_from_minutes` takes no origin argument,
   because an origin is exactly the knob that would let two runs disagree about
   which minutes belong to which bar.
2. **Full constituent counts.** The bucket must hold exactly `N` distinct
   minutes. Four minutes is not a 5m bar with a caveat; it is an exchange outage.
   Incomplete bars are **dropped** — never forward-filled, forward-completed, or
   padded from the period after them.
3. **No membership from the future.** The last constituent must open strictly
   inside the period. A resample that folded in the first minute of the next
   period would still produce full-looking buckets.
4. **No row at or after the boundary.**

Uniqueness and ordering are asserted before any of this, because rule 2 is only
meaningful once they hold: sixty rows in an hour is evidence of a complete hour
only if the sixty are sixty distinct minutes.

## 5. What the exchange did to the data

The 1m archive has **15 discontinuities**, all of them Binance outages, the
longest 5h54m on 2020-02-19 and the last on 2023-03-24. They are enumerated in
the manifest because every clock's completeness is a function of them:

| clock | bars | incomplete bars dropped |
| --- | --- | --- |
| 1m | 2,827,755 | 0 |
| 5m | 565,550 | 4 |
| 15m | 188,514 | 6 |
| 30m | 94,256 | 7 |
| 1h | 47,123 | 14 |
| 4h | 11,767 | 24 |
| 1d | 1,950 | 16 |

A slower clock loses more bars to the same outage, which is the expected shape:
one missing minute destroys one 5m bar and one 1d bar alike.

## 6. The 1h parity check

The 1h clock is **re-derived from the 1m source** rather than read from
`btc_usdt_1h_gen1_raw_pre_styx.parquet`, so that the two can be compared value
by value over the research-visible region. That comparison is what licenses
fitting on the new source at all.

Tolerance: **relative `1e-9`**. Binance publishes BTCUSDT prices as decimal
strings with two decimal places and volumes with eight; both round-trip through
float64 exactly. Agreement between two correct readings of the same bar is
therefore *identity*, and the tolerance is set at the width of a float64
summation-order difference rather than at anything economic. A discrepancy
larger than this is a real disagreement that has to be explained, never absorbed
by loosening the number.

**Result: 47,094 of 47,123 overlapping hours agree — 99.9385%.**

| column | maximum relative difference | hours within `1e-9` | bit-identical hours |
| --- | --- | --- | --- |
| `open` | `0` | 47,123 of 47,123 | 47,123 |
| `high` | `4.77e-3` | 47,121 | 47,121 |
| `low` | `4.34e-3` | 47,121 | 47,121 |
| `close` | `1.53e-3` | 47,119 | 47,119 |
| `volume` | `2.39e-1` | 47,094 | 44,460 |

`volume` is the only column where the two counts differ, and the 2,634 hours
between them are float64 summation-order differences of relative size at most
`2.2e-16` — the archive sums 60 minute volumes and the hourly series does not.
That is the noise floor the tolerance was chosen to sit above, not a
disagreement.

Two kinds of difference, and both are named rather than absorbed:

**(a) 13 hours present only in the committed history.** These are hours in which
the 1m archive holds fewer than 60 minutes. The strict full-constituent rule
makes them *unavailable* rather than partial, which is the conservative
treatment: the committed 1h series reports a bar aggregated over whatever
traded, and this generation declines to trade a bar the minute data cannot
reconstruct.

**(b) 29 hours whose values disagree.** These are an **upstream inconsistency
inside Binance's own archive**, established by a three-way comparison rather than
assumed:

| comparison | disagreeing hours |
| --- | --- |
| committed 1h vs Binance's published **1h** archive | 8 of 47,136 |
| Binance's published **1h** archive vs this 1m-derived 1h | 22 of 47,123 |
| committed 1h vs this 1m-derived 1h | 29 of 47,123 |

Binance's own 1h and 1m series disagree with each other on 22 hours. The
derivation is not what introduces the difference: against Binance's own 1h
archive, **every `open` and every `close` matches exactly** on all 47,123
overlapping hours, and only `volume` (22 hours) and the extremes (2 hours and 1
hour) differ — the shape of a 1m archive that is missing individual trades the
hourly aggregation captured, never of a misaligned or shifted grid.

**Where they are.** All 29 lie between `2020-04-09` and `2022-05-01`. The
earliest reported outer block begins `2023-03-04T07:00Z`, so **none of them
falls in any block whose numbers a checkpoint reports**; they lie in the
training region alone. The 29 timestamps are enumerated in the manifest so that
the claim can be re-checked rather than believed, and it is asserted
mechanically against the fold periods in the P6 preregistration's own tests once
those periods are frozen.

This is a resolution, not a tolerance widened until the problem disappeared: the
cause is located in a specific upstream series, the effect is bounded by
enumeration, and the bound is shown to miss every reported period.

## 7. What is committed

| path | what it is |
| --- | --- |
| `data/research/btc_usdt_multiclock_gen2_1m_pre_boundary.parquet` | the canonical 1m source, 2,827,755 rows |
| `data/research/btc_usdt_multiclock_gen2_manifest.json` | per-month provenance, per-clock digests, gap list, parity record |

The derived clocks are **not** committed. They are cut from the 1m file
deterministically by `nn.multiclock.resample_from_minutes`, and the manifest
records a digest of each, so `tools.verify_multiclock_snapshot` re-cuts all seven
and holds them to it. One source of truth, and no derived file that can drift
away from it.

Reproduce with:

```
python -m tools.acquire_multiclock_source --archive-dir DIR   # network
python -m tools.verify_multiclock_snapshot                    # offline
```

## 8. Defects and open questions

Written before any P6 number existed, so that none of them can later be
mistaken for a response to a result.

1. **The 29 upstream hours are not repaired, only bounded.** Binance publishes
   two series that disagree and neither is documented as authoritative. This
   generation prefers the 1m archive because it is the source every clock is cut
   from, and consistency across clocks matters more than agreeing with a series
   no clock uses.
2. **13 hours are unavailable that the 1h programme had.** The 1h specialist in
   P6 therefore sees marginally fewer bars than P2a's control did. This is a
   consequence of the completeness rule, not a defect to be tuned away, and it
   means P6's 1h cell is *not* a reproduction of P2a's and is not offered as one.
3. **Volume is summed, and quote volume is discarded.** The archive carries
   `quote_volume`, `trades`, and taker-side splits. None enters this generation,
   because P6 changes the clock and adding columns would change two things.
4. **The gap structure is inherited, not modelled.** A bar adjacent to an outage
   is treated as an ordinary bar. `nn.data_pipeline` segments on discontinuities
   so no feature or label crosses one, but nothing marks the neighbourhood of an
   outage as unusual.
5. **1d bars are UTC days.** No exchange-session or funding-window alignment is
   attempted; the epoch grid is the only grid.

None of these is a threshold to search against the outer blocks.
