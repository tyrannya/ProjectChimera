# Derivatives v1 — causal positioning-and-carry information set

Version: `derivatives_v1`
Research checkpoint: **P4** (*does causal `derivatives_v1`, alone or combined
with OHLCV14, add usable information beyond OHLCV14?*) — the fourth information
family, after `smc_v1`'s P2b, `chart_structure_v1`'s P2c and
`microstructure_v1`'s P3, and the first whose *source is not the spot tape at
all*.

**Status: adaptive research evidence. Not a pristine out-of-sample
confirmation.** P4 cannot produce confirmatory evidence under any outcome; the
ceiling is *single-region supported*, and
[`p4_preregistration.md`](p4_preregistration.md) §12 says why.

**Status of the evidence: P4 has not run, and no P4 model has been fitted.**
This document, the engine, the acquisition, the verifier and the runner beside
it were committed *before* the derivatives source could be acquired — outbound
access to `data.binance.vision` and `fapi.binance.com` is denied by the egress
policy in force, exactly as it was when `microstructure_v1` was written — which
is what makes everything below predeclared rather than fitted. There is no P4
cell, no P4 comparison and no P4 result artifact. The interlock at
[`../data/research/p4_stage1_authorisation.json`](../data/research/p4_stage1_authorisation.json)
says `not_authorised`, and the P4-HOLD ledger says `unspent`.

---

## 1. What this document is, and what it is not

**It is not the specification.** Every scientific constant of `derivatives_v1` —
the eight column names, their definitions, their windows, their clips, the
staleness bounds, the warm-up, the venue, the instrument, the sample-universe
rule — is a value in
[`../nn/p4_preregistration.py`](../nn/p4_preregistration.py), inside the payload
that [`p4_preregistration.md`](p4_preregistration.md) hashes. `nn/derivatives.py`
**reads them from there** rather than restating them, so this document cannot
disagree with the preregistration: there is nothing here for it to disagree with.

That is a deliberate difference from `smc_v1.md`, `chart_structure_v1.md` and
`microstructure_v1.md`, each of which is the definition of its own family. P4's
definition was fixed before its data existed and is hashed; a second copy of it
in prose would be a second thing to keep in sync, and the failure that costs is
the one this repository has already had once — a document and a hashed value
drifting apart, with the number deciding and the prose telling a different story.

**It is the implementation record.** What is here is what the preregistration
does *not* fix and an implementation therefore had to choose, plus the operational
detail of turning three public archives into one hourly table.

---

## 2. The columns

Eight, from
[`nn.p4_preregistration.FEATURES`](../nn/p4_preregistration.py), in that order:

| column | family | window | clip |
| --- | --- | --- | --- |
| `drv_funding_last` | funding | — | [-0.01, 0.01] |
| `drv_funding_sum_9` | funding | 9 settlements | [-0.09, 0.09] |
| `drv_funding_z` | funding | 30 settlements | [-5, 5] |
| `drv_oi_log_change_24h` | open interest | 24 hours | [-1, 1] |
| `drv_oi_notional_ratio` | open interest | 168 hours | [0, 10] |
| `drv_oi_price_divergence` | open interest | 24 hours | [-1, 1] |
| `drv_basis` | basis | — | [-0.02, 0.02] |
| `drv_basis_z` | basis | 168 hours | [-5, 5] |

`nn.derivatives.DerivativesSpec.spec_hash` covers the constants *and* the
windows and clips it read from §5, so editing the preregistration moves the
feature-spec hash as well as the preregistration hash. A cell recorded under one
is not comparable with a cell recorded under the other, and
`nn.p2b_compare.check_cells_agree` refuses to join them.

**Funding windows count settlements; open-interest and basis windows count
hours.** That is what §5's own table says — `drv_funding_sum_9` is "the last 9
realised settlements (three days)" and `drv_basis_z` is "the last 168h" — and it
is why the 240-hour warm-up and the 30-settlement window are the same bound.

---

## 3. What the preregistration left open, and what was chosen

Four constructions, recorded as values in `DerivativesSpec` so that each is part
of the spec hash rather than a habit of the code. Each is stated here with the
reading that was taken and with what it costs if the other reading were right.

### 3.1 "The last N" includes the row's own observation

§5 writes "the mean of the last 168h" and "the last 30 visible settlements", and
it names `drv_funding_last` — *the most recent visible settlement* — as the
numerator of `drv_funding_z`. A window that excluded the row's own observation
would not contain the value being standardised, so the inclusive reading is the
one the definition supports.

Recorded because this repository's other families use the opposite convention:
`nn.microstructure._trailing` is strictly prior, by `min_periods` then `shift(1)`.
Both are causal — a row's own observation is available to it — and a silent
disagreement between two families is the kind of thing found in a number rather
than in a diff.

### 3.2 `drv_funding_z` standardises against the raw settlements

§5 standardises `drv_funding_last` against "the last 30 visible **settlements**",
which are the archive's rates rather than this column's clipped copy. So the
numerator is the clipped column and the window is the raw series.

**This cannot bind on the instrument in scope.** Binance caps BTCUSDT's funding
rate at ±0.75%, strictly inside the ±1% clip, so the clipped and unclipped
series are the same numbers, and `tests/test_derivatives.py` asserts it on the
data the engine sees. It is recorded rather than passed over because "it does
not matter here" is a claim, and a claim in a research repository should have a
test under it.

### 3.3 `drv_basis_z` standardises against the clipped basis

§5 names `drv_basis` — the column, clip and all — for both the numerator and the
window, so numerator and window are one series. Unlike §3.2 this one *could*
bind: a 2% clip on the perpetual's premium is inside the range a stressed market
reaches. Taking the literal reading is what keeps the choice out of the
implementation's hands.

### 3.4 A punctured trailing window still produces a value

`min_window_observations = 1`. A trailing mean over the last 168 hours is defined
when the window holds an observation, and §3.0a's own arithmetic — "one missing
day removes every hour of *that UTC day*" — is only true if a punctured window
still produces a value for the hours around it. Requiring a full window would
make one missing archive day remove a whole week.

A row whose **own** observation is missing or stale leaves the universe
regardless, and that is the rule doing the work.

---

## 4. Two consequences that are larger than the preregistration's worked examples

Both follow from the preregistered definitions; neither is a choice. Both are
recorded here because they tighten the availability gate, and a reader deciding
whether to authorise a stage-1 run needs them.

### 4.1 A missing open-interest day costs 48 rows, not 24

`drv_oi_log_change_24h` and `drv_oi_price_divergence` read the snapshot at `t`
**and** at `t − 24h`. §3.4 forbids carrying either forward past an hour. So a day
the metrics archive does not publish removes its own 24 hours *and* the 24 hours
exactly one day later.

§3.0a's worked example — "every hour of that UTC day leaves the sample universe"
— is the staleness rule's own consequence and is correct as far as it goes; the
echo is the feature definitions' consequence on top of it.

Under §8.0's rule (98% of 4,821 rows ≈ 96 hours, no contiguous outage over 48
hours), that means an outer block tolerates **two** missing days, not four, and
two *consecutive* missing days produce a 48-hour outage that is at the limit.

### 4.2 A spine segment boundary inside a block makes that block unavailable

This one is a property of the committed geometry and was true before any archive
was fetched.

The dataset builder drops an indicator warm-up after every market-data gap, so a
two-hour hole in the candle history becomes a three-and-a-half-day hole in the
research spine. §6.2's fourth condition forbids a feature's window bridging a
spine segment boundary, and `derivatives_v1`'s binding window is 30 funding
settlements — 240 hours. Every spine segment therefore begins with 240 rows that
no arm can be scored on, and 240 hours is a contiguous outage where §8.0 permits
at most 48.

The committed spine has fifteen segments. Their starts are rows 0, 854, 1019,
1264, 2419, 3869, 7506, 7933, 9002, 9467, 10462, 10499, 13049, 14094 and
**26998** — and 26998 falls inside the first exploratory outer block
`[26518, 31339)`. So:

| block | rows | segment start inside | verdict on geometry alone |
| --- | --- | --- | --- |
| `[26518, 31339)` | 4,821 | row 26998 | **unavailable** (240h outage) |
| `[31339, 36160)` | 4,821 | none | available |
| `[36160, 40981)` | 4,821 | none | available |
| `[40981, 45802)` | 4,821 | none | available |

Three available blocks clears §3.6's gate, which requires two. But §8.2 requires
**three valid folds and three improved folds**, so stage 1's screen is a 3-of-3
rather than a 3-of-4: one invalid fold under the ten-trade rule, or one fold with
a non-positive delta, ends the checkpoint.

`tests/test_p4_universe.py::test_a_spine_segment_boundary_inside_a_block_makes_it_unavailable`
pins this, and the number is reported per block by the availability gate whatever
it decides. **Nothing here is a reason to change a preregistered constant.** It
is a reason for whoever authorises stage 1 to know in advance how little room the
screen has.

---

## 5. The source

Three archives, fixed in §3 and read from `DATA_SOURCES`:

| field | archive | granularity |
| --- | --- | --- |
| funding rate | `.../futures/um/monthly/fundingRate/BTCUSDT/` | monthly |
| open interest | `.../futures/um/daily/metrics/BTCUSDT/` | **daily** |
| perpetual price | `.../futures/um/monthly/klines/BTCUSDT/1h/` | monthly |

The spot denominator of the basis is not fetched: it is the committed candle
history `data/research/btc_usdt_1h_gen1_raw_pre_styx.parquet`.

`tools/export_derivatives_snapshot.py` writes **one hourly table**, not the
features. Storing the features would put §5's windows inside the *source*, where
a re-export could move them without moving the feature-spec hash. What is stored
is the point-in-time observation each feature is a function of, plus how stale it
is — see `nn.derivatives_sources.DERIVATIVES_COLUMN_KINDS` — and
`nn/derivatives.py` turns those into columns.

Availability is a `0/1` flag and staleness an `int64` age rather than a nullable
instant, because a semantic fingerprint has no representation for "missing": a
`t8` column refuses a `NaT` and an `f8` column folds every `NaN` together.

### 5.1 What stops the acquisition, and what does not

| situation | outcome |
| --- | --- |
| a metrics day the archive does not publish (404) | recorded as a **missing day**; its hours leave the universe for every arm |
| a metrics day whose bytes fail the published checksum | recorded as a missing day, with the reason |
| a metrics day whose member is empty | recorded as a missing day, with the reason |
| a funding or kline period that is absent | **stops** — those sources are continuous by design |
| a transport failure that outlived its retries | **stops** — an archive this tool could not fetch is not one the source does not publish |
| an unreadable ZIP, an ambiguous member, a nested member | **stops** |
| a funding header matching no §3.0b layout | **stops** |
| a timestamp that will not parse inside its archive's own period | **stops** |
| anything at all from `fapi.binance.com` | there is no code path; §3.0a forbids it |

The distinction between the first three rows and the rest is the one §3.0a
draws: a missing day is a *measurement* of what the archive contains, and
everything else is a failure of this reader. Only a measurement may enter the
availability gate.

---

## 6. Causality

Every column is a function of observations published at or before the row's own
instant, on the conservative rule of §4: a funding settlement at `T` is visible
to row `t` only when `T <= t` — the candle's *open*, a full hour before the row
decides — and an open-interest snapshot only when `create_time <= t` and it is at
most an hour old.

The battery §4.4 requires is `tests/test_derivatives.py`, and every item in it
carries a **positive control**: the funding visibility boundary is exercised at
`T = t` and at `T = t + 1h` with the value moving; the open-interest staleness
bound at exactly one hour and one nanosecond past it; the perpetual series and
the spot denominator each shifted by one row; a future observation mutated and
the earlier rows asserted unchanged *and* the later rows asserted changed. A
leakage test that only asserts the right answer proves nothing about whether it
could have found the wrong one.

---

## 7. The sample universe, and why this family has no declared defaults

`smc_v1`, `chart_structure_v1` and `microstructure_v1` all guarantee a finite
value on every row and fill from a declared default. `derivatives_v1` does not,
and the difference is structural rather than stylistic.

P4's control is **re-run on the intersection** (§6.2). An undefined derivatives
row therefore has to leave the universe for *every* arm — including `ohlcv14` —
rather than be imputed for one, because the alternative is a comparison in which
the derivatives arm is scored where its data exists and the control is scored
everywhere. That comparison measures the difference between two market periods
and reports it as a difference between two information sets, and it is the
single easiest way to get P4 wrong.

So `compute_derivatives_features` returns a `defined` mask beside the columns,
`nn/p4_universe.py` intersects it with §6.2's other conditions, and the mask
reaches every arm's scaler, windowing and scoring **from the same array object**.
"The three arms were scored on the same rows" is then a property of how the run
was constructed, not a claim about it — and `nn.p2b_compare` refuses to join two
cells whose universe or per-fold sample-index hashes disagree.

---

## 8. Why this family is not in `nn/causal_families.py`

The registry in `nn/causal_families.py` iterates engines with the signature
`compute(candles)` / `compute_segmented(candles, timeframe)`, and
`tests/test_causal_families.py` checks the properties every such engine shares.
`derivatives_v1` has neither signature: it is a function of an hourly derivatives
source and a spot close series, not of candles, and it returns a definedness mask
rather than a value on every row.

`microstructure_v1` is absent for the same reason. Registering either would mean
widening the registry's contract until it no longer says anything, so both get
their properties asserted in their own suites instead —
`tests/test_derivatives.py` here, `tests/test_microstructure.py` there.

---

## 9. What must still happen before P4 runs

- [x] acquisition tool with `--plan` and `--probe`, no network in `--plan`
- [x] the eight columns implemented against §5, with a spec hash
- [x] the §4.4 leakage battery, each item with a positive control, **passing**
- [x] a snapshot exporter and a fail-closed verifier that recomputes every claim
- [x] the verifier called from inside the only function that loads the source
- [x] a value-level cross-source check with justified explicit tolerances
- [x] `P4` registered in `nn.information_sets.CHECKPOINTS` with its three arms
- [x] every stage-1 row set, snapshot and fold plan passed through `nn.p4_holdout`
- [ ] **the source acquired** — blocked: the egress policy denies both
      `data.binance.vision` and `fapi.binance.com`
- [ ] the availability window established from metadata, and the §3.6/§8.0 gate applied
- [ ] the funding CSV layout matched against §3.0b's allow-list, or refused
- [ ] P4-HOLD's own archive-day coverage established
- [ ] the P4-HOLD snapshot exported, verified, and left uninspected

The four unchecked items above the last are all downstream of one thing: a
network path to the public archive. Nothing about them is a research decision,
and none of them may be worked around by reading a different source.

**Two housekeeping steps belong to the session that acquires the source**, and
they are recorded here so that they are not discovered as failures. `data/` is
ignored except by explicit allow-list, so
`btc_usdt_1h_gen1_derivatives_hourly_pre_p4hold.parquet` and
`btc_usdt_1h_gen1_derivatives_snapshot_manifest.json` each need a `!` line in
`.gitignore`; and the hourly table is larger than the 512 KiB
`check-added-large-files` limit, so it needs the same pre-commit exclusion the
P3 trade aggregate has. Both are deliberate decisions about how this repository
stores a research source, which is why neither is made in advance.

`tests/test_p4_preregistration.py::test_no_p4_market_data_has_been_acquired`
asserts that `data/research/` holds exactly two P4 files — the ledger and the
interlock — and is a tripwire on purpose: "P4 has data now" should be a diff
somebody wrote, not a directory listing that quietly changed.
