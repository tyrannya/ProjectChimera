# P13-A2R2 — SOURCE ACQUISITION AND SOURCE CLOSURE

**SOURCE ACQUISITION ONLY. NO HISTORICAL ECONOMIC RUN. NO G1–G6. NO GOVERNED
RESULT. NO LIVE MONEY.**

| | |
| --- | --- |
| Active design | **P13-A2R2** |
| Active preregistration hash | `sha256:cac2f318e525fb1f0e5892fdd16fcd5febb72853d1a1cfa9fd6c5d3868b7a092` |
| Result state | `P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT YET RUN` |
| Archive host | `data.binance.vision` (only) |
| Acquisition manifest digest | `sha256:ca196161d6f4384a7c88d97acb5a29bbe1fdf31d427f882f5e3f5ef646114823` |
| Source closure digest | `sha256:16710f0432e869d6100e74ed4d2a0b25777068c1234a4f2e49b94d24c3c7ec6e` |
| Generated at | `2500ba9620df6c28c09ab5ed162faaaeea864d09`, clean tree |

This directory is **new**. The environment-only NOT EVALUABLE evidence in
[`../btc_p13_carry/`](../btc_p13_carry/) was generated at `2b1b400e` under the
ORIGINAL design and keeps that design's hash; it is not regenerated, not
rewritten and not superseded by anything here.

## 1. What was acquired

All **260** objects the frozen plan requires — the same plan digest
`sha256:e0caba374d12276e4bf5ebf6b155d0239c9ef7fbb71deacf375c292ee887a405` the
`2b1b400e` evidence records — across 65 months, 2020-01 through 2025-05.

| family | objects | rows read | archive bytes | resolved unit |
| --- | --- | --- | --- | --- |
| `spot_price` (spot klines 1h) | 65 | 47,137 | 2,810,253 | ms ×60, **µs ×5** |
| `perpetual_price` (USD-M klines 1h) | 65 | 47,168 | 2,471,109 | ms ×65 |
| `mark_price` (USD-M markPriceKlines 1h) | 65 | 46,976 | 1,459,417 | ms ×65 |
| `funding_settlement` (USD-M fundingRate) | 65 | 5,896 | 56,987 | ms ×65 |
| **total** | **260** | | **6,797,766** | |

Extracted members total 18,317,947 bytes.

**Checksum discipline.** Every object was fetched with its published `.CHECKSUM`
companion; the companion was parsed, **both** halves checked (a companion naming
a different object is refused), and the digest **compared** against an
independently recomputed sha256 of the received bytes. **260 of 260 verified; 0
unverified; 0 mismatches.** Both digests are recorded per object — the whole
archive object's, which the publisher vouches for, and the extracted CSV
member's, which the rows were parsed from.

**Object identity is the publisher's PATH, not the filename.** Binance names the
spot klines, the USD-M perpetual klines and the USD-M markPriceKlines identically
— `BTCUSDT-1h-{period}.zip` for all three — and distinguishes them only by
directory. Each record carries `archive_relative_path`, and all 260 are distinct.

**No machine-local path appears in this evidence.** Where the bytes sit is local
state, so the manifest records no cache directory and no per-object cache path:
it records the publisher's path plus the digests, which are the same on every
machine. The raw archives are not committed.

## 2. What the sources are

**Timestamp units, resolved per object and never assumed.** Only the SPOT family
changes: 60 months in milliseconds and **2025-01 through 2025-05 in
microseconds**, exactly as Binance's README documents for spot data from
2025-01-01. The perpetual, markPriceKlines and fundingRate families are
milliseconds throughout, matching the preregistration's expectation that "the
futures archives show no equivalent note". `TIMESTAMP_UNIT_POLICY`'s per-object
resolution handled the straddle without a special case.

**Row quality.** Zero contradictory duplicate instants, zero non-positive prices
and zero repeated funding instants, across all 260 objects.

**Funding.** 5,896 settlements, first `2020-01-01T00:00Z`, last
`2025-05-19T00:00Z`, every one published with `funding_interval_hours = 8`; by
calendar year 1,098 / 1,095 / 1,095 / 1,095 / 1,098 / 415. This is a **source
count** — rows the archive publishes — not the number any block applies, which
depends on a holding window and is an economic quantity this closure does not
compute.

**Mark publication.** All **65 of 65** months publish a `markPriceKlines` object.
`MARK_PRICE_FALLBACK`'s per-object substitution is therefore never triggered over
this span. Note this is about the OBJECT: the objects exist, and some of them are
internally incomplete — see §3.

**Research boundary.** 950 rows fall at or after `2025-05-19T08:00:00+00:00` and
were truncated at load in the single boundary-straddling month (304 per kline
family, 38 funding). The maximum surviving instant is `2025-05-19T07:00:00+00:00`,
strictly before the boundary. No other month carried a boundary-crossing row.

## 3. Grid coverage, and the central finding

Against the calendar-generated reference grid of **47,168** hours:

| family | present | missing | gap runs | longest gap |
| --- | --- | --- | --- | --- |
| `spot_price` | 47,137 | **31** | 15 | 5 h |
| `perpetual_price` | 47,168 | **0** | 0 | — |
| `mark_price` | 46,976 | **192** | 5 | **96 h** |

The two families that are short are short in completely different ways. Spot's 31
missing hours are short maintenance windows, most beginning at 02:00 UTC. Mark's
192 missing hours are whole **days** absent from months whose object was
published: `2021-07-01`, `2021-07-24`→`2021-07-27`, `2022-07-31`, `2022-10-02`
and `2023-02-24`.

Both were confirmed independently of the loader by reading raw published CSVs
directly: `BTCUSDT-1h-2021-07.zip` carries 744 of 744 hours under the spot and
perpetual paths and **624 of 744** under the markPriceKlines path.

Under A2R2 every block opens at its calendar boundary, with **no delay** and with
the opening consulting **no mark row** — which is the amendment working as
specified. Held-window coverage:

| block | opens | held hours | held hours missing an execution row | held hours missing a mark row |
| --- | --- | --- | --- | --- |
| 2020 | 2020-01-01T00:00Z | 8,783 | **17** (first `2020-02-09T02:00Z`) | 0 |
| 2021 | 2021-01-01T00:00Z | 8,759 | **13** (first `2021-02-11T04:00Z`) | **120** (first `2021-07-01T00:00Z`) |
| 2022 | 2022-01-01T00:00Z | 8,759 | 0 | **48** (first `2022-07-31T00:00Z`) |
| 2023 | 2023-01-01T00:00Z | 8,759 | **1** (first `2023-03-24T13:00Z`) | **24** (first `2023-02-24T00:00Z`) |
| 2024 | 2024-01-01T00:00Z | 8,783 | 0 | 0 |
| 2025-partial | 2025-01-01T00:00Z | 3,319 | 0 | 0 |

Blocks 2024 and 2025-partial are fully covered. Blocks 2020–2023 are not.

### P13 SOURCE CLOSURE: FUTURE GOVERNED SCREEN NOT EVALUABLE

`MARGIN_AND_LIQUIDATION.liquidation_check` requires its inequality "evaluated at
every hourly grid instant while the position is open", and A2 makes a held bar
with no authorised liquidation touch **screen-wide terminal**
`P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT EVALUABLE`. Both causes are present:
223 held hours lack a required source row, and the first is
`2020-02-09T02:00:00+00:00`.

**This is a source-validity statement and nothing else.** It says the
preregistered sources are insufficient to audit every hour the design would hold
a position through. It is **not** an economic result, **not** a negative finding
about carry, and must never be cited as one. No funding total, basis figure,
block return, gate condition or decision was computed here, and none may be
inferred from this document.

**It is not rescued and not skipped.** The affected periods are not dropped, the
holes are not jumped, the entry is not delayed past them, and no block is
excluded. A2's `forbidden_treatments` names every one of those and refuses it.

**It was foreseen before acquisition.** The block runner committed at `1d5a31e` —
before a single object was fetched — states the consequence in its own module
docstring: *"under this reading a single unpublished hour inside a holding window
terminates the screen … It is flagged for review rather than buried."* That flag
now has a measurement attached to it.

**Open question for the reviewer, deliberately left open.** A2's
`held_bar_mark_absent` is written about a bar "carrying neither a mark high nor a
mark close", while the runner applies it to a held bar missing *any* required
source. On these sources the distinction is material: the mark-absent hours (192)
are the case A2 describes literally, while the execution-absent hours (31,
including the earliest) are the extension. Whether the uniform reading is right is
a preregistration question and **cannot be settled now** — the coverage is
visible, so re-reading the rule at this point would be choosing a rule against a
known outcome. It is recorded for an independent reviewer to settle explicitly,
before any governed run, and in the knowledge that either answer is now
data-informed.

## 4. A defect found and corrected during this acquisition

The first run of this acquisition reported 260 of 260 objects verified while only
130 files existed. The cache was keyed on the object NAME, which three of the four
families share, so all three mapped onto one file — and because a cached object is
reused, the second and third families were never fetched at all: their bytes were
the first family's, verified against the first family's own companion, and passed.

Every check ran and every check passed; they simply ran three times on one object.
The closure then described spot data three times, which is why an earlier draft of
this document reported identical gaps and an identical unit split across the three
kline families. **That was the artifact, not a finding, and every conclusion drawn
from it is withdrawn** — including the claim that the futures archives had also
moved to microseconds, which the corrected data disproves.

The cache is now keyed by the publisher's path, `acquisition_manifest` refuses a
manifest carrying two records for one published path, and a regression test proves
three distinct published objects yield three distinct digests and six fetches. The
65 spot and 65 fundingRate objects that HAD been genuinely fetched were preserved
and re-verified against freshly downloaded companions rather than re-downloaded;
the 130 that had never been fetched were fetched.

## 5. What was deliberately not done

- No alternate venue, no REST endpoint standing in for the archive, no
  third-party mirror, and no S3 origin. `assert_allowed_url` refuses each on the
  parsed host, and the acquisition reached **no** URL outside the frozen plan.
- No authenticated endpoint, no API key, no credential of any kind.
- No `P4-HOLD` read, no Styx read, no P8.
- No object skipped, marked optional, or substituted. The acquisition either
  completes or stops.
- **No governed screen.** `run_offline_screen`, `run_screen` and `evaluate_block`
  were never called over this history, and the closure module cannot call them —
  asserted on its parsed call graph.

## 6. Files

| file | what it is |
| --- | --- |
| `acquisition_manifest.json` | per-object identity: published path, URL, both digests, published checksum, verification state, member name and sizes |
| `source_closure.json` | per-family and per-object source metadata, grid coverage, gaps, units, boundary truncation, and the per-block held-window coverage above |
| `STATUS.md` | this document |

Frozen under
[`../../btc_p13_a2r2_source_acquisition_SHA256SUMS.txt`](../../btc_p13_a2r2_source_acquisition_SHA256SUMS.txt).

**`CURRENT_RESULT_STATE` remains `NOT YET RUN`.** The first governed economic run
is the next chronology stage and requires explicit review of this closure.
