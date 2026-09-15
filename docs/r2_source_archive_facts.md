# R2 — gen4 preflight: first-party source and archive facts

**Phase:** R2 (gen4 engineering preflight), master roadmap `docs/master_roadmap_r0_r18.md` §37, R2.
**evidence_class:** `DIAGNOSTIC`.
**Retrieved:** 2026-09-14 (UTC).
**Base commit:** `13c34c4b89ff3f1a540d749ac41a3ef25aac04b3` (R0 adoption merge, PR #96).
**Revised:** 2026-09-15, after independent read-only review of PR #98 returned
REQUEST CHANGES. The corrections are **prose only** — provenance, scope, novelty
and timing. The reviewer independently re-fetched and re-parsed the first-party
objects and reproduced every measurement below exactly; **no measured value was
changed**, and no remeasurement was performed to support the rewrite.

## What this document is, and is not

This records **measured first-party source facts** for the streams R2's Tier A,
Tier B and (Lane-B-undecided) spot capture needs. R2's purpose is "measure before
freezing"; these are measurements, not decisions.

It is **not**:

- the R2 preflight report — that requires ≥ 30 consecutive days of live
  acquisition on an authorised second host and is not creatable yet;
- a freeze of R4's verification classes — R2 reports source facts, R4 freezes
  the acquisition contract;
- prospective evidence, alpha evidence, a strategy result, or any statement
  about returns. No price, return, PnL or predictive quantity is computed here.

Scientific standing is unchanged by this document. gen3's `prospective_from`
remains `null`. No real-money authority is created.

## Method

Every fact below was retrieved by this session from the **single allow-listed
first-party archive host** the repository already governs:

    https://data.binance.vision

over HTTPS, with no credential and no signature — the same source-integrity
rules `chimera/recorder/contracts/btcusdt-prospective-gen3.json`
(`reconciliation_rule`) and `nn/p13_acquisition.py` (`ALLOWED_HOST`,
`assert_allowed_url`, `parse_checksum_companion`) already impose. Where an
object's content is quoted, its published `.CHECKSUM` companion was fetched and
the digest verified against the bytes received.

Listings were taken from the archive's own S3 listing endpoint:

```
curl "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=<PREFIX>"
```

Object presence was probed with `curl -o /dev/null -w "%{http_code}"` against
`https://data.binance.vision/<KEY>`.

### Environment constraint that is a fact about this host, not about Binance

From the container this session runs in:

| Endpoint | Result |
|---|---|
| `https://data.binance.vision` | **200** — reachable |
| `https://fapi.binance.com/fapi/v1/ping` | **451** — blocked |

`451` is returned by the network path available to this session. It is **not**
evidence about the venue, and it must not be recorded as one. Its operational
consequence is concrete and is carried into the barrier record: **the REST-fed
parts of Tier A — open interest via REST, and the daily `exchangeInfo` snapshot
— cannot be exercised or payload-verified from this authoring environment.**
They must be verified on the authorised second host before live collection.

## Archive families that exist

Verified by prefix listing on 2026-09-14.

`data/futures/um/daily/`: `aggTrades/`, `bookDepth/`, `bookTicker/`,
`indexPriceKlines/`, `klines/`, `markPriceKlines/`, `metrics/`,
`premiumIndexKlines/`, `trades/`

`data/futures/um/monthly/`: `aggTrades/`, `bookTicker/`, `fundingRate/`,
`indexPriceKlines/`, `klines/`, `markPriceKlines/`, `premiumIndexKlines/`,
`trades/`

`data/spot/daily/`: `aggTrades/`, `klines/`, `trades/`

`data/spot/monthly/`: `aggTrades/`, `klines/`, `trades/`

The archive's whole namespace is `data/futures/{um,cm}/{daily,monthly}/…`,
`data/spot/…` and `data/option/…`. **There is no `exchangeInfo`, metadata,
symbol-reference or leverage-bracket family anywhere in it.** Tier A's "daily
`exchangeInfo` snapshot" therefore has no archive path at all and is reachable
only over REST — which is 451 from this environment. This is what blocks R2's
candidate-universe admissibility rule here: listing age, contract type, trading
status, tick/step/`minNotional` filters and leverage brackets all come from
`exchangeInfo`, and none of them can be established from the archive.

Three absences are load-bearing and are stated as absences of a *listed prefix*,
not as an inference from a single 404:

- **No `liquidationSnapshot/` (or any liquidation family) under
  `data/futures/um/daily/` or `data/futures/um/monthly/`.** A listing of
  `data/futures/um/daily/liquidationSnapshot/BTCUSDT/` returns
  `<IsTruncated>false</IsTruncated>` with **zero** `<Key>` entries.
  The adopted corrected audit already records "there is **no
  `liquidationSnapshot`** under `um/daily`", so the daily absence is a
  reproduction, not a discovery. What R2 adds here is narrow and methodological:
  the **monthly** path is checked too, and the absence is established by
  *directory enumeration* — an empty listing — rather than by inferring absence
  from a single 404 on a guessed key.
- **No `bookTicker/` under `data/spot/daily/` or `data/spot/monthly/`.**
- **No `exchangeInfo`/metadata family anywhere under `data/`.**

## Per-stream findings

"Denominator" below means: can this archive supply the *published* minute or
event set that a live-capture coverage ratio divides by, in the sense
`btcusdt-prospective-gen3.json`'s `coverage_rule` uses the term.

| R2 stream | First-party archive | Granularity | Latest verified | Denominator? |
|---|---|---|---|---|
| `um` 1m klines | `futures/um/daily/klines/{SYM}/1m/` | 1 minute | `2026-09-13` (200) | **Yes** — complete minute enumeration |
| `um` markPrice@1s | `futures/um/daily/markPriceKlines/{SYM}/1m/` | 1 **minute** | `2026-09-13` (200) | **Per-minute only** — see cadence mismatch below |
| `um` bookTicker | `futures/um/{daily,monthly}/bookTicker/` | per update | **daily `2024-03-30`; monthly `2024-04`** | **No** — discontinued |
| `um` aggTrade | `futures/um/daily/aggTrades/{SYM}/` | per trade | `2026-09-13` (200) | **Yes**; and `agg_trade_id` is contiguous in every archive sample tested, which makes internal gaps detectable — see the scope limits below |
| `um` funding | `futures/um/monthly/fundingRate/{SYM}/` **monthly only** | per settlement | **`2026-08`** | Yes, at month-scale latency |
| open interest (REST) | `futures/um/daily/metrics/{SYM}/` | **5 minutes** | `2026-09-13` (200) | **Not equivalent** — see below |
| forceOrder | **none** | — | — | **No archive at all** |
| `exchangeInfo` daily | **none** (REST only) | — | — | No; and REST is 451 from this host |
| Tier B depth20@100ms / diff depth | **none at message granularity**; `futures/um/daily/bookDepth/` only | **~30 s band snapshots** | `2026-09-13` (200) | **No** |
| `spot` kline_1m (Lane B) | `spot/daily/klines/{SYM}/1m/` | 1 minute | `2026-09-13` (200) | **Yes** |
| `spot` bookTicker (Lane B) | **none** | — | — | **No archive at all** |

### bookTicker is discontinued — independently reproduced, nothing new found

Probed directly:

```
futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03-29.zip -> 200
futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03-30.zip -> 200
futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03-31.zip -> 404
futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-04-01.zip -> 404
futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2025-01-15.zip -> 404
futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2026-09-10.zip -> 404
```

The **monthly** series has one further object, `BTCUSDT-bookTicker-2024-04.zip`
(200), and that is its last — `2026-06`, `2026-07`, `2026-08` all return 404,
and the monthly key listing ends at `2024-04`.

**Both halves of this were already recorded by the adopted corrected audit, and
R2 discovered neither.** `docs/governance/r0_adoption_2026-09-14/ProjectChimera_full_audit_and_master_roadmap_2026-09-14_corrected.md`
states, in bold: "**`bookTicker` daily objects end 2024-03-30 and monthly
2024-04** for BTCUSDT/ETHUSDT/SOLUSDT". What this section contributes is
**independent first-party reproduction** of that fact on 2026-09-14 — the daily
boundary probed either side, and the monthly tail confirmed by key listing
rather than by a single probe — not a refinement, and not a new object.

Neither series reaches a 2026 prospective period, so the operative conclusion is
unchanged and the gen3 contract's statement — that no contemporary first-party
archive publishes a minute denominator for `um.bookTicker` — holds on current
evidence.

### metrics is 5-minute cadence — confirmed

`futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-09-12.zip`, checksum
verified (`a9e3d74a6a5d6b448ea5de7bf79cbcc1cbe600f8aee99148d7f3c75ad8969b5e`,
matching its published `.CHECKSUM`), contains **289 lines = 1 header + 288
rows**, and 288 = 1440 / 5. Header and first row:

```
create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio
2026-09-12 00:05:00,BTCUSDT,103453.1500000000000000,7992877899.3611860000000000,1.62758097,2.20204500,1.58625258,1.01983000
```

Two parsing facts, verified on the same file: the 288 timestamps are **distinct
and complete but are NOT in chronological order** (the file's first row is
`00:05:00` and its last is `22:00:00`), so a consumer must sort.

And early `metrics` files carry **each row exactly twice**, so a naive row count
double-counts that history. Measured across the transition:

| Object | Rows | Distinct timestamps |
|---|---|---|
| `BTCUSDT-metrics-2020-10-01.zip` | 576 | 288 |
| `BTCUSDT-metrics-2021-05-20.zip` | 576 | 288 |
| `BTCUSDT-metrics-2021-05-25.zip` | 288 | 288 |

so the duplication ends between **2021-05-20 and 2021-05-25**.

Consequence for R2: this archive carries open interest, but at a **fixed
5-minute grid stamped at interval end**. R2's Tier A acquires OI **via REST**
at whatever cadence the recorder polls. The two are not the same series, and
the archive is therefore a *cross-check*, not a denominator for REST polling.
Any claim of equivalence would have to be established, not assumed.

### bookDepth is a sampled band file, not depth — confirmed

`futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2026-09-12.zip`, checksum
verified (`4faa59ec86693f647bacaee07bc6abd339295ad649011b9c8f1675ddf926bcb4`),
contains **34561 lines = 1 header + 34560 rows**. Header and first rows:

```
timestamp,percentage,depth,notional
2026-09-12 00:00:04,-5.00,8993.20000000,681319717.82160000
2026-09-12 00:00:04,-4.00,8295.82900000,629932457.03450000
2026-09-12 00:00:04,-3.00,7466.34000000,568104750.49130000
```

Parsed: **2880 distinct timestamps, each carrying exactly 12 rows** — the
percentage bands `±0.20, ±1.00, ±2.00, ±3.00, ±4.00, ±5.00`. The interval
between consecutive snapshots is **~30 s and jittered, not grid-aligned**
(observed deltas: 29 s ×1064, 31 s ×1055, 30 s ×233, 32 s ×220, 28 s ×215, with
a tail from 24 s to 37 s). So the file is ~2 snapshots per minute of *aggregated
resting quantity and notional inside price bands*.

This is an aggregate summary of book shape. It is **not** an order-book message
stream: there is no per-level and no per-update content. It cannot serve as a
denominator for `depth20@100ms` or for diff-depth, which are of the order of ten
messages per second, and nothing in the archive enumerates them.

One schema change matters for any fixed-width parse: the band set was **10 bands
(`±1,2,3,4,5`) through 2026-01-14 and 12 bands (adding `±0.20`) from 2026-01-15
onward** — and the transition happens *inside* 2026-01-15, whose file carries
both 10-band and 12-band snapshots.

Consequence for R2: **Tier B has no archive denominator at all.** Whatever Tier
B measures about missingness and depth sequence-continuity must come from the
recorder's own self-attested health metrics (sequence-number continuity in the
diff-depth stream being the strongest available internal check), not from
reconciliation against a published truth.

### aggTrade carries a contiguous identifier, and what that does and does not prove

`futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-09-12.zip`, checksum
verified (`c259f9306f0a5d30f965971234b7ba766db35f45f4a6f06dee9d8171c9253cd7`),
header
`agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker`.

Parsed: `n = 296171`, `min = 3447734194`, `max = 3448030364`, `span = 296171`,
duplicates `0`, sorted ascending — so the `agg_trade_id` sequence is **perfectly
contiguous**.

The independent reviewer of this PR reproduced the same property on three
further symbol-days — BTCUSDT `2026-09-11` (n = span = 1 722 572), ETHUSDT
`2026-09-12` (426 423) and SOLUSDT `2026-09-12` (186 131) — each contiguous,
zero duplicates, sorted. So the property is not a single-file accident.

**What this supports, stated exactly.** Within an observed `agg_trade_id`
sequence, a gap is detectable by identifier arithmetic on that sequence alone,
without comparison against a published denominator. That is a genuinely useful
internal check and no other R2 stream offers it.

**What it does not establish**, and must not be read as:

- It does **not** prove a live recorder capture is complete. The measurement is
  a property of *archive objects*. Carrying it to a live capture additionally
  requires that the live stream's identifier carries the same semantics and that
  capture continuity relates to it in the assumed way. `fapi.binance.com` is
  **451** from this environment, so the live payload semantics were **not**
  independently verified here, and no first-party source consulted for this
  document establishes that relationship.
- It does **not** prove `aggTrade` represents every venue event. The adopted
  corrected audit's standing caveat holds unchanged: "`aggTrade` excludes
  insurance-fund and ADL trades, so neither is a complete record". Contiguity is
  completeness *of the published aggregate-trade series*, not of trading
  activity.

So this is a candidate verification *method* worth R4's attention when
verification classes are frozen — not a self-certifying completeness guarantee,
and not a basis for a stronger class than the live-semantics evidence supports.

The `trades` stream does **not** share this property. On the same day,
`BTCUSDT-trades-2026-09-12.zip` (checksum verified,
`5aaf475f90b199390c2e80c620a7b3fc1cc0309b648d9a96862888f69c92d78c`) gives
`n = 791576` against a span of `797633` — **6057 identifiers absent inside the
day**, with zero duplicates. Those are IDs the venue does not publish, not data
the recorder lost, so a contiguity check transplanted from `aggTrade` to
`trades` would raise false alarms.

### funding is monthly-only, and its latency is real

`futures/um/monthly/fundingRate/BTCUSDT/` key listing ends at
`BTCUSDT-fundingRate-2026-08.zip` as of 2026-09-14. There is **no
`fundingRate` family under `futures/um/daily/`**.

This independently confirms the mechanism the gen3 contract already encodes:
"The funding schedule source is a monthly object, so a day inside a month whose
archive has not been published yet is `FUNDING_SCHEDULE_UNAVAILABLE` … That is
expected evidence latency and not a recorder fault." A 30-day R2 window will
end with its most recent days not yet funding-reconcilable, by construction.
R2 must not treat that as an outage, and must not shorten it by substituting a
REST read.

### markPrice cadence mismatch

Live Tier A captures `markPrice@1s`. The archive publishes
`markPriceKlines` at **1-minute** granularity. A coverage ratio built on it
therefore measures *minutes in which a mark observation existed*, not *1-second
updates delivered* — which is exactly the semantics gen3 already adopted
(per-minute mark open/high/low/close). R2 should carry that semantics forward
rather than invent a per-second denominator that no first-party source
publishes.

## Parser hazards verified in the archive itself

These are properties of the published files. Each would silently corrupt a
consumer that assumed otherwise, and each was confirmed by retrieving the files.

1. **Header rows were introduced mid-history.** `klines/BTCUSDT/1m` for
   `2022-06-15` has **1440 rows and no header** (its first field is
   `1655251200000`); `2022-09-15` has **1441 rows with a header** (first field
   `open_time`). A parser that assumes a header silently drops the first bar of
   every pre-2022 file; one that assumes none ingests the header as a row.
2. **`metrics` rows are not chronologically ordered**, and early `metrics` files
   are exactly doubled — see above.
3. **`bookDepth`'s band set changes within 2026-01-15** — see above.
4. **Archive objects are mutable.** The first-party public-data README states,
   verbatim: "Archived files may be updated at a later date as a result of
   recently discovered issues." Reproducibility therefore requires pinning the
   **`.CHECKSUM` digest**, not the date — which is what the gen3 contract's
   acquisition rules already require, and R2 must keep.

## First-party documentation coverage is a real gap

Binance's own public-data README — the "Public data document" linked from the
archive site, read at
`https://raw.githubusercontent.com/binance/binance-public-data/master/README.md`
— has a `### FUTURES` section containing exactly three subsections:
**`AggTrades`, `Klines` and `Trades`**. `metrics`, `bookDepth`, `bookTicker`,
`markPriceKlines`, `indexPriceKlines`, `premiumIndexKlines`, `fundingRate` and
any liquidation family carry **no first-party availability, cadence or
continuity commitment** there.

This is not a reason to avoid those streams — R2 measures what is actually
published. It is a reason to record that **six of the streams R2 itself depends
on are empirically observed rather than contractually documented**, which is
exactly the freedom under which `bookTicker` and the USD-M liquidation archive
were withdrawn without notice. The six, enumerated so the count is recoverable:
**`markPriceKlines`, `bookTicker`, `metrics` (OI), `bookDepth` (Tier B),
`fundingRate`, and any liquidation family**. (`indexPriceKlines` and
`premiumIndexKlines` are also undocumented but are not R2 streams, so they are
outside this count.) R4 should weigh that when it assigns verification
classes, and no R2 measurement should be read as a guarantee of future
publication.

## Object sizes actually retrieved

Recorded because they bear on second-host provisioning. These are **archive
object sizes, one symbol, one day**. They are **not** the R2 live-capture
storage measurement, which can only be made by running the recorder.

| Object | Compressed | Uncompressed CSV |
|---|---|---|
| `BTCUSDT-metrics-2026-09-12.zip` | 11 158 B | 35 891 B |
| `BTCUSDT-bookDepth-2026-09-12.zip` | 554 208 B | 2 010 708 B |

## Publication latency

Measured, not assumed. On 2026-09-14 the daily object for **`2026-09-13` was
already present (200)** for `klines`, `aggTrades` and `metrics`, while
`2026-09-14` was absent (404, expected — the day is incomplete). The
`Last-Modified` header on `BTCUSDT-1m-2026-09-13.zip` is
`Mon, 14 Sep 2026 09:05:01 GMT`.

**Observed** daily publication is therefore **T+1**, landing in the morning UTC
of D+1. This is an observation from a sampled publication on one date, not a
contractual guarantee: the first-party README carries **no publication-latency
commitment** for any futures family (see the documentation-coverage section
below), so T+1 must be treated as current observed behaviour that may change
without notice, and a recorder must not depend on it without its own check.

On that observed behaviour, publication is *earlier* than the gen3 contract's
"once per UTC day, for day D-2" reconciliation cadence, so the cadence carries
margin rather than running at the edge. The size of that margin depends on the
hour the reconciliation job runs — roughly 15 h if it runs at 00:00 UTC on day
D, longer if it runs later in the day — and is not a fixed day.

`fundingRate` is the exception and is month-scale, as above.

## Contradictions and refinements against the prior audit

| Audit claim | Status on current first-party evidence |
|---|---|
| "**`bookTicker` daily objects end 2024-03-30 and monthly 2024-04**" | **Confirmed, both halves.** R2 adds nothing here: the monthly tail was already the audit's finding. R2's contribution is independent first-party reproduction — the daily boundary probed either side, the monthly tail read off the key listing. |
| `metrics` at 5-minute cadence | **Confirmed** (288 rows/day, verified checksum). |
| `bookDepth` is a sampled band file (~30 s cadence, ±1–5% bands), not L2 | **Confirmed**, with the structure stated precisely: **12 rows per snapshot** (bands ±0.20, 1, 2, 3, 4, 5) at **~2 snapshots per minute** (2880 snapshots/day, ~30 s jittered), i.e. 24 rows per minute across two distinct snapshots — never 24 rows in one. R2 adds the ±0.20 band and the mid-file 10→12 band change on `2026-01-15`. |
| `forceOrder` is `DESCRIPTIVE_ONLY`; "there is **no `liquidationSnapshot`** under `um/daily`" | **Confirmed.** The daily absence is the audit's own finding, reproduced. R2 adds only that the **monthly** path is equally empty, and that both were established by directory enumeration rather than by a single 404. The record for the archive family is therefore **ABSENT** at any date, not "ended" — an ended stream leaves usable history; this one leaves none. |

No contradiction of the audit's archive facts was found.

**What this session independently reproduced** (already recorded by the adopted
corrected audit, and *not* discovered here): the bookTicker daily end
`2024-03-30` **and** its monthly tail `2024-04`; the absence of any
`liquidationSnapshot` under `um/daily`; `metrics` at 5-minute cadence;
`bookDepth` as a sampled band file at ~30 s cadence; `fundingRate` monthly-only;
and archive mutability.

**What this session actually adds**, stated without inflation:

- the **monthly**-path check for the liquidation family, and directory
  enumeration (empty listing) rather than single-404 inference as the method for
  all three load-bearing absences;
- the absence of any `exchangeInfo`/metadata family **anywhere** under `data/`,
  and its consequence for the candidate-universe admissibility rule;
- the measured `bookDepth` structure — 12 bands including **±0.20**, ~2
  snapshots/minute, and the 10→12 band change *inside* `2026-01-15`;
- the `metrics` row-order and early-file duplication facts, with the transition
  bracketed between `2021-05-20` and `2021-05-25`;
- the `aggTrade` identifier contiguity **and its scope limits**, together with
  the contrasting `trades` gaps that show the check must not be transplanted;
- the observed T+1 publication latency, as an observation and not a guarantee;
- the four parser hazards above;
- and, throughout, per-object `.CHECKSUM` verification so each number is
  reproducible rather than taken on trust.

## Governed inputs this document does NOT supply

These remain open and are escalated rather than invented here. **None of their
values is chosen in this document**, and none may be chosen by an authoring
session; they are governance inputs.

### Timing: what must be frozen BEFORE live R2 begins

The following **MUST be resolved and frozen before the 30-consecutive-day live
R2 preflight begins**, because each one either decides which data is eligible,
decides how the universe is constructed, or is an input to R2's own
kill/deferral rule — and a value chosen after observing the 30-day run would be
a threshold or a universe selected on the result:

| Input | Why it cannot wait |
|---|---|
| **gen4 coverage thresholds** | An input to the R2 kill rule ("≥ 2 of the 30 days failing coverage thresholds for core streams"). Selecting the threshold after seeing the days' coverage is choosing a pass mark from the result. |
| **The exact definition of "core streams"** | The same kill rule's stream set. Choosing which streams count after seeing which ones degraded is the same defect by another route. |
| **The candidate-universe mechanical rule** — exact volume field (`quote_volume` vs base `volume`), the tie-break, and the named selection date | Decides *which ~20 symbols are recorded at all*. A symbol not captured cannot be added retroactively, so this is a hard precondition on starting collection, not a reporting detail. It is additionally gated on `exchangeInfo`, which has no archive path and is 451 from this environment. |
| **The Tier B storage/replay budget** | The second half of the kill rule ("Tier B cost beyond budget → Tier B shrinks or is deferred"). An undefined budget makes that condition unevaluable, and a budget set after measuring the cost is the same post-hoc selection. It is also an owner spend decision, inseparable from authorising the second host. |

### Timing: what need not block acquisition

| Input | Standing |
|---|---|
| **Trade-through reference-size semantics** | Collection itself does **not** depend on a single fixed size: trade-through is computable after the fact from captured `aggTrade`, `bookTicker` and depth data. A later R2 diagnostic may therefore report the cost envelope **as a function of size over a declared size grid**, which breaks the R2↔R3 circularity without anyone inventing a trading size. **No R3 trading size is selected here**, and the selection of the single deciding size remains R3's. |
| **The daily-return definition** | Must be **frozen before the correlation / effective-N diagnostic is computed**, because effective-N feeds R3's power calculation — but it does not block raw data acquisition. |
| **`evidence_class = DIAGNOSTIC` as a machine-checkable value** | Required **before final R2 report acceptance**, not before collection, for as long as R1-o remains the owning implementation. |

### The inputs themselves

1. **Core-stream coverage thresholds for gen4.** gen3's contract fixes
   `published_coverage >= 0.995`, `wallclock_coverage < 0.990 → RECORDER_OUTAGE`,
   "three flagged days in a window fail the gate" — as **contract text for
   gen3 only**, with no executable enforcement on `main` (the evaluator exists
   only on the unmerged PR #76 branch). R2's own kill rule is "**≥ 2 of the 30
   days failing** coverage thresholds for core streams". "Failing" ≠ "flagged"
   and 2 ≠ 3; "core streams" is undefined anywhere at HEAD.
2. **The reference size** for realised trade-through. R2 is told to measure it;
   R3 cites "the reference size **from R2**". Nothing defines it. This is a
   circular reference and the value feeds R3's deciding slippage envelope, so
   it is not a free engineering parameter.
3. **The universe rule's operative details** — the volume field
   (`quote_volume` vs base `volume`), the tie-break, and the "named date" are
   undefined; and §18's admissibility filters are scoped to the K = 10–12
   campaign universe, not R2's ~20-symbol preflight set.
4. **The daily-return definition** for the correlation diagnostic.
5. **The Tier B storage/replay budget** the kill condition refers to. R0 fixed
   a *candidate* budget and an effort cap; no storage or cost budget exists.
6. **`evidence_class = DIAGNOSTIC`** as a machine-checkable value. `main` emits
   only lowercase `"engineering"`/`"prospective"` in the heartbeat; the manifest
   field and verifier check are R1-o, which is unbuilt.

Item 6 additionally means R2's stated acceptance criterion depends on an R1
deliverable, although R1 and R2 are scheduled in parallel.

Items 1, 3 and 5 are the ones carrying the pre-live freeze requirement above.

## Reproducing this

Every row above is reproducible with `curl` against the one allow-listed host,
using the listing and probe commands given under **Method**. No credential is
required and none was presented. No repository state was mutated to produce it.
