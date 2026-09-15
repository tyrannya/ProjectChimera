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

This document draws on **three distinct evidence classes from three distinct
sources**, kept apart deliberately because they cannot be verified the same way:
archive **objects** retrieved and checksum-verified from the governed host; a
**documentation** claim read from a commit-pinned repository; and archive
**namespace enumeration** taken from a listing origin that is *not* the governed
host and whose admissibility is unresolved.

### 1. Archive object measurements — `data.binance.vision`

**Every archive object measurement below** — every object probe, every row count
and every parsed value — was retrieved by this session from the **single
allow-listed first-party archive host** the repository already governs:

    https://data.binance.vision

over HTTPS, **with no credential and no signature**, which is the same
source-integrity rule `chimera/recorder/contracts/btcusdt-prospective-gen3.json`
(`reconciliation_rule`) and `nn/p13_acquisition.py` (`ALLOWED_HOST`,
`assert_allowed_url`, `parse_checksum_companion`) already impose. Where an
object's content is quoted, its published `.CHECKSUM` companion was fetched and
the digest verified against the bytes received.

**This class does not include namespace enumeration.** Listing which archive
families exist under a prefix was *not* done against this host and is *not*
covered by anything in this section — see **§ 3**, which records the separate
origin those listings actually came from and why it is not interchangeable with
this one.

That no-credential statement is a fact about **this archive object acquisition
only**. It is not a claim that every input R2 will eventually need is
unauthenticated — the leverage-bracket dependency recorded below is not.

### 2. Documentation claims — Binance's public-data repository, commit-pinned

What this document says about what Binance *documents* comes from Binance's own
public-data repository, **not** from the archive host. That repository is served
by `raw.githubusercontent.com`, a **different host under different governance**;
nothing here claims the two are the same, and the archive listing/probe
procedure above cannot reproduce a README finding.

So that this evidence is immutable rather than pinned to a moving branch, the
README is cited at an **exact commit**, never at `master`:

| Field | Value |
|---|---|
| Repository | `binance/binance-public-data` (Binance-owned) |
| Commit | `5c7f3197591c0d54d85dc43066226bc4c671d47a` |
| Path | `README.md` |
| Retrieved from | `https://raw.githubusercontent.com/binance/binance-public-data/5c7f3197591c0d54d85dc43066226bc4c671d47a/README.md` |
| Size | 5144 bytes |
| SHA-256 | `085ab91377aa9325d44f4c7ad27cce4ab381e158403e1d7df2bad39d1a66f7c6` |
| Git blob id | `311354cd82a76bcaaec588e6818e6c12644abef0` |

The commit was resolved with `git ls-remote https://github.com/binance/binance-public-data`,
and the pinned copy was byte-compared against the `master` copy fetched the same
day: identical. A future `master` may differ; this revision cannot.

### 3. Namespace / listing evidence — an S3 origin that is NOT the allow-listed host

**This is a third provenance class, and it is weaker than the other two. It is
recorded here rather than folded into §1, because folding it in would make §1
untrue.**

Object presence and object content (§1) were obtained from the allow-listed host:

```
curl -o /dev/null -w "%{http_code}" "https://data.binance.vision/<KEY>"
```

But *enumeration* of what exists under a prefix — which archive families are
published, and which are absent — was taken from the bucket's S3 listing origin:

```
curl "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=<PREFIX>"
```

**That hostname is not `data.binance.vision`, and this repository's governed
acquisition rule refuses it by name.** `nn/p13_acquisition.py::assert_allowed_url`
rejects any hostname other than `data.binance.vision`, with the message: *"No
alternate venue, no REST endpoint, no mirror and **no S3 origin** may stand in
for it without an explicit amendment."* `tests/test_p13_acquisition.py` pins that
behaviour with this exact URL shape among its refused cases. So the listings
were **not** produced under the P13 procedure, and this document does not claim
they were.

**The allow-listed host cannot substitute.** `https://data.binance.vision/?prefix=…`
returns the site's HTML browser page, not an S3 `ListBucketResult`; it yields
zero `CommonPrefixes`. There is no equivalent listing endpoint on the governed
hostname, so the enumeration cannot simply be redone there.

**Which claims depend on the S3 origin.** Everything that asserts what the
archive namespace *contains or lacks by enumeration*: the archive-family lists
for `futures/um/{daily,monthly}` and `spot/{daily,monthly}`; the "zero keys"
result for `liquidationSnapshot`; the absence of a `bookTicker` family under
spot; the absence of any `exchangeInfo`/metadata family anywhere under `data/`;
and the "last published key" statements for monthly `bookTicker` (`2024-04`) and
monthly `fundingRate` (`2026-08`).

**What does not depend on it.** Every measurement in §1 — every status probe,
every downloaded object, every `.CHECKSUM` verification, every parsed row count
and value — came from the allow-listed host and stands on its own.

**Corroboration from the allow-listed host.** The three load-bearing absences
were re-probed directly on `data.binance.vision`, with a positive control on the
same host and path shape to show the probe is meaningful:

| Probe (on `data.binance.vision`) | Result |
|---|---|
| `futures/um/daily/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-2023-01-01.zip` | 404 |
| `futures/um/daily/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-2024-06-01.zip` | 404 |
| `futures/um/monthly/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-2023-01.zip` | 404 |
| `spot/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2026-09-13.zip` | 404 |
| `spot/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-01-15.zip` | 404 |
| `futures/um/daily/exchangeInfo/BTCUSDT/BTCUSDT-exchangeInfo-2026-09-13.zip` | 404 |
| **control** `futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-09-13.zip` | **200** |

This corroboration is **weaker than enumeration**: a 404 shows one object is
absent, while an empty listing shows a whole family is. The absences are
therefore supported on the governed host by probe, and supported *more strongly*
only by the ungoverned S3 origin. The two are not interchangeable and this
document does not treat them as such.

**Open governance question — recorded, not resolved here.** Whether the S3
listing origin may count as governed evidence for this repository is a
**source-governance decision that has not been made**. `nn/p13_acquisition.py`
is P13's frozen rule and does not by itself bind R2's own acquisition, but no R2
contract yet names an allowed listing source at all. This document does not
invent a retroactive authorisation. It is carried as a **pre-live prerequisite**
in the timing table below.

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
only over REST — which is 451 from this environment.

This is what blocks R2's candidate-universe admissibility rule here, and the
blocking inputs divide into **two different access classes that must not be
conflated**:

- **Public, unauthenticated market metadata — `GET /fapi/v1/exchangeInfo`.**
  Listing/onboard date, contract type, trading status and the symbol filters
  (tick size, step size and the `minNotional`-related filters) come from here.
  No credential is required; the endpoint is simply unreachable from this
  container.
- **Authenticated account metadata — `GET /fapi/v1/leverageBracket`.**
  Notional and leverage brackets are **NOT an `exchangeInfo` field.** They are a
  separate endpoint that Binance classifies as **`USER_DATA`**, which is
  **signed** and therefore requires an API key and secret.

The adopted §18 universe procedure requires rejecting *leverage-bracket
anomalies*. A prerequisite phrased as "`exchangeInfo` is reachable" is therefore
**not sufficient** for that rule: it would let R2 begin with an incomplete
universe check, or discover only at collection time that an authenticated
credential and a second daily snapshot are required.

**Verified first-party**, from Binance's own official connector
`binance/binance-futures-connector-python` at commit
`a6bfbbf10fe2c1b4eb76fc24ffb82eb94bf9df89`:

| | `exchange_info()` | `leverage_brackets()` |
|---|---|---|
| Module | `binance/um_futures/market.py` | `binance/um_futures/account.py` |
| Endpoint | `GET /fapi/v1/exchangeInfo` | `GET /fapi/v1/leverageBracket` |
| Binance's own label | "Exchange Information" | "**Notional and Leverage Brackets (USER_DATA)**" |
| Call | `self.query(url_path)` — unsigned | `self.sign_request("GET", url_path, params)` — **signed** |
| Binance doc path | `…/usds-margined-futures/`**`market-data`**`/rest-api/Exchange-Information` | `…/usds-margined-futures/`**`account`**`/rest-api/Notional-and-Leverage-Brackets` |

**No credential is invented, requested or provisioned by this document**, and it
does **not** decide whether the eventual mechanical universe rule must use
leverage brackets at all. It records the dependency, its endpoint class and its
access cost, so governance can settle that question **before** live R2
collection begins.

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
4. **Archive objects are mutable.** The public-data README states, verbatim:
   "Archived files may be updated at a later date as a result of recently
   discovered issues." (A documentation claim, from the commit-pinned source
   under *Method § 2* — not an archive measurement.) Reproducibility therefore
   requires pinning the
   **`.CHECKSUM` digest**, not the date — which is what the gen3 contract's
   acquisition rules already require, and R2 must keep.

## First-party documentation coverage is a real gap

This is a **documentation claim, not an archive measurement**. Its source is the
commit-pinned README recorded under *Method § 2* — `binance/binance-public-data`
at commit `5c7f3197591c0d54d85dc43066226bc4c671d47a`, SHA-256
`085ab91377aa9325d44f4c7ad27cce4ab381e158403e1d7df2bad39d1a66f7c6` — served by
`raw.githubusercontent.com`, which is **not** `data.binance.vision` and is not
reproducible by the archive listing/probe procedure.

At that pinned revision, Binance's own public-data README — the "Public data
document" linked from the archive site — has a `### FUTURES` section containing
exactly three subsections:
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
  and its consequence for the candidate-universe admissibility rule — including
  that leverage brackets are not an `exchangeInfo` field at all, but a separate
  **signed `USER_DATA`** endpoint;
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
| **The candidate-universe mechanical rule** — exact volume field (`quote_volume` vs base `volume`), the tie-break, and the named selection date | Decides *which ~20 symbols are recorded at all*. A symbol not captured cannot be added retroactively, so this is a hard precondition on starting collection, not a reporting detail. It is additionally gated on **public** `exchangeInfo`, which has no archive path and is 451 from this environment — and, if §18's leverage-bracket condition is retained, on the separate **signed** endpoint in the row below. |
| **Whether the universe rule retains §18's leverage-bracket condition — and if so, how that data is authorised and obtained** | §18 rejects *leverage-bracket anomalies*, but brackets are **not** an `exchangeInfo` field: they come from `GET /fapi/v1/leverageBracket`, which Binance classifies as signed **`USER_DATA`**. Keeping the condition therefore introduces an **authenticated credential dependency** and a second daily snapshot. Whether to keep it is a governance decision, not R2's — but it must be settled *before* collection starts, because discovering the credential requirement mid-run leaves the universe check incomplete or forces a restart. Reaching `exchangeInfo` alone does not satisfy this. |
| **The Tier B storage/replay budget** | The second half of the kill rule ("Tier B cost beyond budget → Tier B shrinks or is deferred"). An undefined budget makes that condition unevaluable, and a budget set after measuring the cost is the same post-hoc selection. It is also an owner spend decision, inseparable from authorising the second host. |
| **The reference-size / trade-through measurement method** | **Withdrawn from "need not block acquisition" — this was wrong.** The adopted capture gives depth (Tier B) to **BTCUSDT and ETHUSDT only**; the other ~18 Tier A symbols get `bookTicker`, which carries only best bid/ask and their sizes (`chimera/recorder/events.py`: `BookTickerEvent` has exactly `bid`, `bid_qty`, `ask`, `ask_qty`), plus `aggTrade`, which is executed prints and not resting depth. **For any size exceeding the displayed top level there is no captured data from which a hypothetical sweep can be reconstructed for those symbols**, so a post-hoc size grid is *not* generically computable and the 30-day run could complete without the data R2's cost envelope and R3's slippage envelope require. The method must therefore be frozen **before** collection and must be identifiable from the data actually captured **for every candidate symbol**. |
| **The gen4-preflight contract-schema interpretation** | The roadmap specifies the R2 contract as `gen4-preflight` with **`prospective_from` absent by schema**, but `chimera/recorder/contract.py` lists `prospective_from` in `REQUIRED_FIELDS` and the parser refuses any file missing a required field. **R2 cannot instantiate the specified contract under today's schema.** Whether "absent" means the key is omitted or is present-and-`null` is not resolved by any governance record located, and is not decided here. |
| **Whether the S3 listing origin is governed evidence** | The archive-family and absence claims rest partly on an S3 listing origin that is not the allow-listed hostname (see *Method § 3*). Whether that source is admissible — or whether an equivalent must be found, or an amendment recorded — is an open source-governance decision. It is listed here because the candidate-universe and stream-inventory work depends on knowing what the archive publishes. |

#### The reference-size problem has exactly two architectural resolutions

Recorded so the choice is visible. **Neither is elected here**, and electing one
is a governance decision, not R2's:

- **A — capture sufficient depth for every candidate symbol.** This makes a
  post-hoc size grid genuinely computable, at the cost of widening Tier B from
  two symbols to ~20. Tier B exists precisely to *measure* whether depth is
  practical, and its storage cost is itself an unresolved governed input, so
  this cannot be adopted silently.
- **B — freeze a reference-size method measurable from Tier A alone.** For
  example a method grounded in *actually observed* sweeps — sequences of
  `aggTrade` prints that cross the prevailing `bookTicker` touch — rather than a
  hypothetical order walking an order book that was never recorded. This keeps
  Tier B at two symbols but constrains what "trade-through" can mean.

**This document does not widen Tier B, and does not select the R3 deciding trade
size.** It records that the two options exist, that they have different capture
consequences, and that one of them must be settled before the 30-day clock
starts.

### Timing: what need not block acquisition

| Input | Standing |
|---|---|
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

**There is no single-host reproduction path for this document, and none is
offered.** The three evidence classes reproduce differently, and one of them
cannot be reproduced on the governed host at all.

**1. Archive object measurements** — reproducible with `curl` against the one
allow-listed archive host, using the **object-probe** command given under
**Method § 1**. For those, no credential is required and none was presented.

**2. Documentation claims** — reproducible from the commit-pinned README
recorded under **Method § 2**: a different host, and a source whose identity is
fixed by commit SHA and content digest rather than by the `master` branch.

**3. Namespace enumeration** — reproducible **only** through the listing origin
described in **Method § 3**, which is `s3-ap-northeast-1.amazonaws.com` and
**not** `data.binance.vision`. Method § 1 contains no listing command and never
did reproduce these claims. The governed host offers no equivalent listing
endpoint — `https://data.binance.vision/?prefix=…` returns the site's HTML
browser page with zero `CommonPrefixes` — so this class **cannot** be reproduced
against the allow-listed host, and no such path is claimed here. That origin is
a distinct provenance class whose **governance admissibility remains
unresolved**; it is carried as a pre-live prerequisite above.

The governed-host absence probes in § 3 **corroborate** certain listing-derived
claims but do **not** reproduce the enumeration: a 404 shows one object absent,
while an empty listing shows an entire family absent. They are weaker evidence
of a different shape, not a substitute.

Neither statement extends to the venue's authenticated endpoints: the
`leverageBracket` dependency recorded above is signed `USER_DATA`, and **no
credential was invented, requested, provisioned or used anywhere in producing
this document**.

No repository state was mutated to produce it.
