# R2 ↔ R1 recorder-core coordination record

**Phase:** R2 (gen4 engineering preflight), running in parallel with R1.
**Status:** `RECORDER_CORE_BARRIER` — R2's recorder-generalisation work is held.
**Base commit:** `13c34c4b89ff3f1a540d749ac41a3ef25aac04b3`.
**Written:** 2026-09-14.
**Revised:** 2026-09-15, after independent read-only review of PR #98. Prose
only; every module claim below was independently re-verified against HEAD by the
reviewer and none changed.

This is an **operational coordination record**, not evidence and not governance.
It exists because R1 and R2 are executed by separate sessions that cannot see
each other's local state, and the adopted roadmap makes them share one recorder
core. Coordination therefore has to live in the repository.

**This record is not to be deleted.** It is dated chronological evidence of what
was verified, and when, and of the barrier that was honoured rather than raced.
Once R1-g and R1-h merge, its status is **superseded by a later reviewed
record** that states the new position and cites this one; the file itself stays
in place so the chronology remains readable.

## Naming correction (read this first)

R2 task prompts have referred to the shared recorder-hygiene subphase as
**"R1-f2"**. **No subphase by that name exists in the adopted roadmap.**
`docs/master_roadmap_r0_r18.md` §37 R1 lists `R1-a … R1-o`, and the substance
the prompt attributes to "R1-f2" maps onto two adopted items:

| Prompt's term | Adopted roadmap | Substance |
|---|---|---|
| "R1-f2" (atomic parquet publication, silence watchdog/reconnect, `recorder.down`/`recorder.up`) | **R1-h** | "Atomic parquet publication and recorder evidence" |
| "R1-f2" (incremental normalisation freshness) | **R1-g** | "Recorder / runner cadence" — publish the last closed minute incrementally |
| "R1-g supply-chain/ops files" | **R1-n** | "Supply chain and operations" |

The adopted roadmap's own `R1-f` is *real staleness detection*, which is a
different item again. This is a **label** mismatch, not a governance conflict:
the semantics the prompt protects are exactly the semantics R1-g and R1-h own,
so the coordination rule is applied unchanged, using the adopted labels. It is
recorded here so that neither session mistakes one letter for another.

## Current state of R1 (verified, not assumed)

As of 2026-09-14 22:50 UTC, on `origin`:

- `origin/main` = `13c34c4b89ff3f1a540d749ac41a3ef25aac04b3` (unchanged since R0 adoption).
- **PR #97 — `R1-a` source identity / CAMPAIGN self-check**
  (`claude/projectchimera-r1-remediation-5demn6`, draft, head
  `fdbb6b2353821162e92e27267d334d8771c228cd`). R1 has started.
- PR #76 (`pr-06/recorder-reconciliation-coverage`, draft, head
  `8a8f4a1f7d754cff190a3668873072a7efbb1542`). Not an R1 PR.

**PR #97 does not touch the barrier.** Its three files are
`tools/demo_run.py`, `tests/test_demo_cli.py` and `docs/demo_runbook.md` —
none of them a recorder module, and none of them a file R2 touches. R1-a and
this R2 work are cleanly disjoint.

**R1-g and R1-h are still not started.** No branch, no PR, and nothing in
`normalize.py`, `streams.py`, `service.py` or `incremental.py` has changed. The
barrier below therefore stands unchanged.

Both sessions should re-verify this from GitHub rather than from this record,
which ages.

## The overlap map

Reconstructed by reading the modules at HEAD, not from prose.

### R2 needs, and where it lands

R2's recorder generalisation is "the gen3 recorder generalised to
(symbol, stream)-keyed files". At HEAD the recorder is single-symbol-per-market
in four concrete ways:

| # | Where | What blocks multi-symbol |
|---|---|---|
| 1 | `chimera/recorder/normalize.py` | `MARKET_COLUMNS` is keyed by the literal market names `um` / `spot`; `columns_for(market)` refuses anything else. Normalised output is one parquet per **market**-day. |
| 2 | `chimera/recorder/streams.py:203` | `WEBSOCKET_STREAMS` is keyed by literal full stream ids (`um.kline_1m`, `um.markPrice`, …). |
| 3 | `chimera/recorder/service.py:552,599` | Funding and mark-price sinks are addressed as the singletons `UM_FUNDING` / `UM_MARK_PRICE`. |
| 4 | `chimera/recorder/events.py:107-117` | Stream ids are module constants and frozen sets (`UM_KLINE_1M`, `KLINE_STREAMS`, `BOOK_TICKER_STREAMS`). |

**What is already general and needs no change:**

- `chimera/recorder/contract.py` places **no limit on the number of markets**.
  Validation only requires each stream's `market.` prefix to name a declared
  market (`contract.py:564-580`). A gen4 contract may declare twenty markets
  with twenty symbols today, so **multi-symbol support is not what the schema
  blocks**.

  **But the schema IS a blocker for the `gen4-preflight` contract, for a
  different reason.** An earlier revision of this record said "the contract
  schema is not the blocker"; that was too broad and is withdrawn. The adopted
  roadmap specifies the R2 contract as `gen4-preflight` with **`prospective_from`
  absent by schema**, while `contract.py` lists `prospective_from` in
  `REQUIRED_FIELDS` and the parser refuses any file missing a required field
  (`missing = [name for name in REQUIRED_FIELDS if name not in payload]` → raise).
  **R2 cannot instantiate the specified contract under today's schema.** This is
  recorded as a pre-live blocker below; it is not resolved here, and
  `contract.py` is not modified by this documentation PR.
- `chimera/recorder/sink.py` is already **(stream)-keyed**: raw paths are
  `raw/<stream_id>/<day>/events.ndjson`, and `RawSink` is constructed per
  stream. Raw storage needs no generalisation.
- `sink.py` already writes atomically — `write_bytes_atomic` (`sink.py:272`)
  does temp-file + `fsync` + `os.replace`, and `write_json_atomic` wraps it.
- `health.py` already writes the heartbeat atomically.

### R1-h's surface, and why it collides

R1-h is "temp-file + fsync + rename for **every parquet** and manifest write;
an in-process **silence watchdog** that reconnects on stream silence; persisted
`recorder.down` / `recorder.up` records".

Verified gaps at HEAD:

- **`normalize.py:896-899` writes the parquet non-atomically:**

  ```python
  parquet = self.parquet_path(market, day)
  parquet.parent.mkdir(parents=True, exist_ok=True)
  frame.to_parquet(parquet, index=False, compression="zstd", compression_level=19)
  parquet_sha = hashlib.sha256(parquet.read_bytes()).hexdigest()
  ```

  This is inside `NormalizedStore.write_day` — **the same function** a
  (symbol, stream) generalisation must change, and the same module that owns
  `MARKET_COLUMNS`. The collision is direct and unavoidable.
- **No silence watchdog exists.** `streams.py` has reconnect-on-error and a
  proactive 23 h 50 m reconnect, but no symbol for `silence`/`watchdog`; a
  socket that goes quiet without erroring is not detected.
- **No persisted `recorder.down`/`recorder.up` records exist** in `health.py`.

### R1-g's surface, and why it collides

R1-g requires the recorder to "publish the last closed minute **incrementally**
(seconds, not the 300 s full-day cadence)". At HEAD
`service.py:107` sets `NORMALIZE_INTERVAL_S = 300.0`, and the maintenance loop
around `service.py:711-733` is built on it. The incremental normaliser's
per-market `DayState` (`incremental.py:282`) is the structure both R1-g's
cadence change and R2's per-symbol fan-out would rewrite.

### Classification

| R2 work | Files | Class |
|---|---|---|
| Multi-symbol/multi-stream recorder generalisation | `normalize.py`, `streams.py`, `service.py`, `incremental.py` | **RECORDER_CORE_OVERLAP** — held |
| `gen4-preflight` contract schema/parser work (`prospective_from` absent by schema) | `chimera/recorder/contract.py` | **NOT owned by anyone yet** — separate reviewed engineering work, not part of this documentation PR and not part of the R1 barrier. Must exist before R2 can instantiate its contract. |
| New Tier A/B parsers (aggTrade, forceOrder, depth, OI) | `events.py`, then wiring into `normalize.py`/`service.py` | **RECORDER_CORE_OVERLAP** once wired; the parsers alone are additive but are not useful unwired, and their payload semantics are unverifiable from this host (`fapi` 451) |
| gen4 archive layouts / reconciliation for new streams | would duplicate `reconcile.py`/`coverage.py` | **Held for a different reason** — R5 owns generalising **PR #76's** implementation; building a second fetcher now would be the alternate implementation the coordination rule forbids |
| Candidate-universe selection module | new module | **Blocked on governed inputs** — volume field, tie-break and named date undefined; admissibility needs public `exchangeInfo` (451 from this host) and, if §18's leverage-bracket condition is retained, the *separate signed* `USER_DATA` endpoint `GET /fapi/v1/leverageBracket` |
| First-party source/archive fact verification | none (documentation) | **NON_OVERLAPPING** — done; see `docs/r2_source_archive_facts.md` |
| This coordination record | none | **NON_OVERLAPPING** — done |

## The barrier

**R2 will not implement the recorder-core generalisation while R1-g and R1-h
are unmerged.** No alternate abstraction, no "temporary" parallel solution, no
race.

### What R1 must land before R2 resumes

1. **R1-h**: atomic parquet publication in `NormalizedStore.write_day`; the
   in-process silence watchdog and its reconnect; persisted `recorder.down` /
   `recorder.up` records. R1-h's own constraint — "gen3's contract hash is
   unchanged by any of this" — is compatible with R2, which introduces a
   *separate* `gen4-preflight` contract and never edits the gen3 file.
2. **R1-g**: incremental publication of the last closed minute, replacing the
   300 s cadence.

### What R2 will do once they merge

Fetch the new `main`, read **the implementation actually merged**, and
generalise *that* to (symbol, stream). R2 will not restore pre-R1 assumptions
and will not re-derive an atomicity or watchdog design of its own. If an R2
branch needs current `main`, it takes it as an **ordinary merge commit** —
never a rebase or force-push of published history.

### What R2 asks R1 not to do

Nothing is asked of R1 beyond its own scope. For symmetry, R2 records that it
has **not** touched, and will not touch while R1 is active, the modules R1-g
and R1-h must change — `normalize.py`, `streams.py`, `service.py`,
`incremental.py`, `health.py` — nor the R1-n supply-chain/ops files
(`.github/workflows/ci.yml`, `requirements-lock.txt`, `Dockerfile`,
`docker-compose.yml`, `deploy/`).

`events.py` is **not** claimed by any R1 item, so R2 may add parsers there. What
R2 will not do while the barrier stands is **wire** them into the normalisation
or service path, because that wiring is the overlap. A parser added without its
wiring would also be unverifiable today: `fapi.binance.com` is 451 from this
environment, so the live payload semantics it must implement cannot be confirmed
against the first-party source, and guessing them is precisely what R2 forbids.

## PR #76 isolation

PR #76 is untouched by this work and is not R2's environment. This session did
not merge, rebase, update, comment on or check out PR #76; did not contact its
VPS; and read nothing from its quarantine. Its coverage/reconciliation code was
identified as existing only on its branch, which is a fact about where the code
lives, established without checking the branch out.

R2 requires a **second host**, physically and logically separate from the PR #76
recorder deployment and its storage root. None is authorised yet, so no live
collection has begun and none may begin here.

## Other blocking preconditions for R2 live collection

Independent of the R1 barrier, live collection additionally requires:

- an authorised second host (not provisioned; no paid infrastructure may be
  bought without owner authorisation);
- the `gen4-preflight` contract identity fixed, which requires the candidate
  universe, which requires the governed inputs listed in
  `docs/r2_source_archive_facts.md`;
- the **public, unauthenticated** venue endpoints reachable from that host —
  `exchangeInfo` and the OI REST endpoint, both **451** from this authoring
  environment;
- a settled governance decision on **authenticated venue metadata**. §18's
  universe procedure rejects leverage-bracket anomalies, and leverage brackets
  are **not** an `exchangeInfo` field: they come from
  `GET /fapi/v1/leverageBracket`, which Binance classifies as signed
  **`USER_DATA`**. Reaching `exchangeInfo` therefore does **not** by itself
  satisfy the universe rule's inputs. Whether the rule keeps that condition —
  and if so how a credential is authorised, held and rotated — must be decided
  before collection starts. No credential is provisioned or requested here. See
  `docs/r2_source_archive_facts.md` for the first-party verification;
- ≥ 30 consecutive days of real elapsed time, which cannot be simulated,
  shortened, or substituted;
- the governed inputs that `docs/r2_source_archive_facts.md` marks as requiring
  resolution **before** live R2 begins, frozen, because each decides
  eligibility, universe construction, what is captured at all, or R2's own
  kill/deferral rule — and so cannot be selected after observing the 30-day run:
  1. the gen4 coverage thresholds;
  2. the exact definition of "core streams";
  3. the candidate-universe mechanical rule (volume field, tie-break, named date);
  4. whether that rule retains §18's leverage-bracket condition and how the
     signed `USER_DATA` data it needs is authorised;
  5. the Tier B storage/replay budget;
  6. **the reference-size / trade-through measurement method**, which must be
     identifiable from the data actually captured for *every* candidate symbol —
     the ~18 Tier A symbols outside BTCUSDT/ETHUSDT have no depth stream, so a
     hypothetical size grid is not reconstructable for them;
  7. **the `gen4-preflight` contract-schema interpretation** — `prospective_from`
     is required by today's parser but "absent by schema" per the roadmap;
- a **source-governance decision on the S3 listing origin**, which the archive
  namespace and absence claims partly rest on and which the repository's
  governed acquisition rule refuses by name (see *Method § 3* of the facts
  record). Recorded separately because it is a provenance question rather than a
  campaign parameter.

## Standing recorded by this document

- **R2 REMAINS OPEN.**
- **R2 IS BLOCKED AT THE R1 RECORDER-CORE COORDINATION BARRIER.**
- **R1-g and R1-h are the canonical adopted roadmap labels** for the shared
  recorder-core work. "R1-f2" is not a roadmap label and is not repository truth.
- **No live 30-day collection is authorised** by this document or by the pull
  request carrying it.
- **No second host is authorised or provisioned.**
- **No R3 work has started.**
- **No prospective boundary is created.** gen3's `prospective_from` stays `null`.
- **No alpha claim is created.**
- **No real-money authority is created.**
- **PR #76 remains untouched.**
