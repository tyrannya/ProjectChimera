# Futures dry-run validation — the protocol, frozen before it was evaluated

**Protocol hash:** `sha256:17a6a44d1fb5aca5eed16151c63a4b827566d664b1e1e78364357c32c6a44f4d`

That hash is a digest of `tools.futures_dry_run.PROTOCOL` — every scenario, every
invariant, the acceptance rule and the replay window, as data. It is asserted by
`tests/test_futures_dry_run.py`, stamped into every report the runner writes, and
rechecked by `python -m tools.futures_dry_run --verify`. **Weakening an invariant
after watching it fail moves the hash**, and a report whose embedded hash does not
match the protocol in the build reading it is rejected rather than believed.

That is the whole mechanism, and it is deliberately small: what stops acceptance
being redefined after the fact is not a promise, it is that the report and the
protocol are the same object and a checksum says so.

---

## 1. What this is, and what it is not

**It is** an operational validation of Futures Execution v1: proof that the
execution layer holds the invariants `futures_execution_v1.md` claims for it.

**It is not** an experiment, and it produces no research evidence:

- not evidence of trading alpha;
- not evidence about real exchange execution quality;
- not evidence about P4, P5 or any research checkpoint;
- not a basis for selecting a model, a feature, a threshold, a horizon or a
  target — and there is nothing in it to select on, because no descriptive metric
  has a criterion attached.

The simulated PnL it reports is a property of
`chimera.futures.venue.DeterministicFillModel`, not of a market. It appears in
the report because an operator wants to see it, and it is labelled every place it
appears.

## 2. What it runs on

Real observed prices from the committed pre-Styx OHLCV snapshot, restricted to
rows `[40981, 45802)` — **outer block 3** of the research fold plan,
`2024-09-04T21:00` to `2025-03-24T17:00`, 4,821 hourly candles.

That window is chosen for what it is *not*:

- it is not sealed. Styx (`2025-08-27T23:00:00+00:00`) is nowhere near it, and
  invariant **I15** checks that rather than trusting it.
- it is not `P4-HOLD`. Rows `[45802, 48211)` were retired unread and are not
  spent on an engineering test. The runner refuses to construct a window that
  reaches them (`_load_replay` raises), so the rule is code rather than care.
- it is already burned. Six research checkpoints have read outer block 3, so
  reading it again costs nothing that was not already spent.

Signals are **scripted**, not modelled: a fixed six-phase cycle
(`LONG, LONG, HOLD, SHORT, SHORT, HOLD`, 240 bars each) turned into probability
vectors and passed through `chimera.contracts.decide` at threshold 0.55, so the
real Pythia interface is exercised while nothing about the signal's *quality* is
claimed. A model here would invite the reading that the resulting PnL says
something about the model.

The path exercised end to end is the real one:

    scripted probabilities -> chimera.contracts.decide -> Signal
      -> FuturesExecutor.target_for -> plan_transition
      -> chimera.risk.RiskEngine.evaluate_entry        (Aegis)
      -> DryRunFuturesVenue.submit                     (Hermes -> venue)
      -> OrderEvent stream -> position, ledger, telemetry

## 3. The invariants

Acceptance is **every invariant holds**. There is no scoring, no partial credit,
and no threshold on any measured quantity — so there is nothing to tune.

| id | claim |
| --- | --- |
| I01 | no impossible order state transition is ever accepted |
| I02 | no order or fill ever reverses a position; every close reaches flat |
| I03 | a reversal is executed as two legs, never as one oversized order |
| I04 | a duplicate venue event changes no position, fee, or ledger entry |
| I05 | an Aegis veto makes execution impossible: the venue is never reached |
| I06 | a reduction succeeds while the risk engine is halted |
| I07 | reconciliation reports agreement when local and reported agree |
| I08 | reconciliation fails closed on disagreement: it never overwrites local state, and trading stops |
| I09 | emergency flatten reaches zero from LONG, from SHORT, from a partial fill and under a mismatch; is a recorded no-op when already flat; and is safe to repeat |
| I10 | restart recovery is correct and idempotent at every persistence boundary, and never assumes flat from an empty memory |
| I11 | funding signs are correct for all four (side, rate sign) combinations, paid and received are not netted, and a settlement is booked once |
| I12 | venue constraints fail closed: missing metadata, below minimum quantity and below minimum notional are refused rather than defaulted |
| I13 | the authenticated live-order route is unreachable, with and without the spot live-trading acknowledgement, and no credential is required |
| I14 | the required telemetry series are emitted by a full replay |
| I15 | the replay reads no row at or beyond P4-HOLD and no row at or beyond Styx |
| I16 | the whole-replay restatement: no impossible transition, no in-order reversal, both sides reached, partial fills occurred, restarts recovered exactly, the halt produced vetoes and blocked no exit, and the account ended flat |

**I05 deserves a note on how it is checked.** The scenario substitutes a venue
subclass whose `submit` raises if it is called at all, then drives an opening
order through a halted engine *and* through an engine whose exposure cap makes
the order impossible. Passing means the venue object was never reached — not that
an order came back rejected, which a venue could also produce.

**I13 likewise.** It is not enough that a live config is refused. The scenario
sets `ENABLE_LIVE_TRADING` to `chimera.safety`'s exact acknowledgement token and
checks that `FuturesExecutionConfig(dry_run=False)` *still* raises, because the
thing a reader might otherwise conclude is that the acknowledgement is the
missing piece. It is not; there is no live path to acknowledge.

## 4. Scripted interventions, declared in advance

A replay that never restarts, never halts and never reconciles reports zero for
three of its descriptive metrics and exercises none of the paths behind them. So
three interventions are scripted at fixed bar indices, and they are in the hashed
protocol rather than tuned into the run:

- **restarts at bars 1200 and 3600** — the executor is dropped, the persisted
  store re-opened, and `recover()` run against the venue's reported position. The
  position and ledger it comes back holding must equal what was there before.
- **a halt window at bars 2200–2300** — inside a SHORT phase, so a position
  exists. The engine is halted and the position flattened *while halted* (I06),
  after which every remaining bar in the window signals into a flat account and
  has its opening order refused by Aegis. That is what makes the veto count
  non-zero and non-trivial.
- **reconciliation every 1000 bars**, against a venue that should agree.

## 5. Descriptive metrics — measured, never optimised

The report records: signals seen and rejected; orders planned, submitted and
rejected; risk vetoes; fills and partial fills; mean slippage in bps; trading
fees; funding paid and funding received, separately; turnover; bars spent LONG,
SHORT and flat, and the LONG/SHORT balance; peak gross exposure; net exposure at
the end; reconciliation errors; emergency flattens; restart recoveries; and the
maximum drawdown of simulated net PnL.

Every one of them is descriptive. None has a threshold, none appears in the
acceptance rule, and the report says so in a field of its own
(`descriptive_metrics_are_not_acceptance_criteria`). If one of them looks bad,
the correct response is to understand why — not to change it, because changing it
would mean changing the execution layer to make a number move, which is the
definition of optimising against a metric that was never a criterion.

## 6. What happens when an invariant fails

Repair the **execution defect** and re-run the **same** protocol. Do not weaken
the invariant; the hash makes that visible rather than possible-and-quiet.

This has already happened once, and it is worth recording because it is the point
of the exercise. The first run of this protocol failed **I08**: a reconciliation
mismatch marked every *open order* for the symbol as
`RECONCILIATION_REQUIRED`, but by the time a mismatch is noticed the orders that
caused it are usually already terminal — so on a disputed position with no open
orders, nothing stopped the next signal from sizing itself against a position
nobody could vouch for. The fix was to record the dispute against the **symbol**
and persist it (`FuturesState.disputed`), cleared only by
`resolve_reconciliation` and its written reason. The protocol did not change.

## 7. What this does not cover

- **Sustained real-time paper operation.** This is a deterministic replay of
  historical candles inside one process. It is not days of wall-clock runtime
  against a live Binance USD-M feed, and this repository has no mechanism for
  that today. It is recorded here as a **later operational requirement**, not
  pretended away: before any real capital is involved, Futures Execution v1 needs
  a sustained paper-trading campaign against the live venue, with reconciliation
  running against a real exchange position that outlives the process.
- **Real venue behaviour.** Actual latency, actual rejection reasons, actual
  partial-fill patterns, actual funding rates. Everything here is scripted or
  modelled, and the model is adverse and deterministic by construction.
- **Anything above 1x or outside isolated margin**, which the config refuses.

## 8. Running it

    make futures-dry-run          # run the protocol and write the evidence
    make futures-dry-run-verify   # recheck the committed report against the protocol
    python -m tools.futures_dry_run --protocol   # print the protocol, run nothing

Evidence lands in [`artifacts/futures_dry_run_v1/`](../artifacts/futures_dry_run_v1/):
`dry_run.json` (the full report, including the embedded protocol) and
`STATUS.md` (the human-readable summary, with the "not evidence of alpha"
statement at the top rather than in a footnote).
