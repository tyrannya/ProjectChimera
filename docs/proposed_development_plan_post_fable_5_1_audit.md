# PROPOSED - NOT YET ADOPTED: development plan after the Fable 5.1 strategic audit

Status: **PROPOSED - NOT YET ADOPTED.** This document is a proposal produced by
the independent audit recorded in
[`fable_5_1_full_project_strategic_audit.md`](fable_5_1_full_project_strategic_audit.md).
It does not replace `docs/current_development_plan.md` or
`docs/research_roadmap.md`, which remain authoritative until the owner adopts
this plan in a separate, explicit decision. Nothing here is a preregistration,
nothing here licenses a result, and nothing here changes any frozen verdict.

Written 2026-09-03 against `main` at `177d4b60c7e137730ce88241b481941b07b4cd30`
and draft PR #67 at `36cdae48877b1d5fa88b2664c127b5307a917751`.

---

## 0. The decision this plan asks the owner to make

Adopt the following, as one decision:

1. **No further deciding use of the four outer blocks** (2023-03-04 to
   2025-05-19). They have been read by ten checkpoints. They remain
   descriptive context and nothing more.
2. **P14 is declined, not opened.** PR #67 is closed with the audit linked.
   The branch stays as the record of a design considered and declined. The
   source-preflight artifact and the sign-convention proof may be
   cherry-picked into `main` with a STATUS note saying the design was
   declined at audit; no evaluator is written.
3. **P8 is withdrawn as moot.** Its opening condition (two eligible modes)
   cannot be met without refitting screened-out clocks, which its own rules
   forbid.
4. **The instrument changes.** The Binance USD-M BTCUSDT perpetual is the
   modelled and executed instrument. Binance spot BTCUSDT is the price control
   and the long leg of the carry position.
5. **The next scientific checkpoint is prospective, not historical.** It is
   the Prospective Validation Campaign described in section 3 (PVC-1), run on
   data that has not yet occurred, under a protocol frozen and independently
   reviewed before the first scored day.
6. **The demo architecture collapses to the Minimum Viable Chimera** in
   section 4. The multi-clock specialist / consensus / router architecture is
   retired. Freqtrade, the four strategies, the inference service, the model
   registry, MLflow and Ray Tune leave the demo path.
7. **The research budget is finite** (section 6) and ends in a decision.

Everything else in the current plan's standing constraints that is not
contradicted above is carried forward unchanged: no real money; Aegis is the
sole risk authority; `P4-HOLD` stays retired and unread; Styx stays sealed and
carries its hindsight-era disclosure; negative results stay visible;
preregistration before result; one research question at a time; unknown
operational state fails closed; the "no live route" two-claims distinction.

---

## 1. Why the direction changes

The audit's evidence, in one paragraph. Every information-set and clock
checkpoint decided on "net return after costs improved in at least 3 of 4
folds", where each fold's net return rests on between 4 and 280 realised
trades. The same OHLCV14 control cell, unchanged except for which rows were
eligible, moves from `+0.106` to `+0.429` in fold 3 (XGBoost, P2b versus P5)
and from `-0.026` to `-0.542` in fold 2 (LightGBM, P2b versus P4). The gate
therefore measures threshold selection and a handful of trades, not the
information set; it cannot resolve a plausible edge and it cannot refute one.
The flat 20 bps per-trade cost is about three standard deviations of a
one-minute BTC return and about one standard deviation of a six-minute
return, so the fast clocks and P14's economic stage were closed by arithmetic
before any fit. Every economic number since v4 prices a SHORT leg that spot
cannot hold. And the four periods are spent: the only non-adaptive evidence
left is the future, which nobody is recording. A plan that keeps reading the
same periods with the same instrument cannot reach a defensible demo.

---

## 2. Stages, backwards from the endpoint

Endpoint: a frozen candidate set operating prospectively in dry-run against
live market data for six months, with per-instrument costs, persistent state,
failure handling, replay parity, operator visibility and no retrospective
parameter changes, followed by an independent audit and a decision.

### S0 - Decision and freeze (week 1)

- Owner adopts or rejects this plan. If adopted: close PR #67; mark P8
  withdrawn in the research-state table (a new state value, `withdrawn`, so
  the verifier keeps working); record the research budget of section 6 in the
  standing constraints.
- Engineering complexity: trivial. Information value: none. Failure: none.

### S1 - Prospective recorder (weeks 1-3, then continuous)

- Question: can the project collect causally clean forward data for the
  instruments it will trade?
- Build: a recorder for Binance USD-M BTCUSDT (1m klines, mark-price klines,
  index-price klines, funding settlements, best bid/ask) and Binance spot
  BTCUSDT (1m klines), writing append-only, checksummed daily files under a
  `gen3` research contract with an explicit "prospective from" instant. The
  recorder stores what the exchange publishes, never interpolates, and
  records its own gaps. Reconciliation against `data.binance.vision` archives
  when they appear, using the existing acquisition and verification tools.
- Pass: 30 consecutive days at >= 99.5% minute coverage per stream, with the
  archive reconciliation agreeing on every overlapping minute within
  tolerance.
- Fail: coverage or reconciliation cannot be sustained. Failure means the
  project cannot do prospective science; it stops until fixed.
- Complexity: small. Information value: high (it enables every later stage).

### S2 - Candidate freeze and PVC-1 preregistration (weeks 2-5)

- Question: which frozen rules are evaluated forward, and by what rule are
  they judged?
- Menu (at most three rules; the multiplicity is declared and paid for):
  - **R1, operational: always-on delta-neutral carry.** LONG spot BTCUSDT
    against SHORT USD-M BTCUSDT perpetual, equal quantity, 1x, cross or
    isolated margin as preregistered, rebalance rule fixed, funding-rate halt
    fixed. Accounting from the P13 Decimal engine, with mark price recorded
    live so the liquidation quantity is always defined. Costs: spot and perp
    maker/taker as executed, plus a spread/slippage model from the recorded
    best bid/ask.
  - **R2, shadow (optional): a fixed-coefficient logistic rule on OHLCV14** at
    a multi-hour horizon on the perpetual, with the coefficients and the
    abstention threshold frozen in the protocol (fitted once on research-
    visible data before the campaign, then never refitted). Scored as a
    signal series; positions simulated at zero size.
  - **R3, shadow (optional): a daily time-series momentum rule** with fixed
    lookback, on the perpetual, as the simplest externally anchored
    directional comparator.
- Per-rule pass/fail written before the first scored day, each with: an
  effect-size floor (net of costs), a minimum number of funding settlements
  (R1) or realised trades (R2, R3) that makes the floor detectable, monthly
  blocks, a dependent-data test (block bootstrap on trade-level or
  settlement-level net returns) reported beside the block count, and a
  multiplicity adjustment across the menu.
- Independent review of the protocol, with a minimum elapsed time (at least
  one week) between commit and the first scored day.
- Pass: the reviewer approves. Fail: the reviewer rejects; redesign; no data
  is scored meanwhile. Complexity: small. Information value: high.

### S3 - Minimum Viable Chimera build (weeks 3-9)

- Question: can the platform run the frozen rules against live data with
  simulated fills, and prove afterwards what it did?
- Build, in this order:
  1. `RiskEngine` persists its full state (peak equity, day-start equity, loss
     streak, cooldown, order-rate window), with two-sided synthetic tests;
     feed-staleness and funding-rate halts added.
  2. A live-data simulated venue in `chimera.futures`: fills at the recorded
     best bid/ask plus a slippage model, per-instrument maker/taker fees,
     funding from observed settlements, no credentials, no live route (the
     package's AST guard keeps asserting that).
  3. A two-leg position (spot leg simulated in the same ledger) for R1.
  4. A demo runner derived from `tools/paper_run.py` with modes and consensus
     removed: recorder stream -> frozen rules -> Aegis -> executor -> store,
     plus an append-only decision log carrying a hash of every input each
     decision read.
  5. A replay-parity harness: replaying the recorder's files through the same
     code path must reproduce every decision and every ledger entry byte for
     byte.
  6. Alerts (halt, reconciliation, feed staleness, restart), one dashboard, a
     runbook, a daily report script, the kill-switch file.
  7. Removal of Freqtrade, its configs, its Docker image and the four
     strategies from the demo branch; archival of `chimera/modes.py`,
     `chimera/consensus.py`, the inference service and the registry; a lock
     file; the `read_text()` encoding, numpy-2.2 dtype and path-separator
     portability fixes.
- Pass: replay parity holds; all 16 invariants of `futures_dry_run_v1` hold
  on live data; the two-leg accounting is hand-traced for LONG, SHORT and the
  hedged position with units, notional, fees, funding and PnL checked.
- Fail: parity failures unresolved after one repair cycle. Failure means the
  platform cannot be trusted to report what it did; it stops until repaired.
- Complexity: medium (the largest engineering item in the plan; capped at
  eight person-weeks, after which the plan is simplified further rather than
  extended). Information value: medium.

### S4 - Soak and parity (weeks 9-11)

- 14 days of continuous operation with scheduled restarts, simulated feed
  outages and reconciliation drills.
- Pass: >= 99% uptime, zero unexplained divergence between the decision log
  and replay, every halt and recovery behaving as specified.
- Fail: repairs, then one repeat. A second failure stops the plan.

### S5 - PVC-1 sustained run (months 3-9)

- The scientific checkpoint. The frozen menu runs for six months. Monthly
  frozen reports are committed as evidence (`evidence_class: prospective`).
  No parameter, cost, rule or menu change. Operational repairs that change no
  decision are allowed and disclosed; a repair that would change a past
  decision invalidates the affected block and is recorded as such.
- Minimum evidence for a verdict: R1 at least 500 funding settlements and at
  least 5 complete monthly blocks; R2/R3 at least the trade count fixed in S2.
- Information value: decisive, in both directions.

### S6 - Closure and independent audit (month 10)

- Per-rule verdicts under the S2 rules. Either a narrow promotion case for a
  separately authorised very small live consideration (a new contract with a
  hard capital cap, written and reviewed before any key is created), or a
  recorded "no deployable alpha under the current mandate".
- An independent audit reconstructs the verdicts from the decision log and
  the recorder files.

### PVC-2 (optional, months 10-16)

Only if S6 leaves a verdict ambiguous by its own preregistered rules. At most
two rules re-frozen; no rule may be tuned on PVC-1's data. After PVC-2 the
project decides regardless.

---

## 3. What PVC-1 is and is not

It is a preregistered forward test of a small frozen menu, with declared
multiplicity, on the instrument that can hold the positions. It is the first
non-adaptive evidence the project will produce. It is not a backtest, not a
smoke, not a claim of alpha, and not permission for real money. A positive
verdict earns a separately written live-authorisation contract, nothing more.

Why carry is the operational spine: it is the only mechanism in the
programme's history with an externally known positive expectation, an
observable payoff (funding settles every eight hours), low turnover, and an
accounting engine already built and hand-traced. Its historical screen was
lost to source rigidity; its prospective screen is cheap and decisive.

Why the directional shadow rules are optional and frozen: the programme's
directional leads are post-selection observations on burned blocks. The only
honest treatment of such a lead is a frozen forward test with the prior that
it is noise, and that is what the shadow rules are. They are limited to two
so that the campaign's multiplicity stays small, and to the logistic family
because it is the only one whose control cells kept a consistent sign across
universes and the only one whose frozen coefficients remove the BLAS
reproducibility problem.

---

## 4. Minimum Viable Chimera - DEMO

```
DATA:        gen3 prospective recorder (perp: 1m klines, mark, index, funding,
             best bid/ask; spot: 1m klines); archive reconciliation; existing
             acquisition and verification tools.
SIGNAL:      R1 carry (operational); R2/R3 frozen directional shadows (optional).
MODEL:       none fitted online; frozen coefficients only.
RISK:        Aegis with persisted full state; 1x cap; daily-loss, drawdown,
             funding-rate, feed-staleness and reconciliation halts; kill switch;
             reductions always available.
EXECUTION:   chimera.futures executor + live-data simulated venue; two-leg
             position; per-instrument fees; no credentials; no live route.
STATE:       atomic store; append-only decision log with input hashes; daily
             snapshots; replay-parity harness.
OBSERVABILITY: existing Prometheus series; four alert rules; one dashboard;
             Telegram optional.
OPERATOR:    runbook; daily report; monthly frozen report; kill switch.
EXCLUDED:    MTST and training; inference service; registry; Freqtrade and
             strategies; modes; consensus; MLflow; Ray Tune; Grafana breadth;
             all checkpoint-specific research modules (frozen history only).
```

---

## 5. Research methodology changes (apply to every future design)

1. Every preregistration states an effect-size floor, the minimum realised
   trade or settlement count that makes it detectable, and the gate's
   false-positive rate and power under a stated null and alternative.
2. Per-instrument cost models (maker/taker, spread, funding); the flat 20 bps
   is retired; the horizon is justified against the cost model.
3. A dependent-data statistical test beside any fold or block count; a
   multiplicity budget per campaign; deflated or bootstrap-adjusted
   risk-adjusted metrics where a Sharpe-like number is reported.
4. An independent review before a checkpoint opens, with a minimum elapsed
   time between preregistration and the first result.
5. Prospective evidence outranks historical evidence for promotion decisions;
   Styx and `P4-HOLD` are never described as prospective.
6. The instrument modelled is the instrument executed.

---

## 6. Finite research budget and stopping policy

| item | budget |
| --- | --- |
| deciding checkpoints on the four outer blocks | 0 |
| new historical checkpoints on genuinely new data (for example cross-asset breadth) | at most 1, only after S5 is running, under section 5 |
| prospective campaigns | at most 2 (PVC-1, PVC-2), six months each, at most three rules each |
| engineering before S5 | capped at eight person-weeks; exceeding it simplifies the plan rather than extending it |
| end of budget | after PVC-2 at the latest (about 14 months from adoption): freeze a candidate for a separately authorised very small live consideration, or record "no deployable alpha under the current mandate" and close or re-mandate |

Kill and pivot criteria are in the audit, section 24, and are adopted
unchanged.

---

## 7. What this plan deliberately does not do

- It does not open P14, P8, P13's economics, `P4-HOLD` or Styx.
- It does not promote any historical lead to a live or paper position.
- It does not enable any authenticated route or increase leverage above 1x.
- It does not rewrite any frozen evidence or any closed checkpoint's verdict.
- It does not re-specify P13 against the data that arrived; the carry rule in
  PVC-1 is a new, prospective design with its own preregistration.
- It does not promise that any rule will pay. The expected outcome, on the
  evidence to date, is that the directional shadows are indistinguishable
  from zero and the carry rule is regime-dependent and marginal. Establishing
  that prospectively is the point.

---

## 8. Adoption checklist

- [ ] Owner decision recorded in `docs/current_development_plan.md` (adopt,
      adopt with changes, or reject), with the date.
- [ ] PR #67 closed with the audit linked; branch retained.
- [ ] P8 marked withdrawn; research-state verifier extended with the state.
- [ ] Standing constraints updated with the finite budget and the instrument
      change.
- [ ] S1 recorder branch opened as the first engineering task.
- [ ] S2 protocol drafted for independent review.
