# PROPOSED: ProjectChimera futures-first roadmap v2

Status: **PROPOSED — NOT YET ADOPTED.**

This document is a proposed replacement for the **future** path after the current S3 remediation is closed. It does **not** supersede or alter the work already in flight in PR #89, and it must not be used to widen that PR.

The current active implementation branch remains responsible for finishing its own S3 / PR-10R remediation, review barrier, exact-head CI, and operator/runbook truthfulness. This proposal begins only after that work is dispositioned and merged or explicitly closed.

Nothing in this document is a research preregistration. Nothing here licenses a result, a live order, real money, a leverage increase, or a change to any frozen historical verdict. It is a roadmap and architecture decision proposal only.

---

## 0. Owner objective

ProjectChimera is to become a **futures-first autonomous crypto-trading research and execution system** with this priority order:

1. **Primary — USD-M perpetual futures.**
   - Autonomous LONG / SHORT directional trading is the principal trading capability.
   - Funding, margin, liquidation, fees, spread, slippage, position state, exchange state and recovery are first-class quantities.
   - The instrument modelled is the instrument executed.

2. **Second — scalping and spot.**
   - Scalping is a native microstructure programme, not a one-minute OHLCV strategy renamed as scalping.
   - Spot is a first-class tradable venue/path, not merely a hedge leg or price reference.
   - Futures scalping has priority inside the scalping lane; spot scalping and spot directional trading share the same data/state substrate where scientifically valid.

3. **Later — other crypto strategies and instruments.**
   - carry / funding / basis;
   - statistical arbitrage and cross-asset breadth;
   - market making;
   - options / volatility;
   - cross-exchange arbitrage;
   - DEX / on-chain strategies;
   - other assets and venues.

The architecture must not prevent those later lanes, but they may not dictate the design before the primary futures and second-priority scalping/spot lanes are scientifically and operationally coherent.

A second non-negotiable objective is a **causal multi-timeframe coherence system**: every decision can know how the market looks simultaneously at several horizons without leaking unfinished higher-timeframe information or future data.

---

## 1. What is preserved from the current programme

The replacement is not a reset. The following work remains valuable infrastructure and should be retained unless a later audit proves otherwise:

- the prospective recorder and append-only raw-data discipline;
- explicit missingness — missing data stays missing;
- deterministic normalisation and source identity;
- Aegis as the sole central risk authority;
- persisted risk state and fail-closed recovery;
- futures dry-run execution machinery;
- recorded-quote fills;
- carry accounting as a tested multi-leg accounting reference;
- the hash-chained decision log;
- replay-parity machinery;
- observability, alerts, reports and runbook infrastructure;
- Windows/Linux portability work;
- frozen historical evidence and every negative result;
- no further deciding use of burned historical outer blocks;
- P4-HOLD remains retired/unread;
- Styx remains sealed;
- no real money without a separate later authorisation contract.

The current S3 work is therefore a **foundation**, not discarded work.

What this proposal changes is the destination and scientific priority after S3: **carry is no longer the operational spine of the future scientific programme, and the old R1/R2/R3 PVC-1 menu is not automatically the next deciding campaign.** Carry remains a legitimate later non-directional lane and an accounting/integration stress case.

---

## 2. Mandatory bridge: finish the current S3 before changing direction

### B0 — Current S3 / PR-10R closure

PR #89 must finish exactly under its current contract. Do not add this roadmap's futures-directional, multi-timeframe, scalping or spot scope to that branch.

Required before this roadmap may become active:

1. Round 7 / the current review barrier finishes on the exact reviewed head.
2. Exact-head CI is green.
3. No unresolved BLOCKER remains in the scope PR #89 claims to close.
4. The PR description/runbook describes only operator actions that are actually executable.
5. Any head movement after review is re-reviewed on the changed delta.
6. The branch is merged only through the project's normal review/merge discipline.

### B1 — S3 lifecycle closure, if still required after PR #89

PR #89 itself records pre-existing lifecycle gaps that are intentionally out of its remit. Before any new prospective scientific campaign starts, those gaps must be either fixed or explicitly removed from the future runtime contract. At minimum inspect:

- persisted HALT -> supported operator recovery;
- `resume` / `resolve` crash-recordability;
- campaign source-identity correctness;
- replay policy for operator records and `seq`;
- partial-leg correction reachability;
- an identity check independent of the ledger fields it is checking;
- replay-safe trading-day / UTC-day semantics;
- all remaining disputes having an explicit safe resolution or terminal-halt policy.

This is still S3 closure. It is not alpha research.

Only after B0/B1 is complete should the owner adopt the remainder of this document.

---

## 3. Target architecture

The target data/decision path is:

```text
PUBLIC MARKET DATA
    |
    v
append-only recorder + provenance + reconciliation
    |
    +---------------------+
    |                     |
    v                     v
microstructure state      causal multi-timeframe state
(tick/trade/book)         (1m -> 5m -> 15m -> 1h -> 4h -> 1d)
    |                     |
    +----------+----------+
               v
        MARKET CONTEXT
   one as-of instant, explicit
   missingness and staleness
               |
        +------+------+----------------+
        |             |                |
        v             v                v
 futures directional  scalping         spot
 strategy family      strategy family  strategy family
        |             |                |
        +-------------+----------------+
                      v
              strategy intents
                      |
                      v
                    AEGIS
             sole risk authority
                      |
                      v
          venue-specific execution
        futures / spot / later venues
                      |
                      v
       accounting + position state
                      |
                      v
      decision log + input hashes
                      |
                      v
             deterministic replay
```

The system must separate:

- **data truth** from strategy interpretation;
- **market context** from model family;
- **strategy decision** from risk permission;
- **risk permission** from venue execution;
- **execution** from accounting/reporting;
- **engineering qualification** from scientific evidence;
- **prospective validation** from permission to trade real money.

---

## 4. Multi-Timeframe Coherence (MTC) — required core capability

The old P7 consensus-v1 result does **not** become a blanket prohibition on multi-timeframe reasoning. The new system must not revive that same majority-vote design under a new name.

### 4.1 Causal clocks

Initial supported context clocks:

- event / trade / book state for microstructure;
- 1s / 5s / 15s derived micro bars only where the source contract can support them;
- 1m;
- 5m;
- 15m;
- 1h;
- 4h;
- 1d.

Not every strategy must use every clock.

At decision instant `t`, a timeframe may contribute only information that was causally available by `t`.

Default rule for v1:

> A higher-timeframe candle contributes as a completed candle only when its close instant is <= the decision instant.

An unfinished H1/H4/D1 candle may not be silently represented as its final OHLC. If a future design wants partial-candle state, that is a **different explicitly named feature family** with its own as-of semantics.

### 4.2 MarketContext

Build one deterministic `MarketContext(as_of=t)` containing per clock, where available:

- return / direction state;
- trend-strength state;
- realised-volatility state;
- range / compression / expansion state;
- volume/activity state;
- futures mark/index/basis/funding state;
- microstructure state where applicable;
- staleness;
- completeness / missingness;
- source/input digest.

The market context is an engineering representation. It does not itself declare LONG or SHORT.

### 4.3 Coherence representation

The first researchable MTC representation should expose a vector rather than hard-code a single vote:

- directional agreement across clocks;
- disagreement structure;
- trend-strength alignment;
- volatility-regime compatibility;
- short-vs-long horizon momentum conflict;
- microstructure confirmation/conflict;
- staleness-weighted availability;
- derivatives-vs-spot disagreement where both are present.

A later model may learn which interactions matter, but any deciding use is frozen before the prospective result.

### 4.4 Required MTC invariants

Tests must prove:

1. changing a future higher-timeframe close cannot change an earlier context;
2. no incomplete higher-timeframe candle is passed as final without an explicit partial-candle feature family;
3. replaying the same event set creates byte-identical contexts;
4. missing input remains missing — no forward/backward fill;
5. stale clocks are marked stale rather than reused as fresh;
6. the exact source rows behind a decision are hash-identifiable;
7. no model or strategy can bypass Aegis because MTC agrees strongly.

---

## 5. Data generations

Do not mutate the existing gen3 contract into the new research surface. Preserve it as provenance.

### D0 — Keep gen3 intact

Gen3 remains the record of the current programme and current S3 machinery.

### D1 — Gen4 source preflight

Before freezing a gen4 contract, perform engineering-only source preflight for the data needed by the futures-first and scalping lanes.

Futures core candidates include, subject to current first-party documentation and observed availability:

- USD-M perpetual kline/trade data;
- mark/index/premium/funding information;
- best bid/ask;
- aggregate trade / trade flow;
- depth/book updates for the scalping lane;
- required exchange metadata for tick size, quantity step and contract rules.

Spot candidates include:

- spot kline/trade data;
- best bid/ask;
- depth/book data for the spot scalping lane.

Exact endpoints, stream names, event fields, sequence rules and reconciliation sources must be verified from current first-party documentation and a real engineering probe **before** the gen4 contract is frozen.

### D2 — Storage sizing rule

Do not guess a monthly microstructure storage number.

For every high-rate candidate stream, run a 24-hour engineering pilot before contract freeze and compute:

```text
compressed_GiB_per_day = compressed_bytes_24h / 2^30
projected_30d_GiB       = compressed_GiB_per_day * 30
minimum_free_capacity   = 1.5 * projected_30d_GiB
                         + compaction/replay scratch requirement
```

Provisioning must use the measured rate, not a README estimate. If a stream makes the approved storage/egress budget impossible, change sampling/retention **before** `prospective_from`, create a new contract identity, and repeat the pilot. Never thin a prospective stream retrospectively because it became expensive.

### D3 — Microstructure integrity

A book/depth contract must define and test:

- snapshot + diff sequencing where the venue requires it;
- duplicate policy;
- out-of-order policy;
- reconnect policy;
- sequence-gap detection;
- reconstruction validity;
- receipt time and exchange/event time;
- stale-book semantics;
- recovery after disconnect;
- exact behaviour when a gap cannot be repaired.

A corrupted or uncertain local book fails closed for scalping decisions.

---

## 6. Stage sequence after S3

The proposed future sequence is:

```text
B0/B1 current S3 closure
      ↓
F0 roadmap adoption / generation freeze
      ↓
F1 gen4 recorder + market-state foundation
      ↓
F2 general futures directional runtime
      ↓
F3 causal MTC implementation + research protocol
      ↓
F4 futures-first candidate freeze
      ↓
F5 Futures Validation Campaign (FVC-1)
      ↓
while FVC-1 accrues:
      SCL1 futures-scalping lane
      SPT1 standalone spot lane
      ↓
SCL2 / SPT2 prospective campaigns
      ↓
U1 unified multi-strategy orchestration
      ↓
U2 sustained paper operation + independent audit
      ↓
L0 separately authorised tiny-live consideration
      ↓
X1 later options / DEX / cross-exchange / other assets
```

No result from one arrow is assumed by the next.

---

## 7. F0 — adopt the futures-first mandate

Governance only.

Record explicitly:

- futures directional autonomy is priority 1;
- scalping and spot are priority 2;
- other families are lower priority;
- MTC is a required platform capability but not presumed alpha;
- carry remains supported but is no longer the primary campaign spine;
- the old carry-first R1/R2/R3 PVC-1 plan is superseded before it starts;
- no old historical result is reinterpreted;
- no prospective result exists merely because this roadmap is adopted.

F0 must not alter the active PR #89 chronology retrospectively.

---

## 8. F1 — gen4 recorder and MarketContext foundation

Goal: collect and represent the data needed for the instrument the project actually intends to trade.

Engineering deliverables:

- new gen4 contract(s), preserving gen3;
- futures and spot source preflight records;
- microstructure raw sink additions where selected;
- deterministic trade/book normalisation;
- causal multi-timeframe resampling;
- `MarketContext` with input hashes and staleness;
- replay of context construction;
- source-quality / sequence-gap / clock-skew observability.

Scientific tasks: **none**.

No model fitting based on the prospective gen4 validation region.

Pass: the exact data contract can be recorded and replayed deterministically under the frozen source semantics.

Fail: source/sequence/storage constraints make the design unreliable. Redesign the contract before activating its prospective boundary.

---

## 9. F2 — general futures directional runtime

The current runtime is carry-shaped. The futures-first target needs a first-class **single-leg directional perpetual position** without breaking the existing carry path.

Required capabilities:

- LONG / FLAT / SHORT perpetual target state;
- open, increase, reduce, close, reverse under explicit transition rules;
- mark-to-market from the venue's liquidation/reference semantics;
- realised and unrealised PnL;
- taker/maker fee representation as actually executed;
- funding paid/received;
- spread/slippage attribution without double charging;
- margin state and liquidation-distance calculation;
- no exposure increase without Aegis approval;
- restart reconstruction against persisted position/accounting state;
- deterministic decision records and replay;
- strategy-agnostic accounting interfaces where practical.

This stage uses synthetic rules/fixtures. It generates no alpha result.

The carry implementation remains as a two-leg stress test and later optional strategy, not as the abstraction every futures trade must pretend to be.

---

## 10. F3 — causal MTC implementation and falsification

This stage tests the **machinery**, then designs one bounded scientific question.

Engineering first:

- implement causal context at the chosen clocks;
- leakage tests at every resampling boundary;
- staleness/missingness tests;
- feature parity between live and replay;
- clock and timezone tests;
- synthetic examples where higher and lower timeframes agree, disagree and are missing.

Only after engineering is frozen may a research checkpoint ask whether MTC adds value.

The primary MTC scientific comparison should be incremental and controlled:

> On the same samples, same target, same execution assumptions and same model family, does adding the preregistered MTC information set improve a futures directional candidate over its single-clock baseline?

Do not answer this with an uncontrolled comparison of two entirely different systems.

The roadmap budget for the initial directional screen is intentionally finite:

- at most 2 model families: one linear and one tree baseline;
- at most 3 decision clocks/horizons selected before the deciding run;
- exactly 2 information arms per cell: single-clock baseline vs MTC-augmented;
- therefore at most 12 deciding cells before multiplicity correction;
- no neural model in the first futures-first screen.

The exact clocks, target and effect-size floor belong in the preregistration, not in this roadmap.

---

## 11. F4 — futures candidate freeze

Primary purpose: produce a **small frozen menu** for forward validation, not find the prettiest backtest.

Candidate selection principles:

- model the USD-M BTCUSDT perpetual directly;
- train only on research-visible/development data;
- burned historical blocks may be used for development but never described as fresh deciding evidence;
- the forward campaign starts strictly after model/rule/config hashes are frozen and independently reviewed;
- no online tuning in FVC-1;
- cost model is venue/instrument specific, not a flat global 20 bps assumption;
- effect-size floor, minimum activity, statistical method and multiplicity budget are all fixed before the first scored minute.

Initial campaign menu: at most **three** futures directional candidates total, preferably fewer.

At least one must be a simple externally intelligible baseline. A complex candidate cannot be promoted merely for beating CASH while failing to beat the appropriate simple baseline net of costs.

---

## 12. F5 — FVC-1, the primary Futures Validation Campaign

FVC-1 is the first priority scientific campaign under this roadmap.

Minimum design:

- USD-M BTCUSDT perpetual is the traded/simulated instrument;
- candidate hashes frozen before start;
- simulated fills use recorded market state and the venue-specific cost model;
- every decision is reconstructible from the decision log and exact input hashes;
- no parameter change after campaign start;
- no threshold rescue;
- no changing a losing LONG/SHORT rule into a different rule after reading interim results;
- monthly frozen evidence;
- dependent-data inference, not IID trade assumptions;
- multiplicity correction across the frozen menu.

### Duration policy

The default maximum campaign duration is **182 calendar days**.

Do not extend it because a promising result is "almost significant". If the preregistered minimum effective activity is not reached, the result is `INSUFFICIENT_ACTIVITY` under the protocol unless an extension rule was frozen before start.

The protocol must calculate, before start, the minimum effective sample size required for its effect-size floor at:

- family-wise alpha <= 0.05 after the declared candidate multiplicity;
- target power >= 0.80 under the preregistered alternative;
- dependence handled through block/bootstrap or another justified dependent-data method.

Trade count alone is not an IID sample-size claim.

### FVC-1 output

One verdict per frozen candidate, once.

A PASS authorises continued paper validation and later safety review. It does not authorise real money.

---

## 13. SCL1 — futures scalping lane (priority 2)

Start its engineering while FVC-1 is accruing so calendar time is not wasted.

Definition for this programme:

> Scalping means decisions whose information and execution depend materially on native microstructure, not merely one-minute candles.

Required data/engineering before a scientific scalping campaign:

- trade / aggregate-trade flow;
- reliable top-of-book and, if selected, locally reconstructed depth;
- spread and microprice;
- depth/order-flow imbalance;
- event/receipt latency measurement;
- sequence integrity;
- taker fills from the touch;
- maker execution only after a separate queue/latency model is validated.

Initial signal families may include, subject to preregistration:

- order-flow imbalance;
- signed trade imbalance;
- microprice / depth imbalance;
- spread/liquidity state;
- very-short-horizon momentum/reversal;
- futures mark/index/spot dislocation;
- liquidation/forced-order information only if a stable first-party source is verified.

Do not begin with dozens of indicators.

The first scalping campaign should prefer **taker-only execution** because maker fills require queue-position assumptions that can create fictitious alpha. A maker lane is a separate later checkpoint.

MTC can contribute higher-horizon context to a scalp, but higher timeframes are context, not a mandatory directional veto. The research question must determine whether the context improves the microstructure baseline.

---

## 14. SPT1 — standalone spot lane (priority 2)

Spot becomes a first-class tradable path.

Initial scope:

- BTCUSDT spot, LONG / FLAT only;
- own venue constraints, fees, quotes and ledger;
- same recorder/provenance/MarketContext framework;
- no assumption that a futures edge transfers to spot;
- no margin shorting in v1;
- optional spot scalping after the shared microstructure substrate is valid.

Scientific comparisons are venue-specific. A spot result and a futures result must not be collapsed merely because the symbol is BTCUSDT.

---

## 15. SCL2 / SPT2 — prospective validation

Scalping and spot receive separate prospective protocols and start instants.

They may overlap FVC-1 in wall-clock time if and only if:

- their rules were frozen before their own first scored data;
- each campaign has its own multiplicity budget;
- interim FVC-1 results are not used to tune them;
- shared data do not make one campaign's holdout a hidden development set for the other.

For scalping, require both a minimum calendar-regime span and an effective-activity threshold; high raw trade count alone is not enough because thousands of adjacent micro trades are strongly dependent.

---

## 16. U1 — unified multi-strategy orchestration

Do **not** rebuild the old mode router before strategies have earned it.

U1 opens only when at least two strategy families have prospectively valid non-negative evidence under their own protocols.

Potential strategy families at that point:

- futures directional;
- futures scalping;
- spot directional/scalping;
- carry/basis;
- later stat-arb.

The orchestrator owns allocation, not scientific rescue.

Required concepts:

- strategy-level risk budgets;
- aggregate BTC delta;
- gross/notional exposure;
- correlated strategy exposure;
- conflicting intents;
- portfolio drawdown and daily-loss state;
- order-rate sharing;
- funding/margin pressure;
- one Aegis decision surface above all strategies.

A losing strategy may not be hidden by portfolio aggregation and called validated.

---

## 17. U2 — sustained autonomous paper operation

Before any authenticated trading route exists, the selected system runs continuously in paper/dry-run form.

Minimum expectations:

- several months of continuous operation after candidate selection;
- scheduled restarts;
- recorder outages;
- stale-book drills;
- state corruption drills;
- exchange metadata changes simulated;
- kill-switch drills;
- reconciliation drills;
- zero unexplained replay divergence;
- decision log and accounting reconstructible independently;
- runbook executed by someone other than the authoring agent.

The autonomous property is operational here: the system can make, risk-check, simulate, record, recover and explain its decisions without an operator manually selecting trades.

---

## 18. L0 — separately authorised live route, only after evidence

The end goal includes the ability to trade, but the authenticated route is deliberately late.

L0 may be designed only after:

1. a strategy has passed its prospective protocol;
2. sustained paper operation is coherent;
3. replay/accounting/safety audits pass;
4. an independent reviewer reconstructs the result;
5. the owner explicitly authorises a new live-trading contract.

Initial live architecture must include at minimum:

- dedicated exchange sub-account;
- API key with trading permission only and withdrawals disabled;
- IP allow-list where supported;
- credentials outside repository and ordinary state directories;
- hard capital cap enforced by both account balance and Aegis;
- leverage capped at 1x initially unless a later independent risk mandate changes it;
- exchange-state reconciliation;
- duplicate-order prevention;
- cancel/flatten procedure;
- local kill switch plus exchange-side emergency procedure;
- parallel dry-run for live-vs-shadow parity;
- explicit operator enable gate;
- no automatic capital top-up;
- any capital increase is a new reviewed decision.

A live executor is not alpha and does not retroactively validate a strategy.

---

## 19. X1 — later expansion

Only after the futures + scalping/spot core is coherent may the project open lower-priority mandates such as:

- delta-neutral carry / funding / cash-and-carry;
- cross-exchange arbitrage;
- cross-sectional/statistical arbitrage;
- market making;
- options/volatility;
- DEX/on-chain execution and MEV-aware research;
- additional perpetual symbols;
- portfolio-level cross-asset breadth.

Each is a new research mandate with its own data/source/cost model. None inherits validation from BTCUSDT futures merely because the infrastructure is shared.

---

## 20. Research-budget discipline

The replacement roadmap broadens capability but must not reopen unlimited historical search.

Standing rules:

- deciding reuse of the burned outer blocks remains 0;
- a new feature family is not a licence for unlimited model search;
- first futures MTC screen: <= 12 deciding cells as defined in F3;
- FVC-1: <= 3 frozen candidates;
- first scalping screen: bounded candidate menu fixed in its preregistration;
- first spot screen: bounded candidate menu fixed in its preregistration;
- no neural model until a simpler baseline has shown a prospectively validated nonlinear residual worth modelling;
- no online adaptive retraining until a static candidate has first been evaluated prospectively;
- negative results remain visible.

A future adaptive/retraining system, if desired, is its own research programme: retraining schedule, decay/expiry, drift detector and promotion rule are frozen before forward evaluation.

---

## 21. Verification discipline for every major stage

Every major engineering PR should prove the claims it makes with some combination of:

- two-sided synthetic controls;
- mutation witnesses for load-bearing guards;
- property/invariant tests;
- deterministic replay;
- independent hand arithmetic for accounting;
- source-provenance checks;
- cross-platform CI;
- exact-head CI;
- frozen-manifest verification;
- no-live-path / credential scans until L0 is explicitly opened.

Mutation tools that rewrite source in place must never run concurrently with the ordinary suite in the same worktree.

For major scientific checkpoints:

```text
preregistration commit + push
        ↓
independent protocol review
        ↓
future start instant
        ↓
result generation
        ↓
closure
        ↓
independent reconstruction/audit
```

No amend/squash/rebase/force-push may erase the chronology that establishes that order.

---

## 22. Calendar estimate

These are planning estimates, not scientific deadlines. Evidence time cannot be compressed by adding agents.

Assuming the current S3 remediation closes around September 2026:

| milestone | best | expected | delayed |
| --- | --- | --- | --- |
| current S3 / PR-10R barrier closed | ~2026-09-10 | ~2026-09-15 | ~2026-09-22 |
| roadmap v2 adoption + F0 | + few days | + 1 week | + 2 weeks |
| gen4 + MTC + general futures runtime ready enough for protocol review | ~6 weeks after adoption | ~9 weeks | ~13 weeks |
| **FVC-1 start** | ~2026-10-22 | ~2026-11-17 | ~2026-12-22 |
| **FVC-1 182-day end** | ~2027-04-22 | ~2027-05-18 | ~2027-06-22 |

Scalping and spot engineering should begin after FVC-1 is frozen/started and proceed in parallel with evidence accrual. Their prospective campaigns therefore need not wait for the end of FVC-1, but must start only on data that was still future when their own protocols were frozen.

Earliest credible tiny-live consideration is therefore measured in **many months, not weeks**. A rough planning range is about **12–15 months from roadmap adoption** if at least one strategy passes and the safety path remains clean. A negative campaign correctly pushes the decision toward no deployment rather than toward faster tuning.

---

## 23. Suggested engineering package sequence

Exact PR numbers should be assigned from repository state after the current S3 branch lands. Do not reuse numbers from the old master plan blindly.

Suggested packages:

1. governance: adopt futures-first v2 after S3 closure;
2. gen4 source preflight + data-rate/storage report;
3. gen4 recorder contract + raw event types;
4. microstructure stream/reconciliation implementation;
5. causal timeframe fabric + MarketContext;
6. MTC invariants and replay parity;
7. general single-leg futures accounting/state machine;
8. futures directional runner integration;
9. futures cost/funding/margin reporting;
10. FVC-1 scientific protocol;
11. FVC-1 operational soak and start gate;
12. futures scalping data/replay engine;
13. futures scalping candidate protocol;
14. standalone spot execution/accounting;
15. spot candidate protocol;
16. unified portfolio/Aegis orchestration only after validation eligibility;
17. sustained autonomous-paper certification;
18. later live-executor design, only if L0 prerequisites are met.

Large packages stay reviewable. One authoring agent must not self-certify the merge of a load-bearing scientific or safety PR.

---

## 24. Explicitly not authorised by this proposal

Until separately authorised, this roadmap does **not** permit:

- changing PR #89 scope;
- starting a campaign before the current S3 lifecycle is coherent;
- using real money;
- creating exchange credentials;
- enabling authenticated order routing;
- increasing leverage above 1x;
- reading P4-HOLD;
- opening Styx;
- rewriting or hiding any historical negative result;
- using the burned outer blocks as fresh deciding evidence;
- claiming MTC works because P7 was negative or because the architecture looks richer;
- calling one-minute OHLCV trading "scalping" without native microstructure;
- assuming spot and perpetual alpha transfer to each other;
- maker-fill backtests without a validated queue/latency model;
- online tuning during a prospective campaign;
- a new mode router before at least two strategy families earn eligibility.

---

## 25. Adoption checklist

Before changing this document from PROPOSED to ADOPTED:

- [ ] PR #89 finished and independently reviewed on its exact head.
- [ ] Current S3 lifecycle gaps dispositioned under B1.
- [ ] `main` SHA re-established from Git, not from this document.
- [ ] owner explicitly accepts the priority hierarchy: futures first; scalping + spot second; everything else later.
- [ ] owner explicitly accepts MTC as a required capability but **not** a presumed source of alpha.
- [ ] old carry-first PVC-1 menu marked superseded before it starts; no result implied.
- [ ] existing gen3/history preserved rather than rewritten.
- [ ] front-door documents updated to point to the adopted v2 roadmap.
- [ ] implementation master plan regenerated from the live repository after adoption rather than mechanically editing stale PR numbers.
- [ ] independent Fable 5.1 strategic review scheduled before the first new scientific protocol opens, with the reviewer allowed to reject this roadmap.

---

## 26. One-line roadmap

> Finish the current S3 truth/recovery/accounting barrier without widening it -> adopt a futures-first mandate -> record a new causally clean gen4 futures/spot/microstructure generation -> build deterministic multi-timeframe market context -> validate autonomous directional USD-M perpetual trading first -> build and prospectively validate native-microstructure scalping and standalone spot second -> unify only strategies that earn eligibility under Aegis -> sustain autonomous paper operation -> independently audit -> consider a separately authorised tiny live route -> only then expand to other crypto markets and strategy families.
