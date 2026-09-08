# PROPOSED: ProjectChimera futures-first roadmap v2

Status: **PROPOSED — NOT YET ADOPTED.**

This document is a proposed replacement for the **future** path after the current S3 remediation is closed. It does **not** supersede or alter the work already in flight in PR #89, and it must not be used to widen that PR.

The current active implementation branch remains responsible for finishing its own S3 / PR-10R remediation, review barrier, exact-head CI, and operator/runbook truthfulness. This proposal begins only after that work is dispositioned and merged or explicitly closed.

Nothing in this document is a research preregistration. Nothing here licenses a result, a live order, real money, a leverage increase, or a change to any frozen historical verdict. It is a roadmap and architecture decision proposal only.

> **Presence on `main` does not mean adoption.** This file may be merged into `main` so collaborators and independent reviewers can read it from the default branch. Until the adoption checklist in section 25 is completed by a separate governance decision, agents may read, critique, expand, or independently audit this proposal, but they MUST NOT treat it as the active roadmap or begin implementing its post-S3 stages merely because the file exists on `main`.

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
               |
               v
       MarketContext(as_of=t)
               |
       +-------+-------+
       |               |
       v               v
 futures directional   scalping / spot
 candidates            candidates
       |               |
       +-------+-------+
               |
               v
      Aegis risk authority
               |
               v
    simulated execution first
               |
               v
  persistent accounting + log
               |
               v
 deterministic replay / audit
```

The essential architectural rule is that **MarketContext is a causally valid representation of what was knowable at a single `as_of` instant**, not a bag of resampled candles.

---

## 4. Multi-Timeframe Coherence (MTC)

### 4.1 Purpose

MTC is a required platform capability, not a presumed source of alpha.

It exists to let a strategy ask:

- what is the microstructure saying now?
- what is the 1m/5m/15m local state?
- what is the 1h/4h/day regime?
- do those views reinforce, contradict or remain silent relative to each other?
- how stale is each view?
- which views are missing?

### 4.2 Causal rule

At decision instant `t`, every timeframe feature must be computed only from information available by `t`.

For a closed higher-timeframe bar:

```text
bar is eligible iff its close instant <= t
```

An unfinished 4h or 1d candle may not contribute its eventual final high/low/close as though those values already existed.

If partial-bar state is ever used, it must be specified as a **different information family** with explicitly causal fields known at `t`; it may not borrow the closed-bar semantics.

### 4.3 Alignment

Each view carries at minimum:

```text
source_time
available_time
as_of
age
complete / partial
missing
```

No forward fill that hides missingness.

No nearest-future match.

No retrospective alignment after the fact.

### 4.4 State categories

The first MTC generation should represent at least:

- directional/trend state;
- realised-volatility state;
- range/compression/expansion state;
- volume/activity state;
- spot/perpetual basis or divergence state;
- funding state;
- price relative to recent causal structure;
- microstructure pressure;
- explicit disagreement/coherence across clocks;
- staleness and missingness.

### 4.5 Coherence is not majority voting

Do not restore P7 consensus-v1 under a new name.

A higher timeframe is not automatically a veto and a lower timeframe is not automatically noise. The role depends on the strategy family and horizon.

For example:

- a multi-hour futures strategy may treat 4h/1d as regime context and 5m/15m as timing;
- a scalper may treat order flow and top-of-book as the primary state while 15m/1h is only a context/risk modifier.

The relationship must be measured prospectively rather than hard-coded from trading folklore.

### 4.6 MTC scientific question

The first MTC experiment is not "does a complicated architecture make money?"

It is:

> Does a causally aligned multi-timeframe context add prospective information or decision value relative to the same candidate family without that context, under the same instrument, horizon, costs and risk rules?

This makes the incremental contribution identifiable.

---

## 5. Future data generation: gen4

The current prospective recorder is a foundation, but the new destination requires a generation whose contract explicitly supports directional futures and native microstructure.

### 5.1 Required core market families

At minimum investigate and freeze first-party sources for:

- USD-M BTCUSDT perpetual klines;
- mark price / index price / funding semantics;
- perpetual best bid/ask;
- public trades / aggregate trades where scientifically suitable;
- native order-book depth if a defensible continuous source is available;
- liquidation / force-order information if a source-valid contract can be maintained;
- open interest if publication semantics and archive/reconciliation are adequate;
- spot BTCUSDT klines;
- spot best bid/ask;
- spot public trades / aggregate trades where suitable.

Every source must have a source preflight before the contract freezes.

### 5.2 Depth/order-book rule

If native depth is used, the system must define and test:

- snapshot + delta bootstrap;
- sequence/update-id semantics;
- gap detection;
- reconnect recovery;
- duplicate handling;
- out-of-order handling;
- local-book reconciliation;
- clock/time basis;
- storage footprint;
- deterministic replay.

A broken local book is worse than no book. Fail closed rather than pretend continuity.

### 5.3 Data-rate/storage report

Before adopting high-rate streams, run an engineering preflight that measures:

- events/s;
- compressed/raw bytes/s;
- CPU;
- memory;
- disk/day;
- replay throughput;
- expected 30/90/182-day storage;
- recovery time after restart.

The roadmap must be operationally affordable before the source contract is frozen.

---

## 6. General single-leg futures runtime

The existing demo path is carry-shaped. A futures-first system needs a general directional state machine that does not require a spot hedge leg.

### 6.1 Required states

At minimum represent safely:

```text
FLAT
LONG
SHORT
REDUCING
HALTED
DISPUTED
RECOVERING
```

A position may not be inferred from the last strategy signal; execution/accounting state is authoritative.

### 6.2 Accounting

The single-leg futures ledger must make explicit:

- signed quantity;
- average entry;
- realised PnL;
- unrealised PnL;
- fees;
- funding paid/received;
- margin allocation;
- liquidation reference;
- slippage evidence;
- free cash / total equity;
- order/fill provenance.

No term may be charged twice merely because it appears both in a fill price and in a metric.

### 6.3 Crash consistency

State/log/ledger ordering must be defined before a campaign starts.

Every restart must either:

- reconstruct one coherent state;
- recover a specifically authorised crash window;
- or refuse.

Never silently manufacture a fresh ledger because a persistent file disappeared.

### 6.4 No real exchange route yet

This is still simulated execution over recorded public data.

No credentials.

No authenticated order route.

No leverage above 1x.

---

## 7. Futures directional candidate programme

This is the primary scientific lane.

### 7.1 Candidate hierarchy

Start with intentionally limited families rather than a model zoo.

Tier A — simple baselines:

- constant/no-trade controls;
- causal momentum/trend;
- causal reversal;
- volatility/regime-conditioned variants.

Tier B — simple learned models:

- regularised linear/logistic family;
- one tree-family candidate (for example LightGBM/XGBoost selected in advance, not both searched indefinitely).

Tier C — later only if earned:

- sequence/deep models;
- representation learning;
- transformers;
- specialised architectures.

Deep learning is not a prerequisite for a useful autonomous trading system.

### 7.2 First bounded MTC screen

For the first deciding MTC screen:

- maximum 2 model families: one linear, one tree;
- maximum 3 preselected horizons/clocks;
- exactly two arms per model/horizon: baseline vs MTC;
- maximum **12 deciding cells**;
- no neural models;
- no post-hoc horizon addition;
- no post-result feature-family rescue.

The purpose is to answer whether the MTC information family deserves promotion, not to optimise the whole strategy universe.

### 7.3 Instrument

Primary deciding instrument:

```text
Binance USD-M BTCUSDT perpetual
```

A strategy trained/evaluated for the perpetual is executed against the perpetual simulator.

Spot may be an input/control where preregistered, but a positive perpetual result is not silently re-labelled as spot alpha and vice versa.

---

## 8. Cost and execution realism

Directional futures and especially scalping cannot reuse one generic cost assumption blindly.

### 8.1 Common terms

Every candidate must account for the terms applicable to it:

- taker/maker fee;
- bid/ask crossing;
- configured or measured slippage;
- funding;
- latency assumption;
- partial fills;
- quantity/price constraints;
- market impact where relevant;
- liquidation/margin rules.

### 8.2 Scalping constraint

A strategy is not a serious scalping candidate if its expected gross edge is of the same order as one spread crossing plus fees and it has no credible maker/queue model.

### 8.3 Taker-first rule

The first scalping campaign should prefer a conservative taker model unless a maker simulation has independently validated:

- queue position;
- cancellation/replace latency;
- fill probability;
- adverse selection;
- partial fills;
- fees/rebates.

Do not manufacture profitable maker fills from touching the best quote.

---

## 9. Primary prospective campaign: FVC-1

FVC-1 = **Futures Validation Campaign 1**.

This replaces the assumption that the old carry-first PVC-1 menu is automatically the next deciding campaign.

### 9.1 Freeze before start

Before the first scored instant, freeze in Git:

- source contract;
- eligible data families;
- candidate count;
- candidate parameters;
- horizon/holding rules;
- MTC information set;
- execution/cost model;
- risk limits;
- scoring metrics;
- minimum activity/data sufficiency rules;
- multiple-testing procedure;
- missing-data behavior;
- stop/continuation rules;
- campaign duration rule;
- exact boundary/start instant.

Independent protocol review occurs before the boundary.

### 9.2 Candidate cap

FVC-1 may carry at most **3 frozen futures candidates**, including any baseline/control required by the protocol.

No online tuning.

No replacing a losing candidate mid-campaign.

### 9.3 Maximum duration

Maximum deciding observation window:

```text
182 UTC days
```

The protocol may define an earlier valid stop only if that stop rule itself is frozen before the first scored day.

### 9.4 Minimum effective sample size

Do not choose an arbitrary number such as "100 trades" and call it statistical sufficiency.

Before the boundary, the protocol must specify an effect-size floor and compute the minimum effective sample size needed for the primary comparison under:

```text
family-wise alpha <= 0.05
power >= 0.80
```

Because trades/returns are serially dependent, effective sample size must be estimated conservatively with a preregistered dependent-data method such as block bootstrap, stationary bootstrap or another justified equivalent.

For scalping, thousands of adjacent fills do not count as thousands of independent experiments.

### 9.5 Primary metrics

The protocol should distinguish:

- predictive/information metrics;
- economic metrics after all costs;
- risk metrics;
- operational/data-validity metrics.

A candidate cannot pass economically merely because a predictive AUC/accuracy is nonzero.

### 9.6 Default failure meaning

A negative FVC-1 is a result.

It does not trigger an in-place rescue search.

Any materially new candidate family requires a new preregistered campaign and consumes the finite future research budget.

---

## 10. Scalping programme — second priority

Scalping starts as a separate scientific lane after the futures directional protocol is frozen/started, so evidence accrual can run while new engineering proceeds.

### 10.1 Native information families

Candidates may use preregistered subsets of:

- top-of-book imbalance;
- spread;
- microprice;
- signed trade flow;
- aggressive buy/sell volume;
- depth imbalance;
- short-horizon realised volatility;
- liquidity depletion/replenishment;
- short-horizon basis/spot-perp divergence;
- liquidation activity where source-valid;
- MTC context as a slower regime input.

### 10.2 Horizons

Scalping is defined by its execution and holding horizon, not by naming a chart timeframe.

The protocol should explicitly define:

- decision cadence;
- expected holding time;
- exit/time stop;
- order type;
- latency model;
- whether more than one decision may occur per second/minute;
- overlap policy for positions/signals.

### 10.3 Separate campaign

A positive or negative FVC-1 does not answer the scalping question.

Scalping receives its own preregistration and prospective boundary.

---

## 11. Spot programme — second priority alongside scalping

Spot becomes first-class.

Required eventually:

- standalone spot position/accounting state;
- spot LONG execution;
- explicit short-unavailable rule unless a margin/borrow product is separately introduced later;
- spot fees/slippage;
- independent spot strategy campaign;
- ability to use spot as a supporting input without conflating instruments.

A future unified portfolio may contain both futures and spot strategies, but each family first earns eligibility independently.

---

## 12. Carry and market-neutral strategies

Carry is retained, but demoted from architectural spine to a later candidate family.

Why retain it:

- structural payoff differs from directional alpha;
- useful multi-leg accounting stress case;
- funding/basis may provide diversification;
- the current work invested in safe accounting remains valuable.

What does not follow:

- the old carry rule is not automatically a production strategy;
- successful accounting is not successful economics;
- a structural carry thesis still requires its own prospective validation.

---

## 13. Later strategy families

After futures directional and scalping/spot are dispositioned, evaluate whether to open:

- cross-sectional/statistical arbitrage;
- market making;
- options/volatility;
- cross-exchange arbitrage;
- DEX/on-chain;
- other coins/markets.

Every new family receives a source-validity review and its own scientific contract.

No family is added merely to increase the chance that something somewhere looks profitable.

---

## 14. Unified orchestration

Do not build the final strategy router before there is something worth routing.

Eligibility to open a unified multi-strategy layer requires at least **two strategy families** to have earned prospective eligibility under their own frozen protocols.

Only then design:

- capital allocation;
- correlated exposure limits;
- strategy conflicts;
- priority and netting;
- cross-strategy drawdown control;
- portfolio-level Aegis rules;
- mode/context routing if it has measurable incremental value.

The old mode/consensus architecture may inform engineering history but is not restored automatically.

---

## 15. Aegis and safety

Aegis stays the sole risk authority.

Future extensions must eventually cover:

- gross/net exposure;
- margin utilisation;
- liquidation distance;
- stale/missing data;
- reconciliation disputes;
- funding extremes;
- position/account mismatch;
- duplicate execution;
- restart safety;
- campaign-specific loss/drawdown limits;
- portfolio limits once multiple families exist.

Reductions/flattening remain reachable when new exposure is refused.

No strategy may bypass Aegis because it is "high confidence".

---

## 16. Autonomous-paper certification

Passing one research campaign is not enough to enable live money.

Before any live-route proposal, the system must demonstrate sustained unattended paper operation over real incoming data.

Required evidence should include:

- restart/recovery exercises;
- stale-feed faults;
- network disconnects;
- duplicate/out-of-order market events;
- state/log reconciliation;
- risk halt and supported recovery;
- no unexplained replay divergences;
- complete daily/monthly reports;
- no secret/manual decision path needed to keep the system running.

A strategy that only works when an operator continually rescues it is not autonomous.

---

## 17. Live-money ladder — not authorised now

No real-money trading is authorised by this roadmap proposal.

A later live-executor design may only be considered after:

1. at least one strategy passes a frozen prospective campaign;
2. independent audit confirms the result and chronology;
3. autonomous-paper certification passes;
4. the execution/accounting system is independently safety-audited;
5. a separate governance change explicitly opens a live route;
6. exchange/account/credential/security design is reviewed separately.

Possible later ladder, not yet authorised:

```text
L0  no authenticated path
L1  authenticated read-only account reconciliation
L2  tiny capped execution, leverage 1x, one strategy
L3  only after independent review of L2 evidence
```

The exact capital amounts are deliberately not set here.

---

## 18. Research-budget discipline

Historical outer blocks remain burned for deciding directional model selection.

Future programme budget:

- no more deciding reads of those historical blocks;
- at most two major prospective directional campaigns before a mandatory strategic review;
- each new strategy family consumes an explicit campaign slot;
- negative results remain visible;
- no post-result feature/horizon/model rescue inside a campaign;
- exploratory work may exist only when clearly labelled non-deciding and separated from governed evidence.

This prevents the roadmap from becoming a perpetual search until a winner appears.

---

## 19. Promotion gates

A candidate strategy may be promoted only if all applicable gates pass.

### Scientific

- frozen before prospective evidence;
- enough effective sample under the preregistered power/effect rule;
- multiple-testing procedure followed;
- primary endpoint passes;
- no forbidden post-hoc selection.

### Economic

- net after applicable fees/spread/slippage/funding;
- no profit dependent on obviously impossible fills;
- acceptable drawdown/tail behavior under the frozen criterion.

### Operational

- data coverage valid;
- replay deterministic;
- no unresolved accounting/reconciliation dispute;
- autonomous soak clean.

### Audit

- independent reviewer can reproduce/reconstruct the verdict;
- Git chronology proves the protocol predates the result.

A strategy that fails one required category does not become production-eligible because another category looks strong.

---

## 20. Independent review programme

Use independent review at several boundaries:

### A. roadmap review

Before v2 is adopted, a fresh Fable 5.1 strategic audit may:

- reject priorities;
- change stage order;
- remove unnecessary components;
- identify missing lanes;
- tighten statistical design;
- challenge MTC;
- challenge scalping economics;
- challenge the six-month horizon;
- challenge the proposed research budget.

The reviewer is not asked to ratify this file.

### B. protocol review

Every deciding campaign gets a fresh read-only review before its boundary.

### C. result audit

Every positive result receives independent reconstruction before any promotion.

For an important negative result, audit when the consequence is to close an entire research family.

---

## 21. Verification requirements for future engineering

Depending on package scope, require:

- targeted unit tests;
- positive/negative controls;
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
