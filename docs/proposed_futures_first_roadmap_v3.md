# ProjectChimera — Futures-First Roadmap v3

**Status:** PROPOSED — DECISION DRAFT, NOT YET ADOPTED IN REPOSITORY GOVERNANCE
**Date:** 2026-09-09
**Current verified main at drafting:** `413d597d9d609fcec79558722f5acff463387412`
**Post-drafting integration baseline:** `eb6fa2e9313bc76b99b944a5cdb448a67dea51ed` (PR #93 merged 2026-09-10)
**Primary open implementation PR:** PR #76 — recorder archive reconciliation and 30-day coverage gate
**Real-money authority:** NONE
**Scientific boundary:** no new prospective scientific boundary is activated by this document

---

## 0. Purpose

This roadmap replaces the strategic/scientific sequencing of
`docs/proposed_futures_first_roadmap_v2.md` while preserving the parts of v2 that
remain sound.

The central change is:

> ProjectChimera must not choose a prospective campaign first and ask whether it
> has enough statistical power later. Power, data scope, cost realism and
> effective sample size must determine the campaign design before the boundary
> is activated.

The target remains an autonomous futures/perpetuals-first research and
execution system, with genuine causal multi-timeframe state, centralised risk,
dry-run-first execution, prospective evidence and explicit separation between
engineering correctness and alpha.

---

# 1. Fixed owner objective

## 1.1 Primary direction

ProjectChimera is primarily a **USD-M perpetual futures LONG/SHORT system**.

The researched instrument must match the executed instrument.

The system must treat as first-class quantities:

- LONG and SHORT position state;
- mark, index, last and fill price as distinct concepts;
- funding;
- maker/taker fees;
- spread;
- slippage;
- market impact where relevant;
- leverage;
- isolated/cross margin semantics where the stage requires them;
- maintenance margin;
- liquidation;
- realised and unrealised PnL;
- account equity;
- free collateral;
- execution uncertainty;
- order/fill provenance;
- exchange state;
- reconciliation;
- restart/recovery;
- deterministic replay.

**Aegis remains the sole central risk authority.**

No strategy, router or execution path may create a second risk authority.

## 1.2 Secondary directions

After the primary directional system has a coherent engineering and scientific
foundation:

1. futures scalping;
2. spot trading;
3. spot scalping.

Scalping means native microstructure work. One-minute OHLCV trading is not to
be relabelled as scalping.

Spot is a legitimate first-class lane, not merely an input source.

## 1.3 Later lanes

Only after earlier lanes justify the abstraction cost:

- funding / basis / carry;
- statistical arbitrage;
- cross-asset strategies;
- market making;
- options / volatility;
- cross-exchange strategies;
- DEX / on-chain strategies;
- additional venues and instruments.

No later lane may distort the architecture of the first coherent futures
system.

---

# 2. Scientific firewalls

The following remain hard constraints.

1. **P4-HOLD remains retired and unread.**
2. **Styx remains sealed and unread.**
3. Burned historical outer blocks remain burned.
4. Negative results remain visible.
5. No historical winner may be relabelled prospective.
6. No new historical holdout may be manufactured from repeatedly inspected
   data.
7. P8 remains unopened unless its existing eligibility condition is genuinely
   satisfied.
8. Engineering readiness is not alpha.
9. Recorder correctness is not alpha.
10. Dry-run correctness is not alpha.
11. Paper/autonomous-demo correctness is not alpha.
12. A positive historical result would still not authorise real money.
13. No real-money route is authorised by this roadmap.

---

# 3. Strategic decision from v2 → v3

Roadmap v2 is **not adopted as-is**.

Its architecture is largely retained, but four load-bearing changes are
required:

### Change A — power before campaign

Do not freeze a BTCUSDT-only 182-day economic campaign merely because 182 days
is operationally convenient.

Before the first scientific boundary, compute whether the candidate design can
actually answer its own question under:

- dependent observations;
- candidate multiplicity;
- realistic effect size;
- volatility clustering;
- serial correlation;
- overlapping trades/labels;
- cost uncertainty.

If the power analysis says the campaign is underpowered, redesign the campaign.
Do not run it and hope.

### Change B — gen4 must preserve breadth

The next recorder generation must be **multi-symbol capable from inception**.

The exact number of symbols is not frozen here.

The initial engineering preflight should test approximately:

- **12–20 liquid USD-M perpetuals** for the broad core tier;
- **2–3 high-liquidity symbols** for full microstructure/depth capture.

The final universe is selected before contract freeze using measured:

- correlations;
- effective sample size;
- storage;
- CPU;
- bandwidth;
- replay throughput;
- source validity.

### Change C — first campaign is information + economics

The first prospective scientific campaign should not immediately try to prove
positive trading Sharpe.

It should first ask whether a predeclared causal information set contains
prospective predictive/decision information **large enough to have economic
relevance after measured costs**.

A statistically significant but economically trivial result is not promotable.

### Change D — risk configuration must actually reach Aegis

At drafting, the demo path contained a confirmed configuration-to-Aegis mapping
defect. That mapping remediation was subsequently implemented and merged in
PR #93 on 2026-09-10, with explicit production mapping and behavioural tests.

The remaining pre-soak risk-wiring blocker is deterministic clock injection for
time-dependent Aegis/order-rate/cooldown behaviour; it must be resolved in a
separate reviewed PR rather than conflated with the completed mapping fix.

---

# 4. Current recorder / PR #76 decision

## 4.1 Gen3 standing

The current recorder contract remains:

`btcusdt-prospective-gen3`

Its `prospective_from` remains:

`null`

### Decision

**Do not activate gen3 as the future scientific contract.**

Gen3 remains useful as:

- recorder engineering validation;
- archive-reconciliation validation;
- provenance validation;
- recovery validation;
- implementation substrate for gen4.

Nothing recorded under current null-boundary gen3 becomes prospective alpha
evidence later.

## 4.2 Current VPS acceptance

The isolated PR #76 recorder deployment remains limited to recorder/reconciliation
engineering evidence.

The first genuine two-day acceptance attempt, covering 2026-09-10 and
2026-09-11 UTC, is preserved as a **failed engineering acceptance campaign**:

- 2026-09-10 passed all minute-stream coverage thresholds and remained
  `UNJUDGEABLE` only because the current-month funding schedule archive was not
  yet available;
- 2026-09-11 independently failed the required minute-stream coverage thresholds
  and was classified as `RECORDER_OUTAGE`;
- both days were frozen/verified and the failed attempt must not be rescued by
  retroactively substituting later days.

A later acceptance campaign may be declared only prospectively after a clean
operational preflight. Gen3 remains `prospective_from = null`, so all such
records remain engineering evidence only.

Do not burden this VPS with gen4 multi-symbol capture. Do not modify it without
a concrete operational reason.

## 4.3 PR #76 disposition

**KEEP + REWORK + INTEGRATE EARLY.**

PR #76 must not be merged merely because its two-day acceptance passes.

Its reusable architecture includes:

- first-party archive reconciliation;
- checksum verification;
- fail-closed archive outcomes;
- full-path cache identity;
- exact coverage arithmetic;
- recorder data treated as immutable evidence;
- explicit distinction between unavailable evidence and recorder outage.

Before integration into the new direction:

1. preserve and learn from the failed first two-day engineering acceptance;
2. complete any later acceptance attempt prospectively, without replacing the
   failed campaign retroactively;
3. freeze the gen4 source contract;
4. generalise reconciliation to the gen4 symbol/stream model;
5. separate stream readiness from campaign-specific eligibility where needed;
6. preserve health-gated semantics for streams with no defensible archive
   denominator;
7. integrate on a fresh current-main lineage with normal Git chronology.

No rebase/squash/force-push is required or preferred.

---

# 5. Gen4 data architecture

## 5.1 Design principle

**Record wide enough to preserve scientific optionality; score narrowly.**

Data capture is the part of the system whose missed history cannot be recovered
later.

But high-rate data must be tiered so the recorder does not waste terabytes on
streams that are not scientifically useful.

## 5.2 Proposed tier structure

### Tier A — broad core universe

Initial preflight target: roughly **12–20 liquid USD-M perpetuals**, final count
chosen after measurement.

Candidate core streams:

- USD-M 1m klines;
- mark price;
- index price where independently available/meaningful;
- funding rate and funding interval metadata;
- aggTrades / public trades;
- open-interest / venue metrics where source-valid;
- instrument metadata;
- tick size;
- step size;
- minimum notional;
- contract status;
- leverage-bracket snapshots;
- funding-formula/version metadata where available.

Spot should be recorded for the major references required by the scientific
protocol, not automatically mirrored one-for-one for every perpetual.

### Tier B — full microstructure

Initial target: **2–3 symbols**.

Candidate streams:

- bookTicker;
- diff depth / native order-book updates;
- aggTrades / trades;
- snapshot required for local-book bootstrap;
- sequence/update identifiers;
- local-book health/reconciliation state.

This tier is the substrate for future scalping and execution research.

### Tier C — periodic metadata

Versioned snapshots of:

- exchange rules;
- symbol metadata;
- fee schedule assumptions;
- leverage brackets;
- funding interval/formula;
- relevant venue parameters.

Historical simulation must never silently apply today's venue metadata to a
past period.

## 5.3 Verification classes

Each stream must be explicitly classified as one of:

### ARCHIVE_RECONCILED

A contemporary first-party archive provides a defensible denominator/object.

### HEALTH_GATED

No adequate archive denominator exists, so integrity is established through
source-specific continuity and consistency checks.

Examples may include:

- update-sequence continuity;
- snapshot/delta consistency;
- L2 top-of-book versus bookTicker consistency;
- reconnect recovery;
- duplicate/out-of-order detection.

### DESCRIPTIVE_ONLY

Recorded for diagnosis/context but not allowed to silently become a campaign
eligibility condition.

No unavailable archive may be replaced with a fake denominator.

---

# 6. Gen4 engineering preflight

Before the gen4 contract freezes, run a separate engineering capture outside the
current PR #76 acceptance environment.

Measure for every candidate stream/symbol tier:

- events/s;
- raw bytes/s;
- compressed bytes/s;
- CPU;
- RAM;
- disk/day;
- disk/month;
- projected 182-day storage;
- projected one-year storage;
- replay throughput;
- recovery time;
- websocket reconnect behaviour;
- REST/API load;
- archive availability;
- cross-symbol correlations;
- missingness;
- source/version changes.

The final gen4 universe is chosen from these measurements.

Do not freeze 12, 20 or 30 symbols merely because the number sounds sensible.

---

# 7. Multi-Timeframe Coherence v3

## 7.1 Required concept

A causal multi-timeframe / hierarchical market-state system is mandatory.

It is not a claim that MTF creates alpha.

It is a representation capability to be tested prospectively.

## 7.2 Causal semantics

Every view carries at minimum:

- `source_time`;
- `available_time`;
- `as_of`;
- `age`;
- `staleness`;
- `complete/partial`;
- `missing`.

A higher-timeframe bar becomes visible only after it closes.

No unfinished 4h/1d candle may leak into a lower-timeframe decision.

The existing causal as-of machinery should be reused rather than replaced.

## 7.3 Preferred architecture

```text
4H / 1D
slow regime / context
        ↓
1H / 15m
setup / directional context
        ↓
5m / 1m
local state / timing context
        ↓
trades / L2 / book
execution state
        ↓
POLICY
target position + confidence + abstain
        ↓
AEGIS
objective safety / exposure / integrity
        ↓
EXECUTION
```

## 7.4 No majority voting

Do not restore P7 consensus under a new name.

Do not hard-code:

- “4H overrides 1m”;
- “three bullish clocks beat two bearish clocks”;
- arbitrary weighted voting.

Represent contradiction explicitly.

Candidate state may include:

- trend by scale;
- realised-volatility regime;
- liquidity/activity regime;
- basis;
- funding;
- open-interest state;
- coherence/disagreement;
- age/staleness;
- missingness.

## 7.5 Prediction, eligibility and risk are separate

MTF market state may:

- enter the predictive model;
- affect confidence;
- affect position sizing;
- define a preregistered strategy-specific abstention/eligibility rule.

MTF market opinion must **not** become a second safety authority.

Aegis hard gates are reserved for objective constraints such as:

- stale/missing required data;
- exchange/position disagreement;
- exposure limits;
- order-rate limits;
- funding caps;
- liquidation distance;
- kill switch;
- persistent-state integrity.

---

# 8. Model ladder

Use the smallest model that can honestly answer the scientific question.

## Tier 0 — null/control

- no-signal;
- no-trade;
- simple deterministic reference rules.

## Tier 1 — first deciding learned model

- regularised linear/logistic family;
- small predeclared feature set;
- no post-hoc rescue;
- no model zoo.

## Tier 2 — one tree challenger

Choose one tree family in advance if justified by the protocol.

Do not search XGBoost/LightGBM/CatBoost indefinitely against prospective data.

## Tier 3 — later only if earned

- sequence models;
- deep learning;
- representation learning;
- transformers;
- RL/policy learning.

Deep learning is not a milestone. It is allowed only if the available sample
size, event rate and scientific question justify its extra variance and search
space.

---

# 9. First prospective scientific campaign

## 9.1 The first question

The first campaign should ask:

> Does a preregistered causal perpetual-futures information set provide
> prospective predictive or decision information beyond the corresponding
> baseline, and is that incremental effect large enough to retain plausible
> economic value after measured execution costs?

## 9.2 Co-primary scientific logic

Promotion requires BOTH:

### Informational relevance

The predeclared information set adds measurable prospective signal relative to
the same candidate without that information.

### Economic relevance

The measured effect clears a predeclared economic floor based on realistic
costs.

A result may be statistically interesting but still fail promotion.

## 9.3 Candidate endpoints to choose during protocol design

The protocol may select, after power analysis and before the boundary:

- conditional expected executable forward return;
- cost-aware classification/calibration;
- time-series IC;
- cross-sectional rank IC;
- another well-justified information metric.

Economic quantities must remain secondary/co-primary in a way that prevents an
economically trivial IC from being called success.

## 9.4 Validation requirements

Before the boundary freeze:

- effect-size floor;
- economic floor;
- exact symbol universe;
- exact horizons;
- candidate count;
- feature families;
- MTF arm definition;
- baseline arm;
- multiplicity method;
- dependence method;
- validation unit;
- missing-data rule;
- cost model;
- stopping rule;
- PASS;
- FAIL;
- abstention rule;
- negative controls.

Required negative controls include at minimum:

1. MTF block removed/zeroed;
2. shuffled-label or equivalently invalid-signal control where scientifically
   appropriate.

## 9.5 Power before boundary

The protocol must compute power using at least:

1. a simple analytical baseline;
2. a dependence-adjusted estimate;
3. block bootstrap / stationary bootstrap / simulation or another justified
   realistic sensitivity analysis.

Report:

- nominal N;
- effective N;
- candidate multiplicity;
- alpha;
- target power;
- minimum detectable effect;
- expected calendar duration.

If the design cannot answer its own question, do not activate it.

---

# 10. Historical evidence policy

The four old outer blocks remain closed to new deciding use.

The old programme is still informative in a restricted way:

### Reusable as negative priors

- SMC / market-structure implementation tested in P2b;
- chart structure tested in P2c;
- hourly-aggregated microstructure tested in P3;
- specific P5 MTF implementation;
- P6/P7 candidate/rule families.

### Not reusable as fresh model-selection evidence

Do not use repeated historical winners/leads to choose the first prospective
candidate.

Historical negative results do not prove entire information families are
impossible.

Example:

> P3 is evidence against the tested hourly-aggregated microstructure design,
> not against event-scale microstructure.

---

# 11. Aegis / risk remediation

Before any unattended autonomous demo, complete a narrow remediation:

**DemoLimits → RiskLimits/Aegis must be explicit, complete and tested.**

At minimum trace and test:

- funding adverse streak;
- loss streak;
- max open positions;
- max orders/minute;
- cooldown;
- max daily loss;
- max data delay;
- max drawdown;
- max exposure per asset;
- max total exposure;
- max funding cost;
- max leverage;
- minimum liquidation distance.

No test harness may maintain its own duplicate mapping.

The production mapping must be the single mapping boundary used by tests.

This is an **unattended-demo blocker**, not an alpha question.

---

# 12. Futures runtime roadmap

## 12.1 Before autonomous demo

Minimum coherent runtime:

- LONG/SHORT;
- 1x isolated only;
- exact signed quantity;
- average entry;
- fees;
- discrete funding settlement;
- realised/unrealised PnL;
- free cash/equity;
- recorded-quote fills;
- position/account reconciliation;
- Aegis;
- crash consistency;
- deterministic decision log;
- replay parity;
- explicit mark/index/last semantics where the runtime uses them.

Do not require institutional-grade exchange simulation before demo.

## 12.2 Before prospective economic campaign

Add/validate as required by the chosen campaign:

- measured lane-specific transaction costs;
- correct fee schedule;
- funding timing;
- spread/slippage distributions;
- fill sensitivity;
- size-dependent impact where material;
- stronger replay equivalence.

If leverage remains fixed at 1x, full leveraged-liquidation fidelity is not a
scientific prerequisite.

## 12.3 Before any future leveraged/live consideration

Required later:

- leverage brackets;
- maintenance amounts/cumulative maintenance;
- bracket-dependent liquidation solve;
- isolated and cross semantics;
- one-way and hedge mode as applicable;
- partial liquidation;
- insurance-fund/ADL semantics as applicable;
- authenticated exchange/account reconciliation;
- venue-side protection;
- independent infrastructure watchdog;
- credential/key isolation;
- live-authorisation contract;
- capital cap;
- leverage cap;
- rollback and kill-switch drills.

None of these requirements authorises real money.

---

# 13. Scalping programme

Scalping remains a strategic priority, but its stages are:

## S-M1 — measurement

Record native:

- trades/aggTrades;
- spread;
- best bid/ask;
- depth;
- sequence integrity;
- book state;
- short-horizon realised moves.

Measure the gross signal/cost envelope.

## S-M2 — execution calibration

Before a maker-based scientific claim:

- queue assumptions;
- fill probability;
- adverse selection;
- cancellation/replace latency;
- partial fill behaviour;
- maker/taker fees;
- depth/impact sensitivity.

## S-M3 — deciding campaign

Only after the measured gross signal is plausibly larger than the realistic
execution-cost envelope.

A conservative taker simulation may be used as a falsification benchmark, but
a deciding taker-first campaign should not be launched merely to rediscover
that fees dominate the horizon.

---

# 14. Spot programme

Spot becomes a first-class lane after the first coherent futures demo and data
substrate.

Spot may support:

- independent spot strategies;
- spot/perp basis/context;
- hedge legs;
- cross-market information;
- later arbitrage/carry.

Spot alpha and perpetual alpha must never be silently substituted for each
other.

---

# 15. Autonomous demo definition

The first autonomous demo is an **engineering milestone**.

Minimum chain:

```text
recorder
→ normalisation
→ MarketContext(as_of=t)
→ frozen rule/model
→ target position
→ Aegis
→ dry-run futures execution
→ ledger
→ hash-chained decision log
→ replay
→ daily report
```

Acceptance should include:

- unattended runtime;
- restart recovery;
- feed outage drill;
- persistent-state recovery;
- reconciliation dispute drill;
- kill-switch drill;
- no unexplained decision-log divergence;
- no manual rescue required for ordinary recovery.

A clean autonomous demo is **not evidence of alpha**.

---

# 16. Evidence ladder

## Rung 1 — engineering validation

Tests, invariants, accounting traces, mutation witnesses, CI.

## Rung 2 — simulator / data validation

Recorder/reconciliation, measured cost model, replay determinism, fill-model
sensitivity.

## Rung 3 — autonomous demo

Unattended operation with failure/recovery drills.

## Rung 4 — economically interesting prospective result

Predeclared, cost-aware, multiplicity-aware, dependence-aware result above both
scientific and economic floors.

## Rung 5 — prospective alpha candidate

Rung 4 plus independent continuation/replication and independent audit.

## Rung 6 — eligible for separate real-money consideration

Only after a new live-authorisation governance contract, infrastructure review,
credential isolation, capital/leverage caps, and operational drills.

Rung 6 means *eligible for consideration*, not approved.

---

# 17. Later multi-strategy architecture

Do not build this abstraction before at least two strategy families have earned
prospective eligibility.

Final shape:

```text
shared market-data substrate
        ↓
shared causal MarketContext
        ↓
strategy lanes
        ↓
target-position intents
        ↓
capital/netting layer
        ↓
AEGIS — sole risk authority
        ↓
shared execution/reconciliation
        ↓
one accounting ledger
        ↓
one hash-chained evidence trail
```

Required properties:

- one risk authority;
- one authoritative account/ledger;
- explicit capital conflicts;
- no duplicate market-state pipelines;
- no hidden strategy-to-venue bypass;
- full trace from strategy intent to final fill and PnL.

---

# 18. Active / legacy component disposition

| Component | Decision |
|---|---|
| recorder machinery | KEEP |
| gen3 contract | RETAIN FOR ENGINEERING, DO NOT ACTIVATE |
| PR #76 reconciliation | KEEP + REWORK |
| Aegis | KEEP; CONFIG WIRING REMEDIATED IN PR #93; DETERMINISTIC CLOCK INJECTION STILL REQUIRED PRE-SOAK |
| futures dry-run domain/executor/store | KEEP |
| current 1x margin model | KEEP FOR DEMO, EXTEND LATER |
| recorded-quote fill model | KEEP FOR EARLY DIRECTIONAL DEMO |
| carry domain/accounting | KEEP, DEMOTE TO LATER LANE |
| carry-shaped demo runner | REWORK TOWARD GENERAL SINGLE-LEG FUTURES |
| decision log | KEEP |
| replay parity | KEEP + STRENGTHEN |
| causal MTF implementation | KEEP / REUSE |
| microstructure feature definitions | KEEP DEFINITIONS, RECOMPUTE AT NATIVE SCALE |
| old checkpoint runners | ARCHIVE / DO NOT REOPEN |
| MTST/deep historical lane | RETIRE FROM ACTIVE PATH |
| Freqtrade runtime path | RETIRED/DISCONNECTED |
| old consensus/mode routing | RETIRED FROM FUTURE ARCHITECTURE |
| MLflow/Ray-style optimisation stack | DEFER/RETIRE UNLESS A FUTURE NEED EARNS IT |
| broad strategy router | DEFER UNTIL MULTIPLE ELIGIBLE LANES |

---

# 19. Roadmap stages

## V3-0 — protect current acceptance

**Status:** ACTIVE / FIRST CAMPAIGN FAILED AND PRESERVED.

**Objective:** finish PR #76 VPS engineering acceptance honestly, preserving
negative operational evidence.

**Observed first campaign:**

- 2026-09-10: minute-stream thresholds passed; funding verdict unavailable due
  to expected monthly archive latency;
- 2026-09-11: required minute-stream thresholds failed and `RECORDER_OUTAGE`
  was recorded;
- the 2026-09-10/11 pair is permanently a failed acceptance attempt and is not
  replaced retroactively.

A subsequent two-day attempt may start only after a clean prospective preflight.

**Forbidden:**

- gen4 on the same VPS;
- boundary activation;
- economic scoring;
- rewriting or substituting failed acceptance days;
- merge merely because the service ran.

---

## V3-1 — Aegis wiring remediation

**Status:** MAPPING REMEDIATION COMPLETE VIA PR #93.

PR #93 merged one explicit `DemoLimits → Aegis` production mapping, reused it
in the test harness, and added behavioural witnesses without changing any
scientific artifact or contract.

### V3-1b — deterministic clock injection

**Objective:** remove wall-clock dependence from time-sensitive Aegis/order-rate/
cooldown behaviour before unattended demo/soak and compressed deterministic
replay.

**Acceptance:**

- one injectable authoritative clock path for relevant risk timing;
- production defaults preserve real-time behaviour;
- replay/tests can provide deterministic time;
- no scientific artifact/contract change;
- no weakening of risk limits.

**Kill condition:**

- any ambiguity that changes risk semantics must be resolved before implementation,
  not guessed.

---

## V3-2 — separate gen4 engineering preflight

**Objective:** measure the real cost of preserving future scientific optionality.

**Scope:**

- candidate multi-symbol core tier;
- limited microstructure tier;
- source preflight;
- storage/network measurement;
- panel correlation / ESS estimate.

**Acceptance:**

- credible bytes/day and one-year projection;
- source validity classified;
- candidate universe supportable operationally;
- no dependency on current PR #76 VPS.

**Forbidden:**

- prospective boundary;
- alpha scoring.

---

## V3-3 — power analysis and first-campaign design

**Objective:** choose an answerable first scientific question.

**Acceptance:**

- analytical baseline;
- dependence-adjusted estimate;
- bootstrap/simulation sensitivity;
- effect floor;
- economic floor;
- multiplicity;
- universe;
- duration;
- PASS/FAIL.

**Kill condition:**

- campaign cannot reach adequate power → redesign before implementation.

---

## V3-4 — freeze gen4 contract

**Objective:** create the acquisition identity that future evidence may use.

**Dependencies:**

- V3-2;
- V3-3.

**Acceptance:**

- exact symbols;
- exact streams;
- verification class per stream;
- missingness semantics;
- storage policy;
- source identity;
- metadata/version policy;
- independent plan review.

**Forbidden:**

- retroactive relabelling of gen3 data.

---

## V3-5 — generalise PR #76 reconciliation

**Objective:** make recorder validity compatible with gen4.

**Scope:**

- multi-symbol;
- new archive-reconciled families;
- health-gated families;
- campaign-specific eligibility;
- immutable evidence semantics.

**Acceptance:**

- real-source verification;
- fail-closed behaviour;
- deterministic offline recomputation;
- full current-main integration verification.

---

## V3-6 — activate gen4 prospective boundary

**Objective:** start irreversible scientific evidence accrual.

**Dependencies:**

- gen4 contract frozen;
- reconciliation readiness;
- production recorder identity verified;
- one qualifying pre-activation engineering period;
- independent boundary review.

**Acceptance:**

- activation commit;
- fresh storage root;
- immutable boundary;
- no mixed contract hashes.

---

## V3-7 — causal MarketContext / MTC fabric

**Objective:** produce one as-of-correct multi-resolution state.

**Acceptance:**

- no unfinished-bar leakage;
- explicit age/staleness;
- explicit missingness;
- contradiction representation;
- deterministic replay.

---

## V3-8 — general single-leg futures runtime

**Objective:** remove the carry-shaped dependency from the primary futures path.

**Acceptance:**

- FLAT/LONG/SHORT/REDUCING/HALTED/DISPUTED/RECOVERING;
- position/account state authoritative;
- 1x isolated;
- exact accounting;
- Aegis;
- recovery;
- replay.

---

## V3-9 — autonomous demo / soak

**Objective:** prove the system can operate unattended.

**Acceptance:**

- predetermined soak duration;
- restart drills;
- feed-outage drills;
- kill-switch drills;
- reconciliation-dispute drills;
- no unexplained replay divergence;
- no operator rescue needed for expected recovery paths.

**Scientific standing:** engineering only.

---

## V3-10 — first prospective information/economic campaign

**Objective:** answer the first scientifically powered market question.

**Acceptance:** only as preregistered in V3-3.

**No online tuning.**

**No replacement of a losing candidate.**

**No post-hoc universe expansion.**

---

## V3-11 — economic campaign

Opened only if V3-10 clears both informational and economic promotion floors.

This is the first stage allowed to ask whether the candidate has robust
cost-adjusted trading value under the defined execution model.

Still no real money.

---

## V3-12 — strategy expansion

Candidate order:

1. futures scalping once microstructure/execution evidence is sufficient;
2. spot directional;
3. spot scalping;
4. carry/basis where justified;
5. cross-sectional/stat-arb;
6. later families.

Each lane earns promotion separately.

---

## V3-13 — multi-strategy orchestration

Opened only when at least two strategy families have earned prospective
eligibility.

Introduce:

- explicit capital allocation;
- netting;
- lane conflicts;
- one Aegis;
- one execution layer;
- one ledger.

---

## V3-14 — future live eligibility framework

Separate project/governance stage.

Must include:

- authenticated execution architecture;
- credential isolation;
- capital cap;
- leverage cap;
- venue/account reconciliation;
- kill/flatten procedures;
- rollback;
- independent infrastructure watchdog;
- failure drills;
- independent audit.

No automatic transition from prospective alpha to live money exists.

---

# 20. Immediate next actions

## ACTION 1 — now

Keep the PR #76 VPS isolated. Preserve the failed 2026-09-10/11 acceptance
campaign and allow a new two-day campaign only after a clean prospective
operational preflight.

## ACTION 2 — next coding PR

Implement V3-1b deterministic clock injection for the time-sensitive Aegis/
order-rate/cooldown path. The `DemoLimits → Aegis` mapping itself is already
remediated by merged PR #93.

## ACTION 3 — in parallel on separate infrastructure

Run the separate gen4 data-rate + source-validity + power preflight. Do not put
gen4 workload on the PR #76 acceptance VPS.

Only after the preflight and power work should the exact symbol universe,
contract and first scientific campaign be frozen.

---

# 21. Decisions locked now

The following are adopted at the strategic-decision level by this v3 draft and
should not be reopened casually:

- futures/perpetuals-first remains the primary direction;
- LONG and SHORT are first-class;
- Aegis remains the sole central risk authority;
- prediction and execution timing are distinct;
- causal Multi-Timeframe Coherence remains mandatory;
- MTC is not majority voting;
- gen3 `prospective_from` remains null;
- gen4 must be multi-symbol capable;
- genuine scalping requires microstructure;
- the first scientific campaign must be power-designed before boundary;
- information alone cannot pass without an economic relevance floor;
- old outer blocks are not reopened;
- autonomous demo is engineering evidence only;
- no real money is authorised.

---

# 22. Decisions intentionally left open until preflight

The following must **not** be guessed now:

- exact gen4 symbol count;
- exact final symbol universe;
- exact primary horizon;
- exact primary information endpoint;
- exact economic floor;
- exact tree challenger;
- whether 182 days is sufficient after the chosen panel/dependence model;
- whether full L2 depth belongs on 2, 3 or more symbols;
- whether leverage above 1x is scientifically useful for the first economic
  campaign.

These are resolved from measured evidence before the relevant boundary.

---

# 23. Later independent audit

A future Fable or other independently authorised cross-model review may audit
this decision.

That review is a **post-decision adversarial checkpoint**, not a reason to leave
ProjectChimera directionless until the reviewer becomes available.

The highest-value later review questions are:

1. Is the chosen panel/effective-sample-size model defensible?
2. Is the selected information + economic endpoint the best first prospective
   question?
3. Is the final gen4 universe large enough to preserve scientific optionality
   without creating unmanageable operational cost?
4. Are any hidden researcher degrees of freedom still large enough to invalidate
   the proposed campaign?
5. Does the final preregistration truly have enough power to answer its own
   question?

---

# 24. One-line roadmap

> **Finish PR #76 engineering acceptance → fix Aegis config wiring → measure a
> separate tiered multi-symbol gen4 design → compute power before campaign →
> freeze gen4 → generalise reconciliation → activate a fresh prospective
> boundary → build causal MTC + general single-leg futures runtime → prove an
> unattended autonomous demo → run the first answerable information+economic
> prospective campaign → open a full economic campaign only if earned → add
> scalping/spot/later lanes one by one → build unified orchestration only after
> multiple lanes qualify → consider live eligibility only under separate future
> governance.**
