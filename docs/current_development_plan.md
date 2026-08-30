# ProjectChimera — current development plan

Status: **working development plan after P5**. This document is intentionally **not** a research preregistration. It records the intended system direction and the order in which future questions should be formalised. Exact horizons, targets, model families, weighting rules and success criteria still have to be preregistered before the corresponding research run.

## Where the project is now

The merged research line through P5 established a reproducible BTC/USDT baseline on **1h observations with a 6h prediction horizon** and then tested five incremental information directions around that same setup.

- P2b: causal market structure / SMC — negative.
- P2c: chart structure — negative.
- P3: hourly aggregation of `aggTrades` microstructure — negative.
- P4: funding, open interest and perpetual basis — screened out at Stage 1.
- P5: fully closed 4h/1d OHLCV context added to the same 1h model — negative.

Those results are evidence about that **1h / 6h research design**. They are not evidence that the original multi-timeframe specialist architecture is useless.

### What P5 did *not* test

P5 used one 1h decision clock and one model per benchmark cell. The 4h and 1d information was supplied as additional causal features to that model.

P5 therefore did **not** test the intended architecture in which:

- 1m has its own specialist model;
- 5m has its own specialist model;
- 15m has its own specialist model;
- 30m has its own specialist model;
- 1h has its own specialist/context model;
- Pythia combines the specialists by their agreement, confidence and calibrated outputs.

Do not cite the negative P5 result as having ruled out this specialist-model / consensus design.

## Target architecture

The intended Chimera architecture is a **multi-clock specialist ensemble**, not one universal model receiving every timeframe as columns.

```text
1m specialist  ─┐
5m specialist  ─┤
15m specialist ─┼─> Pythia consensus / fusion ─> Aegis ─> Hermes ─> futures execution
30m specialist ─┤                            │
1h specialist  ─┘                            └─> Argus telemetry
```

Each timeframe specialist should be allowed to learn the behaviour appropriate to its own temporal scale. The exact target and forecast horizon for each specialist must be fixed **before** its result is read; there is no assumption that every model should predict the same 6h label.

Pythia should consume the specialists' calibrated outputs rather than treating their raw votes as interchangeable. The intended behaviour is:

- strong agreement in the same direction can increase confidence;
- disagreement can reduce confidence, reduce desired exposure or produce HOLD;
- a fast model should not be able to blindly override strongly contradictory slower context;
- the consensus logic must itself be deterministic and preregistered before it is evaluated as an alpha component.

Aegis remains the sole central risk authority. Hermes remains responsible for execution semantics. Argus records the operational state and evidence. Futures Execution v1 remains dry-run only until the later live-safety gates are explicitly passed.

## Trading modes

"Mode" here means **trading style / temporal operating mode**, not market-regime labels such as trend/range/high-volatility.

The current working mapping is:

| trading mode | primary specialists | slower context / confirmation | purpose |
| --- | --- | --- | --- |
| **Scalping** | 1m, 5m | 15m | short-lived entries and precise timing |
| **Day trading** | 5m, 15m | 30m, 1h | intraday setups and direction |
| **Longer / swing** | 30m, 1h | higher context only if separately introduced later | multi-hour / longer setups |

This table is architectural intent, **not a frozen research matrix**. Exact horizons, thresholds, weights and which specialists are mandatory for agreement must be preregistered rather than selected after seeing results.

Market-regime conditioning — trend/range, volatility state, bullish/bearish context, transition state — is a separate possible research axis and must not be confused with these trading modes.

## Development sequence

### 1. Sustained Futures Paper Validation v1

Before real capital, move beyond the deterministic historical replay already completed for Futures Execution v1.

Run the dry-run execution stack for sustained wall-clock operation against a live market-data feed while keeping order placement simulated and real-money routing impossible.

Evidence should cover at least:

- restart and recovery across real process lifetimes;
- reconciliation over time;
- LONG and SHORT operation;
- Aegis vetoes and exposure limits;
- funding accounting;
- simulated partial fills and slippage;
- emergency flatten;
- state persistence and idempotency;
- telemetry continuity and latency;
- zero accidental live-order reachability.

Paper-trading PnL is descriptive operational evidence. It must not be used to choose research models, features, horizons or thresholds.

### 2. Multi-clock data and research contracts

Build the causal data layer required for the intended clocks:

- 1m;
- 5m;
- 15m;
- 30m;
- 1h.

The contract must define, before model comparison:

- exact candle/source semantics;
- causal resampling/alignment rules;
- costs appropriate to the intended trading frequency;
- target construction per clock;
- temporal split design;
- leakage tests;
- what data is research-visible;
- what remains sealed.

Do not manufacture a new holdout merely because the clock changes. P4-HOLD remains retired and Styx remains sealed unless a later mature-system protocol explicitly authorises its one-time use.

### 3. Timeframe specialist baselines

Train and evaluate **separate specialists**, not one pooled multi-timeframe feature matrix.

Each specialist receives its own preregistered research question and appropriate horizon. The goal is to learn whether a usable signal exists at each temporal scale before a consensus layer is allowed to hide individual weaknesses.

The programme must not choose the winning timeframe by trying many clocks/horizons and reporting only the best. The candidate matrix and decision rule must be fixed in advance, and all planned results must remain visible.

At the end of this phase the project should know, for every candidate timeframe:

- whether the specialist has evidence of useful cost-aware predictability;
- its calibrated confidence behaviour;
- its turnover/cost sensitivity;
- whether its failure modes are independent enough to make ensemble agreement meaningful.

### 4. Pythia cross-timeframe consensus

Only after the specialist definitions are frozen should Chimera test the central original idea: **agreement between independently trained temporal specialists**.

The consensus experiment should compare a preregistered fusion rule against appropriate frozen specialist controls. Candidate behaviours may include confidence-weighted agreement, veto/confirmation by slower models and HOLD under material disagreement, but the exact rule must be fixed before its evaluation.

The important question is not "can another model be added?" It is:

> Does cross-timeframe agreement provide robust incremental value beyond the best relevant specialist while controlling costs and turnover?

Do not tune consensus weights against the same outer results used to judge success.

### 5. Trading-mode controller

After specialists and consensus are understood, formalise the operating modes:

- scalping;
- day trading;
- longer / swing.

A mode defines which frozen specialists are active, which act as confirmation/context, the decision cadence and the risk/execution envelope. It must not silently swap models after seeing which one happens to be profitable that week.

This phase is where the architecture becomes a coherent trading system rather than a collection of research models.

### 6. Later research axes only if still needed

If the specialist/consensus line remains insufficient, change one axis at a time. Candidate later questions include:

- cross-asset / market context;
- causal market-regime conditioning;
- deeper representation / model-architecture redesign;
- new data sources such as richer order-book information;
- different target formulations, if separately preregistered.

Do not return to an endless sequence of handcrafted feature families on the same fixed setup merely because the previous family was negative.

### 7. Mature-system freeze

Before spending the sealed final evidence, freeze the candidate system closely enough that a result cannot trigger another round of selection. At minimum freeze:

- active temporal specialists;
- feature/information sets;
- model families and training procedure;
- per-timeframe targets/horizons;
- Pythia consensus logic;
- thresholds;
- trading-mode definitions;
- Aegis risk rules;
- execution semantics;
- operational recovery behaviour.

The sustained paper evidence should already be understood at this point.

### 8. Styx — one maximally independent evaluation

Styx remains sealed throughout ordinary iteration. It is not a checkpoint to view after every idea.

Only a mature frozen candidate may spend it under a predeclared one-time protocol. A disappointing Styx result is a result; it may not be followed by tuning against Styx and another claim of independent validation.

### 9. Very small live allocation

Real money comes only after:

```text
research evidence
→ multi-timeframe specialists
→ Pythia consensus
→ sustained futures paper operation
→ mature-system freeze
→ Styx
→ very small live allocation
```

Initial live operation remains conservative: isolated futures semantics, low leverage, strict Aegis authority, emergency flatten/recovery and continuous telemetry. Larger leverage or allocation is a separate future decision, not an automatic consequence of a positive research result.

## Standing constraints

1. **No real money during the current development stages.**
2. **P4-HOLD remains retired and unread.** It is not a spare validation set.
3. **Styx remains sealed** until the mature-system one-time evaluation.
4. **P5 did not test specialist-model consensus.** Its negative result applies to causal 4h/1d features supplied to the 1h/6h benchmark design.
5. **Every timeframe has its own model in the target architecture.** Do not collapse the intended design back into one universal model merely for convenience.
6. **Trading modes and market regimes are different concepts.** Scalping/day trading/swing describe how the system operates; trend/range/volatility conditioning is a separate research axis.
7. **No timeframe or horizon shopping.** Candidate clocks, horizons and success rules are fixed before results.
8. **Aegis is the sole risk authority.** Pythia expresses desired action/confidence; Hermes executes only what risk permits.
9. **Operational paper metrics do not select alpha.** Execution validation and research evaluation remain separate evidence classes.
10. **Negative results stay visible.** The objective is a defensible system, not a sequence of experiments edited until one looks profitable.

## Current one-line roadmap

**Merged P5 baseline → sustained Futures Paper Validation → causal 1m/5m/15m/30m/1h contracts → separate timeframe specialists → Pythia cross-timeframe consensus → scalping/day-trading/longer-mode controller → later research axes only if needed → mature-system freeze → Styx → very small live allocation.**
