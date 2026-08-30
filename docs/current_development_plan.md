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

## Active research interlock — do not disturb P6/P7/P8 mid-run

An autonomous Claude Code task has already begun the next serial research chain:

- **P6 — multi-clock / timeframe specialists**;
- **P7 — cross-timeframe Pythia consensus**;
- **P8 — automatic trading-mode routing** (scalping / day trading / swing / FLAT, subject to the exact preregistered contract produced by that task).

The external research lessons recorded later in this document were discovered **after that work had started**. They are therefore **not permission to modify an in-flight preregistration, success rule, horizon matrix, target, consensus rule, router rule or evidence after results become visible**.

The correct sequence is:

1. let the already-started P6/P7/P8 task reach a valid scientific boundary;
2. preserve all negative or positive results exactly as produced under their frozen contracts;
3. independently audit the completed work;
4. only then use the external lessons below to design the next roadmap revision.

If P6, P7 or P8 has not yet crossed its own preregistration boundary when a design defect is found, it may be corrected transparently before the first result for that checkpoint. Once a checkpoint has valid results, do not retrofit the external findings into that checkpoint.

In particular, **do not stop P6/P7/P8 merely because a potentially better idea was found on the internet**. Doing so would recreate the same adaptive selection problem the research discipline is meant to prevent.

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

An automatic router must always retain **FLAT** as a first-class outcome. It must select a temporal operating mode from causal market/model state, not by choosing whichever mode recently made the most money.

This phase is where the architecture becomes a coherent trading system rather than a collection of research models.

### 6. Later research axes only if still needed

If the specialist/consensus/router line remains insufficient, change one axis at a time. Candidate later questions include:

- cross-asset / market context;
- causal market-regime conditioning;
- deeper representation / model-architecture redesign;
- new data sources such as richer order-book information;
- different target formulations, if separately preregistered;
- cost-aware abstention / prediction-to-trade gating;
- adaptive retraining and model-expiry rules;
- explicit distribution-shift / out-of-distribution rejection.

Do not return to an endless sequence of handcrafted feature families on the same fixed setup merely because the previous family was negative.

## External research lessons — apply after the in-flight P6/P7/P8 chain

This section records external leads and engineering/research lessons found while comparing Chimera with other crypto-ML/trading projects and papers. These are **not ProjectChimera evidence** and do not override an already-frozen checkpoint. They exist so the next roadmap revision does not forget the lessons.

### A. Prediction quality is not the same thing as tradable edge

A recurring result in external work is that apparently useful predictive models become poor trading systems after turnover, fees and slippage are applied. The important object is not only the forecast but the **forecast-to-position policy**.

A particularly relevant external line uses BTC/USDT walk-forward forecasting with XGBoost/LSTM/Transformer-like models and shows a qualitative pattern that deserves independent replication inside Chimera: unrestricted trading can be destroyed by transaction costs, while a **cost-aware abstention gate** that refuses marginal position changes can drastically reduce turnover and change the economic result.

Post-P8 action:

- create a clean external-replication/sanity benchmark rather than merely borrowing the conclusion;
- test a preregistered cost-aware rule of the form "trade only when predicted edge exceeds an explicit multiple of expected transaction cost";
- compare it with the same frozen prediction stream without the gate;
- never tune the multiplier on the same outer folds used for the verdict.

### B. HOLD / FLAT should be a core alpha decision, not only a risk fallback

Mature ML-trading systems often reject predictions when the current feature state is unlike the training distribution or when the model is stale. FreqAI, for example, treats outlier/dissimilarity handling and model expiration as first-class concerns.

Chimera should eventually distinguish:

```text
model predicts direction
≠
model is sufficiently in-distribution, fresh and confident to justify a trade
```

Post-P8 questions to preregister separately:

- model age / expiration;
- distance from training distribution;
- confidence calibration;
- abstention under disagreement or distribution shift;
- whether abstention improves economic results by avoiding low-quality turnover rather than by hidden hindsight selection.

### C. Adaptive retraining may be part of the hypothesis, not merely maintenance

Crypto is non-stationary. External systems such as FreqAI are designed around repeated retraining rather than assuming one model remains valid indefinitely.

The existing Chimera research machinery deliberately freezes many things for scientific clarity, but a future checkpoint should explicitly ask whether a **predeclared rolling retraining schedule** improves forward performance compared with a frozen/less-frequently-refit control.

Do not silently introduce faster retraining into an existing checkpoint after seeing performance decay. Retraining cadence is itself a research variable.

### D. Target formulation is a major unresolved axis

The project spent many checkpoints on the fixed 1h-observation / 6h target. External work uses materially different targets, including:

- continuous future return;
- next-bar return;
- local-extrema/event labels;
- barrier-based outcomes;
- order-book price-movement labels.

Therefore do not assume that inheriting "six native bars" or any other mechanically convenient horizon is necessarily optimal for later generations. If the active P6 already froze such a rule, preserve it and interpret the result narrowly. **After P8**, target/horizon design should be treated as its own preregistered axis rather than quietly changed inside another checkpoint.

### E. Native short-horizon microstructure deserves a real short-horizon test

P3 used a genuinely new source (`aggTrades`) but compressed it into **hourly** features and evaluated it under the 1h/6h programme. That result does not answer whether microstructure is useful at seconds/minutes horizons.

For genuine scalping research, later candidates may include causally available native-resolution inputs such as:

- aggressive trade-flow imbalance;
- spread and spread changes;
- microprice;
- order-book imbalance;
- queue/depth dynamics;
- short-horizon liquidity and slippage state.

If richer L1/L2 data is introduced, it must be a new source contract with strict timestamp, sequencing and execution-parity checks. Do not retrofit it into P3.

### F. Multi-timeframe consensus can filter risk without creating alpha

External multi-timeframe confirmation experiments show an important failure mode: requiring slower-timeframe agreement can slash trade count and drawdown yet still leave profit factor below one.

Therefore Pythia consensus must not be treated as a magic alpha generator.

The proper interpretation is:

- consensus may reduce false positives and turnover;
- consensus may improve risk characteristics;
- but multiple weak specialists can still produce a weak ensemble.

After P6/P7, inspect whether individual specialists actually have useful signal. If P7 is negative, do not rescue it by adding more voting rules against the same evidence.

### G. Neural networks are not an automatic upgrade

External BTC and order-book studies repeatedly show that XGBoost/logistic baselines can match or beat LSTM/Transformer/DeepLOB-style systems once costs and implementation details are included.

The neural roadmap therefore remains:

1. establish a clean specialist/control interface;
2. freeze the same data universe, target, folds and accounting;
3. compare neural sequence specialists against the strong tree baseline;
4. only promote a neural model if it adds robust economic value, not because it is architecturally fashionable.

Candidate future classes include TCN/dilated CNN, GRU/LSTM and a PatchTST/Transformer-like sequence model. Avoid a large post-hoc architecture search.

### H. Backtest/live parity is a separate source of failure

Freqtrade explicitly maintains lookahead-analysis and recursive-analysis tooling because profitable-looking strategies often rely on future leakage or on indicator values that cannot be reproduced from the limited history available in live operation.

Chimera already has strong leakage/provenance machinery, but later production gates should also compare **actual live/paper feature and decision values against the backtest/replay implementation**, not just test formulas in isolation.

Required later checks should include:

- same candle close semantics;
- same finite lookback availability;
- same model preprocessing/state;
- same rounding and exchange constraints;
- realistic fill assumptions;
- slippage/spread behaviour;
- missed/unfilled order semantics;
- restart parity.

### I. Complexity escalation is a known trap

External live/preprint failure reports describe systems with strong historical classification metrics that remained economically weak after moving to real futures execution, even after adding more model complexity, finer data and more leverage.

Project rule:

> Do not respond to a negative checkpoint by automatically increasing model complexity, leverage or feature count.

First identify which layer failed:

- information;
- target;
- calibration;
- prediction-to-trade gate;
- transaction costs;
- regime/distribution shift;
- execution parity;
- risk sizing.

Then change one axis under a new contract.

### J. External replication should become a formal sanity tool

Before spending many new checkpoints on original ideas, it is valuable to reproduce at least one serious public result whose data/task is close enough to Chimera to be informative.

A proposed post-P8 sanity task is:

```text
BTC/USDT
+ XGBoost
+ continuous/next-horizon return target matching the external paper
+ strict walk-forward
+ explicit trading costs
+ unrestricted prediction-to-trade conversion
vs
+ preregistered cost-aware abstention
```

The point is **not** to import somebody else's claimed return. The point is to answer:

> Can Chimera's data, walk-forward and accounting machinery reproduce the same qualitative finding under independently implemented semantics?

If yes, we learn that target/execution formulation can dominate another feature-family search. If no, the discrepancy itself becomes a high-value debugging/research result.

### K. Sources/leads to revisit during the post-P8 design review

These are working references, not authority rankings:

- Freqtrade / FreqAI documentation and source — adaptive retraining, feature expansion, outlier/dissimilarity handling, model expiration, lookahead analysis and recursive analysis;
- Hummingbot — separation of controllers/strategy logic from executors;
- Jesse — backtesting/optimization workflow and overfitting cautions;
- BTC walk-forward ML-under-transaction-costs work using XGBoost/LSTM/Transformer-like models — prediction-to-trade gating / cost-aware abstention;
- high-frequency BTC/crypto limit-order-book studies — distinction between predictive classification and executable economic value;
- multi-timeframe confirmation studies — confirmation can reduce activity/drawdown without creating positive expectancy;
- FinRL / ensemble-RL work — useful later for architecture comparison, but not evidence that reinforcement learning automatically solves alpha.

Before using any numerical claim from these sources in a future preregistration, re-open the original source, verify the exact methodology and record the citation in the corresponding design document.

## Post-P8 decision review

When the current P6/P7/P8 chain is complete, do **not** immediately start another checkpoint. First produce a short decision review answering:

1. Did any native timeframe specialist show useful cost-aware signal?
2. Did Pythia consensus add value, or only reduce activity/risk?
3. Did AUTO routing add value over fixed eligible modes?
4. Was failure mainly predictive, cost/turnover-related, or execution-related?
5. Would a cost-aware abstention gate have changed the trade count materially without looking at future outcomes?
6. Is the target formulation now the most suspicious common assumption?
7. Is adaptive retraining/model expiration justified as the next independent axis?
8. Does scalping require native trade/L1/L2 data rather than candle-only inputs?
9. Is there enough signal to justify neural-specialist work, or would neural complexity merely obscure a weak target?
10. Which single next checkpoint has the highest information value?

That review, not excitement about whichever cell happens to look best, determines the next roadmap revision.

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
→ Pythia consensus / validated mode logic
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
11. **Do not modify in-flight P6/P7/P8 because of later external research.** Finish or invalidate each checkpoint under its own chronology, then revise the roadmap.
12. **Cost-aware abstention, adaptive retraining, model expiration, target redesign and native short-horizon microstructure are now explicit post-P8 candidates.** They must each be tested under their own preregistered semantics rather than smuggled into an already-read experiment.
13. **Consensus and AUTO routing are filters/selectors, not assumed sources of alpha.** Their value must be measured against strong frozen controls.
14. **Neural complexity is conditional.** Neural specialists are promoted only if they improve robust economic evidence over simpler baselines.
15. **Backtest/live parity must be proven operationally.** A mathematically causal backtest is necessary but not sufficient for live equivalence.

## Current one-line roadmap

**Merged P5 baseline → finish and independently audit the already-started P6 timeframe specialists → P7 Pythia cross-timeframe consensus → P8 automatic scalping/day-trading/swing/FLAT routing → post-P8 external-lessons decision review (cost-aware abstention, target formulation, adaptive retraining/model expiration, distribution shift, native short-horizon microstructure, external replication, neural specialists as warranted) → sustained Futures Paper Validation → mature-system freeze → Styx → very small live allocation.**