# ProjectChimera — Full Adversarial Strategic, Scientific, Quantitative, Architecture, Technology and Live-Readiness Audit, with a Complete Rewritten Master Roadmap

**Reviewer:** independent owner-level reviewer (Claude Fable 5.1, externally selected; subagents inherited the same model).
**Repository:** `tyrannya/ProjectChimera`, `main` = `46921ef1206748c6b7304432a26c8295b7830e27` (verified; merge of PR #95).
**Date:** 2026-09-13.
**Mode:** READ-ONLY. No file in the repository was edited; no commit, branch, push, PR, issue, CI, VPS, recorder or acceptance state was touched. The harness-created local branch `claude/chimera-full-audit-roadmap-d3gaqu` carries no commits and was not pushed to.
**Corrected 2026-09-14** per the Final Owner Corrigendum (its Part A lists every change; Part D lists every edited passage; §37 is replaced in full).
**Standing of this document:** a PROPOSAL from an independent audit. It creates no scientific evidence, no prospective boundary, no alpha claim and no real-money authority. The owner may adopt all, part or none.

Evidence labels used throughout: **[RF]** repository fact (read from code, tests, git, artifacts, with `file:lines`); **[EF]** external fact (cited); **[INF]** inference; **[REC]** recommendation.

---

## Contents

1. Executive Decision
2. Exact Revision and Evidence Reviewed
3. ProjectChimera Current-State Reconstruction
4. Scientific Standing
5. Engineering Standing
6. Existing Architecture Adversarial Review
7. Aegis / Risk Review
8. Execution Review
9. Accounting Review
10. Time / Causality / Replay Review
11. Persistence / Crash / Recovery Review
12. Recorder / Data Architecture Review
13. Research-Methodology Review
14. Futures-First Adjudication
15. Target / Horizon / Frequency Adjudication
16. MTF Adjudication
17. Microstructure Adjudication
18. Universe / Symbol Selection Adjudication
19. Feature-Family Adjudication
20. Model-Ladder Adjudication
21. Evaluation / Multiple-Testing Adjudication
22. Power / MDE / Effective-Sample-Size Adjudication
23. Cost / Execution-Economics Adjudication
24. Technology Landscape and Build-vs-Adopt Decisions
25. Future Architecture Review
26. Autonomous Demo Definition
27. Prospective Campaign Design
28. Strategy Expansion Governance
29. Portfolio / Orchestration Adjudication
30. Live-Trading Readiness Review
31. Security / Infrastructure Review
32. Failure-Mode Pre-Mortem
33. Decision Register
34. Ranked Critical / High / Medium / Low Risks
35. What Must Happen Next
36. What Must Not Happen Next
37. COMPLETE REWRITTEN MASTER ROADMAP (PROPOSAL)
38. Delta: Old Roadmap (v3) vs New Roadmap (§37)
39. Final Go / No-Go Statements

---

## 1. Executive Decision

**ProjectChimera is a well-governed, honestly documented research and dry-run platform with strong provenance, persistence and safety engineering, no demonstrated alpha, no prospective data, and a runtime that cannot yet run unattended or start a campaign. The adopted Roadmap v3 points in the right direction and is wrong in its sequencing, in one internal coherence gap, and in one load-bearing arithmetic assumption. It should be replaced by the roadmap in §37, which keeps its firewalls and its power-before-boundary principle, repairs the runtime before growing it, resolves the endpoint/runtime coherence question, and accepts that economic evidence takes longer than 182 days.**

Ten findings carry the decision (the number in brackets is the section where the evidence lives).

1. **The demo cannot start, cannot run as a service, cannot be resumed by an operator, and cannot detect a dead feed. [RF]** The committed campaign config refuses to build a runner; every CAMPAIGN run halts at SELF_CHECK on a source-identity defect; `run` is one bounded three-minute catch-up pass re-invoked by systemd every 30 s; `resume`/`resolve` cannot succeed from the CLI; the stale-feed veto compares a clock to itself and is identically zero, so a dead recorder leaves an open position silently held with no halt, funding booking or liquidation check; two ordinary crash windows produce disputes no command can clear. Each of these is disclosed somewhere in the repository and none is scheduled by Roadmap v3; independent verifiers rated each MEDIUM rather than HIGH because the runtime is dry-run only and the fixes are small — the point is that they gate the demo, not that money is at risk today (§§8, 10, 11, 26).
2. **The recorder-to-runner cadence makes replay parity structurally unreachable as deployed. [RF/INF]** The recorder rewrites the open day's parquet on a ≥300 s cadence while the runner decides only the newest three minutes; most minutes become `SKIPPED_STALE`, and the parity document itself says such skips make later state diverge (§10).
3. **The money arithmetic is correct; the account model is not general. [RF]** Funding sign, fees, realised PnL, funding deduplication and the two-leg carry identity all hand-trace exactly. But equity exists only inside the carry-shaped ledger the roadmap demotes, the funding-cost entry veto and the loss-streak/cooldown rules have no production caller, equity is valued at kline close while liquidation reads mark price, and the carry ledger's "identity check" is algebraically tautological (§9).
4. **The historical instrument could not see the effects it searched for, and no inference machinery exists to replace it. [RF]** Every deciding gate was a 3-of-4 fold sign count on 4–280 trades per fold with a self-declared coin-null pass rate of 5/16, a 29-point threshold grid as a second fitted parameter, and ~90 cells over 12 readings of the same four blocks. No bootstrap, effective-N, IC, power, MDE or multiplicity code exists anywhere (§§13, 21, 22).
5. **Roadmap v3's first campaign, as sketched, cannot reach its own economic floor. [INF, arithmetic in §22]** 182 days on 12–20 correlated perps detects a time-series IC of ~0.05–0.09 at a 6 h horizon (implausibly large) and a cross-sectional IC of ~0.025–0.03 (plausible), while taker costs of 12–16 bps make any IC below ~0.08 at 4–6 h uneconomic. Information and economics are answerable in different regions; the co-primary design as written would open an underpowered campaign, which v3's own Change A forbids.
6. **Roadmap v3 has a coherence gap between endpoint and runtime. [INF]** A panel information endpoint is cross-sectional evidence; the primary lane and the planned V3-8 runtime are single-symbol time-series. Either the endpoint is time-series (underpowered) or the runtime must be multi-symbol single-leg (unbudgeted). §§14, 15, 27 resolve this by making the runtime multi-symbol and the primary endpoint cross-sectional.
7. **"Mandatory hierarchical MTC" is a premature architectural commitment. [RF/INF]** P5, P6, P6-EXT and P7 are narrow negatives measured with a weak instrument; that licenses a representation *test*, not a locked architecture. The causal as-of machinery is sound and is kept; the hierarchy is simplified to at most two closed-bar slow-scale variables inside the frozen feature set (§16). The runtime must nevertheless be multi-timeframe *capable*: a generic multi-clock causal snapshot is a required engineering property (R7), while hierarchical MTC as a scientific model is not required for campaign 1 and may return only under a new preregistered campaign.
8. **Futures-first survives; the strategy family does not, as stated. [REC]** The USD-M perpetual stays the researched and executed instrument (SHORT is native, taker fees are half of spot's, the validated executor exists). "Directional LONG/SHORT" is a shape, not a hypothesis; the first frozen family is one falsifiable panel hypothesis with an executable-return target and a turnover-throttled policy (§14). The primary lane is multi-symbol, single-leg, directional LONG/SHORT on USD-M perpetuals; spot is reference data, a hedge leg where a lane needs one, and support infrastructure. The carry mechanism, demoted by owner decision without refutation, is the only mechanism whose *economic* evidence is reachable in a six-month window and is recommended as a SECONDARY, OPTIONAL, separately preregistered lane on the runtime that already exists (§27): it never delays R1–R9, has its own hypothesis slot, is not a prerequisite for anything, is not orchestration, receives no priority for arriving sooner, and yields on any resource conflict.
9. **Gen4 is right to be multi-symbol; its numbers are guesses. [RF/REC]** 12–20 symbols and 182 days appear before any measurement; storage was already under-estimated 24× once. The preflight must fix the universe from a measured correlation matrix and cost envelope, and the contract must separate immutable raw capture from regenerable normalisation (§§12, 18).
10. **PR #76 is sound in design and stale in lineage. [RF]** Its reconciliation and coverage code is fail-closed and well-tested offline; it is 129 commits behind `main` with a known conflict, its first acceptance campaign failed on day two, and the roadmap must not wait on it. Both outcomes of campaign #2 are handled in §12 and §37 R5.

**What must happen next (§35):** adopt or decline this proposal (R0); repair the runtime's integrity defects and delete the last live-capable path (R1); start the separate gen4 engineering preflight on a second host (R2). **What must not happen next (§36):** activating any prospective boundary, freezing a gen4 contract, opening a campaign, running the soak, or treating any historical lead as a candidate before R1–R3 are done.

**Confidence:** HIGH on findings 1–4 and 10 (read off code, tests and artifacts; findings 1–3 were produced by subagents and verified by a separate verifier agent, finding 4 by a subagent and the lead, finding 10 by the lead alone from the extracted PR #76 tree); MEDIUM-HIGH on 5–7 (first-principles arithmetic under stated assumptions, produced independently twice — by the lead and by a subagent — with agreeing orders of magnitude); MEDIUM on 8–9 (strategic judgement on evidence that is narrow by construction).

**Process disclosure.** The audit was designed as a dynamic workflow of 23 investigator workstreams, each to be challenged by adversarial verifiers with different lenses, all on the same serving model. The account's usage limit was reached in three successive windows and every agent in flight at those moments failed; the workflow was cut down to cluster investigators with a single combined verifier lens and relaunched after each reset. Five investigator reports completed: accounting (§9) and clocks/persistence (§§10–11), both independently verified by a separate verifier agent against the source and adjusted where the verifier disagreed; target/horizon/power (§§15, 22), Aegis/boundaries/execution (§§6–8) and external methodology plus Binance documentation (§§13, 17, 23, 24), which no verifier reached — their load-bearing claims were instead re-read by the lead in the source or in the cited external documents and are marked [RF]/[EF] only where that re-reading succeeded. The remaining workstreams (current state, governance, recorder/PR #76/gen4, MTF/microstructure/strategy, tests/demo/ops, costs, technology landscape) were performed by the lead directly from the repository, with claims labelled and cited; where a claim rests only on the lead's reading it says so. The adjudication panels planned for contested decisions did not run; the adjudications in §§14–31 are the lead's. No agent modified anything.

---

## 2. Exact Revision and Evidence Reviewed

**Git state [RF].** `HEAD` = `46921ef1206748c6b7304432a26c8295b7830e27` = `origin/main`, an ordinary two-parent merge of PR #95 (second parent `82390d618d8163659369621773a5add109827b72`, first parent the roadmap-adoption merge `1f95bd74af1f22929c76901de1503ed82daf795a`). 305 commits, 57 merge commits; the last 30 merges are ordinary merges with visible chronology. Working tree clean. The values given in the task brief match the repository.

**PR #76 [RF].** Open, draft, "PR-06 — recorder archive reconciliation and 30-day coverage gate", head `8a8f4a1f7d754cff190a3668873072a7efbb1542`, base `95322a991e762ca40a406bb69093d148bcd9d22a` (129 commits behind `main`), +5,735/−114 over 8 files; `tests/test_recorder_no_network.py` changed on both sides, so a merge will conflict there. A read-only copy of its tree was extracted for review; nothing on GitHub, on the VPS, or in any acceptance environment was touched.

**Test suite [RF].** Locked Linux dependencies installed in a disposable copy; full suite run: every test passed except five that require a `.git` checkout (`git ls-files` in `tests/test_reporting_integrity.py`, `nn.source_identity` in `tests/test_source_identity.py`); those 57 tests pass in a local clone. The three integrity verifiers (`verify_research_snapshot`: 23 checks; `verify_trade_snapshot`: 28; `verify_research_state`: 13 checkpoints) pass. Effective verdict: green at HEAD in this environment. PR #95's body reports 5,218 collected on the same head.

**Documents read in full by the lead [RF].** `CLAUDE.md`; `docs/agent_workflow_policy.md`; `docs/current_development_plan.md`; `docs/proposed_futures_first_roadmap_v3.md`; `docs/fable_5_1_full_project_strategic_audit.md`; `docs/architecture.md`; `docs/futures_execution_v1.md`; `docs/futures_dry_run_validation.md`; `docs/replay_parity.md`; `docs/demo_runbook.md` §0; the S0–S6 stage and definition sections of `docs/proposed_demo_implementation_master_plan.md`; the "next" sections of `docs/research_roadmap.md`; `conf/demo/pvc1.json`; `chimera/recorder/contracts/btcusdt-prospective-gen3.json`; `.github/workflows/ci.yml`; `deploy/systemd/chimera-demo.service`; `pyproject.toml`; `.pre-commit-config.yaml`; `.env.example`; `chimera/safety.py`; `chimera/demo/rules_carry.py`; outlines and key functions of `chimera/risk.py`, `chimera/futures/{domain,executor}.py`, `chimera/demo/{runner,feed}.py`, `chimera/recorder/{streams,service,health,incremental,normalize}.py`, `nn/{mtf,multiclock,data_pipeline,evaluate,source_identity}.py`, `tools/demo_run.py`. Subagents read the accounting, clocks/persistence and target/power surfaces in depth (their file:line citations are reproduced in §§9–11, 15, 21, 22).

**Sealed and burned data [RF].** No parquet under `data/research/` was loaded by any participant. No statistic was computed from historical market data. `P4-HOLD` and Styx were not read. PR #76's acceptance data was not touched.

**External evidence.** Where external facts are used (fee schedule, funding cadence, archive availability, methodology literature) they are marked [EF] with a source when a subagent verified them, and [INF] where they rest on the reviewer's knowledge and remain to be confirmed by the owner before they enter a preregistration.

---

## 3. ProjectChimera Current-State Reconstruction

### 3.1 Purpose [RF/INF]
A single-operator research and dry-run platform whose stated aim (v3 §0–1) is an autonomous, futures-first, centrally risk-managed system that produces prospective, preregistered evidence about whether a small systematic crypto strategy has an edge, and that could — only under separate governance — trade real money at 1× on Binance USD-M perpetuals. Everything in the repository so far serves the *evidence-production* half of that aim; nothing serves the *real-money* half beyond the absence of a live route.

### 3.2 Chronology that matters [RF, from `git log`]
- The whole scientific programme (v4, P2a, P2b, P2c, P3, P4, P5, P6, P6-EXT, P7, P8, P13) ran between 2026-08-20 and 2026-08-30 on four historical outer blocks of Binance spot BTCUSDT (2023-03-04 to 2025-05-19). Preregistration-to-evidence margins were minutes to hours (P5 4 h 39 m; P6 28 min; P7 24 min; P6-EXT 8 min; P13 preregistered 17:37 and closed NOT EVALUABLE 17:41 on the same day).
- The first independent audit (PR #68, 2026-09-03) declined P14, withdrew P8, ended deciding use of the four blocks, and set a prospective recorder plus a carry-spine demo as the plan. Its engineering programme (recorder PR-04/05, Aegis PR-03, recorded-quote fills PR-07, config/decision log PR-09, carry ledger PR-08, runner/rules/feed PR-10, replay parity PR-11, observability PR-12, disconnection of the retired runtime PR-13) merged 2026-09-03 to 2026-09-06.
- Roadmap v2 (PR #90, 2026-09-08) and v3 (PR #92, adopted by PR #94 on 2026-09-12) replaced the carry-spine plan with futures-first directional, multi-symbol gen4, and power-before-boundary. V3-1 (PR #93) wired campaign limits into Aegis; V3-1b (PR #95, 2026-09-13) injected a deterministic clock into Aegis and the executors.
- Nothing prospective has ever been recorded into the repository; the gen3 contract's `prospective_from` is `null`; the PR #76 VPS's first two-day acceptance (2026-09-10/11) failed on day two (`RECORDER_OUTAGE`).

### 3.3 Component classification [RF]

| component | class | on runtime import closure | what exercises it |
|---|---|---|---|
| `chimera/recorder/*` — contract, events, sink, normalize, incremental, streams, rest, service, health | IMPLEMENTED (engineering-validated); single-symbol BTCUSDT um+spot, six streams, no aggTrade/depth/OI/forceOrder | recorder process | 768 recorder tests on synthetic servers; one failed VPS campaign |
| `chimera/recorder/reconcile.py`, `coverage.py` (PR #76 only) | IMPLEMENTED on a stale branch; not on `main` | — | offline tests in PR #76 |
| `chimera/demo/*` — runner, feed, rules, rules_carry, rules_shadow, decision_log, reports, telemetry, faults, clock, risk_wiring, inspection, config | IMPLEMENTED but **carry-shaped and not runnable as a campaign** | yes | `tests/test_demo_*` (≈1,100 tests); never run end-to-end against a live recorder (runbook §0) |
| `chimera/carry/*` — hedge, ledger, accounting, factory | IMPLEMENTED; the only actionable strategy shape today (spot-long/perp-short) | yes | lifecycle, parity and hedge tests; hand-traced by the accounting subagent |
| `chimera/futures/*` — domain, executor, store, venue, fills, accounting | IMPLEMENTED; dry-run only; MARKET orders; one-way; 1× isolated; 16/16 frozen invariants | yes | `artifacts/futures_dry_run_v1` (hashed protocol) |
| `chimera/risk.py` (Aegis) | IMPLEMENTED; persisted state; config mapping (PR #93); injected clock (PR #95); several rules unreachable | yes | `tests/test_risk_*`, `test_demo_risk_wiring.py` (75 tests) |
| `chimera/metrics.py`, `notify.py`, `conf/alerts_demo.yml`, Grafana | IMPLEMENTED | yes | promtool in CI; `test_demo_observability.py` |
| `chimera/safety.py`, `tools/run_bot.py`, `strategies/*`, `conf/<exchange>.<mode>.json`, Freqtrade `Dockerfile` | HISTORICAL / DISCONNECTED; **the only live-capable code**, double-gated | no | `test_retired_runtime_disconnected.py`, `test_config_and_cli.py` |
| `chimera/modes.py`, `consensus.py`, `inference_client.py`, `nn/infer_service.py`, `nn/registry.py` serving side | HISTORICAL / DISCONNECTED | no | frozen-evidence tests; `make smoke` in CI |
| `nn/train.py`, `model_def.py` (MTST), `walkforward.py`, `evaluate.py`, `benchmark*.py`, `information_sets.py`, `data_pipeline.py`, `dataset.py`, `wf_diagnostics.py` | IMPLEMENTED research pipeline; HISTORICAL for deciding use | no | frozen artifacts recomputed in CI |
| `nn/p2b*`, `p4*`, `p5*`, `p6*`, `p7*`, `p13*` | HISTORICAL-SUPERSEDED frozen checkpoint code (~26k lines) | no | evidence tests |
| `nn/mtf.py`, `multiclock.py`, `microstructure.py`, `trade_aggregates.py`, `smc.py`, `chart_structure.py`, `derivatives*.py`, `regime.py` | IMPLEMENTED feature families, retired from active use; the as-of machinery in `mtf`/`multiclock` is reusable | no | leakage battery |
| `data/research/*` | Styx rows physically absent; **P4-HOLD rows physically present** in the raw parquet, sealed by ledger and code path, not by absence | — | `verify_research_snapshot` |
| Roadmap v3 stages V3-2 (gen4 preflight), V3-3 (power), V3-4 (contract), V3-5 (generalised reconciliation), V3-6 (boundary), V3-7 (MarketContext), V3-8 (single-leg runtime) | **ROADMAP ONLY — zero code** (no `gen4`, no `MarketContext` in `chimera/`, no effective-N/bootstrap, no multi-symbol recorder, no single-leg runner) | — | — |
| `mlruns/`, `[tracking]`, `[tune]` extras | EXPERIMENTAL / UNUSED | no | — |
| `knowledge/` | non-authoritative vault, near-empty | — | — |
| Infra: `docker-compose.yml` (recorder, demo, Prometheus 2.53, Grafana 11.1, Alertmanager 0.27; legacy services behind a profile), hardened systemd units, `Dockerfile.demo/.recorder` on `python:3.11-slim` | IMPLEMENTED, never operated end-to-end for a demo | — | compose config parsed in CI; images not built in CI |

Sizes [RF]: `chimera/` 26,320 lines; `nn/` 37,830; `tools/` 11,759; `tests/` 73,817 across 118 files (3,802 test functions; ~5.2k collected). Test code exceeds all source code.

### 3.4 Strategy families, models, evaluation, execution, risk, recorder, demo, infrastructure — one paragraph each [RF]
- **Strategy families in code:** three-class cost-aware direction classifiers on OHLCV14 and five retired feature families (historical); a two-leg delta-hedged carry rule (`R1_carry`) with unfrozen parameters (runtime); a shadow linear-score rule that cannot size. No single-leg directional rule exists on the runtime path.
- **Models:** MTST transformer (retired), untuned logistic regression / LightGBM / XGBoost (historical); no model is served, no model artifact is promoted (`artifacts/models/` does not exist).
- **Evaluation:** nested walk-forward, purge = horizon, inner-fold threshold grid (29 points), k-of-4 fold sign gates; no statistical inference machinery.
- **Execution:** dry-run MARKET-only executor with a deterministic adverse fill model and a recorded-quote fill model (touch + configured slippage); order state machine; reconciliation against a venue seeded from the store; idempotent events; atomic store.
- **Risk:** Aegis `RiskEngine` with persisted state, kill-switch file, halt/resume, exposure/drawdown/daily-loss/order-rate/staleness/funding-streak/liquidation-distance rules; several rules unreachable (§7).
- **Recorder:** websocket + REST capture of six streams into append-only NDJSON.gz raw files and regenerable per-day parquet, with heartbeat and per-stream last-event age metrics; alerts at the Prometheus layer.
- **Demo:** file-driven bounded catch-up passes; carry-shaped; cannot start a campaign; replay parity exact only for restart-free, operator-free runs.
- **Infrastructure:** one VPS (PR #76's, isolated), systemd units, docker-compose stack, Prometheus/Grafana/Alertmanager; CI on GitHub Actions (Ubuntu + Windows), locked Linux deps, pre-commit, promtool.

### 3.5 Unresolved blockers, dead architecture, debt [RF]
Blockers: campaign cannot start (config, identity defect); runner not a daemon; operator lifecycle unusable; stale-feed veto vacuous; catch-up cadence mismatch; disputes without clearing path; parity policy for restarts and operator actions undecided; PR #76 lineage; no gen4 design or preflight code. Dead/unused: modes, consensus, inference service, registry serving side, Freqtrade path, MLflow, Ray Tune, MTST. Research debt: P6 dirty-tree fits (disclosed), P4 snapshot and P4-HOLD coverage record not committed, v2/v3 walk-forward runs absent, LR non-reproducible across BLAS builds, P13 execution-gap ambiguity, HOLD-majority doc defect. Engineering debt: lock without hashes, Actions by tag, floating base images, no property-based tests, no crash-injection at persistence boundaries, stale docs after PR #95.

---

## 4. Scientific Standing

**Answered, all negative, all narrow [RF].** v4 (MTST loses to buy-and-hold 20/20), P2a (model family does not matter; XGBoost +0.007 indistinguishable from zero), P2b/P2c/P3/P4/P5 (each added information family negative at 1 h/6 h under a 3-of-4 gate), P6/P6-EXT (no native clock viable under XGBoost; secondary LR/LightGBM leads recorded and refused), P7 (consensus negative vs an oracle benchmark, 13 trades in one mode). P8 withdrawn unopened; P13 closed on source validity, never economically run; P14 declined before opening.

**What the negatives license [INF].** They are evidence against specific implementations at one horizon, one cost model, one asset — measured with an instrument whose false-positive rate was 31% and whose power against a Sharpe-1 alternative was ~0.65–0.77 (§21). They do not establish that any family is useless; they do establish that the four outer blocks are burned and that any historical "lead" is post-selection.

**Standing of the seals [RF].** Styx: rows physically absent from every committed parquet; the instant is a contract constant; hindsight-era ceiling disclosed. P4-HOLD: rows physically present in `data/research/btc_usdt_1h_gen1_raw_pre_styx.parquet` (raw rows 45802–48210); the seal is a ledger (`state: retired`) plus a code path (`nn.p4_holdout.assert_holdout_release`), i.e. procedural, not physical. This is not dishonesty; it is a governance fact the roadmap should state (§13).

**Prospective standing [RF].** None. No prospective data exists; no campaign protocol is frozen; no boundary is active. The next deciding evidence, by adopted policy, comes from data that has not yet occurred.

**Standing of the demo and dry-run evidence [RF].** Engineering only: 16 dry-run invariants under a hashed protocol on burned historical candles; a paper smoke that places zero orders; a runtime that has never run against a live recorder.

**Alpha claim:** none exists and none is possible from any committed artifact. **Economic claim:** none. **Real-money authority:** none.

---

## 5. Engineering Standing

**Strong [RF].** Atomic persistence everywhere (temp + fsync + replace; the carry ledger also fsyncs its directory); a hash-chained, per-record-fsynced decision log with torn-tail recovery that preserves removed bytes; startup triage that names four recovery causes before the first append; fail-closed loading (unreadable files never overwritten; unbootstrapped executors refuse to plan); idempotent order events and funding settlements persisted in the store; a level-triggered, persisted kill switch re-checked every tick; reductions and emergency flatten always available while halted; Decimal money path with hand-traceable arithmetic; a deterministic runner clock on the decision path with a frozen-clock control test; AST-asserted absence of any live route in `chimera/futures`; leakage-safe research pipeline; hashed preregistrations and frozen evidence manifests verified in CI; two-OS CI on a lock file.

**Weak [RF].** The runtime is carry-shaped and single-symbol; the campaign profile cannot start; the runner is not a service; operator lifecycle commands fail; the stale-feed veto is vacuous; the catch-up cadence skips most minutes as deployed; the recorder rewrites the open day's parquet in place; two crash windows create permanent disputes; several Aegis rules are unreachable; equity, liquidation and funding read three different prices; replay parity cannot reach PARITY across a restart or an operator action; no property-based or crash-injection test harness; supply chain pinned by tag/version rather than SHA/hash; no external dead-man switch; no backup/restore drill; nothing built for V3-2 through V3-8.

**Net [INF].** The evidence core (persistence, log, recorder capture, futures domain, Aegis skeleton) is a sound foundation. The operational shell around it (service loop, operator lifecycle, staleness, cadence, dispute clearing, multi-symbol, single ledger) is where the next engineering must go — before any soak, and before any boundary that a soak would validate.


---

## 6. Existing Architecture Adversarial Review

Source: an independent Aegis/boundaries/execution subagent traced every path from decision to venue and ran two-sided synthetic controls for the new HIGH finding; its verifier did not complete (usage limit), so the lead re-read every cited line for the findings used here and confirmed the load-bearing one in the source (`chimera/demo/risk_wiring.py:269-276`; `chimera/risk.py:775-799`).

**Boundary map, as it exists [RF].** Recorder → normalised minute rows (`BookMinute` = last top of book before the minute closed; spot book in receipt time, perp in exchange time) → `FeedCursor.state_for` → `MarketState` (closes, mark, mark high, index, books, last funding rate; `complete`/`missing`) → `CarryRule.evaluate` → `HedgeTarget` (an unsigned BTC quantity) → `HedgedPosition.plan` → per-leg `TargetPosition` → `FuturesExecutor.execute_target` → `plan_transition` → `_run_intent` → `_ask_aegis` → `RiskEngine.evaluate_entry` (OPEN/INCREASE only) → `DryRunFuturesVenue.submit` → events → per-leg `Ledger` → `CarryLedger` (reconciled by level differences each cycle) → `FuturesStore` ×2, `carry_ledger.json`, `risk.json`, `runner_state.json`, decision log. Telemetry reads; it never writes state. The only wall-clock reader on the decision host is telemetry.

**Who answers each question [RF].**

| question | authorities found | verdict |
|---|---|---|
| what position exists? | `FuturesStore` per leg (execution truth); `CarryLedger` hedged quantity (economic view); decision-log `position_after` (evidence); the dry-run venue's `reported_position` (seeded from the store) | one execution truth with a derived economic view and a self-referential "venue"; consistent by construction, but the reconciliation attests self-consistency only (§11) |
| what equity exists? | `CarryLedger` (free cash + levels) via `HedgedPosition.mark_to_market`; `RiskState.equity` (a float copy); telemetry `net_pnl = equity − capital`; executor `net_pnl` (realised − fees + net funding, no unrealised) | **two headline numbers**: carry equity and executor net PnL, each correct on its own terms; no general account ledger (§9) |
| what risk applies? | `RiskEngine` for pre-trade approval and account-level guards; `HedgedPosition.liquidation_touched` for continuous liquidation risk with its own maintenance-rate constant; `CarryLedger` disputes; the runner's halt calls | **Aegis is the sole pre-trade authority, not the sole risk authority** (§7) |
| what time is authoritative? | `RunnerClock` = minute close on the decision path; exchange/receipt times persisted by the recorder | one decision clock, but it cannot see elapsed time (§10) |
| what price is authoritative? | kline close for equity; mark/mark-high for liquidation; settlement mark for funding; recorded touch for fills | three valuation prices, not unified (§9) |
| what trade happened? | `OrderRecord.applied_events` (idempotent) → per-leg ledger → carry ledger → decision-log `execution` block | one truth, replicated by design |
| what cost was paid? | executor fees/funding; carry ledger fees/slippage (attribution); reports | one computation, replicated; slippage measured against close rather than mid |
| what state survived restart? | five files with no cross-file transaction; triage plus `reconstruct()` re-derive consistency; `risk.json` absent → defaults (§7) | fail-closed except for the risk file's absence |

**Hidden second authorities [RF/INF].** (1) The carry layer decides continuous liquidation risk with `HedgeConfig.maintenance_margin_rate = 0.004`, duplicated in `rules_carry.py` and the venue table (`hedge.py:209, 876-911`; `rules_carry.py:192-196`). (2) `HedgedPosition.plan`/`correct` pre-check `risk.state.halted`/`is_disputed` directly rather than through an Aegis decision. (3) The CLI hard-codes `capital = Decimal("1000000")` and the recorder contract id outside the hashed campaign config (`tools/demo_run.py:116-118`), so a campaign's identity does not name its capital or its data contract. (4) The dry-run "venue" is the store seeded back to itself.

**The abstraction "prediction emits target; execution owns how; Aegis owns risk" [INF].** Correct as a separation of concerns and implemented at *order* granularity only: for a two-leg atomic target Aegis never sees the target, so it can approve a state the strategy never asked for (a naked perp SHORT after the spot leg is vetoed or refused), and it counts both hedge legs as gross exposure. For a single-leg runtime the gap collapses to one order per transition; for the multi-symbol runtime this proposal requires, a *target-set* pre-check ahead of per-order checks is the missing piece (§25, R8).

**Verdict [INF].** The evidence core is sound; the boundaries are honest but not yet the ones the primary lane needs: one account ledger, one valuation rule, one risk authority for continuous as well as pre-trade risk, and a target-level Aegis check.

---

## 7. Aegis / Risk Review

**What is genuinely enforced [RF].** Every exposure-increasing order reaches `DryRunFuturesVenue.submit` only through `FuturesExecutor._run_intent → _ask_aegis → RiskEngine.evaluate_entry`; there is no branch that skips it for OPEN/INCREASE; only `emergency_flatten` passes `bypass_risk_gate=True`; the rate-limit halt is re-checked after `record_order` (`executor.py:687-732, 603-610`). Reductions cannot become increases or reversals (side first-class, magnitude quantity, through-zero fills refused, reversals as two legs, venue reduce-only modelling, flatten-not-reaching-zero disputes the symbol). Aegis state is complete, atomic, semantically hashed, fail-closed on unreadable or unknown-schema files; the kill switch is level-triggered, checked at construction, start and every tick, persisted, and cannot be cleared by a crash. The `DemoLimits → RiskLimits` mapping is single, exhaustive and import-checked (`risk_wiring.py:97-165, 279-339`); both the CLI and the test harness build Aegis through it. The injected clock is real: one bound `RunnerClock.time` reaches Aegis and both executors, and read-only inspection injects a raising clock.

**Findings.**

| id | sev | finding [RF] | consequence | resolve by |
|---|---|---|---|---|
| AEG-1 | **HIGH** (new; confirmed by the lead in the source; no independent verifier ran) | `build_risk_engine` constructs the engine (which loads persisted equity, peak, day and day-start) and then unconditionally calls `engine.update_equity(float(capital))` (`risk_wiring.py:269-276`). `update_equity` rolls the UTC day and sets `day_start_equity = equity` when the day changed, and halts when `(peak − equity)/peak ≥ max_drawdown_pct` (`risk.py:775-799`). Under process-per-pass supervision every day roll happens at a process start, so a healthy campaign is halted "max daily loss breached" on the first true mark ≤ 98% of nominal capital after a day roll, and any campaign whose persisted peak ≥ capital/(1 − 0.05) ≈ 105.3% is halted "max drawdown breached" at the next start; the subagent's two-sided control reproduced both (restart=false → not halted; restart=true → halted 2.50% ≥ 2.00%; +6% peak → halted 5.66%; +4% → not halted). `day_start_equity`, `daily_pnl` and `equity` are inside `risk.state_hash`, a replay MUST_MATCH field, so live and replay diverge after the first day roll even without a halt. Not disclosed in the runbook. | A false, terminal halt of a healthy soak; unexplained replay divergence. | R1 (seed equity only when no persisted state exists; two-sided tests across a day boundary and with a persisted peak above capital; parity across a day boundary with restarts) |
| AEG-2 | HIGH→MEDIUM (see §11 REC-1; disclosed) | No executable operator path out of a persisted HALT; CAMPAIGN cannot tick; `flatten` persists a spurious halt reason. | Every halt is terminal; drills cannot pass. | R1 |
| AEG-3 | HIGH (design; per-order approval) | Aegis approves per order, not per target: the spot leg can be vetoed or refused after the perp leg filled; `HedgedPosition.correct()` has no caller; PARTIAL is retried implicitly every minute with no timeout; `liquidation_touched` reads `min(spot, perp) = 0` for a one-legged position and returns False; each hedge leg counts as a separate open position and as gross exposure (`hedge.py:521-540, 892-894`; `executor.py:1013-1019`; runbook §0 sixth). | Unbounded one-legged exposure, unmonitored for liquidation, after a partial-leg refusal. | R1 (wire `correct()` with a bounded timeout or flatten a PARTIAL after N minutes; check the larger leg); R8 (`evaluate_target` before any leg is sent) |
| AEG-4 | MEDIUM | `risk.json` absence fails open: `FileNotFoundError` on a sound filesystem yields defaults (unhalted, no dispute, no funding streak, zero peak) (`risk.py:557-572`); nothing compares the loaded snapshot with the log's last `risk.state_hash` or HALT record, unlike the store and ledger continuity checks. | Deleting one file silently clears a persisted halt with no record. | R1 (refuse a missing `risk.json` when the log already has records, or require an operator note; compare with the log's last hash and write a RECOVERY record) |
| AEG-5 | MEDIUM | Dead limits on the demo path: `loss_streak_limit`/`cooldown_seconds` (no caller of `record_trade_result`), `max_funding_cost_rate` (no `funding_rate` passed), `max_data_delay_s` (delay ≡ 0), `max_orders_per_minute` (at most two approvals per campaign minute) (`risk_wiring.py:62-90`; `runner.py:1076, 1091`). | Configured limits that enforce nothing; false assurance in the config. | R1 (wire or remove from the schema; staleness guard outside the decision path) |
| AEG-6 | MEDIUM | Continuous and portfolio risk live outside Aegis: liquidation touch decided in the carry layer with its own constant (three copies of 0.004); disputes decided by the ledger; halts issued by the runner; Aegis's own liquidation guard is entry-only. | Ownership spread across packages; drift between constants changes which layer "sees" a touch. | R8 (move continuous position guards into Aegis fed by executor `MarginState`; source the maintenance rate from `SymbolConstraints` only) |
| AEG-7 | LOW | Operator flatten recorded as `FlattenCause.RISK_HALT` although `OPERATOR` exists; capital and contract id outside the hashed config; stale test prose about a wall-clock Aegis and a fixture rate override of 64. | Evidence-quality defects. | R1 |

**What belongs in Aegis and what must not [REC].** In: pre-trade approval of every exposure-increasing order *and* of the target-position set (portfolio gross/net/per-symbol caps); continuous account guards (drawdown, daily loss from persisted equity, with the day defined once); continuous position guards (liquidation distance and maintenance from executor-computed margin state); integrity gates (kill switch, dispute state, staleness mark from an operational guard, funding-cost and funding-streak rules); order-rate limits. Out: scientific campaign limits (trades per day, allowed symbols, blocks, windows) — these belong in the campaign contract and are enforced by the policy layer; execution-quality thresholds (quote age, fill divergence) — these belong in the venue/fill model and produce rejections, not vetoes; exchange-risk handling (maintenance windows, rate-limit bans, listenKey expiry) — the adapter's, reported to Aegis as health inputs; operational risk (disk, clock skew, host health) — the supervisor's and the dead-man switch's, reported to Aegis as health inputs. Aegis remains one engine per runtime process; two dry-run lanes are two engines with separate dry-run capital; only R13 introduces one engine over several lanes.

---

## 8. Execution Review

**What exists [RF].** A deterministic order state machine with a total transition table (`PLANNED → RISK_APPROVED → SUBMITTED → ACKNOWLEDGED → PARTIALLY_FILLED/FILLED`, with `CANCELLED/REJECTED/FAILED` and `RECONCILIATION_REQUIRED` requiring explicit resolution); MARKET orders only; one-way mode; 1× isolated enforced at construction; idempotent events by `event_id` persisted with the record; `filled_quantity ≤ intent.quantity` enforced (over-delivery flagged); order ids as local sequence numbers seeded past persisted ids; quantities quantised down to the step, never up; minimum-notional waived only for reduce-only; reconciliation compares local to "reported" and never overwrites local; `recover()` idempotent at five restart boundaries; sixteen hashed invariants held on burned historical candles; `RecordedQuoteFillModel` crossing to the recorded touch plus configured adverse slippage, refusing stale, crossed or future books and fills better than the touch; `DeterministicFillModel` adverse and partial-fill capable for the validation protocol.

**Absent by design, adequate for dry-run, required before live [RF/INF].** Client-order-id ↔ venue-order-id mapping; venue order query; retry; cancel/replace; timeout and unknown-state semantics ("Unknown error" means UNKNOWN execution status and must not be treated as failure — Binance general-info [EF]); REST/WS state model; user-data stream consumption (fills, commission and realised PnL arrive as `ORDER_TRADE_UPDATE` events, not in the REST response [EF]); venue rate limiter (2,400 weight/min, 300 orders/10 s, 1,200 orders/min; 418 bans [EF]); `listenKey` keepalive (60 min) and 24 h stream closure [EF]; maintenance-window handling; hedge mode (reduce-only is unavailable in hedge mode [EF]); leverage brackets and cumulative maintenance amounts; ADL/insurance semantics; partial liquidation; funding cadence per symbol (8 h default, 4 h or 1 h on some contracts, caps via `fundingInfo` [EF]); GTX post-only expiry semantics; `newClientOrderId` charset/length (`^[\.A-Z\:/a-z0-9_-]{1,36}$`) [EF]. None of this should be built now; all of it belongs to R14 as a separate package.

**Findings.**

| id | sev | finding [RF] | resolve by |
|---|---|---|---|
| EXE-1 | MEDIUM | `_resolve_stranded_orders` cancels only `PLANNED`/`RISK_APPROVED`; `SUBMITTED`/`ACKNOWLEDGED` orders left by a crash stay non-terminal indefinitely (asserted by `tests/test_futures_store.py:454-462`); `require_ready` does not block on them. Adequate while the venue is synchronous and in-process; against a real venue this is the duplicate-submission window. | R1 (mark non-terminal submitted orders `FAILED("no venue confirmation after restart")` or `RECONCILIATION_REQUIRED` on recover so the open-order set is finite); R14 (venue query before any retry) |
| EXE-2 | MEDIUM | Aegis consulted per order (AEG-3): a two-leg target can end one-legged; the multi-symbol runtime needs a target-set check. | R8 |
| EXE-3 | LOW | `FuturesStore.open` does not catch `PositionError`/`FuturesError` from semantically invalid persisted positions, so such a file crashes construction instead of taking the designed UNREADABLE path (`store.py:162-178`; `domain.py:223-244`). | later |
| EXE-4 | LOW | Fill-model reference for slippage attribution is the kline close, not the mid; no bps figure emitted (A12 gaps). | R3 (deciding cost semantics frozen as hashed code shared by runtime and evaluator; mid-referenced bps slippage reported as a non-deciding diagnostic per §23) |

**Before demo vs before live [REC].** Before demo: EXE-1 finite open-order set; target-level Aegis check for multi-symbol; unified valuation price; a staleness guard; nothing exchange-facing. Before live (R14): everything in the "absent by design" list, drilled on testnet with an acceptance checklist drawn from the primary documentation (market orders return `FILLED` immediately with `RESULT` but fills arrive on the user stream; crossing post-only orders `EXPIRE`; reduce-only rejects `-2022`; `MIN_NOTIONAL` waived for reduce-only; `-1021` timestamp errors; `listenKeyExpired` is independent of disconnection [EF]).

**Is the runtime carry-shaped? [RF]** Yes: `DemoRunner` owns exactly one `HedgedPosition`; the rule emits an unsigned hedged quantity; equity lives in the carry ledger; the CLI hard-codes the contract and capital. What V3-8/R8 reuses unchanged: `chimera/futures` domain, executor, store, venue constraints, fills, per-leg ledger; Aegis persistence and mapping; the decision log; replay tooling. What it rewrites: the rule interface (signed per-symbol target set with confidence and abstain), the position owner (N single-leg positions), the account ledger, the Aegis portfolio scope, and the runner's tick around a snapshot rather than a two-leg plan.

---

## 9. Accounting Review

Source: an independent accounting subagent hand-traced three worked examples and then executed them against the real code in a scratch directory (no repository write); the lead re-read the cited lines for every HIGH finding. Verifier verdicts are incorporated where they arrived (§1 process disclosure).

**What is correct and is kept [RF].**
- One funding sign convention, `cash_flow = −sign(side) × notional × rate`, implemented identically in `chimera/futures/accounting.py:96`, `chimera/carry/accounting.py:402`, `nn/p13_carry.py:571` and `chimera/risk.py:960`; LONG pays a positive rate, SHORT pays a negative one; paid and received are separate magnitudes never netted; deduplicated by settlement id in the futures ledger and by instant in the carry ledger, with a torn-booking dispute on restart.
- Fees on notional at the executed (slipped) fill price, taker only, quantised to 1e-8, refused if negative; realised PnL booked only on reductions as `(price − entry) × qty × sign`, VWAP unchanged by reductions, reversal refused; leverage applied exactly once and pinned at 1×; Decimal end to end on the money path with floats only at the Aegis boundary and metrics.
- Hand traces agree to the last digit: (a) LONG 0.5 BTC entered at 60,013.0 (ask 60,001 + 2 bps), fee 15.00325, funding +0.0001 at mark 60,010 → −3.0005 paid, exit 60,487.9, realised +237.45, net +204.324275; (b) SHORT with a −0.0001 rate pays −2.9995, net +205.175525; (c) carry 0.5 BTC on 100,000 capital: entry equity 99,970.993 = capital − fees 22.507 − measured slippage 6.5, funding received +1.50075, basis 7.00 → 10 gives equity 99,977.49375 = capital + Q·(entry_basis − basis) + net funding − fees, and full close returns free cash with zero residue. No double counting between the executor ledgers and the carry ledger: fees enter free cash exactly once through the per-cycle delta.
- A12 "slippage is measured, not spent" is implemented and tested against an executor oracle; the P13 port is asserted term by term against `nn/p13_carry.py`; the daily report is a pure function of ledger-effect blocks.

**Findings.**

| id | sev | finding [RF unless marked] | consequence | resolve by |
|---|---|---|---|---|
| ACC-1 | HIGH→MEDIUM (verifier: scheduled design item, nothing computes a wrong number at HEAD) | No general authoritative account ledger exists for a single-leg futures runtime. `FuturesExecutor.Ledger` holds fees, funding, realised PnL and turnover but no capital, free cash, unrealised or equity (`chimera/futures/accounting.py:109-126`); equity is defined only by `HedgedPosition.mark_to_market` over `CarryLedger` fields (`chimera/carry/hedge.py:849-854`), the ledger v3 demotes. | The V3-8 runtime either grows a third ledger or bolts equity onto the executor; Aegis halts on whichever equity was wired last. | before the runtime rework (R8 design) |
| ACC-2 | HIGH→MEDIUM (verifier: dry-run only, gap disclosed in code) | The A10 side-aware funding-cost entry veto is unreachable: `HedgedPosition._execute` calls `execute_target` without `funding_rate` (`chimera/carry/hedge.py:524-528`); `risk_wiring.py:84-90` documents it as NOT REACHABLE; only the adverse-streak halt (after three paid settlements) protects entries. `record_trade_result` has no production caller either, so `loss_streak_limit` and `cooldown_seconds` are dead on the demo path (`chimera/demo/risk_wiring.py:62-71`). | `conf/demo/pvc1.json` limits that enforce nothing at entry; approvals the adopted design says should be vetoes. | R1 / R8 |
| ACC-3 | MEDIUM→LOW (verifier: the tautology reproduces, but the runbook already states the check cannot see a leg wrong by a fill, so "false assurance" is refuted as a characterisation) | `CarryLedger.check_identity` is algebraically tautological: `entry_basis` is derived from the same two VWAPs the residual is computed from (`chimera/carry/ledger.py:545, 708-713`); corrupting a leg VWAP moved free cash by −493 with residual 0 and no dispute; the repository's own violation tests reach a residual only by hand-editing state. | The code comment and plan text overstate the check; the runbook does not. | before gen4 freeze (R1) |
| ACC-4 | MEDIUM | Valuation prices are not unified: equity marks the perp at kline close, liquidation and margin distance at mark price (`hedge.py:844-848, 895, 903`), funding at the settlement mark; v3 §12.1 requires explicit mark/index/last semantics before demo. | Halts and liquidation checks read different prices in the same minute; evidence equity ≠ venue equity. | before demo (R8) |
| ACC-5 | MEDIUM→LOW (verifier: the adopted preregistered model; matters only for a spot/carry economic claim) | Spot leg fidelity: the spot BUY fee is charged in quote at full quantity, while Binance spot deducts the fee from the base asset received unless paid in BNB [EF: Binance spot fee FAQ, fetched 2026-09-13]; the spot leg is modelled as an isolated 1× "futures" position with a maintenance rate and a fictitious liquidation price handed to Aegis (`chimera/carry/factory.py:64-77`). | A real hedge is short ≈ 10 bps of spot; the hedged invariant is exact only in the simulator. | before any economic claim on the carry lane |
| ACC-6 | LOW | Liquidation formula is a first-order approximation (documented, anti-conservative for SHORT by a small amount); the demo's "isolated check" is only the price threshold; a naked perp leg in PARTIAL is not liquidation-checked (`hedge.py:892-911`). | Immaterial at 1×; wrong to reuse above 1×. | before real money |
| ACC-7 | LOW | Aegis rolls its day at the minute close, so the 23:59 minute and the 00:00 settlement fall into the next Aegis day while the report dates them to the previous day; the 00:00 settlement never enters any `daily_pnl` (`runner.py:1076, 1117, 1294`; `risk.py:775-783`). | Daily-loss statistics and daily reports cannot be reconciled exactly. | before demo |
| ACC-8 | LOW | Research-side `TargetSpec` fee (5 bps "spot taker") is half the demo's spot fee; A12's two recorded gaps (no bps slippage figure; slippage measured against close, not mid) remain open. | Research and execution cost models are not the same object. | R3 (one cost-model module imported by both the runtime and the evaluator; frozen as hashed code) |

**Single-ledger decision [REC].** Promote a venue-agnostic account ledger (capital, free cash, per-position margin levels, fees, funding paid/received, realised; unrealised derived at a stated mark price) of which the carry ledger becomes the two-position special case; keep the executor ledger as the per-leg cash-flow journal; state the valuation rule once (perp unrealised/equity at mark; spot inventory at last) and test that equity and liquidation read the same price in one tick. `docs/futures_execution_v1.md` §10's "RiskEngine persists only its halt" is stale (RiskState now persists peak/day-start/daily PnL) and should be corrected.

---

## 10. Time / Causality / Replay Review

Source: an independent clocks/persistence subagent classified every clock hit in `chimera/`, `tools/`, `nn/` and traced the runner's tick; the lead confirmed the two HIGH findings in the source (`chimera/demo/runner.py:1076, 1091`; `chimera/recorder/service.py:107, 717-720`).

**Clock classification [RF].**
- MARKET EVENT TIME (decision-relevant, persisted, deterministic): the runner clock as actually driven — `RunnerClock.observe(minute_ns + MINUTE_NS)` at `runner.py:463, 1076, 2348` (the minute's close, never a receipt stamp, despite `clock.py:10` saying `max(receipt_ns)`); `MarketState.minute_ns` (kline open); fill-model instants (minute close); funding `funding_time_ms`; Aegis day roll, order-rate window and cooldown on the injected clock since PR #95.
- EXCHANGE PROCESSING TIME (persisted): perp bookTicker/markPrice `E`; kline close `T`.
- LOCAL RECEIPT TIME (persisted, not decision-relevant): `time.time_ns`/`monotonic_ns` on every raw event; spot bookTicker has no exchange time, so its canonical time *is* receipt time, labelled `book_time_basis` — one normalised row therefore mixes an EXCHANGE-domain perp book with a RECEIPT-domain spot book (labelled; consumed by nothing on the decision path).
- DECISION TIME (injected): `RiskEngine`, both executors and the ledger's `_now_text` receive the runner clock; the inspection path injects a raising clock.
- OPERATIONAL WALL TIME (telemetry only): heartbeat, feed-age gauge, settlements re-read trigger (`st_mtime`), recorder maintenance cadence; provenance stamps in artifacts. A frozen-clock control test proves none reaches a record.

**Which row is visible [RF].** A decision at minute t reads minute t's own closed kline, the last book before t's close, mark close/high of t, fills at t's close crossing that book, and stamps `runner_now_ns = t + 60 s` with zero decision-to-fill latency (disclosed in `fills.py:48-57`). A row exists only if the recorder accepted a closed frame (`x == true`) *and* the open day's parquet has been re-rendered.

**Findings.**

| id | sev | finding | consequence | resolve by |
|---|---|---|---|---|
| TIME-1 | HIGH→MEDIUM (verifier: no non-test caller passes a wall-derived `now`; a dead feed produces no new minutes and therefore no new entries, so the consequence is a silent *hold*, not uncontrolled entries; recorder-host alerts exist) | The stale-feed veto is structurally vacuous: `risk.note_feed(minute_ns + MINUTE_NS, self.clock.now_ns)` compares the minute close to a clock just set to that same minute close (`runner.py:1076, 1091`), so delay ≡ 0; `MarketState.feed_age_ns` ≡ 0 (`feed.py:589-593`); `evaluate_entry` receives `data_delay_s=None`; `max_data_delay_s: 180` enforces nothing. A dead recorder with an open hedge is a silent indefinite hold: no veto, no halt, no funding booking, no liquidation check; the only wall-clock feed-age gauge exists during a pass. | The demo cannot detect its own feed dying. | R1 (operational staleness guard outside the decision path: recorder heartbeat age or newest-minute-vs-wall-clock in the CLI, halting with a named reason; two-sided test) |
| TIME-2 | HIGH→MEDIUM (verifier: constants and paths confirmed; the ≥ 40% figure is an unmeasured inference; skipped minutes are position-neutral and the first daily report would expose them) | Cadence mismatch: the recorder re-renders the open day's parquet on `NORMALIZE_INTERVAL_S = 300` stretched by a 10% duty cycle (`service.py:107, 717-720`); the runner decides only the newest 3 minutes and writes older pending minutes as `SKIPPED_STALE` (`runner.py:2320-2322`); as deployed (one pass per 30 s) new minutes arrive in batches of ≥ 5, so ≥ 40% of minutes are never decided or liquidation-checked, and `docs/replay_parity.md:92-95` concedes skipped minutes make later state diverge. | Replay parity is structurally unreachable on any day with a position; the evidence stream is mostly skip records. | R1 (daemon loop; incremental last-minute publication or a catch-up limit aligned to the producer cadence and hashed; a cadence-shaped test) |
| TIME-3 | LOW | The decision-log day file is keyed by `utc_day(runner_now_ns)` = minute close, so the 23:59 record lands in the next day's file (`decision_log.py:799`); reports and parity windows select by file name. | Off-by-one record at every midnight. | R1 |
| TIME-4 | LOW | Stale texts after PR #95: `docs/replay_parity.md:131-142` (Aegis day on the wall clock — false now), `runner.py:2518-2531` (`order_times`/`cooldown_until` called wall clock), `clock.py:10`; the risk-hash exclusion now hides deterministic rate-limit/cooldown state from parity. | Misdirected review; weaker parity than possible. | R1 (re-include the fields before any campaign log exists) |

**Replay determinism [RF].** Hashed inputs are the canonical `MarketState`, the risk snapshot minus excluded fields, the config hash (paths excluded) and the contract hash; canonical JSON sorts keys; Decimal money; sequence-based order ids; deterministic event ids; `software.{revision, source_digest, dirty, python}` is the only environment leak and is labelled `ENVIRONMENT_PARITY`. The clock does not resume from persisted state (`clock_now_ns` written, never read); `start()` seeds it from the cursor's next minute. **What parity proves today:** identical files + identical code ⇒ identical decision bytes, except across restarts (`seq` shift), skipped minutes and operator actions. It proves nothing about live latency, receipt-time effects or an external venue because none enter the decision path. There is no live mode; `run` and `replay` walk the same `FeedCursor`.

**Research as-of alignment [RF].** Sound: bars visible only at or after close; incomplete bars dropped; grid shift and boundary overrun refused; shift-detection controls at boundary rows.

---

## 11. Persistence / Crash / Recovery Review

**Files and atomicity [RF].** `spot_store.json`, `perp_store.json` (temp + fsync + replace), `carry_ledger.json` (plus directory fsync), `risk.json`, `runner_state.json` (`write_json_atomic`), `decision_log/YYYY-MM-DD.ndjson` (append + per-record fsync + hash chain), `KILL_SWITCH` (presence). No cross-file transaction; consistency is re-derived at start by triage (log vs runner state vs stores/ledger) and `reconstruct()` (stores vs ledger vs applied funding).

**Write order in one tick [RF]:** feed read → `note_feed` (risk persist) → funding (perp store save; ledger in memory; risk persist) → `_save_ledger` → FUNDING record → `save_state` → liquidation check → rule → plan → apply (per order: Aegis, `record_order`, transition, store save; fills; exposure persist) → reconcile-ledger in memory → optional reconciliation → mark-to-market → `_save_ledger` → `update_equity` (risk persist) → DECISION record → `save_state`. Log after state. On disagreement the LOG is authoritative for the chain head and the decided minute (LOG_AHEAD adopts both), the STORES are authoritative for position (LOG_BEHIND excludes the minute and re-ticks; re-tick is idempotent), and the LEDGER must be at least as advanced as the log (`ledger_behind_log`). Triage order: forged → torn → LOG_AHEAD → LOG_BEHIND, then `reconstruct()`; a halt wins.

**Crash matrix [RF; ✔ = idempotent/monotone/fail-closed and recoverable, ✘ = fail-closed but unrecoverable or inconsistent].**

| # | gap | result | verdict |
|---|---|---|---|
| 1 | order PLANNED/RISK_APPROVED saved → crash before submit | stranded order cancelled on recover | ✔ |
| 2 | SUBMITTED saved → crash before fill | no exposure; order left for reconciliation (which compares the store to a venue seeded from the store) | ✔ / note |
| 3 | perp leg filled and store saved → crash before spot leg or before `_save_ledger` | `ledger_store_mismatch` dispute → HALT, **no clearing path** | ✘ |
| 4 | both legs filled, stores saved → crash before `_save_ledger` | as 3 | ✘ |
| 5 | `_save_ledger` done → crash before `update_equity`/DECISION | LOG_BEHIND_STATE detected; RECOVERY record; minute excluded and re-ticked without re-execution | ✔ |
| 6 | DECISION appended → crash before `save_state` | LOG_AHEAD_OF_STATE; head and minute adopted from the log | ✔ |
| 7 | torn append | TORN_TAIL; bytes preserved; chain re-verified | ✔ |
| 8 | funding: perp store saved → crash before carry ledger save | `funding_booking_torn` dispute, **no clearing path** | ✘ |
| 9 | FUNDING record appended → crash before DECISION | minute re-ticked; settlement skipped via `settled` | ✔ |
| 10 | `risk.halt` persisted → crash before HALT record | restart re-halts and writes the record | ✔ |
| 11 | `risk.resume` persisted → crash before RESUME record | READY with no record (`runner.py:2100-2104`) | ✘ |
| 12 | LIQUIDATION_TOUCH + `save_state` → crash mid emergency reduce | dispute + halt; flatten still reduces | ✔ fail-closed, ✘ unrecoverable |
| 13 | executor `reconcile` mismatch persisted → crash before `note_reconciliation` | Aegis un-disputed; `reconstruct` ignores the store dispute; `require_ready` still blocks the next order | ✔ fail-closed, evidence inconsistent |
| 14 | disk full | `execution_error` halt; risk persist swallowed; HALT append swallowed; `save_state` swallowed → restart with no trace unless the cause recurs (`risk.py:540-541`; `runner.py:1990-1991`) | ✘ |
| 15 | any state file unreadable | store unbootstrapped and never overwritten; risk halted; ledger disputed; runner refuses; forged log never continued | ✔ |
| 16 | kill switch present | level-triggered halt at construction, start and every tick; persisted; a crash cannot clear it | ✔ |

**Findings.**

| id | sev | finding [RF] | resolve by |
|---|---|---|---|
| REC-1 | HIGH→MEDIUM (verifier: fully disclosed; the runbook §6 blockquote records that `start()` followed by `resume()` succeeds on the fixture, so the CLI repair is small) | No documented operator command can leave a persisted HALT or clear a dispute: `resume`/`resolve` build a runner without `start()` and always refuse (`tools/demo_run.py:230-248`; `runner.py:2077, 829-843`); every CAMPAIGN run halts at SELF_CHECK because `_software()` calls `source_identity()` without its required `root` and reads a dict with `getattr` (`tools/demo_run.py:164-184`; `nn/source_identity.py:153`); CLI `flatten` on CAMPAIGN persists a spurious Aegis halt (`allow_dirty`). Disclosed in the runbook as "open S3 blockers"; deferred by `runner.py:281-289`; unscheduled in v3. | R1 |
| REC-2 | HIGH→MEDIUM (verifier: the windows are milliseconds and execution minutes are rare; a "re-derivation resolve" would be exactly the silent overwrite the code refuses) | Crashes in the ordinary execution and funding windows (rows 3, 4, 8) produce disputes with no clearing path and end a campaign permanently; no SIGTERM handler exists, so `systemctl stop` inside an execution minute is such a crash. | R1 (SIGTERM deferral past PERSISTENCE; save the carry ledger immediately after each leg's store save to shrink the window; an explicit protocol rule that a crash-affected block is judged only by the frozen validity evaluator (INVALID if the halt is not recovered within the frozen window or lacks a record; never a negative result; never substituted), the campaign extending per the frozen rule; any operator resolve must display the persisted levels and record a written reason — never a silent re-derivation) |
| REC-3 | MEDIUM | The recorder writes the normalised parquet in place (`normalize.py:898`, unlike every other writer), and a torn read becomes a persisted, CLI-unclearable `feed_unreadable` halt (`runner.py:1079-1089`; `RestartPreventExitStatus=3`). | R1 (temp + fsync + replace; retry transient read failures) |
| REC-4 | MEDIUM→LOW (fold into REC-1's CLI repair) | `resume()` clears the durable halt before the RESUME record is appended and without `_require_recordable` (row 11). | R1 |
| REC-5 | MEDIUM→LOW | Reconciliation is self-referential in dry-run (store vs a venue seeded from the store); a persisted store-level dispute is not re-raised by `reconstruct()`; the RECONCILIATION evidence kind attests internal consistency only. | label the record with its comparison basis now; real-venue requirement in R14 |
| REC-6 | LOW | Disk-full behaviour is fail-silent for the halt reason; the fault module models no crash-at-boundary, forced-restart or disk-full faults (master plan line 1584 lists them); crash windows are covered ad hoc in `tests/test_demo_runner.py`. | R1 (systematic harness) |

**Verifier-raised findings.**
- (accounting verifier, MEDIUM, [INF]) Live-versus-replay funding-booking minute can diverge at every settlement: the runner books a settlement at instant T in the tick of the minute that closes at T (`runner.py:1530`), but in live operation the pass may process that minute before the recorder has written the settlement row (written ~60 s after the instant), so live books it one minute later than replay. Resolve in R1 together with the cadence fix (a minute is decidable only once its settlement row, if any, exists).
- (clocks verifier, LOW, [RF]) A down recorder leaves no evidence record at all, contrary to the systemd unit's comment that it "reports INCOMPLETE_STATE": `catch_up()` returns `[]` with no record when no newer minute exists (`runner.py:2302-2318`). Resolve in R1 by writing a `FEED_STALLED` operational record from the staleness guard.

**Assessment [INF].** The persistence and evidence core is a KEEP: it is more careful than most production systems. The operational shell (operator lifecycle, dispute clearing, staleness, cadence) is where the demo fails, and none of it is scheduled by v3.


---

## 12. Recorder / Data Architecture Review

Standing: the dedicated recorder/PR #76/gen4 subagent did not complete inside the usage window; this section rests on the lead's reading of `chimera/recorder/*` on `main`, the read-only extract of PR #76, the gen3 contract, the clocks subagent's recorder findings (§10), and the external-documentation subagent's archive facts [EF].

**Recorder on `main` [RF].** One asyncio websocket client per venue endpoint (USD-M market, USD-M public, spot on `data-stream.binance.vision`), exponential backoff 1–60 s with jitter, ping/pong 20 s, proactive reconnect at 23 h 50 m, per-stream reconnect counters, a rolling clock-skew estimator; REST pollers for funding; an append-only NDJSON.gz raw sink per stream-day with atomic writes and a late-event file; a minute normaliser (last closed kline frame wins with conflicts recorded; last book before the close; mark open/high/low/close; per-day parquet plus meta with digest and gap list); an incremental normaliser with a sealed cache for the open day (rebuildable, never evidence); a service with heartbeat every 30 s, per-stream last-event age, missing-minute counters and disk-free metrics; health-gated streams (`um.bookTicker`, `spot.bookTicker`) outside `required_for_coverage`; a contract that hashes canonical material (endpoints are code constants, not contract fields), keeps `prospective_from: null`, inherits the P4-HOLD and Styx seals, and defines activation as a reviewed commit writing to a fresh storage root. Alerts exist for stale streams (> 180 s), recorder down (> 120 s heartbeat), data gaps (> 7 missing minutes), clock skew (> 5 s) and disk low — on the recorder's own host.

**Attack [RF/INF].**
- *Websocket reliance and silent stalls:* the client counts reconnects on errors and closes; a socket that stays open and delivers nothing (the A3 retired-base failure) is detected only by the Prometheus alert on `last_event_age`, not by an in-process watchdog that reconnects. → R2: in-process silence watchdog per stream.
- *REST recovery:* kline gap-fill after reconnect is designed; spot REST gap-fill is not asserted to work (A11 notes `api.binance.com` returns 451 from that environment). Provenance distinguishes a REST-filled minute from a live one only if the raw record says so — the raw layer stores the source stream, so yes, but the normalised row does not carry a `source` flag. → R4: per-minute provenance flag.
- *Gap semantics:* a minute with no closed kline has no row (never interpolated) — correct; downstream consumers must treat "no row" as missing, which the feed does (`complete = False`).
- *In-place parquet write:* `frame.to_parquet(final_path)` (`normalize.py:898`) is the one non-atomic writer (§11 REC-3).
- *Schema versioning:* raw event schema and parser versions are not stamped per file; a parser change would alter normalised output without changing the contract hash; the contract hashes streams and semantics but not the code version. → R4: bind contract hash + code version + schema version in every day manifest.
- *Clock synchronisation:* receipt wall and monotonic clocks are persisted; skew is measured and alerted, never applied; no chrony requirement is documented.
- *Cross-stream consistency:* none computed (kline vs aggTrade OHLCV; bookTicker vs depth top; markPrice vs markPriceKlines) — PR #76 adds mark reconciliation only.
- *Duplicates and sequence gaps:* kline frames deduplicated by the closed flag; bookTicker by update id with out-of-order counted; no depth stream, so no `U/u/pu` continuity logic exists — required before any Tier B capture.
- *Partial minutes at start/stop and day rollover:* handled by canonical-day file naming and late files; the last minute of the day is filed under the next day by the runner, not the recorder (§10 TIME-3).

**PR #76, read-only [RF].** `reconcile.py` (1,895 lines) and `coverage.py` (860 lines) add: one allow-listed host; `.CHECKSUM` companion fetched every time, filename validated, SHA-256 recomputed over received bytes; redirects refused; exactly one ZIP member; per-layout epoch units declared per file and cross-checked against the object's day/month; ten `ArchiveOutcome` values none of which becomes an empty published set; a full-path-keyed cache re-verified against the venue's digest on every run; minute-by-minute comparison at 1e-9 relative tolerance with zero compared exactly; no recorder file opened for writing (asserted, including against an idempotent rewrite); integer-arithmetic thresholds (`agreeing × 1000 ≥ published × 995`; wall-clock `< 990`); funding equality on settlement time and rate only; `schedule_established` false when the monthly object is absent, unverified, unparseable or covers only part of the day → `FUNDING_SCHEDULE_UNAVAILABLE`, which does not pass and is not an outage; the qualifying streak as the maximal run of consecutive passing days at or after the boundary; index reconciliation reported but gating nothing; bookTicker absent from the archive gate. Design verdict: **sound and fail-closed**; the three BLOCKERs its review found (streak measured only backwards; the fourth `schedule_established` clause unimplemented; urllib redirect-following) are fixed in the head. Integration debt: base `95322a99`, 129 commits behind; `tests/test_recorder_no_network.py` changed on both sides; A11's spot transport change and PR-13's disconnection landed after the base; the tree must be merged on a fresh lineage with an ordinary merge. Acceptance design: two real days reconciled against published archives; daily objects land ~07:00–08:15 UTC on D+1 and the monthly funding object after month end [EF], so a two-day acceptance can be minute-stream-complete on D+2 but funding-`UNJUDGEABLE` until the month closes — which is exactly what campaign #1's day one showed; day two failed on coverage (`RECORDER_OUTAGE`), i.e. the recorder itself, not the archive, failed — the strongest argument for the R2 preflight hardening (silence watchdog, disk/CPU headroom, supervision drills) before any campaign #2.

**Archive availability that changes the gen4 design [EF, fetched 2026-09-13 by the external-documentation subagent].** On `data.binance.vision` under `futures/um`: `aggTrades`, `bookDepth`, `bookTicker`, `indexPriceKlines`, `klines`, `markPriceKlines`, `metrics`, `premiumIndexKlines`, `trades` exist daily; `fundingRate` monthly only; **`bookTicker` daily objects end 2024-03-30 and monthly 2024-04** for BTCUSDT/ETHUSDT/SOLUSDT; there is **no `liquidationSnapshot`** under `um/daily`; `bookDepth` is a sampled band file (~30 s cadence, ±1–5% bands), not L2; `metrics` is 5-minute; historical objects are occasionally re-uploaded with new `LastModified` and the README says archives may be updated. Consequences: bookTicker and any depth capture are HEALTH_GATED for a 2026 campaign (the gen3 contract already says so — confirmed externally); OI and long/short-ratio REST histories keep ~30 days and must be captured live; `forceOrder` pushes only the largest liquidation per symbol per second (since 2026-04-10) and `aggTrade` excludes insurance-fund and ADL trades, so neither is a complete record; reconciliation artifacts must store the verified digest with the object's `ETag`/`LastModified` so a re-upload reads as an archive event, not tampering.

**Gen4 answers (Part IV) [REC].**
- *Record from day one (Tier A, all core symbols):* 1 m klines, `markPrice@1s` (mark, index, estimated settle, funding rate, next funding time), `bookTicker`, `aggTrade`, funding (REST), OI and long/short ratios (REST, 5-min), `forceOrder` (descriptive), daily `exchangeInfo`/leverage-bracket/fee-assumption snapshots (Tier C). *Tier B (BTCUSDT, ETHUSDT):* `depth20@100ms` or diff depth with REST snapshot bootstrap and `U/u/pu` continuity, health-gated, raw only (no reconstruction in the recorder). *Do not record:* raw `trades` (aggTrades suffice), anything for symbols outside the frozen candidate list.
- *Add later without contaminating evidence:* any stream added after a contract freeze is usable only by campaigns whose contract includes it; adding it does not invalidate a running campaign; it goes to a new contract id and, if a campaign wants it, a new boundary.
- *Streams requiring a new boundary if added later:* any stream the campaign's feature function or label reads (klines, markPrice, bookTicker, aggTrades, funding); descriptive streams (forceOrder, OI, depth) do not.
- *Verification classes:* ARCHIVE_RECONCILED — klines, markPriceKlines, indexPriceKlines, premiumIndexKlines (derived from markPrice), aggTrades, fundingRate (monthly, with the latency rule), metrics (OI, 5-min); HEALTH_GATED — bookTicker, depth, markPrice@1s itself (sequence/age/consistency against markPriceKlines); DESCRIPTIVE_ONLY — forceOrder, exchangeInfo snapshots.
- *Data-quality gates before any campaign:* per-stream coverage ≥ 99.5% of published minutes for reconciled streams over ≥ 30 consecutive days; ≥ 99.0% wall-clock minutes for health-gated streams with `last_event_age` p99 < 60 s; depth sequence-continuity ≥ 99.9% of updates with reconnect recovery drilled; clock skew median < 1 s, p99 < 5 s; cross-stream consistency (kline OHLCV vs aggTrade-derived OHLCV within tick tolerance on ≥ 99.9% of minutes; bookTicker mid within 0.1% of kline close at the minute close on ≥ 99.9%); duplicate and out-of-order rates reported; zero unexplained `RECORDER_OUTAGE` days in the qualifying period; archive reconciliation agreement for every reconcilable stream on every published day.
- *Preflight duration:* ≥ 30 consecutive days on the production host (separate from the PR #76 VPS), including at least one exchange maintenance window and one funding-rate extreme; completeness measured with per-stream denominators (published minutes for reconciled streams; wall-clock minutes and health proxies for gated streams); the same 30-day gate re-run on the production recorder identity before R6 activation.
- *Raw/normalised separation and binding:* raw NDJSON.gz per (symbol, stream, day) immutable with SHA-256; normalised parquet regenerable and versioned by `(contract_hash, recorder_code_version, schema_version, normaliser_version)` stamped in every day manifest; a normaliser change bumps the version and regenerates, never edits, evidence.
- *Storage [INF, to be measured in R2]:* bookTicker dominates (gen3 measured ~37 GiB/month gzipped for one symbol's two books); aggTrades for BTCUSDT ≈ 1–3 M events/day (≈ 100–300 MB/day raw); depth20@100ms ≈ 0.9 M messages/day per symbol; 15 core + 2 depth symbols plausibly 0.5–1.5 TB/year gzipped — provision headroom and measure before freezing.


---

## 13. Research-Methodology Review

**The central research question, as the repository actually asks it [RF/INF].** Four incompatible questions coexist: (1) the historical programme's "does information set X improve net return after a flat 20 bps cost in ≥ 3 of 4 folds under a cost-aware three-class label and an inner-selected threshold" (an economic fold-count gate on a classifier); (2) P13's "does an always-on delta-hedged carry earn robust net returns across calendar blocks" (a structural payoff, never run); (3) Roadmap v3 §9's "does a predeclared causal information set carry prospective predictive or decision information beyond baseline, large enough to retain economic value after measured costs" on a panel (information + economics co-primary, endpoint left open among conditional executable return, cost-aware calibration, time-series IC, cross-sectional rank IC); (4) the runtime's only actionable strategy, a two-leg carry rule, while the roadmap's primary lane is single-leg directional. No document states the bridge from any information endpoint to a target position, and no document requires endpoint, policy and runtime to be the same object. §§14, 15 and 27 fix one question per lane and add the coherence rule.

**Adaptivity map [RF].** One asset, one venue, four temporal blocks (2023-03-04 to 2025-05-19) of 4,821 hourly rows each, fold k's inner block being fold k−1's outer block; ~90 fitted cells; ~45 deciding and ~14 secondary gate applications; 12 readings; preregistration-to-evidence margins of 4 minutes to 4 h 39 m; no independent pre-open review before 2026-08-30; the OHLCV14 control cell moving from −0.043 to +0.429 (fold 3, XGBoost) and −0.026 to −0.542 (fold 2, LightGBM) under row-eligibility changes alone. The prior audit's five load-bearing findings are confirmed against the repository: the instrument cannot see the effects searched for; the flat 20 bps foreclosed fast clocks by arithmetic; every economic number priced a SHORT leg spot cannot hold; the four blocks are spent; the operational architecture was built for an ensemble that does not exist (now disconnected).

**Evidence-governance mechanisms, as enforced [RF].** Preregistration payload hashing with superseded-hash chains and pre-number amendments (P4: four; P13: A1/A2/A2R1/A2R2); `artifacts/*SHA256SUMS.txt` verified in CI; `tools.verify_research_state` regenerating the state table from artifacts and failing CI on prose/artifact disagreement; `tests/test_sealed_boundary.py` making Styx an instant; Styx rows physically absent; negative results and thin trade counts pinned by tests (`test_p6_evidence.py` names the eight sub-10-trade cell-folds); `would_have_passed` secondary leads recorded rather than hidden; the P13 gate as the one template with a sample floor and an effect floor. **Procedural rather than physical:** P4-HOLD rows are present in the committed raw parquet and sealed by ledger and code path; the "dirty tree refusal" is enforced in the demo's SELF_CHECK (currently for the wrong reason) but not in the research runners (P6 fits ran dirty and were accepted); evidence classes exist as labels, not as a machine-checkable ladder.

**Fitness for a prospective campaign [INF].** Historical preregistration guaranteed *ordering*. A prospective campaign needs, in addition: an activation commit that stamps `prospective_from` and a fresh storage root (present in the gen3 design); a no-peek rule during accrual; an interim-look policy; a mechanical, outcome-blind block-validity rule with a frozen target of VALID blocks, a frozen extension cap and a NOT EVALUABLE outcome (the engineering-invalidation versus scientific-failure split made non-adaptive); no retroactive substitution of blocks; deciding economic semantics frozen before the boundary and identical in confirmation; a minimum elapsed time and an independent review between preregistration and first scored day; a mandatory historical-adaptivity disclosure in the preregistration; and a machine-readable evidence class on every artifact so exploratory work cannot be promoted by prose. Roadmap v3 has the first and requires the review; it lacks the rest. §37's governance classes and Git policy add them.

**Prior-audit follow-through [RF].** Implemented: recorder, demo collapse, dependency lock, Freqtrade disconnection, `pandas<3`, P14 declined, P8 withdrawn, burned blocks closed, per-instrument fee parameters in config. Superseded by v3 by owner decision, with reasons recorded: BTC-only → multi-symbol (statistical breadth); carry-spine → directional-first (owner objective; the carry argument was outranked, not refuted). Quietly reintroduced: a *mandatory* hierarchical MTC representation after P5/P6/P7 (§16). Still open: independent pre-open review with a minimum elapsed time; effect-size floor and dependence-aware test (required in v3 §9.5, unimplemented); the funding halt and feed-staleness halt (present in config, unreachable in code — §§7, 10).

**Verdict [INF].** Strict about ordering and honesty, not yet strict about statistics or about the prospective mechanics; the first can be preserved unchanged, the second is R3.

---

## 14. Futures-First Adjudication

**Question.** Should the primary direction remain USD-M perpetual futures, LONG/SHORT, as adopted in v3 §1?

**Alternatives weighed [INF unless marked].**

| direction | for | against | engineering cost from HEAD |
|---|---|---|---|
| Spot directional | simplest instrument; no funding, no liquidation; deepest history | no SHORT (every historical economic number priced a SHORT leg spot cannot hold — prior audit finding 3, unrefuted [RF]); taker fee ~2× the perp's; the only spot executor is the retired Freqtrade path | high (rebuild a spot runtime or resurrect Freqtrade) |
| Spot + hedge | removes market beta of a spot book | needs the perp anyway; two-venue reconciliation | as carry |
| **Perp directional LONG/SHORT (adopted)** | SHORT native; lower taker fee; mark/index/funding/bookTicker archives; validated dry-run executor; Aegis integration exists | funding cost inside every hold; liquidation semantics (negligible at 1×); the current runtime is carry-shaped, so a *single-leg multi-symbol* runtime still has to be built | medium (R8) |
| Funding/carry (spot long / perp short) | the only mechanism with an observable payoff (funding settles every 8 h by default, i.e. 3×/day, with some symbols on 4 h or 1 h intervals [EF]); low degrees of freedom; two-leg runtime **exists** [RF]; economic evidence reachable in months | capacity-limited; regime-dependent (funding sign flips); pays two legs' fees plus a base-asset spot fee not modelled [RF §9]; teaches nothing about prediction | low (R1 fixes only) |
| Relative value / basis | as carry, more variants | more degrees of freedom; venue-specific | medium |
| Cross-sectional market-neutral panel | removes the common crypto factor, which is what makes a panel worth ≈ 14 rather than ≈ 2 effective symbols [INF, §22]; LONG/SHORT natural on perps | doubles trades and costs versus a single-leg book; needs simultaneous multi-symbol execution | medium (same R8 runtime, portfolio caps in Aegis) |
| Slower (daily) horizon | costs become second-order | ~2 years to power anything [INF, §22] | none extra |
| Maker microstructure | the only way sub-hour horizons survive costs | needs depth recorder, queue/adverse-selection model, a different project | high |

**Decision: KEEP FUTURES-FIRST as the instrument; MODIFY the strategy family; ADD the carry mechanism as a parallel low-cost economic lane. [REC]**

1. *Instrument:* the Binance USD-M perpetual is the researched and executed instrument for every directional claim. Spot is a hedge leg and reference data, not a first lane. Spot scalping is dropped from the roadmap.
2. *First strategy family (one hypothesis, not a shape):* "Price-based state on a liquid-perpetual panel — cross-sectional relative return (momentum/reversal) and time-series trend and realised-volatility state at 1 h–1 d scales — carries information about vol-standardised executable 4–8 h forward returns beyond a no-information baseline, and a low-turnover LONG/SHORT policy built on it clears measured taker costs." The cross-sectional component is primary because it is the only endpoint the first window can power (§22); the panel of single-leg positions it implies is exactly what the R8 runtime executes.
3. *Carry lane (SECONDARY, OPTIONAL):* a separately preregistered, always-on or funding-gated delta-hedged carry rule on BTCUSDT (and possibly ETHUSDT) in dry-run on the existing two-leg runtime, as its own process, state directory, Aegis instance, ledger and dry-run capital. It is the only candidate whose *economic* evidence is reachable inside the first accrual window (≈ 546 settlements per symbol in 182 days). It is not orchestration; it is a second independent dry-run campaign with its own single hypothesis slot. It must not delay R1–R9, is not a prerequisite for the primary lane, does not replace futures-first and receives no priority because its evidence may arrive sooner; if resources conflict, the primary directional futures lane wins. Election is an owner decision at R0 with a deadline at R4 (its spot streams must be in the gen4 contract). If the owner declines it, nothing else in the roadmap changes.
4. *Why not carry as the spine (the prior audit's design)?* The owner's fixed objective is a directional futures system (v3 §1.1), and the directional lane needs the multi-symbol runtime built regardless. Running carry *instead* would defer that build; running it *alongside* costs little because its runtime exists, and it is permitted only while it costs the primary lane nothing material.

**Evidence that would reverse this later.** (a) Preflight shows the perpetual data or venue semantics are unreliable (coverage, funding cadence anomalies) — then spot becomes the reference instrument and the directional lane pauses. (b) The first campaign shows the funding term dominates the directional economics (e.g., the economic screen fails only because of funding) — then the family is redesigned around funding-aware holds, not the instrument. (c) Both the directional and carry lanes return prospective negatives — then "no deployable alpha under the current mandate" is recorded and any new mandate (maker execution, breadth beyond crypto) gets its own budget.

**Coherence rule adopted [REC].** Endpoint ↔ policy ↔ runtime must match by construction: a cross-sectional endpoint implies a multi-symbol single-leg runtime with portfolio caps; a time-series endpoint implies per-symbol single-leg positions; a carry endpoint implies the two-leg runtime. The preregistration states which; the runtime that will execute it must have passed its soak (R9) before the boundary is activated, and R9 PASS is a recorded input to R6.

---

## 15. Target / Horizon / Frequency Adjudication

**What exists [RF].** One label family: three-class direction on the close-to-close return over 6 bars with a ±20 bps dead zone equal to 2×(5 bps fee + 5 bps slippage), baked into `TargetSpec` (`chimera/contracts.py:54-64`, `nn/data_pipeline.py:207-224`); fill assumed at the decision bar's own close with zero latency and an additive flat 20 bps (`nn/evaluate.py:42-43, 241`); threshold chosen on the inner block from a 29-point grid maximising compounded net return with a 10-trade floor (`nn/evaluate.py:917-968`); positions are −1/0/+1; no sizing; calibration reported only as a 10-bin top-label ECE. The runtime fills at the recorded ask/bid plus slippage at or after the decision (`chimera/futures/fills.py:12-25`) — researched timing ≠ executable timing. The "retired" 20 bps cost still lives inside the label.

**Adjudication [REC].**

| question | decision | reason |
|---|---|---|
| Is classification still appropriate as the primary target? | **No — REPLACE.** Primary target = vol-standardised executable forward return over H: r̃ = (P_exit − P_entry)/(P_entry·σ̂_t) − c_rt/σ̂_t − funding over the hold (A10 sign), with P_entry = the first recorded top-of-book on the crossing side at decision instant + a predeclared latency budget, P_exit likewise at t+H. Secondary = sign of the same executable return with one analytic threshold. | A target-position system with abstention and sizing needs E[r|x]; classification discards magnitude and makes the threshold the alpha; the executable definition removes the same-close optimism and puts funding inside the instrument. |
| Cross-sectional or time-series? | Primary endpoint cross-sectional rank IC of r̃ across the panel per decision; secondary time-series IC (m = 2, Holm). | Power (§22). |
| Horizon | **Frozen by a pre-result rule, not chosen now.** Compute κ_H = c_rt/σ_H per symbol from preflight data for H ∈ {1h, 2h, 4h, 8h, 24h}; choose the shortest H whose panel-median κ_H ≤ κ* (predeclare κ* ≈ 0.10–0.15); prefer H aligned to the 8 h funding cadence. Under taker costs and BTC-like volatility this lands at 4–8 h [INF]; 1 h is marginal; sub-hour is foreclosed until maker costs are measured. | Both the lead's and the subagent's arithmetic put κ_4h ≈ 0.12 and κ_6h ≈ 0.095 at 14 bps round trip and 0.6%/h volatility. |
| Decision frequency | 1 h decision clock with overlapping labels modelled (N_eff ≈ N/H) or decisions every H bars; the preregistration picks one. | Overlap is a known ~H× inflation of nominal N (§21). |
| Turnover | The policy that maps scores to target positions carries a hysteresis band frozen in the preregistration so that the expected holding period is ≥ 2 days at a 1 h clock. | Turnover, not horizon, is what makes taker costs survivable (§23). |
| Threshold / abstention | **REPLACE the grid.** Trade iff |E[r̃|score]| > k·c_rt with k predeclared (e.g. 1.5), or a single preregistered p*; coverage sensitivity reported as exploratory only; never re-selected during accrual. | The grid is the largest hidden degree of freedom in the historical programme (§21). |
| Calibration | Drop as an endpoint. If probabilities size positions, freeze an isotonic/Platt map on pre-boundary engineering data and report Brier skill and reliability slope on the traded region with block-bootstrap intervals. | ECE is non-proper and bin-dependent [RF]. |
| Funding inside the label | Yes, for holds crossing a settlement, with the A10 sign convention. | Instrument = executed instrument (constraint 37). |
| Consistency with execution | The recorder must capture the quotes the label needs at decision instants for every core symbol (bookTicker is already recorded; aggTrades added in gen4); the runtime's fill model and the label's price rule are the same function. | Otherwise the campaign IC is not the IC of the executable position [RF §21]. |

**What stays as history [RF].** `compute_target`, the dead-zone label and the 29-point grid remain the historical definitions for reproducing frozen artifacts; they are not to be reused by the campaign evaluator.

---

## 16. MTF Adjudication

**Evidence [RF].** P5 (closed 4h/1d bars as columns on a 1 h model) negative; P6/P6-EXT (native specialists) negative under the deciding family; P7 (consensus) negative against an oracle benchmark with 13 trades in one mode and unbounded staleness — all measured with the instrument of §21. The as-of machinery is sound: `as_of = searchsorted(close_times, t, side="right") − 1` plus a shift-detection control at boundary rows and a count of rows reading a bar that had not closed (`nn/mtf.py:13-35, 258-260, 368-520`); bars are cut from the 1 m grid only when every constituent minute exists (`nn/multiclock.py:246-247`); the runtime's `MarketState` already carries `minute_ns`, `complete`, `missing`, `feed_age_ns` (`chimera/demo/feed.py:162-234`). No MarketContext/MTC fabric exists in `chimera/`.

**Attack on the v3 §7 design [INF].** (1) Unfinished-bar leakage is guarded in research; in a live runtime it depends on the recorder's `x == true` flag and on an explicit `available_time` that `MarketState` does not yet carry. (2) Three time domains (candle close, exchange event time, receipt) must be named per input. (3) `nn/regime.py` statistics are computed over scored rows and are hindsight; they must never be features. (4) Degrees of freedom: clocks × state variables × coherence definitions × staleness rules multiply into hundreds of design points before a model; v3 forbids voting (good) but "represent contradiction explicitly" is itself a feature-engineering family. (5) The narrative "context → setup → timing → execution" reduces to "slow-scale state conditions the short-horizon return distribution" — a two-feature claim.

**Decision: SIMPLIFY and DEFER hierarchical MTC as a scientific model; RETAIN the causal machinery; REQUIRE multi-timeframe capability in the runtime. [REC]** MULTI-TIMEFRAME CAPABILITY = REQUIRED ENGINEERING PROPERTY. HIERARCHICAL MTC AS A SCIENTIFIC MODEL = NOT REQUIRED FOR CAMPAIGN 1. V3-7 is re-scoped to a generic multi-clock causal `MarketSnapshot` (R7): for every input and every contract-declared timeframe it carries `source_time`, `available_time`, `as_of`, `age`, `staleness`, `complete`, `missing`, and it derives multiple closed-bar clocks causally from the base 1 m stream, so the runtime is not architecturally limited to one timeframe; it is tested with two-sided synthetic leak controls (a planted future leak must be detected; a clean series must pass; the `nn/mtf.py` shift control is the template). The first campaign's deciding feature set is restricted to the preregistered decision clock plus at most two preregistered slow-scale, closed-bar, explicitly aged state variables (1 d realised volatility; 1 d or 4 h trend sign); every other available clock is capability only and creates no scientific degree of freedom during R10. No hierarchy, no coherence state, no regime labels, no voting, no "HTF always wins", no mandatory multi-layer narrative state. A richer hierarchy returns only under a new preregistered campaign after the first prospective result warrants it. v3 §21's "causal Multi-Timeframe Coherence remains mandatory" should be revised to "causal as-of semantics and multi-clock capability remain mandatory; hierarchical coherence does not".

---

## 17. Microstructure Adjudication

**Recorder capability today [RF].** Streams: `um.kline_1m`, `um.markPrice@1s` (mark, index, estimated settle, funding rate, next funding time), `um.bookTicker`, `um.funding` (REST), `spot.kline_1m`, `spot.bookTicker`. No aggTrade, depth, forceOrder or open interest. Measured raw rate ≈ 37 GiB/month gzipped, 99.4% of it bookTicker (master plan S1 row, amended A3).

**Classification [INF; first campaign at a 4–8 h horizon; execution measurement separate].**

| stream / signal | predictive use, campaign 1 | execution use (demo, economic campaign) | verdict |
|---|---|---|---|
| aggTrades | OPTIONAL (hourly aggregation negative in P3; native scale untested) | ESSENTIAL: realised spread, trade-through, volume-at-price for fill/impact calibration | record day one, Tier A, archive-reconcilable |
| raw trades | NOT JUSTIFIED (aggTrades suffice) | — | do not record |
| bookTicker | OPTIONAL as a spread feature | ESSENTIAL: fills, spread, label prices, staleness | keep; health-gated |
| L2 partial depth (depth5/20) | PREMATURE | HIGH-VALUE for 2 symbols: impact and queue calibration | Tier B day one, ≤ 3 symbols, descriptive/health-gated |
| L2 diffs @100 ms + snapshot bootstrap | PREMATURE | OPTIONAL (maker-execution mandate only) | defer unless preflight shows it is cheap; sequence-continuity gate required |
| depth imbalance / OFI / microprice / trade sign / aggressor flow | NOT JUSTIFIED at 4–8 h | HIGH-VALUE later for execution timing | derive later from frozen raw streams; no new boundary needed if raw was in the contract |
| queue state | PREMATURE | PREMATURE | — |
| liquidations (forceOrder) | OPTIONAL | HIGH-VALUE cheap context | record day one, DESCRIPTIVE_ONLY |
| open interest (REST) | OPTIONAL (P4 negative at 6 h) | HIGH-VALUE cheap | record day one, health-gated |
| basis / premium index | ESSENTIAL (cost, funding expectation) | ESSENTIAL | day one (inside markPrice stream) |
| funding | ESSENTIAL (cost inside horizon) | ESSENTIAL | day one, monthly-archive reconciled |

**Archive facts that bind the microstructure tier [EF, fetched 2026-09-13; see §12].** The `data.binance.vision` USD-M archive publishes `bookTicker` daily objects only up to 2024-03-30 (monthly to 2024-04) for the symbols checked, publishes no `liquidationSnapshot`, publishes `bookDepth` as a sampled band file at roughly 30-second cadence (±1–5% bands, not L2), and publishes `metrics` at 5-minute cadence. Consequently no first-party minute denominator exists for quote or depth capture over a 2026 campaign: quote/depth streams are HEALTH_GATED (self-attested with health metrics), never ARCHIVE_RECONCILED; `forceOrder` carries only the largest liquidation per symbol per 1000 ms and `aggTrade` excludes insurance-fund/ADL trades, so neither may be preregistered as a count. A gen4 contract that assumes a quote archive denominator, or a feature built on liquidation counts, is invalid by construction.

Predictive and execution datasets differ and should be labelled as such in the contract: campaign 1's predictive set is closed klines (any clock), mark/index/funding and cross-symbol returns; its execution set is bookTicker + aggTrades (+ depth for ≤ 3 symbols) used only to measure costs and fills. The microstructure tier exists first to *measure execution*, not to find alpha; a maker/scalping mandate is a separate project with its own budget.

---

## 18. Universe / Symbol Selection Adjudication

**What v3 says [RF].** "Roughly 12–20 liquid USD-M perpetuals" and "2–3 microstructure symbols", final count after measurement (§5.2, §6).

**Attack [INF].** A number before a measurement is a guess; storage was already mis-estimated 24× once. Pairwise return correlations among large-cap perps are commonly 0.6–0.9, so the *effective* number of independent time-series symbols is ≈ 1.4–3 (equicorrelation N/(1+(N−1)ρ), N = 15, ρ = 0.7 → 1.4; ρ = 0.3 → 2.9); breadth helps a time-series endpoint little and a cross-sectional endpoint a lot (≈ N − 1 after demeaning if residual correlations are small). Beyond ~10–12 names the marginal ESS gain shrinks while operational cost, multiplicity temptation and delisting risk grow.

**Procedure adopted [REC].** Frozen before the boundary, from pre-boundary data only: rank USD-M perpetuals by 90-day median daily USD volume as of a named pre-boundary date; require listing age ≥ 12 months, uniform 8 h funding interval (symbols on 4 h/1 h intervals are excluded from campaign 1 to keep cost semantics uniform; Binance publishes an 8 h default with per-symbol 4 h/1 h exceptions [EF]), sane tick/step/minNotional filters, no leverage-bracket anomalies; take the top K. K is fixed by the preflight: compute the effective number of symbols from the correlation matrix of per-symbol score×return products (participation ratio (Σλ)²/Σλ²) and choose the smallest K at which the effective number stops growing materially; provisional K = 10–12 [INF]. No refresh during a campaign; a delisting or trading halt → forced flat, the symbol's block is invalidated (engineering invalidation, not a negative); a universe change is a new boundary. Depth tier: BTCUSDT and ETHUSDT; a third only if preflight shows the marginal cost is trivial. Survivorship is avoided by construction (selection date precedes the boundary; the list is hashed).

---

## 19. Feature-Family Adjudication

| family | class | basis [RF unless marked] |
|---|---|---|
| OHLCV14 (`chimera/features.py`) | **foundational control** | causal, scale-free, prefix-invariance tested; mean-zero after costs everywhere |
| realised volatility / range / volume, closed-bar slow variants | foundational | mechanistic (sizing, vol scaling) |
| cross-symbol market factor (panel mean return, beta) | **incremental, high value on a panel** | mechanistic; absent today (single asset) [INF] |
| funding rate / basis (1–2 features) | incremental | P4 negative as *prediction* at 6 h; still a cost term inside any hold crossing a settlement |
| SMC (`nn/smc.py`, 39 columns) | scientifically weak / fashionable vocabulary | P2b negative; ablation: only `liquidity` marginally positive; heavy overlap with OHLCV14 |
| chart structure (30) | scientifically weak | P2c negative; every mean delta negative |
| hourly microstructure (32) | redundant at multi-hour horizons | P3 negative at 1 h aggregation |
| HTF context as columns (`mtf_v1`, 28) | redundant / incremental | P5 negative; keep ≤ 2 slow variables |
| regime labels (`nn/regime.py`) | **high overfit risk (hindsight)** | diagnostic over scored rows; never a feature |
| technical indicators beyond OHLCV14 | redundant | — |

**Frozen set for campaign 1 [REC]:** OHLCV14 at the decision clock; ≤ 2 slow closed-bar state variables; 1 cross-sectional market-return feature (panel demeaning is part of the endpoint anyway); 1 funding/basis feature. Everything else is exploratory and labelled. The "information set" the campaign tests is the cross-sectional/slow-state block relative to the same candidate without it (the negative control is the candidate with that block zeroed).

---

## 20. Model-Ladder Adjudication

**History [RF].** MTST transformer (72k parameters on ~300–550 non-overlapping windows per fold; lost to buy-and-hold 20/20) → untuned LR/LightGBM/XGBoost on flattened windows → XGBoost named deciding family → P7 replays. Roadmap v3's ladder (null → regularised linear → one tree challenger → deep only if earned) exists in prose only; no Tier-1 candidate, coefficient-freeze or refit-schedule code exists; the only experiment grid tooling targets the retired MTST (`nn/experiment.py:74-82`).

**Decision [REC].**
- Tier 0: no-signal (always abstain) and one deterministic reference rule (time-series momentum sign), frozen.
- Tier 1 (deciding): one L2/elastic-net linear model on the vol-standardised executable target with the frozen feature set; λ chosen once by a predeclared blocked inner-CV on pre-boundary engineering data (the preflight window); coefficients frozen and hashed; **no refits during accrual** (preferred) or a predeclared calendar refit lane (e.g. every 28 days, same λ, expanding window) reported separately and counted in the multiplicity budget.
- Tier 2 (optional challenger): one XGBoost at the P2a fixed configuration, counted as m = 2 under Holm, or labelled exploratory; justified a priori, not by burned-block ranks.
- Tier 3: none. Reopening conditions for deep models are the prior audit's three (≈ 10⁵ effectively independent labels; a prospectively validated nonlinear edge of a tree over the linear model; a cost model under which that horizon is tradable).
- Model-selection allowance during the campaign: zero. Seeds fixed. The only "selection" is the analytic threshold from measured cost.

**What additional capacity answers [INF].** Nothing until a linear model shows prospective incremental information; nonlinearity is a second hypothesis that needs its own multiplicity slot, not a milestone.

---

## 21. Evaluation / Multiple-Testing Adjudication

**Instrument reconstruction [RF].** Nested walk-forward with expanding train, one inner and one outer block per fold (4,821 hourly rows ≈ 201 days each), purge = horizon by index arithmetic, no embargo beyond the horizon, fold k's inner block = fold k−1's outer block (disclosed), every bar labelled (overlap ≈ 6× nominal N for classification metrics), greedy non-overlapping trades for economic metrics, retraining once per fold, no rolling refit study. Gates: 3-of-4 fold sign counts (P2b–P7; P4/P5 added mean and worst-fold conditions); coin-null pass probability 5/16 self-declared (`nn/p4_preregistration.py:1163-1169`); `annualised_sharpe` i.i.d.-scaled with no HAC; no bootstrap, effective-N, IC, power, MDE, FWER/FDR, DSR, PBO or SPA code anywhere in `nn/`, `tools/`, `chimera/` or `tests/`. Threshold: 29-point grid on 10–100 inner trades per fold, selected values scattering 0.36–0.72 across families and folds.

**Researcher degrees of freedom, quantified [RF/INF].** ~90 fitted cells (82 benchmark directories + 10 walk-forward directories); ~45 deciding and ~14 secondary gate applications; 12 readings of the same four blocks; per checkpoint, the free choices before a verdict were: v4 (verdict rule in code, architecture grid), P2a (family set chosen after v4), P2b/P2c/P3 (six model×arm verdicts with no single deciding cell), P4/P5 (one predeclared cell), P6 (five per-clock verdicts + ten secondary), P6-EXT (two + four), P7 (two). Never varied: symbol (1), venue (1), cost model (1: 20 bps), horizon rule (1), label family (1), refit cadence (1). Operating characteristic of the gate [INF]: P(pass | coin) = 0.3125; P(pass | per-fold P(positive) = p) = 4p³(1−p) + p⁴ → 0.48 at p = 0.6, 0.65 at 0.7, 0.82 at 0.8; a Sharpe-1 strategy has p ≈ Φ(1·√0.55) ≈ 0.77, so the gate had ≈ 0.77 power at α ≈ 0.31. Under a coin null ~45 applications would yield ~14 chance passes; zero occurred — consistent with untuned learners slightly below zero after costs, and uninformative about small real effects.

**Fit of the standard methods to the prospective design [INF/REC].** Holm (FWER) fits the small primary family (m ≤ 2). BH-FDR fits labelling the exploratory family only. Deflated Sharpe Ratio fits as a *disclosure* of the historical ~90-cell search, not as a gate for a single frozen candidate. PBO/CSCV fits a retrospective assessment of the historical threshold/family search (descriptive, allowed under constraint 34 as non-deciding), not a frozen prospective candidate. White's reality check / Hansen's SPA fit only if a challenger is compared to a benchmark on dependent data (use SPA with the same stationary bootstrap when Tier 2 is included). Nested selection is unnecessary in-campaign because no selection happens. The prospective window is the holdout; interim inspection is allowed only through a predeclared alpha-spending stopping rule.

**Decisions [REC].** (1) The historical evaluator is KEPT as a historical/engineering fixture and DROPPED as the deciding instrument for any prospective claim. (2) The campaign evaluator computes the endpoint on non-overlapping (or overlap-modelled) decision events, reports nominal N and N_eff, and derives every standard error from a frozen stationary block bootstrap with the Politis–White automatic block length. (3) Multiplicity budget: one primary endpoint, one candidate, optional one challenger (m ≤ 2, Holm), one horizon, one decision clock, universe as one panel (no per-symbol tests), one economic screen that gates R11 rather than promotion, two predeclared negative controls not counted, everything else BH-labelled exploratory. (4) Every prospective preregistration carries a mandatory "historical adaptivity disclosure" block (cells, gates, readings, shared families) and states that the candidate's prior is post-selection.

---

## 22. Power / MDE / Effective-Sample-Size Adjudication

**Unit of evidence [REC].** For an information endpoint: a non-overlapping decision event per symbol, with time dependence handled by a joint-in-time stationary bootstrap and symbol dependence by cross-sectional demeaning (or an effective-symbol count). For an economic endpoint: a realised trade (abstained events do not count). For robustness: independent time blocks (2–6 per 182 days — too few to decide anything; report descriptively). Raw rows are never the unit.

**First-principles arithmetic [INF; assumptions: hourly σ ≈ 0.6% for a major perp; taker round trip 12–16 bps (central 14); pairwise raw-return correlation 0.3–0.7, to be measured; per-decision IC 0.02–0.05; Gaussian score–return model; 182 days].**
- Overlap: H-bar labels on a 1-bar clock give N_eff,time = N/H (728 events per symbol at 6 h; 546 at 8 h; 182 at 24 h).
- Symbols: time-series effective count N/(1+(N−1)ρ) ≈ 1.4 (ρ 0.7) to 2.9 (ρ 0.3) for N = 15; cross-sectional demeaned design ≈ N − 1 = 14 if residual correlations are small (verify with the participation ratio on preflight data).
- SE(IC) ≈ 1/√N_eff; MDE = 2.49·SE (one-sided α 0.05, power 0.8) or 2.80·SE (Holm, m = 2).

| design (182 d, 15 symbols) | N_eff | SE(IC) | MDE m = 1 | MDE m = 2 |
|---|---|---|---|---|
| 6 h, time-series, ρ 0.7 | 1,019 | 0.031 | 0.078 | 0.088 |
| 6 h, time-series, ρ 0.3 | 2,111 | 0.022 | 0.054 | 0.061 |
| 6 h, cross-sectional (14 eff.) | 10,192 | 0.0099 | 0.025 | 0.028 |
| 24 h, time-series, ρ 0.7 | 255 | 0.063 | 0.156 | 0.175 |
| 24 h, cross-sectional (14 eff.) | 2,548 | 0.020 | 0.049 | 0.055 |

Regime and volatility clustering inflate these SEs by a further ≈ 1.2–2× for raw-return endpoints (less for vol-standardised ones); the bootstrap measures it. The lead's independent calculation (daily decisions, 12 symbols, 4 effective, 0.7 regime haircut) gives MDE ≈ 0.11 at 1 d and ≈ 0.045 at 4 h for a time-series endpoint — the same orders of magnitude.

- Economic relevance of a detectable IC: gross per event ≈ 0.8·IC·σ_H trading every event with a binary position, ≈ 1.75·IC·σ_H when trading only the top decile. At 6 h (σ_H ≈ 1.5%): IC 0.03 → 3.6 / 7.9 bps against ≈ 14 bps cost — uneconomic. At 24 h (σ_H ≈ 2.9%): 7 / 15 bps — break-even only when selective. IC 0.05 at 24 h selective → 25 bps > cost.
- Economic floor power: with per-trade net-return SD ≈ σ_H and a 5 bps floor, z = 2.5 needs ≈ 5,600 effective trades; 182 days at ~100 trades per symbol gives ~140–300 effective (time-series) or ~1,400 (cross-sectional); reaching 5,600 cross-sectionally takes ≈ 2 years; a 10 bps floor ≈ 183 days (borderline).

**Conclusion [INF].** 182 days on ~15 symbols can detect an incremental IC of ≈ 0.03 only with a cross-sectional/panel endpoint at ≤ 8 h and m ≤ 2. It cannot confirm a 5 bps net-per-trade floor. Under taker costs, economic relevance requires either IC ≥ 0.05 at ≥ 24 h, or a low-turnover policy whose holding period is days while the decision clock is hours, or maker execution. The regions where information is detectable and where economics are confirmable barely overlap in a 6-month window. Therefore: the first campaign is information-primary with an economic *falsification screen* (point estimate above the floor and bootstrap upper bound not below it → "not falsified"), and economic confirmation (R11) is sized from the measured effect — likely 1–2 years, or a larger effective universe. This is the arithmetic v3 §22 said must not be guessed; it has now been done twice independently and the roadmap must plan for it.

**The implementable pre-result method, to be frozen as code before the boundary [REC].** Inputs (pre-boundary only): per-symbol executable returns at the decision clock for candidate H; the measured cost model; a placeholder or frozen candidate score series (for dependence structure only); the correlation matrix of per-symbol score×return products. Layer 1 (analytical): N_eff,time = N/(1 + 2Σρ_k) with ρ_k the autocorrelation of u_t = s_t·r_{t,H} up to the Politis–White bandwidth; N_eff,sym = (Σλ)²/Σλ² of the correlation matrix of the per-symbol u series; N_eff = product; SE = sd(u)/√N_eff. Layer 2 (stationary bootstrap): Politis–Romano resampling of whole-panel time indices (same blocks for all symbols), expected block length from the Politis–White automatic rule, centred under the null → SE_boot and N_eff,boot = N·Var_iid/Var_boot. Layer 3 (full-pipeline null simulation): run the exact frozen pipeline (scoring, analytic threshold, cost charging, gate, Holm) on ≥ 1,000 null panels built by joint circular time shifts of returns beyond H + max lookback + block length, and by block-shuffled labels; record the empirical α of the exact decision rule; inject synthetic effects to trace the power curve; MDE = smallest effect with power ≥ 0.8. Outputs committed and hashed into the preregistration: nominal N; N_eff (three layers); m; α; target power; MDE in IC, in bps net per trade and in annualised Sharpe; expected calendar duration to the effect floor; cost ±50% sensitivity; the target number N of VALID scored blocks, the block length, the maximum extension E and the R11 continuation size N′ (or its deterministic sizing rule). The block-length *rule* and the endpoint estimator are frozen, not the numbers; PASS/FAIL uses only the frozen estimator; interim looks only via a predeclared alpha-spending rule.

---

## 23. Cost / Execution-Economics Adjudication

**Cost assumptions in the repository [RF].** Historical: flat 20 bps round trip per realised trade at every clock, fill at the decision close, zero latency, 5 bps slippage allowance; the "retired" cost still lives in `TargetSpec` (5 bps fee + 5 bps slippage, described as a "retail spot taker fee" while the demo's spot leg pays 10 bps). Runtime: per-leg fee/slippage from an unfrozen campaign config; `RecordedQuoteFillModel` crosses to the recorded touch and adds configured slippage; slippage is measured as an attribution, not double-charged; measured against the kline close rather than the mid and never emitted in bps (A12's two open gaps); no maker path; no depth/impact; no reject/partial-fill model beyond fault injection; funding at 8 h assumed; a `maintenance_margin_rate = 0.004` hard-coded in the carry rule; spot BUY fee charged in quote at full quantity, whereas Binance spot deducts the fee from the base asset received unless paid in BNB [EF, MEDIUM], so a real hedge would be short ≈ 10 bps of spot.

**Cost envelope vs signal [INF; σ_d ≈ 2.5% BTC-like, √time scaling, taker round trip ≈ 12–13 bps on BTCUSDT, 15–20 bps on liquid alts, ≈ 22 bps on spot].**

| horizon | σ_H (bps) | 12 bps / σ_H | 20 bps / σ_H | breakeven IC vs 12 bps (top-20% gated) |
|---|---|---|---|---|
| 1 m | 6.6 | 1.8 | 3.0 | — |
| 5 m | 15 | 0.8 | 1.4 | — |
| 15 m | 26 | 0.47 | 0.78 | — |
| 1 h | 51 | 0.24 | 0.39 | ≈ 0.17 |
| 4 h | 102 | 0.12 | 0.20 | ≈ 0.084 |
| 1 d | 250 | 0.05 | 0.08 | ≈ 0.034 |

Reading: with taker execution a per-trade edge survives costs only at ≥ daily horizons for plausible ICs, or at 4 h with IC ≥ 0.08. **Turnover is the lever:** a 1 h decision clock is fine if the position flips only every ~2 days (cost ≈ 12 bps / holding-days ≈ 6 bps/day against σ_d 250 bps → required daily IC ≈ 0.03). Hence the hysteresis rule of §15 is part of the frozen candidate.

**Required cost fidelity by stage [REC; decision-relevant semantics frozen in R3 and identical in R10 and R11].**

| stage | fidelity | how obtained | deciding? |
|---|---|---|---|
| information endpoint (R10) | measured median half-spread and taker fee per symbol; funding inside the horizon; label on executable quotes; no fill model beyond the frozen crossing rule | gen4 preflight bookTicker/aggTrade statistics (≥ 30 d), frozen as code in R3 | yes |
| economic screen (R10) and confirmation (R11) | the frozen deciding model: fee tier and BNB assumption; touch-crossing executable price at t + Δ with Δ frozen; conservative per-symbol slippage envelope (p90 trade-through for the reference size); funding by settlement per symbol interval; turnover on every target change; frozen reject/partial-fill rule; notional capped below the impact threshold | R2 measurements frozen as hashed code in R3; the same bytes in runtime and evaluator | yes — identical in R10 and R11 |
| autonomous demo (R9) | the same model; correctness, not realism, is the claim; shadow economics sealed | same | no (engineering only) |
| stress / diagnostic analyses (R11, R14) | measured slippage distributions; testnet reject/partial rates; hypothetical impact at larger sizes; mid-referenced slippage in bps; spot base-asset fee for any spot leg | testnet adapter; Tier B depth for BTC/ETH | never — reported separately; may trigger a new preregistration (C) |
| real money (R16+) | actual fills, fees, funding from the account statement reconciled daily against the frozen model; parity report | authenticated adapter; canary | operational only |

**Freeze rule [REC].** Every economic semantic that affects the endpoint — fee assumptions, spread rule, executable-price rule, slippage estimator or conservative envelope, funding treatment, turnover accounting, reject/partial-fill treatment, latency assumption, impact rule and the notional cap that keeps impact out of scope — is frozen in R3 as hashed code before the boundary. R10 and R11 test the same scientific/economic object. Newly observed execution, testnet or live-operational information may enter R11 only (A) through a deterministic update rule preregistered before R10, (B) as a non-deciding stress/diagnostic analysis, or (C) by opening a new preregistration and campaign. The confirmatory period adds data; it never redefines success.

**Fees and funding cadence [EF, confirmed by the external-documentation workstream against Binance primary documentation on 2026-09-13; not independently re-verified by a second agent].** Binance USD-M VIP0 maker 0.02% / taker 0.05% before discounts (0.018%/0.045% when paid in BNB; Binance FAQ 360033544231); spot VIP0 0.10%/0.10% with the fee deducted from the asset received unless paid in BNB; funding every 8 h by default with some symbols on 4 h or 1 h intervals; funding caps and the premium-index formula are venue-published and change by announcement. The campaign contract must record the fee tier, the BNB-discount assumption and the funding interval per symbol as frozen assumptions dated to the fetch, and must either exclude non-8 h symbols from campaign 1 or model their interval explicitly per symbol (the audit recommends exclusion). Because funding is a regime-dependent cost that can dominate a slow directional rule, the contract must also preregister a tail/drawdown component of the economic screen (maximum funding-inclusive drawdown per block) rather than a mean-only floor.

**Liquidation at 1× [RF].** Distance ≈ 99.6% (`entry × 0.004`); the default `min_liquidation_distance_pct = 0.5` is slack by construction and would bite at 2×; the closed-form used is a first-order approximation that is anti-conservative for SHORTs by a small amount (documented). Fidelity becomes material only if leverage above 1× is ever authorised, which nothing in this audit recommends before real-money Gate 4.


---

## 24. Technology Landscape and Build-vs-Adopt Decisions

Standing of this section: the external technology-landscape workstream did not complete inside the usage window (see §1 process disclosure); the verdicts below rest on the repository's actual choices [RF] and on the reviewer's knowledge of the alternatives [INF]. Maintenance status and licences of any alternative must be confirmed by the owner before adoption; no verdict here depends on an unverified claim about an external project.

| area | current choice [RF] | credible alternatives [INF] | verdict | justification | stage |
|---|---|---|---|---|---|
| Exchange ingestion (public market data) | in-house asyncio websocket client per endpoint with backoff, ping/pong, 23 h 50 m proactive reconnect, REST pollers | ccxt.pro, cryptofeed, binance-connector-python, tardis-machine (replay/normalised historical) | **KEEP** in-house; REVISE for multi-symbol/stream generalisation and an in-process silence watchdog | the code is small, tested against synthetic servers, and records provenance (`time_basis`, receipt clocks) that generic libraries do not; the failure mode that matters (retired base URL delivering silence, A3) was found by the project's own preflight | R2 |
| Historical archives | first-party `data.binance.vision` objects with `.CHECKSUM`, per-layout epoch units (PR #76) | tardis.dev (paid), Kaiko (paid) | **KEEP** | first-party, checksummed, free; the reconciliation code is the project's strongest data asset | R5 |
| Event storage | append-only NDJSON.gz per stream-day (raw, immutable) + per-day parquet (normalised, regenerable) | ClickHouse, QuestDB, TimescaleDB, DuckDB files, kdb+ | **KEEP** files; **DEFER** DuckDB as an *analytics* reader over parquet (no server) | single node, auditability and immutability matter more than query speed; a database adds an authority that is not the raw file | later |
| Parquet/Arrow/pandas | pyarrow + pandas 2.x | Polars, DuckDB | **KEEP**; DEFER Polars | the evidence path is Decimal-based and small; a second dataframe library is a second source of float-rendering drift | — |
| Order-book reconstruction | none | cryptofeed book, tardis-machine, hftbacktest, nautilus_trader | **DEFER** until a maker-execution mandate; if Tier B depth is recorded, store raw diffs and *do not* reconstruct in the recorder | reconstruction is notoriously subtle (sequence gaps, snapshot alignment); the recorder must stay "computes nothing" | R2/R4 |
| Streaming/event architecture | plain asyncio, filesystem handoff between two processes | Redis Streams, NATS, Kafka | **KEEP**; DROP brokers | one host, two processes, minute cadence; a broker is a second authority and a new failure mode | — |
| Backtesting / simulation | custom nested walk-forward (research); custom event-driven dry-run venue (runtime) | vectorbt, backtrader, zipline-reloaded, nautilus_trader, hftbacktest, jesse, Lean | **KEEP** both; **REVISE** research evaluator with the inference module (§22); REVISIT hftbacktest only for a maker/queue study | vectorised frameworks cannot express the runtime's persistence/replay semantics; adopting nautilus for the campaign would replace a validated, hashed, replayable venue with a framework whose fill model the project does not control | — |
| Execution simulation | recorded-quote touch crossing + configured slippage; deterministic adverse model | queue-position models (hftbacktest), impact models | **KEEP** for demo, R10 and R11 (the deciding fill model is frozen as hashed code in R3 and identical in R10 and R11); measured slippage distributions and a size-dependent impact term for BTC/ETH from Tier B depth enter only as non-deciding stress/diagnostic analyses (§23 freeze rule) or through routes (A)/(C) | § 23 fidelity table | R3 (freeze); R11/R14 (stress only) |
| Model training / feature pipelines | scikit-learn, LightGBM/XGBoost, PyTorch (retired MTST); in-house causal feature functions | feature stores, Ray, Optuna | **KEEP** scikit-learn + one tree library; **DROP** torch from the active path; DROP feature stores | one frozen linear candidate needs none of it | R3 |
| Experiment tracking / model registry | hashed artifact directories + SHA256SUMS + research-state verifier; MLflow and Ray Tune present but unused | MLflow, W&B, DVC | **KEEP** hashed artifacts; **DROP** MLflow and Ray Tune extras | the registry is the git history; a tracking server is a second, mutable authority | R1 |
| Orchestration | systemd units; manual/cron reports | Prefect, Dagster, Airflow, systemd timers | **KEEP** systemd; ADD systemd timers for daily reports and reconciliation; DROP workflow engines | scale does not justify a scheduler service | R1 |
| Monitoring / alerting | Prometheus + Grafana + Alertmanager; per-stream age metrics; heartbeat files | Uptime Kuma, Healthchecks-style dead-man endpoints | **KEEP**; **ADD** an off-host dead-man check for recorder and runner heartbeats | today every alert lives on the host it watches | R1 |
| Deployment | one VPS, systemd (hardened units), docker-compose alternative, python:3.11-slim images | Nomad, k8s, immutable images | **KEEP** systemd; DEFER containers as the primary deployment; DROP orchestration platforms | single operator; the hardening directives already present are the right layer | — |
| Secrets | `.env` files, redaction, pre-commit refusal | sops/age, systemd `LoadCredential`, 1Password CLI, Vault | **REVISE**: systemd `LoadCredential` (or sops/age-encrypted files) for any future key; never in the runner's working directory | needed only at R14; decide the mechanism then, but design the live package around it | R14 |
| Live trading framework | none (no live route) | nautilus_trader live, hummingbot, freqtrade live, ccxt | **DEFER** to R14; do not adopt a framework for the first live stage; build a thin authenticated adapter in a separate package around the official Binance USD-M REST + user-data stream, with explicit fields | the project's value is in its provenance and safety semantics, which frameworks hide; a thin adapter is auditable | R14 |
| Exchange abstraction | none needed (one venue) | ccxt | **DEFER**; a second venue is a new mandate | — | — |
| CI | GitHub Actions (tags), lock without hashes, two OS legs, promtool, integrity checks | SHA-pinned actions, `pip --require-hashes`, image digests | **REVISE** pins; **KEEP** the rest; ADD a replay-determinism job and a synthetic 48 h soak-in-CI once R1 lands | supply-chain posture is the cheapest fix with the largest downside if ignored | R1 |
| Freqtrade | retired/disconnected; configs, image and strategies in tree | — | **DROP** (delete) | the only live-capable path; a large dependency; on no evidence path | R1 |
| Inference service / registry serving | disconnected | — | **DROP** from the active tree after R1 (keep in history) | a frozen linear candidate runs in-process | R1 |

Where in-house code re-implements something notoriously hard [INF]: exchange order-state reconciliation against a venue that outlives the process (R14 — do not improvise; model it on the exchange's documented order lifecycle and user-data stream, and drill it on testnet), L2 book reconstruction (deferred), and fill simulation with queue position (deferred). None of these is on the first campaign's path.

---

## 25. Future Architecture Review

**Principle [REC].** One host, two long-running processes (recorder, runner) plus batch jobs, filesystem as the only channel, exactly as the master plan §2.1 designed — until a live stage adds a third process (the authenticated adapter) in its own package and user. No message broker, no database server, no orchestration platform at any stage covered here.

| layer | target design | one process or several | when |
|---|---|---|---|
| Market-data ingestion | in-house recorder generalised to (symbol, stream); one websocket connection per venue endpoint; REST pollers; in-process silence watchdog that reconnects and flips health | recorder process | R2 |
| Canonical event schema | `RawEvent(stream, symbol, canonical_ns, time_basis, receipt_wall_ns, receipt_mono_ns, dedup_key, payload)` — unchanged; schema version stamped per file; per-stream parsers versioned | recorder | R2/R4 |
| Historical storage | raw NDJSON.gz per stream-day, immutable, SHA-256 in a day manifest with contract hash + code version + schema version; first-party archives for reconciliation | files | R4 |
| Live storage | the same raw files, plus per-day normalised parquet regenerable from raw, written atomically; the *last closed minute* published incrementally for the runner | files | R1/R2 |
| Feature computation | one pure function from a `MarketSnapshot` to the frozen feature vector; the same function in research, replay and live | runner (and offline evaluator) | R7 |
| Causal feature state | `MarketSnapshot` generic over contract-declared causal clocks: per input and per timeframe `source_time`, `available_time`, `as_of`, `age`, `staleness`, `complete`, `missing`; closed-bar clocks derived causally from the base 1 m stream (multi-timeframe capability is a required engineering property); the deciding subset fixed by a hashed allow-list (decision clock + ≤ 2 slow variables in campaign 1) | runner | R7 |
| Model service | none; frozen coefficients loaded from a hashed artifact in-process | runner | R3 |
| Strategy decision layer | the frozen candidate: snapshot → scores → policy (analytic threshold, hysteresis) → `TargetPositionSet` | runner | R8 |
| Target-position layer | per-symbol `TargetPosition` (side, quantity, confidence, abstain) under portfolio caps; the carry lane emits a `HedgeTarget` in its own process | runner | R8 |
| Execution state machine | the existing `chimera/futures` order state machine per symbol; MARKET + reduce-only semantics; idempotent events | runner | R8 |
| Exchange adapter | none until R14; then a separate package with its own AST guard set, signed REST + user-data stream, unknown-state resolution, rate-limit backoff | adapter process (separate user) | R14 |
| Aegis | one engine per runtime process; evaluates the target-position *set* (portfolio gross/net/per-symbol caps, order rate, staleness, dispute state, kill switch, funding cap, daily loss and drawdown from persisted equity) and every exposure-increasing order; reductions ungated; scientific campaign limits (max trades/day, allowed symbols) live in the campaign contract and are enforced by the policy layer, not by Aegis | runner | R8 |
| Accounting | one account ledger (capital, free cash, per-position margin levels, fees, funding paid/received, realised; unrealised at mark); executor ledgers as per-leg journals | runner | R8 |
| Reconciliation | dry-run: labelled self-consistency; live: every minute against the exchange position via the user-data stream, `HALT` on disagreement, written-reason resolution | runner / adapter | R8 / R14 |
| Persistence | atomic files as today; write order store → ledger → log; log arbitrates; SIGTERM deferred past PERSISTENCE; directory fsync everywhere | runner | R1 |
| Decision journal | the existing hash-chained log with `seq` and `OPERATOR` policies fixed; evidence-class label per record | runner | R1 |
| Telemetry | Prometheus as today + off-host dead-man; alert semantics matching a daemon | both | R1 |
| Operator controls | CLI `status/flatten/resume/resolve/kill` all working against persisted state, each recorded; no state-file edits ever | runner | R1 |
| Deployment | hardened systemd units; separate Unix users for recorder, runner, (later) adapter; separate state roots; pinned dependencies and images | host | R1/R14 |
| Recovery | triage as today plus audited dispute re-derivation; crash-injection harness in CI | runner | R1 |

**What is over-engineered today [INF].** Modes/consensus/router, the inference service and registry serving side, MLflow/Ray Tune, Grafana breadth, the Freqtrade stack, and ~26k lines of frozen checkpoint code on the active tree (keep in history, off the runtime path). **What is under-engineered [INF].** The operational shell (§5), multi-symbol single-leg execution, one account ledger, the statistical inference module, the gen4 multi-stream recorder, and the crash/clock/exchange-ambiguity test harnesses.

---

## 26. Autonomous Demo Definition

**What "demo" means today [RF].** A file-driven, bounded, carry-shaped catch-up pass that cannot start under the campaign profile (§§3, 10, 11). It proves persistence and replay semantics on synthetic fixtures; it does not prove unattended operation.

**Definition adopted [REC].** An autonomous demo is a *daemon* consuming a *live* recorder's files with ≤ 2 minutes end-to-end latency, driving single-leg LONG/SHORT dry-run positions per campaign symbol (and, in a separate process, the two-leg carry position) under one Aegis per process, running unattended for a predetermined window with drills, and reproducing every decision byte-for-byte in replay. Data source: the gen4 recorder's engineering root or the production recorder's pre-activation root; never a prospective root, because the soak precedes activation (R9 before R6) and no prospective data exists while it runs. Exchange connectivity: none. Fills: recorded-quote crossing + measured slippage. Accounting: the single account ledger. Risk: Aegis with all configured rules reachable.

**ENGINEERING DEMO PASS gates [REC].**
1. ≥ 14 consecutive UTC days unattended on live recorder data under a SOAK profile (the frozen candidate runs in SHADOW; scoring is forbidden until R10's first scored day; the candidate's shadow economics are sealed at soak end as SEALED-DIAGNOSTIC and are not an input to the activation decision).
2. ≥ 2 planned restarts and ≥ 1 unplanned `SIGKILL` mid-minute, each recovered with a `RECOVERY` record and no state-file edit.
3. ≥ 1 injected feed outage ≥ 10 min → staleness halt or `INCOMPLETE_STATE` as designed, automatic continuation when data resumes; ≥ 1 recorder-process restart during the soak.
4. Kill-switch drill: file → halt persisted across a restart → `resume` with a note, recorded. ≥ 1 reconciliation-dispute drill cleared through the CLI with a written reason. ≥ 1 disk-low drill.
5. Daily report generated every day; every alert rule fired at least once (synthetic) and delivered off-host.
6. Replay of every soak day = PARITY, with the `seq`-across-restart and `OPERATOR` policies fixed before the soak.
7. Zero manual state-file edits; zero unexplained divergence; zero halts without a record; zero minutes skipped as stale while the recorder was healthy.
**KILL:** an unexplained divergence not root-caused within one repair cycle; a silent position change; a halt without a record; a recovery that required editing state; > 1% of healthy-recorder minutes skipped.
**Separation:** the soak report is frozen as ENGINEERING evidence and contains engineering metrics only; the candidate's shadow PnL is logged for Aegis, sealed at soak end and opened only after R10 closes as a shadow-vs-live diagnostic; it may not select anything and may not inform the R6 decision; a passed soak changes no prospective standing; the head that passed the soak is the head R10 runs (head-freeze rule, §37.2). **Order:** R9 PASS is a recorded prerequisite of R6 — SOAK PRECEDES PROSPECTIVE ACTIVATION.

---

## 27. Prospective Campaign Design

**Choice: TWO-STAGE IN ONE ACCRUAL for the directional lane (information-primary with an economic falsification screen that gates a separately sized confirmation), plus an ECONOMIC-FIRST carry lane run in parallel as its own campaign. [REC]** Information-first alone (v3) invites a second selection step at the economic stage and, as sketched, cannot reach its economic floor; economic-first alone on the directional lane is underpowered for any window under two years (§22); the two-stage design commits both readings before the boundary and sizes the confirmation from measured effects.

**Lane A — directional panel campaign.**
- *Preregistration:* committed and pushed ≥ 7 days before the boundary with an independent review recorded; hashed payload as P13's pattern; includes the historical-adaptivity disclosure block, the frozen power report, and the coherence statement (endpoint ↔ policy ↔ runtime).
- *Frozen universe:* K symbols by the §18 procedure; listed by name and hash; delisting → symbol block invalidated.
- *Frozen streams:* the gen4 contract's Tier A (klines, markPrice, bookTicker, aggTrades, funding, OI, forceOrder, exchangeInfo snapshots); only streams in the contract may be used.
- *Frozen features:* §19; feature function hashed.
- *Frozen target:* vol-standardised executable forward return over H (§15); H fixed by the κ rule from preflight data.
- *Frozen models:* Tier 0 references; Tier 1 linear with λ from a predeclared blocked CV on pre-boundary data and coefficients hashed; optional Tier 2 challenger counted (m = 2).
- *Model-selection allowance:* none during accrual; hyperparameter allowance: none; refit schedule: none (or a predeclared calendar lane reported separately and counted).
- *Cost model (deciding; identical in R10 and R11):* per-symbol fee tier and BNB assumption; touch-crossing executable price at t + Δ with Δ frozen; conservative per-symbol slippage envelope (p90 trade-through for the reference size); funding by settlement per symbol interval; turnover on every target change; frozen reject/partial-fill rule; notional capped below the impact threshold; all frozen from preflight as hashed code in R3; ±50% sensitivity reported, never deciding; changes only by (A) a preregistered deterministic update rule, (B) non-deciding stress analysis, or (C) a new preregistration (§23).
- *Primary endpoint (information):* incremental cross-sectional rank IC of the Tier 1 candidate versus the same candidate with the information block zeroed, computed on non-overlapping decision events pooled across the panel, SE from the frozen stationary bootstrap, Holm over m ≤ 2. PASS iff the lower one-sided bound exceeds the preregistered floor (set from the power report so that MDE ≤ floor at the planned duration).
- *Economic screen (co-primary gate for R11, not for promotion):* net conditional executable return of the implied low-turnover LONG/SHORT panel policy on traded events after measured costs and funding; NOT FALSIFIED iff the point estimate ≥ floor and the bootstrap upper bound ≥ floor; FALSIFIED otherwise. The floor is expressed in bps net per unit gross exposure per day and equals the measured cost of one round trip per expected holding period plus a margin fixed in the preregistration.
- *Abstention:* analytic threshold (§15); hysteresis band frozen.
- *Minimum duration:* the larger of a calendar cap and the event count the power report requires; monthly blocks; a predeclared stopping rule (alpha-spending) as the only interim look; expected ≥ 6 months, likely 9–12 for m = 2 [INF].
- *Sample/power gate:* the frozen module's outputs (N_eff, MDE, duration) committed before the boundary; if MDE > floor at the cap, the design is declined.
- *No-peek policy:* engineering dashboards only (coverage, staleness, halts, parity, block validity); at each block end the validity verdict is computed from engineering records only, committed and hashed first; economic and information readouts are computed only for VALID blocks, by the frozen evaluator, logged and hashed; an INVALID block never receives a readout.
- *Engineering invalidation (not a result; mechanical, non-adaptive):* the contract freezes the invalidity criteria (required-stream coverage below threshold on more than the frozen number of days; replay-parity failure; a halt not recovered within the frozen window or without a record; recorder identity or contract-hash change; a runtime head change outside an ENGINEERING-FIX PR; any manual state edit; delisting/halt of a symbol → that symbol's block, or the whole block if the minimum symbol count fails), the evaluator (`nn/prospective/validity.py`, engineering records only, no prices/scores/positions/PnL), the target number N of VALID blocks, the deterministic calendar-extension rule (contiguous blocks until N VALID), and the maximum extension E (proposal ⌈N/2⌉). An INVALID block stays in chronology, labelled, reported, is never a negative result and is never replaced. If E is exhausted without N VALID blocks the campaign is NOT EVALUABLE / ENGINEERING FAILURE — neither PASS nor scientific FAIL; any successor is a new campaign on a new boundary with no pooling.
- *Scientific failure (a result):* endpoint below floor at the end → immutable negative; candidate retired for this target/horizon/universe.
- *Outcomes:* information FAIL → negative, R12; information PASS + screen FALSIFIED → information-only result, no promotion, R12 decides whether a redesigned low-turnover policy merits a new campaign; information PASS + NOT FALSIFIED → R11 confirmation of the identical object (same candidate, same deciding economics, same evaluator hash), with N′ VALID blocks fixed in the preregistration or by a sizing rule preregistered as deterministic in the measured effect; E exhausted → NOT EVALUABLE / ENGINEERING FAILURE, R12.
- *Negative controls:* information block zeroed (must fail); block-shuffled labels and time-reversed panel (must fail at the nominal rate).
- *Evidence created:* PROSPECTIVE for this candidate. *Not created:* any alpha claim, any real-money eligibility.

**Lane B — carry economic campaign (SECONDARY, OPTIONAL; parallel; never on the critical path).** Elected by the owner at R0 with a deadline at R4 (its spot streams must be in the gen4 contract); own preregistration hash and own hypothesis slot; own process, state root, Aegis instance and ledger; runs on the same R6 boundary if ready, otherwise waits for its own later contract and boundary; never delays R1–R9 or R6; yields on any resource conflict.
- Existing two-leg runtime after R1; BTCUSDT (ETHUSDT optional); always-on or funding-gated rule with parameters frozen from pre-boundary funding statistics; primary endpoint: net return after both legs' measured frictions (including the base-asset spot fee) per settlement, aggregated by monthly block; PASS iff the mean per-settlement net return over ≥ 6 blocks exceeds a preregistered floor with a block-bootstrap lower bound above zero and the worst block above a preregistered drawdown floor; negative funding regimes are part of the result, not an invalidation; engineering invalidation as Lane A. Multiplicity: m = 1; disclosed as a second programme-level hypothesis. Evidence: PROSPECTIVE for the carry mechanism.

**No retroactive substitution, in both lanes.** A failed block, symbol, day or campaign is never replaced; a new attempt is a new preregistration on a new boundary.

---

## 28. Strategy Expansion Governance

Rules [REC]:
1. A negative campaign retires the candidate family for the same (target, horizon, universe); it is recorded as an immutable negative with its full disclosure block. A NOT EVALUABLE campaign (engineering failure) retires nothing scientifically: the cause is fixed, the runtime re-soaked, and the successor rule pre-committed in the preregistration applies automatically (at most one same-design successor on a new boundary, never decided on outcomes, no pooling; the predecessor's VALID-block readouts stay sealed until the successor closes; a second exhaustion is terminal). An ABORTED campaign (owner stop, or a change to a contract-hashed module during accrual) consumes the candidate's slot and permits no same-design rerun.
2. A new candidate must change at least one design axis a priori (mechanism, target, horizon, information set, universe) and receives its own R3 (power report, review, ≥ 7-day gap). It may read the negative campaign's *descriptive* statistics (turnover, cost realisation, coverage) but not its per-decision outcomes for design.
3. Budget: at most two directional candidates and one carry candidate across the programme before a mandatory stop-and-decide review; after two directional negatives, record "no deployable directional alpha under the current mandate".
4. Stopping: a lane closes on its second negative; the programme closes or opens a new mandate (maker execution; breadth beyond crypto perps) with a new budget.
5. Reuse: engineering evidence (soaks, parity), DIAGNOSTIC preflight statistics and the cost tables are reusable across candidates; PROSPECTIVE evidence belongs to the candidate that earned it and to no other.
6. The "≥ 2 prospective-eligible strategy families before orchestration" rule is **KEPT and tightened**: orchestration opens only when two families hold CONFIRMATORY standing (R11 PASS), never merely because two families exist or two lanes ran.

---

## 29. Portfolio / Orchestration Adjudication

**When justified [INF/REC].** Only after two families reach CONFIRMATORY standing. Before that, "portfolio" means the Aegis portfolio caps inside one lane (gross ≤ 1× capital, net exposure cap, per-symbol cap, drawdown budget) which R8 must implement because a panel of single-leg positions is already a portfolio. Two dry-run lanes run as two processes with separate dry-run capital; that is not orchestration.

**What orchestration would need, when earned.** Explicit capital allocation per lane; netting of opposing intents on the same symbol; gross/net caps at the account level; concentration limits per symbol and per regime cluster; a drawdown budget per lane and total; correlation-aware exposure using the measured effective-N rather than a VaR model (a historical-simulation floor at most at this scale); one Aegis, one execution layer, one account ledger, one evidence trail. Strategy conflicts are resolved by a frozen priority or by netting, never by a discretionary override at runtime.

**Verdict:** DEFER (R13), with the Aegis portfolio scope built earlier (R8).

---

## 30. Live-Trading Readiness Review

**Today [RF].** No authenticated route exists in `chimera.futures` (construction with `dry_run=False` raises; AST guards forbid network and credential imports); the legacy Freqtrade path is live-capable and double-gated. Nothing has ever been reconciled against a real exchange position. The master plan §20 sketches a sound credential architecture. Nothing satisfies any live gate.

**Gates [REC; parameters are proposals to be frozen by the owner before any live stage].**
- **Gate 1 — scientific eligibility.** R11 CONFIRMATORY PASS for the candidate that will trade, under the same object and the same deciding economic semantics as R10; an independent audit reconstructing the verdict from the decision log and recorder files; a continuation period not contradicting it. No engineering evidence counts; no demo, soak or canary counts.
- **Gate 2 — execution eligibility.** The R9 soak gates re-run in full on the exact deployable head (the runtime plus the adapter package in testnet mode) and frozen as ENGINEERING evidence; a separate authenticated adapter package exercised ≥ 30 days on testnet or read-only live reconciliation (a read-only key with no trading permission; no trade-capable key before Gate 4 is signed) with: idempotent client order ids; unknown-state resolution by order query before any retry; partial fills across restarts; cancels; reduce-only semantics; rate-limit backoff; `listenKey` keepalive and expiry; maintenance-window handling; websocket loss and REST ambiguity drills; reconciliation every minute with `HALT` on disagreement; zero unresolved disputes at the end; a drill log frozen as ENGINEERING evidence.
- **Gate 3 — operational eligibility.** Dedicated sub-account; trade-only API key with withdrawals disabled and an IP allow-list, permissions re-verified daily; secrets outside the repository and the runner's working directory (systemd `LoadCredential` or encrypted files); separate Unix users and state roots for recorder, runner and adapter; SHA-pinned CI, hash-pinned dependencies, digest-pinned images; backups restored in a drill; off-host dead-man switch; chrony with the existing skew alert; documented VPS baseline and firewall; audit log of every operator action; an out-of-process exchange-side "cancel all and close" procedure drilled on testnet.
- **Gate 4 — capital governance.** A signed live-authorisation contract, reviewed before any trade-capable key is created, fixing: starting capital (a small USDT sum the owner can lose entirely), max gross exposure ≤ 1× capital, leverage 1× (config and exchange account setting), max single-order notional, max daily loss, max cumulative loss and max drawdown (halt + flatten + key rotation), per-symbol cap, funding-cost cap, overnight policy, shutdown triggers, two-person enable, rollback path. The numbers come from the soak's measured turnover and the campaign's realised volatility, frozen then.
- **Gate 5 — canary.** ≥ 30 days; ≤ 10% of the authorised cap and above minNotional × the largest campaign order; 1×; the campaign's symbols; MARKET + reduce-only identical to dry-run; parallel dry-run for parity; daily reconciliation of fills, fees and funding against the account statement; abort on any dispute, unexplained divergence, daily-loss breach, key-permission change or parity failure; escalation = stop and audit, never resize. A profitable canary is operational evidence only.
- **Gate 6 — controlled scale-up.** Increase only by a new decision with its own evidence period and audit; ≤ 2× per step; ≥ 60 days between steps; only after realised costs and parity match the model; automatic decrease or halt on drawdown, parity breach or dispute; no profit-based acceleration.

---

## 31. Security / Infrastructure Review

**Present [RF].** Hardened systemd units (`NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp`, `PrivateDevices`, `ProtectHome`, `RestrictNamespaces`, `MemoryDenyWriteExecute`, `StateDirectory`); `.env`-based secrets with redaction and a pre-commit refusal of `.env`; `detect-private-key` hook; no committed secrets found by pattern search at HEAD; Prometheus/Grafana/Alertmanager pinned by version; `python:3.11-slim` base images; Actions pinned by major tag; a 170-line lock without hashes; skew, disk-low, stale-stream and heartbeat alerts; no backup, no off-host monitor, no documented VPS baseline (provider and region are not in the repository), no log-rotation policy, no chrony requirement.

**Recommendations sized to a single-operator single-VPS system [REC].** SHA-pin Actions; `--require-hashes` for the lock; digest-pin images; off-host dead-man checks for both heartbeats; nightly encrypted backup of the recorder root and state directories with a quarterly restore drill; chrony with the existing skew alert as the acceptance check; journald rotation limits; separate Unix users per process; firewall allowing only SSH (key-only, non-default port optional) and Prometheus scrape from a known host; automatic security updates for the OS, never for Python dependencies; no GitHub Actions job may ever hold an exchange secret; for R14, secrets via `LoadCredential` or encrypted files decrypted only by the adapter's user, and a documented key-rotation procedure. Reject: containers as the primary isolation, Kubernetes/Nomad, a secrets server, multi-region failover — none is justified by the system's scale, and each adds an authority.


---

## 32. Failure-Mode Pre-Mortem

Premise: ProjectChimera has, some time later, published a false alpha claim and/or lost real money badly. The most plausible paths, ranked by likelihood × severity × (1 − detectability); each mapped to the roadmap gate that must close it. [INF throughout; existing controls are RF.]

| rank | path (class) | narrative | L | S | D | existing control | missing control | gate |
|---|---|---|---|---|---|---|---|---|
| 1 | Inherited instrument (FALSE ALPHA) | The campaign evaluator reuses the fold-count gate and the threshold grid; an IC of 0.04 on 3-of-4 monthly blocks is read as PASS; nobody computed N_eff or the null rate of the exact rule. | H | H | L | constraints 34–39; v3 §9.5 prose | executable, hashed power/inference module; single frozen estimator | R3 |
| 2 | Silent dead feed (MONEY) | A live adapter inherits a runtime whose staleness veto is identically zero; the websocket stalls; positions are held through a 10% move with no halt; funding accrues; the operator sees `up=1` between passes. | H | C | L | `RecorderStreamStale` alert on the recorder host | operational staleness guard on the decision host; external dead-man switch; exchange-side stop procedure | R1, R14 |
| 3 | Interim peeking (FALSE ALPHA) | Monthly reports show PnL; the team "just looks"; a losing candidate is quietly replaced or the universe expanded "for coverage"; the campaign is extended after a good month. | H | H | L | v3 §19 V3-10 prohibitions | no-peek policy; blind engineering dashboards; predeclared stopping rule; evidence-class labels | R3, R10 |
| 4 | Cost optimism (BOTH) | The economic screen uses BTC's spread for alts, the VIP0 taker fee for a fee-tier that changes, and 8 h funding for a symbol that settles every 4 h; the "not falsified" screen passes; R11 is opened; live fills cost 2× the model. | H | H | M | measured-cost requirement in v3 §12.2 | per-symbol measured cost table frozen in the preregistration; funding-interval check; parity of canary fills vs model | R2, R3, R16 |
| 5 | Operator hand-edit (MONEY) | A dispute with no clearing path halts the campaign; under time pressure the operator edits `carry_ledger.json`; the runtime resumes on a fabricated equity; Aegis halts too late. | M | H | M | runbook forbids it | audited resolve paths; no state edits by policy and by test | R1 |
| 6 | Unknown order state (MONEY) | Live adapter: an order times out; the retry duplicates it; the position is 2× intended; reconciliation lags a minute. | M | C | M | idempotent client order ids in the dry-run design | exchange-adapter unknown-state resolution by order query before any retry; user-data stream; 30-day drills | R14 |
| 7 | Endpoint/runtime mismatch (FALSE ALPHA) | A cross-sectional IC passes; the runtime trades one symbol at a time on the time-series score; the "validated" candidate is not what runs. | M | H | L | none | coherence rule; runtime soak on the same policy before boundary | R3, R8, R9 |
| 8 | Leakage via available_time (FALSE ALPHA) | A 4 h bar is used at its open + 1 minute because the snapshot has no `available_time`; the live system cannot reproduce the backtest's features. | M | H | M | research shift controls | runtime snapshot with `available_time`; two-sided leak controls in CI | R7 |
| 9 | Correlated panel collapse (MONEY) | Ten "independent" perps move as one on a BTC shock; gross cap per symbol is respected, portfolio loss is 10× the per-symbol expectation. | M | H | H | none (single symbol today) | portfolio gross/net caps in Aegis; effective-N-aware exposure; drawdown budget | R8, R14 |
| 10 | Leverage creep (MONEY) | After a good canary, leverage is raised to 2× "to make it matter"; the liquidation approximation is anti-conservative for SHORT; a wick liquidates. | M | C | H | 1× pinned in config and executor | live contract fixes 1×; bracketed liquidation solve before any change | R15, R18 |
| 11 | Secret compromise (MONEY) | API key in an env file on a shared VPS; key has withdrawal rights; account drained. | L | C | M | `.env` never committed; secret redaction | trade-only key, withdrawals disabled, IP allow-list, secrets outside the repo, sub-account, daily permission check | R14 |
| 12 | Dependency/config drift (BOTH) | A pandas or pyarrow release changes float rendering; replay diverges; or a config change lands untested and halts are silently disabled. | M | M | M | lock file; config hash in the log | hash-pinned deps; digest-pinned images; environment-parity check in CI | R1 |
| 13 | Recorder gap invisible to coverage (FALSE ALPHA) | Coverage counts minutes; bookTicker is health-gated and stalls for hours; labels are computed on stale quotes; IC is inflated by stale-quote reversion. | M | H | L | per-stream age metrics | cross-stream consistency gates; label-quote staleness refusal in the evaluator | R2, R4 |
| 14 | Canary profits reinterpreted (BOTH) | A profitable 30-day canary is cited as validation; capital is scaled. | M | H | L | v3 §16 rung text | Gate 6 rules; explicit "canary is operational evidence" labelling | R16, R18 |
| 15 | Model drift / regime shift (MONEY) | A candidate frozen in a trending regime is run through a crash; no drift monitor; drawdown budget hit only after the fact. | M | H | M | drawdown/daily-loss halts | preregistered drawdown budget and halt-then-review; no refits | R10, R17 |
| 16 | Wrong funding sign or cadence (MONEY) | A symbol switches to 4 h funding mid-campaign; the model books 8 h; hedge ratio decisions are wrong. | L | M | M | A10 sign convention; monthly archive reconciliation | funding-interval metadata snapshot per day; refusal on interval change | R4 |
| 17 | Exchange maintenance/outage (MONEY) | Orders rejected during maintenance; the adapter retries into the reopening auction; fills at extremes. | M | M | H | none | maintenance-window handling in the adapter; drill | R14 |
| 18 | Timestamp domain mixing (FALSE ALPHA) | Spot book in receipt time, perp in exchange time; a cross-venue feature uses both; a 300 ms skew becomes a "signal". | L | M | L | `time_basis` labels | snapshot per-input time basis; skew alert already exists | R7 |
| 19 | Governance decay (BOTH) | Preregistration-to-first-result gaps return to minutes; the same session designs and reads. | M | H | L | plan-first policy | ≥ 7-day gap and independent review recorded in the PR | R3 |
| 20 | Deployment mistake (MONEY) | The demo and the live process share a state directory or a config; the live process replays a demo cursor. | L | C | M | none | separate hosts/users/state roots; live package separate | R14 |

**Single most likely false-alpha path:** #1 (an inherited, unpowered instrument plus #3 interim peeking) — the programme has already shown it can design and read a checkpoint in 8 minutes.
**Single most likely real-money-loss path:** #2 (a runtime that cannot see its own feed die, transplanted to a live venue without an out-of-process stop).

---

## 33. Decision Register

Verdicts: KEEP / REVISE / REPLACE / DEFER / DROP. Evidence labels as in §1. "Resolve" names the roadmap phase by which the decision must be settled.

| # | decision | current choice [RF] | verdict | evidence | main failure mode | alternative | resolve |
|---|---|---|---|---|---|---|---|
| 1 | Futures-first | USD-M perpetual is primary; spot a later lane (v3 §1) | **KEEP** (instrument) | SHORT native; taker fee ≈ half of spot; validated dry-run executor; funding/mark archives | funding cost dominating the economics of a slow directional rule | spot directional (no SHORT), carry-first | R0 |
| 2 | Strategy family | "directional LONG/SHORT" (a shape) | **REVISE** | no hypothesis stated; historical negatives narrow; power favours a panel endpoint | a vocabulary, not a mechanism, gets preregistered | one falsifiable panel hypothesis (§14) + optional carry lane | R3 |
| 3 | BTC-only vs multi-symbol | multi-symbol-capable gen4 (v3 Change B) | **KEEP** | breadth is the only route to effective sample size; single-asset history is spent | multiplicity and ops cost grow with K | BTC+ETH only (too little ESS) | R2 |
| 4 | Symbol count | 12–20 (v3) | **REPLACE** with a procedure | number precedes measurement; effective N ≈ 1.4–3 for time-series, ≈ K−1 cross-sectional | frozen guess; retrospective selection | rank by 90-day volume as of a pre-boundary date; K from measured effective N; provisional 10–12 | R2 → R4 |
| 5 | Microstructure subset | 2–3 full-depth symbols | **REVISE** | depth serves execution measurement, not campaign-1 prediction; 99% of storage today is bookTicker | terabytes for no scientific use | BTCUSDT + ETHUSDT depth5/20 day one; diffs deferred | R2 → R4 |
| 6 | OHLCV14 baseline | control feature set | **KEEP** as control, not as hypothesis | mean-zero after costs everywhere; causal | mistaking the control for the candidate | — | — |
| 7 | MTF / MTC | "mandatory", "locked" hierarchy (v3 §7, §21) | **REVISE** (multi-timeframe capability required in the runtime; hierarchical MTC as a scientific model not required for campaign 1) | P5/P6/P7 narrow negatives; as-of machinery sound; hundreds of design DoF | leakage via `available_time`; state explosion; a runtime limited to one timeframe | generic multi-clock causal snapshot; ≤ 2 slow closed-bar variables in the deciding set | R3, R7 |
| 8 | SMC / chart structure | retired feature families in tree | **DROP** from active use (keep history) | P2b/P2c negative; overlap with OHLCV14 | vocabulary re-entering a preregistration | — | — |
| 9 | Target type | 3-class dead-zone direction with 20 bps baked in | **REPLACE** | classification discards magnitude; threshold is the alpha; same-close fill | validation-mined threshold decides the sign | vol-standardised executable forward return; secondary binary with analytic threshold | R3 |
| 10 | Prediction horizon | 6 bars (6 h at 1 h; 6 min at 1 m) | **REPLACE** with the κ rule | cost/vol arithmetic; fast clocks foreclosed by the label | horizon by convenience | shortest H with median κ_H ≤ κ*, funding-aligned; likely 4–8 h | R3 |
| 11 | Decision frequency | per bar | **REVISE** | overlap ≈ H× nominal N | SE too small by √H | 1 h clock with overlap modelled, hysteresis so holding ≈ days | R3 |
| 12 | Logistic/linear baseline | LR untuned as one of three families | **KEEP** as the Tier-1 deciding family | only family with sign-consistent controls; interpretable; freezable coefficients | BLAS non-reproducibility if refit | freeze coefficients, not the fitting procedure | R3 |
| 13 | XGBoost | historical deciding family | **KEEP as optional single challenger** (m = 2) or exploratory | P2a +0.007 indistinguishable from zero; least stable control | promoted because it "won" on burned blocks | — | R3 |
| 14 | Deeper models | MTST retired; Tier 3 "if earned" | **DROP** for this roadmap | 72k parameters on ~500 independent windows; lost 20/20 | reintroduced as a milestone | reopen only under the three conditions (§20) | — |
| 15 | Cost model | flat 20 bps retired in text, alive in `TargetSpec`; config fees in demo | **REPLACE** | §23 | optimism at fast clocks, spot fee understated | per-symbol deciding cost semantics (fee, spread, executable price, latency, slippage envelope, funding, turnover, reject/partial rule, impact cap) frozen as hashed code in R3; identical in R10 and R11 | R2 → R3 |
| 16 | Recorder | in-house gen3 single-symbol | **KEEP + REVISE** (generalise; silence watchdog; incremental last minute; atomic parquet) | sound provenance; single symbol; in-place parquet write | multi-stream generalisation done ad hoc | adopt a library (loses provenance) | R1, R2 |
| 17 | PR #76 generalisation | archive reconciliation + coverage on a stale branch | **KEEP + REWORK** with both-outcome branches | fail-closed design; 129 commits behind; conflict file | merged as-is onto a stale base, or the roadmap waits on its VPS | rewrite (wasteful) | R5 |
| 18 | Aegis | sole risk authority; persisted; injected clock; several rules unreachable | **KEEP + REVISE** (scope; portfolio caps; wire or remove dead rules; staleness guard outside the decision path) | evaluate_entry is the only door; reductions ungated; dead limits | limits that enforce nothing; equity from the demoted ledger | a second authority (never) | R1, R8 |
| 19 | Target-position architecture | prediction emits target; execution owns how; Aegis owns risk — per order, single hedge | **KEEP the abstraction; REVISE to a per-symbol `TargetPositionSet` under portfolio caps** | correct separation; cross-sectional endpoint needs a set | Aegis approving legs, not portfolios | — | R8 |
| 20 | Execution engine | in-house dry-run executor, MARKET only, one-way, 1× | **KEEP** for demo and campaigns; a separate authenticated adapter later | 16 hashed invariants; idempotent events | framework adoption hiding semantics | nautilus/hftbacktest (deferred) | R8, R14 |
| 21 | Carry legacy | two-leg runtime is the only actionable rule; v3 demotes to a later lane | **KEEP** as a SECONDARY, OPTIONAL parallel economic lane in its own process; never delays or gates the primary lane; primary wins on conflict | only mechanism with reachable economic evidence; runtime exists | orchestration by stealth; priority by stealth | drop (loses cheap evidence) | R0 (election), R4 (deadline), R10 |
| 22 | Single-leg futures runtime | none (V3-8 roadmap only) | **BUILD** (R8), multi-symbol, one ledger | the primary lane cannot run without it | built single-symbol and rebuilt later | — | R8 |
| 23 | Demo runner | bounded catch-up pass; carry-shaped; cannot start a campaign | **REVISE → daemon** with the R1 repairs | §§10, 11 | soak on a runner that skips most minutes | — | R1 |
| 24 | Data store | NDJSON.gz raw + parquet normalised + JSON state files | **KEEP**; DuckDB as an optional reader later | immutability and auditability | a database becomes a second authority | ClickHouse/QuestDB | — |
| 25 | CI | Actions by tag; lock without hashes; two OS legs; integrity checks; smoke | **KEEP + REVISE** pins; add replay-determinism and synthetic-soak jobs | supply-chain posture | silent dependency drift changing replay bytes | — | R1 |
| 26 | Deployment | one VPS; hardened systemd; compose alternative | **KEEP**; separate users and state roots; off-host dead-man; backups | scale | alerts on the host they watch | containers/k8s | R1, R14 |
| 27 | Orchestration | deferred until ≥ 2 eligible families | **KEEP, tightened** to ≥ 2 CONFIRMATORY families; portfolio caps earlier in Aegis | §29 | premature allocation logic | — | R13 |
| 28 | Live venue approach | none; Freqtrade path live-capable | **DROP Freqtrade; DEFER adapter to R14** as a thin separate package | the only live-capable path is retired code | live route surviving in the tree | framework live modules | R1, R14 |
| 29 | Campaign design | information + economic co-primary, 182 d | **REPLACE** with two-stage in one accrual + separately sized confirmation of the identical object; N VALID blocks with a frozen extension cap and a NOT EVALUABLE outcome | §22 | underpowered economic floor read as "no value" | economic-first (underpowered), information-only (invites a second selection) | R3 |
| 30 | Evaluation instrument | fold-count gates, grid threshold, no inference | **DROP for prospective use; KEEP as history** | §21 | inherited instrument | frozen three-layer inference module | R3 |
| 31 | Governance mechanics | ordering guarantees; firewalls | **KEEP + REVISE** (classes, no-peek, interim looks, ≥ 7-day gap, disclosure block) | §13 | prose promotion of exploratory work | — | R0, R3 |
| 32 | Leverage | 1× pinned | **KEEP** through every phase of this roadmap | liquidation approximation; slack guard | leverage creep after a canary | — | R15+ |

---

## 34. Ranked Critical / High / Medium / Low Risks

Ranking combines likelihood, severity and detectability for the programme as it would proceed under v3 without this proposal. [INF]

**CRITICAL**
1. Opening a prospective campaign with the historical evaluator or an unfrozen inference method (false alpha) — §§21–22, R3.
2. Transplanting the current runtime's staleness, cadence and dispute semantics into a live adapter without an out-of-process stop and exchange-side reconciliation (money) — §§10–11, R1/R14.
3. Interim peeking and candidate/universe substitution during accrual (false alpha) — §27, R10.

**HIGH**
4. The 182-day co-primary economic floor failing for lack of power and being read as "no economic value", or an economically trivial IC being promoted — §22, R3.
5. Endpoint/runtime incoherence (a validated cross-sectional candidate executed as a single-symbol time-series rule) — §14, R3/R8.
6. Operator hand-edit of state files under a dispute with no clearing path — §11, R1.
7. Cost model optimism per symbol (spread, fee tier, funding cadence) surviving into the economic screen — §23, R2/R3.
8. Correlated panel collapse with per-symbol caps only — §29, R8.
9. Leakage through a snapshot without `available_time` in a live runtime — §16, R7.
10. Leverage creep after a profitable canary — §30, R15/R18.

**MEDIUM**
11. Silent-hold on recorder death in the demo (position-neutral in dry-run; material in live) — §10.
12. Catch-up cadence skipping minutes; parity structurally diverging — §10.
13. Crash windows ending a soak/campaign; disk-full fail-silence — §11.
14. Recorder in-place parquet write racing the runner — §11.
15. Dead Aegis limits (funding veto, loss streak, cooldown) giving false comfort — §7.
16. Valuation-price inconsistency (close vs mark) inherited by the single-leg runtime — §9.
17. Live-vs-replay funding-booking minute divergence — §11.
18. Dependency/action/image drift changing replay bytes — §31.
19. PR #76 merged onto a stale base or the roadmap blocked on its second campaign — §12.
20. MTC hierarchy built before any prospective signal exists — §16.

**LOW**
21. Decision-log day boundary and Aegis-day off-by-one-minute — §§9–10.
22. Tautological identity check overstated in comments — §9.
23. Spot base-asset fee unmodelled (carry lane only) — §9.
24. Stale documentation after PR #95 — §10.
25. HOLD-majority documentation defect — §15.
26. Down recorder leaving no evidence record — §11.


---

## 35. What Must Happen Next

1. **R0 — Owner decision on this proposal** (adopt, adopt in part, or decline), recorded in a governance PR with the exact SHA; dispose of PR #76 under the R5 branch rule; record the finite budget.
2. **R1 — Runtime integrity remediation**, as separate reviewed engineering PRs R1-a … R1-o in the order given in §37 R1: source-identity/CAMPAIGN self-check; AEG-1 persisted-equity re-seeding; `risk.json` continuity; daemon/service semantics; clock hardening (including the injected operational clock); real staleness detection from the operational clock; recorder/runner cadence with the funding-settlement-minute rule; atomic parquet publication and recorder down/up evidence; dispute lifecycle and crash/restart semantics (including disk-full behaviour); replay-parity policy for `seq` and `OPERATOR` with deterministic risk fields re-included in the hash; unreachable Aegis rules wired or removed; unified valuation price; Freqtrade deletion; supply chain and operations (SHA-pinned Actions, hashed lock, digest-pinned images, off-host dead-man check, chrony, backup/restore drill, separate users); evidence-class labels and the verifier check. Acceptance: 72 h SOAK on an engineering root with a planned restart and a `SIGKILL`, PARITY on every day, zero `SKIPPED_STALE` minutes while the recorder was healthy.
3. **R2 — gen4 engineering preflight on a second host**, ≥ 30 days, Tier A for ~20 candidates and Tier B for BTC/ETH, producing the measured rates, the correlation matrix and effective symbol count, the per-symbol cost envelope and the archive-availability table.
4. **R3 — Scientific design and the frozen power module**, independent review, ≥ 7 days before any boundary.
5. **R4 gen4 contract freeze** (with the Lane B election settled), then **R5 reconciliation generalisation** and the production recorder's ≥ 30-day pre-activation qualification, accruing in parallel with the next item.
6. **R7 (multi-clock causal `MarketSnapshot` and the frozen feature function; contract-agnostic parts may start early) → R8 (multi-symbol single-leg runtime with one ledger and Aegis set scope) → R9 soak** (≥ 14 days, drills, PARITY every day, shadow economics sealed).
7. Only after R3, R4, R5, **R9 PASS**, the qualification and the independent boundary review: **R6 boundary activation → R10**. SOAK PRECEDES PROSPECTIVE ACTIVATION.


---

## 36. What Must Not Happen Next

- Do not activate any prospective boundary (gen3 or gen4) before R3 (merged ≥ 7 days earlier), R4, R5, **R9 PASS**, the ≥ 30-day production-recorder qualification and the independent boundary review are all recorded; the soak precedes activation, never follows it.
- Do not freeze a gen4 contract, a symbol count, a horizon or an economic floor from the numbers in v3 §5–6; freeze them from R2 measurements and R3 arithmetic.
- Do not run the autonomous demo/soak on the current runner: it cannot start a campaign, cannot be resumed, cannot detect a dead feed, and skips most minutes; run it only on the R8 runtime, on an engineering or pre-activation root, and never read its shadow economics before R10 closes.
- Do not promote any historical lead (LR on fast clocks; LR OHLCV14 control; P7's mean-constituent behaviour) into the first candidate; the candidate's family is chosen a priori and disclosed as post-selection.
- Do not build the hierarchical MTC fabric (the multi-clock causal snapshot of R7 is required; the hierarchy as a scientific model is not), orchestration, an inference service, a model registry, or any live adapter now.
- Do not reopen P4-HOLD, Styx, P8, P13 or P14; do not split burned blocks; do not re-run the historical evaluator for any deciding purpose.
- Do not merge PR #76 as-is onto a stale base; do not let the strategic sequence wait on its second campaign.
- Do not treat a green demo, a green soak, a positive canary or a positive backtest as scientific evidence; do not let any lane's positive result cross the real-money boundary without all four eligibility conditions.
- Do not raise leverage above 1× at any stage covered by this roadmap.
- Do not design and read a campaign in the same session; do not shorten the ≥ 7-day preregistration-to-boundary gap.
- Do not change any deciding economic semantic, validity criterion, extension cap, evaluator or contract-hashed module between R10 and R11, except a parameter update applied exactly and logged under an update rule preregistered in R3 (route A, which may never make the deciding economics less conservative); any other such change ends the campaign as ABORTED and any successor is a new preregistration and a new campaign; a runtime-head change beyond ENGINEERING-FIX that touches no hashed module invalidates the block in which it lands and requires a re-soak before the next block is scored (§37.2).
- Do not let the carry lane delay, gate or outrank the primary directional futures lane.


---

## 37. COMPLETE REWRITTEN MASTER ROADMAP (PROPOSAL, adoption-ready; corrected 2026-09-14)

**Status:** PROPOSAL produced by an independent read-only audit and corrected by this corrigendum. It creates no scientific evidence, no prospective boundary, no alpha claim and no real-money authority. Adoption is an owner decision recorded in a governance PR (R0).

**Starting state [RF, as of the audit]:** `main` = `46921ef1206748c6b7304432a26c8295b7830e27` (merge of PR #95). PR #76 open/draft at `8a8f4a1`, 129 commits behind `main`, with a known conflict in `tests/test_recorder_no_network.py`. gen3 recorder contract with `prospective_from = null`. No prospective data. No campaign protocol frozen. P4-HOLD retired/unread; Styx sealed/unread; P8 withdrawn; four outer historical blocks BURNED. Real-money authority: NONE.

**Primary lane, stated once:** Binance USD-M perpetual futures, multi-symbol, single-leg, directional LONG/SHORT, at 1× leverage. Spot is reference data, a hedge leg where a lane needs one, and support infrastructure. The carry lane (Lane B) is SECONDARY and OPTIONAL: it never delays the primary lane, has its own hypothesis slot, is not a prerequisite for anything, is not orchestration, and yields on any resource conflict.

### 37.0 Phase map

```
ID   phase                                                     class                         irreversible?
R0   Adoption & freeze (owner)                                 governance                    no
R1   Runtime integrity remediation                             engineering                   no
R2   gen4 engineering preflight (separate host)                engineering + DIAGNOSTIC      no
R3   Scientific design, power module, campaign contract        PREREGISTERED design          yes (merge)
R4   gen4 contract freeze                                      data governance               no (until R6)
R5   Reconciliation generalisation (PR #76 → gen4)             engineering  [PASS/FAIL branches]  no
R7   Multi-clock causal MarketSnapshot + frozen feature fn     engineering (MTF-capable)     no
R8   Multi-symbol single-leg futures runtime + Aegis set scope engineering                   no
R9   Autonomous demo / soak                                    ENGINEERING evidence only     no
R6   Prospective boundary activation  (AFTER R9 PASS)          irreversible governance act   YES
R10  First prospective campaign (accrual, no-peek)             PROSPECTIVE evidence          YES (first scored day)
R11  Economic confirmation and continuation                    CONFIRMATORY evidence         no
R12  Strategy expansion governance                             science governance            no
R13  Portfolio / orchestration (conditional)                   engineering                   no
R14  Live-readiness qualification (Gates 2–3, Gate-4 text)     engineering + operational     no
================  REAL-MONEY AUTHORITY BOUNDARY (four conditions)  ================  YES
R15  Owner authorisation (Gate 4 executed)                     owner governance              no (revocable until an order exists)
R16  Canary (Gate 5)                                           operational                   YES (first order)
R17  Controlled real-money operation                           operational                   —
R18  Scale-up / scale-down governance (Gate 6)                 owner + operational           —
```

Listed in execution order. IDs are stable labels inherited from the audit; **R6 executes after R9** (Part B). Dependencies, exhaustively: R1 ∥ R2 (both need R0); R3 needs R2 (≥ 30 measured days); R4 needs R3 (and the Lane B election, deadline R4); R5 needs R4; R7 needs R3 (frozen feature function) and R4 (declared streams/clocks) for acceptance, ENG-START-EARLY for the generic snapshot type; R8 needs R1 and R7 for acceptance, ENG-START-EARLY for the runtime skeleton; R9 needs R8 and a running gen4 recorder (the production recorder's pre-activation root from R4, or the R2 preflight recorder if R2 records that it keeps running); the ≥ 30-day production-recorder qualification starts at R4's production deployment and accrues in parallel with R5 and R7–R9, but a day counts only once R5's accepted reconciliation has been run over it (offline recomputation, retroactive where needed), so the qualification record — produced by R5 — closes only after R5 acceptance; R6 needs R3 (merged ≥ 7 days earlier), R4, R5, **R9 PASS**, the completed qualification and the independent boundary review; R10 needs R6; R11 needs R10 PASS on both floors; R12 follows any R10/R11 outcome; R13 needs ≥ 2 CONFIRMATORY families; R14 needs R11 Gate 1 for the candidate that will trade and re-runs the R9 gates on the exact deployable head; R15–R18 are sequential behind the real-money boundary. Re-entry edges: R12 → R3 (a new candidate) → R7 feature-function PR → R9 re-soak on the new head → a new R6 with all eight inputs → R10; R12 (NOT EVALUABLE; the automatic same-design successor, at most once) → R9 re-soak → new R6 → R10; R6 → DECLINED → R12; R5 (a stream downgraded in a new contract version) → R4 re-freeze, restarting the qualification clock under the new hash and re-establishing R7/R9 acceptance if any declared stream or clock changed.

### 37.1 Governance classes and status labels used throughout

| class | meaning | may be reused for |
|---|---|---|
| ENGINEERING | tests, invariants, soak logs, replay parity, coverage | any later stage; never as alpha evidence |
| DIAGNOSTIC | descriptive statistics on engineering-root or pre-activation data (rates, spreads, correlations, coverage, cost envelopes) | design inputs before a boundary; never as a result |
| EXPLORATORY | any model/feature/rule evaluation on pre-boundary or engineering-root data | hypothesis generation; must be labelled; never promotable |
| PREREGISTERED | a hashed design committed and pushed before the data it governs exists or is read | governs exactly one campaign |
| PROSPECTIVE | evidence accrued at or after an activated `prospective_from` under a frozen contract and preregistration | promotion decisions |
| CONFIRMATORY | a second, independent prospective period under the same frozen candidate and the same deciding economics that does not contradict the first | eligibility for real-money consideration (Gate 1) |
| ADAPTIVE | any evaluation whose design was chosen after seeing outcomes from the same data | disclosure only |
| BURNED | the four outer historical blocks; P4-HOLD; anything read ≥ 2 times for deciding purposes | descriptive context, fixtures; never deciding |
| SEALED-DIAGNOSTIC | a hashed artifact that exists but may not be read before a named event: the soak's shadow economics until R10 closes; a NOT EVALUABLE campaign's VALID-block readouts until its successor closes or the candidate is retired; a DECLINED or ABORTED candidate's sealed material, which is never opened | opened only at the named event, then DIAGNOSTIC (or ADAPTIVE context where the label says so); never for descriptive context before it |

Block and campaign status labels (R10/R11): a scored block is **VALID** or **INVALID** (mechanical, blind to outcomes; INVALID blocks stay in chronology and reporting forever and are never a negative result); a campaign ends **PASS**, **NEGATIVE** (scientific), **NOT EVALUABLE / ENGINEERING FAILURE** (only: the maximum extension exhausted before N valid blocks), or **ABORTED** (an owner stop, or a change to a contract-hashed module during accrual: not PASS, not NEGATIVE, not NOT EVALUABLE; the reason recorded; the candidate's slot consumed; no same-design rerun; all readouts stay sealed). A candidate whose owner declines activation after a passed soak is **DECLINED** (slot consumed; sealed shadow economics never opened; re-election only via a new R3, a fresh soak and a fresh seal).

Every artifact directory carries an `evidence_class` field in its manifest; `tools.verify_research_state` refuses a front-door document that cites a non-PROSPECTIVE artifact as prospective evidence, a non-CONFIRMATORY artifact as confirmatory evidence, or an INVALID block as a result (new check; R1-o).

### 37.2 Git / PR policy (all phases)

Ordinary commits; ordinary merge commits; no amend/squash/rebase/force-push on `main` or on any branch carrying a preregistration, a boundary activation, an evidence freeze, a validity verdict or a negative result; immutable negative results; PR-specific exact-head CI required for every merge into `main`; every scientific/governance PR names its exact base SHA and the SHA it produces; large research PRs stay draft until independently audited; evidence directories are frozen with SHA256SUMS in the same PR that creates them; a failed acceptance, an INVALID block or a NOT EVALUABLE campaign is never replaced by a later one (a new attempt is a new directory with a new prospective declaration); preregistration PRs sit ≥ 7 calendar days between merge and first scored instant with one independent review recorded in the PR. **Head-freeze rule (R9 → R10 → R11):** the runtime head that passed R9 is the head that runs the campaign; the only permitted changes are ENGINEERING-FIX PRs (reviewed; no change to any contract-hashed module — feature function, model artifact loader, policy, cost model, endpoint evaluator, validity evaluator; replay parity re-proven on ≥ 3 prior days; recorded as an engineering event in the campaign log). Any other change to a non-contract-hashed module lands as an INVALID block and requires a re-soak (on an engineering recorder root or, read-only and unscored with its shadow economics sealed, on the prospective root) before the next block is scored; any change to a contract-hashed module ends the campaign as ABORTED — the object changed — and any successor is a new preregistration (route C). **Validity-before-readout rule:** a block's validity verdict is computed by the CI/verifier job, never by hand, from engineering records only; the job commits the verdict, its input manifest and its hash in the block's evidence PR before the readout job runs; that PR is reviewed by a person who certifies having read no readout of the block; a manually produced verdict is itself an invalidity criterion.

### R0 — Adoption and freeze

**PURPOSE:** record the owner's decision on this roadmap and the dispositions it implies.
**WHY:** nothing below is legitimate without an explicit adoption; the roadmap's budget and lane structure must be fixed before work starts.
**INPUT PREREQUISITES:** the completed audit and this corrigendum.
**EXACT WORK:** (a) a governance PR adopting or declining this roadmap by section, naming the audit SHA and this corrigendum; (b) disposition of PR #76 under R5's branch rule (it is not merged in R0); (c) a record that gen3 stays engineering-only and that its `prospective_from` stays `null`; (d) the finite budget: at most **two directional candidates** and at most **one carry candidate** across the programme before a mandatory stop-and-decide review, each candidate holding one first campaign (R10), at most one automatic same-design re-run after a NOT EVALUABLE outcome, and one continuation (R11), with ABORTED and DECLINED consuming the slot; zero deciding use of burned blocks; an engineering effort cap for R1–R9 set by the owner (the audit suggests ≤ 10 person-weeks); (e) the Lane B election opened (elect / decline / defer to R4); (f) the standing rule that the primary directional lane wins any resource conflict.
**TECHNICAL DESIGN:** documents only; regenerate front-door state blocks with the existing tooling.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** none.
**DATA NOT USED:** everything.
**TESTS / VERIFICATION:** `tools.verify_research_state`; CI on the exact head.
**ACCEPTANCE:** decision recorded with date and SHA; budget and lane rule visible in `docs/current_development_plan.md`.
**KILL / STOP CONDITIONS:** owner declines → Roadmap v3 stands unchanged and this proposal is archived as a document.
**EVIDENCE CREATED:** none.
**EVIDENCE NOT CREATED:** any.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none; no authority created.
**DEPENDENCIES:** none.
**NEXT MAY ASSUME:** an adopted plan, a fixed budget, the lane rule, and a recorded PR #76 disposition.
**PR BOUNDARIES:** one docs/governance PR.

### R1 — Runtime integrity remediation (engineering prerequisites)

**PURPOSE:** make the existing runtime honest before it grows.
**WHY:** the demo cannot start under the campaign profile, cannot run as a service, cannot clear its own disputes, cannot detect a dead feed, skips most minutes as stale, can halt a healthy campaign at a day roll, and cannot prove parity across a restart [RF §§7, 10, 11]. Growing a multi-symbol runtime on top would carry every defect forward.
**INPUT PREREQUISITES:** R0.
**EXACT WORK** (each item a separate reviewed engineering PR with a two-sided test; none changes scientific standing):
- **R1-a Source identity / CAMPAIGN self-check:** `tools/demo_run.py::_software` passes the checkout root to `source_identity()` and reads the mapping by key; a genuinely dirty tree is refused on CAMPAIGN, a clean tree passes; the committed campaign config must build a runner.
- **R1-b AEG-1 persisted-equity re-seeding:** `build_risk_engine` never calls `update_equity(capital)` after loading a persisted state; capital seeds equity only on first start; on restart the persisted equity is reconciled against the ledger and a disagreement is a dispute; two-sided tests: (a) a restart across a UTC day boundary after a ≥ 2% intraday mark move → no halt; (b) a persisted peak ≥ capital/(1 − max_drawdown_pct) → no halt on restart, while a genuine breach still halts; (c) replay PARITY across a day boundary with a restart (`risk.state_hash` equal).
- **R1-c `risk.json` continuity (AEG-4):** a missing or unreadable `risk.json` when the decision log already holds records is a dispute, not a default; the loaded snapshot is compared with the log's last `risk.state_hash`/HALT record; a `RECOVERY` record is written; the same continuity the store and ledger already have.
- **R1-d Daemon / service semantics:** a continuous READY loop (sleep to the next expected minute close + grace), graceful SIGTERM deferred past PERSISTENCE, heartbeat every 30 s, persistent metrics endpoint; the systemd unit becomes a real long-running service; `RunnerDown` becomes a liveness alert; process-per-pass supervision is removed.
- **R1-e Clock hardening:** RiskEngine and executors refuse construction without an injected clock under CAMPAIGN/SOAK profiles; no `time.time` default reachable from the demo path; the daemon's operational clock (used only for staleness, heartbeats and the stall tick) is itself injected and distinct from the decision clock; a clock-hostility test.
- **R1-f Real staleness detection:** the age is computed as the injected operational wall clock minus the newest available minute close (or as the recorder heartbeat age), never as the decision clock compared with itself (today the comparison is identically zero by construction); compared against `max_data_delay_s` in the runner's READY gate, outside the per-order decision path; a stale feed produces a recorded `FEED_STALLED` halt (positions held; funding and liquidation checks continue on the last valid state via a stall tick driven by the operational clock), automatic continuation when data resumes; two-sided test: a frozen recorder with advancing wall time halts within threshold + grace, a healthy feed never halts; `max_data_delay_s` is enforced here or removed from the schema.
- **R1-g Recorder / runner cadence:** the recorder publishes the last closed minute incrementally (seconds, not the 300 s full-day cadence); the runner decides every closed minute since its last decided minute (no three-minute cap) or records each skip with a reason; a minute whose close coincides with a funding instant is decidable only once its settlement row exists in the recorder output (or the recorder has marked the instant as no-settlement) — a late settlement row defers the decision rather than skipping the booking; two-sided test: live and replay book the same settlement in the same minute; acceptance: zero `SKIPPED_STALE` minutes while the recorder is healthy.
- **R1-h Atomic parquet publication and recorder evidence:** temp-file + fsync + rename for every parquet and manifest write; an in-process silence watchdog that reconnects on stream silence; persisted `recorder.down` / `recorder.up` records in the recorder's own log so a dead recorder leaves evidence; gen3's contract hash is unchanged by any of this (gen3 stays engineering-only).
- **R1-i Dispute lifecycle and crash/restart semantics:** every dispute kind has exactly one clearing path (`resolve` reachable from the CLI, requires a note, re-books what it must, refuses what it cannot); `resume` and `resolve` succeed from the CLI via `start()` with log-tail clock seeding; `flatten` on CAMPAIGN without `allow_dirty` and without persisting a spurious halt reason; `HedgedPosition.correct()` wired with a bounded timeout or deleted; one-legged positions liquidation-checked per leg; `resume` ordering fixed; the state-machine crash harness (kill at every persistence step, recover, assert consistency) added to CI, including a disk-full fault; on any persistence failure (ENOSPC included) the runner exits non-zero after attempting to append the halt reason to a pre-allocated emergency record, so a restart never lacks a trace.
- **R1-j Replay parity policy:** decide `seq` (per-kind sequence or exclusion of operational kinds) and `OPERATOR` (deterministic replay counterpart from a committed operator-action file); re-include deterministic risk fields in the hash; the 48 h synthetic fixture reaches PARITY with a restart and an operator action.
- **R1-k Unreachable Aegis rules:** the funding-cost entry veto made reachable (pass `funding_rate` to `execute_target`) or removed; loss-streak and cooldown limits either driven by a real `record_trade_result` caller or removed from the config schema; each wiring/removal has a breach-refused and non-breach-allowed test; the unsafe-default detector (a configured limit that no code path enforces) added to CI.
- **R1-l Unified valuation price:** one documented valuation price (mark) for equity, drawdown and liquidation; the close-vs-mark inconsistency removed from the ledger path that R8 will inherit.
- **R1-m Retire the live-capable legacy:** delete the Freqtrade path, strategies, exchange configs, its Dockerfile, the three tests that import it and the `trade` extra; keep `make smoke` by moving it off `nn.infer_service` or retire it with the inference service.
- **R1-n Supply chain and operations:** SHA-pin GitHub Actions; `--hash` the lock; digest-pin base images; an off-host dead-man check fed by recorder and runner heartbeats; documented chrony requirement with the existing skew alert; backup/restore drill for the recorder root and the state directory; separate Unix users and state roots for recorder and runner.
- **R1-o Evidence-class labels:** the `evidence_class` manifest field, the 37.1 status labels, and the `verify_research_state` checks described there.
**TECHNICAL DESIGN:** no new abstractions; fixes inside existing modules; the four-item minimum test architecture (state-machine crash harness; clock hostility; replay-determinism CI job; unsafe-default detector) plus a synthetic-soak CI job.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** synthetic fixtures; gen3 engineering recorder data.
**DATA NOT USED:** burned blocks, P4-HOLD, Styx, any sealed data, for any purpose.
**TESTS / VERIFICATION:** the unit and two-sided tests named per item; the four harnesses; exact-head CI per PR.
**ACCEPTANCE:** CI green on the exact head of each PR; a 72 h SOAK-profile run unattended on an engineering recorder root with one planned restart and one `SIGKILL`, PARITY on every day, zero `SKIPPED_STALE` minutes while the recorder was healthy, zero manual state edits.
**KILL / STOP CONDITIONS:** none scientific; an item whose fix would require a design change beyond its module is split into its own reviewed PR rather than widened.
**EVIDENCE CREATED:** ENGINEERING.
**EVIDENCE NOT CREATED:** any scientific evidence.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R0.
**NEXT MAY ASSUME:** a service-shaped runner whose decisions replay across restarts, whose halts and disputes are clearable from the CLI, whose Aegis limits all enforce something, and a recorder whose files are atomic and whose last minute is fresh.
**PR BOUNDARIES:** R1-a … R1-o as separate PRs; no PR touches a scientific artifact or a contract hash.

### R2 — gen4 engineering preflight (separate host)

**PURPOSE:** measure before freezing.
**WHY:** v3's 12–20 symbols and 182 days are unmeasured guesses; storage was already under-estimated 24× once [RF].
**INPUT PREREQUISITES:** R0; a second host that is never the PR #76 VPS; no interaction with the gen3 acceptance environment.
**EXACT WORK:** multi-symbol capture under an ENGINEERING contract id (`gen4-preflight`; `prospective_from` absent by schema) of Tier A streams (1 m klines, markPrice@1s, bookTicker, aggTrade, funding, OI via REST, forceOrder, daily exchangeInfo snapshots) for ~20 USD-M perpetuals chosen by 90-day median USD volume as of a named date, plus Tier B (depth20@100ms or diff depth with snapshot bootstrap) for BTCUSDT and ETHUSDT, plus — if Lane B is elected or undecided — spot kline_1m and spot bookTicker for BTCUSDT and ETHUSDT, for ≥ 30 consecutive days. Measure: events/s, bytes/s raw and gz, CPU, RAM, disk/day, projected 182 d/365 d, replay throughput, recovery time, reconnect behaviour, REST weight use, archive availability per stream, missingness, depth sequence-continuity rate, clock skew, cross-stream consistency (kline vs aggTrade OHLCV; bookTicker vs depth top). Compute the cross-symbol daily-return correlation matrix and the effective number of symbols; compute the per-symbol cost envelope (median/p90 half-spread at decision instants, realised trade-through for a reference size, fee tier, funding interval).
**TECHNICAL DESIGN:** the gen3 recorder generalised to (symbol, stream)-keyed files; raw NDJSON.gz per stream-day immutable; normalised parquet regenerable; contract hash + code version + schema version stamped in every day manifest; the R1-h atomicity and watchdog carried over.
**SCIENTIFIC DESIGN:** none — DIAGNOSTIC only; no model, no label, no return statistic beyond correlation, volatility and cost envelopes.
**DATA USED:** live public streams on the preflight host.
**DATA NOT USED:** burned blocks; no outcome-bearing statistic of any candidate.
**TESTS / VERIFICATION:** synthetic-server tests for new parsers; parity tests full-vs-incremental normalisation per stream; manifest checksum verification.
**ACCEPTANCE:** a preflight report with every measurement above, frozen with SHA256SUMS, `evidence_class = DIAGNOSTIC`.
**KILL / STOP CONDITIONS:** ≥ 2 of the 30 days failing coverage thresholds for core streams → fix operations before R4; Tier B cost beyond budget → Tier B shrinks or is deferred (recorded).
**EVIDENCE CREATED:** DIAGNOSTIC.
**EVIDENCE NOT CREATED:** prospective; exploratory candidate evaluations.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R0.
**NEXT MAY ASSUME:** measured rates, a correlation matrix and effective-N, per-symbol cost envelopes, archive verification classes per stream.
**PR BOUNDARIES:** recorder-generalisation PRs (offline core first, then live collection), then the preflight report PR.

### R3 — Scientific design, power module and campaign contract

**PURPOSE:** choose an answerable question and freeze it, including everything that will decide R10 and R11.
**WHY:** the historical programme measured noise with an instrument that could not see its effects; v3 says power before boundary but defines no method; a confirmatory period must replicate, not re-score.
**INPUT PREREQUISITES:** R2's diagnostics (correlation matrix, cost envelopes, rates, archive classes); the standing negative priors; the Lane B election status.
**EXACT WORK:**
1. *The frozen candidate (Lane A):* instrument (USD-M perpetuals); universe rule and K (§18 procedure, provisional 10–12); decision clock (1 h); horizon H by the κ rule (likely 4–8 h, funding-aligned); the deciding feature set = features on the decision clock plus **at most two preregistered slow closed-bar variables** (e.g., 1 d realised volatility; 1 d or 4 h trend sign), each with its clock, `available_time` rule and maximum age declared; model class with fixed hyperparameters (Tier 0 references; Tier 1 linear with λ from a predeclared blocked CV on pre-boundary engineering data; optional Tier 2 challenger, m = 2); calibration map; abstention (analytic threshold) and hysteresis; target-position mapping (per-symbol side/quantity/confidence/abstain under portfolio caps); the coherence statement (cross-sectional endpoint ↔ panel policy ↔ multi-symbol single-leg runtime).
2. *Frozen deciding economic semantics (used identically in R10 and R11), as hashed code:* per-symbol fee tier and BNB-discount assumption dated to the fetch; spread rule (recorded bookTicker touch at the decision minute close); executable-price rule (buy at ask, sell at bid, at decision time + Δ); latency Δ (frozen constant; the quote used is the first at or after t + Δ); slippage: a conservative per-symbol envelope (p90 measured trade-through for the reference size from R2, as a frozen constant or a frozen deterministic function of size and spread); funding by settlement at each recorded funding timestamp with the symbol's interval (non-8 h symbols excluded from Lane A); turnover accounting (fees and spread on every change of target notional; hysteresis inside the policy); reject/partial-fill treatment (dry-run venue model: full fill at the executable price; an engineering-rejected order is a missed decision with no phantom fill; the rule is frozen); impact rule (the campaign's dry-run notional per symbol is frozen below the measured depth threshold, so impact is outside the deciding model; anything above is stress only). ±50% cost sensitivity is reported, never deciding.
3. *The ESS/MDE/inference module,* frozen as code and report (§22's three layers: analytical N_eff; stationary bootstrap with automatic block length; whole-pipeline null simulation calibrating the exact decision rule including Holm over m ≤ 2), run on R2 engineering data and null controls only; outputs: N_eff, MDE (IC, bps net, annualised Sharpe), duration to floor, **the target number N of VALID scored blocks**, block length (calendar month, UTC), the alpha-spending interim-look rule, and the R11 continuation size N′ (or its deterministic sizing rule in the measured R10 effect).
4. *The campaign contract:* endpoints and floors (§27); blocks; **block-validity rules**: (i) mechanical invalidity criteria (any required stream below its coverage threshold on more than the frozen number of days in the block; a replay-parity failure on any day; a runner halt not recovered within the frozen window or any halt without a record; a recorder identity or contract-hash change; a runtime head change other than an ENGINEERING-FIX PR; any manual state edit; a symbol delisting/halt → that symbol's block invalid, the whole block invalid if the contract's minimum symbol count is not met); (ii) the evaluator `nn/prospective/validity.py`, reading engineering records only and no prices, scores, positions or PnL; (iii) validity-before-readout ordering; (iv) the deterministic calendar-extension rule (contiguous blocks until N VALID blocks); (v) the maximum extension E (proposal: E = ⌈N/2⌉ blocks; frozen); (vi) exhaustion → NOT EVALUABLE / ENGINEERING FAILURE; (vii) the pre-committed successor rule: at most one automatic same-design successor after a NOT EVALUABLE outcome (new hash, new boundary, re-soak first, no pooling), taken once the engineering cause is fixed and never decided on outcomes; the predecessor's VALID-block readouts stay SEALED-DIAGNOSTIC until the successor closes or the candidate is retired; a second exhaustion is terminal for the design and consumes its slot; an owner stop, or a change to a contract-hashed module during accrual, is ABORTED (slot consumed, no same-design rerun, readouts sealed); (viii) INVALID blocks permanently reported; stopping rule; minimum duration; no-peek rule; engineering-invalidation vs scientific-failure split; PASS/NEGATIVE statements; negative controls; the historical-adaptivity disclosure block; the head-freeze rule; **the R11 rule**, with its non-inferiority statement and N′: R11 uses the identical candidate and identical deciding economics; new execution/testnet/live-operational information enters only (A) through a deterministic update rule preregistered here that can never make the deciding economics less conservative than R10's (e.g., "replace the slippage envelope with max(frozen p90, p90 of ≥ 500 testnet fills) computed by frozen code"), (B) as non-deciding stress/diagnostic analysis, or (C) by a new preregistration and campaign.
5. *Lane B contract (only if elected):* own hash; BTCUSDT (ETHUSDT optional); always-on or funding-gated two-leg carry rule with parameters frozen from pre-boundary funding statistics; endpoint per §27 Lane B; m = 1 in its own slot; the same validity, extension, no-peek, head-freeze and shadow-economics-seal rules and the same DECLINED consequence; a Lane B elected after this PR gets its own preregistration PR (own hash, own review) merged ≥ 7 days before R6, otherwise it waits for its own later contract and boundary.
6. *Independent pre-open review* recorded in the PR; **≥ 7 days** between the preregistration merge and the boundary.
**TECHNICAL DESIGN:** `nn/prospective/` package: contract JSON hashed as P13's pattern; the ESS/MDE code; the endpoint evaluator that reads only the recorder root and the decision log; the validity evaluator; the cost-model code imported by both the runtime and the evaluator so the same bytes decide.
**SCIENTIFIC DESIGN:** per §27 as corrected (two-stage in one accrual: information primary, economic falsification screen gating R11).
**DATA USED:** R2 engineering-root data for null calibration, cost measurement, and the one-time predeclared λ-selection blocked CV and coefficient fit of the Tier 1 candidate (procedure and outputs hashed, recorded in the historical-adaptivity disclosure block, labelled EXPLORATORY, never a go/no-go input).
**DATA NOT USED:** burned blocks for any deciding purpose; no evaluation of the candidate's endpoint or economic screen on any pre-boundary data other than the predeclared λ-selection CV and the null-calibration controls, and none of it reported or used as evidence.
**TESTS / VERIFICATION:** hash tests; two-sided synthetic controls for the endpoint code (planted signal detected; null passes at the nominal rate); two-sided controls for the validity evaluator (a planted coverage gap → INVALID; a clean block → VALID; the evaluator has no import path to price or PnL data); two-sided controls for the cost code; a two-sided coherence control (on a synthetic block the evaluator's economic readout recomputed from the decision log equals the runtime ledger's simulated PnL to the cent, and a planted mismatch between the policy's target set and the evaluator's assumed positions is detected).
**ACCEPTANCE:** contract merged with hash recorded; review recorded; power report shows MDE ≤ the floor at the planned N (or N raised until it does); the deciding economic code hashed into the contract; the R11 rule present.
**KILL / STOP CONDITIONS:** no design reaches adequate power within an accrual of ≤ 18 months including the maximum extension → do not proceed to R4 for that design; record it as declined.
**EVIDENCE CREATED:** PREREGISTERED (design); DIAGNOSTIC (power report).
**EVIDENCE NOT CREATED:** any result.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R2 (≥ 30 days). ENG-START-EARLY: the module code and contract templates may be written during R2; acceptance needs R2's outputs.
**NEXT MAY ASSUME:** a frozen, reviewed, powered contract whose deciding code, validity rules, extension rule and R11 rule cannot change without a new preregistration.
**PR BOUNDARIES:** one preregistration PR per lane (draft until reviewed); the `nn/prospective/` code PRs precede it.

### R4 — gen4 contract freeze

**PURPOSE:** create the acquisition identity for the prospective recorder.
**WHY:** a stream absent from the contract cannot serve the campaign; verification classes must match what first-party archives actually publish [EF, audit §§12, 17].
**INPUT PREREQUISITES:** R2 report; R3's universe and streams; the Lane B election (deadline: this phase).
**EXACT WORK:** the contract JSON: exact symbols; exact streams per tier with verification class — ARCHIVE_RECONCILED (klines, aggTrades, markPriceKlines, indexPriceKlines, premiumIndexKlines, fundingRate via the monthly archive with its latency rule, metrics at 5-minute cadence, bookDepth as a sampled band file only); HEALTH_GATED (markPrice@1s; bookTicker, whose daily archive ended 2024-03-30; depth diffs/snapshots; all self-attested with health metrics and no denominator); DESCRIPTIVE_ONLY (forceOrder, which carries only the largest liquidation per symbol per second; OI/long-short REST histories; exchangeInfo snapshots); **plus spot kline_1m and spot bookTicker for Lane B's symbols as HEALTH_GATED capture iff Lane B is elected**; minute-key and missingness semantics; a per-minute provenance flag in the normalised schema (LIVE / REST_FILLED / LATE) with a test that a REST-filled minute is so labelled; storage layout (raw immutable, normalised regenerable); checksum scheme; metadata policy (daily exchangeInfo snapshot; leverage brackets; fee schedule assumption recorded); source identity; `prospective_from: null`.
**TECHNICAL DESIGN:** contract schema versioned; the contract hash binds code version and schema version in every day manifest.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** R2 measurements.
**DATA NOT USED:** any outcome data.
**TESTS / VERIFICATION:** contract-schema tests; a test that every stream named by R3's feature function and cost model is in the contract.
**ACCEPTANCE:** independent plan review; hash recorded; production deployment of the recorder under this pre-activation hash begins; the ≥ 30-day qualification clock starts here and accrues in parallel with R5 and R7–R9, but a day counts as qualifying only once R5's accepted reconciliation has been run over it (offline recomputation, retroactive where needed), so the qualification record closes only after R5 acceptance; any coverage or reconciliation failure restarts the clock.
**KILL / STOP CONDITIONS:** none.
**EVIDENCE CREATED:** none.
**EVIDENCE NOT CREATED:** any.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R2, R3.
**NEXT MAY ASSUME:** a frozen contract that R6 will activate; a stream added later cannot be used by the campaign whose contract lacks it, and adding a stream to a new contract does not invalidate an existing campaign.
**PR BOUNDARIES:** one contract PR; the production-deployment PR.

### R5 — Reconciliation generalisation (PR #76 → gen4), with explicit branches

**PURPOSE:** recorder validity for gen4.
**WHY:** PR #76's reconciliation and coverage code is fail-closed and well-tested offline but single-symbol and 129 commits stale [RF]; gen4 needs (symbol, stream)-keyed reconciliation with the archive classes of R4.
**INPUT PREREQUISITES:** R4; the outcome of PR #76's second acceptance campaign, whichever it is (the roadmap does not wait for it).
**EXACT WORK:**
- *IF campaign #2 PASSES:* it establishes only that the gen3 recorder captured and reconciled two days on that VPS (ENGINEERING). Merge PR #76 on a fresh current-`main` lineage with an ordinary merge (resolve the `tests/test_recorder_no_network.py` conflict; exact-head CI); then generalise `reconcile.py`/`coverage.py` to (symbol, stream) with the contract's `minute_indexed_required` as the single source; add archive layouts for aggTrades, markPriceKlines, indexPriceKlines, premiumIndexKlines, metrics, bookDepth; keep HEALTH_GATED semantics; keep the funding monthly-archive latency rule (`FUNDING_SCHEDULE_UNAVAILABLE` is not an outage). No boundary activation follows from the pass.
- *IF campaign #2 FAILS:* classify the cause (recorder outage; archive-latency mis-read; VPS operations; reconciliation defect). Keep the offline code and tests; fix the cause in a reviewed PR; harden the preflight (silence watchdog, disk/CPU headroom, supervisor); no campaign #3 until a ≥ 7-day clean SOAK on the same host; campaign #3 is declared prospectively and cannot substitute failed days. gen4 proceeds on its own host regardless; PR #76's code is integrated into gen4 once its offline tests pass on gen4 layouts even if VPS acceptance is pending; the gen4 recorder's own qualification (R4 → R6) is the operative gate.
- *In both branches:* R1–R4, R7–R9 do not wait for the outcome.
**TECHNICAL DESIGN:** reconciliation as a pure offline recomputation over raw files and archives; every `ArchiveOutcome` fail-closed.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** gen4 preflight and pre-activation raw files; first-party archives.
**DATA NOT USED:** any outcome data.
**TESTS / VERIFICATION:** offline determinism tests; real-source verification of every ARCHIVE_RECONCILED stream on ≥ 2 real days; fail-closed tests per outcome.
**ACCEPTANCE:** the above green on the exact head; reconciliation running on the production pre-activation recorder; the qualification record (≥ 30 qualifying days: per-day coverage, reconciliation agreement and source identity of the production recorder) frozen with SHA256SUMS once the clock completes.
**KILL / STOP CONDITIONS:** a reconciliation defect that cannot be made deterministic → the affected stream is downgraded to HEALTH_GATED in a new contract version before R6, never silently; a new contract version is a new R4 freeze: the qualification clock restarts under the new hash, and R7/R9 acceptance is re-established if any declared stream or clock changed.
**EVIDENCE CREATED:** ENGINEERING, including the qualification record that R6 inputs (5)–(6) consume.
**EVIDENCE NOT CREATED:** any scientific evidence.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R4.
**NEXT MAY ASSUME:** every ARCHIVE_RECONCILED stream is reconciled daily and deterministically; health-gated streams carry health metrics; the qualification record exists or its clock is running.
**PR BOUNDARIES:** the PR #76 merge (or its fix PR); the generalisation PRs per stream family; the qualification-record PR.

### R7 — Multi-clock causal MarketSnapshot and the frozen feature system

**PURPOSE:** one as-of-correct input snapshot per decision, generic over every clock the contract declares.
**WHY:** the live runtime must carry the causal semantics the research code already has (`nn/mtf.py`), and the runtime must not be architecturally limited to one timeframe — while the first campaign's science stays narrow.
**INPUT PREREQUISITES:** R3 (frozen feature function and declared clocks), R4 (declared streams). ENG-START-EARLY: the generic snapshot type and the closed-bar derivation may be implemented during R2–R4 against a schema of declared clocks.
**EXACT WORK:** (a) `MarketSnapshot`: for every input and every timeframe, `source_time`, `available_time` (= close time + the observed recorder processing delay from the raw receipt stamp), `as_of`, `age`, `staleness`, `complete`, `missing`; (b) closed-bar clocks (e.g., 5 m, 15 m, 1 h, 4 h, 1 d) derived causally from the base 1 m stream — a bar exists only when every constituent minute exists and the bar's `x == true` close has been received; (c) the feature function computed from the snapshot only, with the deciding set restricted by the contract (decision clock + ≤ 2 slow closed-bar variables); other declared clocks are computed and logged as capability, never fed to the deciding model in campaign 1; (d) no regime labels, no coherence state, no voting, no "HTF wins"; (e) the same function in research, replay and live. **MULTI-TIMEFRAME CAPABILITY = REQUIRED ENGINEERING PROPERTY. HIERARCHICAL MTC AS A SCIENTIFIC MODEL = NOT REQUIRED FOR CAMPAIGN 1.**
**TECHNICAL DESIGN:** a pure function `snapshot → feature vector`; clocks declared in the contract; feature-set membership enforced by a hashed allow-list; the snapshot logged per decision.
**SCIENTIFIC DESIGN:** none beyond the frozen feature set; a richer hierarchy may return only under a new preregistered campaign.
**DATA USED:** engineering-root or pre-activation recorder data.
**DATA NOT USED:** burned blocks; no outcome-bearing evaluation.
**TESTS / VERIFICATION:** two-sided synthetic leak controls (a planted future leak is detected; a clean series passes); the `nn/mtf.py` shift control as template; replay determinism; a test that a non-allow-listed clock cannot enter the deciding vector.
**ACCEPTANCE:** feature values from live snapshots equal features recomputed offline from the recorder root, byte for byte, over ≥ 7 days on every declared clock.
**KILL / STOP CONDITIONS:** none scientific.
**EVIDENCE CREATED:** ENGINEERING.
**EVIDENCE NOT CREATED:** any scientific evidence.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R3, R4 (acceptance).
**NEXT MAY ASSUME:** a causal, multi-clock snapshot whose deciding subset is fixed by hash.
**PR BOUNDARIES:** snapshot-type PR; closed-bar derivation PR; feature-function PR (hash-tested).

### R8 — Multi-symbol single-leg futures runtime and Aegis set scope

**PURPOSE:** replace the carry-shaped runner with the runtime the primary lane needs.
**WHY:** a cross-sectional endpoint is executed as a panel of single-leg positions; no such runtime exists [RF]; equity lives only in the demoted carry ledger; Aegis approves per order, not per target set.
**INPUT PREREQUISITES:** R1, R7 (acceptance). ENG-START-EARLY: the runtime skeleton may be built on the R1 runner during R3–R7.
**EXACT WORK:**
- **R8-a `TargetPositionSet`:** per campaign symbol: side, quantity, confidence, abstain; produced by the frozen candidate; the carry lane keeps `HedgeTarget` in its own process.
- **R8-b Execution:** one `chimera/futures` state machine per symbol (MARKET + reduce-only semantics, idempotent events, finite open-order set).
- **R8-c Single authoritative account ledger:** capital, free cash, per-position margin, fees, funding paid/received, realised, unrealised at mark; derived from the FuturesStore; the CarryLedger retired from the primary path and kept for Lane B.
- **R8-d Aegis set scope:** an `evaluate_target(TargetPositionSet)` entry point evaluating the set before any leg is sent (portfolio gross/net/per-symbol caps, order rate, staleness, dispute state, kill switch, funding cap, daily loss/drawdown from persisted equity); `evaluate_entry` kept for every exposure-increasing order; a set refused in part is refused whole; reductions ungated; scientific campaign limits (max trades/day, allowed symbols) enforced by the policy layer from the contract, not by Aegis; `record_trade_result` wired or its rules removed (finishing R1-k for the new path).
- **R8-e Crash consistency:** single write order store → ledger → log; the log arbitrates disagreement; recovery fail-closed and idempotent.
- **R8-f Per-symbol dispute isolation:** one symbol's dispute halts that symbol; portfolio halt only on limits.
- **R8-g Reconciliation:** dry-run labelled self-consistency; live semantics reserved for R14.
- **R8-h Continuous position guards inside Aegis (AEG-6):** liquidation distance and maintenance margin evaluated inside Aegis every tick from executor-computed `MarginState`; the maintenance rate read from `SymbolConstraints` only and the carry-layer constants (three copies of 0.004) deleted; two-sided test (a touch is halted; a non-touch is not) and a single-source test for the maintenance rate.
**TECHNICAL DESIGN:** reuse of `chimera/futures` domain/executor/store; one Aegis engine per runtime process; no message broker, database server or orchestration platform.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** engineering-root or pre-activation recorder data; synthetic fixtures.
**DATA NOT USED:** burned blocks; no outcome-bearing evaluation.
**TESTS / VERIFICATION:** the state-machine crash harness; two-sided limit tests for every Aegis rule (a breach refused; a non-breach allowed); replay parity across restarts; multi-symbol invariants (no cross-symbol exposure leakage; portfolio caps bind; a partial set refusal sends nothing).
**ACCEPTANCE:** the 16 dry-run invariants re-run under the new runtime plus the multi-symbol invariants; the 48 h synthetic fixture at PARITY with a restart, an operator action and a partial-set refusal; the R3 coherence control passes on that fixture.
**KILL / STOP CONDITIONS:** none scientific.
**EVIDENCE CREATED:** ENGINEERING.
**EVIDENCE NOT CREATED:** any scientific evidence.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R1, R7.
**NEXT MAY ASSUME:** a multi-symbol single-leg dry-run runtime with one ledger, set-scoped Aegis, and clearable disputes.
**PR BOUNDARIES:** R8-a … R8-h as reviewed PRs; the runtime never links to a live venue module.

### R9 — Autonomous demo / soak (engineering evidence only)

**PURPOSE:** prove unattended operation of the exact runtime head that will run the campaign, before the irreversible boundary.
**WHY:** the boundary cannot be undone; every defect that a soak can reveal must be found first. **SOAK PRECEDES PROSPECTIVE ACTIVATION.**
**INPUT PREREQUISITES:** R8 accepted; a gen4 recorder running on the production recorder's pre-activation root (R4) or on the R2 preflight recorder if R2 records that it keeps running (never a prospective root — none exists before R6); the R3 contract merged; the `seq` and `OPERATOR` replay policies fixed.
**EXACT WORK and ACCEPTANCE — the ENGINEERING DEMO PASS gates (§26 as corrected):** (1) ≥ 14 consecutive UTC days unattended on live recorder data under a SOAK profile, the frozen candidate running in SHADOW (targets computed, positions simulated); Lane B, if elected, soaks in its own process under its own PASS/KILL record, its outcome never enters the primary lane's R9 verdict, and a Lane B soak failure defers Lane B to its own later contract and boundary; (2) ≥ 2 planned restarts and ≥ 1 unplanned `SIGKILL` mid-minute, each recovered with a `RECOVERY` record and no state-file edit; (3) ≥ 1 injected feed outage ≥ 10 min → staleness halt as designed and automatic continuation; ≥ 1 recorder-process restart; (4) kill-switch drill persisted across a restart and cleared by `resume` with a note; ≥ 1 reconciliation-dispute drill cleared through the CLI; ≥ 1 disk-low drill, accepted only if the halt reason is present after restart; (5) daily report every day, containing engineering metrics only (no equity, PnL or position-level economics); every alert rule fired at least once (synthetic) and delivered off-host; (6) replay of every soak day = PARITY; (7) zero manual state edits, zero unexplained divergence, zero halts without a record, zero minutes skipped as stale while the recorder was healthy. **Shadow-economics seal:** the candidate's simulated PnL is logged (Aegis needs it) but is not in the soak report, not an input to the R6 decision and not readable by anyone before R10 closes: the ledger's economic fields are written to a sub-root that is hashed daily and read only by the runtime and Aegis, and the sub-root is hashed at soak end as SEALED-DIAGNOSTIC. **Head record:** the exact head, contract hash and feature-function hash of the soak are recorded; R10 runs that head (37.2).
**TECHNICAL DESIGN:** the R1 daemon, R7 snapshot, R8 runtime, unchanged; drills scripted and logged as operator actions.
**SCIENTIFIC DESIGN:** none; scoring of the candidate is forbidden during the soak.
**DATA USED:** engineering-root or pre-activation recorder data.
**DATA NOT USED:** prospective data (none exists); burned blocks.
**TESTS / VERIFICATION:** the gates above; replay parity per day; the drill log.
**ACCEPTANCE:** all seven gates met; the soak report frozen with SHA256SUMS as ENGINEERING evidence; the shadow-economics seal recorded.
**KILL / STOP CONDITIONS:** an unexplained divergence not root-caused within one repair cycle; a silent position change; a halt without a record; a recovery that required editing state; > 1% of healthy-recorder minutes skipped. A failed soak is repaired and re-run in full on the new head; soak days are never spliced.
**EVIDENCE CREATED:** ENGINEERING (the soak report); SEALED-DIAGNOSTIC (shadow economics).
**EVIDENCE NOT CREATED:** alpha; any prospective standing; any input to activation other than PASS/FAIL.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R8; a running gen4 recorder.
**NEXT MAY ASSUME:** a runtime head certified for unattended operation, restart, drills and parity — the only head R10 may run; the R6 review may now proceed.
**PR BOUNDARIES:** the soak-report PR (evidence directory + SHA256SUMS); ENGINEERING-FIX PRs found by the soak are merged before the report is frozen and trigger a full re-run.

### R6 — Prospective boundary activation (irreversible; executes AFTER R9 PASS)

**PURPOSE:** start irreversible evidence accrual under a frozen contract on a soaked runtime.
**WHY:** the boundary cannot be undone; it must be the last gate before accrual, after everything repairable has been repaired and proven. Roadmap v3 activated it fourth; this roadmap activates it last.
**INPUT PREREQUISITES (all, separately recorded):** (1) the R3 contract merged ≥ 7 days earlier with its review; (2) the R4 contract hash; (3) R5 reconciliation running on the production recorder; (4) **R9 PASS** with its frozen soak report and head record; (5) the qualification record produced by R5: ≥ 30 qualifying pre-activation days on the production deployment under the pre-activation hash, the clock starting at the R4 deployment, each day counted only once R5's accepted reconciliation has been run over it, with coverage ≥ thresholds on every required stream and reconciliation agreement on every ARCHIVE_RECONCILED stream (any failure restarts the 30-day clock; no substitution); (6) production recorder source identity verified (part of that record); (7) an independent boundary review recorded in its own governance PR **after** R9 PASS, having read the soak report, the contract, the power report and the qualification record; (8) the owner's explicit activation decision, which may not reference the sealed shadow economics.
**EXACT WORK:** the activation commit sets `prospective_from` to the earliest UTC midnight after all eight inputs; the activated hash writes to a fresh storage root; no mixed hashes; the campaign log opens with the contract hash, the runtime head, and the feature-function, model, policy, cost-model, evaluator and validity-evaluator hashes; Lane B opens on the same boundary in its own process iff it is elected, its preregistration PR was merged ≥ 7 days earlier, and its own R9 record met every gate — readiness is mechanical and may not reference its sealed shadow economics; otherwise it waits for its own later contract and boundary.
**TECHNICAL DESIGN:** the gen3 activation pattern (fresh root; hash stamped in every manifest).
**SCIENTIFIC DESIGN:** none; this phase creates the boundary and no result.
**DATA USED:** none for any decision.
**DATA NOT USED:** the sealed shadow economics; any candidate outcome; burned blocks.
**TESTS / VERIFICATION:** activation-schema tests; a check that the first prospective day is recorded under the activated hash and root; the verifier refuses activation if any of the eight inputs is missing from the PR.
**ACCEPTANCE:** activation PR merged; first prospective day recorded under the activated hash.
**KILL / STOP CONDITIONS:** a coverage failure in the qualifying period restarts the clock; a failed or unrepaired soak → no activation; an owner decision not to activate after a PASSED soak → the candidate is DECLINED with the reason recorded: its engineering soak report stays ENGINEERING evidence, its sealed shadow economics are never opened, the decline consumes the lane's candidate slot (R12), and re-election requires a new R3, a fresh soak and a fresh seal.
**EVIDENCE CREATED:** the boundary (no result).
**EVIDENCE NOT CREATED:** any result; any claim.
**PROSPECTIVE STANDING:** CHANGES — everything after the boundary is prospective for the contract's campaign(s).
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R3, R4, R5, R9 PASS, the qualification, the review.
**NEXT MAY ASSUME:** a live boundary; a head-frozen runtime; a frozen contract; a fresh root; no pre-boundary outcome ever read.
**PR BOUNDARIES:** the boundary-review record PR (governance), then one activation PR containing nothing else.

### R10 — First prospective campaign (design per §27 as corrected)

**PURPOSE:** the first answerable market question, accrued without adaptation.
**WHY:** every historical result was adaptive or underpowered; this is the first evidence that can count.
**INPUT PREREQUISITES:** R6; the R9 head; the contract.
**EXACT WORK:** accrual under the contract on the head-frozen runtime; no online tuning; no candidate replacement; no universe change; no refit outside a predeclared calendar lane. **No-peek:** engineering dashboards only (coverage, staleness, halts, parity, block validity). **At every block end, in this order:** (1) the validity evaluator is run by the CI/verifier job (never by hand) on engineering records only; the job commits the VALID/INVALID verdict, its input manifest and its hash in the block's evidence PR, opened before the readout job runs and reviewed by a person who certifies having read no readout of that block (a manually produced verdict is itself an invalidity criterion); (2) for a VALID block only, the frozen evaluator computes the block readout, logged and hashed; INVALID blocks get no readout; (3) interim decisions only through the alpha-spending rule. **Extension:** accrual continues in contiguous calendar blocks until N VALID blocks exist or the maximum extension E is exhausted; the count is mechanical. **Head:** ENGINEERING-FIX PRs only (37.2). **Lane B**, if elected, accrues in parallel in its own process under its own contract on the same boundary and is reported separately.
**TECHNICAL DESIGN:** runtime unchanged from R9; the campaign log; the R3 evaluator and validity code; per-block evidence directories.
**SCIENTIFIC DESIGN:** per the contract: information-primary cross-sectional endpoint; economic falsification screen; negative controls; Holm over m ≤ 2; deciding economics identical to R11's.
**DATA USED:** prospective data from the activated root, read only by the frozen evaluator at block ends.
**DATA NOT USED:** any pre-boundary data for a deciding purpose; burned blocks; the outcomes of INVALID blocks (never computed).
**TESTS / VERIFICATION:** daily replay parity; the validity evaluator's determinism (re-run offline gives the same verdict); the per-block hash chain; the verifier's check that no readout precedes its validity verdict; the coherence control re-run per block.
**ACCEPTANCE (campaign end):** N VALID blocks → the preregistered PASS / NEGATIVE statements applied by the frozen evaluator (information endpoint and economic screen); or E exhausted → **NOT EVALUABLE / ENGINEERING FAILURE**.
**KILL / STOP CONDITIONS:** *engineering, mechanical:* a block INVALID → excluded from scoring, campaign extends per rule; a head change outside ENGINEERING-FIX that touches no contract-hashed module → that block INVALID and a full re-soak before the next block is scored; a change to a contract-hashed module → campaign ABORTED (the object changed; any successor is route C); an owner stop → campaign ABORTED (37.1): the reason recorded, the candidate's budget slot consumed, no same-design rerun, all block readouts left sealed, no partial claim. ABORTED is never recorded as NOT EVALUABLE, which is reserved for exhaustion of E. *Scientific:* none mid-campaign except the alpha-spending rule.
**EVIDENCE CREATED:** PROSPECTIVE (PASS or NEGATIVE), a NOT EVALUABLE engineering-failure record, or an ABORTED record; the full VALID/INVALID chronology in every case.
**EVIDENCE NOT CREATED:** an alpha claim; real-money eligibility; confirmatory standing.
**PROSPECTIVE STANDING:** consumes the boundary for this candidate at the first scored day.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R6.
**NEXT MAY ASSUME:** an immutable campaign outcome with a complete block chronology. PASS on both floors → R11 with the identical object. Information PASS + screen FALSIFIED → information-only result, no promotion, R12. NEGATIVE → R12 (family retired). NOT EVALUABLE → R12 (engineering cause fixed, re-soak, the pre-committed automatic same-design successor on a new boundary if the slot allows; no pooling; readouts sealed until it closes). ABORTED → R12 (slot consumed).
**PR BOUNDARIES:** one evidence PR per block (validity verdict, then readout, each with SHA256SUMS); the closure PR; ENGINEERING-FIX PRs.

### R11 — Economic confirmation and continuation (same object as R10)

**PURPOSE:** confirm, not discover: replicate the R10 result on new data under an unchanged success definition.
**WHY:** one prospective PASS is one observation; confirmation adds data and must not redefine success.
**INPUT PREREQUISITES:** R10 PASS on both floors; an independent audit reconstructing the R10 verdict from the decision log and recorder files; the continuation size N′ and the non-inferiority rule from the R3 contract (N′ fixed there, or produced by a sizing rule preregistered there as deterministic in the measured R10 effect).
**EXACT WORK:** the same frozen candidate, on the same head-frozen runtime (ENGINEERING-FIX only), under the **identical deciding economic semantics** (fee, spread, executable price, latency, slippage envelope, funding, turnover, reject/partial rule, impact rule), for ≥ N′ VALID blocks (proposal: N′ ≥ ⌈N/2⌉ and ≥ 3) with the same validity/extension mechanics. PASS = the continuation does not contradict the R10 result under the preregistered non-inferiority rule. New execution, testnet or live-operational information enters only: **(A)** through an update rule preregistered in R3 as deterministic, applied exactly and logged; **(B)** as non-deciding stress/diagnostic analyses reported separately and labelled (measured slippage distributions, testnet reject/partial rates, hypothetical impact at larger sizes, mid-referenced slippage in bps, the opened SEALED-DIAGNOSTIC shadow-vs-live comparison from R9); or **(C)** by a new preregistration and a new campaign, which R11 is not. The confirmatory period may add data; it may not redefine success.
**TECHNICAL DESIGN:** runtime and evaluator unchanged; a separate non-deciding stress module that cannot write to the deciding readout.
**SCIENTIFIC DESIGN:** replication of one preregistered object; no new hypothesis; m unchanged.
**DATA USED:** prospective data after R10's close.
**DATA NOT USED:** R10's data for re-scoring; pre-boundary data; burned blocks.
**TESTS / VERIFICATION:** as R10; a test that the evaluator, cost-model and validity-evaluator hashes equal R10's.
**ACCEPTANCE:** CONFIRMATORY PASS by the frozen evaluator; independent audit reconstructing it.
**KILL / STOP CONDITIONS:** contradiction → CONFIRMATION FAILED, immutable, eligibility withdrawn, R12; NOT EVALUABLE as in R10; any change to a contract-hashed module → the continuation ends as ABORTED (route C for any successor), without a confirmatory result.
**EVIDENCE CREATED:** CONFIRMATORY (or an immutable confirmation-failure record); DIAGNOSTIC stress reports.
**EVIDENCE NOT CREATED:** real-money authority; any claim beyond eligibility for consideration.
**PROSPECTIVE STANDING:** **Gate 1** — eligibility for real-money *consideration*, nothing more.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R10 PASS.
**NEXT MAY ASSUME:** a candidate with CONFIRMATORY standing; a realised-cost diagnostic record from which Gate 4 numbers may be derived; any mismatch between the frozen deciding model and observed execution is a (C) trigger, never a silent revision.
**PR BOUNDARIES:** continuation block PRs; closure PR; the stress-analysis report PR, labelled non-deciding.

### R12 — Strategy expansion governance

**PURPOSE:** decide what follows any campaign outcome without adaptive drift.
**WHY:** an adaptive sequence of campaigns can masquerade as one programme.
**INPUT PREREQUISITES:** any R10 or R11 outcome record, or a DECLINED record from R6.
**EXACT WORK:** apply the rules: (1) NEGATIVE retires the candidate family for the same (target, horizon, universe), recorded with its full disclosure block; (2) NOT EVALUABLE retires nothing scientifically — the engineering cause is fixed by reviewed PRs, the runtime re-soaked (R9), and the pre-committed successor rule of R3 applies automatically: at most one same-design successor (new hash, a new boundary with all eight R6 inputs again, no pooling), never decided on outcomes; the predecessor's VALID-block readouts stay SEALED-DIAGNOSTIC until the successor closes or the candidate is retired, and are then disclosed as ADAPTIVE context, never as a design input; a second NOT EVALUABLE is terminal for the design and consumes the slot; (2b) ABORTED and DECLINED consume the candidate's slot and permit no same-design rerun; (3) a new candidate changes ≥ 1 design axis a priori and re-enters at R3 (power report, review, ≥ 7-day gap), then must pass R7 (feature-function hash), R9 (re-soak on the new head) and the full R6 input list before its boundary; it may read a negative campaign's descriptive statistics (turnover, cost realisation, coverage) but not its per-decision outcomes; (4) budget: ≤ 2 directional candidates and ≤ 1 carry candidate before a mandatory stop-and-decide, each candidate holding one R10, at most one automatic NOT-EVALUABLE re-run and one R11; after two directional negatives, record "no deployable directional alpha under the current mandate"; (5) engineering evidence, DIAGNOSTIC statistics and cost tables are reusable; PROSPECTIVE and CONFIRMATORY evidence belongs to the candidate that earned it; (6) orchestration (R13) opens only with ≥ 2 families holding CONFIRMATORY standing.
**TECHNICAL DESIGN:** governance documents; the front-door verifier counts candidates against the budget.
**SCIENTIFIC DESIGN:** governance only.
**DATA USED:** campaign outcome records (descriptive).
**DATA NOT USED:** per-decision outcomes for design; burned blocks.
**TESTS / VERIFICATION:** verifier; CI.
**ACCEPTANCE:** a decision PR naming the outcome record's SHA and the next permitted action.
**KILL / STOP CONDITIONS:** budget exhausted → stop-and-decide review; no further campaign without a new mandate and budget.
**EVIDENCE CREATED:** none.
**EVIDENCE NOT CREATED:** any.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R10 / R11.
**NEXT MAY ASSUME:** a recorded next step inside the budget.
**PR BOUNDARIES:** one governance PR per decision.

### R13 — Portfolio / orchestration (conditional)

**PURPOSE:** allocate across lanes only once two families hold CONFIRMATORY standing.
**WHY:** premature allocation logic is a design degree of freedom and an operational burden; nothing today needs it.
**INPUT PREREQUISITES:** R12's decision; ≥ 2 CONFIRMATORY families (the second arrives from Lane B or through the R12 → R3 re-entry loop).
**EXACT WORK:** capital allocation across lanes; netting of opposing intents; gross/net caps; per-symbol concentration; drawdown budgets per lane and in total; correlation-aware exposure with a historical-simulation floor only (no VaR model beyond it); one Aegis engine over the lanes that share an account (as §7 and §29 state), one account ledger, one evidence trail; separate dry-run lanes on separate accounts keep separate engines.
**TECHNICAL DESIGN:** an extension of R8's set scope; no message broker, database server or orchestration platform.
**SCIENTIFIC DESIGN:** the allocation rules are a governance contract, not a hypothesis; they are hashed before use.
**DATA USED:** CONFIRMATORY campaign records (descriptive).
**DATA NOT USED:** any pre-boundary outcome; burned blocks.
**TESTS / VERIFICATION:** two-sided cap tests; replay parity under orchestration.
**ACCEPTANCE:** a ≥ 14-day dry-run soak under orchestration with PARITY every day.
**KILL / STOP CONDITIONS:** fewer than two CONFIRMATORY families → not opened.
**EVIDENCE CREATED:** ENGINEERING.
**EVIDENCE NOT CREATED:** alpha.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R12.
**NEXT MAY ASSUME:** a validated allocation layer, if and only if opened.
**PR BOUNDARIES:** allocation-layer PRs; the orchestration soak-report PR.

### R14 — Live-readiness qualification (Gates 2 and 3; Gate-4 contract text)

**PURPOSE:** make the execution and operational conditions of the real-money boundary true and provable.
**WHY:** nothing in the repository has ever been reconciled against a real exchange position; the only live-capable path is retired code [RF].
**INPUT PREREQUISITES:** R11 Gate 1 for the candidate that will trade (the adapter is built in this phase, after Gate 1; testnet credentials and, for the read-only reconciliation option, a read-only key with no trading permission are the only credentials permitted before R15; no trade-capable key exists before R15's token); R9 PASS on the campaign head.
**EXACT WORK:** *Gate 2 — execution:* an authenticated exchange adapter as a separate package with its own AST guard set; signed REST + user-data stream; idempotent client order ids; unknown-state resolution by order query before any retry; partial fills across restarts; cancels; reduce-only semantics; rate-limit backoff; `listenKey` keepalive and expiry; maintenance-window handling; websocket-loss and REST-ambiguity drills; reconciliation every minute with HALT on disagreement; ≥ 30 days on testnet or read-only live reconciliation; zero unresolved disputes; drill log frozen. *Gate 3 — operational:* dedicated sub-account; the trade-only key specification (withdrawals disabled, IP allow-list, daily permission re-verification) drilled on testnet — the live trade-only key is created in R15 and these items are re-verified within 7 days of go-live; secrets outside the repository and working directory; separate Unix users and state roots for recorder, runner, adapter; SHA-pinned CI, hash-pinned dependencies, digest-pinned images; backups restored in a drill; off-host dead-man switch; chrony; documented host baseline and firewall; audit log of every operator action; an out-of-process exchange-side "cancel all and close" procedure drilled on testnet. *Gate 4 text:* the capital-governance contract drafted with numbers derived from R9–R11 measurements (starting capital the owner can lose entirely; gross exposure ≤ 1× capital; leverage 1× in config and on the exchange account; max order notional; daily/cumulative loss and drawdown halts with flatten and key rotation; per-symbol cap; funding-cost cap; overnight policy; shutdown triggers; two-person enable; rollback path). *Deployable-head re-soak:* the R9 gates re-run in full (dry-run, ≥ 14 days, drills, PARITY every day) on the exact deployable head — the runtime plus the adapter package in testnet mode — frozen as ENGINEERING evidence; this report is the artifact real-money condition (2) names. *Diagnostic:* a non-deciding comparison of testnet fills against the frozen deciding cost model, which may trigger a (C) new preregistration but never alters R11.
**TECHNICAL DESIGN:** the adapter in its own package and Unix user; the runtime unchanged; dry-run and live share the same order semantics.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** testnet; read-only live account data.
**DATA NOT USED:** campaign outcomes for design; burned blocks.
**TESTS / VERIFICATION:** the Gate 2 drill list, each drill logged; the Gate 3 checklist; the crash harness against the adapter's state machine.
**ACCEPTANCE:** the deployable-head re-soak PASS; Gate 2 drill log frozen as ENGINEERING evidence; Gate 3 checklist PASS, re-verified within 7 days of any go-live; Gate 4 text ready to sign.
**KILL / STOP CONDITIONS:** any unresolved dispute at the end of the drill period → the 30-day period repeats; any Gate 3 item failing at re-verification → no go-live.
**EVIDENCE CREATED:** ENGINEERING / OPERATIONAL.
**EVIDENCE NOT CREATED:** scientific evidence; authority.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R11 (Gate 1); R9 (its gates re-run here on the deployable head).
**NEXT MAY ASSUME:** Gates 2 and 3 PASS on the exact head to be deployed; a Gate-4 contract awaiting signature.
**PR BOUNDARIES:** adapter-package PRs; the drill-log evidence PR; the operations-checklist PR; the Gate-4 contract PR (draft until signed).

### ================= REAL-MONEY AUTHORITY BOUNDARY =================

No phase above may place a real order. Crossing requires **ALL FOUR**, separately recorded, each mapped to a §30 gate:
1. **SCIENTIFIC ELIGIBILITY = Gate 1:** R11 CONFIRMATORY PASS under the same object and the same deciding economics as R10, with an independent audit reconstructing the verdict.
2. **ENGINEERING ELIGIBILITY = R9 PASS + R14 re-soak + Gate 2:** R9 PASS on the campaign head, the R14 re-soak PASS on the exact deployable head, and the execution drills PASS on that head.
3. **OPERATIONAL ELIGIBILITY = Gate 3:** the operational checklist PASS, re-verified within 7 days of go-live.
4. **OWNER AUTHORISATION = Gate 4:** the signed live-authorisation contract naming capital cap, leverage 1×, symbols, order types, abort rules, the two-person enable procedure and the rollback path.

A profitable backtest, demo, soak or canary never satisfies (1). Engineering or demo success is never alpha. No historical positive result is prospective. Nothing in this roadmap, in the audit or in the corrigendum creates real-money authority.

### R15 — Owner authorisation (Gate 4 executed)

**PURPOSE:** record the owner's bounded, revocable authority to run the canary.
**WHY:** authority must be explicit, written, and one of four gates, never the only one.
**INPUT PREREQUISITES:** conditions (1)–(3) recorded as PASS; the Gate-4 contract text from R14.
**EXACT WORK:** a governance PR recording the signed contract; the acknowledgement token is created only after the PR merges; the trade-capable API key is created only after the token exists; both are logged.
**TECHNICAL DESIGN:** documents; the runtime's live gate reads the token and the contract hash.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** none.
**DATA NOT USED:** any.
**TESTS / VERIFICATION:** the verifier checks that the four condition records exist and are current before the live gate can open.
**ACCEPTANCE:** PR merged; token and key existence logged.
**KILL / STOP CONDITIONS:** any of conditions (1)–(3) lapsed or contradicted → no authorisation; authority is revocable at any time by removing the token.
**EVIDENCE CREATED:** none.
**EVIDENCE NOT CREATED:** any.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** authority created for the canary only, bounded by the signed contract.
**DEPENDENCIES:** R11, R14.
**NEXT MAY ASSUME:** a bounded authority for R16 and nothing beyond it.
**PR BOUNDARIES:** one governance PR.

### R16 — Canary (Gate 5)

**PURPOSE:** prove that real execution matches the dry-run model at a size that cannot matter.
**WHY:** the first real order is irreversible; it must be small, parallel-shadowed and reconciled daily.
**INPUT PREREQUISITES:** R15.
**EXACT WORK:** ≥ 30 days; ≤ 10% of the authorised cap and above minNotional × the largest campaign order; 1×; the campaign's symbols only; MARKET + reduce-only semantics identical to dry-run; a parallel dry-run for parity; daily reconciliation of fills, fees and funding against the account statement; abort on any reconciliation dispute, unexplained divergence, daily-loss breach, key-permission change or parity failure; escalation = stop and audit, never resize.
**TECHNICAL DESIGN:** the R14 adapter; the R8 runtime; the parallel dry-run process.
**SCIENTIFIC DESIGN:** none; a profitable canary is operational evidence only.
**DATA USED:** live account data; the campaign's recorder data.
**DATA NOT USED:** any outcome for candidate design.
**TESTS / VERIFICATION:** daily reconciliation report; daily parity report.
**ACCEPTANCE:** 30 days with zero aborts, zero unreconciled items, parity every day; the canary report frozen as OPERATIONAL evidence.
**KILL / STOP CONDITIONS:** as listed; any abort ends the canary; a repeat needs a new R15 authorisation.
**EVIDENCE CREATED:** OPERATIONAL.
**EVIDENCE NOT CREATED:** scientific evidence; scale authority.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** real orders at canary size under the contract.
**DEPENDENCIES:** R15.
**NEXT MAY ASSUME:** execution fidelity at canary size, nothing about larger sizes.
**PR BOUNDARIES:** the canary evidence PR.

### R17 — Controlled real-money operation

**PURPOSE:** operate at the authorised cap under the same rules.
**WHY:** the cap, not the result, is the authority.
**INPUT PREREQUISITES:** R16 accepted.
**EXACT WORK:** continue at the authorised cap; monthly frozen reports; monthly parity; daily reconciliation; any change to the candidate is a new R3 followed by a dry-run campaign — money never runs an unconfirmed candidate.
**TECHNICAL DESIGN:** unchanged.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** live account data.
**DATA NOT USED:** live outcomes for candidate design.
**TESTS / VERIFICATION:** monthly reports and parity.
**ACCEPTANCE:** ongoing; each monthly report frozen.
**KILL / STOP CONDITIONS:** the contract's halt triggers (daily/cumulative loss, drawdown, dispute, parity breach, key change); halt + flatten + key rotation.
**EVIDENCE CREATED:** OPERATIONAL.
**EVIDENCE NOT CREATED:** scientific evidence; scale authority.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** bounded by the contract.
**DEPENDENCIES:** R16.
**NEXT MAY ASSUME:** a realised-cost and parity record for R18.
**PR BOUNDARIES:** monthly evidence PRs.

### R18 — Scale-up / scale-down governance (Gate 6)

**PURPOSE:** change size only by decision, never by momentum.
**WHY:** leverage and size creep after a profitable period is the classic failure.
**INPUT PREREQUISITES:** R17 with ≥ 60 days of frozen reports since the last step.
**EXACT WORK:** increase only by a new decision with its own evidence period and audit; ≤ 2× per step; ≥ 60 days between steps; only after realised costs and parity match the frozen deciding model (a mismatch is a (C) trigger: new preregistration before any scale-up); automatic decrease or halt on drawdown, parity breach or reconciliation dispute; no profit-based acceleration; leverage stays 1× throughout this roadmap.
**TECHNICAL DESIGN:** unchanged.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** frozen monthly reports.
**DATA NOT USED:** live outcomes for candidate design.
**TESTS / VERIFICATION:** the audit per step.
**ACCEPTANCE:** each step recorded in a governance PR with its evidence period.
**KILL / STOP CONDITIONS:** automatic decrease/halt as listed.
**EVIDENCE CREATED:** OPERATIONAL.
**EVIDENCE NOT CREATED:** scientific evidence.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** bounded by the current step's contract amendment.
**DEPENDENCIES:** R17.
**NEXT MAY ASSUME:** nothing beyond the current step.
**PR BOUNDARIES:** one governance PR per step.

---

---

## 38. Delta: Old Roadmap (v3) vs New Roadmap (§37)

| topic | Roadmap v3 (adopted 2026-09-12) | this proposal | why |
|---|---|---|---|
| Sequence | V3-1b → V3-2 preflight → V3-3 power → V3-4 gen4 freeze → V3-5 reconciliation → V3-6 boundary → V3-7 MTC → V3-8 single-leg runtime → V3-9 demo → V3-10 campaign | (R1 ∥ R2, both from R0) → R2 feeds R3 design/power → R4 freeze → (R5 reconciliation ∥ [R1 + R7 multi-clock snapshot] → R8 runtime → R9 soak ∥ the ≥ 30-day recorder qualification, whose days count once R5 reconciles them) → **R6 boundary only after R9 PASS and the boundary review** → R10 campaign → R11 confirmation of the identical object (edges per §37 Part B) | v3 activates the boundary (V3-6) before the runtime that will execute the campaign exists or has soaked; the boundary is irreversible and should be the last gate before accrual, not the fourth |
| Runtime defects | not scheduled (V3-1b covered only the clock) | R1: identity, lifecycle, staleness, cadence, disputes, parquet atomicity, resume ordering, parity policy, dead limits, valuation price | the demo cannot start or run unattended today |
| First campaign | information + economic co-primary, ~182 days, 12–20 symbols | two-stage in one accrual: information primary (cross-sectional panel IC), economic falsification screen gating a separately sized confirmation (R11); duration from the frozen power report (≥ 6 months, likely 9–12); K from measured effective N; N VALID blocks with a mechanical, outcome-blind validity rule, a frozen extension cap and a NOT EVALUABLE outcome; deciding cost semantics frozen in R3 and identical in R11 | arithmetic: the co-primary economic floor is unreachable in 182 days at taker costs |
| Strategy family | "directional LONG/SHORT" | one hypothesis: price-based panel state → vol-standardised 4–8 h executable returns, low-turnover LONG/SHORT policy; plus a SECONDARY, OPTIONAL carry lane that never delays or outranks the primary lane | a shape is not a hypothesis; carry is the only mechanism with reachable economic evidence in the first window |
| Endpoint/runtime coherence | not addressed | coherence rule; R8 runtime is multi-symbol single-leg | a panel endpoint is cross-sectional; a single-symbol runtime cannot execute it |
| Target | open (§22) | vol-standardised executable forward return; secondary binary; analytic threshold; hysteresis frozen | classification with a grid threshold is the largest hidden degree of freedom |
| Horizon | open | κ-rule from preflight, likely 4–8 h; funding-aligned | first-principles cost envelope |
| MTC | "mandatory", "locked" | causal as-of semantics and multi-timeframe *capability* mandatory in the runtime (generic multi-clock snapshot); hierarchical MTC as a scientific model deferred; ≤ 2 slow variables in the deciding set | narrow negatives; premature commitment; the runtime must not be limited to one timeframe |
| Universe | 12–20 + 2–3 | procedure + measured effective N; provisional 10–12 + 2 | a number before a measurement is a guess |
| gen4 streams | tiers listed | tiers with day-one/add-later/never per stream, raw-immutable vs normalised-regenerable, verification classes, storage estimates, cross-stream gates | contract must bind code, schema and contract versions |
| PR #76 | KEEP + REWORK + INTEGRATE EARLY | same, with both-outcome branches and decoupling of gen4 from its acceptance | the sequence must not wait on a VPS campaign |
| Aegis | remediated wiring; clock injection | scope defined (what is and is not Aegis); dead rules removed or wired; portfolio caps; single ledger | several rules are unreachable; equity lives in the demoted ledger |
| Demo | "engineering milestone" with a drill list | a daemon on live recorder data with quantified PASS/KILL gates, run BEFORE the boundary on the head that will run the campaign, shadow economics sealed | the current runner is a bounded pass; the boundary is irreversible |
| Evaluation | "power before boundary" in prose | frozen executable three-layer power/inference module; multiplicity budget m ≤ 2; disclosure block | nothing implements V3-3 |
| Governance | firewalls + evidence ladder | firewalls kept; evidence classes machine-labelled; no-peek and interim-look rules; ≥ 7-day gap and review; Git policy | prospective mechanics were missing |
| Expansion | lanes listed; ≥ 2 families before orchestration | budget (2 directional + 1 carry candidates), axis-change rule, stop-and-decide; ≥ 2 CONFIRMATORY families before orchestration | prevents an adaptive sequence masquerading as one programme |
| Live | V3-14 list | Gates 1–6 with parameters or freeze procedures; a visibly separate real-money boundary with four conditions; canary and scale-up rules | v3 stops at "framework" |
| Supply chain / ops | not addressed | SHA/hash/digest pins; off-host dead-man; backups; chrony; separate users | cheapest fixes with the largest downside |
| Freqtrade | RETIRED/DISCONNECTED | DELETE in R1 | the only live-capable path |

---

## 39. Final Go / No-Go Statements

- **READY / NOT READY FOR NEXT ENGINEERING PHASE — READY**, for R1 (runtime integrity remediation) and R2 (gen4 engineering preflight on a separate host). Both are engineering-only, create no evidence and can start now. Not ready for V3-4 through V3-9 as v3 sequences them.
- **READY / NOT READY TO FREEZE GEN4 DATA CONTRACT — NOT READY.** Blockers: no preflight measurements (rates, storage, archive availability per stream, cross-stream consistency, correlation matrix, effective N, cost envelope); no raw-immutable/normalised-regenerable separation or contract/code/schema version binding designed; no decision on which streams are day-one; PR #76 reconciliation not generalised; universe procedure not run.
- **READY / NOT READY TO ACTIVATE A NEW PROSPECTIVE BOUNDARY — NOT READY.** Blockers: no frozen contract (R4); no preregistered campaign with frozen deciding economics and validity rules (R3); no frozen power/inference module (R3); no gen4 reconciliation (R5); no runtime that can execute the campaign's policy (R7, R8); **no R9 PASS — the soak must precede activation**; no independent boundary review; no ≥ 30-day qualifying pre-activation period.
- **READY / NOT READY FOR AUTONOMOUS DEMO — NOT READY.** Blockers: campaign profile cannot start (config; `_software()` identity defect); runner is a bounded pass, not a daemon; `resume`/`resolve` cannot succeed from the CLI; stale-feed veto identically zero; catch-up cadence skips most minutes; crash windows create disputes with no clearing path; recorder writes the open day in place; replay parity has no policy for restarts or operator actions; the runtime is carry-shaped and single-symbol; no single account ledger; funding veto and loss-streak rules unreachable.
- **READY / NOT READY FOR PROSPECTIVE ECONOMIC CLAIMS — NOT READY.** Blockers: no prospective data exists; no campaign is preregistered; the historical instrument cannot support any claim; economic confirmation requires R11 after an information PASS, which requires everything above.
- **READY / NOT READY FOR REAL MONEY — NOT READY.** Blockers: all four eligibility conditions fail — scientific (no confirmatory evidence), engineering (no authenticated adapter, no exchange reconciliation, no soak), operational (no credential architecture, no off-host kill switch, no backups, no pinned supply chain), owner authorisation (no live contract exists). Nothing in this audit, in the demo, in a future soak, or in a future canary changes the scientific condition; only R10 → R11 → independent audit can.

The proposed roadmap is a PROPOSAL. It does not become authoritative by being written; it creates no evidence, no boundary, no claim and no authority.
