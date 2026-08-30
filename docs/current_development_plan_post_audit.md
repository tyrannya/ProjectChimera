# ProjectChimera — post-audit development-plan addendum

Status: **authoritative supplement to `docs/current_development_plan.md` after the independent Fable 5 post-merge audit of `main` at `a72f94e021be61df2851b746d9d3ee741df09d0d`.**

This file records decisions discovered after the current development plan was written. It is a planning/governance document, **not** a research preregistration and not evidence that any future strategy works.

**Consolidated.** Its authoritative conclusions — the four required historical disclosures, the narrowed P6/P7 readings, the futures-versus-Freqtrade live-capability distinction, and the selection of structural carry as the next axis — now live in the normal front-door documents: [`current_development_plan.md`](current_development_plan.md), [`research_roadmap.md`](research_roadmap.md) and [`../README.md`](../README.md). This addendum is retained as the fuller record of *how* each conclusion was reached and of the contract requirements the structural screen had to satisfy; it is no longer the only place any of them is stated, and the front-door documents are not "older" than it.

## START HERE — post-audit state

The independent audit returned **COHERENT WITH REQUIRED REPAIRS**. Seven independent audit paths reconstructed the recent science, data boundaries, accounting, execution and safety machinery rather than trusting PR prose. Headline P6/P6-EXT/P7 values were independently replayed from committed row-level evidence; preregistration-before-result ordering was verified for P6, P6-EXT, P7 and P8; the futures accounting and dry-run safety paths survived independent LONG/SHORT dimensional checks and live-reachability probes.

The audit **does not reopen or change any frozen verdict**. It changes the precision of the historical record and the choice of the next research question.

Current research state remains:

- `P6`: answered, negative under its preregistered XGBoost deciding family;
- `P6-EXT`: answered, negative; `4h` and `1d` not viable under that screen;
- `P7`: answered, negative against the preregistered fold-wise best-constituent benchmark;
- `P8`: preregistered, **NOT OPENED**, because zero trading modes are currently eligible;
- `P4-HOLD`: retired, checkpoint remains `null`, not a spare holdout;
- Styx: sealed from repository use, but its historical/blinding limitation below is now part of the record;
- Futures execution: dry-run only in the Chimera futures chain;
- sustained paper alpha validation: **not started**;
- real-money promotion: **NO-GO**.

## Required historical disclosures

### 1. Styx is an internal non-read seal, not a truly prospective blind holdout

The fixed Styx instant is `2025-08-27T23:00:00+00:00` and the multi-clock research-visible cutoff is `2025-05-19T08:00:00+00:00`, but the repository's formal research programme and sealing machinery were authored in August 2026, after those market dates had already occurred.

Consequences:

- the seal is valuable against **subsequent in-repository adaptivity and accidental reads**;
- it does **not** prove that designers were historically blind to the market action after the chosen boundary when the protocol was authored;
- any future Styx result must be described as a one-shot, repository-sealed historical evaluation with a **hindsight-era ceiling**, not as pristine prospective evidence;
- a genuinely stronger future confirmation source is a strategy frozen before **future wall-clock data that has not yet occurred**, followed by sustained prospective paper observation.

This does not invalidate P1-P7 negative findings. It does cap any future positive claim made from Styx.

### 2. P6 primary-fit source provenance has a reproducibility ceiling

The fifteen primary P6 cells record the preregistration revision `a56df4641d0b3b1f9ea2554373ce23e4f6dfdef2`, but also record `dirty: true`, two untracked Python source files and a source digest representing that dirty tree. The committed per-sample predictions and their economics replay exactly, but the exact fitting source state is not recoverable as one committed Git tree.

Therefore:

- the P6 verdict remains the historical verdict because its committed predictions and gate arithmetic are reproducible;
- do not claim the **fit itself** can be reconstructed exactly from a clean checkout;
- P6-EXT and P7 have stronger source pinning and are not granted P6's provenance defect by association;
- future research cells must refuse or prominently fail a primary-evidence fit from a dirty/uncommitted source tree unless the full source state is itself committed and recoverable.

### 3. "Frozen evidence is never rewritten" is era-dependent, not universally historical

Earlier P2b/P2c-era derived reports/manifests were regenerated and some operational evidence was replaced after repairs; later generations became progressively stricter about immutable primary evidence. Future documentation must distinguish:

- immutable/frozen **primary economic evidence** under the modern discipline;
- derived reports that may be regenerated from immutable inputs;
- historical exceptions already disclosed in Git history;
- code/document repairs made after closure without changing the frozen economic outcome.

Do not rewrite old primary numbers to make the history aesthetically cleaner.

## Scientific interpretation after the audit

### P6 / P6-EXT

P6 establishes only that, under its frozen BTCUSDT multi-clock design and preregistered XGBoost deciding rule, none of the seven tested clocks cleared the viability screen. The result must not be expanded into "short timeframes do not work".

Secondary-family facts remain leads, not promoted evidence:

- Logistic Regression would have cleared the P6 screen on `1m`, `5m`, `15m`;
- LightGBM would have cleared it on `1m`, `5m`;
- a secondary Logistic Regression result also looked stronger on `4h` in P6-EXT;
- the design correctly refused winner-shopping because XGBoost was named deciding before fitting.

A new checkpoint that merely declares one of those already-seen families the winner on the **same four burned blocks** is not fresh confirmation and has low information value. New data or a genuinely new question is required for such a follow-up to become meaningful.

The native momentum baseline on fast clocks was extremely weak (approximately trading itself toward `-1` under costs). "Beats momentum" therefore provided little discrimination there. P6 still failed on its other required conditions, so the verdict does not move.

### P7

P7 remains negative **against the preregistered fold-wise hindsight-best constituent benchmark**:

- SCALPING: `1/4` folds improved, mean delta `-0.0265515`;
- DAY_TRADING: `1/4`, mean delta `-0.034336`.

The audit adds a crucial interpretive note: the consensus beat the **mean constituent** in all four folds of both modes in the independent reconstruction. That weaker benchmark was not the preregistered decision rule and cannot rescue P7, but it means the statement "consensus necessarily destroyed component value" is too broad.

P7-v1 conclusions remain narrow because:

- day trading realised only 13 trades across four folds and one fold traded zero times;
- the frozen alignment rule had no staleness bound and observed slower votes could be almost four days old;
- none of the constituent XGBoost specialists was individually viable;
- the fold-wise best benchmark is deliberately oracle-like and difficult to beat.

Do not retrofit P7. If cross-timeframe fusion is revisited, staleness, minimum effective sample/trade requirements and benchmark choice must be fixed in a new preregistration before new evidence is read.

### P8

P8 is **preregistered only**. There is no AUTO implementation/result and the opening condition is mechanically false: zero modes are eligible. Do not open it to rescue P7.

## Engineering and safety interpretation

The independent audit found the Chimera futures accounting and dry-run executor coherent:

- leverage applied exactly once;
- quantity/notional separated;
- fees use notional bases;
- LONG/SHORT signs are symmetric;
- funding signs/interval semantics are correct under the implemented model;
- partial-fill fee accounting conserves correctly;
- `net = realised - fees + funding` reconciles in checked examples;
- persisted state/reconciliation/emergency flatten remain fail-closed;
- the committed dry-run and zero-order paper-smoke artifacts are engineering evidence, not alpha evidence.

Terminology must remain precise: **the `chimera.futures` path has no live route**, but the repository also contains a legacy Freqtrade spot pathway that is deliberately double-gated for live operation. Do not make a repository-wide claim that no live-capable code exists. Nothing in current evidence authorises enabling either a futures live path or the legacy Freqtrade live path.

Unknown/corrupt account state remains fail-closed for exposure increases, while safe risk reduction must remain available.

## Test hardening required by the audit

The real-evidence P5/P7 tests reconstruct the committed negative decisions well, but negative evidence alone is not a two-sided control for the decision functions.

Required repair:

- synthetic P5 supportive-case test proving the 3-of-4 gate can produce `supportive_adaptive`;
- synthetic P7 supportive case proving the conjunction `improved_folds >= 3` **and** `mean_delta > 0` can produce `supportive_adaptive`;
- synthetic P7 counter-controls proving a positive mean cannot compensate for fewer than three positive folds and that three positive folds cannot compensate for a non-positive mean.

These tests harden decision code only. They do not alter any result artifact.

## Dependency / reproducibility housekeeping

The audited multi-clock path was validated under pandas 2.x and a reviewer reproduced a fail-closed-but-wrong timestamp-resolution behavior under pandas 3.x assumptions. Until the code is explicitly made version-agnostic and tested there, the project should constrain the core dependency to `pandas>=2.1,<3.0` rather than silently accepting an unverified major version.

Also close stale PR #50 because the corresponding P4 Stage-1 authorisation was incorporated through merged PR #51; leaving the superseded PR open misstates repository state.

## Post-audit decision: the next checkpoint

The mandatory decision review has now been performed with an independent external audit. The selected next research axis is **structural/non-directional alpha**, specifically a preregistered BTC spot/perpetual delta-hedged funding/basis carry feasibility screen.

This supersedes the prior working preference to make cost-aware directional abstention the immediate next checkpoint. Cost-aware abstention, target redesign, adaptive retraining/OOD, native L1/L2 microstructure and neural specialists stay in the roadmap; they are **not** deleted. They are simply not the next experiment.

### Why structural funding/basis is first

The directional line has repeatedly read the same four adaptive historical windows. A new target/model/gating variation on those blocks would add another adaptive design point and any positive would be difficult to interpret.

Funding/basis carry asks a more orthogonal question whose payoff mechanism is not "predict the next BTC direction":

```text
LONG spot BTC
+
SHORT BTC perpetual
≈ delta-neutral directional exposure

candidate carry
= funding received/paid
+ basis convergence/divergence
- spot trading costs
- perpetual trading costs
- slippage
- hedge rebalancing costs
- financing / transfer / operational frictions modelled by the contract
```

P4 does **not** answer this question. P4 used funding/open-interest/basis as predictive information for directional BTC trading. A hedged carry strategy uses funding/basis as the payoff mechanism itself.

This checkpoint is still historical and adaptive. Historical funding regimes are already public knowledge in 2026, so it must be called an **exploratory structural carry feasibility screen**, not confirmation of a future edge.

## Contract requirements for the structural carry feasibility screen

Before any result is calculated, its preregistration must freeze at least:

1. exact venues and instruments for both legs;
2. whether the first screen is same-venue synthetic/replication, spot+USD-M, or another precisely named construction;
3. canonical spot, perpetual mark/index, funding and basis data sources with hashes/provenance;
4. timestamp and funding-payment semantics;
5. hedge ratio and rebalancing rule;
6. whether hedge sizing uses spot, mark, index or another predeclared price;
7. entry/exit rule based only on information available at the decision instant;
8. no use of future realised funding to decide entry;
9. maker/taker assumptions for both legs;
10. explicit slippage/spread model for both legs;
11. turnover and rebalancing costs;
12. funding sign convention and payment base;
13. basis PnL and realised/unrealised accounting;
14. capital/margin model, isolated-margin assumptions and liquidation/stress handling;
15. treatment of borrowing/financing, transfers and venue-specific constraints where applicable;
16. missing-data/outage/reconciliation behavior — fail closed;
17. temporal evaluation structure based on funding/regime periods rather than reusing the directional P6/P7 success bar without justification;
18. minimum sample/payment count and thin-sample diagnostic fixed before results;
19. viability gate fixed before results, including net-of-all-modelled-frictions robustness;
20. stress cases for higher fees/slippage, delayed hedge/rebalance and adverse basis movement;
21. evidence ceiling stating that the historical screen is exploratory/adaptive;
22. explicit prohibition on reading P4-HOLD or Styx to select parameters or rescue the screen.

Do not choose thresholds by searching historical profitability and then call the winner preregistered. If a threshold family must be evaluated, define a chronological inner-selection rule and an untouched evaluation layer before the first result.

## Promotion path if the structural screen is positive

A positive historical structural screen is **not** permission for real money and is not enough to spend Styx.

Preferred sequence:

```text
post-audit repairs
→ preregister structural carry feasibility
→ run and close the historical screen
→ independent audit
→ build/verify a dry-run structural execution path if warranted
→ freeze a paper protocol before future data arrives
→ sustained prospective paper across sufficient funding/regime events
→ mature-system freeze
→ decide whether the weakened historical Styx adds useful one-shot evidence
→ only then consider a separately authorised very-small live experiment
```

If the structural screen is negative, keep the negative result and return to the decision review. Likely remaining axes include cost-aware directional abstention/external replication, target redesign, adaptive retraining/OOD, native microstructure/L1-L2 and neural specialists. Do not launch them all in parallel.

## GO / NO-GO after the audit

- Current `main` as the historical record: **GO with these disclosures/test repairs merged**.
- Design the next preregistration: **GO after/alongside non-scientific remediation; do not calculate structural results before the preregistration commit is pushed**.
- Sustained paper as alpha validation now: **NO-GO** — no eligible directional mode or validated structural alpha exists yet.
- P8 now: **NO-GO**.
- P4-HOLD: **NO-GO; remains retired and unread**.
- Styx now: **NO-GO**; future use must carry the hindsight-era disclosure.
- Any real-money promotion now: **NO-GO**.
- Increase leverage to rescue economics: **NO-GO**.

## Standing post-audit constraints

1. The August-2026 authorship versus 2025 data/seal dates is always disclosed when describing Styx or historical confirmation strength.
2. P6's dirty-tree fitting provenance is never silently upgraded to exact clean-checkout reproducibility.
3. P6/P7 negative results remain narrow and visible.
4. Secondary P6 winners are leads, not fresh evidence and not a reason to relabel P6.
5. P7's better-than-mean behavior is descriptive only; the preregistered fold-wise-best verdict remains negative.
6. P8 remains unopened until genuinely new evidence creates at least two eligible modes under its frozen opening rule.
7. `P4-HOLD` remains retired/unread and Styx remains sealed.
8. The next research mechanism is structural carry; do not mix it with directional feature/model rescue in the same checkpoint.
9. Historical structural research is exploratory/adaptive; genuinely stronger confirmation must eventually be prospective.
10. Aegis remains the sole central risk authority for exposure increases; risk reduction remains available during halts.
11. The Chimera futures chain stays dry-run only during this research stage.
12. The legacy Freqtrade live capability does not create permission to trade.
13. Accounting dimensional checks remain part of every new strategy review.
14. Backtest/live or replay/paper parity is a promotion requirement.
15. Short green paper/testnet runs are not an alpha promotion gate.
16. Robustness diagnostics are not tuning surfaces.
17. One research question at a time.

## Current one-line roadmap

**P6/P6-EXT negative → P7-v1 negative → P8 preregistered but NOT OPENED → independent Fable 5 audit: COHERENT WITH REQUIRED REPAIRS → merge disclosures + positive-control tests + pandas-2.x constraint → preregister ONE exploratory Structural Funding/Basis Carry Feasibility Screen → run/close → independent audit → if warranted, build structural dry-run + freeze prospective paper protocol → sustained future paper → mature freeze → carefully decide whether historically sealed Styx adds value → only then consider separately authorised very-small live.**
