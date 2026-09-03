# ProjectChimera - full independent strategic, scientific and engineering audit

Status: **independent audit record, written from a fresh session on 2026-09-03.**
It is not a preregistration, not a research result and not the authoritative
development plan. The authoritative plan remains `docs/current_development_plan.md`
until the owner adopts or rejects the companion proposal
`docs/proposed_development_plan_post_fable_5_1_audit.md` (labelled PROPOSED -
NOT YET ADOPTED).

Audited state:

| item | value |
| --- | --- |
| `main` at audit time | `177d4b60c7e137730ce88241b481941b07b4cd30` (matches the handoff) |
| draft PR #67 head | `36cdae48877b1d5fa88b2664c127b5307a917751` (matches the handoff; base is `main`; mergeable; CI green) |
| P14 preregistration hash, recomputed from `nn/p14_preregistration.py` | `sha256:830943664906c8cffbdae3b03b8f78e23339123c5d10831a85957ac958eb9b12` (matches the document and the PR) |
| research state, verified by `tools.verify_research_state` | v4, P2a, P2b, P2c, P3, P4, P5, P6, P6-EXT, P7 answered; P8, P13, P14 preregistered |
| CI | green on `main` (`177d4b6`) and on the PR #67 head; the Tests job takes about 7 minutes on Ubuntu |
| local Windows test run (Python 3.13, numpy 2.2.3) | environmental failures only: 26 numpy-dtype errors at `nn/mtf.py:255` and a handful of path-separator assertions; no failure concerns a research number |

Scientific firewall observed by this audit: no P14 statistic was computed, no
P13 economics were run, P8 was not opened, `P4-HOLD` and Styx were not read, no
model was fitted, no data was downloaded beyond public listings and fee pages,
no live route was touched, nothing was merged. Every number quoted below is
copied from a committed artifact or preregistration, or is labelled as external
literature or as this audit's own inference.

Evidence labels used throughout: **[A]** the repository establishes it,
**[B]** published external source, **[C]** this audit's inference,
**[D]** this audit's recommendation.

Method: six fresh-context reconstruction lanes (operational architecture; model
and evaluation pipeline; v4/P2a/P2b/P2c/P3; P4/P5; P6/P6-EXT/P7/P8; P13) read
code, preregistrations, frozen artifacts and Git chronology rather than PR
prose. The auditor read the P14 preregistration, the artifact index, the CI
surface and the front-door plan documents directly, spot-checked every
load-bearing lane claim against the underlying JSON, and read
`docs/current_development_plan.md` and `docs/research_roadmap.md` only after the
reconstruction was complete.

---

## 1. Executive verdict

**ProjectChimera is a well-governed research platform with strong safety and
provenance engineering and no demonstrated alpha, and its current research
direction is not a coherent path to a defensible demo.** The verdict is
PARTLY: the governance and the execution substrate are worth keeping; the
research programme needs to change what it measures and where it looks; P14
should not be the next checkpoint.

Five findings carry the verdict.

1. **The measuring instrument cannot see the effects being searched for.** [A][C]
   Every information-set checkpoint from P2b to P5 decided on "outer net return
   improved in at least 3 of 4 folds" where a fold's net return rests on between
   4 and 280 realised trades. The same OHLCV14 control cell, same features,
   same four periods, same costs, differing only in which rows were eligible,
   returns `+0.106` in fold 3 under P2b and `+0.429` under P5 for XGBoost and
   `-0.026` in fold 2 under P2b and `-0.542` under P4 for LightGBM (section 5).
   A gate applied to a statistic this unstable neither confirms nor refutes an
   information family. The repository states its own coin-null false-positive
   rate at 5/16 per gate and applied the gate about 45 times to the same four
   blocks with no multiplicity control and no statistical test anywhere.
2. **The frozen 20 bps per-trade cost model foreclosed the fast clocks before
   they were tested.** [A][B][C] A 20 bps round trip is roughly 3 standard
   deviations of a one-minute BTC return and roughly 1 standard deviation of a
   six-minute return (section 10). P6's 1m/5m cells, P7's scalping mode and the
   proposed P14 Stage 2 could only ever return "negative on tradability". The
   project has spent most of its historical budget where a pass was
   arithmetically implausible regardless of signal quality.
3. **The research instrument is not the executable instrument.** [A] Every
   economic number since v4 prices a synthetic long/short position on Binance
   spot, whose SHORT leg spot cannot express. The only executor that can hold
   both directions is the dry-run USD-M perpetual layer, and no research
   checkpoint has ever modelled the perpetual.
4. **The four outer blocks are spent, and the future has not been collected.**
   [A] The same four 201-day periods (2023-03-04 to 2025-05-19) have been read
   by ten checkpoints in ten days of wall-clock research (2026-08-20 to
   2026-08-30). The only non-adaptive evidence left is data that has not
   happened yet, and the repository has no prospective collection running: its
   last pulled candle is 2026-08-20.
5. **The operational architecture is built for an ensemble that does not
   exist.** [A] The mode router, the consensus module and the paper chain route
   specialists that were screened out; under committed evidence they decide
   FLAT on every bar. Freqtrade, the inference service, the registry, MLflow and
   Ray Tune are on no path any committed artifact has exercised.

What follows from this: stop reading the four blocks; do not open P14; start
prospective data collection immediately; freeze a small, economically
plausible candidate set for a preregistered prospective validation campaign on
the perpetual (with spot as the hedge leg); collapse the demo architecture to
the futures dry-run executor, Aegis, a live-data simulated venue, a decision
log and telemetry; and accept "no deployable alpha under the current mandate"
as a legitimate end state reachable within about twelve months.

CONFIDENCE: HIGH on findings 1, 3, 4 and 5 (they are read off artifacts);
MEDIUM-HIGH on finding 2 (it rests on external volatility figures and
first-principles arithmetic, not on any P14 data).

---

## 2. Repository and evidence state verified

- `main` is `177d4b6`; the PR #67 branch adds one commit (`36cdae4`), 22 files,
  +3,641/-32 lines, all documentation, preregistration, tests and a source
  preflight artifact. No evaluator, no signal module, no P14 number exists.
  `tests/test_p14_preregistration.py` asserts all of that.
- The hash in the document, the PR body and the module agree, and the module's
  payload reproduces it byte for byte.
- All 20 `artifacts/*_SHA256SUMS.txt` manifests are verified by
  `tests/test_p2b_evidence.py` in CI; the P6/P6-EXT/P7 lane independently
  recomputed 96 of 96 digests with 0 mismatches.
- Preregistration-before-evidence holds for every checkpoint that has a
  preregistration (P4, P5, P6, P6-EXT, P7). The margins are small: P2b 56
  minutes, P6 27 minutes, P7 25 minutes, P6-EXT 8 minutes between the
  gate-fixing commit and the evidence commit. No post-hoc change to any
  threshold, cost, fold, horizon, model configuration or verdict rule was found
  by any lane.
- Two reproducibility gaps are disclosed in the repository and confirmed: the
  P4 derivatives snapshot and the P4-HOLD coverage record are not committed,
  and the v2/v3 walk-forward source runs are absent, so those aggregates cannot
  be re-derived from a clean checkout. A third is confirmed from the artifacts:
  the logistic-regression OHLCV14 control exists in three numerically different
  versions across P2a, P2b/P2c and P3 because `lbfgs` reductions differ across
  BLAS builds; fold 3 of one re-run moved from 31 to 110 trades and from
  `-7.0%` to `-15.7%` with no code or data change
  (`docs/research_reproduction.md` section 7.5). [A]
- Data eras: research-visible `2020-01-04` to `2025-05-19T08:00Z`; `P4-HOLD`
  `2025-05-19T08:00Z` to `2025-08-27T23:00Z` (2,409 hourly rows, retired unread;
  the only P4-HOLD-adjacent information ever obtained is 101 archive HEAD
  responses); Styx `2025-08-27T23:00Z` to `2026-08-20` (sealed, hindsight-era).
  Nothing has been collected after 2026-08-20. [A]
- The programme's wall-clock chronology: a June-2025 LLM-generated skeleton
  (documented as broken in `docs/engineering-audit.md`), fourteen months idle,
  a rebuild merged 2026-08-18, then v4 through P7 answered between 2026-08-20
  and 2026-08-30, P13 preregistered 2026-08-30 and closed 2026-09-02, P14
  preregistered 2026-09-03. Four checkpoints (P6, P7, P6-EXT, P8) were
  designed, preregistered, run and closed inside 95 minutes on 2026-08-30. [A]

---

## 3. What ProjectChimera actually is today

Code volume (lines): `nn/` 38,248 of which about 25,900 are checkpoint-specific
research modules and about 12,400 are the reusable pipeline; `chimera/` 5,518
(of which `futures/` 3,163); `strategies/` 937; `tools/` 9,163; `tests/` 45,845
across 88 files and 2,563 test functions, 33 of which read documentation prose.
Test code exceeds all source code combined. [A]

Component map (KEEP / MODIFY / DEFER / RETIRE is this audit's recommendation
for the route to a first demo; it is not a statement about code quality):

| component | what it solves | real today? | justified? | verdict |
| --- | --- | --- | --- | --- |
| DATA: archive acquisition, snapshot manifests, research contracts, fingerprints (`nn/data_pipeline`, `nn/multiclock`, `nn/p13_sources`, `tools/*snapshot*`) | causal, checksum-verified historical sources with a sealed boundary | yes | yes; the strongest asset | KEEP; add a prospective recorder |
| FEATURES: OHLCV14 (`chimera/features.py`) plus five retired families | scale-free causal features shared by research and strategy | yes | as a baseline only | KEEP as baseline; RETIRE `smc_v1`, `chart_structure_v1`, `microstructure_v1`, `derivatives_v1`, `mtf_v1` from active use |
| LABELS: fixed +/-20 bps band on a 6-bar close-to-close return, 3 classes | cost-aware direction | yes | see section 10 | MODIFY (per-instrument costs; horizon tied to cost arithmetic) |
| MODELS: MTST, XGBoost, LightGBM, logistic regression | directional classification | no edge shown | see section 6 | RETIRE MTST; XGBoost and LR as baselines; RETIRE LightGBM from the deciding set |
| TRAINING: `nn/train.py`, `nn/walkforward.py`, `nn/benchmark.py` | leakage-safe nested walk-forward | yes | yes for its purpose | KEEP; DEFER further use until new data exists |
| EVALUATION: `nn/evaluate.py`, `nn/p2b_compare.py`, decision modules | per-trade cost-aware metrics, fold-count gates | yes | the gate is underpowered | MODIFY (effect-size floor, trade-count floor, statistical test, per-instrument costs) |
| ARTIFACT REGISTRY / EVIDENCE INDEX (`artifacts/README.md`, `nn/research_state.py`, SHA256SUMS) | frozen evidence, machine-derived state | yes | yes | KEEP |
| INFERENCE (`nn/infer_service.py`, `nn/registry.py`, `chimera/inference_client.py`) | serve a promoted model over HTTP | no promoted model exists | premature | DEFER |
| STRATEGIES (`NNPredictorStrategy`, `SwingSpot`, `ScalpFutures`, `ArbMM`) | Freqtrade entry points | not on any artifact path; two are declared dead | no | RETIRE |
| CONSENSUS (`chimera/consensus.py`) | P7 rule | P7 negative | no | RETIRE (archive) |
| MODE ROUTER (`chimera/modes.py`) | select among eligible modes | zero eligible modes; decides FLAT | premature | RETIRE (archive) |
| AEGIS (`chimera/risk.py`) | single risk authority, kill switch | yes | yes | KEEP; MODIFY persistence |
| HERMES / EXECUTION (`chimera/futures/`) | dry-run USD-M perpetual execution, accounting, reconciliation | yes; 16/16 invariants validated | yes; the only executor that can hold both directions | KEEP; add a live-data simulated venue and a two-leg position |
| FREQTRADE (`conf/*.json`, `tools/run_bot.py`, `strategies/`) | legacy spot execution engine | used by no committed artifact | redundant for the demo | RETIRE from the demo path |
| PAPER RUN (`tools/paper_run.py`) | replay chain through modes/Aegis/Hermes | replay only; places zero orders | partly | MODIFY into the demo runner (drop modes/consensus) |
| MONITORING (`chimera/metrics.py`, Prometheus, Grafana, alerts) | telemetry | yes | yes, minimal set | KEEP minimal |
| TELEGRAM (`chimera/notify.py`) | optional operator alerts | guarded; silent without credentials | optional | KEEP optional |
| RESEARCH GOVERNANCE (preregistration modules, hashes, forbidden lists, research-state verifier) | prevents post-hoc rescue | yes | yes, with the gaps in section 10 | KEEP; MODIFY (power, multiplicity, independent pre-open review) |
| MLflow, Ray Tune | tracking, tuning | referenced by no artifact | no | REMOVE |

---

## 4. Full scientific history reconstruction

Shared substrate of everything through P7: Binance spot BTCUSDT; `1h` gen1 or
the gen2 1m-derived clocks; OHLCV14; 20 bps round trip charged once per
realised trade; label = sign of the 6-bar close-to-close return outside a
+/-20 bps band; threshold chosen on the inner block from a 29-point grid with a
10-trade floor; four outer blocks of 4,821 hourly rows:

| fold | outer block |
| --- | --- |
| 0 | 2023-03-04T07:00Z to 2023-09-24T17:00Z |
| 1 | 2023-09-24T17:00Z to 2024-04-12T14:00Z |
| 2 | 2024-04-12T14:00Z to 2024-10-30T11:00Z |
| 3 | 2024-10-30T11:00Z to 2025-05-19T08:00Z |

| checkpoint | question | design | primary gate | result | learned | not learned |
| --- | --- | --- | --- | --- | --- | --- |
| v1-v3 | MTST baseline, earlier generations | same as v4 | none | superseded; v2/v3 source runs not committed | pre-correction metrics | nothing reusable |
| v4 | is MTST on OHLCV14 economically viable? | MTST 72,323 parameters (d_model 64, 2 layers, 4 heads, seq_len 64), 5 seeds, 4 folds, 10 epochs patience 3 | none written; verdict string in code | mean outer net `-0.0193`; beat CASH 9/20; beat buy-and-hold 0/20 (B&H `+0.606`); per-fold means `-0.037, +0.194, -0.180, -0.054` | MTST loses money after costs; weak baselines (majority `-0.667`, momentum `-0.794`) are vacuous floors | whether any model could pay 20 bps on this design |
| P2a | model family or information? | LR, LightGBM, XGBoost untuned on MTST's own 64x14 windows; zero seed dispersion | none written; code committed 3h33m before evidence | XGB `+0.0070`, LGBM `-0.0013`, LR `-0.0558`; XGB per fold `-0.102, +0.033, -0.008, +0.106` on 80/11/4/16 trades | model complexity does not help; XGBoost the strongest family by a hair | whether `+0.007` is distinguishable from zero (it is not; section 5) |
| P2b | does `smc_v1` (39 columns) add information? | 3 models x 3 arms; 3-of-4 folds improved vs control | fixed 56 minutes before evidence | best 2/4; two arms with positive mean; xgboost combined 1/4 | negative for this family | anything about the family at other horizons |
| P2c | does `chart_structure_v1` (30 columns) add information? | as P2b | as P2b | best 2/4; every mean delta negative | negative | same |
| P3 | does hourly `microstructure_v1` (32 columns from 3.38 billion aggTrades) add information? | as P2b | fixed 22h51m before evidence | best 2/4; two positive means with 2/4 | negative at 1h aggregation | short-horizon trade flow (P14's stated gap) |
| P4 | does `derivatives_v1` (funding, OI, basis; 8 columns) add information? | as P2b plus Stage 1 screen with availability rule, MIN_OUTER_TRADES 10, P4-HOLD | preregistered 2026-08-23; 4 amendments before fit; evidence 2026-08-28 | block 0 unavailable (240h OI gap); deciding screen 1 of 3 valid folds improved, mean `-0.0388`, worst `-0.0931`; screened out; P4-HOLD retired unread | negative for this design at 6h | anything about derivatives at horizons spanning several funding settlements |
| P5 | does causal 4h/1d context (`mtf_v1`, 28 columns) add information? | as P2b on a 96.4%-eligible universe | preregistered 4h39m before evidence | xgboost combined 1/4: `+0.115, -0.075, -0.040, -0.184`; fold 2 combined arm had 8 trades | negative | the specialist architecture (P5 is one model with extra columns) |
| P6 | do native-clock specialists (1m/5m/15m/30m/1h) extract cost-aware signal? | 3 families per clock, XGBoost deciding; 6 native bars; same 20 bps; 3-of-4 positive folds and positive mean and beats momentum | preregistered 27 minutes before evidence; P6 fits from a dirty tree (disclosed) | every XGBoost clock exactly 2/4; means `+0.030, +0.011, +0.009, -0.009, -0.027`; secondary LR would pass on 1m/5m/15m and LightGBM on 1m/5m; momentum baseline about `-1.0` on fast clocks | no clock viable under the deciding family; 25 of 84 cell-folds positive gross but negative net | whether the fast clocks carry signal under a cost model that permits a pass |
| P6-EXT | do 4h and 1d specialists carry signal? | as P6 on two slow clocks | preregistered 8 minutes before evidence | XGBoost 4h 0/4 (`-0.201`), 1d 1/4 (`-0.192`); momentum beat both in 2/4; LR 4h 3/4 (`+0.118`), LR 1d mean `+0.282` on 2/4 | SWING not eligible | same post-selection-lead question |
| P7 | does consensus among frozen specialists add value over the fold-wise best constituent? | strict majority with veto; 1m and 5m decision clocks; fits nothing | preregistered 7 minutes after P6 evidence, 25 minutes before its own | 1/4 in each mode; means `-0.0266` and `-0.0343`; day trading realised 13 trades, one fold zero; stale votes up to 3d 20h | consensus-v1 negative against an oracle benchmark | whether fusion adds value at all (benchmark oracle-like; trade counts tiny) |
| P8 | can a causal router choose among eligible modes? | preregistered, unopened | opening needs 2 eligible modes; there are 0 | not run | nothing | nothing |
| P13 | does always-on delta-neutral spot-long/perp-short carry earn robust net returns? | 6 calendar-year blocks, G1-G6, Decimal accounting engine; 4 amendments; 260 archive objects verified | preregistered 2026-08-30 | economics never run; closed on source validity because 192 held hours lack a mark-price row and the design authorised no substitute | source metrology; a source-availability look-ahead class; "sufficiency before freeze" | anything about carry; about 17,000 lines for zero economic information |
| P14 (proposed) | does 1m signed trade-flow imbalance predict the next 1m bar, and survive 20 bps? | one column, no constants, three stages, spot, same four blocks | preregistered; not opened | none | none yet | see section 9 |

Overstatement check: no front-door document overstates any artifact; several
understate caveats (P5 fold-2 thinness; P4's "2 of 4" quoted over all four
folds when the deciding screen was 1 of 3 valid). Four small documentation
transpositions were found (listed in the lane reports and section 10). [A]

---

## 5. Hypothesis-independence and adaptivity map

**How much independent evidence exists.** [A][C] There is one asset, one venue,
and four temporal periods. Every checkpoint after v4 reads those four periods,
and fold k's inner block is fold k-1's outer block, so even the four are not
four independent draws. Counting what was evaluated against them:

| checkpoint | cells | gate applications | reading of the blocks |
| --- | --- | --- | --- |
| v4 | 5 seeds | 1 | 1st |
| P2a | 3 | 3 | 2nd |
| P2b (+ablation, +regimes) | 9 + 6 + 1 | 6 | 3rd-5th |
| P2c | 9 | 6 | 6th |
| P3 | 9 | 6 | 7th |
| P4 | 9 | 6 + 1 screen | 8th (3 of 4 blocks) |
| P5 | 9 | 6 | 9th |
| P6 | 15 | 5 deciding + 10 secondary | 10th |
| P6-EXT | 6 | 2 + 4 secondary | 11th |
| P7 | 2 modes + 7 replays | 2 | 12th |
| total | about 90 fitted cells | about 45 deciding gates, 14 secondary | |

Against a coin null the repository's own gate passes with probability 5/16 for
"3 of 4 improved" (about 0.28 after the positive-mean conjunction). Forty-five
independent applications would be expected to yield about a dozen spurious
passes; zero occurred. The most economical reading is that added feature
families systematically degrade an untuned learner out of sample (so the
per-fold improvement probability is below one half), and that P6's absolute
cells sit slightly below zero after costs. Neither reading says anything about
the presence of a small real effect, because the instrument cannot resolve one.

**Correlated variants.** [C] P2b, P2c, P3, P4 and P5 are five draws of one
hypothesis: "some hand-designed transformation of Binance BTCUSDT history adds
information to OHLCV14 for a 6-hour direction call at 20 bps". P6 and P6-EXT
change the clock but keep the features, horizon-in-bars and cost model. P7 and
P8 are functions of P6. Only P13 and P14 ask different questions, and P13 asked
its on public 2020-2025 funding history that the designer had lived through.
The genuinely orthogonal axes the project has never touched are: the executable
instrument (perpetual), breadth (more than one asset), execution style (maker
versus taker), and the future.

**The same cell across checkpoints.** [A] The OHLCV14 control is the closest
thing the project has to a replicated measurement. Same model, same fourteen
columns, same four periods, same costs; the only difference is which rows were
eligible (P4 masks rows without derivatives coverage from 2020-09; P5 masks
rows without a fresh 4h/1d bar).

| control cell | fold 0 | fold 1 | fold 2 | fold 3 | trades per fold |
| --- | --- | --- | --- | --- | --- |
| XGBoost, P2b/P2c/P3 | -0.102 | +0.033 | -0.008 | +0.106 | 80 / 11 / 4 / 16 |
| XGBoost, P4 universe | -0.020 | +0.051 | -0.104 | -0.043 | 14 / 15 / 17 / 42 |
| XGBoost, P5 universe | -0.228 | +0.048 | +0.134 | +0.429 | 207 / 26 / 29 / 47 |
| LightGBM, P2b/P2c/P3 | -0.047 | -0.189 | -0.026 | +0.256 | 30 / 96 / 21 / 59 |
| LightGBM, P4 universe | -0.018 | -0.112 | -0.542 | +0.185 | 3 / 280 / 170 / 28 |
| LightGBM, P5 universe | -0.114 | -0.026 | -0.111 | +0.080 | 76 / 50 / 72 / 78 |
| logistic, P2a (unpinned BLAS) | -0.292 | +0.055 | +0.159 | -0.145 | |
| logistic, P2b/P2c | -0.007 | +0.066 | +0.131 | -0.157 | 28 / 11 / 75 / 110 |
| logistic, P3 (different BLAS) | -0.020 | +0.055 | +0.159 | -0.070 | 30 / 9 / 72 / 31 |
| logistic, P4 universe | +0.088 | +0.031 | +0.164 | +0.248 | 30 / 22 / 48 / 89 |
| logistic, P5 universe | +0.009 | +0.078 | +0.203 | -0.044 | 2 / 41 / 89 / 116 |

The fold-3 XGBoost figure ranges from `-0.043` to `+0.429`, and the fold-2
LightGBM figure from `-0.026` to `-0.542`, under perturbations that no one
would call a change of hypothesis. This is the central methodological fact of
the programme: the primary statistic is dominated by threshold selection on
noisy probabilities and by a handful of trades, not by the information set.
Every "k of 4" verdict from P2b to P6-EXT should be read with that in mind.

One pattern is worth naming without promoting: the logistic-regression control
is net-positive in 13 of its 20 cell-folds above, while the tree controls flip
sign across universes. This is a post-selection observation on burned blocks
and licenses nothing; it is recorded because it bears on which family belongs
in a prospective candidate menu (section 6).

---

## 6. Model family verdicts

### MTST Transformer

[A] `nn/model_def.py`: 2 encoder layers, d_model 64, 4 heads, FFN 128, dropout
0.1, learned positional embedding, last-timestep head; 72,323 parameters;
input 64 x 14; trained on CPU for at most 10 epochs with patience 3 (best
epochs 3-6); AdamW 3e-4, weight decay 1e-4, class-weighted cross-entropy.
Fitted exactly once in the programme (v4). Training samples per fold 20,763 to
35,157 overlapping windows, i.e. roughly 300-550 non-overlapping windows;
parameters exceed independent observations by one to two orders of magnitude.
Mean outer net `-0.0193`; lost to buy-and-hold in 20/20 run-folds; two of three
untuned tabular models on the same flattened inputs matched or beat it (P2a).
Wall-clock training cost is not recorded in any artifact. No model artifact has
ever been promoted; `artifacts/models/` does not exist.

[C] Nothing in v4 shows useful nonlinear structure: the seed spread (`-0.045`
to `+0.018`) is as large as the effect, and a linear model on the same windows
is within noise. A 64-bar sequence model on 45,000 hourly bars of one asset is
an architecture in search of a dataset.

Verdict: **RETIRE.** Keep the code as history; do not run it as a benchmark
(the tabular baselines already bound it from above at a fraction of the cost).
CONFIDENCE: HIGH.

### XGBoost

[A] Deciding family for P4, P5, P6, P6-EXT (named before fitting); untuned
(200 trees, depth 6, lr 0.05, hist, single thread); the only family with a
positive P2a mean (`+0.0070`); every fast-clock P6 cell exactly 2/4; its
control cell is the least stable of the three across universes (section 5).

[C] Its "secondary leads" (positive means with 2/4) are exactly what a null
generates. There is no basis for another XGBoost checkpoint on these blocks.

Verdict: **KEEP AS BENCHMARK ONLY.** A strong, cheap tabular baseline for any
future dataset; no longer the deciding family; not a candidate.
CONFIDENCE: HIGH.

### LightGBM

[A] Never decisive; numerically the least stable control (fold 2: `-0.026` to
`-0.542`); adds a third gradient-boosting dependency for no distinct
information.

Verdict: **RETIRE from the research family set.** Keep the wrapper if it costs
nothing; drop it from any future cell design. CONFIDENCE: HIGH.

### Logistic regression

[A] Multinomial lbfgs, C=1, untuned; the interpretable null; net-positive in
13 of 20 control cell-folds; would have passed P6's screen on 1m/5m/15m and
P6-EXT's on 4h; not reproducible bit-for-bit across BLAS builds (a 0.02
threshold grid amplifies third-decimal probability differences into
different trade sets).

[C] Its value is as the simplest model that any candidate must beat, and as the
one family whose post-selection leads are at least sign-consistent. If any
directional rule is frozen for prospective validation, a fixed-coefficient
logistic model (or a rule derived from one) is the defensible choice, not a
tree ensemble.

Verdict: **KEEP ACTIVE** as the interpretable baseline and as the only
directional family eligible for a frozen prospective candidate. Freeze the
coefficients, not the fitting procedure, so the BLAS issue disappears.
CONFIDENCE: MEDIUM-HIGH.

### Neural models generally

[A][C] No evidence supports further neural work. The conditions under which it
would become rational again: (i) a dataset with at least the order of 10^5
effectively independent labels (which for a single asset at multi-hour horizons
means breadth across many assets, not more history), (ii) a demonstrated,
prospectively validated nonlinear edge of a tree model over the linear
baseline, and (iii) a cost model under which the horizon where the edge lives
is tradable. Until all three hold, neural work is spending on architecture what
the problem cannot pay back. CONFIDENCE: HIGH.

---

## 7. Market and instrument verdicts

External facts used [B]: Binance spot BTC/USDT base fee 0.10% per side (0.075%
with BNB); USD-M perpetual base fees 0.02% maker / 0.05% taker; funding every
8 hours; `data.binance.vision` publishes for spot: `aggTrades`, `klines`,
`trades`, and for USD-M futures: `aggTrades`, `bookTicker`, `fundingRate`,
`indexPriceKlines`, `klines`, `markPriceKlines`, `premiumIndexKlines`,
`trades` (monthly listings checked 2026-09-03). There is no historical order
book archive for spot; futures has best bid/ask (`bookTicker`) and daily depth
snapshots. Sources: Binance fee schedule summaries (tradersunion.com,
bitdegree.org), Freqtrade documentation, Binance public-data S3 listing.

| instrument | data | fees | shortable | execution | fit to hypotheses | now or later |
| --- | --- | --- | --- | --- | --- | --- |
| BTC spot | deepest history (2017+), 1m klines with taker split, aggTrades | 10 bps taker at VIP0; the project's 5 bps assumption is optimistic for spot | no | Freqtrade or nothing; no SHORT | poor for long/short research; fine as a price control and the long leg of carry | data and hedge leg only |
| BTC USD-M perpetual | klines, funding, mark, index, premium, bookTicker from 2020 | 5 bps taker, 2 bps maker at VIP0; the project's cost model matches this instrument, not spot | yes (native) | the dry-run executor already exists and is validated | correct for every long/short design and for carry | **now** |
| spot + perp combined | basis, funding, both tapes | two legs | hedged | needs a two-leg position the executor does not yet hold | correct for structural carry | now, for carry |
| delta-hedged structural (carry) | funding + basis | 25 bps round trip both legs; low turnover | n/a | two-leg dry run | the only mechanism with an externally known positive expectation and observable payoff | **now**, as the demo spine |
| other crypto assets | as BTC | as BTC | perp | as BTC | the only route to statistical breadth for directional ML | later, and only with a new mandate |
| cross-asset context | mixed | n/a | n/a | n/a | untested; low prior at these horizons | later |

Direct answers:

- **Should spot remain the primary research market?** No. [D] It cannot hold
  the SHORT leg the research has priced since v4, and its retail taker fee is
  double the modelled one. CONFIDENCE: HIGH.
- **Should perpetual futures become the primary research market?** Yes. [D]
  The executable instrument, the cheaper taker instrument, the instrument whose
  archives carry funding, mark and best bid/ask, and the instrument the
  repository's own executor already models. The second decision review's
  reason for deferring it (mark-price gaps) is over-weighted: at 1x and
  delta-neutral the liquidation quantity the mark serves is a rounding term,
  and a prospective recorder captures the mark live with no archive gap
  problem. CONFIDENCE: HIGH.
- **Market-agnostic at the model layer?** Yes; the features are already
  scale-free and the contracts already carry the instrument. [A]
- **Spot retained only as data/control/hedge leg?** Yes. [D]

---

## 8. Feature and information verdicts

| family | hypothesis | evidence | redundancy / causality / leakage | cost | verdict |
| --- | --- | --- | --- | --- | --- |
| OHLCV14 | scale-free technical state carries 6h direction | mean-zero after costs across every model (v4, P2a); the control for everything | causal by construction; prefix-invariance tested; decision at bar close with fill at that close (optimistic, disclosed) | trivial | keep as the **baseline**, not as a default alpha hypothesis |
| SMC / structure (39) | market-structure events add information | P2b negative; ablation shows only `liquidity` with a positive marginal mean | high overlap with OHLCV14 (tree importance 0.41 on OHLCV14) | moderate | retired |
| chart structure (30) | classical patterns add information | P2c negative; every mean delta negative | overlaps OHLCV14 | moderate | retired |
| hourly microstructure (32) | trade-tape statistics add information at 1h | P3 negative | aggregation to the hour removes the mechanism's timescale; alignment verified to the row | 48.75 GB acquisition | retired at 1h |
| derivatives positioning (8) | funding/OI/basis predict 6h direction | P4 screened out; block 0 unavailable | funding settles 8-hourly against a 6h horizon (the preregistration itself flags this) | moderate | retired at 6h; funding belongs in the payoff (carry), not in a 6h predictor |
| HTF context (28) | closed 4h/1d bars add information at 1h | P5 negative | duplicates OHLCV14 at slower clocks | low | retired |
| multi-clock specialists | native-clock models extract signal | P6/P6-EXT negative under XGBoost; five secondary passes consistent with null | same columns at seven clocks; costs foreclosed the fast ones | high (gen2 pipeline) | the data pipeline is kept; the specialist architecture is retired |
| cross-timeframe consensus | agreement adds value | P7 negative vs an oracle; 13 trades in one mode | function of dead specialists | low | retired |
| native 1m trade flow (P14) | signed taker imbalance predicts the next minute | not run | see section 9 | low | do not run as designed |

Should OHLCV14 remain the default baseline? **Yes as the control against which
any candidate is measured; no as the default hypothesis.** Has the project spent
too much budget extracting direction from transformed OHLCV? **Yes**: five
feature families and seven clocks of the same fourteen columns on one asset,
measured with an instrument that cannot resolve the effect sizes in play. Is
native microstructure a rational next axis? **Not at a taker cost model and not
on spot, and not before the future is being recorded.** The rational version of
that axis is a maker-execution question on the perpetual with book data, which
is a different project than the one currently funded. [D] CONFIDENCE: HIGH.

---

## 9. P14 / PR #67 adversarial audit

What the preregistration gets right [A]: one column with no constant to
search; the sign fixed in advance; the machine-readable twin hashed and the
hash recomputed here; a source preflight run before freezing, with the
`taker_buy_base_asset_volume` identity proved against the trade tape on 4,320
of 4,320 minutes; a forbidden list that closes every usual door; the
disclosure that Stage 1 is permissive and condition 3 is weak; the admission
that the anchor's claim is contemporaneous; the honest statement that the
SHORT leg is not executable on spot. As a piece of preregistration craft it is
the best in the repository.

What it gets wrong is the question. Issues, classified:

**BLOCKER B1 - the economic stage is arithmetically foreclosed by the design's
own cost model.** [B][C] Using K33 Research's published average daily BTC
volatility (2.24% in 2025, 2.80% in 2024, 3.34% in 2022) and square-root-of-time
scaling, a one-minute return has a standard deviation of roughly 6-9 bps and
a mean absolute size of roughly 5-7 bps. Stage 2 charges 20 bps per trade held
for one minute. A direction oracle with 100% accuracy on the unconditional
distribution nets about `-13 bps` per trade. Conditioning on extreme `|tfi|`
raises the conditional move size but also admits low-volume minutes (the ratio
is volume-normalised), and no public trade-flow signal has been shown to
deliver a 20 bps conditional mean at one minute after fees. The reachable
result states are therefore NOT EVALUABLE, NEGATIVE, or NEGATIVE ON
TRADABILITY; EXPLORATORY CANDIDATE is nominally allowed and practically
unreachable. A checkpoint whose informative branch is closed by arithmetic
before it runs has near-zero expected information value. The preregistration
concedes the ingredients ("a cost model expressed per trade does not become
cheaper because trades are shorter"; "condition 1 is what binds") without
drawing the conclusion.

**BLOCKER B2 - even its best case does not advance the demo route.** [A][C] It
is an eleventh reading of the four burned blocks, on spot, for a one-minute
taker strategy. A positive would be a candidate for a 1-minute execution
stack the project does not have (no live feed, no 1m paper path; the existing
executor was validated on hourly candles), and its SHORT half cannot be
executed on the modelled instrument.

**MAJOR M1 - same-close fill at one minute.** [A][C] Entry at the exact close
print of the decision bar with zero latency is tolerable at 1h and not at 1m,
where a few hundred milliseconds are a material fraction of the bar and the
close is one trade at the bid or the ask. The 5 bps slippage allowance is
the only offset. The bias is towards a false positive.

**MAJOR M2 - last-print (bid-ask bounce) confound.** [C] A minute of heavy
aggressive buying tends to close at the ask; the next close-to-close return is
then biased against the predicted sign by roughly half the spread. The effect
is small against a 6-9 bps standard deviation but is systematic over 10^5 to
10^6 rows, which is exactly the regime where Stage 1 decides. The design has
no mid-price and does not discuss the confound; a Stage 1 failure would be
reported as "no information" when it may be "the close print is the wrong
reference at this clock".

**MAJOR M3 - programme-wide cost model mismatch, inherited.** [B][A] 5 bps fee
plus 5 bps slippage per side corresponds to USD-M perpetual taker fees, not to
Binance spot (10 bps at VIP0). For spot-modelled cells the round trip is
understated by about 10 bps. This barely matters at 6h and matters a great deal
at 1m.

**MAJOR M4 - "external-replication-first" overstates what the anchor buys.**
[A][B] Silantyev (2019) studies BitMEX XBTUSD with trade and quote data and
reports that trade-flow imbalance explains *contemporaneous* price change
better than order-flow imbalance (abstract as indexed; the full text was not
read by the author or by this audit). P14 changes the venue, the instrument,
the dependent variable and the claim. The anchor fixes a sign convention and
nothing else; the result narrative must not describe P14 as a replication.

**MAJOR M5 - Stage 1 carries no effect-size floor.** [A] With 10^5-10^6 rows
per block, `A > B` by one part in 10^5 passes. The document says so; the
consequence is that Stage 1 contributes essentially no information: if any
persistence exists at all (and the literature on signed order flow says it
does in every market studied), Stage 1 passes, and the checkpoint's outcome is
decided by Stage 2, which B1 has already decided.

**MINOR m1** - Stage 0 is framed as a control "that can fail"; same-bar
agreement between aggressive flow and price change is close to mechanical
(price impact), so Stage 0 is a data-integrity check, which is fine, but the
framing invites over-reading a pass.

**MINOR m2** - `nn/research_state.py:248` and `tools/verify_research_state.py:40`
call `read_text()` without an encoding; the verifier fails on a Windows
cp1251 locale (`UnicodeDecodeError`) and passes under `PYTHONUTF8=1`. Not a
P14 defect; found while verifying the PR.

**MINOR m3** - the 30-trade floor and the 10-trade threshold floor constrain
nothing at 1m (hundreds of trades per fold are guaranteed by any theta on the
grid); harmless.

**NON-ISSUES** - hash integrity; chronology (nothing has been run); the
source contract and preflight; the sign convention; the theta grid and
inner-only selection; the 3-of-4 plus positive-mean rule (reasonable for a
model-free statistic); the exclusion rules; the safety prohibitions; the
forbidden list; multiplicity discipline (one signal, one clock, one gate).

Answers to the specific challenges:

- Why spot: inertia ("every checkpoint reads spot"). Wrong instrument for a
  long/short rule. Why BTCUSDT: fine. Why 1 minute: fixed by an anchor that
  says nothing predictive; the worst horizon for the frozen cost model. Formula
  and sign: correct and proved. Bar timing and causal boundary: correct.
  Missing intervals: handled. Outer blocks: burned. Stage 0: integrity check.
  Stage 1: uninformative by construction. Stage 2: foreclosed. Best-constant
  comparator: a fair hindsight floor. Positive-mean rule: fine. 20 bps: wrong
  for spot, right for the perpetual. Threshold selection: fine. Evidence
  ceiling: stated correctly. Contemporaneous versus predictive: correctly kept
  apart. Multiplicity: handled. Can Stage 0 forfeit a valid mechanism: only
  through a data defect, which is its purpose. Does Stage 1's sample size make
  passing trivial: yes, and the document admits it. Is sign agreement
  economically meaningful: no; the document admits that too. Is the economic
  stage reachable under a sensible gate: no (B1). Does the design answer a
  question worth another historical checkpoint: no.

**Recommendation: REPLACE P14 WITH A DIFFERENT NEXT CHECKPOINT.** [D] The
reasons are information-value reasons (B1, B2, M5) and a design-level
mismatch between horizon and cost model, not any P14 outcome, because none
exists. A "repair" that made P14 worthwhile would change the horizon family,
the instrument and the execution/cost model, each of which the preregistration
itself classifies as a different checkpoint; that is a replacement, not a
repair. Handling of PR #67: do not merge it as the next research contract.
Close it with this audit linked, keeping the branch as the record of a
design considered and declined; if the owner wants the source preflight and
the sign-convention proof in `main`, cherry-pick the artifact directory alone
with a STATUS note that the design was declined at audit. The replacement is
the prospective validation campaign in section 20.
CONFIDENCE: HIGH on the recommendation; MEDIUM-HIGH on B1's arithmetic.

---

## 10. Research-methodology audit

| item | finding | status |
| --- | --- | --- |
| temporal / label / preprocessing / normalisation leakage | windows end before labels begin; scaler fit on train only; thresholds on inner only; early stopping never sees outer; poisoning tests past the visible bound leave outputs byte-identical | **strong** [A] |
| purge / embargo | purge equals the horizon (6 bars) by index arithmetic; no extra embargo; adequate because training only expands forward | adequate [A] |
| fold contamination | fold k's inner block is fold k-1's outer block; disclosed; reduces four folds to fewer than four draws | disclosed, uncorrected [A] |
| nested walk-forward | correctly implemented; fold geometry fixed by contract | strong [A] |
| multiple testing | about 45 deciding gates and 14 secondary evaluations on four blocks; no FWER/FDR; coin-null FPR 5/16 self-declared | **weak** [A] |
| repeated-holdout adaptation | twelve readings of the same blocks; the repository labels every result adaptive and refuses confirmation claims | disclosed; the only cure is new data [A] |
| researcher degrees of freedom | pre-fixed grids, costs, folds, models; forbidden lists; hashes | strong on paper; see "pace" below |
| metric and gate selection | net return after costs, 3-of-4 folds; no effect-size floor; trade counts 4-280 per fold; no statistical test; Sharpe i.i.d. with no HAC or deflation | **weak** [A] |
| survivor bias | single asset chosen before the programme; none within it | fine |
| source substitution | refused everywhere (P13 forfeited rather than substitute) | strong, arguably rigid [A] |
| missing data | segment discipline; no forward fill; availability rules | strong [A] |
| transaction-cost realism | 20 bps per trade, flat, no maker/taker split, no spread, no impact; matches perp taker not spot; applied identically at every clock | **inadequate at fast clocks; optimistic on spot** [A][B] |
| turnover | reported; the momentum baseline churns to `-1.0` on fast clocks, the majority baseline loses 75.6% per fold, so "beats baselines" is vacuous | disclosed [A] |
| slippage | 5 bps flat; fill at the decision close with zero latency | optimistic [A] |
| funding / liquidation | absent from every directional cell (spot); modelled only in the futures executor and the unrun P13 engine | not applicable yet |
| market impact | none | acceptable at the sizes contemplated |
| class imbalance | SHORT 17,597 / HOLD 9,308 / LONG 18,897 at 1h; HOLD is not the majority class, contradicting `nn/train.py` and `docs/ml_pipeline.md`; HOLD share 80% at 1m | documentation defect [A] |
| sample dependence / effective N | overlapping 64-bar windows and 6-bar labels; realised trades are the binding unit and are tiny | **the central weakness** [A] |
| statistical vs economic significance | conflated in the opposite direction from usual: economically sized gates applied to statistically meaningless samples | [C] |
| baseline quality | momentum baseline evaluated on standardised `ema_cross` (deadband in training-set standard deviations; "positive" means above the training mean), contradicting its docstring; no test covers it | defect [A] |
| probability calibration | ECE reported and gated at promotion; no recalibration | adequate |
| abstention logic | threshold on the winning class chosen on inner; correct | adequate |
| selective reporting | none found; negatives are the most visible thing in the repository | strong [A] |
| post-hoc rescue | none found in any lane | strong [A] |
| pace | ten checkpoints in ten days; preregistration-to-evidence gaps of 8-56 minutes; no independent review before a checkpoint opened until the first Fable audit on 2026-08-30 | **the preregistrations are ordering guarantees, not reviewed commitments** [A][C] |
| P13 rigidity | an empty surrogate tuple plus terminal treatment turned an archive publication defect in a rounding-term quantity into a lost checkpoint; the daily `markPriceKlines` objects were never probed | [A][C] |

Overall: **strict about process, not strict enough about statistics, and rigid
where flexibility was cheap.** The discipline that prevents post-hoc rescue is
genuinely ahead of common practice. The discipline that would have told the
team, before P2b, that the gate could not resolve a plausible effect size was
absent, and it is the absence that let five checkpoints be spent measuring
noise. Required changes to the methodology (all before any new evidence):

1. every preregistration states an effect-size floor, a minimum realised-trade
   count per fold that makes the floor detectable, and the gate's
   false-positive and power estimates under a stated null and alternative;
2. per-instrument cost models (maker/taker, spread) and a horizon justified
   against them;
3. a statistical test appropriate to dependent data (block bootstrap or
   stationary bootstrap on trade-level returns) reported beside the fold
   count, and an explicit multiplicity budget per campaign;
4. an independent review gate before a checkpoint is opened, with a minimum
   elapsed time between preregistration and first result;
5. no further use of the four outer blocks for deciding evidence.

---

## 11. Software-architecture audit

| component | classification | evidence |
| --- | --- | --- |
| `chimera/risk.py` (Aegis) | ESSENTIAL NOW | single gate; halted checked first; halt persisted; kill-switch file; unreadable halt file starts halted |
| `chimera/futures/*` (Hermes, ledger, venue, store) | ESSENTIAL NOW | dry-run only by construction; `dry_run=False` raises; leverage other than 1 raises; 16/16 invariants on 4,821 candles; atomic persisted state; reconciliation halts; emergency flatten |
| `chimera/features.py`, `chimera/contracts.py` | ESSENTIAL NOW | shared feature definition and decision rule |
| `chimera/metrics.py`, Prometheus, alert rules | USEFUL SOON (minimal set) | wired; bounded labels; one documented metric (`mode_active_seconds_total`) does not exist |
| Grafana dashboards | PREMATURE beyond one panel set | nothing has ever run long enough to look at |
| `chimera/notify.py` | USEFUL SOON, optional | silent without credentials; token redacted |
| `tools/paper_run.py` | USEFUL SOON after modification | replay-only today; `LiveSource` refuses because no specialist persists an estimator; routes through modes/consensus that must go |
| `chimera/modes.py`, `chimera/consensus.py` | SHOULD BE REMOVED (archived) | route dead specialists; FLAT on every bar; P8 unreachable |
| `strategies/*` and Freqtrade | REDUNDANT for the demo | imported in four files; used by no committed artifact; `ScalpFutures` and `ArbMM` declared dead; Freqtrade cannot hold a two-leg position and its dry-run funding model is weak by its own documentation |
| `nn/infer_service.py`, `nn/registry.py`, `chimera/inference_client.py` | PREMATURE | no model to serve; a frozen rule runs in-process |
| MLflow, Ray Tune | SHOULD BE REMOVED | optional extras referenced by no artifact |
| Docker topology (Freqtrade image, inference image, Prometheus, Grafana, Alertmanager) | PREMATURE in breadth | `docs/dry_run.md` records that no container has ever been started end to end |
| research modules `nn/p*.py` | future-only / historical | keep as frozen history; not on any runtime path |

Are we building an execution platform around strategies with no demonstrated
edge? **Yes, in two of three layers.** The futures execution layer and Aegis
are justified because any demo needs them and they are validated. The
mode/consensus/router layer and the inference/registry/Freqtrade layer are
built for an architecture the evidence has retired. Development on those two
layers should stop now. [D] CONFIDENCE: HIGH.

Two engineering findings that matter for a sustained demo [A]:

- `RiskEngine` persists only its halt. Peak equity, day-start equity, the loss
  streak, the cooldown deadline and the order-rate window reset on restart, so
  a drawdown halt cannot fire on pre-restart history. Must be fixed before any
  multi-day run.
- The test suite is environment-sensitive: green on Ubuntu with numpy 2.4.6,
  but 26 tests fail on numpy 2.2.3 (`nn/mtf.py:255`, a datetime-division
  dtype error) and several evidence-coverage tests assert Linux path
  separators. There is no lock file; dependencies float (freqtrade 2026.8,
  xgboost 3.2.0, torch 2.14 in the last CI run).

---

## 12. Risk and execution audit

Aegis [A]: `RiskEngine.evaluate_entry` is reached by the Freqtrade path through
`confirm_trade_entry` and by the futures executor through `_ask_aegis` for
every exposure-increasing order; reductions, closes and flattens bypass it by
design so a halted engine can still reduce risk; `emergency_flatten` passes
`bypass_risk_gate=True`. `max_leverage` is checked on every entry
unconditionally; funding-rate and liquidation-distance limits apply only when
the caller supplies them, and the executor falls back to `liquidation_price=None`
on a `PositionError`, so that limit can silently not apply for one order.
Strategy code cannot reach the venue except through the executor. Fail-closed
behaviour: inference failures become HOLD; a missing model artifact yields no
signals; a raising risk module blocks the order; an unreadable halt file starts
the engine halted; reconciliation mismatch halts and records a symbol-level
dispute that `require_ready` refuses on.

Live reachability [A]: the futures chain has no authenticated route at all
(asserted by AST over the package; `LiveFuturesNotImplemented`). The legacy
Freqtrade spot path is live-capable in principle and double-gated
(`ENABLE_LIVE_TRADING=I_UNDERSTAND_THE_RISK` and `--mode live`; every committed
config keeps `dry_run: true`; CI asserts the refusal). Credentials would be
read only by the Freqtrade launcher. Removing the Freqtrade path removes the
only live-capable code in the repository, which is itself an argument for
removing it during the demo stage.

Execution [A][B][C]: Freqtrade supports futures with isolated margin and
shorting but cannot hold a two-leg spot/perp position, and its documentation
warns that dry-run funding accounting is unreliable. The custom layer models
funding, partial fills, reconciliation and restart, and is the only substrate
that can express the demo's positions. Direct exchange integration for
real orders is out of scope until a candidate survives prospective
validation. Verdict: **the first demo needs the custom futures layer with a
live-data simulated venue, and does not need Freqtrade.** [D]

Paper/live isolation: adequate today; strengthen by deleting the live-capable
path rather than gating it.

Drawdown / exposure / correlation controls for the first demo: exposure cap at
1x notional, a daily-loss halt, a drawdown halt with persisted peak equity, a
funding-rate halt for the carry position, a feed-staleness halt, and a
reconciliation halt. Correlation controls are unnecessary for one instrument.

---

## 13. Project viability

| dimension | verdict | basis |
| --- | --- | --- |
| software | **PARTIAL** | the safety, provenance and dry-run execution core is solid and tested; nothing has ever run sustained against live data; two persistence and portability gaps; two layers built for a retired architecture |
| research | **PARTIAL** | credible at producing negatives and preventing rescue; not capable, as designed, of detecting a plausible positive; its historical data is spent |
| alpha | **NO** | no cell passed any preregistered gate; the leads are consistent with noise |
| economic | **NO** (evidence); structurally doubtful for taker-cost directional trading of one asset at multi-hour horizons | section 10 arithmetic and the control-cell instability |
| demo / paper | **PARTIAL** | the platform could run a FLAT or a carry position in dry-run today; there is no ML candidate that has earned a demo |
| live | **NO** | nothing authorises it and nothing should |

"NO DEPLOYABLE ALPHA FOUND UNDER CURRENT MANDATE" is a legitimate final
conclusion and, on the present evidence, the most likely one for the
directional-ML mandate. It is not yet the conclusion for the structural
mandate, which has never been tested.

---

## 14. KEEP

1. Preregistration discipline: hashed machine-readable designs, forbidden
   lists, superseded-hash chains, amendment-before-number.
2. The evidence index and the research-state verifier (a document that
   contradicts the artifact tree is a CI failure).
3. Source acquisition and provenance: checksum-verified archives, semantic
   fingerprints, research contracts with an immutable boundary, the
   no-forward-fill alignment code from P13.
4. The leakage battery and nested walk-forward implementation.
5. Aegis as the single authority for exposure increases, with reductions
   always available.
6. The futures dry-run executor, ledger, venue constraints, store and the
   16-invariant validation protocol.
7. The Decimal accounting engine and hand-traced witnesses from P13.
8. Negative results kept visible, with narrow closures.
9. OHLCV14 as the control and logistic regression as the interpretable null.
10. The operating guide's separation of executor and independent reviewer.

## 15. MODIFY

1. Evaluation: add an effect-size floor, trade-count floor, dependent-data
   test and multiplicity budget to every future design.
2. Costs: per-instrument maker/taker/spread models; retire the flat 20 bps.
3. Instrument: perpetual as the modelled and executed instrument; spot as
   hedge leg and control.
4. Aegis: persist full state; add feed-staleness and funding-rate halts.
5. `tools/paper_run.py`: strip modes/consensus; drive a frozen rule set; add
   a decision log with input hashes; add a live-data simulated venue.
6. Portability: pin dependencies (lock file); fix `read_text()` encodings;
   fix the numpy-2.2 dtype error; make path assertions OS-neutral.
7. Documentation: correct the majority-class claim, the momentum-baseline
   description, `mode_active_seconds_total`, the `max_leverage` caveat and the
   "P7 pins its sources properly" sentence (P7 cells record `dirty: true`).

## 16. DEFER

1. The inference service, registry and promotion gates (until a candidate
   exists that needs serving).
2. Neural models (until the conditions in section 6 hold).
3. Cross-asset breadth (a new mandate, after the prospective campaign).
4. Styx and P4-HOLD: keep sealed; do not plan around them; they are weak,
   hindsight-era, one-shot checks at best.
5. Any maker-execution or order-book research (needs a live book recorder
   first).

## 17. RETIRE

1. MTST as an active model.
2. LightGBM from the research family set.
3. `smc_v1`, `chart_structure_v1`, `microstructure_v1`, `derivatives_v1`,
   `mtf_v1` as active feature families (keep as frozen history).
4. The multi-clock specialist architecture, `chimera/consensus.py` and
   `chimera/modes.py` (archive; P7 negative; P8 unreachable).
5. P8: withdraw the preregistration as moot (its opening condition cannot be
   met without refitting screened-out clocks, which its own rules forbid).
6. Freqtrade, its configs, its Docker image and the four strategies, from the
   demo path (delete after the demo protocol is frozen).
7. MLflow and Ray Tune extras.
8. P14 as designed.

## 18. STOP DOING

| stop | why | what would reopen it |
| --- | --- | --- |
| reading the four outer blocks for deciding evidence | twelve readings; the instrument cannot resolve the effects; every result is adaptive by the repository's own labels | nothing; they are spent for decisions and remain useful only as descriptive context |
| directional feature-family search on OHLCV transforms of one asset | five families negative; control-cell instability shows the gate measures noise | breadth (many assets) with a powered design, or a cost model that opens faster clocks (maker execution with book data) |
| Transformer / neural work | no evidence of nonlinear structure; parameters exceed independent observations | the three conditions in section 6 |
| timeframe searches | seven clocks screened; the fast ones were cost-foreclosed | a per-clock cost model that is not foreclosed by arithmetic |
| consensus and router work | negative and unreachable respectively | two prospectively validated specialists |
| Freqtrade and inference-service engineering | on no demo path | a promoted model or a single-leg spot strategy with an edge |
| historical checkpoints designed and run in the same session | preregistration-to-evidence gaps of minutes remove the point of preregistration | an independent pre-open review and a minimum elapsed time |
| flat 20 bps per trade at every clock and instrument | wrong for spot, foreclosing for fast clocks | per-instrument models |
| spot as the modelled instrument for long/short rules | the SHORT leg does not exist | none; spot is the hedge leg |
| treating Styx as future confirmation | hindsight-era | none; prospective data replaces it |

---

## 19. Current roadmap verdict

**PLAN NEEDS MAJOR REVISION.** [D]

`docs/current_development_plan.md` and `docs/research_roadmap.md` are honest,
detailed and internally consistent, and their standing constraints are mostly
right. They are wrong about direction in four places:

1. Their target architecture (five specialists, Pythia, mode controller,
   router) has been refuted or made unreachable by their own evidence, yet it
   remains the "intended Chimera architecture" and shapes the engineering.
2. Their next checkpoint (P14) has near-zero information value (section 9).
3. Their promotion path still runs "historical checkpoint -> historical
   checkpoint -> ... -> paper" with no finite budget; the roadmap's own text
   says the blocks can support "not much" after eight readings and then
   schedules a twelfth.
4. They defer the instrument axis for a reason that does not survive a
   prospective design, and they treat structural carry as a historical
   screen when it is cheaply and decisively testable forward.

What the plan gets right and the proposal keeps: one question at a time;
sufficiency before freeze; negatives visible; Aegis authority; no live
trading; the Styx disclosure; the refusal to promote secondary leads.

The replacement plan is `docs/proposed_development_plan_post_fable_5_1_audit.md`
(PROPOSED - NOT YET ADOPTED). Its skeleton follows.

---

## 20. Proposed roadmap from the current state to the first serious demo run

Designed backwards from the endpoint: a frozen candidate set operating
prospectively in dry-run on live market data for six months, with realistic
per-instrument costs, persistence, failure handling, replay parity, operator
visibility and no retrospective changes.

| stage | question | why | input | output | pass | fail | failure means | stops or redirects | engineering | information value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 Decision and freeze (week 1) | adopt this plan? decline P14? | a plan that keeps reading burned blocks cannot reach a demo | this audit | adopted plan; PR #67 closed; research budget frozen | owner decision recorded | none | n/a | n/a | trivial | none |
| S1 Prospective recorder (weeks 1-3, then continuous) | can the project collect causally clean forward data? | every later stage needs it; none exists | Binance USD-M BTCUSDT and spot BTCUSDT streams: 1m klines, mark, index, funding, best bid/ask | gen3 contract; append-only checksummed daily files; coverage report | 30 consecutive days at >= 99.5% minute coverage, reconciled against the published archives when they appear | cannot collect or reconcile | the project cannot do prospective science | stops | small | high (enables everything) |
| S2 Candidate freeze and prospective preregistration (weeks 2-5) | which frozen rules are evaluated forward, and by what rule? | the only non-adaptive evidence available | P13 accounting engine; the control-cell record; external anchors | PVC-1 protocol: one operational rule (delta-neutral carry, 1x) plus at most two shadow directional rules with frozen parameters; per-rule pass/fail; multiplicity budget; 6-month window; monthly blocks | independent review approves before the first scored day | reviewer rejects | design work continues; no data is scored | redirects | small | high |
| S3 Minimum Viable Chimera build (weeks 3-9) | can the platform run the frozen rules against live data with simulated fills? | the demo substrate | futures executor, Aegis, metrics | live-data simulated venue; two-leg position; persisted Aegis state; decision log; alerts; runbook; replay-parity harness | replay of recorder data reproduces every decision byte for byte; all 16 dry-run invariants hold on live data | parity failures unresolved after one repair cycle | the platform cannot be trusted to report what it did | stops until repaired | medium | medium |
| S4 Soak and parity (weeks 9-11) | does it survive restarts, outages, reconciliation drills? | sustained operation is the claim | S3 | soak report | 14 days, >= 99% uptime, zero unexplained divergence | otherwise | not ready | repairs, then repeat once | small | medium |
| S5 PVC-1 sustained run (months 3-9) | do the frozen rules pay net of modelled costs, prospectively? | the scientific checkpoint | S2 protocol, S3 platform | monthly reports; no parameter changes | per the preregistered rules | per the preregistered rules | for the carry rule: the mechanism does not pay in this regime; for shadow rules: the lead was noise | see section 24 | none | **decisive** |
| S6 Closure and independent audit (month 10) | what did PVC-1 establish? | one decision, made once | S5 | verdicts; either a promotion case for a separately authorised tiny live consideration, or NO DEPLOYABLE ALPHA | audit confirms the verdicts | n/a | n/a | ends the campaign | small | high |

Additional historical alpha experiments are **not justified** under the
current data. P14 is not the last historical checkpoint; P7 was. If, and only
if, S5 leaves an ambiguous verdict, one second campaign (PVC-2) may re-freeze
at most two rules for a further six months. After two campaigns the project
decides.

---

## 21. Finite historical research budget

- Deciding use of the four outer blocks: **zero** further checkpoints.
- New historical checkpoints on genuinely new data (for example cross-asset
  breadth): **at most one**, and only after S5 is running, under the modified
  methodology of section 10, with an independent pre-open review.
- Prospective campaigns: **at most two**, six months each, with a frozen menu
  of at most three rules per campaign and a declared multiplicity budget.
- Stopping policy: after PVC-2 at the latest (about 14 months from adoption),
  either freeze a candidate for a separately authorised very small live
  consideration or record "no deployable alpha under the current mandate" and
  either close the project or open a new mandate (breadth or maker execution)
  with its own budget.

---

## 22. Minimum Viable Chimera - DEMO

```
DATA:
  gen3 prospective recorder: Binance USD-M BTCUSDT (1m klines, mark, index,
  funding, best bid/ask) and Binance spot BTCUSDT (1m klines); append-only,
  checksummed, reconciled against public archives; the existing archive tools
  for backfill and verification.

SIGNAL:
  R1 operational: always-on delta-neutral carry (spot long / perp short, 1x,
  equal quantity), frozen entry/exit/rebalance rules, funding-rate halt.
  R2, R3 shadow (optional, at most two): frozen directional rules with fixed
  parameters (a fixed-coefficient logistic rule on OHLCV14 at a multi-hour
  horizon, and/or a daily time-series momentum rule), signals logged and
  scored, positions simulated at zero size or not at all.

MODEL:
  none fitted online; frozen coefficients only.

RISK:
  Aegis with persisted full state; 1x exposure cap; daily-loss halt; drawdown
  halt; funding-rate halt; feed-staleness halt; reconciliation halt; kill-switch
  file; reductions always available.

EXECUTION:
  chimera.futures executor with a live-data simulated venue (fills at observed
  bid/ask plus a slippage model; per-instrument maker/taker fees; funding from
  observed settlements); a two-leg position for R1; no credentials; no live route.

STATE:
  FuturesStore atomic JSON; append-only decision log (inputs hash, outputs,
  veto reasons); daily snapshots; replay-parity harness.

OBSERVABILITY:
  Prometheus metrics (existing series); alert rules for halt, reconciliation,
  feed staleness, restart; one dashboard; Telegram optional.

OPERATOR:
  runbook; daily report script; kill switch; monthly frozen report.

COMPONENTS EXCLUDED:
  MTST and training stack; inference service and registry; Freqtrade, its
  configs, image and strategies; modes and consensus; MLflow; Ray Tune;
  Grafana beyond one dashboard; every checkpoint-specific research module
  (kept in the repository as frozen history).
```

Compared with the current repository, roughly a third of the operational code
is needed (Aegis, futures, features, contracts, metrics, acquisition and
verification tools, governance), roughly a third is premature (modes,
consensus, inference, registry, Freqtrade strategies, Docker breadth), and
roughly a third is historical research code that belongs in the record and on
no runtime path. [C]

---

## 23. Demo entry criteria

1. S1 recorder passes 30 days.
2. S2 protocol independently reviewed and committed before the first scored
   day, with per-rule pass/fail, effect-size floors, minimum settlement and
   trade counts, and a multiplicity budget.
3. S3 replay parity: the platform reproduces every decision from recorded data.
4. S4 soak: 14 days, restarts and outage drills, all dry-run invariants hold
   on live data.
5. Aegis state persistence and the feed-staleness and funding halts merged and
   tested with two-sided synthetic controls.
6. No live-capable path on the demo branch (Freqtrade removed or unreachable).
7. Per-instrument cost model documented and hand-traced for LONG, SHORT and
   the two-leg position.

---

## 24. Project kill / pivot criteria

- **Kill (directional mandate):** no shadow rule meets its preregistered rule
  in PVC-1 and PVC-2. Record "no deployable directional alpha under the current
  mandate".
- **Kill (structural mandate):** the carry rule's net return after modelled
  costs is not positive over the campaign, or its worst monthly block breaches
  the preregistered floor, and PVC-2 repeats it.
- **Pivot to a new mandate** (breadth or maker execution) only with a fresh
  budget and a fresh preregistration; never by re-reading burned blocks.
- **Operational kill:** replay parity cannot be restored after one repair
  cycle; a reconciliation or safety invariant is breached without explanation;
  the recorder cannot maintain coverage for 30 consecutive days.
- **Signal half-life:** any rule whose realised holding period is shorter than
  the platform's measured decision-to-fill latency is dropped from the menu.
- **Complexity:** if engineering effort on the platform exceeds a preset cap
  (for example eight person-weeks) before S5 starts, stop and simplify.

---

## 25. Top 10 risks

1. Continuing to spend the historical budget on the four blocks because each
   checkpoint is cheap and the pace is exhilarating.
2. Opening P14 and reading "negative on tradability" as evidence about trade
   flow.
3. Promoting a post-selection lead (LR on fast clocks; LR OHLCV14 control)
   into a demo without a prospective protocol.
4. The demo running an architecture (modes/consensus) that decides FLAT
   forever and being mistaken for validation.
5. Aegis state loss on restart during a multi-day run.
6. Dependency drift (no lock file) silently changing frozen-number
   reproducibility, already demonstrated for logistic regression.
7. Treating Styx or P4-HOLD as prospective evidence.
8. Carry-specific tail risks (funding sign flips, basis dislocation, exchange
   halts) being under-modelled in the two-leg simulation.
9. A live-capable Freqtrade path surviving into the demo branch.
10. Owner fatigue with a programme that has produced only negatives; the cure
    is a finite budget with a decisive endpoint, not another checkpoint.

## 26. Top 10 strengths

1. Preregistration with hashed machine-readable designs and amendment chains.
2. The evidence index and the research-state verifier.
3. Checksum-verified, semantically fingerprinted, boundary-sealed sources.
4. A leakage battery with two-sided synthetic controls.
5. Aegis: single authority, fail-closed, persisted halt.
6. A futures dry-run executor validated against 16 frozen invariants.
7. Decimal accounting with hand-traced witnesses.
8. Negative results published narrowly and kept visible.
9. Honest front-door documents; no overstatement found by any lane.
10. An operating guide that separates executor from independent reviewer.

---

## 27. Previous decisions this audit disagrees with

1. Selecting P14 as the next checkpoint (second decision review): information
   value near zero (section 9).
2. Deferring the instrument axis because of mark-price archive gaps: the
   quantity the mark serves is a rounding term at 1x, and a prospective
   recorder removes the gap.
3. A flat 20 bps per-trade cost at every clock: it foreclosed P6's fast
   clocks, P7's scalping mode and P14 before any fit, and it understates
   spot fees.
4. Running ten checkpoints on four blocks in ten days without a power or
   effect-size analysis: the programme measured noise with precision.
5. The 3-of-4 fold gate with trade counts of 4-280 and no statistical test.
6. P13's forfeiture design (empty surrogate tuple, terminal treatment) and
   the closure's omission of the daily-archive probe; and the first Fable
   audit's choice to test carry historically rather than prospectively.
7. Building modes, consensus, router, inference service and paper chain before
   any specialist was viable.
8. Keeping Freqtrade as "the execution engine" in the README while every
   validated path uses the custom layer.
9. Describing P14 as "external-replication-first".
10. Preregistration-to-evidence gaps of minutes with no independent pre-open
    review (repaired only by the audits that followed).

## 28. Previous decisions this audit agrees with

1. Preregistration before result, amendments before numbers, superseded
   hashes kept.
2. Refusing to promote P6's secondary leads and refusing to retrofit P7.
3. Retiring P4-HOLD unread after a failed screen.
4. Disclosing Styx's hindsight-era ceiling.
5. Not opening P8.
6. Dry-run-only futures with no live route, and the two-claims distinction
   about live capability.
7. Source sufficiency before freeze.
8. The evidence-class labels (primary, derived, operational).
9. Refusing to redesign P13 against the data that arrived (the audit would
   have probed the daily archive first, but agrees with not re-specifying).
10. The choice of untuned simple models as the family set after v4.

---

## 29. Immediate next action - exactly one

**Build and start the prospective market-data recorder (S1) for Binance USD-M
BTCUSDT and spot BTCUSDT, committed under a gen3 research contract, before any
other research or engineering.** Every day without forward data is a day the
only decisive evidence the project can still obtain is not being collected.
Declining P14 is a decision the owner records in the same week; it is not
work.

Then, in order: S2 (freeze the PVC-1 menu and protocol; independent review),
S3 (Minimum Viable Chimera), S4 (soak and parity), S5 (six-month run), S6
(closure and audit).

---

## 30. Final 12-month strategic view

Months 0-3: recorder running; PVC-1 protocol frozen and reviewed; Minimum
Viable Chimera built; soak passed; the retired layers archived; Freqtrade
removed from the demo branch; dependencies pinned.

Months 3-9: PVC-1 runs untouched. Monthly frozen reports. The most likely
outcome on present evidence: the carry rule is marginally positive or flat
net of costs with identifiable regime dependence, and the shadow directional
rules are indistinguishable from zero. That outcome is a success of the
programme, not a failure: it is the first non-adaptive evidence the project
will have produced.

Months 9-12: closure and independent audit. Either a narrow promotion case
for a separately authorised very small live consideration of the carry rule
(with the demo's operational record as the primary evidence), or a recorded
"no deployable alpha under the current mandate" and a decision about a new
mandate. In neither case does the project return to the four blocks.

The overall idea - a disciplined, safety-first, reproducible platform for
finding out whether a small systematic crypto strategy can be run honestly -
is worth continuing under a finite budget. The idea that a single-asset
directional ML classifier on transformed OHLCV would be that strategy is not,
and the evidence the project itself produced is what says so.

---

## Appendix A. Direct answers to the twenty-five special questions

1. Continue researching BTC spot? No, as the modelled instrument for long/short
   rules; yes as data and the carry hedge leg.
2. Futures replace spot as the main instrument? Yes.
3. Transformer remain? No; retire.
4. XGBoost remain? As a benchmark only.
5. LightGBM remain? No; retire from the family set.
6. Logistic regression remain? Yes, as the baseline and the only directional
   family eligible for a frozen prospective candidate.
7. OHLCV14 remain the default feature baseline? As the control, yes; as the
   default hypothesis, no.
8. Is native 1m trade flow worth P14? Not as designed (taker cost, spot,
   burned blocks, one-minute horizon).
9. Should PR #67 be merged? No; close it with this audit linked; optionally
   cherry-pick the preflight artifact with a "declined" note.
10. Is P8 worth keeping preregistered? No; withdraw as moot.
11. Is the mode-router architecture premature? Yes; retire.
12. Is consensus worth further research? No.
13. Is Freqtrade still the right execution framework? No; the custom futures
    layer is the demo substrate.
14. Is the custom futures layer necessary? Yes; it is the only executor that
    can hold the positions the research and the demo need.
15. Is Aegis architecturally sound? Yes, with the persistence fix and two new
    halts.
16. Are we over-engineered? Yes, in the mode/consensus/router and the
    Freqtrade/inference layers; not in safety, provenance or execution.
17. Are we over-testing the same historical data? Yes; twelve readings.
18. How many additional historical checkpoints should be allowed? Zero on the
    four blocks; at most one on genuinely new data, after the prospective
    campaign starts.
19. Exact evidence required before demo trading: section 23.
20. Exact evidence required before any real-money consideration: a completed
    six-month prospective campaign in which a frozen rule meets its
    preregistered rule; replay parity; an independent audit of the campaign;
    a separately written live-authorisation contract with a hard capital cap;
    none of which exists.
21. Shortest defensible path to demo trading: S0-S4 in about ten to twelve
    weeks, then S5.
22. If 50% of ProjectChimera had to be removed today: Freqtrade and the four
    strategies, modes, consensus, the inference service and registry, MLflow,
    Ray Tune, MTST training, the five retired feature families' active code,
    the Grafana breadth, and the checkpoint-specific research modules from any
    runtime path (kept as history).
23. Five core concepts to keep: preregistration with hashes; sealed,
    fingerprinted, causal data; a single fail-closed risk authority; a
    validated dry-run executor with persisted state; negative results kept
    visible.
24. Highest-value thing currently missing: prospective data collection (and,
    behind it, a statistically powered evaluation design).
25. Is the overall idea worth continuing? Yes, under a finite budget with a
    decisive endpoint; not in its current research direction.

## Appendix B. External sources consulted

- K33 Research figures on BTC average daily volatility (2022: 3.34%, 2024:
  2.80%, 2025: 2.24%) as reported by coindesk.com and other summaries.
- Binance fee schedules as summarised by tradersunion.com and bitdegree.org
  (spot 0.10% base; USD-M futures 0.02% maker / 0.05% taker base).
- Freqtrade documentation, "Leverage" page (futures support, isolated/cross
  margin, funding-rate limitations in backtesting).
- Binance public data S3 listings for `data/spot/monthly/` and
  `data/futures/um/monthly/` (checked 2026-09-03).
- Silantyev, E. (2019), Order flow analysis of cryptocurrency markets, Digital
  Finance 1, 191-218 (abstract as indexed by the publisher and Semantic
  Scholar; full text not read).
- Anastasopoulos, Gradojevic, Liu, Maynard, Tsiakas, Order Flow and
  Cryptocurrency Returns (2025 working paper / 2026 journal version), abstract
  as indexed: order flow has out-of-sample predictive power for
  cryptocurrency returns robust to short-selling constraints and transaction
  costs. Full text not read; cited only as a reason the trade-flow axis is
  not dead, at the right horizon and cost model.
- Standard references for the methodology points: Bailey, Borwein, Lopez de
  Prado and Zhu, The Probability of Backtest Overfitting (J. Computational
  Finance, 2017); Bailey and Lopez de Prado, The Deflated Sharpe Ratio (J.
  Portfolio Management, 2014); Harvey, Liu and Zhu, ... and the Cross-Section
  of Expected Returns (Review of Financial Studies, 2016); Lopez de Prado,
  Advances in Financial Machine Learning (Wiley, 2018); Arnott, Harvey and
  Markowitz, A Backtesting Protocol in the Era of Machine Learning (J.
  Financial Data Science, 2019); Gu, Kelly and Xiu, Empirical Asset Pricing
  via Machine Learning (Review of Financial Studies, 2020), for the role of
  cross-sectional breadth in ML return prediction.
