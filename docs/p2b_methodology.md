# P2b — does causal market structure add information beyond OHLCV14?

Research checkpoint: **P2b**.
Research contract: `btc-usdt-1h-gen1`, sealed at `2025-08-27T23:00:00+00:00`.
Status: the design below was fixed, and this document written, before any P2b
outer-validation number was read.

**This document contains no results, deliberately.** The numbers are a separate
artifact under `artifacts/benchmark/`, indexed in
[`../artifacts/README.md`](../artifacts/README.md). A methodology written beside
the numbers it describes cannot be shown to have preceded them, and a verdict
rule stated after the fact is not a rule.

Feature definitions: [`smc_v1.md`](smc_v1.md). The exact command sequence from a
fresh clone: [`research_reproduction.md`](research_reproduction.md).

---

## 1. The research question

> Is there measurable, causal structure in the price series that the frozen
> fourteen OHLCV columns do not carry?

**Why this is an information question and not an architecture one.** The v4
generation asked whether an MTST Transformer on OHLCV14 is economically viable
and answered no. The obvious next move — a larger model, a different family — is
the move P2a already made and ruled out. Given *exactly the same samples*,
untuned logistic regression, LightGBM and XGBoost spanned a narrow band of mean
outer net return either side of zero, and the strongest of them was a
gradient-boosted tree rather than the Transformer. Whatever is binding, it is
not the model family; see `artifacts/benchmark/btc_p2a_comparison/` for that
evidence.

That leaves the information itself as the standing candidate explanation.
Fourteen scale-free transformations of five numbers per hour may simply not
contain a directional signal that survives fees and slippage at this horizon.
P2b tests that by changing the feature columns and nothing else.

**What an answer looks like.** Not "did the best cell make money". The
comparison that matters is *one model, one fold, one sample set, three column
sets*: the same estimator, fitted on the same rows, with the same label, the
same costs and the same threshold rule, differing only in which columns it was
allowed to see. Any difference between two such cells is attributable to the
information, because nothing else was free to move — and the machinery in
§5 exists to make "nothing else was free to move" a checked property rather
than an intention.

---

## 2. What is held fixed

Every value below is the value P2a ran under, and P2a held each of them at the
value the frozen v4 walk-forward ran under. None of them is a P2b choice, and
none is exposed as a tuning surface on the P2b command line.

| held fixed | value | fixed by |
| --- | --- | --- |
| market | Binance `BTC/USDT`, `1h` | the contract's scope, enforced by `require_scope` |
| label horizon | 6 candles | `target_spec` in the dataset metadata |
| fee | `0.0005` per side | `target_spec` |
| slippage | `0.0005` per side | `target_spec` |
| cost threshold | `0.002` | `target_spec` |
| sequence length | 64 candles | `nn.p2b.SEQ_LEN` |
| folds | 4 nested temporal folds, expanding training window | `nn.p2b.FOLDS` |
| fold fractions | train ≥ 45%, inner 10%, outer 10%, step = outer size | `nn.p2b.MIN_TRAIN_FRAC` / `INNER_VAL_FRAC` / `OUTER_VAL_FRAC` |
| threshold selection | inner-validation block only | `nn.evaluate.select_threshold` |
| threshold grid | 29 points, `0.34` to `0.90` step `0.02` | `nn.benchmark.threshold_grid` |
| threshold objective | maximise net return after round-trip costs, subject to at least 10 realised trades; falls back to the most permissive grid point when no candidate clears the floor | `nn.benchmark.THRESHOLD_OBJECTIVE` |
| trade floor | 10 | `nn.p2b.MIN_TRADES` |
| model configurations | the three predeclared specs, reused verbatim from P2a | `nn.simple_models.SIMPLE_MODELS` |
| scaling | `nn.dataset.StandardScaler` fitted on the fold's training rows only | `nn.train.prepare_research_windows` |
| seed | run seed 42; fold seed `42 + fold` | `nn.p2b.SEED` |
| Styx boundary | `2025-08-27T23:00:00+00:00` | `nn/research_contracts/btc-usdt-1h-gen1.json` |

The three model configurations are conservative library defaults with the
depth and estimator budget stated up front, and are recorded in full in every
artifact's provenance block:

| model | configuration | seed |
| --- | --- | --- |
| `logistic_regression` | multinomial `lbfgs`, `C=1.0`, `max_iter=2000`, `class_weight=None` | takes none |
| `lightgbm` | 200 trees, `learning_rate=0.05`, `num_leaves=31`, no bagging, `deterministic=True`, `n_jobs=1` | `random_state` |
| `xgboost` | 200 trees, `max_depth=6`, `learning_rate=0.05`, `tree_method="hist"`, `device="cpu"`, `n_jobs=1` | `random_state` |

**Fold roles are the walk-forward's fold roles.** Training fits the scaler and
the model. Inner validation selects the decision threshold and nothing else.
Outer validation is frozen evaluation: `nn.train.score_frozen_split` fits
nothing, and `plan.outer` reaches that one call in the whole module.

**The samples are not merely matched, they are the same arrays.** P2b builds no
dataset, window or split of its own. It plans folds with
`nn.walkforward.plan_nested_folds`, materialises train and inner-validation
windows with `nn.train.prepare_research_windows`, and scores the outer block
through `nn.train.score_frozen_split` — the three functions P2a and the MTST
walk-forward go through. The only addition is
`nn.simple_models.flatten_windows`, which turns a `(seq_len, n_features)`
sequence into one row in a declared order: column `j = t * n_features + f`,
oldest timestep first.

---

## 3. What varies

The feature columns, and only those.

There is no flag on `python -m nn.p2b` that can change a model's
hyperparameters, the fold geometry, the costs, the horizon or the boundary.
`--information-set` is the experiment; `--model` selects which of the three
predeclared estimators the arm is measured with.

---

## 4. The three arms

| arm | columns | flattened width | what it is |
| --- | --- | --- | --- |
| `ohlcv14` | 14 | 896 | P2a's control, re-run here rather than copied |
| `smc_v1` | 39 | 2496 | causal market structure alone, no OHLCV14 column present |
| `ohlcv14_plus_smc_v1` | 53 | 3392 | both, concatenated in that order |

Flattened width is `seq_len × columns` with `seq_len = 64`: every estimator
receives the whole 64-candle window, not the latest candle. Handing a tabular
model only the decision row would answer a different question — "is one candle
enough?" — and would make any gap between two arms uninterpretable.

`smc_v1`'s 39 columns are six declared families — `structure` (8), `liquidity`
(6), `breaks` (6), `sweeps` (6), `displacement` (5), `fvg` (8) — each with one
exact formula, one firing rule, and nine predeclared constants, all in
[`smc_v1.md`](smc_v1.md). Two properties of that spec are load-bearing here
rather than cosmetic:

- **Append invariance.** `F(candles[0:N]) == F(candles[0:N+K])[0:N]` for every
  `N` and `K`. A pivot confirmed three candles late becomes observable three
  candles late, and never earlier. Without this, the added columns would carry
  the future and the comparison would measure leakage.
- **No NaN on any row.** Where the underlying state does not exist yet the value
  is a declared default and a paired availability flag is `0`.
  `nn.data_pipeline.build_dataset` drops rows with NaN features, so a single NaN
  column would evaluate the arms on *different rows* and confound the whole
  checkpoint with the sample universe. §5 depends on this.

**`ohlcv14` is re-run, not copied.** P2a's control numbers exist and could have
been quoted. They are not, because the comparison would then be between one
number produced by `nn.benchmark` on a locally built dataset and another
produced by `nn.p2b` on the committed snapshot. Re-running it makes every
comparison in P2b two live numbers from one code path over one file.

---

## 5. The common sample universe

P2b's answer is a *difference* between two numbers, and a difference is only
about the information while everything else is identical. The failure this
guards against is quiet: a new feature family warms up over its own first rows,
those rows are dropped, every later row shifts by the number dropped, fold
boundaries resolved by row index land on different candles, and the two arms are
then measured over different market periods. Nothing raises. The report still
prints four folds. The comparison is simply no longer about the features.

The defence has three layers.

**Guaranteed by construction.** `nn.information_sets.build_information_set_views`
builds one `ResearchData` per arm over one shared row spine. The views hold *the
same array objects* — not equal ones — for dates, targets, future returns,
closes and segment ids; only `features` differs. `nn.dataset.sample_indices` is
a function of the split, the sequence length, the horizon and the segment ids
alone, so the views cannot select different rows. Construction fails closed on
anything that would move one: a spine timestamp with no raw candle, a non-finite
market-structure value, or a processed segment the raw candle history
contradicts.

**Proved per fold.** `AlignedResearchSamples.prove_alignment` re-derives the
sample index for every view, every fold and every block, and asserts the arrays,
the labels on them and the timestamps on them are identical. It returns the
evidence rather than a boolean: the artifact records each block's row range,
sample count, first and last row, first and last timestamp, and a SHA-256 over
the sample-index array. A claim of alignment that nothing recomputes is a
comment, not a guarantee.

**Proved across cells.** One P2b run is one cell — one information set, one
model, four folds — because nine independent single-threaded cells finish far
sooner on four cores than one process would. Splitting the work cannot weaken
the parity claim; it strengthens it. Each cell scores the majority and momentum
rule floors and the CASH and buy-and-hold economic references *on the control
view's columns*, whatever arm it is testing. Those are properties of the rows
and the fold geometry, not of the model or the feature set, so on identical data
they are one value across all nine cells. `nn.p2b_compare` refuses to aggregate
until they agree exactly, along with the contract, the snapshot hashes, the fold
sizes and periods, the label and costs, the threshold rule, the combined
feature-spec hash, and the per-fold sample-index hashes from the alignment
proof. Nothing but the data could make nine separately-run processes agree on
all of that.

That third layer is not redundant. The runner materialises only the views it
needs — the control plus the arm under test — so the `ohlcv14` cell holds a
single view and its own alignment proof has nothing to compare against. What
proves the control cell scored the same rows as the other eight is the
cross-cell comparison of the recorded hashes, and only that.

The rule floors are computed from the control's columns for a second reason
besides the fact that `MomentumBaseline` reads `ema_cross`, which the `smc_v1`
arm does not contain: a floor that changed with the information set would stop
being a floor, and would stop being usable as the data-level proof above.

---

## 6. Fold geometry, and the row the snapshot cannot resolve

The fractions are taken of the **canonical research region** — the 48,217 rows
of the canonical processed dataset that lie before the sealed instant — giving
`min_train = 21,697`, `inner = outer = step = 4,821`, four folds. Fold `k`
trains on rows `[0, 21697 + k·4821)`, selects on the 4,821 rows that follow, and
is reported on the 4,821 after those. Step equals the outer size, so the four
outer blocks are contiguous and partition one stretch of the research region: no
row is reported as the result of two folds.

| fold | train | inner validation | outer validation |
| --- | --- | --- | --- |
| 0 | `[0, 21697)` | `[21697, 26518)` | `[26518, 31339)` |
| 1 | `[0, 26518)` | `[26518, 31339)` | `[31339, 36160)` |
| 2 | `[0, 31339)` | `[31339, 36160)` | `[36160, 40981)` |
| 3 | `[0, 36160)` | `[36160, 40981)` | `[40981, 45802)` |

**The subtlety.** P2b runs from the committed research snapshot under
`data/research/`, which is a *prefix* of the canonical processed dataset,
truncated at row 45,802 — the last row any outer block reaches. That snapshot
therefore **cannot resolve the sealed boundary itself**, because by design it
contains no sealed row to resolve against: the boundary is defined as the first
row at or after `2025-08-27T23:00:00+00:00`, and every row present is strictly
before it.

Resolving the boundary from the data on disk would produce 45,802 — the file's
own length — and taking fractions of *that* would silently plan four different
folds over four different market periods while raising nothing. So the geometry
is not resolved from the snapshot at all. `research_rows = 48217` is read from
the snapshot manifest's record of the canonical research region, the plan is
expressed in the canonical dataset's row numbers, and then two assertions run
**before anything is fitted**:

- the plan's last outer row must fall inside the rows actually present, or the
  run aborts telling the operator to re-export the snapshot rather than shrink
  the folds to fit;
- that same row must equal the manifest's declared `max_outer_end_row`, or the
  run aborts on the grounds that the snapshot and the runner disagree about
  which rows research may reach and neither is permitted to guess.

The runner additionally refuses to start unless the manifest declares
`contains_styx: false` and `styx_rows_exported: 0`, refuses to run under a
contract whose hash differs from the one the snapshot was exported under, and
refuses to *write* a prediction whose row index reaches the boundary. The
snapshot's own integrity — hashes, row counts, spans, and a direct timestamp
comparison against the sealed instant — is
`tools.verify_research_snapshot`'s job and is not duplicated in the runner.

Because the plan is expressed in canonical row numbers, the four outer blocks
are the same rows the v4 walk-forward and P2a reported on. That is what makes
the control arm comparable to P2a's control, and it is also the reason §10
exists.

---

## 7. The statistical unit

**The statistical unit is one temporal outer period. There are four of them.**

There is no seed dimension in P2b and there is not meant to be one. The three
estimators are deterministic given their inputs; `logistic_regression` takes no
seed argument at all. P2a ran five seeds, and its own artifact index records the
consequence: the deterministic configurations produced zero seed dispersion, so
"15 of 20 run-folds" was never twenty observations — it was four observations
reported five times, and logistic regression's five seeds were five identical
copies of one piece of evidence.

Everything P2b reports is therefore `n of 4`. Nothing in the output multiplies
four periods by a replication count, and `nn.p2b_compare` emits no other
denominator.

Four is a small number, and calling it the unit is what keeps it visible.
Aggregates over four folds are reported with min, median and max beside the
mean, because with four observations the range says more than the standard
deviation does.

---

## 8. Reproducibility: the native thread count is part of the experiment

`LogisticRegression`'s `lbfgs` solver reduces through multi-threaded BLAS, and a
floating-point reduction whose order depends on the thread count is not
reproducible. Measured on this dataset, fold 0's outer probabilities move by up
to **0.025** between a one-thread and a four-thread fit. That is not a rounding
difference: it is enough to select a different point off the 0.02-spaced
threshold grid, and therefore to report a materially different net return for
the same data, the same code and the same seed.

So `nn.p2b` pins native pools to `FIT_THREADS = 1` with `threadpoolctl`, once,
around the whole cell. LightGBM and XGBoost were already pinned at `n_jobs=1` in
their predeclared configurations for the same reason; this closes the remaining
hole.

**Where the pin is entered matters, and the first version got it wrong.**
`threadpool_limits` can only limit pools that are already loaded when it is
entered, and `SimpleModelSpec.build` imports its library lazily. Entering the
limit and *then* fitting therefore left scipy's OpenBLAS and scikit-learn's
libgomp at four threads on fold 0 — the import had not happened yet — and pinned
on folds 1 to 3, where it had. Fold 0 ran under a thread configuration no other
fold in the run used, and the pool snapshot written into the artifact named
neither library. An adversarial review caught it; measured on this build the
probabilities were bit-identical either way, so it was a broken guard rather
than a broken number, but a guard that cannot be checked is not one.

The estimator's library is now loaded once before the pin is entered, so all
four native pools are pinned for every fold. The observed pool state — every
pool's API, thread count and library version — is recorded in each artifact's
provenance, so the claim can be checked against the run rather than trusted.

This is a correctness setting, not a performance one, and it is why the nine
cells are run as nine processes rather than one threaded one: parallelism across
cells is free, parallelism inside a fit is not.

---

## 9. The predeclared verdict rule

Fixed before any number was read, and recorded in
`nn.p2b_compare.VERDICTS` so the output cannot pick its own bar. Counted on net
return after costs, per model, per arm, against the `ohlcv14` control on the
same rows:

| folds improved | reading |
| --- | --- |
| 4 of 4 | consistent improvement — evidence worth continuing on |
| 3 of 4 | evidence worth continuing on |
| 2 of 4 | regime-dependent, inconclusive |
| 1 of 4 | weak evidence against |
| 0 of 4 | negative evidence |

What follows from each reading is written down in advance too, in
[`research_roadmap.md`](research_roadmap.md): consistent improvement continues
with market-structure diagnostics and does **not** tune `smc_v1`'s predeclared
constants; a two-of-four split is described and does **not** get a fitted regime
filter; no improvement freezes the negative result and moves to the next causal
family. A negative result is a result, and the reusable machinery — the
alignment layer, the parity proof, the recomputation path — is the durable
output either way.

---

## 10. Adaptive status

**P2b was designed after P2a's outer results had been seen.** Its four outer
blocks are the same four the v4 walk-forward and P2a already reported on, and
they have now been read repeatedly. That makes P2b's outer evidence **adaptive
research evidence, not a pristine out-of-sample test**, and every artifact it
writes says so in its own payload.

This is stated rather than worked around because the alternative — carving a
fresh holdout out of the research region for each new checkpoint — would spend
the sealed block a slice at a time. The sealed block is what pristine evidence
is for, and it stays shut: it is not planned over, not fitted on, not selected
on and not scored, and every artifact records `sealed_test: false`.

---

## 11. What P2b deliberately does not do

- **No hyperparameter tuning.** No Optuna, no Ray Tune, no grid or random
  search, no early stopping, no per-fold variation beyond the seed, and no flag
  that can change a model's configuration.
- **No threshold selection on outer data.** The threshold is chosen on the inner
  block and applied to the outer block unchanged. The inner selection score is
  recorded outside the `outer_validation` block precisely so it can never be
  aggregated as a result.
- **No search over the feature constants.** `smc_v1`'s nine constants were
  predeclared as round, interpretable numbers and are frozen in the spec object
  whose hash every artifact records. Searching them against these four outer
  blocks would convert a result into a fit.
- **No MTST.** P2a already established that the model family is not what is
  binding; re-running a Transformer here would add cost and a second sample
  construction path without changing the question.
- **No order blocks.** Every definition in common use needs either a
  discretionary choice of which candle in a cluster counts, or knowledge of what
  price did afterwards. Neither survives the causality rule, so they are
  deferred until a rigorous causal definition exists rather than approximated
  ([`smc_v1.md`](smc_v1.md) §0).
- **No feature selection of any kind**, inside a fold or across folds. Every arm
  receives all of its columns, and `feature_selection: none` is written into
  every artifact.

Two post-hoc analyses sit beside the canonical result and label themselves as
post-hoc in their own output: `nn.p2b_ablation` (what each market-structure
family contributed *given the others were present*, over the six families
declared before any result existed) and `nn.p2b_regimes` (what the four outer
periods actually were). Neither selects anything, neither is part of the
verdict, and no regime filter is fitted from either.

---

## 12. Where the evidence lands

Nine cells at `artifacts/benchmark/btc_p2b_<information_set>_<model>/`, each
holding `p2b.json`, `p2b.md` and `outer_predictions.parquet`, and one join at
`artifacts/benchmark/btc_p2b_comparison/`.

The join is not a concatenation. Before it reports anything, `nn.p2b_compare`
proves the cells scored the same rows (§5) and then **rebuilds every reported
trading and classification number from the persisted per-sample predictions**,
through `nn.evaluate`, and refuses to continue on any disagreement. A report
that contradicts its own predictions is a report about nothing, and there is no
way to notice that by reading it. The annualised Sharpe and the candle-level
drawdown are the declared exceptions: they need the candle price path, which the
prediction file does not carry, and they are listed as not recomputed rather
than silently skipped.

Which of those directories is authoritative for which question is recorded in
[`../artifacts/README.md`](../artifacts/README.md), not inferred from directory
names.
