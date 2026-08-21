# Reproducing the research from a fresh clone

Every command on this page runs on one machine with a network connection long
enough to install packages, and nothing else. **No VPS, no exchange
credentials, no private dataset, and no access to the sealed block.** The
research checkpoint P2b runs entirely from data committed to this repository.

What that costs: it is not the whole pipeline. `nn.train`, `nn.walkforward` and
the dry-run stack still need a locally built dataset from
`make backfill` + `make features`, and the v4 MTST generation cannot be
re-derived here at all — its source dataset is not committed and no fingerprint
can be computed for it after the fact ([`../artifacts/README.md`](../artifacts/README.md)).
What *is* reproducible from a clone is P2b: the nine benchmark cells, their
join, and every test that guards them.

Design and rationale: [`p2b_methodology.md`](p2b_methodology.md). Feature
definitions: [`smc_v1.md`](smc_v1.md).

---

## 0. Before anything: run from the repository root

`nn.p2b`, `nn.p2b_compare`, `nn.p2b_ablation` and `nn.p2b_regimes` default their
`--manifest` and their `--out` to repository-relative paths. Run them from the
root of the checkout, or pass every path explicitly. `make` targets do this for
you.

---

## 1. Clone and install

```bash
git clone https://github.com/tyrannya/projectchimera.git
cd projectchimera

python --version          # 3.11 or newer; pyproject requires it
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install -e ".[benchmark,ml,dev]"
```

Three extras, and each is load-bearing:

| extra | why it is needed |
| --- | --- |
| `benchmark` | scikit-learn, LightGBM and XGBoost — the three estimators. It also pulls `threadpoolctl`, which `nn.p2b` imports directly to pin native thread pools (§8 of the methodology). |
| `ml` | **torch.** `nn.p2b` imports `nn.train` for the windowing, scaling and frozen-scoring functions, and `nn.train` imports torch at module scope for the MTST definition. No Transformer is fitted in P2b, but the import still has to resolve. |
| `dev` | pytest, for §6. |

`make setup` installs `.[all]` and the pre-commit hooks instead, which is always
correct if you are unsure.

**If the torch install fails at `download.pytorch.org`.** Some networks block
that host by egress policy. Plain PyPI carries the CPU wheels this repository
needs, so install torch on its own first and then the rest:

```bash
pip install torch
pip install -e ".[benchmark,ml,dev]"
```

Nothing in P2b runs on a GPU — the estimator configurations are pinned to
`device="cpu"` and `n_jobs=1` — so a CPU-only torch wheel is not a compromise
here.

---

## 2. Verify the research snapshot

```bash
make verify-research-snapshot          # python -m tools.verify_research_snapshot
```

Expected tail:

```
23 checks passed; no row at or after the sealed instant.
```

It recomputes every claim rather than reading it, and fails closed on the first
disagreement, printing `research snapshot REJECTED - <check>: <detail>` and
exiting non-zero. The 23 checks cover:

- the manifest is readable, declares schema `chimera.research-snapshot/1`, and
  carries all 31 required keys with the right types;
- the named research contract is committed here, and its recomputed SHA-256
  matches what the manifest records — a contract file edited after export
  changes its semantic identity and is rejected, not tolerated;
- the SHA-256 of all three referenced files;
- **the seal, against the data**: every timestamp in both Parquet files is
  compared directly with `2025-08-27T23:00:00+00:00`. This check runs *before*
  any digest or row-count check, so a snapshot holding a sealed row is rejected
  for holding one rather than for whatever hash that row happened to disturb;
- row counts, the `[0, rows)` row range, and the first/last timestamp span of
  both files against the manifest;
- the semantic research-input digests of the raw candles and of the processed
  prefix — recompression, a different Parquet writer or a moved file leave these
  alone, while one changed price does not;
- that the rows present cover the last row the outer folds reach.

**What a failure means.** Not "re-export and continue". The tool exists because
a fresh clone inherits no proof — it has three files and a manifest asserting
things about them, and every number a run produces comes from the files, not
from the manifest's description of them. A rejection means the files and their
claims have drifted apart
(a rebuilt Parquet, a hand-edited manifest, a partial checkout, a merge that
resurrected an older file), and any number produced from them would be a number
about an unknown input. A seal-check failure is more serious still: it means the
committed data reaches into the sealed block, and no research may be run against
it. Note that the rejection counts offending rows and never prints them — a
verifier that displayed the offending candle would leak the rows it exists to
keep sealed.

`make check` runs this verifier as part of the acceptance gate, before pytest.

---

## 3. What the committed research data is — and what it is not

Three files plus a manifest under `data/research/`:

| file | rows | span |
| --- | --- | --- |
| `btc_usdt_1h_gen1_raw_pre_styx.parquet` | 49,551 | `2020-01-01T00:00` .. `2025-08-27T22:00` |
| `btc_usdt_1h_gen1_ohlcv14_outer_coverage.parquet` | 45,802 | `2020-01-04T06:00` .. `2025-05-19T07:00` |
| `…_outer_coverage.parquet.meta.json` | — | feature spec, target spec, class balance, validation report |
| `btc_usdt_1h_gen1_snapshot_manifest.json` | — | hashes, spans, and the canonical reference |

The raw file is the unprocessed OHLCV history the market-structure engine is run
over. The processed file is the OHLCV14 research spine: 14 features, `target`,
`future_return`, `segment_id`, already past indicator warm-up and horizon
trimming. It is the row space the fold plan is expressed in.

**What it deliberately is not:**

- **It holds no sealed row.** Not one. The raw file stops at the last candle
  strictly before the Styx instant; the processed file stops far earlier still.
  `contains_styx: false` and `styx_rows_exported: 0` are in the manifest, and §2
  checks both against the data rather than taking the manifest's word.
- **The processed file is a prefix, not the dataset.** It is
  `rows [0, 45802)` of the canonical processed dataset, truncated at the last
  row any outer block reaches. The canonical research region is 48,217 rows, so
  2,415 research-visible rows exist that this snapshot does not carry. They are
  omitted because no P2b fold touches them.
- **It therefore cannot resolve the sealed boundary itself.** Resolving the
  boundary from these rows would return the file's own length, 45,802, and
  planning fractions of *that* would silently produce four different folds over
  four different market periods. `research_rows = 48217` comes from the
  manifest's `canonical_reference` block, the plan is expressed in canonical row
  numbers, and `nn.p2b` asserts the plan's last row both fits inside the
  snapshot and equals the manifest's declared `max_outer_end_row` before it fits
  anything. See [`p2b_methodology.md`](p2b_methodology.md) §6.
- **It cannot re-derive v4 or P2a.** Those ran on the full canonical dataset,
  which is not committed. The snapshot's `canonical_reference` records the v4
  artifact and research-input hash it was cut against, but the verifier cannot
  re-derive them from a research clone and does not pretend to — it checks that
  block for internal consistency only and labels it as the exporter's testimony.

---

## 4. Run P2b

### One cell

```bash
make p2b-cell SET=smc_v1 MODEL=xgboost
```

which is exactly:

```bash
python -m nn.p2b --checkpoint P2b --information-set smc_v1 --model xgboost \
    --out artifacts/benchmark/btc_p2b_smc_v1_xgboost
```

`--checkpoint` is required and is not inferred: `ohlcv14` is the control of both
P2b and P2c, so the arms cannot say which research question a cell is answering.
A cell that will not say is refused before it answers one, and so is an arm that
does not belong to the checkpoint named — `--checkpoint P2c --information-set
smc_v1` is a mislabelled cell and is unrunnable rather than plausible. P2c's own
cells are `make p2c-cell SET=chart_structure_v1 MODEL=xgboost`.

`SET` is one of `ohlcv14`, `smc_v1`, `ohlcv14_plus_smc_v1` for P2b, and
`ohlcv14`, `chart_structure_v1`, `ohlcv14_plus_chart_structure_v1` for P2c.
`MODEL` is one of `logistic_regression`, `lightgbm`, `xgboost`. The remaining
flags — `--manifest`, `--research-contract`, `--seed`, `--seq-len`,
`--min-trades` — default to the predeclared values and exist to be *recorded*,
not to be swept. There is no flag that can change a model's hyperparameters, the
fold geometry, the costs or the boundary.

**The runner verifies the research snapshot before it fits anything.** All 23
checks of `tools.verify_research_snapshot` run inside `load_snapshot`, not
beside it in a `make` target, so `python -m nn.p2b` is exactly as safe as
`make p2b-btc`. A snapshot whose manifest has stopped describing its files
produces zero model fits and a named rejection.

Each cell writes `p2b.json`, `p2b.md` and `outer_predictions.parquet`
(18,939 outer samples, four folds) into its `--out` directory.

### All nine, in sequence

```bash
make p2b-btc
```

### All nine, three at a time

The nine cells are independent and every estimator is pinned to one native
thread inside the runner, so running several at once is both reproducible and
roughly three times faster on four cores. It cannot change a single number:
nothing is shared between cells except the input files.

```bash
mkdir -p artifacts/p2b_logs
for s in ohlcv14 smc_v1 ohlcv14_plus_smc_v1; do
  for m in logistic_regression lightgbm xgboost; do echo "$s $m"; done
done | xargs -P 3 -n 2 sh -c '
  python -m nn.p2b --checkpoint P2b --information-set "$1" --model "$2" \
      --out "artifacts/benchmark/btc_p2b_$1_$2" \
      > "artifacts/p2b_logs/$1_$2.log" 2>&1' _
```

Logs go outside `artifacts/benchmark/` on purpose: that directory is the evidence
index's row space, and a stray file in it is a directory a reader has to rule
out.

**Do not edit repository source while a batch is running.** Each cell records a
SHA-256 over every repository module its process imported, and §5 refuses to
join cells whose digests differ. That is the point — nine cells built by nine
revisions of the runner would otherwise join without a word — but it means a
mid-batch edit costs you the cells that ran before it.

Leave one core free: three concurrent fits on a four-core machine keeps the
machine responsive and avoids the memory spike of a fourth.

### How long, and how much memory

Indicative wall clock on four cores. These are order-of-magnitude guidance for
planning a run, not a benchmark:

| cell | per fold | per cell (4 folds) |
| --- | --- | --- |
| `ohlcv14 × *` (896 inputs) | under a minute to a few minutes | a few minutes |
| `smc_v1 × *` (2,496 inputs) | a few minutes | ~15–30 minutes |
| `ohlcv14_plus_smc_v1 × xgboost` (3,392 inputs) | **~15 minutes** | **~1 hour** |

The widest XGBoost cell is the long pole; the two other widest cells are the
next longest. Expect the full grid to take roughly **2–3 hours** at three-way
parallelism, and appreciably longer in sequence.

Memory scales with the flattened matrix. The largest training block is fold 3 of
the combined arm: 35,157 samples × 64 timesteps × 53 features in float32 ≈ 477
MB for the windows, before the estimator builds its own internal copy. Budget
roughly 1.5–2 GB per concurrent cell, so **8 GB of RAM is comfortable for three
at a time**; the `ohlcv14` cells need a fraction of that.

---

## 5. Join the cells

```bash
make p2b-compare
```

which is:

```bash
python -m nn.p2b_compare \
    --runs artifacts/benchmark/btc_p2b_ohlcv14_logistic_regression \
           artifacts/benchmark/btc_p2b_ohlcv14_lightgbm \
           artifacts/benchmark/btc_p2b_ohlcv14_xgboost \
           artifacts/benchmark/btc_p2b_smc_v1_logistic_regression \
           artifacts/benchmark/btc_p2b_smc_v1_lightgbm \
           artifacts/benchmark/btc_p2b_smc_v1_xgboost \
           artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_logistic_regression \
           artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_lightgbm \
           artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_xgboost \
    --out artifacts/benchmark/btc_p2b_comparison
```

It writes `p2b_comparison.json` and `p2b_comparison.md`. It writes **nothing**
if the cells cannot be compared, if any cell's persisted predictions disagree
with the committed snapshot, or if any cell's report disagrees with its own
predictions: each raises and exits non-zero. That is the intended behaviour — an
aggregate over cells that did not score the same rows, or did not run the same
code, would be a comparison of sample universes wearing a comparison of
information sets.

### P2c, the other checkpoint

Identical in every respect except the columns and the checkpoint name:

```bash
make p2c-btc        # nine cells: ohlcv14, chart_structure_v1, ohlcv14_plus_chart_structure_v1
make p2c-compare    # -> artifacts/benchmark/btc_p2c_comparison
```

Its cells land in `btc_p2c_<information_set>_<model>/` and carry the same file
names — `p2b.json`, `p2b.md`, `outer_predictions.parquet`. The names are the
*runner's*, not the checkpoint's; every one of those files states its own
checkpoint and research question in its first two fields, and
`nn.p2b_compare` refuses to join cells that disagree about either. A glob wide
enough to catch both checkpoints' cells fails closed rather than averaging
twelve arms of two feature families into one table.

### Optional, post-hoc

Both label themselves as post-hoc in their own output, and neither is part of
the predeclared verdict.

The leave-one-family-out ablation needs its six extra cells run first — there is
no Make target that produces them, because they are not canonical P2b evidence:

```bash
MODEL=xgboost
for f in structure liquidity breaks sweeps displacement fvg; do
  python -m nn.p2b --checkpoint P2b --information-set "ohlcv14_plus_smc_v1_minus_$f" \
      --model "$MODEL" \
      --out "artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_minus_${f}_${MODEL}"
done
make p2b-ablation MODEL=xgboost
```

Six more wide cells is roughly another six hours of XGBoost time — about two at
three-way parallelism. Budget accordingly, or run the ablation against
`lightgbm`.

```bash
make p2b-regimes        # descriptive: what the four outer periods were
```

---

## 6. Run the tests

```bash
pytest tests/test_smc_features.py tests/test_research_snapshot.py tests/test_p2b.py
```

118 tests, and they need no artifacts and no run — they build their own
synthetic fixtures and read the committed snapshot.

| file | tests | what it guards |
| --- | --- | --- |
| `test_smc_features.py` | 42 | the 39 feature definitions, the six families, the nine constants, the no-NaN guarantee, and append invariance at several `(N, K)` on synthetic fixtures *and* on real pre-Styx BTC candles. Written independently of `nn/smc.py` against [`smc_v1.md`](smc_v1.md). |
| `test_research_snapshot.py` | 31 | the verifier itself, by corrupting a copied snapshot 31 ways — a flipped byte, an edited row count, a rewritten manifest claim, a resurrected file — and requiring the named check to fail each time. A verifier that has never been shown to reject is decoration. |
| `test_p2b.py` | 45 | the information sets, the alignment layer, the alignment proof, and — the assertion that matters most — that `nn.p2b_compare.recompute_cell` *reports* a fold whose stored net return has been edited away from what its own predictions imply. |

`make test` runs the whole suite instead. Do not reach for it while a benchmark
is running: it is long, and it is not what tells you a P2b run is sound.

---

## 7. What to check before trusting the output

Four things. All four are in
`artifacts/benchmark/btc_p2b_comparison/p2b_comparison.json`, and the last three
are summarised at the top of `p2b_comparison.md`.

Every one of them also *stops the run* when it fails, so a comparison file that
exists has already passed all four. Read them anyway on any artifact whose
production you did not watch: recording the evidence instead of asserting the
conclusion is what makes that possible.

### 7.1 The alignment proof

Per fold, per block, each cell recomputed which rows it would score and asserted
every view agreed. Read the evidence:

```bash
jq '.alignment.folds[] | {fold, outer: .outer_validation}' \
   artifacts/benchmark/btc_p2b_comparison/p2b_comparison.json
```

Expect four folds, contiguous outer row ranges `[26518,31339)`, `[31339,36160)`,
`[36160,40981)`, `[40981,45802)`, non-zero `samples`, and a
`sample_index_sha256` on every block. Outer periods should read 2023-03 →
2023-09, 2023-09 → 2024-04, 2024-04 → 2024-10, 2024-10 → 2025-05. An empty
block, an outer range that overlaps its neighbour, or a missing hash means the
plan did not survive the data.

Then the half of the proof that is about the columns rather than the rows:

```bash
jq '{checked: .alignment.checked, per_view: .alignment.checked_per_view,
     join: .alignment.join_evidence}' \
   artifacts/benchmark/btc_p2b_comparison/p2b_comparison.json
```

`join_evidence.matches` must be `true` and **both**
`matches_under_plus_one_shift` and `matches_under_minus_one_shift` must be
`false`. That pair is the point: it says the join was checked by something that
would have failed if a market-structure value were sitting one candle off — a
one-row shift is a one-candle look-ahead on all 39 columns at once, and nothing
else in the proof can see it. `raw_candles_seen_by_the_engine` should read
**47,136** — one past the raw row behind the spine's last row — and not the raw
file's 49,551: the candle history is truncated before the engine runs, so it
cannot reach past the last row any fold scores. `raw_rows_first` should be `78`,
the OHLCV14 warm-up the spine already discarded and the market-structure engine
still gets to build state over.

Note that `.alignment` is one cell's block, and `checked` is shorter for a
single-view cell — it says so in words rather than reporting cross-set
comparisons it never made.

### 7.2 The cross-cell parity block

Nine independently-run processes agreeing on every property of the *rows* is
what proves they scored the same rows — nothing but the data could make them
agree.

```bash
jq '.parity' artifacts/benchmark/btc_p2b_comparison/p2b_comparison.json
```

Expect nine entries under `cells`, the nine-item `identical_across_cells` list
(contract and hash, snapshot identity, fold sizes and periods, per-fold
sample-index hashes, label horizon and costs, threshold rule, feature-spec hash,
majority and momentum baseline reports, CASH and buy-and-hold references), and
the conclusion that a difference between two cells can only be the information
set or the model.

`parity.code.source_digest` is the tenth thing they must agree on, and it is
about the runner rather than the data: a SHA-256 over every repository module
each process imported. `parity.code.revisions` may legitimately hold more than
one git revision — a documentation commit between two cells moves `HEAD`
without changing a line any cell executes — which is exactly why the digest, not
the revision, is what the comparison enforces.

If you ran only some of the nine, `cells` will show only those. A comparison of
three cells is a valid comparison of three cells; it is not the P2b result.

### 7.3 The snapshot anchoring

Everything in §7.4 rebuilds a cell's numbers from the cell's own persisted
columns, so a cell whose labels were mis-joined would reproduce its own wrong
numbers exactly and report nothing. This is the check against data the run did
not produce:

```bash
jq '.snapshot_anchoring' artifacts/benchmark/btc_p2b_comparison/p2b_comparison.json
```

Expect `cells_checked: 9`, `problems: 0`, and `rows_checked: 170451`
(9 × 18,939). Every scored row's timestamp, label and realised return was
compared against `data/research/btc_usdt_1h_gen1_ohlcv14_outer_coverage.parquet`
at the row index the cell recorded. A low `rows_checked` means you compared
fewer cells than you think you did.

### 7.3.1 The planned-row binding

`snapshot_anchoring` asks whether each persisted row agrees with the snapshot at
the index the file claims. It cannot see a *wrong selection*: a scorer that
persisted a different set of rows — consistently, each row's own timestamp,
label and return copied correctly — passes it without a mark. This is the check
that closes that:

```bash
jq '.planned_row_alignment' artifacts/benchmark/btc_p2b_comparison/p2b_comparison.json
```

Expect `cells_checked: 9`, `folds_checked: 36`, `rows_checked: 170451`, and
**every counter zero**: `missing_folds`, `unplanned_folds`,
`non_integer_row_index`, `duplicate_rows`, `unsorted_rows`, `count_mismatches`,
`sample_index_hash_mismatches`, `first_last_mismatches`, `cross_fold_rows`,
`snapshot_value_mismatches`. The hash counter is the load-bearing one: it
compares a SHA-256 over the persisted `row_index` bytes against the digest
`prove_alignment` recorded from the fold geometry before anything was fitted.

### 7.4 The independent-recompute mismatch count

```bash
jq '.independent_recompute | {cells_checked, folds_checked, mismatches}' \
   artifacts/benchmark/btc_p2b_comparison/p2b_comparison.json
```

Expect `cells_checked: 9`, `folds_checked: 36`, `mismatches: 0`.

**`mismatches` is always 0 in a file that exists** — a non-zero count raises
before anything is written. The field is there so a reader can see that the
check ran, and over how many cell-folds. What matters is that `folds_checked` is
36 and not, say, 12: a recomputation over a third of the cells proves a third of
the report.

Two numbers are deliberately *not* recomputed, and say so in `not_recomputed`:
the annualised Sharpe and the candle-level maximum drawdown need the candle
price path, which the prediction file does not carry.

### And the labels on the evidence, if you are auditing rather than running

```bash
jq '{sealed_test, unit: .statistical_unit, dependence: .folds_are_not_independent,
     adaptive: .adaptive_status, contains_styx: .snapshot.contains_styx}' \
   artifacts/benchmark/btc_p2b_comparison/p2b_comparison.json
```

`sealed_test` must be `false` and `contains_styx` must be `false`. The other
three are the honest labels on the evidence: four temporal periods with no seed
replication; four periods that are not four independent draws, because each
fold's inner-validation block is the previous fold's outer block; and outer
blocks designed after earlier results on those same blocks had been seen. See
[`p2b_methodology.md`](p2b_methodology.md) §7 and §10.

---

## 8. Freezing what you produced

```bash
python -m tools.freeze_evidence --out artifacts/btc_p2b_SHA256SUMS.txt \
    artifacts/benchmark/btc_p2b_{ohlcv14,smc_v1,ohlcv14_plus_smc_v1}_*
make freeze-evidence MANIFEST=artifacts/btc_p2b_SHA256SUMS.txt
```

The first command records a SHA-256 for every file it is given; the second
(`tools.freeze_evidence --verify`) re-hashes them and reports anything that
moved or vanished, with no exemptions.

**A manifest covers primary evidence only.** A cell and its predictions cannot
be rebuilt without re-fitting, so a byte change in one is a change in the
research result. A comparison is *derived* — regenerated from those cells
whenever the aggregator improves — and freezing refuses it rather than promising
something the workflow is designed to break:

```bash
$ python -m tools.freeze_evidence --out artifacts/demo.txt artifacts/benchmark/btc_p2b_*
refusing to freeze derived evidence: artifacts/benchmark/btc_p2b_comparison, ...
```

Note the glob in the working command: it names the nine cells and excludes the
comparison, the ablation table and the regime description, each of which
declares `evidence_class: "derived"` in its own JSON. What pins those instead is
`tests/test_p2b_evidence.py`, which asserts their fold counts, verdicts and
integrity counters directly.

Freezing refuses to overwrite an existing manifest, on purpose: a manifest is
the repository's own statement about what a past run produced, and regenerating
it in place is how a result quietly becomes whatever the code does today. A
checkpoint that needs new numbers gets a new artifact directory and a new
manifest under a new name, and which directory is authoritative for which
question is recorded in [`../artifacts/README.md`](../artifacts/README.md)
rather than inferred from directory names.

### Indexing it

`artifacts/README.md` is the index of which artifact is authoritative for which
question. The index's first line says everything under `artifacts/` is a record
of a run that has already happened, and `tests/test_reporting_integrity.py`
enforces that: a CURRENT aggregate must be backed by committed source runs, and
`test_the_index_names_exactly_one_current_generation_per_research_question`
compares the set of questions carrying a CURRENT generation against a hard-coded
mapping. **Declaring where evidence will land is not recording that it did**, so
the rows go in when the directories do, in the same commit as the evidence.

P2b's research question is `btc_p2b_information_set_benchmark` and P2c's is
`btc_p2c_information_set_benchmark`. They are the *third* and *fourth* questions
in the index; v4's and P2a's rows keep their own CURRENT status, because CURRENT
is unique per question and neither information-set checkpoint supersedes anything
above it. Per checkpoint that is ten rows: one per cell,
`produced by nn.p2b`, `source runs: itself (<set> x <model>)`,
`source runs present: n/a`; and one for the comparison,
`produced by nn.p2b_compare`, naming the nine cells,
`source runs present: yes`. All `metric semantics: current`. The same commit has
to add the question to the test's expected mapping and to its `CURRENT_*` path
sets — the index and the guard move together, or the guard is not guarding this
question.
