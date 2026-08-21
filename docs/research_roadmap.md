# Research roadmap

What has been asked, what was answered, and what the next question is.

This file exists because a research programme that cannot say what it already
ruled out will keep re-asking the same question with a new library. Each entry
names a checkpoint, the question it was designed to answer, and the answer —
including when the answer is "no".

Rules that apply to every row below:

- **The Styx boundary never moves.** `2025-08-27T23:00:00+00:00` for
  `btc-usdt-1h-gen1`. A new research generation is a new contract file
  describing a new *question*; it may not manufacture a new holdout.
- **A negative result is a result.** Two of the four checkpoints so far are
  negative, and they are the reason the later ones ask what they ask.
- **Adaptive evidence is labelled.** Every checkpoint after v4 reuses outer
  blocks whose results have already been seen, which makes them adaptive
  research evidence rather than pristine out-of-sample tests. The sealed block
  is what pristine evidence is for, and it stays shut.

---

## Answered

### v4 — is an MTST Transformer on OHLCV14 economically viable?

**No.** Across five seeds and four outer folds, MTST beat the majority and
momentum floors in all 20 run-folds, beat CASH in 9 of 20, beat buy-and-hold in
0 of 20, and had mean outer net return −0.019272. Evidence:
`artifacts/diagnostics/btc_regimes_v4/`.

### P2a — is the failure the model family, or the information?

**The model family is not what is binding.** Given exactly the same samples —
MTST's own windows, flattened — untuned logistic regression, LightGBM and
XGBoost span roughly −5.6% to +0.7% mean outer net return. XGBoost became the
strongest OHLCV14 baseline and was positive in about half the temporal periods.
Evidence: `artifacts/benchmark/btc_p2a_comparison/`.

Two things about P2a matter more than its numbers:

- There are only **four unique temporal periods**. P2a ran five seeds, but
  logistic regression takes no seed at all and the tree models are deterministic
  given theirs, so "15 of 20" was never 20 observations.
- Buy-and-hold beat everything over these windows, at full exposure. That is a
  reference, not a competitor, and it is not a reason to select anything.

That is what turned the next question into an information question.

---

### P2b — does causal market structure add information beyond OHLCV14?

**No.** Three untuned models times three information sets (`ohlcv14`, `smc_v1`,
`ohlcv14_plus_smc_v1`) over four temporal outer folds on one proven-identical
sample universe. Not one of the six model-arm comparisons improved on the
OHLCV14 control in more than **two of four** folds, against a bar of three fixed
before any number was read. Evidence:
[`artifacts/benchmark/btc_p2b_comparison/`](../artifacts/benchmark/btc_p2b_comparison/).

| model | `smc_v1` − control | `ohlcv14_plus_smc_v1` − control |
| --- | --- | --- |
| logistic regression | 0 of 4 | 2 of 4 |
| LightGBM | 2 of 4 | 1 of 4 |
| XGBoost | 1 of 4 | **1 of 4** |

Three things about this result matter more than the numbers:

- **The mean would have lied.** `lightgbm x smc_v1` and `xgboost x smc_v1` both
  have a *positive* mean delta while improving only two and one of four folds.
  Pooling four periods would have reported them as gains. The count-the-folds
  rule was predeclared for exactly this and it fired on real data.
- **The control is not the weak link.** XGBoost on `ohlcv14` reproduced P2a's
  frozen seed-42 evidence bit-for-bit — four fold returns, four thresholds —
  from a different code path reading the committed snapshot rather than the
  canonical dataset.
- **The periods were all up.** All four outer blocks had positive total return
  (+18.9%, +162.0%, +4.5%, +43.4%) at low directionality (efficiency ratio
  0.0030-0.0552): choppy uptrends. Buy-and-hold beats every arm over these
  windows, at full exposure, and is a reference rather than a competitor.

What survives is the machinery, which was always the more durable half: the
alignment layer, the parity proof, the snapshot anchoring, the recomputation
path and the source-digest identity are reusable by any later information set.

**Do not respond to this by tuning `smc_v1`'s constants.** They were predeclared,
and searching them against these four outer blocks would convert a negative
result into a fitted one. `docs/smc_v1.md` §9 records the defects worth fixing in
a future `smc_v2`; none of them is a threshold to search.

### P2c — does causal classical chart structure add information?

**No, and more cleanly than P2b.** Specified in
[`chart_structure_v1.md`](chart_structure_v1.md), implemented in
`nn/chart_structure.py`, same three-arm design against `ohlcv14`. Evidence:
[`artifacts/benchmark/btc_p2c_comparison/`](../artifacts/benchmark/btc_p2c_comparison/).

| model | `chart_structure_v1` − control | `ohlcv14_plus_chart_structure_v1` − control |
| --- | --- | --- |
| logistic regression | 1 of 4 | **0 of 4** |
| LightGBM | 1 of 4 | 2 of 4 |
| XGBoost | 1 of 4 | **1 of 4** |

None reached the 3-of-4 bar, and unlike P2b **every mean delta is negative** —
there is no arm where pooling the four periods would even have flattered it.

**Exploratory and adaptive, and that matters more here than anywhere.** By the
time P2c ran, these same four outer blocks had been read by v4, P2a, P2b, the
P2b ablation and the P2b regime description. A positive P2c would have needed
heavy discounting; a negative one needs none, which is the asymmetry that makes
negative results the cheap ones to trust.

The control was re-run for P2c rather than copied from P2b, and again reproduced
P2a's frozen seed-42 XGBoost evidence exactly. The comparison that carries weight
is against **P2a**, not between the two checkpoints: P2a read the canonical
56,790-row dataset through `nn.benchmark`, and these read the 45,802-row
committed snapshot through `nn.p2b` and its alignment layer. Identical fold
returns across two runners and two data paths is evidence that adding a feature
family does not perturb the control. (When P2c first ran it also carried a
different source digest from P2b's; after the provenance remediation both
checkpoints run one revision of the runner, so that particular difference is
gone and the P2a comparison is the one doing the work.)

**Do not respond to this by tuning `chart_structure_v1`'s constants either.**
`WIN_SHORT` and `WIN_LONG` were fixed in advance and searching them against
these four outer blocks would convert a negative result into a fitted one.
`docs/chart_structure_v1.md` §9(i) and §9(j) record the two defects found after
the spec was frozen; neither is a threshold to search.

### What the two negative answers together do and do not license

The strongest claim these results support is narrow, and worth writing down
before someone reaches for a wider one:

> Under the current BTC/USDT 1h research design, the tested OHLCV-derived model
> and feature families did not produce robust incremental performance across the
> four temporal research folds. That is evidence against spending the next
> checkpoint on another hand-designed transformation of the same hourly bars. It
> is **not** proof that all possible OHLCV-derived alpha is impossible.

Three families of transformation, three untuned model families and one label at
one horizon, measured over four adaptive folds of one asset on one exchange, is
not a proof about a space. It is a reason to change what the next checkpoint
spends its budget on.

---

## In flight

### P3 — does trade-level microstructure add information beyond OHLCV14?

**Unanswered. The machinery is built and tested; the data could not be
acquired.**

Both structure families disappointed, and every family so far is a function of
the same five numbers per hour. So P3 does not transform those numbers again: it
changes the *source*. `microstructure_v1`
([`microstructure_v1.md`](microstructure_v1.md)) is 32 causal columns computed
from individual executions — Binance's public spot `aggTrades` archive — folded
into hourly sufficient statistics: trade intensity, aggressive-flow imbalance,
trade size against a trailing distribution, arrival burstiness, price
displacement per unit of traded volume, absorption-*like* quantities, the
intra-hour distribution of activity, and an eight-bin approximation of volume at
price.

The design is the same three-arm design P2b and P2c used, against the same
control, with everything except the columns held at the values P2a ran under:

| arm | columns |
| --- | --- |
| `ohlcv14` | 14 |
| `microstructure_v1` | 32 |
| `ohlcv14_plus_microstructure_v1` | 46 |

Three untuned models, four temporal outer folds, the same target, costs,
threshold rule and sealed boundary. The continuation bar is predeclared and is a
count of folds, not a mean: 3 of 4 or better is worth continuing on, 2 of 4 is
inconclusive, 0–1 of 4 is negative.

**What stopped it.** The trade source has to be fetched from
`data.binance.vision`, and outbound access to that host — and to
`api.binance.com` — is denied by the egress policy of the environment this
checkpoint was developed in. No archive could be downloaded, so **no P3 cell has
been fitted, no P3 number exists, and `artifacts/` contains no P3 evidence.**
Nothing was substituted for the missing data: a P3 result computed from
synthetic trades would be a fabricated research finding, and this repository has
two negative checkpoints precisely because it does not do that.

**What is built and tested anyway**, because it is the durable half and it is
what the acquisition needs to exist first:

- `tools/export_trade_snapshot.py` — streams the official archives one period at
  a time, folding each into hourly rows and deleting it before requesting the
  next, so a five-year acquisition needs one archive of disk rather than a
  billion rows of it. `--plan` lists what it would fetch without touching the
  network; `--probe` measures a couple of real archives and projects the whole
  acquisition from measurements rather than from an estimate.
- `nn/trade_aggregates.py` — the aggregation, the aggressor rule, the size and
  volume-at-price histograms, and the per-archive epoch-unit resolution. Binance
  spot archives carry milliseconds up to 2024 and microseconds from 2025-01-01;
  the unit is resolved by requiring every timestamp inside the archive's own
  calendar period, and an archive that fits no unit or two is refused rather
  than read under a guess. Everything downstream is int64 UTC nanoseconds.
- `tools/verify_trade_snapshot.py` — 27 checks, fail-closed, recomputing every
  claim the manifest makes. `nn.p2b` runs it in the only function that loads the
  trade source, so a corrupt snapshot produces zero model fits by any path.
- `nn/microstructure.py` and the alignment layer — the join is proved by
  re-deriving two columns from the source two different ways, and a matrix
  rolled one hour forward is rejected.
- The leakage battery in `tests/test_p3_leakage.py`: append invariance, future
  mutation invariance, the half-open hour boundary, gap resets, order invariance
  within an hour, duplicate rejection, the rolled matrix, the future-quantile
  attack, a Styx breach stopping the run before any fit, and semantic-identity
  falsification — each with a positive control beside it.

**What the acquisition costs, so far as can be stated without running it.** The
window is 2019-12-01 to 2025-05-19T08:00 — one month of warm-up before the
research spine, ending at the hour after its last candle, which is over three
months before Styx. That is 84 archives (65 monthly, 19 daily). No archive that
could contain a sealed trade is ever requested, so the seal is a property of the
plan rather than of a filter applied afterwards. The committed snapshot is the
hourly aggregate, which is 472 canonical bytes per hour and does not grow with
trade count; the raw trades are read once and discarded. At the observed Parquet
compression that is roughly 13 MB for the full window — which is larger than the
512 KiB `check-added-large-files` limit in `.pre-commit-config.yaml`, so how this
repository stores a research source of that size is a decision for whoever runs
the acquisition, not one this branch should make in advance.

**The estimate of one to two billion trades and 50–150 GB is provisional and is
not this repository's measurement.** `make trade-probe` exists to replace it
with one.

---

## Next, once P3 has an answer

If P3 is positive, the first thing to establish is whether the gain survives
without `ms_log_trade_count`, the one non-stationary column in the family
([`microstructure_v1.md`](microstructure_v1.md) §6) — as a description of where
the signal sat, not as a new arm to select.

If P3 is negative, the next genuinely new source is **funding, open interest and
basis**, not order-book/OFI. Two reasons, and they are about what the evidence
would mean rather than about which is more interesting:

- **P3 would already have tested the fine-grained-flow hypothesis.** Trade flow
  and L1/L2 order-flow imbalance are the same family of claim at two
  resolutions. A negative P3 makes it likely that an OFI checkpoint on the same
  four blocks would answer the same way, at several times the acquisition cost —
  L2 reconstruction needs full-depth updates, not a daily archive.
- **Funding, OI and basis are a different *kind* of information.** They are
  positioning and cost-of-carry, published at low frequency, cheap to acquire
  from the same public archive, and they say something about the state of
  leveraged participants that neither candles nor executions can. That
  independence is what makes it a real second experiment rather than a finer
  version of the first.

Order book and OFI stay on the list; they belong after a positioning checkpoint,
and they need their own contract and their own acquisition work.

---

## Standing constraints

- No live trading, no execution optimisation, no risk-parameter tuning while the
  research question is open. There is nothing to execute yet.
- No hyperparameter search against outer validation. Ever.
- No coin-switching in response to a disappointing BTC result. A different
  market is a different contract and a different question, not a second attempt
  at this one.
- Frozen evidence is never rewritten. A checkpoint that needs new numbers gets a
  new artifact directory and a new checksum manifest; see
  [`../artifacts/README.md`](../artifacts/README.md). A manifest covers *primary*
  evidence only — the cells and their per-sample predictions, which cannot be
  rebuilt without re-fitting. Comparisons and ablation tables are derived from
  those cells, are regenerated whenever the aggregator improves, and are pinned
  by regenerating them and checking what they say rather than by a hash.
