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
  0.012-0.055): choppy uptrends. Buy-and-hold beats every arm over these
  windows, at full exposure, and is a reference rather than a competitor.

What survives is the machinery, which was always the more durable half: the
alignment layer, the parity proof, the snapshot anchoring, the recomputation
path and the source-digest identity are reusable by any later information set.

**Do not respond to this by tuning `smc_v1`'s constants.** They were predeclared,
and searching them against these four outer blocks would convert a negative
result into a fitted one. `docs/smc_v1.md` §9 records the defects worth fixing in
a future `smc_v2`; none of them is a threshold to search.

---

## Next

### P2c — does causal classical chart structure add information?

Specified in [`chart_structure_v1.md`](chart_structure_v1.md) and implemented in
`nn/chart_structure.py`, with 48 tests written from the spec without reading the
engine. Not yet wired into any information set and not yet benchmarked.

The same three-arm design, against `ohlcv14` — which P2b leaves as the best
information set this repository has, by default rather than by merit.

**P2c will be exploratory and adaptive, and must be labelled as such.** By the
time it runs, the same four outer blocks will have been read many times. It can
generate hypotheses; it cannot confirm one.

### After that: microstructure-lite

If both market structure and classical chart structure disappoint, the next
family is not another transformation of the same OHLCV bars. Every family so far
is a function of five numbers per hour, and there is a limit to how much
independent information those can hold. The next genuinely *new* information
would be:

- trade-level history: trade intensity, aggressive-volume imbalance
- volume-at-price structure: high- and low-volume nodes
- price displacement per unit of volume; absorption
- breakout quality measured against participation rather than range
- market-time context

and later, if that pays: funding, basis, open interest, L1/L2 depth, order-flow
imbalance, multi-timeframe context.

**None of those datasets exist in this repository, and none should be fabricated
to make a roadmap look complete.** Acquiring them is its own piece of work, with
its own contract and its own sealed boundary.

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
  [`../artifacts/README.md`](../artifacts/README.md).
