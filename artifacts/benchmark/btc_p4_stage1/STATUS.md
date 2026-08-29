# CURRENT

The deciding P4 Stage-1 screen: the preregistered rule that was to decide whether the
single-use P4-HOLD holdout would be spent. It compares the `ohlcv14_plus_derivatives_v1`
arm against the `ohlcv14` control on `xgboost` at a cost multiplier of 1.0, on the
exploratory blocks that passed the availability rule, and reports the outcome.

**It did not pass.** Of the four preregistered exploratory blocks, three were available
and all three were valid; one improved. The mean combined-minus-control net-return delta
over the valid folds is `-0.038821333333333326` and the worst fold is
`-0.09306799999999998`, against a rule requiring three improved folds, a mean above zero
and a worst fold no lower than `-0.02`. `screen.passed` is `false` and `screen.outcome`
is `screened_out`.

**So P4 stopped here.** There was no Stage 2 and no re-fit. `P4-HOLD` was **retired
unread** — never opened, scored, evaluated, used for model selection, or used for any P4
result — and the ledger records `state: retired` with `checkpoint: null`. Retiring rather
than banking it is deliberate: publishing this Stage-1 result is what would make a later
reuse of the same holdout adaptive. Styx remains sealed; nothing in P4 reaches the sealed
rows.

**Not a research result, and it says so.** Its own `evidence_class` reads *"exploratory
screen on burned blocks; not a research result"*. These blocks had already been observed
by earlier checkpoints, and the rule is a screen with a stated false-continuation rate
under a coin null of 0.3125 — its only job was to decide whether to spend a holdout, and
a screen cannot confirm a hypothesis. The nine-cell aggregate behind it is
`benchmark/btc_p4_comparison`; the cells themselves are frozen under
`btc_p4_stage1_SHA256SUMS.txt`.

**Frozen on its own.** `stage1.json` is the deciding report and names its own manifest,
`btc_p4_screen_SHA256SUMS.txt`, in its `frozen_evidence` field — a single-file freeze of
the decision, taken at the moment it was made. This `STATUS.md` is index metadata written
afterwards and is deliberately **not** covered by that manifest: adding it would mean
restating what the frozen report says its own freeze contains, and rewriting the deciding
evidence after the fact is the one thing a decision freeze exists to prevent.
