# CURRENT

One P5 cell: the `lightgbm` model on the `ohlcv14_plus_mtf_v1` information set — both, 42 columns, concatenated in that order — over
four temporal outer folds, under the `btc-usdt-1h-gen1` research contract.

A source run for `benchmark/btc_p5_comparison`, which is where the aggregate over all
nine cells lives, and for `benchmark/btc_p5_decision`, which applies the preregistered
rule. This directory on its own is one arm of a three-arm comparison and says nothing
about the question by itself.

P5's control is re-run on P5's sample universe rather than reproduced from P2a's frozen
numbers, so it is **not** byte-identical to the P2b, P2c and P3 controls. That is
correct rather than a defect: `mtf_v1` is undefined until each higher clock has warmed
up, and comparing an arm scored where its data exists against a control scored
everywhere would measure two market periods and report the difference as an information
set (`docs/p5_preregistration.md` §6.1).

**Adaptive research evidence.** These four outer blocks had been read by v4, P2a, P2b,
the P2b ablation, the P2b regime description, P2c, P3 and P4 before P5 ran. Frozen under
`artifacts/btc_p5_SHA256SUMS.txt`.
