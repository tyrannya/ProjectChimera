# `mtf_v1` — the OHLCV14 engine on a 4h and a daily clock

Research checkpoint: **P5** (*does strictly causal higher-timeframe OHLCV context
— `mtf_v1`, alone or combined with OHLCV14 — add usable information beyond
OHLCV14?*)

**P5 ran and is negative.** The deciding `xgboost` comparison improved 1 of 4
temporal outer folds against a bar of 3; fold deltas `+0.11508`, `-0.075359`,
`-0.039844`, `-0.183647`. §9.2 of the preregistration applies: these constants
are not tuned, there is no `mtf_v2`, and the next research move changes axis.
Section 8 below was written before that result existed and is unchanged by it.

Status: **adaptive research evidence**. Not a pristine out-of-sample
confirmation, and it cannot become one: P5's four outer blocks have been read by
eight prior passes. See [`p5_preregistration.md`](p5_preregistration.md) §9.

Preregistered in full at [`p5_preregistration.md`](p5_preregistration.md), hash
`sha256:dc4bd73a078a166e366381c2297bcdad0328c5e08da8def928e8e1f37f04ed8c`. **That document is the authority on every constant below.** This one
says what the family *is* and what implementing it turned up; it does not restate
a number the preregistration fixes, because two copies of a constant are one
opportunity for them to differ.

Implemented in [`../nn/mtf.py`](../nn/mtf.py). Spec hash `63d748a3321ef076b5011ab5b7adff6217b37d164f5960c1d2bb6d745d9b26c5`.

---

## 1. What it is, in one paragraph

`mtf_v1` is **not a new feature family.** It is
`chimera.features.compute_features` — the fourteen columns the control is built
from, unchanged, including the window lengths — evaluated over 4h and 1d bars
instead of 1h bars, and aligned to each 1h decision row by the last higher-
timeframe bar that had **closed**. Twenty-eight columns: `mtf_4h_<name>` and
`mtf_1d_<name>` for each of the fourteen.

That is the whole idea, and its economy is the argument for spending a checkpoint
on it. Four families have now been designed, specified, implemented and refuted;
this one required no design decisions about *what to measure*, only about *when a
row is allowed to see it*.

## 2. The three rules that make it causal

**A bar is used only when it is complete.** A higher-timeframe bar is
`open=first, high=max, low=min, close=last, volume=sum` over its constituent 1h
candles, on a fixed UTC grid, and only if **every** one of them is present. The
committed history has 15 gaps totalling 32 missing hours; a bar built from three
of its four is not a 4h bar, and forward-filling one puts an unobserved hour into
every row that reads it. Measured: 20 of 11,792 4h bars and 16 of 1,966 1d bars
are dropped.

**A bar is visible only when it has closed.** The as-of index is
`searchsorted(close_times, t, side="right") - 1`, so the row at `t` reads the last
bar whose close time is at or before `t`. A row landing exactly on a close does
see that bar; the next one is invisible to it. There is no branch in which a bar
contributes to a row inside its own window.

**A row with no fresh context is ineligible, not served a stale one.** If a
dropped bar leaves a hole, the nearest complete bar is more than one bar old, and
that row is excluded from the sample universe rather than given eight-hour-old
context while its neighbours get fresh context.

Measured on the committed history, all three rules together leave **44,171 of
45,802** spine rows eligible, with the ineligible ones contiguous at the head and
every fold's inner and outer block complete. §6.4 of the preregistration records
that measurement and the fact that it was taken before any fit.

## 3. Why the join needed a different witness

Every other family in this repository is checked for a shifted join by rolling
one recomputed column by ±1 row and asserting it stops matching —
`smc_body_ratio` for the candle families, `ms_qty_imbalance` for P3.

**That control is degenerate here.** A higher-timeframe column is *piecewise
constant* across the rows inside one bar, so a whole-column ±1 shift matches
almost everywhere and would report a clean join whether or not the join was
clean. It is exactly the failure mode
[`p2b_methodology.md`](p2b_methodology.md) records for P2c, where rolling all 30
chart columns one candle forward left the old check reporting `matches: true`.

So `nn.mtf.mtf_join_evidence` evaluates the shift at the **boundary rows** — the
rows where the as-of bar index changes, and the only rows where a shift is
detectable — and reports how many there were, so the coverage does not have to be
taken on trust. On the committed history that is 11,212 boundary rows for 4h and
1,871 for 1d, and both the +1 and the −1 shift fail to match, which is what makes
the match evidence rather than an artefact.

Two witnesses, chosen so they fail differently: `ret_1` rebuilt **positionally**
from the complete-bar closes at the indices the join selected, and `atr_norm`
rebuilt **by close timestamp** through a lookup keyed on the bar's own close time.
A positional check alone would still pass if the join and the re-derivation were
shifted together.

## 4. Contiguity — the decision that was not obvious

The complete-bar series is treated as contiguous: feature state is *not* reset at
a dropped bar. `smc_v1` and `chart_structure_v1` do reset at a market-data gap,
so this is a deliberate departure and §3.6 of the preregistration carries the
measurement that settled it, taken before any fit.

The short version: a per-segment reset leaves 30,563 of 45,802 rows eligible with
outer block 0 at 0.621. That does not merely cost sample — it fails the
availability rule and **changes which folds exist**, so "3 of 4" would then be a
statement about a different experiment.

The reason those two families reset is that their state is *structural*: a swing
high inferred across a hole is a claim about prices nobody saw. A moving average
over observed closes makes no such claim. Every bar that survives the
completeness rule is fully observed and strictly in the past of the row reading
it; dropping a bar removes an unobserved bar from a series, it does not invent
one.

## 5. Feature spec

Deliberately by reference: the fourteen names, their formulas and their window
lengths are `chimera.features._FEATURE_NAMES` and
`chimera.features.FeatureSpec()`, unchanged, and
`tests/test_p5_leakage.py::test_the_family_is_the_ohlcv14_engine_on_a_wider_bar`
recomputes the family and compares it against `compute_features` run directly on
the bar series. There is no second list here to drift from the first.

The two families an ablation could remove are `mtf_4h` and `mtf_1d`, fixed in
`nn.mtf.MTF_FEATURE_FAMILIES` rather than derived from whichever clock turned out
to matter. **No ablation is part of P5**; the partition exists because
`InformationSet` requires families to partition columns, and choosing the groups
after seeing the result would make a later ablation a search.

## 6. The sample universe

`mtf_v1` is undefined until each higher clock has warmed up, so P5 runs on a
restricted universe — computed once and applied to all three arms **from the same
array object**, so "the arms were compared on the same rows" is a property of
construction. That is the mechanism
[`p4_preregistration.md`](p4_preregistration.md) §6.2 established for P4's
punctured derivatives feed, reused unchanged.

The consequence is the one P4 named: P5's `ohlcv14` control is **re-run on P5's
universe** rather than reproduced from P2a's frozen numbers. It will therefore
*not* be byte-identical to the P2b, P2c and P3 controls, and that is correct —
comparing an arm scored where its data exists against a control scored everywhere
would measure two market periods and report the difference as an information set.

## 7. What this family does not do

- **It does not add a source.** The bars are cut from the candles the control
  already reads, which is why P5 adds no new way to reach Styx.
- **It does not rescale the windows.** A 26-bar EMA on 4h bars is a 104-hour
  lookback; on 1h bars the same constant is 26 hours. Matching the lookbacks in
  hours would make the higher-timeframe arm a smoothed copy of the control — a
  question about smoothing, not about timeframe.
- **It does not look at a partially formed current bar.** A live system *could*
  read the 4h bar in progress, and that is a genuinely different family with a
  genuinely different leakage profile. It is not this one.

## 8. Defects and open questions, recorded before any result

Written here now so that a later `mtf_v2` — if the programme ever wants one — has
a list that was not assembled after seeing what P5 said. **None of these is a
threshold to search against P5's folds.**

(a) **Two clocks, chosen rather than derived.** 4h and 1d are the two most widely
watched higher timeframes for an hourly trader, which is a reason and not a
measurement. A family that also carried 12h, or that carried only 1d, is a
different family.

(b) **The warm-up is inherited, not justified for a wider bar.** 78 bars is
`FeatureSpec.warmup`, chosen for 1h bars on a three-time-constants argument. It is
reused unchanged because rescaling it would be a second axis moving, but 78 daily
bars is 78 *days*, and the argument for that number was never made about a daily
series. It costs 1,631 rows, all of them training rows.

(c) **The staleness bound is one bar, and one bar is a lot on a daily clock.**
A row 23 hours past a daily close reads context that is nearly a full session old.
That is correct — it is the freshest closed daily bar there is — but it means the
daily columns are much smoother than the 4h ones by construction, and the family
cannot distinguish "the day has been quiet" from "the day has barely started".

(d) **Volume aggregation is a plain sum.** For a venue with a session structure
that would be wrong; for a 24/7 perpetual market it is the obvious choice, and it
was not tested against an alternative.

(e) **The daily grid is UTC.** Every crypto venue publishes it that way, so this
is the reproducible choice rather than the informative one. A market whose
participants think in a different day would want a different grid, and picking
one after seeing results would be exactly the search this section exists to
foreclose.
