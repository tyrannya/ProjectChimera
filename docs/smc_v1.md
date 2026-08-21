# SMC v1 — causal market-structure information set

Version: `smc_v1`
Status: predeclared before any P2b outer-validation result was observed.
Research checkpoint: **P2b** (*does causal measurable market structure add usable
information beyond OHLCV14?*).

This document is the **definition**, not a description. Every feature below has
one exact formula and one exact firing rule. `nn/smc.py` implements it and
`tests/test_smc_features.py` pins it. If the two disagree, one of them is a bug;
neither is free to reinterpret the other.

---

## 0. Why this family, and what it is not

"Smart money concepts" as practised is discretionary: a human draws a level, and
a second human draws a different one. That is not a research information set,
because it cannot be recomputed. What *is* recomputable is the measurable price
geometry the vocabulary points at — confirmed swing points, breaks of those
points, liquidity resting at equal highs and lows, wicks that take that liquidity
and fail, candles whose bodies dominate their range, and unfilled three-candle
imbalances.

That geometry is what `smc_v1` measures. It carries no claim that the geometry is
predictive; the whole point of P2b is to find out.

**Order blocks are deliberately not implemented.** Every definition in common use
requires either a discretionary choice of which candle in a cluster counts, or
knowledge of what price did afterwards. Neither survives the causality rule in
§2. Deferred until a rigorous causal definition exists.

---

## 1. Domain and inputs

Input is a **contiguous candle segment**: rows `t = 0 … T-1` of `(o, h, l, c, v)`
with strictly consecutive timestamps one `timeframe` apart. Segments are the ones
`nn.data_pipeline._contiguous_segment_ids` already assigns, so a missing-candle
gap starts a new segment.

**All state resets at every segment start.** No pivot, structure, equal level,
sweep, break, or fair-value gap may reference a candle on the other side of a
gap. A 14-hour exchange outage is not a 1-hour transition, and the only honest
thing to do with the state that spanned it is to drop it.

### 1.1 Scale normalisation

Every price-valued quantity is divided by ATR so the feature is comparable across
a 2020 BTC price of $7,000 and a 2025 price of $100,000.

```
TR[t]  = max( h[t] - l[t],
              |h[t] - c[t-1]|,
              |l[t] - c[t-1]| )            TR[0] = h[0] - l[0]
ATR[t] = wilder(TR, 14)[t]                 wilder = ewm(alpha=1/14, adjust=False)
```

This is the same `_true_range` / `_wilder` pair `chimera.features` uses for
`atr_norm`, imported rather than reimplemented.

Divisions guard against a degenerate flat market:

```
atr_den[t] = max( ATR[t], 1e-12 * c[t] )
```

### 1.2 Fixed constants

Predeclared. **Not tuned, and not to be tuned against any outer-validation
result.** They are round numbers chosen for interpretability, not search output.

| symbol | value | used by |
| --- | --- | --- |
| `ATR_PERIOD` | 14 | all normalisation |
| `PIVOT_LEFT` | 3 | pivot detection |
| `PIVOT_RIGHT` | 3 | pivot detection **and confirmation delay** |
| `BREAK_ATR` | 0.10 | BOS / CHOCH |
| `SWEEP_ATR` | 0.05 | liquidity sweeps |
| `EQUAL_ATR` | 0.25 | equal highs / equal lows |
| `DISPLACEMENT_BODY_ATR` | 1.00 | displacement candles |
| `DISPLACEMENT_BODY_RATIO` | 0.60 | displacement candles |
| `FVG_MIN_ATR` | 0.05 | fair-value gaps |

---

## 2. The causality rule

> A pivot at candle `i` detected with `PIVOT_RIGHT = 3` **cannot be known at
> candle `i`**. It becomes observable at candle `i + 3` and not one candle
> earlier.

Formally: let `raw_swing_high[i]` be the geometric predicate of §3.1. Its value
depends on `h[i-3 … i+3]`. It is therefore *undefined* to a decision made at row
`i`, `i+1` or `i+2`, and defined from row `i+3` onward.

Every feature in this document is a function of the candle history `≤ t` **and of
the confirmed-pivot set**

```
S_high(t) = { i : raw_swing_high[i]  and  i + PIVOT_RIGHT <= t }
S_low(t)  = { i : raw_swing_low[i]   and  i + PIVOT_RIGHT <= t }
```

which by construction only contains pivots whose confirming candle has already
closed. This yields the property the tests enforce:

**Append invariance.** For every `N`, `K > 0` and every feature column `F`:

```
F( candles[0 : N] )        ==        F( candles[0 : N+K] )[0 : N]
```

Appending future candles never alters an already-observable historical value.
`tests/test_smc_features.py::test_append_invariance_*` asserts this on synthetic
fixtures and on real pre-Styx BTC candles at multiple `(N, K)`.

Implementation consequence: the engine is a **single forward pass**. At row `t`
it first absorbs the pivot at index `t - PIVOT_RIGHT` (if any), then emits the
row's features. There is no backward fill, no centred window, and no second pass.

---

## 3. Definitions

### 3.1 Pivots

```
raw_swing_high[i] = 1   iff   3 <= i <= T-1-3
                          and h[i] >  h[j]  for all j in [i-3, i-1]
                          and h[i] >= h[j]  for all j in [i+1, i+3]

raw_swing_low[i]  = 1   iff   3 <= i <= T-1-3
                          and l[i] <  l[j]  for all j in [i-3, i-1]
                          and l[i] <= l[j]  for all j in [i+1, i+3]
```

The asymmetry (strict left, non-strict right) is deliberate and is the tie-break
rule: on a flat plateau of equal highs the **earliest** candle is the pivot, and
exactly one candle is. Without it a plateau produces either zero pivots or
several, depending on floating-point noise.

Confirmed at `i + 3`, per §2.

### 3.2 Structure state

Maintained left to right. At row `t`, after absorbing the pivot at `t-3`:

- `last_sh` / `last_sh_i` — price and index of the newest **confirmed** swing high
- `last_sl` / `last_sl_i` — same for lows
- `active_sh` — the newest confirmed swing high **not yet broken** (`None` after a
  bullish break, until a new swing high is confirmed)
- `active_sl` — symmetric
- `dir` ∈ {-1, 0, +1} — structure direction, `0` until the segment's first break

### 3.3 Breaks: BOS and CHOCH

A **bullish break** fires at `t` iff `active_sh` exists and

```
c[t] > active_sh + BREAK_ATR * atr_den[t]
```

It is a **BOS** (continuation) when `dir >= 0` and a **CHOCH** (change of
character) when `dir == -1`. On firing: `dir := +1`, `active_sh := None`.

A **bearish break** fires iff `active_sl` exists and
`c[t] < active_sl - BREAK_ATR * atr_den[t]`; it is a BOS when `dir <= 0` and a
CHOCH when `dir == +1`. On firing: `dir := -1`, `active_sl := None`.

Break on the **close**, not the wick: a wick through a level that closes back
inside is the sweep of §3.5, and calling both a break would erase the
distinction the family exists to measure. When both directions qualify on one
candle (possible only in a violent outside bar), the bullish test is evaluated
first and the bearish test then sees the updated state; this is a declared
tie-break, not an accident.

### 3.4 Equal highs and equal lows (resting liquidity)

When a swing high at index `i` is confirmed at `t = i+3` and a previous confirmed
swing high `prev_sh` exists:

```
equal_high fires at t   iff   |h[i] - prev_sh| <= EQUAL_ATR * atr_den[t]
```

and creates an **equal-high level** at `max(h[i], prev_sh)` — the liquidity sits
at the higher of the pair. Symmetrically for equal lows, at `min(l[i], prev_sl)`.

A level stays **active** until price trades through it: an equal-high level `L`
is removed at the first `u` with `h[u] > L`; an equal-low level `L` at the first
`u` with `l[u] < L`. Removal is checked before the row's features are emitted, so
a level taken out this candle is not reported as active on it.

### 3.5 Liquidity sweeps and reclaims

A **high-side sweep** fires at `t` iff `active_sh` exists and

```
h[t] >  active_sh + SWEEP_ATR * atr_den[t]      (the wick takes the level)
c[t] <= active_sh                                (the close does not hold it)
```

A **low-side sweep**: `l[t] < active_sl - SWEEP_ATR * atr_den[t]` and
`c[t] >= active_sl`.

A sweep does **not** clear `active_sh` / `active_sl`: the level was probed, not
broken, and the liquidity question is still open.

**Reclaim** confirms the sweep failed. After a high-side sweep at index `s`, a
`sweep_high_reclaim` fires at the first `t > s` with

```
c[t] < l[s]
```

— the close is below the *entire* sweep candle. The pending reclaim is cancelled
by a bullish break (which refutes the failure thesis) and replaced by any newer
high-side sweep. Symmetrically, `sweep_low_reclaim` fires at the first `t > s`
with `c[t] > h[s]`, cancelled by a bearish break.

This definition takes no lookback-window parameter on purpose: a window length
would be a tuned constant, and `bars_since_sweep_*` already exposes staleness to
the model as a feature rather than hiding it in a threshold.

### 3.6 Displacement

```
body[t]       = c[t] - o[t]
body_ratio[t] = |body[t]| / (h[t] - l[t])          0 when h[t] == l[t]

displacement_bull[t] = 1  iff  body[t] > 0
                           and |body[t]| >= DISPLACEMENT_BODY_ATR * atr_den[t]
                           and body_ratio[t] >= DISPLACEMENT_BODY_RATIO

displacement_bear[t] = 1  iff  body[t] < 0  and the same two magnitude tests
```

Purely local to candle `t`; no state, and trivially causal.

### 3.7 Fair-value gaps

The three-candle imbalance ending at `t` (requires `t >= 2` **within the
segment**):

```
bullish FVG at t  iff  l[t] > h[t-2]  and  (l[t] - h[t-2]) >= FVG_MIN_ATR * atr_den[t]
                  interval [ h[t-2], l[t] ]

bearish FVG at t  iff  h[t] < l[t-2]  and  (l[t-2] - h[t]) >= FVG_MIN_ATR * atr_den[t]
                  interval [ h[t], l[t-2] ]
```

An FVG stays **active** until fully filled: a bullish gap `[lo, hi]` is removed at
the first `u > t` with `l[u] <= lo`; a bearish gap at the first `u > t` with
`h[u] >= hi`. Fills are processed before the row's features are emitted.

"Fully filled" rather than "touched" is a declared choice: a partial touch leaves
an imbalance, and the alternative would make the feature a near-duplicate of
`hl_range`.

---

## 4. The 39 features

Column order is fixed and is part of the spec. `V` marks an availability flag —
they exist because `0.0` is a legal value for the distance features, so
"no confirmed structure yet" needs its own encoding rather than a magic number.

### A. structure (8)

| # | column | definition |
| --- | --- | --- |
| 1 | `smc_structure_valid` | V: 1 iff both `last_sh` and `last_sl` exist |
| 2 | `smc_dist_swing_high_atr` | `(last_sh - c[t]) / atr_den[t]`, else 0 |
| 3 | `smc_dist_swing_low_atr` | `(c[t] - last_sl) / atr_den[t]`, else 0 |
| 4 | `smc_age_swing_high` | `log1p(t - last_sh_i)`, else 0 |
| 5 | `smc_age_swing_low` | `log1p(t - last_sl_i)`, else 0 |
| 6 | `smc_structure_direction` | `dir` ∈ {-1, 0, +1} |
| 7 | `smc_range_width_atr` | `(last_sh - last_sl) / atr_den[t]`, else 0 |
| 8 | `smc_range_position` | `(c[t] - last_sl) / (last_sh - last_sl)`, clipped to `[-2, 3]`; `0.5` when invalid or width ≤ 0 |

Ages are `log1p` rather than raw: a raw age is unbounded and grows without limit
inside a long segment, which would hand a standardised model a proxy for
"how far into the segment are we" instead of a structural measurement.

### B. equal levels / liquidity (6)

| # | column | definition |
| --- | --- | --- |
| 9 | `smc_equal_high` | event, §3.4 |
| 10 | `smc_equal_low` | event, §3.4 |
| 11 | `smc_eqh_active` | V: 1 iff an equal-high level ≥ `c[t]` is active |
| 12 | `smc_eql_active` | V: 1 iff an equal-low level ≤ `c[t]` is active |
| 13 | `smc_eqh_dist_atr` | `(min{L active, L >= c[t]} - c[t]) / atr_den[t]`, else 0 |
| 14 | `smc_eql_dist_atr` | `(c[t] - max{L active, L <= c[t]}) / atr_den[t]`, else 0 |

### C. breaks (6)

| # | column | definition |
| --- | --- | --- |
| 15 | `smc_bos_bull` | event, §3.3 |
| 16 | `smc_bos_bear` | event, §3.3 |
| 17 | `smc_choch_bull` | event, §3.3 |
| 18 | `smc_choch_bear` | event, §3.3 |
| 19 | `smc_break_magnitude_atr` | on a break, signed `(c[t] - level) / atr_den[t]`; 0 on a non-break candle |
| 20 | `smc_bars_since_break` | `log1p(t - last_break_i)`, 0 if no break yet |

### D. sweeps (6)

| # | column | definition |
| --- | --- | --- |
| 21 | `smc_sweep_high` | event, §3.5 |
| 22 | `smc_sweep_low` | event, §3.5 |
| 23 | `smc_sweep_high_reclaim` | event, §3.5 |
| 24 | `smc_sweep_low_reclaim` | event, §3.5 |
| 25 | `smc_bars_since_sweep_high` | `log1p(t - s_high)`, 0 if none |
| 26 | `smc_bars_since_sweep_low` | `log1p(t - s_low)`, 0 if none |

### E. displacement (5)

| # | column | definition |
| --- | --- | --- |
| 27 | `smc_body_signed_atr` | `(c[t] - o[t]) / atr_den[t]` |
| 28 | `smc_body_abs_atr` | `abs(c[t] - o[t]) / atr_den[t]` |
| 29 | `smc_body_ratio` | §3.6 |
| 30 | `smc_displacement_bull` | event, §3.6 |
| 31 | `smc_displacement_bear` | event, §3.6 |

### F. fair-value gaps (8)

| # | column | definition |
| --- | --- | --- |
| 32 | `smc_fvg_bull` | 1 iff a bullish FVG was created at `t` |
| 33 | `smc_fvg_bear` | 1 iff a bearish FVG was created at `t` |
| 34 | `smc_fvg_bull_size_atr` | its size / `atr_den[t]` on the creating candle, else 0 |
| 35 | `smc_fvg_bear_size_atr` | same, bearish |
| 36 | `smc_bull_fvg_active` | V: 1 iff some active bullish gap has `hi <= c[t]` |
| 37 | `smc_bear_fvg_active` | V: 1 iff some active bearish gap has `lo >= c[t]` |
| 38 | `smc_bull_fvg_dist_atr` | `(c[t] - max{hi : active bull, hi <= c[t]}) / atr_den[t]`, else 0 |
| 39 | `smc_bear_fvg_dist_atr` | `(min{lo : active bear, lo >= c[t]} - c[t]) / atr_den[t]`, else 0 |

**Total: 39 columns**, in six named families used by the P2b ablation study.

---

## 5. No-NaN guarantee, and why it matters

Every column is finite on **every** row of every segment, including row 0. Where
the underlying state does not exist yet, the value is the declared default (`0.0`,
or `0.5` for `smc_range_position`) and the paired availability flag is `0`.

This is not cosmetic. `nn.data_pipeline.build_dataset` drops any row with a NaN
feature. If SMC columns could be NaN, the OHLCV14 information set and the SMC
information set would be evaluated on **different rows**, and every comparison in
P2b would be confounded by the sample universe rather than by the information.
A no-NaN engine makes the common sample universe of `nn/information_sets.py`
provable by construction instead of by hope.

Warm-up rows are not a NaN problem but they are a real one: the first candles of
a segment genuinely have no confirmed structure. They are handled the same way
the OHLCV14 set handles its own indicator warm-up — the processed research
dataset already discards `FeatureSpec.warmup = 78` rows at the head of every
segment, and the SMC engine is run over the **full** segment including those 78
candles, so the structure state entering the first retained row was built from
real history rather than from a truncated view.

---

## 6. What this spec does not claim

- It does not claim these features are predictive. P2b measures that.
- It does not claim these are *the* SMC definitions. They are *a* set of exact,
  causal, reproducible ones, fixed in advance so that a result cannot be
  manufactured by redefining a level after seeing a return.
- It does not claim the constants in §1.2 are optimal. They were not searched,
  and searching them against outer validation is forbidden.

---

## 7. Public API

Pinned here so the engine and the tests that check it can be written against the
same contract rather than against each other.

```python
# nn/smc.py

SMC_SPEC_VERSION: str = "smc_v1"

#: The 39 columns of §4, in that exact order. Never reordered: the feature-spec
#: hash and every persisted prediction record depend on it.
SMC_FEATURE_NAMES: tuple[str, ...]

#: Family name -> its columns. The six families of §4, used by the P2b ablation.
SMC_FEATURE_FAMILIES: dict[str, tuple[str, ...]]

@dataclass(frozen=True)
class SMCSpec:
    atr_period: int = 14
    pivot_left: int = 3
    pivot_right: int = 3
    break_atr: float = 0.10
    sweep_atr: float = 0.05
    equal_atr: float = 0.25
    displacement_body_atr: float = 1.00
    displacement_body_ratio: float = 0.60
    fvg_min_atr: float = 0.05

    def to_dict(self) -> dict[str, float]: ...

    def spec_hash(self) -> str:
        """sha256 over canonical JSON of version + constants + feature names."""

def smc_feature_columns() -> list[str]: ...

def compute_smc_features(
    df: pd.DataFrame, spec: SMCSpec | None = None
) -> pd.DataFrame:
    """SMC features for ONE contiguous segment.

    ``df`` needs ``open``/``high``/``low``/``close`` (``volume`` is accepted and
    unused). The result has ``df``'s index and exactly ``SMC_FEATURE_NAMES`` as
    columns, all float64, and contains no NaN and no infinity.
    """

def compute_smc_features_segmented(
    candles: pd.DataFrame, timeframe: str, spec: SMCSpec | None = None
) -> pd.DataFrame:
    """Split ``candles`` on market-data gaps, compute per segment, concatenate.

    Segments come from :func:`nn.data_pipeline._contiguous_segment_ids`, so the
    boundaries are the ones the processed dataset already used. State never
    crosses a boundary.
    """
```

---

## 8. Order of operations within a candle

Several rules in §3 can fire on the same candle, and which one wins changes the
output. The order is therefore part of the specification rather than an artefact
of how the loop happens to be written. At row `t`, in this sequence:

1. **absorb** the pivot at index `t - PIVOT_RIGHT`, if any — updating
   `last_sh`/`last_sl`, `active_sh`/`active_sl`, and firing `equal_high` /
   `equal_low` (§3.4) with the level it creates
2. **retire** equal-high and equal-low levels price has now traded through (§3.4)
   and fill fair-value gaps price has now closed (§3.7)
3. **create** a fair-value gap from `(t-2, t-1, t)` (§3.7)
4. **break** — bullish tested first, then bearish against the updated state (§3.3)
5. **sweep** (§3.5)
6. **reclaim** a pending sweep (§3.5)
7. **emit** row `t`'s 39 values

Consequences that follow from this order and are part of the definition:

- A pivot confirmed at `t` is immediately breakable and sweepable at `t`. This is
  forced by §2 and is why a BOS often lands on the confirmation candle itself.
- A level created at step 1 that this candle's high already exceeds is retired at
  step 2 and is never reported active — which is what §3.4's "a level taken out
  this candle is not reported as active on it" means.
- A break at step 4 cancels a pending reclaim before step 6 can fire it.
- A new sweep at step 5 replaces the pending sweep, so a reclaim guarded by
  `s < t` cannot fire on a candle that is itself a fresh sweep on the same side.
  On the committed BTC history this binds on 165 candles, about 13% of all
  reclaims, so it is a materially observable choice rather than a formality.
- `prev_sh` in §3.4 is `last_sh` — the newest previously confirmed swing high,
  **including one that has already been broken**. §3.2 distinguishes `last_sh`
  from `active_sh` precisely so this can be said.
- When both directions break on one candle, `smc_break_magnitude_atr` reports
  the **bearish** value: step 4 writes the bullish magnitude and the bearish
  test then overwrites it. Dormant — no such candle exists in the committed
  history — but declared, because a rule that has never fired is still a rule.

### 8.1 "else 0" is per quantity, not gated on the flag

In §4 rows 2–5, the fallback applies to the quantity's *own* precondition:
`smc_dist_swing_high_atr` and `smc_age_swing_high` need only `last_sh`, and the
low pair needs only `last_sl`. `smc_structure_valid` (row 1) requires both, and
so do rows 7–8. On the committed BTC history there are 222 rows where
`smc_structure_valid == 0`, and on 92 of them exactly one side is confirmed and
that side reports a real value while the flag reads 0. That is intended: gating
all seven columns on the flag would discard a measurement that exists.

### 8.2 Pivot detectability is not a function of the frame length

§3.1's bound reads `3 <= i <= T-1-3`, which mentions `T`. It is safe only
because that `3` is `PIVOT_RIGHT`: a pivot at `i > T-1-PIVOT_RIGHT` could not be
*confirmed* inside the frame either, so the two readings coincide and append
invariance holds. Read it as "detectable, and only ever consulted at
`i + PIVOT_RIGHT`". An implementation that treats it as a property of the pivot
rather than of confirmation can copy it into a two-pass design and silently
break causality.

### 8.3 The engine drifted from this list once, and was brought back

§8 was written down at commit `a7a25d2` (01:09:36 UTC), before any P2b cell ran.
The chronology is in git rather than in the artifacts as they stand today, which
have since been regenerated: the P2b cells **as first committed**, at `685764b`,
record runner revisions `c840627` (01:39:48) and `f5b409f` (01:48:27) — thirty
and thirty-nine minutes after §8 landed. `git show
685764b:artifacts/benchmark/btc_p2b_smc_v1_xgboost/p2b.json` is the check.

The engine, committed before §8 existed, created the fair-value gap **after**
the reclaim rather than at step 3, and nobody noticed for the length of two
checkpoints — the loop reads plausibly in either order and every behavioural
test in `tests/test_smc_features.py` passed under both.

It changed nothing. Steps 4–6 neither read nor write gap state, and a gap
cannot be retired by the candle that created it in either arrangement, so the
two orders are **byte-identical**: 0 of 1,786,278 cells differ over the 45,802
rows P2b scores, 0 of 1,932,489 over the full committed pre-Styx history, and 0
over twelve synthetic frames. The engine was moved to the declared order anyway,
because a specification the code does not follow cannot be used to check the
next change, and this one was predeclared. `tests/test_smc_features.py`
transcribes both lists and compares them step for step, which is the only check
that would have caught the drift.

---

## 9. Known defects in `smc_v1`, deferred

These were found by running the engine over the committed pre-Styx history
**after** `smc_v1` had been frozen and the P2b benchmark had started. They are
recorded here rather than fixed, because editing a predeclared feature
specification part-way through the checkpoint it was declared for is exactly the
move this repository exists to prevent. A fix is `smc_v2`'s job.

Four are recorded. **(a)**, **(b)** and **(c)** make the features **weaker**
than intended: a collision that hides a distinction, a name that misdescribes a
signed quantity, and a tail a standardiser will struggle with. **(d)** is not a
predictive defect at all — it is a hole in the hash recipe, and what it can
corrupt is the *identity* of a run rather than the information in a column.
None of the four can leak future information, and none can manufacture a
positive result.

**(a) `smc_bars_since_*` cannot distinguish "never" from "just now".**
`smc_bars_since_break`, `smc_bars_since_sweep_high` and
`smc_bars_since_sweep_low` are `log1p(t - i)`, which is `0.0` on the event candle
itself — the same value §4 uses for "this has not happened yet". Measured on the
committed history: 3,485 zero rows for `smc_bars_since_break`, of which 3,151 are
break candles and 334 are rows before the segment's first break. A model cannot
tell the two states apart. §4 introduces availability flags precisely because
`0.0` is a legal value, and these three columns then reintroduce the collision.
`log1p(1 + t - i)` would fix it without a new column.

Measured again on the 45,802 rows P2b actually scores, rather than on the full
raw history: `smc_bars_since_break` has 2,912 zeros, every one of them an event
candle and none of them "never happened"; `smc_bars_since_sweep_high` has 1,546,
likewise all event candles; `smc_bars_since_sweep_low` has 1,683, of which 127
genuinely mean "no low sweep has occurred in this segment". So the collision
fires on 127 research rows. On the event candles the ambiguity is resolvable
from elsewhere in the same row — `smc_bos_*`, `smc_choch_*` and `smc_sweep_*`
are 1 exactly then — which is why this degrades three columns rather than
corrupting them.

**(b) `smc_range_width_atr` is signed, and "width" is the wrong word for it.**
It goes negative when the newest confirmed swing high sits below the newest
confirmed swing low — which happens after price runs past an old high without
confirming a new one. 75 rows of the committed history are negative, reaching
−4.62. The value is well defined and informative; the name is not. Row 8 guards
`width <= 0` explicitly, so the case was known when row 7 was written.

**(c) Equal-level distances have a very heavy tail.** §3.4 levels never expire,
so `smc_eqh_dist_atr` reaches 181 ATR when a level from an earlier regime stays
active until price finally trades through it. This follows from the same
no-arbitrary-window reasoning as §3.5 and is deliberate, but these two columns
are the ones a standardiser will struggle with, and a later version should
consider a scale-bounded transform rather than a lookback parameter.

**(d) A provenance defect, not a predictive one: the spec pins the hash recipe
but not the JSON type of the integer constants.** `spec_hash()` hashes `dataclasses.asdict(spec)`, which keeps
`atr_period`, `pivot_left` and `pivot_right` as `int` — so the canonical JSON
contains `"atr_period":14`, not `14.0`, and an implementation that casts them to
float to satisfy the `dict[str, float]` annotation in §7 gets a different hash.
The reference value is
`3421312fc8d8687e158b5dc269f65c76bfa6916ec4643f3063cf9473d8a36649`.
