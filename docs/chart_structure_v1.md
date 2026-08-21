# Chart structure v1 — causal classical-pattern information set

Version: `chart_structure_v1`
Status: predeclared before any P2b outer-validation result was observed.
Research checkpoint: **P2b** (*does causal measurable market structure add usable
information beyond OHLCV14?*) — the second family, after `smc_v1`.

This document is the **definition**, not a description. Every feature below has
one exact formula and one exact firing rule. `nn/chart_structure.py` will
implement it and `tests/test_chart_structure.py` will pin it. If the two
disagree, one of them is a bug; neither is free to reinterpret the other.

---

## 0. Why this family, and what it is not

Classical chart patterns fail as a research information set in two separate
ways, and they have to be fixed separately.

**They are discretionary.** A human draws a triangle; a second human draws a
different one. Two analysts disagreeing about where the trendline goes is not a
measurement, because it cannot be recomputed.

**They are confirmed by the future.** This is the worse defect and it is
usually invisible, because it lives inside the published definition rather than
inside the backtest. "A double top is a double top once it breaks the
neckline." The *label* is attached to the second peak; the *information* that
produced the label arrived tens of candles later. Anything that marks the
second peak is showing the model a shape that had not happened yet. The same
hole sits inside the standard flag ("a continuation pattern — it continues"),
the standard triangle ("it broke out upward, so it was ascending") and the
standard failed breakout ("that high was a false break"). A family built out of
these definitions cannot be falsified, because every instance was selected by
its own outcome.

What *is* recomputable, and computable from the past alone, is the geometry the
vocabulary points at: rolling range compression and where price sits inside it,
the slope of the highs and the slope of the lows and whether the two converge,
a least-squares channel with its dispersion and its fit quality, the ratio of
short-horizon realised range to long-horizon, distance beyond a rolling
extreme, a close back inside after such a distance, two comparable confirmed
swings with a trough between them, and a measured impulse followed by measured
compression.

That geometry is what `chart_structure_v1` measures. It carries no claim that
the geometry is predictive; the whole point of P2b is to find out.

### 0.1 Continuous over categorical

Wherever a continuous measure exists, this spec emits the measure and not a
label. "Ascending triangle", "descending triangle" and "symmetric triangle" are
three names for three regions of one continuous axis — the relative sign of the
two boundary slopes — and §3.3 emits that axis (`chart_slope_asymmetry`, in
`[-1, 1]`) instead of the three names. A slope is strictly more informative
than *is a triangle*, and a categorical label is a threshold applied to a
measurement: it would introduce a constant that has to be chosen, and any
choice would eventually be defended by how well it scored.

The rule has one exception, stated once here rather than argued each time: two
**availability flags** (§4 family F) and nothing else. They exist for the same
reason `smc_v1` §4 has them — `0.0` is a legal value for a signed distance, so
"no confirmed structure yet" needs its own encoding rather than a magic number.

### 0.2 Deliberately not implemented

Head and shoulders, wedges as labels, cup-and-handle, harmonic ratios,
measured-move targets, and any trendline drawn through two chosen extremes.
Each requires either a discretionary choice of which points count, or a
confirmation event that lies in the future of the row being labelled. Neither
survives §2. Deferred until a rigorous causal definition exists, exactly as
`smc_v1` §0 defers order blocks.

### 0.3 Overlap with `smc_v1` is expected, and is not a defect

Both families are functions of the same OHLC series, and §3.7 reuses `smc_v1`
§3.1's pivots unchanged. The combined information set will therefore contain
correlated columns. Correlation between two predeclared families is a modelling
fact for P2b to report, not a leak; the alternative — dropping a measurement
because a different family already saw the same candles — would make each
family's content depend on the order the families were written in.

---

## 1. Domain and inputs

Input is a **contiguous candle segment**: rows `t = 0 … T-1` of `(o, h, l, c, v)`
with strictly consecutive timestamps one `timeframe` apart. Segments are the
ones `nn.data_pipeline._contiguous_segment_ids` already assigns, so a
missing-candle gap starts a new segment.

**All state resets at every segment start.** No rolling window, pivot, channel
fit, pending breakout or impulse may reference a candle on the other side of a
gap. A 14-hour exchange outage is not a 1-hour transition, and a 60-bar window
that spans one is not a 60-bar window. On the committed pre-Styx history this
is 16 segments, the shortest 80 candles and the longest 21,297.

### 1.1 Scale normalisation

Every price-valued quantity is divided by ATR so the feature is comparable
across a 2020 BTC price of $7,000 and a 2025 price of $100,000.

```
TR[t]  = max( h[t] - l[t],
              |h[t] - c[t-1]|,
              |l[t] - c[t-1]| )            TR[0] = h[0] - l[0]
ATR[t] = wilder(TR, 14)[t]                 wilder = ewm(alpha=1/14, adjust=False)
```

This is the same `_true_range` / `_wilder` pair `chimera.features` uses for
`atr_norm` and `nn/smc.py` uses for its denominator, imported rather than
reimplemented. It is computed **per segment**, so the Wilder recursion restarts
at each segment start and never carries a level across a gap.

Divisions guard against a degenerate flat market with the identical rule:

```
atr_den[t] = max( ATR[t], 1e-12 * c[t] )
```

`nn.data_pipeline.validate_ohlcv` rejects any non-positive price, so
`atr_den[t] > 0` strictly and the guard is the only epsilon in this document.
Every other division is either guarded by a declared default for its degenerate
case (§5) or has a denominator that is provably positive (§3.1).

### 1.2 Rolling windows

Two window lengths, both **trailing and left-truncated**. For a window length
`n` at row `t` of a segment:

```
w_n(t) = min(n, t+1)                        effective length, >= 1 always
I_n(t) = [ t - w_n(t) + 1 , t ]             inclusive, entirely <= t
```

and one **prior** window, which excludes the current candle:

```
I_l^-(t) = [ max(0, t - WIN_LONG) , t - 1 ]     empty at t = 0
```

Shorthand used below: `I_s = I_{WIN_SHORT}`, `I_l = I_{WIN_LONG}`,
`w_s = w_{WIN_SHORT}`, `w_l = w_{WIN_LONG}`.

Truncation happens at the **start** of the segment and never at the end. This is
the whole causality argument for the window features in one sentence: a window
that is short because the segment just began is a window computed on less
history, which is honest; a window that reaches past `t` is a window computed on
history that has not happened, which is not. In pandas terms this is
`rolling(n, min_periods=1)` on the segment's own frame — never `center=True`,
never `shift(-k)`, never `bfill`.

On the committed pre-Styx history 98.1% of rows have a full 60-bar window and
99.4% have a full 20-bar window, so truncation is a real but small minority
case. `chart_window_fill` (§4 A1) reports `w_l(t) / WIN_LONG` on every row, so
the model can see exactly how much history each row's windows were built from
instead of having to infer it.

### 1.3 Fixed constants

Predeclared. **Not tuned, and not to be tuned against any outer-validation
result.** They are round numbers chosen for interpretability, not search output.
There are six, and the count is deliberately small: every constant is a degree
of freedom that a later disappointing result would invite someone to spend.

| symbol | value | used by |
| --- | --- | --- |
| `ATR_PERIOD` | 14 | all normalisation (§1.1) |
| `WIN_SHORT` | 20 | fast range box, realised-range numerator, flag/pole split |
| `WIN_LONG` | 60 | range box, boundary slopes, channel fit, prior extremes, pending-breakout expiry |
| `TOUCH_ATR` | 0.25 | boundary-touch tolerance (§3.2) |
| `PIVOT_LEFT` | 3 | pivot detection (§3.7) |
| `PIVOT_RIGHT` | 3 | pivot detection **and confirmation delay** (§3.7) |

`ATR_PERIOD`, `PIVOT_LEFT` and `PIVOT_RIGHT` carry `smc_v1`'s values on purpose:
the two families then normalise against a byte-identical denominator and see an
identical swing set, so any difference P2b measures between them is a difference
of definition rather than of parameterisation. `TOUCH_ATR` is `smc_v1`'s
`EQUAL_ATR`, for the same reason — "these two highs are the same level" is the
same judgement in both families and should not be made twice with two numbers.

`WIN_SHORT` and `WIN_LONG` are new and are the only genuinely free choices here.
20 and 60 bars are one day and two and a half days on the 1h research data; they
were picked as round numbers with a 3:1 separation, not searched. Note what is
*not* in this table: no expiry constant for a pending breakout (§3.6 reuses
`WIN_LONG`), no minimum separation for the two tops of a double top (§3.7 gets
`>= 4` bars for free from the pivot geometry), and no clipping bounds (§5).

### 1.4 The column budget

**24 to 32 columns. This set has 30.** The budget is a hard constraint, not a
style preference, and the arithmetic is why. The P2b models flatten a
`SEQ_LEN = 64` candle window, so each column costs 64 model inputs:

| information set | columns | flattened inputs |
| --- | --- | --- |
| `ohlcv14` | 14 | 896 |
| `smc_v1` | 39 | 2,496 |
| `chart_structure_v1` | 30 | 1,920 |
| all three combined | 83 | 5,312 |

The smallest training fold in the P2b geometry is `min_train = 21,697` rows. At
5,312 inputs that is roughly four training rows per input, against a target
whose signal-to-noise ratio is assumed low enough that P2b exists to test
whether there is any. Every column added past that point buys a little
information and a lot of variance, and the variance lands on exactly the
comparison the checkpoint is trying to make.

The budget also disciplines the *definition*: it forces the choice of the
continuous measure over the label, the shared axis over the pair of one-sided
indicators, and the two families of §3.6 and §3.7 to justify each column
individually. Where two spellings of the same information were available, the
narrower one was cut, and §9 records which.

---

## 2. The causality rule

> A feature at row `t` may be a function of rows `<= t` **within the same
> segment**, and of nothing else. Not of row `t+1`, not of a centred window, not
> of a pattern's eventual resolution, and not of a candle on the other side of a
> gap.

Three clauses follow from it and each one is testable.

**(a) Windows truncate left.** §1.2. Every rolling aggregate at row `t` is taken
over `I_n(t) ⊆ [segment_start, t]`.

**(b) Pivots carry a confirmation delay.** A pivot at candle `i` detected with
`PIVOT_RIGHT = 3` **cannot be known at candle `i`**. It becomes observable at
candle `i + 3` and not one candle earlier. Formally, with `raw_swing_high[i]`
the geometric predicate of §3.7:

```
S_high(t) = { i : raw_swing_high[i]  and  i + PIVOT_RIGHT <= t }
S_low(t)  = { i : raw_swing_low[i]   and  i + PIVOT_RIGHT <= t }
```

Only these sets are ever consulted. This clause is `smc_v1` §2 unchanged.

**(c) Stamp-at-availability.** *An event is written on the row where it became
knowable, and is never backdated to the row where it began.* This is the clause
that classical pattern definitions violate, and it is the reason several
features below look unlike their textbook namesakes:

- a failed breakout is stamped on the **reclaim candle**, not on the breakout
  candle it invalidates (§3.6);
- a double top is never stamped at all — §3.7 emits the *live* geometry of the
  two newest confirmed swing highs on every row, including the distance from the
  close to the candidate neckline, and lets the row where price actually closes
  through the neckline be the row that says so;
- a flag is never stamped — §3.8 emits the prior impulse and §3.5 emits the
  current compression, both measured over past bars, on every row.

**Append invariance.** For every `N`, `K > 0` and every feature column `F`:

```
F( candles[0 : N] )        ==        F( candles[0 : N+K] )[0 : N]
```

Appending future candles never alters an already-observable historical value.
`tests/test_chart_structure.py::test_append_invariance_*` must assert this on
synthetic fixtures and on real pre-Styx BTC candles at multiple `(N, K)`,
exactly as `tests/test_smc_features.py` does for `smc_v1`. A spec whose
append invariance is only argued in prose is a spec whose engine will eventually
be written with a `center=True` in it.

Implementation consequence: the engine is a **single forward pass**. At row `t`
it absorbs the pivot at index `t - PIVOT_RIGHT` (if any), resolves pending
breakout state from earlier rows, then emits the row's features. There is no
backward fill, no centred window, and no second pass.

---

## 3. Definitions

### 3.1 Window primitives

Aggregates over a window `I` (all indices `<= t`, §1.2):

```
H_n(t) = max{ h[u] : u in I_n(t) }
L_n(t) = min{ l[u] : u in I_n(t) }
```

Least-squares slope of a series `y` over `I` of effective length `w`, with the
window's own index as the regressor:

```
x[u]   = u - t                     u in I,  so x runs over [-(w-1), 0]
slope(y, I) = sum_u (x[u]-xbar)(y[u]-ybar) / sum_u (x[u]-xbar)^2      w >= 2
slope(y, I) = 0                                                       w == 1
```

The denominator is `w(w^2-1)/12`, a function of `w` alone and strictly positive
for `w >= 2`, so this division needs no guard and no epsilon. The slope's unit
is price per bar; every slope emitted in §4 is divided by `atr_den[t]` and is
therefore in ATR per bar.

**Least squares, not a two-point trendline.** A classical trendline is drawn
through two chosen extremes, and *which two* is precisely the discretionary
choice §0 rejects. The least-squares line through every high in the window is
unique, is a continuous function of the data, and moves smoothly rather than
jumping when a new extreme replaces an old one. It is a different object from a
hand-drawn trendline and this spec does not pretend otherwise: it measures the
same thing the trendline is a proxy for.

### 3.2 The range box and boundary touches

The box is `[L_l(t), H_l(t)]`, the extremes of the trailing `WIN_LONG` window
inclusive of `t`. Its width in ATR and the close's position inside it are
features A2 and A3.

Because `t ∈ I_l(t)`, we have `L_l(t) <= c[t] <= H_l(t)` by construction, so the
position is in `[0, 1]` with no clipping. (Contrast `smc_v1` row 8, whose box
comes from confirmed pivots that price is routinely outside of, and which
therefore needs an explicit clip to `[-2, 3]`.)

A **boundary touch** counts bars that came close to the box edge:

```
touch_high(t) = #{ u in I_l(t) : h[u] >= H_l(t) - TOUCH_ATR * atr_den[t] }
touch_low(t)  = #{ u in I_l(t) : l[u] <= L_l(t) + TOUCH_ATR * atr_den[t] }
```

reported as fractions of `w_l(t)` so they are bounded in `(0, 1]` and comparable
across the truncated windows of §1.2. The bar that *is* the extreme always
counts, so the fraction is never zero.

The tolerance uses `atr_den[t]` — one scale for the whole window — rather than
each bar's own contemporaneous ATR. Declared choice: a mixed-scale tolerance
would make the count depend on how volatility moved *inside* the window, which
is a second measurement smuggled into a first.

### 3.3 Boundary slopes and convergence

Over `I_l(t)`:

```
s_hi(t) = slope(h, I_l(t)) / atr_den[t]        ATR per bar
s_lo(t) = slope(l, I_l(t)) / atr_den[t]
conv(t) = s_lo(t) - s_hi(t)                    > 0 iff the boundaries converge
asym(t) = (s_hi + s_lo) / (|s_hi| + |s_lo|)    0 when the denominator is 0
```

`conv` is the continuous measure of "triangle": positive when the highs are
falling relative to the rising lows, negative for a broadening formation, near
zero for a parallel channel.

`asym` is the continuous measure of *which* triangle, and it is bounded in
`[-1, 1]` by construction:

| classical name | geometry | `asym` |
| --- | --- | --- |
| ascending triangle | flat highs, rising lows | ≈ +1 |
| symmetric triangle | falling highs, rising lows, equal magnitude | ≈ 0 |
| descending triangle | falling highs, flat lows | ≈ -1 |

One column replaces three labels and three thresholds, and it degrades
gracefully: a not-quite-ascending triangle reads 0.7 rather than flipping
between two categories.

### 3.4 The trend channel

Least squares on the closes over `I_l(t)`:

```
b(t)     = slope(c, I_l(t))
a(t)     = cbar - b(t) * xbar
r[u]     = c[u] - (a(t) + b(t) * x[u])         u in I_l(t)
sigma(t) = sqrt( mean_u r[u]^2 )               residual mean is 0 by construction
R2(t)    = 1 - sum_u r[u]^2 / sum_u (c[u]-cbar)^2      0 when the denominator is 0
```

`b / atr_den` is the channel's slope in ATR per bar, `sigma / atr_den` its
residual dispersion in ATR — jointly, "how steep" and "how wide". `R2` is the
fit quality in `[0, 1]`: a clean channel scores high, a chop scores low, and the
model gets to decide whether a steep slope with `R2 = 0.1` means the same thing
as a steep slope with `R2 = 0.9`. It is clipped into `[0, 1]` only to absorb
floating-point error; that clip cannot change a value by more than float noise
and is not a tuned constant.

Position within the channel is the standardised current residual:

```
chan_pos(t) = r[t] / sigma(t)      0 when sigma(t) <= max(1e-9*atr_den[t], 1e-12*c[t])
```

The guard is a **relative threshold, not `sigma == 0`**. `sigma` is a computed
square root, and on a perfectly-fitted window it lands near `1e-14` rather than
on zero — so an exact test misses, and the ratio of two rounding errors is a
number of order one. That would hand the model a measurement of nothing,
bounded by `sqrt(w-1)` and indistinguishable from a real reading. Nine rows of
the committed pre-Styx history do exactly this under an exact guard. The
threshold is relative to ATR because `sigma` is a price-scale quantity and
`1e-9` ATR is not a channel.

**Two floors, not one.** `atr_den` itself floors at `1e-12 * c` (§1.1), so on a
stretch where ATR has decayed toward that floor the composed threshold reaches
`1e-21 * c` — about five orders *below* the ULP of `c`, which is where the noise
actually lives. An ATR-relative test alone therefore cannot fire in the one
regime it exists for: on 200 real candles followed by 300 frozen ones, 216 of
the frozen rows still emitted a nonzero standardised residual. The absolute
`1e-12 * c` floor closes it. Neither floor changes any row of the committed
history.

`r[t]` is one of the `w_l` residuals that `sigma` is computed from, and for a
mean-zero set of `w` numbers `max|r_i| / sqrt(mean r^2) <= sqrt(w-1)`. So
`chan_pos` is **mathematically bounded** by `sqrt(WIN_LONG - 1) < 7.7` and needs
no clip. This is the general pattern in §5: bound by construction where
possible, clip nowhere.

### 3.5 Volatility contraction and expansion

Two ratios of a short-horizon measure to a long-horizon one, one built from the
range box and one from realised true range:

```
range_ratio(t) = (H_s(t) - L_s(t)) / (H_l(t) - L_l(t))       1.0 when denom <= 0
tr_ratio(t)    = mean{TR[u] : u in I_s(t)} / mean{TR[u] : u in I_l(t)}
                                                             1.0 when denom <= 0
```

Both are bounded by construction, because `I_s(t) ⊆ I_l(t)`:
`range_ratio ∈ [0, 1]`, and `tr_ratio ∈ [0, WIN_LONG/WIN_SHORT] = [0, 3]`. Both
equal exactly 1 while `w_s == w_l` during warm-up, which is the correct reading:
no contraction is measurable yet.

They are not redundant. `range_ratio` compares *extremes* and answers "has price
stopped making new highs and lows"; `tr_ratio` compares *average bar ranges* and
answers "have the candles got smaller". A drift into the middle of an old range
moves the first and not the second; a quiet grind to new highs moves the second
and not the first.

Neither is normalised by ATR, because both are already ratios of like to like —
applying `atr_den` to numerator and denominator would cancel.

### 3.6 Breakout, and failed breakout

The reference levels are the extremes of the **prior** window (§1.2), which
excludes `t`. Excluding the current candle is what makes a breakout measurable
at all: against an inclusive window, `c[t]` can never exceed `H_l(t)` by
definition and the feature would be identically zero.

```
prior_high(t) = max{ h[u] : u in I_l^-(t) }        undefined at t = 0
prior_low(t)  = min{ l[u] : u in I_l^-(t) }        undefined at t = 0

breakout_up(t) = (c[t] - prior_high(t)) / atr_den[t]        0.0 at t = 0
breakout_dn(t) = (prior_low(t) - c[t]) / atr_den[t]         0.0 at t = 0
```

Both are **signed and continuous on every row**, not events: negative inside the
range, zero at the level, positive beyond it. There is no `BREAK_ATR`-style
threshold, because a threshold would turn a measurement into a label and add a
constant. Breakout is measured on the **close**, matching `smc_v1` §3.3's choice
and for the same reason — a wick through a level that closes back inside is a
different phenomenon, and `smc_v1` already measures it as a sweep.

A **failed breakout** needs state, and it is the sharpest test of §2(c).
Conventional usage stamps the breakout candle as false once price comes back;
that stamp is future information. Here:

```
open   a pending up-break at s     iff  c[s] > prior_high(s)
       recording the level         R = prior_high(s)
replace it with a newer one        at any s' > s with c[s'] > prior_high(s')
resolve it at the first t > s      with c[t] < R
       emitting  failed_up(t) = (R - c[t]) / atr_den[t]   > 0, on that row only
expire it unresolved               once t - s > WIN_LONG
```

and symmetrically for the down side: a pending down-break opens at `c[s] <
prior_low(s)` with `R = prior_low(s)`, resolves at the first `t > s` with
`c[t] > R`, emitting `(c[t] - R) / atr_den[t]`.

`chart_failed_breakout_up_atr` is therefore zero on every row except the one
where price closed back inside, where it carries **how far** inside — a
magnitude, in the same idiom as `smc_v1`'s `smc_break_magnitude_atr`. Both
constituent events are strictly in the past at the row that reports them, which
is exactly what makes the feature causal where the textbook version is not.

The expiry costs no new constant: a pending break is dropped once the level that
defined it has rolled out of the `WIN_LONG` window, because at that point the
level is no longer part of the geometry any other feature can see. `smc_v1` §3.5
refused a lookback window on the grounds that it would be a tuned constant and
exposed staleness as a feature instead; here the window that defines the level
supplies the horizon for free, so the same reasoning arrives at an expiry
instead of a `bars_since` column.

### 3.7 Pivots, and double-top / double-bottom structure

Pivots are `smc_v1` §3.1 verbatim, restated so this document is self-contained:

```
raw_swing_high[i] = 1   iff   3 <= i <= T-1-3
                          and h[i] >  h[j]  for all j in [i-3, i-1]
                          and h[i] >= h[j]  for all j in [i+1, i+3]

raw_swing_low[i]  = 1   iff   3 <= i <= T-1-3
                          and l[i] <  l[j]  for all j in [i-3, i-1]
                          and l[i] <= l[j]  for all j in [i+1, i+3]
```

The strict-left / non-strict-right asymmetry is the plateau tie-break: on a run
of equal highs the earliest candle is the pivot, and exactly one candle is.
Confirmed at `i + 3` per §2(b). The bound `i <= T-1-3` is a statement about
*detectability*, and it is safe only because that `3` is `PIVOT_RIGHT` — see
`smc_v1` §8.2, which this spec inherits along with the definition.

At row `t`, let `i2 > i1` be the two newest indices in `S_high(t)`. Define:

```
m(t)          = min{ l[u] : u in [i1, i2] }        the intervening trough
dt_offset(t)  = (h[i2] - h[i1]) / atr_den[t]       signed
dt_trough(t)  = (min(h[i1], h[i2]) - m(t)) / atr_den[t]       >= 0
dt_neckline(t)= (c[t] - m(t)) / atr_den[t]         signed
```

and symmetrically, with `j2 > j1` the two newest indices in `S_low(t)` and
`M(t) = max{ h[u] : u in [j1, j2] }`:

```
db_offset(t)  = (l[j2] - l[j1]) / atr_den[t]       signed
db_peak(t)    = (M(t) - max(l[j1], l[j2])) / atr_den[t]       >= 0
db_neckline(t)= (M(t) - c[t]) / atr_den[t]         signed
```

All of `[i1, i2]` is `<= i2 <= t - 3`, so `m(t)` and `M(t)` are functions of the
past. No separation constant is needed: the pivot geometry forces
`i2 - i1 >= PIVOT_LEFT + 1 = 4` — if `i2 <= i1 + 3` then `h[i1] >= h[i2]` from
`raw_swing_high[i1]`'s right-hand condition and `h[i2] > h[i1]` from
`raw_swing_high[i2]`'s left-hand condition, which cannot both hold.

**What each column measures, and what "double top" is doing here.**
`dt_offset` is the continuous version of "the two tops are comparable": near
zero is a double top, clearly positive is a higher high (uptrend structure),
clearly negative is a lower high (weakening). One signed column replaces a
threshold and a binary. `dt_trough` is the depth of the valley between the two
tops, which is what separates a real double top from two adjacent wiggles, and
it is measured against the *lower* of the two tops so the shallow-second-top
case cannot inflate it.

**`dt_neckline` is the causal replacement for neckline confirmation, and the
substitution is not free.** The conventional definition waits for a close below
the trough and then declares the structure valid — retroactively, as of the
second top. This spec cannot do that and does not try. Instead it emits, on
every row, the signed ATR distance from the current close to the candidate
neckline `m(t)`: positive while price is still above it, crossing zero exactly
on the candle where price closes through. The information the conventional
definition adds *at the second top* is unavailable and stays unavailable; the
information it adds *at the break* is available and is emitted at the break.
§9 states plainly what this costs.

Both structures come with an availability flag, because `dt_offset = 0.0` is the
most double-top-like value there is and must not collide with "fewer than two
confirmed swing highs exist yet".

### 3.8 Impulse and pole

The flag and the pennant are a large directional move followed by a small
sideways one. Both halves are measurable over past bars:

```
c_at(k)     = c[ max(0, t - k) ]                   clamped at the segment start
pole(t)     = (c_at(WIN_SHORT) - c_at(WIN_LONG)) / atr_den[t]
impulse(t)  = (c[t] - c_at(WIN_SHORT)) / atr_den[t]
```

`pole` is the net displacement over the older part of the long window — the
candidate impulse — and `impulse` is the net displacement over the most recent
`WIN_SHORT` bars, the candidate flag. Both are signed and in ATR. The two
windows partition `I_l(t)` at exactly the point where §3.5's `range_ratio`
splits it, which is what makes the three columns readable together.

**No `flag_score` column is emitted.** A flag is the conjunction "large `|pole|`,
small `range_ratio`, and `impulse` small or counter to `pole`", and this spec
emits the three measurements rather than a hand-chosen product of them. The
product would be a modelling decision — which powers, which weights, which
sign convention — dressed up as a measurement, and the models in P2b exist to
make modelling decisions. A composite would also be untestable in the way the
rest of this document is testable: there is no right answer to check it against.

---

## 4. The 30 features

Column order is fixed and is part of the spec. `V` marks an availability flag.

### A. range (6)

| # | column | definition |
| --- | --- | --- |
| 1 | `chart_window_fill` | `w_l(t) / WIN_LONG` ∈ (0, 1] — how full this row's long window is |
| 2 | `chart_range_width_atr` | `(H_l - L_l) / atr_den[t]` |
| 3 | `chart_range_pos` | `(c[t] - L_l) / (H_l - L_l)` ∈ [0,1]; `0.5` when the width is 0 |
| 4 | `chart_range_pos_fast` | same over `I_s`; `0.5` when the width is 0 |
| 5 | `chart_touch_high_frac` | `touch_high(t) / w_l(t)` ∈ (0,1], §3.2 |
| 6 | `chart_touch_low_frac` | `touch_low(t) / w_l(t)` ∈ (0,1], §3.2 |

### B. compression (4)

| # | column | definition |
| --- | --- | --- |
| 7 | `chart_high_slope_atr` | `slope(h, I_l) / atr_den[t]`, §3.3 |
| 8 | `chart_low_slope_atr` | `slope(l, I_l) / atr_den[t]`, §3.3 |
| 9 | `chart_convergence_atr` | `chart_low_slope_atr - chart_high_slope_atr`, > 0 iff converging |
| 10 | `chart_slope_asymmetry` | `(s_hi + s_lo) / (\|s_hi\| + \|s_lo\|)` ∈ [-1,1]; `0` when the denominator is 0 |

### C. channel (4)

| # | column | definition |
| --- | --- | --- |
| 11 | `chart_trend_slope_atr` | `b(t) / atr_den[t]`, §3.4 |
| 12 | `chart_trend_r2` | `R2(t)` ∈ [0,1]; `0` when `sum (c-cbar)^2 == 0` |
| 13 | `chart_channel_disp_atr` | `sigma(t) / atr_den[t]` |
| 14 | `chart_channel_pos` | `r[t] / sigma(t)`; `0` when `sigma(t) <= max(1e-9*atr_den[t], 1e-12*c[t])`; bounded by `sqrt(w_l - 1)` |

### D. volatility (2)

| # | column | definition |
| --- | --- | --- |
| 15 | `chart_range_ratio` | `(H_s - L_s) / (H_l - L_l)` ∈ [0,1]; `1.0` when the denominator is 0 |
| 16 | `chart_tr_ratio` | `mean(TR over I_s) / mean(TR over I_l)` ∈ [0,3]; `1.0` when the denominator is 0 |

### E. breakout (4)

| # | column | definition |
| --- | --- | --- |
| 17 | `chart_breakout_up_atr` | `(c[t] - prior_high(t)) / atr_den[t]`; `0` at `t = 0` |
| 18 | `chart_breakout_dn_atr` | `(prior_low(t) - c[t]) / atr_den[t]`; `0` at `t = 0` |
| 19 | `chart_failed_breakout_up_atr` | `(R - c[t]) / atr_den[t]` on the resolving row, else `0`, §3.6 |
| 20 | `chart_failed_breakout_dn_atr` | `(c[t] - R) / atr_den[t]` on the resolving row, else `0`, §3.6 |

### F. patterns (10)

| # | column | definition |
| --- | --- | --- |
| 21 | `chart_dt_valid` | V: 1 iff `\|S_high(t)\| >= 2` |
| 22 | `chart_dt_offset_atr` | `(h[i2] - h[i1]) / atr_den[t]`, else 0 |
| 23 | `chart_dt_trough_atr` | `(min(h[i1], h[i2]) - m(t)) / atr_den[t]`, else 0 |
| 24 | `chart_dt_neckline_atr` | `(c[t] - m(t)) / atr_den[t]`, else 0 |
| 25 | `chart_db_valid` | V: 1 iff `\|S_low(t)\| >= 2` |
| 26 | `chart_db_offset_atr` | `(l[j2] - l[j1]) / atr_den[t]`, else 0 |
| 27 | `chart_db_peak_atr` | `(M(t) - max(l[j1], l[j2])) / atr_den[t]`, else 0 |
| 28 | `chart_db_neckline_atr` | `(M(t) - c[t]) / atr_den[t]`, else 0 |
| 29 | `chart_pole_atr` | `(c_at(WIN_SHORT) - c_at(WIN_LONG)) / atr_den[t]`, §3.8 |
| 30 | `chart_impulse_atr` | `(c[t] - c_at(WIN_SHORT)) / atr_den[t]`, §3.8 |

**Total: 30 columns**, in six named families, within the 24–32 budget of §1.4.

As in `smc_v1` §8.1, **"else 0" is per quantity, not gated on the flag**: rows
22–24 need only two confirmed swing highs and rows 26–28 only two confirmed
swing lows. A segment where one side has confirmed twice and the other has not
reports real values on the side that exists.

**There are no `bars_since_*` columns**, and their absence is deliberate. The
models see a 64-candle window of every column, so an event that happened within
the last 64 bars is already visible as a nonzero value at a known offset; a
recency column would only add information about events *older* than the window.
`smc_v1` §9(a) records what those columns cost when they are added carelessly —
`log1p(t - i)` is `0.0` both on the event candle and when the event has never
happened, a collision that its own §4 introduced availability flags to avoid.
Here the same information is carried by columns that are continuous on every
row (17, 18) or by the window itself (19, 20), and the collision cannot arise.

---

## 5. No-NaN guarantee, and why it matters

Every column is finite on **every** row of every segment, including row 0.

This is not cosmetic. `nn.data_pipeline.build_dataset` drops any row with a NaN
feature. If these columns could be NaN, the `ohlcv14`, `smc_v1` and
`chart_structure_v1` information sets would be evaluated on **different rows**,
and every comparison in P2b would be confounded by the sample universe rather
than by the information. A no-NaN engine makes the common sample universe of
`nn/information_sets.py` provable by construction instead of by hope, and
`build_information_set_views` must apply the same `np.isfinite` assertion to
these columns that it already applies to the SMC ones.

The complete list of degenerate cases and their declared values:

| condition | affected | value |
| --- | --- | --- |
| `t = 0` (`w = 1`) | 7–14 (slopes, fit, dispersion, position) | `0.0` |
| `t = 0` (prior window empty) | 17, 18 | `0.0` |
| `H_l == L_l` (flat long window) | 3 | `0.5` |
| `H_s == L_s` (flat short window) | 4 | `0.5` |
| `H_l == L_l` | 15 | `1.0` |
| `mean TR over I_l == 0` | 16 | `1.0` |
| `sum (c - cbar)^2 == 0` | 12 | `0.0` |
| `sigma(t) == 0` | 14 | `0.0` |
| `\|s_hi\| + \|s_lo\| == 0` | 10 | `0.0` |
| fewer than two confirmed swing highs | 21–24 | `0.0` |
| fewer than two confirmed swing lows | 25–28 | `0.0` |
| `t - k < 0` in §3.8 | 29, 30 | clamped to the segment's first close |
| `ATR[t] <= 1e-12 * c[t]` | all ATR-normalised | `atr_den` guard, §1.1 |

Two of these are worth stating out loud because they are collisions:
`chart_breakout_up_atr = 0.0` at `t = 0` is also what a close exactly at the
prior high would report, and `chart_range_pos = 0.5` on a flat window is also
what a mid-range close reports. Neither is ambiguous in practice, because
`chart_window_fill` identifies `t = 0` exactly (`1/60`) and a flat window forces
`chart_range_width_atr = 0`. The resolving column is named here so that a future
reader does not have to rediscover it, which is precisely what `smc_v1` §9(a)
failed to do.

**Nothing in this spec is clipped to a chosen bound.** Where a value could be
unbounded it is either bounded by construction (§3.4's `chart_channel_pos` by
`sqrt(w_l - 1)`, §3.5's ratios by 1 and 3, §4's fractions by 1) or normalised
against a reference level that is at most `WIN_LONG` bars old (17–20) or at most
one pivot spacing old (22–28). On the committed pre-Styx history consecutive
confirmed swing points are a median of 9 bars apart and never more than 46, so
the level a family-F column references is at most 49 bars old. `smc_v1` §9(c)
`smc_v1` §9(c) reports a 181-ATR distance because its equal-level references
never expire. Columns 17-20 genuinely cannot do that, because their reference is
re-derived from the window every row. **Columns 22-28 can**, and §9(f) records
it: their pivot reference has no window bound, so a collapsed `atr_den` makes the
ratio arbitrarily large
tail cannot occur here, and no clip constant is needed to prevent it.

**Warm-up** is not a NaN problem but is a real one: the first candles of a
segment genuinely have less history. They are handled the same way the OHLCV14
set handles its own indicator warm-up — the processed research dataset already
discards `FeatureSpec.warmup = 78` rows at the head of every segment, and
78 > `WIN_LONG` = 60, so **every retained row has a full long window** and every
truncated-window row is discarded before it reaches a model. The engine must
still be run over the **full** segment including those 78 candles, so that the
pivot and pending-breakout state entering the first retained row was built from
real history rather than from a truncated view.

---

## 6. What this spec does not claim

- It does not claim these features are predictive. P2b measures that.
- It does not claim these are *the* chart-pattern definitions. They are *a* set
  of exact, causal, reproducible ones, fixed in advance so that a result cannot
  be manufactured by redrawing a triangle after seeing a return.
- It does not claim to reproduce what a discretionary chartist sees. A
  least-squares line through 60 highs is not a hand-drawn trendline, and
  §3.7 is not a double top as a technician would mark one. Where the
  conventional definition needs the future, this one is strictly weaker, and §9
  says by how much.
- It does not claim the constants in §1.3 are optimal. They were not searched,
  and searching them against outer validation is forbidden.
- It does not claim 30 columns is the right number. It claims the budget in
  §1.4 is binding and that these 30 were chosen under it.

---

## 7. Public API

Pinned here so the engine and the tests that check it can be written against the
same contract rather than against each other.

```python
# nn/chart_structure.py

CHART_SPEC_VERSION: str = "chart_structure_v1"

#: The 30 columns of §4, in that exact order. Never reordered: the feature-spec
#: hash and every persisted prediction record depend on it.
CHART_FEATURE_NAMES: tuple[str, ...]

#: Family name -> its columns. The six families of §4, used by the P2b ablation:
#: "range", "compression", "channel", "volatility", "breakout", "patterns".
CHART_FEATURE_FAMILIES: dict[str, tuple[str, ...]]

@dataclass(frozen=True)
class ChartSpec:
    """The predeclared constants of §1.3.

    Frozen because they were fixed before any P2b outer-validation result was
    observed and tuning them against one is forbidden; a mutable spec would make
    "the constants were not searched" a claim about discipline rather than about
    the object recorded in the artifact.
    """

    atr_period: int = 14
    win_short: int = 20
    win_long: int = 60
    touch_atr: float = 0.25
    pivot_left: int = 3
    pivot_right: int = 3

    def to_dict(self) -> dict[str, int | float]: ...

    def spec_hash(self) -> str:
        """sha256 over canonical JSON of version + constants + feature names."""

def chart_feature_columns() -> list[str]: ...

def compute_chart_features(
    df: pd.DataFrame, spec: ChartSpec | None = None
) -> pd.DataFrame:
    """Chart-structure features for ONE contiguous segment.

    ``df`` needs ``open``/``high``/``low``/``close`` (``volume`` is accepted and
    unused). The result has ``df``'s index and exactly ``CHART_FEATURE_NAMES`` as
    columns, all float64, and contains no NaN and no infinity.
    """

def compute_chart_features_segmented(
    candles: pd.DataFrame, timeframe: str, spec: ChartSpec | None = None
) -> pd.DataFrame:
    """Split ``candles`` on market-data gaps, compute per segment, concatenate.

    Segments come from :func:`nn.data_pipeline._contiguous_segment_ids`, so the
    boundaries are the ones the processed dataset already used. State never
    crosses a boundary.
    """
```

`to_dict()` returns `dataclasses.asdict(spec)`, which keeps `atr_period`,
`win_short`, `win_long`, `pivot_left` and `pivot_right` as `int` — so the
canonical JSON contains `"win_long":60`, not `60.0`. The annotation is
`dict[str, int | float]` rather than `smc_v1` §7's `dict[str, float]` precisely
because that mismatch is `smc_v1` §9(d): an implementation that casts the
integers to float to satisfy the annotation produces a different hash. The
hash payload is

```python
json.dumps(
    {
        "version": CHART_SPEC_VERSION,
        "constants": spec.to_dict(),
        "feature_names": list(CHART_FEATURE_NAMES),
    },
    sort_keys=True,
    separators=(",", ":"),
)
```

and the reference value for the default `ChartSpec()` is
`0f62f35d87cd92abe439959976de288d72a8adbf98eab804b05e594c131e946b`.

`nn/information_sets.py` gains a `chart_v1` information set and a
`ohlcv14_plus_chart_v1` combination alongside the existing three, built the same
way — sharing the spine arrays by object identity so `prove_alignment()` keeps
proving that every cell scored the same rows.

---

## 8. Order of operations within a candle

Several rules in §3 can fire on the same candle, and which state they see
changes the output. The order is therefore part of the specification rather than
an artefact of how the loop happens to be written. At row `t`, in this sequence:

1. **absorb** the pivot at index `t - PIVOT_RIGHT`, if any — appending to
   `S_high` / `S_low` (§3.7)
2. **compute** the prior extremes `prior_high(t)`, `prior_low(t)` from
   `I_l^-(t)`, which excludes `t` (§3.6)
3. **expire** any pending breakout older than `WIN_LONG` bars (§3.6)
4. **resolve** a pending breakout against the level `R` it recorded, emitting
   `chart_failed_breakout_*_atr` — up side tested first, then the down side
   (§3.6)
5. **open or replace** a pending breakout if `c[t]` closed beyond a prior extreme
   (§3.6)
6. **compute** the window aggregates of §3.1–§3.5 and §3.8 over `I_s(t)`,
   `I_l(t)`
7. **emit** row `t`'s 30 values

Consequences that follow from this order and are part of the definition:

- A pivot confirmed at `t` immediately becomes `i2` or `j2`, so a double-top
  geometry can change on the confirmation candle itself. This is forced by
  §2(b): the confirmation candle is the first row on which the pivot exists.
- Step 4 precedes step 5, so a single candle can both resolve a pending up-break
  and open a fresh down-break. That is the correct reading of a large red candle
  that closes back through the old range and out the other side, and both
  columns fire.
- Step 4 uses the level `R` recorded when the pending break opened, not
  `prior_high(t)`. The two differ once the window has rolled, and using the
  current one would mean the "failure" was measured against a level the breakout
  never happened at.
- Step 5 replaces rather than stacks: only one pending break per side exists at
  a time, and it is always the newest. A second breakout that closes even higher
  supersedes the first, because the question "did the breakout fail" is about
  the level price is currently beyond.
- Step 2 excludes `t` while step 6's `I_l(t)` includes it, so
  `chart_breakout_up_atr` and `chart_range_pos` are computed against *different*
  windows on the same row, deliberately: one asks "did we exceed what came
  before", the other "where are we inside what we have".

---

## 9. Judgement calls, and what each one costs

Every entry here is a place where the conventional definition uses information
this spec is not allowed to use. They are recorded now, before any result, so
that a disappointing P2b outcome cannot later be explained away by claiming the
features were meant to be something else.

**(a) Double top / double bottom lose their confirmation, and that is a real
loss.** The conventional pattern is selected by its own outcome: only the tops
that were followed by a neckline break are called double tops, which is a
survivorship filter applied by the future. Dropping it means family F fires on
every pair of comparable confirmed swing highs, the great majority of which a
technician would never have marked. Concretely, this spec gives up (i) the
retroactive validity stamp at the second top, and (ii) the entire selection
effect that makes published double-top statistics look impressive. What it keeps
is the live geometry — the two tops' offset, the trough depth, and the signed
distance from the close to the neckline, which crosses zero on the candle where
a technician would confirm. If the pattern carries information, it must show up
in that geometry; if it only shows up after conditioning on the break, then what
was being measured was the break.

**(b) The flag is emitted as three measurements, not as a pattern.** §3.8 gives
the pole and the drift, §3.5 gives the compression, and no column says "flag".
The cost is that a model must discover the conjunction itself, which trees do
awkwardly and a linear model cannot do at all. The alternative was a composite
score with hand-chosen weights, which would have been a modelling decision with
no right answer to test it against, and would have hidden the three components
behind it. This is the one place where the continuous-over-categorical
preference costs the family real expressive power, and it is a deliberate trade.

**(c) Failed breakout is stamped late, by exactly the design.** The event lands
on the reclaim candle, so the feature can never mark the false high as false at
the time it printed — which is the only thing a discretionary trader wants from
it. Nothing can be done about that without reading the future. The magnitude
emitted on the reclaim candle is what remains.

**(d) Triangles are least-squares lines, not trendlines.** §3.1. A trendline
drawn through the two most extreme highs would be closer to practice and would
also be a discretionary choice of two points out of sixty; the least-squares
line is unique and continuous. The cost is that a textbook triangle whose
boundary is defined by exactly two touches registers only weakly in `s_hi`, and
a channel with one large outlier registers a slope that no chartist would draw.

**(e) Two columns are exact functions of two others.** `chart_convergence_atr`
is `chart_low_slope_atr - chart_high_slope_atr` and `chart_slope_asymmetry` is a
ratio of the same pair. A linear model gains nothing from either. They are
emitted because the P2b model set includes tree ensembles, which split on one
column at a time and cannot form a difference or a ratio across two. The
redundancy is declared, costs 2 × 64 = 128 model inputs, and is the reason
family B has four columns rather than two.

**(f) `WIN_SHORT` and `WIN_LONG` are the exposed choices.** Everything else in
§1.3 is inherited from `smc_v1`. Twelve of the thirty columns change value if
those two numbers change, and no evidence is offered that 20 and 60 are the
right ones — only that they were fixed in advance, are round, and are separated
by a factor of three so the ratios in §3.5 have room to move. If a later version
wants different windows, that is `chart_structure_v2`, declared before it is
run, not a parameter swept inside this checkpoint.

**(g) The `t = 0` collisions in §5 are resolvable but not flagged.** Rows 17 and
18 report `0.0` on a segment's first candle, the same value a close exactly at
the prior extreme would report. A dedicated availability flag would cost 64
model inputs to disambiguate 16 rows in the entire committed history — one per
segment — none of which reach a model anyway, since the 78-row warm-up of §5
discards every truncated-window row before the dataset is built.
`chart_window_fill` already identifies those rows exactly for any caller that
runs the engine outside the research pipeline. The budget of §1.4 decided this;
it is recorded because the reasoning is a cost-benefit call and not a principle,
and a future caller that skips the warm-up drop inherits the collision.

**(h) Nothing in this family looks at volume.** Classical pattern practice
leans on volume confirmation heavily (a breakout "on volume", a flag on
declining volume). It is omitted because volume normalisation across a five-year
BTC history is a separate problem with its own scale-stationarity failure modes,
and solving it badly inside this family would contaminate a measurement that is
otherwise purely geometric. `ohlcv14` carries `volume_change` and `volume_z`;
whether that is enough is a question for a later checkpoint.

**(f) Columns 22-28 are not bounded, and §5's argument does not cover them.**
Their numerator references a pivot pair with no window bound, so it can be
arbitrarily old, while the denominator is the current `atr_den`. On a
synthetic frame that passes `validate_ohlcv` — real candles followed by a long
frozen stretch, then one tick — `chart_dt_neckline_atr` reaches **360,430 ATR**,
three orders past the 181-ATR tail §5 claims cannot occur here. Latent on the
committed history, where the smallest `ATR/close` is 7.9e-4 against a break
point near 2e-7. Recorded rather than clipped: a clip constant would be a tuned
constant, which §1.3 forbids, and the honest fix is a floor on `atr_den` that
cannot fall below a real price scale. Deferred to `chart_structure_v2`.

**(g) `chart_slope_asymmetry` is three-valued in practice, not the continuous
axis §3.3 describes.** `(s_hi + s_lo) / (|s_hi| + |s_lo|)` is exactly ±1
whenever the two slopes share a sign, which over a 60-bar window is almost
always: measured on the 49,551 committed pre-Styx rows it is exactly ±1 on
**95.83%** of them and below 0.5 in absolute value on 2.04%. §3.3's claim that
"a not-quite-ascending triangle reads 0.7 rather than flipping between two
categories" is wrong as written — it does flip. It also correlates 0.985 with
`sign(chart_trend_slope_atr)`, so a flat-highs/rising-lows ascending triangle
and a plain uptrend both read +1.0, which is the distinction §3.3's own table
says the column draws. It costs 64 of the 1,920 model inputs the §1.4 budget is
fighting over and should probably be replaced in `chart_structure_v2` by
something that separates those two states.
