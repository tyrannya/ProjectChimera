# trading_modes_v1 — temporal operating states

Engineering, not a research checkpoint. **Nothing here claims alpha for any
mode**, nothing here selects a mode by how it has been doing, and nothing here
sends an order to a venue.

## 1. What a mode is, and what it is not

A **mode** is a trading *style*: how fast a decision is taken, and which clocks
it reads. Four operating states:

| mode | decision clock | primary clocks | context clocks | purpose |
| --- | --- | --- | --- | --- |
| `SCALPING` | `1m` | `1m`, `5m` | `15m` | seconds-to-minutes entries |
| `DAY_TRADING` | `5m` | `5m`, `15m` | `30m`, `1h` | intraday, minutes to hours |
| `SWING` | `30m` | `30m`, `1h`, `4h` | `1d` | multi-hour to multi-day |
| `FLAT` | — | — | — | no position |

A mode is **not** an instrument, a venue, a margin type, or a strategy category.
`SCALPING` is not "futures"; `SWING` is not "HODL"; none of them is
spot-versus-margin, arbitrage, or copy trading. Those are different axes and this
selector does not touch them.

**Execution for this generation is fixed and outside the mode's control:**
Binance USD-M perpetual futures, isolated margin, exactly 1x leverage, dry-run
only. No real money, no authenticated order route.

## 2. Eligibility — where a mode's right to exist comes from

> A mode is `ELIGIBLE` only when **every specialist it names has been screened
> and found viable**.

Eligibility is a fact about the **committed evidence tree**, not about anything
observed at run time. `chimera.modes.SpecialistStatus` carries `screened` and
`viable`, both read from a closed, preregistered checkpoint, and a clock absent
from the status map counts as unscreened — a specialist nobody measured is not a
specialist that passed.

### 2.1 Current eligibility: none

| clock | checkpoint | screened | viable |
| --- | --- | --- | --- |
| `1m` | P6 | yes | **no** |
| `5m` | P6 | yes | **no** |
| `15m` | P6 | yes | **no** |
| `30m` | P6 | yes | **no** |
| `1h` | P6 | yes | **no** |
| `4h` | P6-EXT | yes | **no** |
| `1d` | P6-EXT | yes | **no** |

Every clock has been screened under a preregistered gate and **none is viable**.
Therefore:

| mode | eligible | reason |
| --- | --- | --- |
| `SCALPING` | **no** | `specialist_not_viable` (`1m`, `5m`, `15m`) |
| `DAY_TRADING` | **no** | `specialist_not_viable` (`5m`, `15m`, `30m`, `1h`) |
| `SWING` | **no** | `specialist_not_viable` (`30m`, `1h`, `4h`, `1d`) |
| `FLAT` | yes | — |

**The controller can currently select only `FLAT`, and that is the correct
behaviour.** `FLAT` is a first-class successful outcome: a system with no
eligible mode is working when it holds no position. The scaffold exists so that a
mode can be expressed if a future checkpoint ever supports one — not so that one
can be run now.

Note in particular that `SWING` would be ineligible even if the machinery were
perfect, because `4h` and `1d` were screened by `P6-EXT` and failed. **Calling
30m/1h-only operation "swing" is not an option**: the mode is defined by the
clocks it names.

## 3. The decision inside an eligible mode

`chimera.consensus.decide` — the same function `P7` measured, not a second
implementation that agrees with it today. Each mode's specialist set and
agreement count:

| mode | rule | measured by |
| --- | --- | --- |
| `SCALPING` | 2 of {`1m`,`5m`,`15m`}, `15m` vetoes | **P7A — negative** |
| `DAY_TRADING` | 3 of {`5m`,`15m`,`30m`,`1h`}, `1h` vetoes | **P7B — negative** |
| `SWING` | 3 of {`30m`,`1h`,`4h`,`1d`}, `1d` vetoes | **never measured** |

`SWING`'s rule follows the same shape as the two that were measured, and **it has
never been evaluated by any checkpoint**. It is written down so the mode has a
definition, not because there is evidence for it. A future checkpoint that wanted
to measure it would preregister it first.

## 4. No automatic selection

Choosing automatically *between* eligible modes is a research question — `P8`,
[preregistered](p8_preregistration.md) and **not opened**. `chimera.modes` has no
`AUTO` value and `decide_mode` takes an operator's standing declaration, so a
scaffold cannot quietly run an unopened checkpoint in production.

**Forbidden as a selection input, permanently:** recent realised PnL, outer-fold
performance, backtest rank, post-hoc best timeframe or horizon, or any knowledge
of which mode "wins" the current period. `chimera.modes.assert_no_profit_input`
is applied by the test suite to the source of every function that can influence
which mode is entered, so an edit that introduced one fails a test rather than a
review.

## 5. Transitions

**One active directional mode at a time in v1.** A mode change with an open
position is not a re-target: the new mode's horizon and cadence are different, so
the position it would inherit is one it never chose.

`chimera.modes.plan_mode_transition` is deterministic given `(from_mode, to_mode,
position_is_flat)`:

| situation | plan |
| --- | --- |
| same mode | nothing |
| mode changed, position flat | nothing to unwind |
| mode changed, position open | **flatten, then reconcile**, before the new mode may act |

So `SCALPING LONG → DAY_TRADING LONG` is never a silent inheritance: it is
`SCALPING LONG → flatten → reconcile → DAY_TRADING LONG`.

## 6. The chain

```
timeframe specialists          (frozen, per clock)
  -> per-mode Pythia consensus (chimera.consensus.decide)
  -> mode controller           (chimera.modes.decide_mode — eligibility, never profit)
  -> Aegis                     (chimera.risk.RiskEngine — holds every veto)
  -> Hermes                    (chimera.futures.FuturesExecutor — dry-run only)
  -> Binance USD-M futures dry-run venue
  -> Argus                     (chimera.metrics — bounded mode telemetry)
```

The controller **does not bypass Aegis** and cannot. It emits a signal; every
exposure-increasing order still passes `RiskEngine.evaluate_entry`, and a veto,
an emergency flatten or a reconciliation dispute is unaffected by which mode is
active.

The controller also may not: raise leverage, switch instrument family, enable
margin borrowing, switch exchange, choose a different coin, or open two
contradictory directional modes at once. None of those is expressible — the mode
carries a clock set and nothing else, and `FuturesExecutionConfig` refuses
anything but dry-run 1x isolated at construction.

## 7. Telemetry

`chimera/metrics.py` publishes `chimera_mode_*` with **bounded labels only**: a
`TradingMode` value or a `ReasonCode` value, both enums. No free-text reason, no
order id, no symbol, no price.

`mode_selected`, `mode_eligible`, `mode_decisions_total{mode,reason}`,
`mode_transitions_total{from_mode,to_mode}`,
`mode_transition_flattens_total`, `mode_active_seconds_total`,
`mode_consensus_state_total`, `mode_risk_vetoes_total{mode,reason}`.

**Deliberately absent: any per-mode return.** A mode metric reporting how well a
mode had been doing is precisely the profit-based selection input §4 forbids.
Exposure, turnover, fees, funding, slippage and drawdown are already published,
unlabelled, by the futures family — which is sufficient while exactly one mode is
active.

## 8. Defects and open questions

1. **`SWING`'s consensus rule is unmeasured.** Its shape follows the two P7
   measured; nothing more should be read into it.
2. **Eligibility is a static table today.** `SpecialistStatus` is supplied by the
   caller and nothing yet derives it automatically from the artifact tree, so a
   future checkpoint's verdict has to be wired in deliberately. That is the
   conservative direction — a mode cannot become eligible by accident — but it is
   a manual step.
3. **No specialist serves live predictions.** P6 fitted its specialists and
   measured them; it did not persist estimators, so there is no multi-clock
   inference service. The paper runner replays frozen predictions instead, and
   says so.
4. **Mode dwell time is measured in wall-clock seconds**, which conflates a slow
   market with a slow decision cadence.
5. **One mode at a time is a v1 constraint, not a finding.** Whether two
   non-contradictory modes could run together is a question no checkpoint asked.
