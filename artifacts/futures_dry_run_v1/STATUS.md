# OPERATIONAL

## Futures Execution v1 — dry-run operational validation

**PASS** — 16 of 16 declared invariants observed, all holding.

**Evidence class: operational.** This is engineering validation of the execution
layer, produced by a deterministic in-process simulation. It is **not** evidence of
trading alpha, not evidence about real exchange execution quality, and not evidence
about any research checkpoint. Nothing here selected a model, a feature, a
threshold, a horizon or a target, and the simulated PnL below is a property of
`DeterministicFillModel` rather than of a market.

Protocol: `sha256:17a6a44d1fb5aca5eed16151c63a4b827566d664b1e1e78364357c32c6a44f4d` — frozen in
`tools/futures_dry_run.py` and `docs/futures_dry_run_validation.md` before this ran.

## Invariants

| id | held | claim |
| --- | --- | --- |
| `I02` | yes | no order or fill ever reverses a position; every close reaches flat |
| `I03` | yes | a reversal is executed as two legs, never as one oversized order |
| `I05` | yes | an Aegis veto makes execution impossible: the venue is never reached |
| `I06` | yes | a reduction succeeds while the risk engine is halted |
| `I01` | yes | no impossible order state transition is ever accepted |
| `I04` | yes | a duplicate venue event changes no position, fee, or ledger entry |
| `I07` | yes | reconciliation reports agreement when local and reported agree |
| `I08` | yes | reconciliation fails closed on disagreement: it never overwrites local state, and trading stops |
| `I09` | yes | emergency flatten reaches zero from LONG, from SHORT, from a partial fill and under a mismatch; is a recorded no-op when already flat; and is safe to repeat |
| `I10` | yes | restart recovery is correct and idempotent at every persistence boundary, and never assumes flat from an empty memory |
| `I11` | yes | funding signs are correct for all four (side, rate sign) combinations, paid and received are not netted, and a settlement is booked once |
| `I12` | yes | venue constraints fail closed: missing metadata, below minimum quantity and below minimum notional are refused rather than defaulted |
| `I13` | yes | the authenticated live-order route is unreachable, with and without the spot live-trading acknowledgement, and no credential is required |
| `I14` | yes | the required telemetry series are emitted by a full replay |
| `I15` | yes | the replay reads no row at or beyond P4-HOLD and no row at or beyond Styx |
| `I16` | yes | over the whole replay: no impossible transition appears in any order's history, no position reverses inside a single order, both LONG and SHORT exposure occur, partial fills occur, restarts recover to the position that was actually filled, a halt produces vetoes and blocks no exit, and the account ends flat |

## Descriptive metrics — not acceptance criteria

Replay: 4821 hourly candles, `2024-09-04 21:00:00+00:00` to `2025-03-24 17:00:00+00:00`,
outer block 3 of the research fold plan. Nothing below has a threshold, and nothing
below may be optimised against.

| metric | value |
| --- | --- |
| `bars` | `4821` |
| `emergency_flattens` | `2` |
| `fills` | `22` |
| `flat_bars` | `1561` |
| `funding_paid` | `25.650904420` |
| `funding_received` | `25.817651830` |
| `long_bars` | `1920` |
| `long_short_balance` | `0.588957` |
| `max_simulated_drawdown` | `89.891946720` |
| `mean_slippage_bps` | `5.000441` |
| `net_exposure_at_end` | `0` |
| `net_pnl_simulated` | `549.783089410` |
| `orders_planned` | `122` |
| `orders_rejected` | `100` |
| `orders_submitted` | `22` |
| `partial_fills` | `22` |
| `peak_gross_exposure` | `2165.16780` |
| `period_end` | `2025-03-24 17:00:00+00:00` |
| `period_start` | `2024-09-04 21:00:00+00:00` |
| `realised_pnl` | `562.2820000` |
| `reconciliation_errors` | `0` |
| `restart_recoveries` | `2` |
| `risk_vetoes` | `100` |
| `short_bars` | `1340` |
| `signals_rejected` | `0` |
| `signals_seen` | `4821` |
| `trading_fees` | `12.66565800` |
| `turnover` | `25331.31600` |

## Not covered

- sustained real-time paper operation against a live Binance USD-M feed, over days of wall-clock time. This repository has no mechanism for it and this protocol is a deterministic replay; it is recorded in docs/futures_dry_run_validation.md as a later operational requirement rather than pretended away
- real venue latency, rejection and partial-fill behaviour
- funding rates actually published by Binance; the schedule here is scripted
