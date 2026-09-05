# ADOPTED A10 — Aegis funding-cost sign correction

Status: **ADOPTED 2026-09-06 by owner decision, pre-evidence.**

This document is a narrow governance amendment to the authoritative engineering plan
`docs/proposed_demo_implementation_master_plan.md`. It supersedes exactly one contradictory
formula in section 7.2, row `funding rate (entry)`, until that plan text is mechanically folded
back into the amendment table.

## Correction

Section 7.2 currently states the side-aware funding cost as:

`-sign(side) * rate`

The correct cost paid by the position is:

`sign(side) * rate`

where `LONG = +1` and `SHORT = -1`.

The frozen execution accounting convention is:

`funding_cash_flow = -sign(side) * notional * rate`

and negative cash flow means the position paid funding. Therefore the corresponding positive
cost rate is the negation of cash-flow-per-notional:

`funding_cost_rate = sign(side) * rate`.

This yields the intended four cases:

| side | funding rate | position |
| --- | ---: | --- |
| LONG | positive | pays |
| LONG | negative | receives |
| SHORT | positive | receives |
| SHORT | negative | pays |

## Classification and scope

This is a **pre-evidence sign-convention correction**, not a new scientific criterion.
It resolves a contradiction between section 7.2's literal formula, that row's own prose
("veto an increase when the position would pay"), and the already-frozen execution accounting
sign convention.

Unchanged:

- `max_funding_cost_rate = 0.0005`;
- the veto condition and its purpose;
- every other Aegis rule;
- all recorder thresholds and semantics;
- the prospective recorder contract and its hash;
- `prospective_from = null`;
- P4/P13 historical semantics and artifacts;
- no model, statistic, economic result, or live route is authorised.

Roadmap PR-03 implements this corrected sign convention. This amendment authorises no other
PR-03 behaviour and no PR-08 implementation.
