# ADOPTED A12 — measured slippage is evidence, not a second cash debit

Status: **ADOPTED 2026-09-07 by owner decision, pre-evidence.**

This document is a narrow governance amendment to the authoritative engineering plan
`docs/proposed_demo_implementation_master_plan.md`. It resolves an ambiguity between section
6.5's definition of slippage, section 6.6's `free_cash` line, and the fill model section 5.2
specifies, for the one fill model in which slippage is represented inside the price.

## The ambiguity

Section 6.6 states the cash line as:

```
free_cash = capital - spot_notional_at_entry - fees_paid - slippage_paid + net_funding + realised_on_closed_legs
```

Section 6.5 defines slippage as "the difference between the fill price and the mid at
decision, per fill, per leg, reported in bps and in quote currency" — a measurement of a
price, not a transfer.

Section 5.2 specifies `RecordedQuoteFillModel` as filling at
`quantize_price(ask * (1 + slippage))` for a BUY and `quantize_price(bid * (1 - slippage))`
for a SELL: the crossing to the recorded ask or bid, and the configured slippage **on top of
that crossing**, are both inside the executed price. `spot_notional_at_entry` is computed
from that executed price — the executor's own VWAP.

Read together and literally, the three sections charge the same money twice: once inside
`spot_notional_at_entry`, and again as `- slippage_paid`.

**This was genuinely ambiguous, and it is not being claimed otherwise.** Section 6.6's line
is correct for the model it was written against. The frozen historical arithmetic in
`chimera.carry.accounting` fills at `entry.spot_fill` and charges
`spot_notional * costs.spot_slippage` as a separate modelled friction on an **un-slipped**
notional; there the two terms are disjoint and subtracting both is right. The demo fills
against a recorded book, so its notional is already the slipped one. Nothing in the adopted
text said which of the two models section 6.6's line governed, and the plan predates the
build that made the question answerable.

## Resolution

For `RecordedQuoteFillModel`, and for it alone:

- the recorded bid/ask crossing is embedded in the executed fill price;
- the configured fill slippage is also embedded in the executed fill price, and therefore in
  the executor's VWAP;
- inventory basis, margin and realised PnL are computed from that executed VWAP;
- **measured slippage is an evidence and reporting quantity.** It is accumulated per leg
  and reported in quote currency;
- **measured slippage must not be debited again from `free_cash`;**
- fees remain separate, explicit cash debits — they are charged by the venue on top of the
  fill price and are not inside it;
- funding remains its own cash and equity term, under A10's sign convention;
- realised PnL is derived from the executed entry and exit prices.

Section 6.6's cash line is therefore read, for this fill model, as:

```
free_cash = capital - spot_notional_at_entry - fees_paid + net_funding + realised_on_closed_legs
```

with `spot_notional_at_entry` evaluated at the executed VWAP, and `slippage_paid` reported
beside the cash line rather than subtracted from it.

Two things section 6.5 says about slippage are **not** changed by this amendment and are
**not** claimed to be satisfied by the current build. They are recorded so the gap is
visible rather than implied away:

- 6.5 asks for slippage "reported in bps and in quote currency". No bps figure for slippage
  is emitted anywhere in `chimera/demo`; the accumulators and the `ledger_effect` block are
  quote currency only. This amendment neither adds the bps reporting nor excuses its
  absence.
- 6.5 and 5.2 define the measurement's reference as the **mid** at decision, while
  `HedgedPosition._frictions` measures against the minute's **close**. On the synthetic
  fixture the two coincide, so no test separates them. That discrepancy pre-dates this
  amendment, has no cash effect under the resolution above, and is left where it is.

## Scope, and what this amendment does not do

**This applies only to a fill model in which slippage is represented in price.** It must not
be generalised. A future fill model that charges an economically distinct, non-price slippage
cost — a separate fee, a rebate forgone, a borrow charge modelled as slippage — would have
two disjoint terms again, and section 6.6's line as originally written would be the correct
one for it. Such a model requires its own amendment; this one does not authorise it.

The historical arithmetic in `chimera.carry.accounting` is untouched and stays as it is: it
uses the disjoint-terms model, deliberately, and no P13 or P4 artifact changes.

## Why it is not cosmetic

The double debit grew monotonically with turnover and reached `risk.update_equity`, so a
campaign that traded enough would eventually have halted on a drawdown limit measured against
cash it had never spent — a phantom-equity failure arriving slowly rather than all at once.
It also put `equity` in every `ledger_effect` block, and therefore the whole reported cost and
equity series, at odds with the executed prices the same records carry.

## Classification and scope of change

This is a **pre-evidence engineering-specification disambiguation**, not a new scientific
criterion and not a change to any decision rule.

Unchanged:

- every rule, rule parameter, threshold and veto condition;
- fee rates, and the fact that fees are a cash debit;
- the funding sign convention (A10) and the separate `funding_received` / `funding_paid`
  accumulators;
- section 6.5's definition of slippage, and its reporting requirement, which this
  amendment leaves exactly as it stands (see the two gaps recorded above);
- the prospective recorder contract and its hash;
- `prospective_from = null`;
- P4 and P13 historical semantics and artifacts, and the frozen
  `chimera.carry.accounting` arithmetic;
- no model, statistic, economic result, or live route is authorised by it.

Made while `prospective_from` is `null`, before any acquisition, and while no prospective
minute existed. `CarryLedger.book_costs` already implements these semantics; this amendment
is the adopted authority they were missing, not a change of behaviour.
