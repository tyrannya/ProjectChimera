# Replay parity

*Section 10 of the adopted demo implementation master plan. Added by PR-11.*

## What it claims, and what it does not

Replay parity is a statement about **reproducibility**. It says:

> Running the decision logic again over the same recorded minutes produces the
> same decisions.

It says nothing whatever about whether those decisions are good, profitable, or
worth making. A campaign can be in perfect parity and lose money; parity is what
makes the decision log *evidence* rather than an anecdote, because it establishes
that the log is a function of the recorded inputs and not of the machine, the
hour, or the run.

## Running it

```
python -m tools.replay_parity \
    --config conf/demo/pvc1.json \
    --root <recorder storage root> \
    --live-log <state dir>/decision_log \
    --days 2026-11-23 2026-11-24
```

or `make demo-replay-parity DEMO_DAYS="2026-11-23 2026-11-24"`.

Exit codes: `0` parity, `1` diverged, `2` refused (nothing to compare).

The tool:

1. **copies** the recorder's normalized and funding files for the range into a
   scratch directory. A copy rather than a live read, so a recorder still
   writing cannot change what the replay sees halfway through — and so a parity
   run can never write anywhere near the recorder's own files. A test asserts
   the recorder's tree is untouched, by mtime.
2. runs `DemoRunner` in replay mode with an **empty state directory** and a
   `RunnerClock` fed from the recorded minutes, never the wall clock.
3. compares the replay's decision log with the live one.

## What must match

Section 10's list, and the tool's `MUST_MATCH` is exactly it:

`kind` · `minute` · `seq` · `runner_now_ns` · `inputs` · `rule` · `signal` ·
`requested_action` · `risk` · `execution` · `position_after` · `ledger_effect` ·
`veto_or_rejection`

Every one is perturbed one at a time by a parametrized test, so a field that was
named in the list but never actually read by the comparison would fail the
suite. A must-match list the tool does not honour is a promise nobody keeps.

### `runner_now_ns`

The plan's table puts this in the right-hand column but its text says it "is
compared with tolerance zero because it is derived from recorded receipt stamps,
not the wall clock". **Tolerance zero is exact comparison**, and that is how the
tool treats it. Reading the column instead of the text would drop the single
field that demonstrates the clock is not the wall clock — the property the whole
replay design rests on.

## What may differ

**Operational records.** `STARTUP`, `SHUTDOWN`, `RECOVERY`, `HALT` and `RESUME`
are aligned by `(minute, kind, ordinal)` and their contents are not compared,
because "the replay may have fewer restarts". Dropping a `STARTUP` is not a
parity failure; dropping a `DECISION` is.

**Everything else is compared, including the kinds PR-10R made reachable.**
`FUNDING`, `RECONCILIATION`, `LIQUIDATION_TOUCH`, `SKIPPED_STALE` and
`INCOMPLETE_STATE` all go through the full must-match comparison. None of them is
exempted to make a run green: a replay that booked a different funding flow,
reconciled to a different outcome, or missed a settlement diverges.

**The alignment key carries an ordinal.** A minute may produce more than one
record of a kind — catching up across a settlement boundary books two `FUNDING`
settlements in one minute — and the key is `(minute, kind, n)` so each is
compared to its own counterpart. Keyed on `(minute, kind)` alone the later record
overwrote the earlier one on both sides, so the earlier one was compared to
nothing and a replay that emitted fewer of them produced no `replay_only` entry
at all. Where a minute produces one record of a kind, which is every case before
PR-10R, the ordinal is `0` and the key is the old one.

**The environment.** If the two runs declare a different `software.python` or
`software.libs`, the whole run is labelled **`ENVIRONMENT_PARITY`** and reported
separately. It is not counted as a plain pass, and the difference is printed. A
parity result that quietly absorbed an environment change would be answering a
different question from the one asked.

## Why it is deterministic

Nothing on the decision path reads a clock, a random number, or an unordered
collection:

| source of nondeterminism | how it is removed |
| --- | --- |
| wall clock | `RunnerClock` is `max(receipt_ns)` over what was read; it reads no clock at all |
| floats in money | `Decimal` throughout fills, fees, funding and the ledger |
| Aegis's floats | it receives the same float inputs in replay, so its decisions replay exactly |
| dict iteration order | canonical JSON sorts keys; the rule registry preserves registration order |
| order ids | sequence-based, per executor |
| event ids | deterministic in the dry-run venue |
| the config's identity | `config_hash` excludes paths, so two hosts and a scratch directory hash alike |
| the risk state's identity | `risk.state_hash` excludes `order_times`, `cooldown_until` and `day`, which are wall clock and host date rather than decision semantics |

The last two were found by PR-10's determinism test and are the reason parity
holds across directories at all.

## On a divergence

The tool prints the **first** divergent record and both versions, and stops
describing. One upstream cause produces hundreds of downstream differences, and
a wall of them hides the one that matters.

A divergence is never repaired by the tool. Section 10's failure criterion is
"any mismatch in a must-match field", and a parity tool that could paper over
one would be worse than no tool at all.

**Explained exclusions.** Section 10 permits one divergence to be explained away:
a minute the runner started from `LOG_BEHIND_STATE` "is explained and excluded
once". PR-10R makes that executable and gives it a second, identical case.

The tool excludes a **minute** — never a record kind — when the *live* log says
that minute was not one the campaign decided from its files:

| live record | minute excluded | why a replay cannot reproduce it |
| --- | --- | --- |
| `SKIPPED_STALE` | the skipped minute | the live process came back from an outage and the minute was already older than `max_catchup_minutes`. Which minutes were stale depends on when the process restarted, and no recorded file holds that. |
| `RECOVERY` | `recovery.evidence_excluded_minute` | section 9.3: the minute a crash left inconsistent "is excluded from the campaign's evidence and counted in the monthly report". |

Every exclusion is reported. `explained_exclusions` lists each excluded minute
with its reason, in the JSON and in the printed summary, and the count is printed
even when it is zero. A run with no restarts and no crashes excludes nothing, and
the 48-hour synthetic acceptance is such a run — so its parity result is an exact
comparison of every record, not a comparison with something taken out of it.

## Scope

PR-11 provides the tool, this document and the fixture test. The **soak-window**
requirement — S4's "zero unexplained divergences over the full soak window" — is
a campaign activity, not a test, and belongs to the operating stage rather than
to this PR.
