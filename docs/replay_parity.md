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

Exit codes: `0` parity, `1` diverged, `2` refused (nothing to compare, or a
committed operator action the replay could not perform).

A day on which an operator ran `flatten` or `resume` is replayed with the
committed operator-action file for it (R1-j, below):

```
python -m tools.replay_parity --config conf/demo/pvc1.json --root <recorder storage root> \
    --live-log <state dir>/decision_log --days 2026-11-23 \
    --operator-actions <the committed operator-action file for the day>
```

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

`seq` is compared under R1-j's policy, as the derived `parity_seq` (below), and
the operator kinds' `operator` block is compared as well.

### `runner_now_ns`

The plan's table puts this in the right-hand column but its text says it "is
compared with tolerance zero because it is derived from recorded receipt stamps,
not the wall clock". **Tolerance zero is exact comparison**, and that is how the
tool treats it. Reading the column instead of the text would drop the single
field that demonstrates the clock is not the wall clock — the property the whole
replay design rests on.

## What may differ

**Operational records.** `STARTUP`, `SHUTDOWN`, `RECOVERY`, `HALT`,
`FEED_STALLED` and `FEED_RESUMED` are left out of the comparison: their
presence and contents are not compared, and they do not count towards
`parity_seq` (below), because "the replay may have fewer restarts". Dropping a
`STARTUP` is not a parity failure; dropping a `DECISION` is. R1-j's reading of
each:

| kind | why it is operational | what must still be reproduced |
| --- | --- | --- |
| `STARTUP` / `SHUTDOWN` | how often the process started and stopped; a replay starts once | nothing: a clean restart moves no decision state, which the next comparable record's `risk` block proves |
| `RECOVERY` | what a restart found | a crash's damaged minute is still named, by the exclusions below |
| `HALT` | a halted campaign writes one again at every restart | the halt itself: no minute is decided while it stands (a missing `DECISION` diverges), `halted` and `halt_reason` are in the next `risk` hash, and a `resume` names what it cleared in its compared `operator` block |
| `FEED_STALLED` / `FEED_RESUMED` | when the LIVE service found the feed stale; a replay reads finished files and is never stale | nothing: no minute is decided during a stall |

`RESUME` was in this set until R1-j. It is the completion record of an operator
`resume`, and is now compared with `OPERATOR`.

## R1-j: the replay parity policy

The roadmap's item, verbatim (`docs/master_roadmap_r0_r18.md`):

> **R1-j Replay parity policy:** decide `seq` (per-kind sequence or exclusion of
> operational kinds) and `OPERATOR` (deterministic replay counterpart from a
> committed operator-action file); re-include deterministic risk fields in the
> hash; the 48 h synthetic fixture reaches PARITY with a restart and an operator
> action.

### `seq`: exclusion of operational kinds

Before R1-j, section 10's table required `seq` to match byte for byte *and*
permitted the replay fewer restarts, and those two cannot both hold. `seq` is
one counter over **every** record (`DecisionLog.append` assigns it before it
knows the kind), so each extra live `STARTUP`, `SHUTDOWN`, `RECOVERY`,
`FEED_STALLED` or `FEED_RESUMED` shifted the `seq` of every record after it. A
clean restart in the middle of a run produced one `seq` divergence per later
record.

**Decision: exclusion of operational kinds** (`SEQ_POLICY`,
`chimera.parity-seq/1`).

1. **Persisted `seq` is unchanged.** It is still the record's position in the
   whole hash-chained log. `request_seq`, the chain verification, R1-c's
   history, R1-i's unfinished-request triage and the reports all read it as
   before, and old logs mean what they always meant. Nothing is renumbered.
2. **Parity compares `parity_seq`**, which is derived and never persisted: a
   record's 1-based position among its own log's comparable records (neither an
   operational kind nor in an excluded minute), in log order. It is reported
   under that name, never as `seq`.
3. **Operational records do not influence it.** A restart or a stall shifts no
   `parity_seq`.
4. **Repeated kinds within one minute stay ordered.** Records are aligned on
   `(minute, kind, ordinal)`, and each has its own `parity_seq`, so reordering
   two `FUNDING` records of one minute diverges.
5. **`OPERATOR` request/completion links stay resolvable.** `request_seq` is
   the request's persisted `seq`. The comparison resolves it in its own log and
   compares what it names: `{"parity_seq": n}` of a `requested` `OPERATOR`
   record, or `{"unresolved": seq}`.
6. **Old logs** are compared under the same rule. Their operational records
   are left out, and their raw `seq` must still rise strictly (below).

The ordering is still compared: a swapped, dropped or added comparable record
changes the `parity_seq` of the records around it. And each side's raw `seq`
must rise strictly through its records, or the record is reported as a `seq`
divergence on that side, because `parity_seq` is derived from log order.

**Rejected: a per-kind sequence.** Numbering each kind separately would also
survive restarts, but it would lose the order *between* kinds: a `FUNDING`
booked after the minute's `DECISION` instead of before it, or a
`RECONCILIATION` moved across a decision, would still match. It also has no
answer for `request_seq`, which links two records by the global counter.
**Rejected: renumbering the persisted `seq`.** It would edit the meaning of a
field that `request_seq`, the chain, R1-c and every committed log already use.

### `OPERATOR`: a replay counterpart from a committed operator-action file

An operator command is intentional state mutation, and section 9.4 puts
`OPERATOR` in the evidence set, so it is not made invisible here. Instead the
operator's actions are written down in a committed file and the replay performs
them.

**The file** (`chimera/demo/operator_actions.py`, schema
`chimera.operator-actions/1`):

```json
{
  "actions": [
    {
      "after_minute": "2026-09-20T05:59:00+00:00",
      "command": "flatten",
      "id": "r1j-48h-flatten-after-restart",
      "note": "R1-j synthetic 48 h fixture: ..."
    }
  ],
  "schema": "chimera.operator-actions/1"
}
```

* **Canonical.** The file's bytes must be `json.dumps(document, indent=2,
  sort_keys=True, ensure_ascii=True)` plus one newline. One set of actions
  therefore has one hash. The report states that hash, and every replayed
  operator record carries it.
* **When.** `after_minute` is the minute the runner had last decided when the
  operator ran the command. That is also the `minute` of the command's own
  `OPERATOR` records, so the anchor is a recorded minute, never a `seq` (which
  restarts shift) and never a wall-clock time. Several actions at one anchor are
  applied in file order. `after_minute` may not decrease through the file, and
  a command may appear only once at one anchor.
* **What.** `flatten` and `resume`, each with a non-blank note (stored exactly as
  the runner records it, stripped) and no selector: both are whole-campaign
  commands.
* **`resolve` is recognised and refused.** Every dispute it clears is a fact
  about files a crash, a disk or an operator damaged. A replay starts from an
  empty state directory, so the dispute does not exist there, and the file
  cannot pretend it does. Its selector is still checked, so zero or two
  selectors are reported as ambiguous first.
* **Refused, fail-closed:**
  * an unknown schema, command or key;
  * a missing, blank or padded note;
  * an ambiguous or impossible selector;
  * a duplicate id, or a duplicate command at one anchor;
  * decreasing order;
  * a malformed, non-UTC, `Z`-spelled or off-boundary minute;
  * a non-canonical or non-ASCII file.

**How the replay performs an action.**
* `replay_with_actions` decides the minutes up to the anchor, then calls
  `DemoRunner.apply_operator_action`.
* A live operator command always runs in a fresh process (the state
  directory's lock keeps the service out). That process's `start()` seeds the
  decision clock at the next minute's close, and the command's records carry
  that instant. So the replay first observes the same instant
  (`DemoRunner._fresh_start_instant_ns`). The clock is a maximum, so this moves
  nothing a later minute decides.
* It then calls the CLI's own `flatten` / `resume`. That method stamps
  `operator.replay_action: {id, file_hash}` on the records the action writes.
* **A halted anchor.** A runner that halts while deciding a minute leaves its
  cursor on the minute before, and a live `resume` is anchored there. So at an
  anchor that carries a `resume`, the replay must be halted too. It attempts the
  next minute, as the service did, and that attempt must halt without deciding
  the minute.
* **Nothing is skipped.** The tool refuses with exit 2 in each of these cases:
  * an action outside the replayed range;
  * an anchor the replay never reached;
  * a resume whose halt did not happen in the replay;
  * a command the runner refuses (for example `resume` on a runner that is not
    halted).

**What is compared** (`OPERATOR_KINDS`: `OPERATOR` and `RESUME`):
* **The must-match fields**, as for every comparable record. These include:
  * `minute` (the anchor);
  * `runner_now_ns`;
  * `position_after`;
  * `ledger_effect`;
  * `parity_seq`.
* **The whole `operator` block**:
  * command, note and phase;
  * `position_before` and `cleared`;
  * any `still_disputed`;
  * the request/completion link, resolved as above.
* **The one key set aside is `replay_action`.** It is required on every
  replayed operator record, and must name the supplied file's hash. It is
  forbidden on a live record, because a "live" log carrying it was written by a
  replay.
* **Missing and extra actions.** A live action the file omits is `live_only`,
  and an extra one is `replay_only`. Both name the command and phase.
* **Wrong actions.** A wrong note, anchor or command diverges or is refused.

**Known limit.** Aegis can halt in the middle of a run without a runner
`HALT`, for example on the order-rate limit. Only a restart's `start()` turns
that into a runner `HALT`. A replay never restarts, so it is not halted where
the operator resumed, and that `resume` is refused rather than performed on a
runner in another state. Such a day cannot reach `PARITY` today, and the
refusal says why.

### Re-included risk fields: `risk.state_hash` is a named policy

`risk.state_hash` used to leave out `order_times`, `cooldown_until` and `day`
as "wall clock and host date". That stopped being true with R1-e: Aegis's only
clock on the demo path is the runner's recorded decision clock
(`RunnerClock.time`). The audit's TIME-4 found that the exclusion then hid
deterministic rate-limit and cooldown state from parity.

| field | writer | clock | deterministic in replay | policy 1 | policy 2 |
| --- | --- | --- | --- | --- | --- |
| `schema`, `equity`, `peak_equity`, `day_start_equity`, `daily_pnl`, `open_positions` | `update_equity`, fills | decision clock | yes | hashed | hashed |
| `day` | `update_equity`'s day roll | decision clock (`datetime.fromtimestamp(clock())`) | yes | **excluded** | **hashed** |
| `order_times` | `record_order`; pruned by every persist | decision clock | yes | **excluded** | **hashed** |
| `cooldown_until` | `record_trade_result` (no caller on the demo path, so always `0.0`; wiring it is R1-k's) | decision clock | yes | **excluded** | **hashed** |
| `consecutive_losses` | `record_trade_result` (no caller) | none | yes | hashed | hashed |
| `halted`, `halt_reason`, `kill_switch` | `halt`, `resume`, `check_kill_switch` | none | yes, except a kill switch, which is an operator's file | hashed | hashed |
| `stale_feed_since` | `note_feed`, which has no caller on the demo path since R1-f, so always `null` | would be the operational clock | yes (constant) | hashed | hashed |
| `reconciliation_disputed` | `note_reconciliation` | none | yes | hashed | hashed |
| `funding_adverse_streak`, `funding_halt` | `note_funding_settlement`, `resume` | none | yes | hashed | hashed |

**Policy 2 excludes nothing.** Each record that states a hash now says which
policy it was taken under: `risk: {state_hash, hash_policy, decisions}`.

* `chimera.risk-hash/1` is the legacy policy. It is what a `risk` block without
  `hash_policy` means, which covers every record written before R1-j.
* `chimera.risk-hash/2` is what this build writes.

**Compatibility.** R1-c hashes `risk.json` under the policy the log's own last
statement names, never under the policy this build writes. So:

* **A legacy hash is compared with a legacy hash.** The first start of this
  build on a certified R1-i state finds no false dispute.
* **The records that follow are written under the new policy.**
* **A real mismatch still seals**, under either policy.
* **A policy this build does not know is a dispute**, never guessed.
* **R1-c's `RECOVERY` block names the policy** its two hashes were taken under.

R1-c's crash-window proofs had to learn one thing. `order_times` moves as
bookkeeping of the Aegis writes the windows already name:

* `record_order` appends the decision-clock instant of an approved order;
* every persist prunes entries 60 seconds old or more.

So each window is also tried on every order window the statement can have
held, and the full hash must still match. The candidate windows are built like
this:

* **Survivors:** a prefix of the found window, each still inside the window at
  the file's `updated_at`.
* **Appended entries:** the rest of the found window, each at or after the
  statement's clock.
* **Pruned entries:** put back in front of the survivors. They are drawn only
  from the log's own clock instants in the 60 seconds up to the statement, and
  only when the file's `updated_at` (the decision-clock instant of its last
  persist) shows they had aged out.
* **Size:** at most `max_orders_per_minute + 1` entries.

A kill after an order's persist is now proved as the `order` window, or as the
window it accompanied. Without this the R1-i crash harness's correction
scenarios sealed on an ordinary crash.

### The 48 h acceptance witness

`tests/test_r1j_parity_48h.py` (marker `replay_determinism`, run by CI's
`replay-determinism` job) builds 48 consecutive hours of synthetic recorder
files. The window runs from 2026-09-19T00:00Z to 2026-09-20T23:59Z, with
settlements through 2026-09-21T00:00Z, the close of the last minute.

The live side is `tools.demo_run`, one process per step, while the recorder's
days grow:

| process | command | minutes |
| --- | --- | --- |
| 1 | `run --once` | D1 00:00 – 11:59 |
| 2 | `run --once` (the restart) | D1 12:00 – D2 05:59, across the UTC midnight |
| 3 | `flatten --note ...` | after D2 05:59 |
| 4 | `run --once` | D2 06:00 – 23:59 |

The replay is `tools.replay_parity` with the committed
`tests/fixtures/r1j_operator_actions_48h.json`.

The required result is `PARITY`, with:

* `divergences == 0`;
* `live_only == []`;
* `replay_only == []`;
* `explained_exclusions == []`.

The same 48 hours without the committed counterpart diverge on the flatten.

**The alignment key carries an ordinal.** A minute may produce more than one
record of a kind — catching up across a settlement boundary books two `FUNDING`
settlements in one minute — and the key is `(minute, kind, n)` so each is
compared to its own counterpart. Keyed on `(minute, kind)` alone the later record
overwrote the earlier one on both sides, so the earlier one was compared to
nothing and a replay that emitted fewer of them produced no `replay_only` entry
at all. Where a minute produces one record of a kind, which is every case before
PR-10R, the ordinal is `0` and the key is the old one.

**The Aegis day.** Since R1-e Aegis rolls its trading day on the runner's
recorded decision clock, not the host's, so `day_start_equity`, `daily_pnl` and
`day` are functions of the minutes read. Since R1-j `day` is hashed too
(policy 2, above).

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
| the risk state's identity | Aegis reads only the recorded decision clock (R1-e), so `risk.state_hash` covers every field of the snapshot (policy `chimera.risk-hash/2`, R1-j); records written before R1-j name no policy and are read under the legacy one, which left out `order_times`, `cooldown_until` and `day` |

The last two were found by PR-10's determinism test and are the reason parity
holds across directories at all. The restart and the operator action are
handled by R1-j's policy above.

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
| `SKIPPED_STALE` | the skipped minute | written only by builds before R1-g, which retired the catch-up cap: the live process came back from an outage and the minute was already older than `max_catchup_minutes`. Which minutes were stale depends on when the process restarted, and no recorded file holds that. Kept so those logs still compare. |
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
