# Demo campaign runbook

How to start, watch, halt, resume and report on the demo deployment — the
prospective recorder plus the dry-run campaign runner — and what an operator is
not allowed to do while a campaign is running.

Every command in this document is one this repository supports at the commit it
was written against. `tests/test_demo_runbook.py` feeds each of them to the
argparse parser of the tool it names, and checks each `make` target against the
`Makefile`, so a command that stops existing fails a test rather than failing an
operator at 03:00.

**No research result appears anywhere in this document, and none can.** The
demo campaign answers no research question. P8 was withdrawn and has no result.
P13 is preregistered and unrun. The prospective protocol PVC-1 has not been
written, so nothing here has been evaluated against it and no number produced by
these procedures is evidence for or against any hypothesis. `make
verify-research-state` is what checks that claim against the repository, and it
runs in CI.

## 0. Three things that are true today, before you follow any of this

**0.1 The committed campaign configuration cannot start a campaign, and must not
be edited so that it can.** `conf/demo/pvc1.json` carries `"protocol_hash":
null` and therefore carries no `rules` block — the config schema refuses rule
parameters in a `CAMPAIGN` profile until the protocol that froze them is named
by hash. `tools/demo_run.py` builds `CarryParams.from_config(...)` for
`R1_carry` and that raises on an empty mapping. So `make demo-run` against the
committed file stops before the runner exists. That is the correct outcome:
choosing rule parameters before the protocol is frozen is choosing them from
recorded data.

This is not only `run`. `status`, `flatten`, `resume` and `resolve` all build the
runner through the same `_load(...)`, so against the committed file every one of
them exits with the same `RuleError` naming the eight absent `R1_carry`
parameters — including the kill-switch procedure in section 8. Read that error as
"no campaign is configured", not as a fault. `tools/demo_report.py --day` is the
exception and works against any configuration, because a report reads persisted
files and never constructs a runner.

The `demo` container in `docker-compose.yml` is subject to the same thing: it is
part of the default stack because it is part of the demo deployment, but until a
campaign configuration with frozen parameters exists it exits on start and
`restart: unless-stopped` retries it. Before the protocol is frozen, start
`recorder prometheus grafana alertmanager` and leave `demo` out.

The sections below are the procedure the campaign will follow once the protocol
is preregistered and both the hash and the parameters are committed; until then
they are exercised against a `TEST` or `SOAK` configuration the operator writes
and does not commit.

**0.2 A `CAMPAIGN` run stops at `SELF_CHECK` on this build.**
`tools/demo_run.py::_software()` cannot establish the source identity — it calls
`nn.source_identity.source_identity()` with no argument where the function
requires a root, and then reads attributes off what is in fact a dict — so both
failures fall into its `except Exception` and the block it returns carries
`"dirty": true`. `DemoRunner.self_check` refuses a run with `dirty` unless
`--allow-dirty`, and `--allow-dirty` is refused on a `CAMPAIGN` profile. The
refusal is right; the reason it fires is a defect. Repairing it belongs to the
change that owns the runner, not to this document, which records it so an
operator is not left guessing at a halt whose cause is a bug in the identity
block rather than a dirty working tree.

**0.3 `tools.demo_run run` is one bounded catch-up pass, not a daemon.** It
processes at most `max_catchup_minutes` minutes — 3 by default — closes the log
and exits 0. Supervision is what re-invokes it: `deploy/systemd/chimera-demo.service`
uses `Restart=always` with `RestartSec=30s` as the interval between passes, and
the compose service restarts for the same reason. Two consequences an operator
must not mistake for faults:

* the metrics endpoint on 9103 exists only while a pass is in flight, so
  `up{job="demo"}` is 0 between passes and the `RunnerDown` alert in
  `conf/alerts_demo.yml` cannot be read as liveness on this build;
* `systemctl status chimera-demo` shows the unit inactive between passes.

A third, worth reading before you trust a funding number. The runner does now
settle funding: it books each recorded settlement in
`funding/um/settlements.ndjson` that falls inside the window
`open_instant < settlement <= now`, once and only once, and writes one `FUNDING`
record per settlement. `CarryLedger.funding_paid`, `funding_received` and
`chimera_demo_funding_adverse_streak` all move, and `FundingAdverseStreak` can
fire. What remains true of the panel is narrower and still worth knowing: the
telemetry pre-creates both `direction` children at 0 so `rate()` and `increase()`
are defined from the first scrape, so **a flat zero on that panel does not
distinguish "no settlement has fallen inside this position's window yet" from "the
position was flat across every settlement so far".** Neither does the daily
report: its `funding` block carries `net`, `paid`, `received`, a count of
settlement minutes and a count of `FUNDING` records, and both cases produce zeros
in all of them. Telling them apart means reading the recorder's
`settlements.ndjson` against the position's own open and flat intervals, and
nothing in this build does that for you.

The per-settlement detail -- settlement id, rate, mark price, quantity, notional,
signed cash flow and direction -- is in the **`FUNDING` records of the decision
log**, one per settlement, not in the daily report.

Section 8.1 of the adopted plan describes a continuous `READY` loop and the
runner has the state machine for one; the CLI does not run it. That is a runner
gap, recorded here rather than hidden behind a restart policy that makes a
bounded pass look like a service.

A fourth, about which profile can reach any of this. `tools/demo_run.py::_software`
calls `nn.source_identity.source_identity()` with no argument, and that function
requires a `root`. The `TypeError` is caught by the surrounding
`except Exception`, which reports `dirty: True` — so **every** run declares a
dirty tree, however clean the checkout is. `SELF_CHECK` then refuses a `CAMPAIGN`
profile, and `--allow-dirty` is itself refused for `CAMPAIGN`, so on that profile
the runner reaches `HALT` and no `DECISION`, `FUNDING`, `RECONCILIATION`,
`LIQUIDATION_TOUCH`, `SKIPPED_STALE` or `INCOMPLETE_STATE` record is ever
written. On `CAMPAIGN`, `STARTUP` and `HALT` are the whole set; the other ten
kinds are reachable on `SOAK` and `TEST`, and there only with `--allow-dirty`,
because `self_check` refuses a dirty tree on **every** profile and this bug makes
every tree read as dirty.

The remedy is **not** just passing the argument. `source_identity` returns a
`dict`, and `_software` reads it with `getattr(identity, "revision", "")` and
`getattr(identity, "dirty", False)` — attribute access on a mapping, which yields
the defaults. Add the argument alone and every run would record an empty revision
and `dirty: False`, so a campaign on a genuinely dirty tree would pass
`SELF_CHECK` unchallenged. That is worse than the halt it replaces. The fix is
the argument **and** subscript access, and it belongs to the change that owns
`tools/demo_run.py`.

A fifth, about the disputes an operator cannot clear — which is all of them.
`resolve` is scoped to a leg's `RECONCILIATION` dispute and, as section 6 records,
cannot run from the CLI at all in this build. `stale_leg`,
`ledger_store_mismatch`, `funding_booking_torn`, `asymmetric_close`,
`ledger_unreadable`, `ledger_capital_mismatch`, `{leg}_ledger_regressed` and
`{leg}_store_unreadable` have no command that ends them either. **There is no supported way to clear
them, and editing the state files by hand is not one this runbook offers** — see
section 6 for why a safety flag cleared without its record is worse than a
campaign left halted. That is narrower than it was: `resolve`
used to clear whatever the carry ledger was disputing, which set a flag and fixed
nothing — a torn funding booking stayed unbooked and the cash stayed short while
the campaign resumed on a ledger it had been told to distrust. Refusing is the
safer half of the fix; the other half, a `resolve` that actually re-books, is not
in this change.

A ninth, and read this one before comparing a report against a hand
calculation. **Slippage is measured, not spent.** The campaign's equity is
`equity = capital - fees + realised + net_funding + unrealised`, with no
slippage term, and `ledger.slippage` is an attribution the reports show beside
it. (`ledger.realised` is the two legs' price realisation only; funding reaches
`free_cash` through `book_funding` and is its own term here.) Section 6.5 defines slippage as "the difference between the fill
price and the mid at decision" — a measurement of a price — and in this build
that difference is already inside the price everything else is computed from:
`RecordedQuoteFillModel` crosses to the recorded touch and applies the
configured slippage on top of it, so the executor's VWAP is the slipped price,
and the inventory, the margin and the realised PnL are all valued at it.

Section 6.6's cash line does read `- slippage_paid`, and it is correct for the
model it was written for: the frozen arithmetic in `chimera/carry/accounting.py`
fills at an un-slipped price and charges a separate modelled slippage rate, so
there the two terms are disjoint. They are not disjoint here. Deducting the
measured slippage as well charged the crossing twice, and the error grew with
turnover and reached `risk.update_equity` — a campaign that traded enough would
have halted on a drawdown limit against cash it never spent. If you are
reconciling a report to section 6.6 by hand, use the equity line above.

An eighth, about what a dispute does and does not cost. The carry ledger books
each leg at its own level — the spot inventory at the spot leg's VWAP, the
perpetual's 1x margin at the perpetual's — and moves `free_cash` by the change
in those levels. So `asymmetric_close` no longer means "no cash was returned":
the leg that really did close returns its own principal or margin, the leg still
holding keeps its own, both legs' fees, realised PnL and slippage are recorded
either way, and the position is disputed on top because one hedged quantity
cannot describe a half-closed pair. Reading `quantity: 0` beside a non-zero
`spot_principal` is that state, and it is the honest one: the hedge is gone and
the spot leg is not.

`{leg}_ledger_regressed` is new and should never be seen. It means an executor
reported LESS cumulative fees than the carry ledger has already booked for that
leg, which fees cannot do — `Ledger.book_fee` refuses a negative — so it means
that executor's accumulators were reset underneath the ledger
(`FuturesStore.adopt_after_unreadable` is the one thing in the tree that does
it, and no CLI reaches it). Which history is the real one is an operator
judgement and no command records it, so the campaign stops rather than crediting
itself the difference — and it stays stopped.

A sixth, for anyone reading a `PARTIAL`. `HedgedPosition.correct()` implements
section 6.3's correction policy — a bounded retry, then a
`HEDGE_CORRECTION` flatten — and the runner never calls it. A position left with
one leg filled is retried implicitly by the next minute's `plan()` at a re-sized
quantity, with no correction timeout and no cap on how long it stays one-legged;
`liquidation_touched` reads `min(spot, perp)`, which is zero for such a position,
so it is not liquidation-checked either. Flatten it by hand (section 7) rather
than waiting for a timeout that does not exist.

A seventh, about one halt reason you will not see. `CarryLedger.check_identity`
compares `Q x (entry_basis - current_basis)` against `spot_pnl + perp_pnl`, and
both sides are computed from the same ledger fields, never from the legs' stores.
The residual is non-zero only when `entry_basis` disagrees with
`perp_entry - spot_entry`, and nothing in this build can make it disagree:
`book_position` writes all three together and clears them together.
A test can force it by assigning `perp_entry` directly
(`tests/test_carry_ledger.py::test_an_identity_violation_disputes`), which is why
the rule is not dead code -- but for a ledger this build wrote,
`identity_violation` is unreachable. It cannot detect "a leg that is wrong by a
fill", which is what section 6.5 introduces it for, so its absence is not
evidence that the legs agree.

## 1. Preconditions

```
make recorder-preflight
```

Can this network actually receive every stream the recorder subscribes to? It
imports no recorder code and writes nothing.

Then read the campaign configuration you intend to run and confirm that it names
the campaign you mean and that its `protocol_hash` is not `null`. A configuration
whose hash is null is section 0.1's case and there is nothing to start.

## 2. Start

Directly:

```
python -m tools.recorder --base-dir data --log-level INFO run --metrics-port 9102
python -m tools.demo_run --config conf/demo/pvc1.json --root data run --metrics-port 9103
```

Supervised:

```
sudo systemctl start chimera-recorder chimera-demo
```

Containerised:

```
docker compose up -d recorder demo prometheus grafana alertmanager
```

Confirm on `http://127.0.0.1:9103/metrics` that `chimera_demo_up` is 1 and that
`chimera_demo_state{state="READY"}` is 1 while a pass is in flight, and on
`http://127.0.0.1:9102/metrics` that `chimera_recorder_up` is 1 for every
stream. Both ports are bound on all interfaces by the Prometheus client and must
be firewalled to the Prometheus host; the compose file publishes them on
127.0.0.1 only.

## 3. Stop

```
sudo systemctl stop chimera-demo
docker compose stop demo
```

The unit sends `SIGTERM`. A stop between the state write and the log write is
the recoverable case the runner's `SELF_CHECK` is built for; a stop in the
middle of a record is not reachable, because `DecisionLog.append` fsyncs each
record before it returns.

## 4. Clean restart

Stop, read what the runner believes, start again:

```
sudo systemctl stop chimera-demo
python -m tools.demo_run --config conf/demo/pvc1.json --root data status
sudo systemctl start chimera-demo
```

`status` prints `last_minute_processed` and `last_record_hash`, which is what
the next start will continue from.

**Do not delete `state/demo/runner_state.json` to "clean" a restart.** The
runner refuses an unreadable state file rather than starting from an empty
cursor, and starting from an empty cursor would reprocess minutes that already
have records — two records for one minute, in an append-only log that cannot
withdraw either.

## 5. Recovery verification: the `STARTUP` and `RECOVER` checks

```
python -m tools.demo_report --config conf/demo/pvc1.json --day 2026-09-19
```

Read, in the JSON:

* `chain.ok` is `true` and `chain.faults` is empty;
* `records_by_kind.STARTUP` went up by exactly one for this start;
* `halts.recoveries` is what you expect — one after an unclean stop, zero after
  a clean one;
* `input_coverage.by_kind` for the kinds you are reading. On this build
  `kinds_the_runner_can_write` lists all twelve, so every zero means "it did not
  happen"; the block is what lets you check that rather than assume it, and on a
  build where a path went unreachable again it is what would say so.

A torn tail shows as a fault in `chain.faults`. **The report does not repair
it.** `chimera.demo.decision_log.recover_tail` is the only repair in this
codebase and the runner performs it on start; a report that repaired its own
input would be changing the evidence it was asked to describe.

## 6. Reconciliation dispute

`ReconciliationMismatch` fires. Read the day first:

```
python -m tools.demo_report --config conf/demo/pvc1.json --day 2026-09-19 --markdown
```

`reconciliation.dispute_halts` counts the halts whose cause collapsed to
`dispute`, and `halts.events` carries each halt's `detail` verbatim.

Then inspect both stores and the two states that carry a dispute of their own.
The daily report is computed from the decision log alone, on purpose, so the
current contents of these files are a separate question and you have to look:

```
python -m json.tool state/demo/spot_store.json
python -m json.tool state/demo/perp_store.json
python -m json.tool state/demo/carry_ledger.json
python -m json.tool state/demo/risk.json
```

`FuturesState.disputed` maps symbol to description in each store,
`CarryLedgerState.disputed` and `.resolutions` are in the ledger, and
`RiskState.reconciliation_disputed` is in `risk.json`. Only when you have read
both legs' stories and can say which one is right:

```
python -m tools.demo_run --config conf/demo/pvc1.json --root data resolve \
    --symbol "BTC/USDT:USDT" \
    --note "venue statement matched the local store at 14:07Z; ticket OPS-118"
```

The note is mandatory in three separate places and it is evidence, not
decoration: it is written into an `OPERATOR` record and it appears in the daily
report under `reconciliation.operator_resolutions`. A note that does not say
what was checked is a resolution nobody can audit.

> **In this build that command always refuses, and no operator path clears a
> reconciliation dispute.** `tools/demo_run.py` constructs the runner and calls
> `resolve` without `start()`, so the runner clock has observed nothing;
> `resolve` checks that it can write the `OPERATOR` record before it changes
> anything, and refuses with *"resolve cannot be recorded"*. That refusal is the
> correct half — the alternative, which this build shipped until it was caught,
> was clearing the store dispute and Aegis's copy and *then* dying, leaving the
> safety state changed with nothing in the log to say who changed it. But it
> leaves the command unusable.
>
> **There is no supported way to clear a reconciliation dispute in this build,
> and this is an open S3 blocker.** Editing `spot_store.json` /
> `perp_store.json` and `risk.json` by hand would clear the flags, and it is
> **not** a procedure this runbook offers: section 8.3 requires that a change to
> the safety state carry the record naming who changed it and why, and a hand
> edit changes that state with nothing in the decision log at all — strictly
> worse than the refusal, and worse than leaving the campaign halted. A halted
> campaign with a durable, explained dispute is a recoverable situation; a
> campaign whose safety flags were edited by hand is not evidence any more.
>
> The fix is to seed the clock from the **log's tail** rather than from the state
> file, which is a change with its own crash matrix: an earlier attempt seeded it
> from the state file instead and made `start()` raise on the crash section 9.3
> exists to recover from. It is recorded here rather than improvised, and it is
> tracked with the `resume` blocker in section 7 as one piece of work.

### If `carry_ledger.json` itself cannot be read

`CarryLedger.open` disputes and leaves the damaged bytes exactly as it found
them; `save` then refuses to overwrite them. The position is `DISPUTED` and the
campaign is halted, which is correct: that file is the only record of what the
position did, and a ledger that resets itself is indistinguishable from one that
never traded.

`flatten` still works and still reduces exposure — a corrupt file is no reason to
leave a real position standing — but the `OPERATOR` record it writes carries **no
`ledger_effect` block**, because the runner is holding a placeholder and any
economics built from it would be invented. That omission is the correct
behaviour and not a missing field. The day's report is still produced; it simply
carries the last economics a real ledger held.

Repair means **restoring the whole file from a good copy**, with the damaged
bytes preserved alongside it under a different name and an incident recorded
(section 15). That is a different act from editing a safety flag by hand: it puts
back a file the campaign itself wrote, rather than asserting a state no record
supports.

**Not any copy, and this is the part that decides whether the campaign can be
started again at all.** SELF_CHECK refuses a ledger holding less cumulative
`fees`, `slippage`, `funding_paid_total` or `funding_received_total` than the
decision log has already committed (`ledger_behind_log`). Those accumulators
cannot fall, and every ledger save precedes the record that quotes it, so no
crash produces a ledger behind the log — it means the file was deleted,
truncated, or replaced by an older one. Deleting it is refused by the same rule
once the campaign has booked a nonzero fee, slippage or funding amount — the
guard compares those accumulators, so a campaign whose committed blocks are all
zero has nothing to be behind, and one that has paid nothing loses nothing by
starting its ledger again.

So the copy has to be **at least as recent as the log's last `ledger_effect` and
its last `FUNDING` record**. A backup older than those is refused again, with no
dispute for any operator command to clear, and **the campaign cannot be
started** — the daily backup that predates this morning's settlement is not a
usable restore point. That is the fail-closed contract working as designed, and
it is also a real operational limit: back the state directory up **with** the
decision log and at the same instant, or accept that a campaign whose ledger is
lost is a campaign that ends.

While the ledger is in that state `flatten` still reduces exposure, but writes
neither a ledger nor a `ledger_effect` — a file that cannot speak for the
campaign must not be allowed to start speaking for it through the emergency
path. `resume` refuses too, for the same reason and with the same remedy: an
operator note cannot make the file hold what it does not hold, and clearing the
halt without repairing the ledger used to end in a traceback rather than a
refusal. That applies to a **restored older copy** exactly as it does to a
deleted one: the copy loads normally, so nothing about the file itself says it is stale,
and only the comparison against the log does. Without that, one `flatten` would
persist the stale file, quote it into an `OPERATOR` record, and leave the log's
own cumulative slippage running backwards — which an append-only evidence log
cannot take back.

**One thing that will mislead you while you work through this.** Aegis persists
the halt in `risk.json`, and it is not cleared by repairing whatever caused it.
So a later CLI start still announces *"Starting in HALTED state from risk.json"*
with the ORIGINAL halt text — after the ledger has been restored correctly, after
the cause is gone. Read that line as "a halt is on disk", never as "the halt you
just fixed is still true": the authority on the current cause is the SELF_CHECK
result of the run you are looking at, not the sentence Aegis carried forward.
This is a reporting defect in the operator lifecycle, recorded here because
following section 6 is exactly when you meet it, and repairing it belongs to the
change that owns the lifecycle.

## 7. Kill switch, flatten, resume

```
touch state/demo/KILL_SWITCH
```

**`state/demo/KILL_SWITCH`, not `user_data/KILL_SWITCH`.** The path in section
11.5 of the adopted plan is `chimera/risk.py`'s default, which is the Freqtrade
path. `tools/demo_run.py` constructs the demo's `RiskEngine` with
`kill_switch_path = <state_dir>/KILL_SWITCH`, so the demo's switch is under the
runner's own state directory. Mechanical correction, recorded here rather than
left as a trap.

Confirm `RunnerHalted` fires. If the position has to come down:

```
python -m tools.demo_run --config conf/demo/pvc1.json --root data flatten \
    --note "kill switch pulled during the 03:10Z venue incident; removing exposure"
```

To come back, in this order:

```
rm state/demo/KILL_SWITCH
python -m tools.demo_run --config conf/demo/pvc1.json --root data resume \
    --note "venue incident closed; both stores reconciled; feed ages under 30s"
```

The kill-switch check is level-triggered: while the file is there every check
re-asserts the halt, so the file must be gone before `resume` is attempted.
`resume` refuses an empty note and refuses to run when the runner is not in
`HALT`.

> **In this build that second command always refuses, and there is no supported
> operator path out of a persisted HALT. This is an open S3 blocker.**
>
> `resume` refuses when the runner is not in `HALT`, and a runner reached
> through the CLI's **`resume`, `resolve` and `status`** paths never is. Those
> three construct a `DemoRunner` and act on it without calling `start()`, so the
> object is in its constructor's `STARTUP` state; the halt is on **disk**, in
> `risk.json`, and nothing on those paths reads it back into the runner. (`run`
> and `flatten` do call `start()` and do reach `HALT` — the gap is not that the
> CLI can never be halted, it is that the commands which exist to LEAVE a halt
> are the ones that never look.) Observed end to end: after a kill-switch halt
> and a clean shutdown, `risk.json` holds `halted: true` with `halt_reason:
> kill_switch`, and a fresh process reports `STARTUP` and answers *"the runner
> is not halted; there is nothing to resume from"*. Removing the kill-switch
> file first does not change it — the refusal is about the runner's own state,
> not the switch.
>
> **The repair is smaller than it looks, and the direction matters.** Calling
> `start()` first is not a second obstacle, it is the missing step: measured on
> the same fixture, `start()` returns `HALT` and `resume()` then **succeeds** —
> the runner reaches `READY` and `risk.json`'s `halted` flips back to false. What
> is missing is that the CLI's resume path adopts the persisted campaign state
> before it decides whether there is anything to resume from. It cannot simply be
> `start()` as it stands, because on a `CAMPAIGN` profile section 0.2's
> `_software()` defect makes `start()` halt on `source_identity` first, and
> `resume` would then be clearing a halt whose recorded cause is a bug rather
> than the operator's. Both have to be fixed together.
>
> The same shape blocks `resolve` (section 6), for a different reason on the
> same path. Between them, a campaign that halts cannot be returned to `READY`
> by any documented command.
>
> **Do not hand-edit `risk.json` to work around this.** It appears to work and
> it destroys the audit trail: section 8.3 requires that a change to the safety
> state be accompanied by the record that says who made it and why, and an
> operator who clears `halted` by hand has changed the safety state with nothing
> in the decision log at all — strictly worse than the refusal. If a campaign is
> halted and must be stopped, stop it; the halt, its cause and its evidence are
> already durable.
>
> Closing this is a runtime change with its own crash matrix — the durable halt
> must not be cleared before the `RESUME` record is provably writable — and it
> is tracked as its own piece of work rather than improvised here.

### What is on disk after a dispute halt, and what is not

A halt that follows a booking persists the carry ledger **before** it writes the
`HALT` record, so the file and the in-memory ledger agree at the moment the
process stops. That ordering is deliberate and is not a property of a clean
shutdown: a process killed at the halt boundary leaves the same bytes behind.
It matters most for slippage, which no executor accumulates — fees and realised
PnL would be re-derived from the executors on the next start, and slippage would
simply be gone.

The reverse is also true and is the reason the `HALT` record itself carries no
`ledger_effect` on most paths: most halts run before the minute's
`mark_to_market`, so there is no equity to report and the record says nothing
rather than something unbacked. A `HALT` without a `ledger_effect` block is
normal. A `ledger_effect` that no file backs would not be.

## 8. Disk low

`DiskLow` fires. Find out which filesystem — the alert carries the firing
series, so `chimera_recorder_disk_free_bytes` and `chimera_demo_disk_free_bytes`
name the recorder's data directory and the runner's state directory
respectively.

```
df -h data state/demo
```

**Never delete a normalized day or a decision-log day file to make room.** A
coverage day that vanished is a hole in the coverage gate that no later run can
fill, and a log day that vanished breaks the day-to-day hash link the chain
verifier walks — every record after the hole becomes unverifiable. Add capacity,
or move the archive copies to another filesystem.

## 9. Data gaps

`DataGapToday` fires.

```
python -m tools.recorder --base-dir data status --json
```

A gap is recorded, never filled. The runner writes an `INCOMPLETE_STATE` record
for a minute it could not read completely and decides nothing on it; the daily
report lists those under `minutes.incomplete_detail` and counts them in
`minutes.incomplete`.

**Restarting the recorder to "fix" coverage is a change to the campaign.** It is
recorded as an incident (section 15) before it is done, or it is not done.

## 10. Stale feed

`RecorderStreamStale` fires. Look at `chimera_recorder_up{stream=...}` and
`chimera_recorder_reconnects_total` to tell a reconnecting stream from a dead
one.

A stale feed is a veto, not a halt: the runner keeps ticking and refuses to
increase exposure while the data is old. Nothing needs an operator unless the
recorder process itself is down, which is `RecorderDown`.

## 11. Process heartbeat

Three different questions, three different series, and they are routinely
confused:

* `chimera_recorder_heartbeat_timestamp` and `chimera_demo_heartbeat_timestamp`
  are wall clock at the last state change. They answer "is the process alive".
  The recorder also writes `health/heartbeat.json` under its storage root every
  30 seconds, which survives the process and is what to read after a crash.
* `chimera_demo_last_minute_age_seconds` answers "how far behind the data is the
  runner", which is what rises during a catch-up and is not a liveness problem.
* `chimera_demo_feed_age_seconds{market=...}` answers "how old is the newest
  minute available for this market", which is the recorder's problem, not the
  runner's.

Read section 0.3 before concluding that a demo heartbeat gap means the runner
died.

## 12. Daily reporting

```
make demo-report DEMO_DAY=2026-09-19
python -m tools.demo_report --config conf/demo/pvc1.json --day 2026-09-19 --markdown
```

The report is computed from the decision log and writes no file; redirect it to
keep it. Read `chain.ok`, `minutes.incomplete`, `halts.count`, `halts.causes`,
`reconciliation.dispute_halts`, `ledger.equity.worst` and `funding.net`.

**The report scores nothing.** It counts halts and names their causes because
operations needs that; it classifies `HALT`, `RESUME` and `RECOVERY` as
`unclassified` and says so in the payload, because whether a campaign's halts
count against it is a question for the prospective protocol and that protocol
does not exist yet. Nothing in the daily report is evidence for or against a
hypothesis.

## 13. Replay parity

```
make demo-replay-parity DEMO_DAYS="2026-09-19"
mkdir -p state/demo/parity
python -m tools.replay_parity --config conf/demo/pvc1.json --root data \
    --live-log state/demo/decision_log --days 2026-09-19 --json
python -m tools.demo_report --config conf/demo/pvc1.json --day 2026-09-19 \
    --parity-dir state/demo/parity
```

Redirect the `--json` run to `state/demo/parity/2026-09-19.json` so the daily
report can quote it. Exit 0 is parity, 1 is a divergence, 2 is a refusal to
compare at all.

**A divergence stops the campaign until it is explained.** It is never repaired
by running the comparison again: a replay that agrees on the second attempt has
told you the comparison is not deterministic, which is a worse finding than the
divergence.

## 14. Monthly freeze — what will happen, and what happens today

```
python -m tools.demo_report --config conf/demo/pvc1.json --month 2026-10 --freeze
python -m tools.freeze_evidence --verify artifacts/pvc1_2026-10_SHA256SUMS.txt
```

**Today the first command refuses, exits 2 and writes nothing at all** — no
directory, no partial artifact, no manifest — because `conf/demo/pvc1.json`
carries `protocol_hash: null` and the recorder contract carries
`prospective_from: null`. There is no prospective protocol to compute a month
against, so there is no month to freeze. This is the designed behaviour and not
a fault to work around: a monthly artifact produced without a preregistered
protocol would be a result whose rules were chosen after the data existed.

When a protocol has been preregistered and a campaign has actually run, the
freeze writes the month and a checksum manifest beside the other manifests under
`artifacts/`. `tools.freeze_evidence` refuses to overwrite an existing manifest,
and the report asks for that refusal *before* it writes anything, so a second
`--freeze` of a month that is already frozen leaves the frozen bytes exactly as
they were. A corrected month is therefore a **new** manifest under a new name,
with the superseded one kept; it is never a regenerated one.

Nothing in this section has happened. No prospective artifact exists in this
repository, no research-question row has been added for one, and this document
does not authorise a campaign, a promotion, or the use of real money.

## 15. Incident recording

Every operator command that changes state already writes its own evidence: an
`OPERATOR`, `RESUME` or `HALT` record carrying a mandatory note.

Everything else that touched the campaign — a host restart, a disk replacement,
a recorder restart, a Prometheus outage, a clock correction, an accidental
`docker compose down` — is recorded in the campaign incident log alongside that
day's report, with the wall-clock time, what was observed, what was done and by
whom. An incident that exists only in a chat message did not happen as far as
the evidence is concerned, and a campaign whose gaps are explained only in
someone's memory is a campaign whose gaps are unexplained.

## 16. What an operator may never do during a campaign

* **Change the configuration, a rule, a rule parameter, a cost, or a limit.**
  Every one of them is inside `config_hash`, so changing one mid-campaign splits
  the campaign into two campaigns and neither of them is the one that was
  preregistered.
* **Edit, truncate, reorder, delete or tidy a decision-log file**, or its
  `.ndjson.truncated` companion. The chain verifier starts its walk at the very
  first record precisely so that removing the head of the evidence is visible.
* **Restart the recorder to fix coverage** without recording it as an incident
  first. A coverage number that improved because the operator intervened is a
  coverage number about the operator.
* **Clear a reconciliation dispute without inspecting both stores**, or with a
  note that does not say what was checked.
* **Resume from `HALT` without reading the halt's `detail`.** The note on
  `resume` is supposed to state what was checked, and it cannot if nothing was.
* **Regenerate a frozen manifest in place.** A superseded manifest is kept and a
  correction gets a new name.
* **Run with `--allow-dirty` on a `CAMPAIGN` profile.** The runner refuses, and
  the refusal is the point: a campaign whose records name a working tree nobody
  can reconstruct is a campaign that cannot be reproduced.
* **Delete or edit anything under `artifacts/`, `data/research/`, or any
  `docs/p*_preregistration.md`.**
* **Change a threshold, a cost model, a fold, a horizon or a success rule after
  seeing a number.** There is no procedure in this document for rescuing a
  result, because there is no such thing.
* **Touch a live venue, a credential, or leverage above 1x.** No such path
  exists in this code, and adding one is not an operator action.
