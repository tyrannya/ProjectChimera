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
recorded data. The sections below are the procedure the campaign will follow
once the protocol is preregistered and both the hash and the parameters are
committed; until then they are exercised against a `TEST` or `SOAK`
configuration the operator writes and does not commit.

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

Section 8.1 of the adopted plan describes a continuous `READY` loop and the
runner has the state machine for one; the CLI does not run it. That is a runner
gap, recorded here rather than hidden behind a restart policy that makes a
bounded pass look like a service.

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
docker compose --profile demo up -d recorder demo prometheus grafana alertmanager
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
docker compose --profile demo stop demo
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
* `input_coverage.by_kind` for the kinds you are reading, because a zero there
  is ambiguous on this build and the block says which zeros mean "the runner
  cannot write this yet" and which mean "nothing happened".

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
`artifacts/`. `tools.freeze_evidence` refuses to overwrite an existing manifest.
A corrected month is therefore a **new** manifest under a new name, with the
superseded one kept; it is never a regenerated one.

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
