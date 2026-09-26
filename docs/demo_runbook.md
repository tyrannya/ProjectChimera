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

**0.2 The source-identity halt is REPAIRED (R1-a). A `dirty` halt now means a
dirty tree.**
Until R1-a, `tools/demo_run.py::_software()` could not establish the source
identity — it called `nn.source_identity.source_identity()` with no argument
where the function requires a root, and then read attributes off what is in fact
a dict — so both failures fell into its `except Exception` and the block it
returned carried `"dirty": true`. **Every** run declared a dirty tree, however
clean the checkout, and since `DemoRunner.self_check` refuses a run with `dirty`
unless `--allow-dirty`, and `--allow-dirty` is itself refused on a `CAMPAIGN`
profile, every `CAMPAIGN` stopped at `SELF_CHECK`.

`_software` now passes the checkout root and reads the mapping by key, and an
identification git only half-answered (a tree it can list but has no commit to
name) still fails closed rather than reporting a clean tree it cannot name. So a
genuinely clean checkout passes `SELF_CHECK` and a genuinely dirty one is still
refused — which is what the refusal always meant.

**Read the `revision` before reaching for `git stash`.** `dirty: true` is still
also the FAIL-CLOSED answer when the identity could not be established at all —
the tree is not a git checkout, git is unavailable, a `.py` file under a source
root is hidden by `.gitignore`, or git named no revision for it — and the broad
`except` discards the specific reason, so the halt text reads the same either
way. The block tells the two apart: a genuinely dirty tree carries a real
40-hex `revision` and `source_digest`, and every fail-closed case carries
**empty** ones. A real revision means commit or stash and run again; an empty
one means the runner could not identify its own source, and committing changes
nothing.

This repaired the identity block only. The other reasons a `CAMPAIGN` cannot yet
be operated end to end are unchanged, and the first of them is that no
`CAMPAIGN` configuration can build a runner at all — see the fourth entry in
section 1.

**0.3 `tools.demo_run run` is a long-running service (R1-d).** It starts the
runner once — `STARTUP`, `SELF_CHECK`, `RECOVER`, which is where every restart
check runs — and then loops: it catches up the minutes the recorder has
published, waits for the next minute close plus `ready_grace_seconds` (5 s,
section 8.1's READY grace), and repeats. It stops on `SIGTERM` or `SIGINT`, and
only between minutes: the signal handler records the request, the minute in hand
finishes — its `PERSISTENCE` included — and the process writes `SHUTDOWN` and
exits 0. While it is waiting that takes under a second. It exits 3 on `HALT`, at
start or during a pass, and does not loop past a halt; anything unexpected ends
it with a traceback and a non-zero exit, which the unit restarts a bounded number
of times. `run --once` is the old single bounded pass and `run --replay FROM TO`
is bounded too; neither is the service.

Until R1-d, `run` was one bounded pass and systemd's `Restart=always` with
`RestartSec=30s` was the scheduler. That is gone, and so are its consequences:
the metrics endpoint on 9103 lives as long as the process, `systemctl status
chimera-demo` shows the unit active, and `RunnerDown` is a liveness alert
(section 11).

What R1-d does not change, and must not be read into it:

* each wake decides what the recorder has published by then, and the recorder
  publishes on its own cadence, so a wake can find nothing new. Since R1-g every
  pending minute is decided, in order, with no cap (section 0.5).
* a halted service exits 3 rather than staying up in `HALT`, so `RunnerHalted`
  is visible only for a moment and `RunnerDown` fires two minutes later. Leaving
  `HALT` is still section 7's procedure; the halt and dispute lifecycle is
  R1-i's.
* the wait runs on the host's wall clock -- since R1-e, the injected
  OPERATIONAL clock described just below. Nothing it reads reaches a record.
* `flatten`, `resume` and `resolve` refuse, exit 2, while the service holds the
  state directory's lock: a second writer beside a running service would fork
  the decision log's hash chain. Stop the service first (section 7).

**Two clocks (R1-e).** The runner decides on the DECISION clock only:
`RunnerClock`, advanced by the recorded instants it reads, so the same files
replayed decide the same way. Aegis and both executors cannot be built without
it on any profile -- CAMPAIGN, SOAK or TEST -- because `build_risk_engine` and
`build_hedged_position` refuse to build without it. The service's wake-ups, the
heartbeat and the `*_age_seconds` gauges run on the OPERATIONAL clock: host wall
time, which enters the process in one place, `tools/demo_run.main`, and reaches
the service loop and the telemetry as the same single clock. It never reaches a
record: `tests/test_r1e_clock_hardening.py` runs one recorded day under hosts set
to 2100 and to 1971, with operational clocks about nine and a half years apart
that leap forward and step backwards, and requires every persisted byte to be
identical. Since R1-f the operational clock also decides one thing more:
WHETHER the feed is stale (0.4). It decides nothing a record says. The stall
records carry only recorded facts, and `tests/test_r1f_real_staleness.py` runs
two services under hostile hosts and different operational clocks through a
stall and requires every persisted byte to be identical.

**0.4 Real staleness: the READY gate and `FEED_STALLED` (R1-f).** Before every
pass, including one where no minute arrived, the service asks whether the feed
is fresh. The answer comes from the **recorder's heartbeat**
(`<root>/health/heartbeat.json`, rewritten every 30 s), not from the recorder's
normalized minutes -- and R1-g, which publishes each minute within seconds,
leaves it so (section 0.5):

    through = min(heartbeat_ns, streams["um.kline_1m"].last_event_ns)
    age     = operational now - through

Both terms count:

* `heartbeat_ns` is when the recorder last wrote the file, on the host's wall
  clock.
* `last_event_ns` is the newest perpetual kline frame it has received. It is
  stamped with that frame's minute **open**, which is how the recorder stamps a
  candle, partial frames included. The open is used as-is: it is a lower bound
  on when the exchange last spoke. The close of a forming candle lies in the
  future.

Taking the earlier of the two means neither fact can vouch for the other:

* a recorder process that keeps beating cannot make a dead kline stream fresh;
* a kline stamp cannot make a heartbeat file fresh once the file has stopped
  being rewritten;
* no other stream, however busy (mark price, book, spot), can stand in for the
  perpetual's klines.

The age is recomputed from the service's own operational clock on every pass.
It keeps growing after the recorder dies, rather than freezing at the age its
last heartbeat reported. The feed is **stale** when:

* `age` is above `max_data_delay_s` (180 s in `conf/demo/pvc1.json`;
  exactly 180 s is still fresh);
* `age` is negative (an instant ahead of the host's clock means the clocks
  disagree, so the age is unknown);
* the heartbeat file is missing, unreadable, or of another schema;
* the `um.kline_1m` entry is absent, has never delivered an event, or is not
  `up`. `up` is the recorder's own `connected and not halted`, so a latched
  storage halt is stale on the first pass that reads it, even though its last
  event is seconds old.

On a healthy recorder the age stays well under the limit. The heartbeat is at
most 30 s old, and the kline stamp is at most a minute older than the heartbeat
(at most two minutes and a delivery delay if only closed frames arrive). That
is about 90 s, and at worst about 152 s.

The recorder and the service must share a host clock, as they do when both run
with `--root data` on one machine.

The newest normalized minute no longer decides anything here. It is still
written into the stall records (`feed.newest_minute`) as a recorded fact, and
the `chimera_demo_last_minute_age_seconds` gauge still shows its age, but it is
diagnostic only. A live recorder whose first normalization pass has not run yet
has no minute at all, and that is not a stall: the service is READY with
nothing to decide.

Nothing about the process's own startup ever counts as freshness. While the
feed is fresh, the service also wakes at `through + max_data_delay_s +
ready_grace_seconds`. So a dead kline stream or a dead recorder is declared
stalled within the limit plus the 5 s grace of the last instant vouched for,
for any limit.

A stale feed does three things:

* it writes one `FEED_STALLED` record, carrying the newest minute and the limit
  but no operational instant;
* the runner enters `FEED_STALLED` (`chimera_demo_state{state="FEED_STALLED"} ==
  1`); and
* each wake from then on runs the **stall tick** instead of a catch-up.

The stall tick re-reads the last processed minute from the recorder's files and
runs the tick's own safety checks on it:

* the **kill switch**;
* **funding**, within that minute's window, so only a settlement row that
  arrived late for an instant already passed can be booked, once, and never one
  the feed has not reached;
* the **liquidation** check against that minute's recorded mark.

It evaluates no rule, places no order, and does not move the decision clock.
**Positions are held.** A stale feed alone never flattens anything. A real
liquidation touch on the last recorded mark flattens and halts exactly as it
would in a tick, and its records say `liquidation_touch`, not a stall.

When the heartbeat vouches for a fresh feed again by the same test, the gate
writes one `FEED_RESUMED` record and returns to READY by itself. No `resume` and
no note are needed, because nothing was halted: Aegis is not told about a stall
and holds no flag for it. The pending minutes are then processed as usual: every
one of them, in order (R1-g). An open stall survives a restart: the service comes back
`FEED_STALLED`, read from the log, and leaves it only when the gate sees a fresh
heartbeat.

`FEED_STALLED` is not `HALT`. A `HALT` (Aegis, a dispute, the kill switch, a
liquidation) exits 3 and needs an operator. `FEED_STALLED` keeps the process up,
keeps the heartbeat beating, and clears itself. The gate never moves a runner
out of `HALT`, and a halt that happens during a stall, such as the kill switch,
stands. SIGTERM during a stall stops the service cleanly, as it does between
any two minutes.

`max_data_delay_s` was a dead limit before R1-f. The only check fed by it, in
`tick`, compared a minute's close with a decision clock just set to that close,
so the age was zero by construction. That call is gone, and the READY gate is
now the only staleness authority. Aegis's own `note_feed` mark is no longer fed
by the demo.

**What R1-f does not change: the recorder publishes no faster.** Today's
recorder re-renders the open day every 300 s (`NORMALIZE_INTERVAL_S`), and later
in the day less often: a 66 s pass stretches the interval to 660 s under
`NORMALIZE_DUTY_CYCLE`. Both are longer than the 180 s limit. That is why the
gate reads the heartbeat and not the newest normalized minute.

The first R1-f head did read the newest minute. The independent review (F1)
found that it recorded a `FEED_STALLED` / `FEED_RESUMED` pair around almost
every publication gap on a perfectly healthy recorder, and that those pairs
turned a healthy day's replay parity from PARITY to DIVERGED on `seq` alone.

On this build a healthy recorder writes **no** stall records, at 300 s or at
660 s. `tests/test_r1f_real_staleness.py` runs two hours of each and requires
zero, and a healthy gated day is byte-for-byte the ungated one, `seq` included.

A genuine outage still writes a pair, and still shifts the global `seq` of every
later record, as a restart does. A genuine outage is any of:

* a dead kline stream;
* a dead recorder;
* a halted stream;
* a heartbeat that happens to catch the stream in the middle of the venue's
  24-hour forced reconnect (`up` false for one beat).

How `seq` is compared across those is R1-j's question. R1-g publishes each
closed minute within seconds and deliberately keeps the heartbeat as the
authority (section 0.5), so a `FEED_STALLED` still means "the recorder's heartbeat
has not vouched for the perpetual kline stream within 180 s".

Follow-ups the independent review recorded and R1-f leaves alone:

* F3: Aegis's `update_equity` can halt without a runner HALT, in a tick and in a
  stall tick alike. Inherited; R1-i.
* F4: `chimera/demo/reports.py` names a `DemoRunner._feed_record` that does not
  exist; the gate is `check_feed`.
* F5: the classification vocabulary differs between the log's kind sets and the
  parity tool's.
* F6: a test name and its content disagree.
* F7: the negative-age rule under a small clock skew.

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

**0.5 Cadence and the funding minute (R1-g).** Three changes, one requirement:

* **The recorder publishes each minute within seconds.** Beside the 300 s
  maintenance pass (rotation, freeze, gap-fill), a publisher looks every second
  for a newly *settled* minute and renders the open day incrementally when one
  appears. A minute is settled once every stream folded into its row -- both
  klines, the perpetual's mark and both books -- has delivered an event at or
  after its close, or once the recorder's clock is 10 s past it
  (`PUBLISH_SETTLE_CAP_S`), so a closed candle is never published beside a mark
  or book aggregate that is still filling. A quiet stream therefore delays
  minutes by at most 10 s and then they appear with that stream's columns
  empty, as before. Every render and every freeze holds one lock, one pass uses
  one horizon for both markets, and the perpetual is written last.
* **The runner decides every pending minute, in order.** No cap and no
  `SKIPPED_STALE`: `max_catchup_minutes` is retired, and a configuration that
  still carries it is refused by name (remove the key). Old logs holding
  `SKIPPED_STALE` still read, report and replay as before. A stop request during
  a long backlog is taken between minutes: the minute in hand completes and the
  next start resumes at the one after it.
* **A funding minute waits for its settlement.** A minute whose close is at or
  after the venue's announced `next_funding_time_ms` is decided only once
  `funding/um/settlements.ndjson` holds that instant's row, or once
  `funding/um/observed.json` -- the query windows of the recorder's successful,
  complete funding polls -- covers the instant with 30 s to spare. Until then the
  minute is **deferred**: nothing is written, the cursor and the clock do not
  move, and the service tries again at its next wake. The recorder's scheduled
  poll 60 s after each instant normally clears it within a minute or two.

A deferral writes **no record** (a replay of the finished files never defers,
so a record would split the two logs); it is visible only as a `minute ...
deferred:` warning in the runner's log, repeated at every wake. A deferral that
lasts is the recorder failing to ask: `observed.json` stale or missing, funding
polls failing in the recorder's log. Do not create or edit `observed.json` by
hand -- it is the recorder's statement of what it asked, and a hand-written one
turns "the recorder does not know" into "there was no settlement". The runner
also defers, rather than halts, on a normalized day it catches mid-rewrite (the
recorder rewrites a day in place until R1-h); the same file failing again
unchanged still halts `feed_unreadable`.

The per-settlement detail -- settlement id, rate, mark price, quantity, notional,
signed cash flow and direction -- is in the **`FUNDING` records of the decision
log**, one per settlement, not in the daily report.

A fourth, about which profile can reach any of this. **This entry was written
when `_software` declared every tree dirty; R1-a repaired that (section 0.2), and
the paragraphs below have not been re-derived for the runtime that repair
exposes.** Read them as the record of a build whose `CAMPAIGN` halted at
`SELF_CHECK`, and re-check anything you are about to rely on.

What holds today is the conclusion, not the cause. A `CAMPAIGN` still reaches no
`DECISION`, `FUNDING`, `RECONCILIATION`, `LIQUIDATION_TOUCH`, `SKIPPED_STALE` or
`INCOMPLETE_STATE` record — but it now stops **before a runner is built at all**,
not at `SELF_CHECK`: `chimera/demo/config.py` refuses a `CAMPAIGN` carrying
`rules` while `protocol_hash` is null, and `CarryRule` has no defaults, so
`tools.demo_run` raises *"R1_carry needs min_basis, …"* out of `_load`. The
committed `conf/demo/pvc1.json` is refused for exactly that reason. Repairing it
means freezing the S2 protocol, which is a later phase's, so it is not repaired
here. The account that follows — of what a `CAMPAIGN` log holds, and of what
`flatten` does on that profile — describes the `SELF_CHECK` halt and no longer
describes how a `CAMPAIGN` fails.

The account, as it stood: a `CAMPAIGN` that has only ever run as one holds
`STARTUP` and `HALT` and nothing else on this build — `OPERATOR` included. The
CLI's `flatten` does call `start(allow_dirty=True)` itself, which appends a
`STARTUP` and a `HALT` of its own, but it then refuses instead of acting: no
`CAMPAIGN` start reaches the tick loop, because `SELF_CHECK` halts before
`RECOVER`, so
`cursor.last_minute_processed` is still null and `flatten` stops on *"nothing has
been processed yet, so there is nothing to flatten"* — before
`_require_recordable`, before either leg moves, and before the `OPERATOR` append.
It stops by raising, and the CLI's `flatten` branch does not catch it, so what an
operator meets there is a traceback rather than a refusal. (The cursor is stored
per state directory and not per profile, so pointing a `CAMPAIGN` config at a
directory some `SOAK` or `TEST` run already advanced hands `flatten` a minute and
this stops holding. Do not do that for a second reason: the two runs would share
one decision log under two `config_hash` values.) `resume` refuses on this
profile too (section 7), and `resolve` cannot run from the CLI at all (a fifth,
below). Reaching any other kind needs a `SOAK` or `TEST` profile. That no longer
requires `--allow-dirty` on a clean checkout: `self_check` still refuses a dirty
tree on **every** profile, but since R1-a a clean tree reads as clean, so reserve
the flag for a tree you know is dirty and mean to run anyway. That is a
necessary condition and not a promise that each remaining kind is then reachable:
`RESUME` stays out of reach on those profiles as well, because the CLI arrives at
`resume` without `start()` and the runner is therefore never in `HALT` (section
7). Where section 6's emergency procedure IS followable is `SOAK` or `TEST`: once
a minute has been processed there, a `flatten` does append its `OPERATOR` record.

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

Confirm on `http://127.0.0.1:9103/metrics` that `chimera_demo_up` is 1, that
`chimera_demo_state{state="READY"}` is 1 while the service waits between
minutes, and that `chimera_demo_heartbeat_timestamp` moves at least every 30
seconds; and on
`http://127.0.0.1:9102/metrics` that `chimera_recorder_up` is 1 for every
stream. Both ports are bound on all interfaces by the Prometheus client and must
be firewalled to the Prometheus host; the compose file publishes them on
127.0.0.1 only.

## 3. Stop

```
sudo systemctl stop chimera-demo
docker compose stop demo
```

The unit sends `SIGTERM`, and the runner defers it past persistence: the minute
in hand runs to completion — its ledger, its `DECISION` record and its runner
state are all written — and only then is `SHUTDOWN` written and the process
exits 0. During the wait between minutes that is under a second; the unit allows
`TimeoutStopSec=120` for the minute in hand and sends `SIGKILL` after it. A
`SIGKILL` is the crash case section 9.3's triage recovers on the next start; a
stop in the middle of a record is not reachable, because `DecisionLog.append`
fsyncs each record before it returns.

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

### What a restart does to Aegis's equity

**A restart no longer re-seeds equity from the configured capital.** It used to:
`build_risk_engine` ended with an unconditional `update_equity(capital)` that ran
*after* the persisted state had been restored, and `update_equity` is a guard,
not a setter. Two things followed, and you would have met both.

* A campaign that had genuinely earned a peak of `capital / (1 -
  max_drawdown_pct)` or more — for `pvc1.json`, 1,052,632 against the demo's
  1,000,000 — **halted on its own startup** with a drawdown it had never taken,
  measured from a peak it really had against an equity it did not have.
* A restart across UTC midnight opened the new day's loss budget at `capital`
  rather than at what the account was worth when the day rolled, so an account
  legitimately down a few percent over several days had its whole cumulative
  fall re-measured as one day's loss and halted on the first ordinary minute.

Capital now seeds equity **only on a genuine first start** — read off whether
there was a `risk.json` to restore, not off whether equity happens to equal
capital, which is also what a campaign that gave back its gains looks like. On a
restart the persisted `equity`, `peak_equity`, `day_start_equity`, `daily_pnl`,
counters, cooldowns and funding state are all left exactly as the previous
process wrote them, and the UTC day rolls on the first real mark rather than on
construction.

### When a missing `risk.json` is not a first start

"There was no `risk.json`" is the right reading of a first start and the wrong
reading of a campaign that has already run. The decision log is what tells the
two apart, and it is now read before anything is seeded.

**The log is evidence here, not a second risk authority.** Aegis remains the
central risk authority; what the log answers is one question — *is the persisted
Aegis state believable?* — and when the answer is no, the campaign stops rather
than trading on a state nobody can vouch for. Nothing is rewritten to make the
two agree: not the log, and not `risk.json`.

**What is compared.** Every start reads the campaign's **verified** decision log
and finds its last statement about the risk state. That is not the last line of
the file and not the last record that happens to contain a hash:

* `DECISION`, `FUNDING`, `RECONCILIATION` and `LIQUIDATION_TOUCH` records carry a
  `risk.state_hash` — the whole Aegis snapshot, hashed. Those **restate** the
  state, and `risk.json` must hash to the newest one.
* a `HALT` record states that the campaign halted and restates no hash, so what
  is checked against it is that `risk.json` is halted too.
* a `RESUME`, an `OPERATOR` record (a flatten, a settled equity) or an
  `INCOMPLETE_STATE` record can move the state and restate nothing, so nothing is
  checked against them. The next decided minute restates the hash and the window
  closes. **Do not read "the last line is the risk state" into this**: an
  ordinary flatten really does move the equity and really does write a record
  with no risk block, and a comparison that ignored that would refuse the most
  routine thing in this runbook.

**What is a dispute.** Four findings, each written to the log as a `RECOVERY`
record whose `recovery.cause` names it, and each halting the campaign:

| `recovery.cause` | what was found |
| --- | --- |
| `RISK_STATE_ABSENT` | there is no `risk.json` and the log holds records. Not a first start; the configured capital is **not** seeded. |
| `RISK_STATE_UNREADABLE` | `risk.json` exists, could not be believed, and the log holds records. The bytes are preserved exactly as they are. |
| `RISK_STATE_PRE_SCHEMA` | `risk.json` is the pre-schema halt record — a halt claim and no account state — and the log holds records. |
| `RISK_STATE_MISMATCH` | `risk.json` is readable and current, and is not the state the log last recorded: a different hash, or running where the log says halted. |

**Nothing writes the questioned file before the verdict.** The finding is
reached from the file as it was read, and until it is settled nothing — not the
kill switch, not R1-b's equity reconciliation, not an operator command — may
change it: an absent `risk.json` stays absent and a present one stays
byte-identical. Three of the findings are settled at once and seal. A
`RISK_STATE_MISMATCH` is settled later in the same start, once the crash triage
below has run, and Aegis is held unchanged until then. If the finding stands,
the file is still exactly the bytes that were found when the campaign halts on
`risk_continuity:`, whatever its equity says against the carry ledger. If a
crash explains it, the file is released, and only **then** does R1-b reconcile
its equity against the ledger — so a crash that also left the two equities
apart still halts on `equity_dispute:`, now after the verdict rather than
before it. (Until PR #102's third remediation that reconciliation ran first,
and a stale or foreign file whose equity was not the ledger's had R1-b's
`equity_dispute:` halt persisted over it before R1-c ruled: the found bytes were
lost, the campaign reported R1-b's reason instead of R1-c's, and every restart
recorded the file it had itself rewritten as a new finding.)

The `RECOVERY` record is written **once per finding**, not once per restart: a
disputed campaign gets started repeatedly and a record each time would bury the
finding in copies of itself. The `HALT` record each restart writes is unchanged.
A finding that recurs after a genuine repair — the file restored, a minute
decided, the file lost again — is recorded again, because the log restated the
hash in between. Each sealed restart's own `STARTUP` still says it was sealed.

**What is deliberately not a dispute.**

* A first start: no records, so there is nothing to be continuous with.
* A halted campaign restarting halted. It keeps the reason it halted for.
* A `risk.json` that is *more* cautious than the log — halted where the log's
  newest statement is a `RESUME`. Refusing that would be a refusal with no safety
  in it.
* A crash inside a tick. Aegis persists before the record that restates it, so a
  process killed in between leaves a `risk.json` genuinely **ahead** of the log —
  which looks exactly like a swapped file. Section 9.3's own triage already names
  that crash (`LOG_BEHIND_STATE`, `LOG_AHEAD_OF_STATE`, `TORN_TAIL`), excludes
  what it must and continues, and a continuity halt on top would take that
  recovery away. So a `RISK_STATE_MISMATCH` stands down when the triage really
  found a crash, and only then. The three findings no crash can produce — an
  absent file, an unreadable one, a pre-schema one — never stand down.
* A crash inside one of five narrow **Aegis-only** windows, and only when it is
  *proved*. Aegis persists before the record that would restate it, and in
  these windows nothing but Aegis moved, so section 9.3's triage — stores,
  ledger accumulators, chain head — finds nothing. Each is proved by reverting
  exactly the fields that window writes to the one value they held before, and
  requiring the result to reproduce the log's **full** `risk.state_hash`:
  * `halt` — a bare `RiskEngine.halt` (a rule exception, a dispute, any
    `_halt` site) before its `HALT` record: `halted`/`halt_reason` back to
    `false`/`""`.
  * `kill_switch_halt` — the kill switch found on a running engine before its
    `HALT` record: the mirror, `halted` and `halt_reason` back, and the reason
    must be one of the two `check_kill_switch` writes. Whether the switch file
    is still there at the restart does not matter; the mirror is what persisted.
  * `kill_switch_mirror` — the same look on an already-halted engine: the mirror
    alone.
  * `equity` / `equity_halt` — the tick's `update_equity` (with or without the
    drawdown / daily-loss halt it raises) before the minute's `DECISION`, on the
    **same UTC day and below the peak**. The prior equity is in the log in
    plaintext — the last `DECISION`'s (or flatten's) `ledger_effect.equity` —
    and the proof additionally requires the carry ledger to hold exactly the
    found equity (the same crash wrote the ledger first) and the real
    `update_equity`, replayed on the reconstructed prior, to reproduce the file.
    A funding minute is this window too.

  A proved window writes the `RISK_STATE_MISMATCH` `RECOVERY` record, whose
  `detail` names the window, and does **not** seal: the campaign halts on the
  guard's own reason (or, for a plain equity window, continues and re-decides
  the minute). `recovery.risk_continuity.halt_transition_explains_mismatch` is
  `true` exactly when the proved window persisted a halt.

  **Still sealed, and canonical R1-i's:** the same `update_equity` window when
  it **rolled the UTC day** (every UTC midnight) or **set a new peak**. The
  values those overwrote are not unavailable: the prior `day_start_equity` is the
  equity of whichever write first touched the previous UTC day, and the prior
  `peak_equity` the running maximum of every equity Aegis was handed, and both
  may be reconstructable from the configured capital or a bounded scan of the
  log's earlier records. But neither is restated by the record the crash
  preceded, so proving either window takes historical reconstruction —
  replay-shaped logic over the campaign's equity history — rather than the
  narrow one-step local inverse the five proved windows use, and that is R1-i's
  work. So is a crash between `note_funding_settlement` and its `FUNDING`
  record (unless section 9.3's triage finds that crash through the ledger), one
  that moved the feed mark, and any combination of two windows. A sealed crash
  window looks exactly like a swapped file; record it as an incident and stop.
* A log that does not verify. A forged log halts the campaign on `log_forged:`,
  which is the graver finding and the one that owns it; an edited log may not
  condemn a state file. It also may not excuse one: a forged log beside a missing
  `risk.json` still refuses to seed.

**A sealed run's records state nothing about the file.** The engine is sealed
when a finding stands — it halts in memory and writes nothing — so neither the
`HALT` that run appends nor an `OPERATOR` record from a `flatten` issued in it
(permitted; that is what `HALT` is for) persisted anything. Every `STARTUP`
record carries `risk_continuity_sealed`, which says whether **that run** was
sealed, and the next start skips **every** record of a sealed run when it looks
for the log's last statement — `HALT`, `RESUME`, `OPERATOR`, `INCOMPLETE_STATE`,
and also any record carrying a `risk.state_hash`. A run that could not write
`risk.json` cannot prove what it holds, and when the disputed file was itself
halted a sealed engine's hash is exactly that file's hash, so reading it would
clear the dispute with no repair at all. It is per run: the next `STARTUP`
states its own value, and a `RECOVERY` record by itself says nothing about it (a
proved crash window writes one and runs unsealed).

A sealed engine also cannot move **in memory**. It refuses `resume` and an
adopted equity outright, so `resume` and `resolve --equity` are refused against
a continuity seal and change nothing (`resolve --equity` exits refused and
prints no state), and it ignores the observations a `flatten` or a
reconstruction reports to it, so those still complete. Without that, a state
changed only in memory would be restated by the next hash-bearing record as if
it had been written.

**Older logs.** A `STARTUP` without the field reads as `false` (unsealed). That
is correct for every log written by a `main` build from before PR #102, none of
which can seal, and every build from `74edc25` on writes the field on every
`STARTUP`. It is **not** correct for a log written by an intermediate, unmerged
build of PR #102 from `dded833` up to and including `471c1b7`: those could
already seal but did not yet write the field (it arrived in `74edc25`), so a
sealed run of theirs would be read as unsealed and its `HALT`/`OPERATOR`
records as statements about `risk.json`. The compatibility condition is
therefore: **no state directory that a build merged from PR #102 reads was ever
run by one of those intermediate builds.** Whether that holds is for the owner
to attest separately; this document does not attest it.

That is what makes a repair work: restore the state directory from a backup taken
after the log's last restatement and the comparison falls back to that
restatement, which the restored file matches — and the repaired run starts
unsealed, so what it does (a `flatten` before any minute is decided included) is
read as it always was. Without it R1-c's own halt would disagree with a
correctly restored file for ever, or a flatten during the seal would read as
`MOVED` and silently clear a dispute nothing had repaired.

**Reading the finding.** `status` prints a `risk_continuity` block and it does
not start the runner, which matters when the campaign will not start:
`disputed`, `fault`, `risk_state_load` (`MISSING` / `LOADED` / `LEGACY` /
`UNREADABLE`), `log_statement`, `log_records`, and the two hashes when there are
two. The same block is inside the `RECOVERY` record under
`recovery.risk_continuity`.

**What an operator may do about it today: nothing in this document, and that is
deliberate.** The campaign fails closed and stays there. Editing `risk.json` by
hand is not a supported recovery and never becomes one — section 16 — and R1's
own acceptance criterion is "zero manual state edits". Restoring the state
directory from a backup taken *after* the log's last restatement is the only
thing that clears a `RISK_STATE_ABSENT` or `RISK_STATE_UNREADABLE` finding
without a new command, and a backup older than that will be refused again, for
the same reason it is refused for `carry_ledger.json` in section 6. A clearing
path with a mandatory note — one per dispute kind, reachable from the CLI — is
**canonical R1-i's** item and is not in this build. If you meet one of these
findings, record it as an incident (section 15) and stop.

### Two halt reasons you can now meet at startup

On a restart the persisted equity is **reconciled** against `carry_ledger.json`,
which is the campaign's accounting authority. They agree exactly in an ordinary
run — the runner hands `update_equity` the `float()` of the very `Decimal` the
ledger recorded — so a disagreement means something happened to one of the two
files. Neither is overwritten to make them agree:

| `halt_reason` begins | what it means |
| --- | --- |
| `equity_dispute:` | `risk.json` and `carry_ledger.json` state different equities. The reason names both numbers. |
| `equity_reconciliation:` | `carry_ledger.json` could not be read at all, so the claim in `risk.json` cannot be checked. |
| `risk_continuity:` | `risk.json` does not continue the campaign's decision log. The four findings are in the previous subsection; the reason names which one. |

**Where to read the reason.** `status` reports `risk_halted` and the ledger's
`last_equity`, and both are useful here — but its `halt_reason` field is the
*runner's* own, which is only set by a `start()` in the same process, so for a
halt raised while the engine was being built it prints `null`. The authoritative
text is the `halt_reason` inside `risk.json`. Read that, and compare `risk.json`'s
`equity` against the ledger's `last_equity` before deciding which file is wrong.
The usual causes are a state directory restored from a partial backup, a file
copied between hosts, a `carry_ledger.json` that was truncated, or the crash
window named below.

### Clearing an equity dispute

```
python -m tools.demo_run --config conf/demo/pvc1.json --root data resolve \
    --equity --note "the ledger is the campaign's accounting; checked both files"
```

This is the one clearing path for this dispute, and it needs to exist rather than
deferring to `resume`: **`resume` does not settle it.** `resume` clears the halt
flag and changes neither equity, so the next process reads the same disagreement
and halts again before a tick can re-synchronise them — which is why an earlier
revision of this section was wrong to say the halt merely inherits section 7's
open `resume` blocker. It does not. It needs its own command, and this is it.

What it does, and what it refuses:

* it adopts what the **carry ledger** accounts for as Aegis's equity, because the
  ledger is where the campaign's cash and marks live and Aegis's equity is a
  reading the runner hands it. Both files are left as they are otherwise; neither
  is rewritten to match the other;
* it records an `OPERATOR` decision-log entry naming both numbers and your note;
* it moves `equity`, `peak_equity` (upward only — an adopted reading never lowers
  a high-water mark) and `daily_pnl`, and **nothing else**: not the UTC day, not
  the day's starting equity, not the order window, the cooldown, the loss streak,
  the funding streak or any reconciliation dispute;
* it **refuses** when there is no equity dispute, when Aegis is halted on anything
  else — a drawdown breach, a liquidation touch, a kill switch — and when the
  ledger may not speak. So it is not a second `resume` and cannot be used as one;
* if the adopted equity is itself a breach, Aegis stays halted on **that**, named.
  The dispute is still settled; what is left is a genuine halt you resume in the
  ordinary way;
* running it twice is refused the second time and changes nothing.

For `equity_reconciliation:` the ledger is what could not be read, so restore
`carry_ledger.json` from a copy at least as recent as the log's last
`ledger_effect` and `FUNDING` records **first**; the command refuses until the
ledger speaks for the campaign again. **Do not hand-edit `risk.json` or
`carry_ledger.json`** — section 16, and for the reason section 7 gives.

### Where a disagreement can actually come from

The runner has exactly one writer into Aegis's equity — `tick`'s
`update_equity(float(mark.equity))` — and five places that mark the ledger and
persist it, so a crash is not the only way the two can part. An earlier revision
of this section said it was, and that was wrong. Against the current runner:

* **the crash window.** `tick` persists the ledger immediately *before* it calls
  `update_equity`, so a process killed between those two statements leaves the
  risk state one mark behind the ledger. The disagreement is real, the halt is
  correct, and the two equities differ by a single minute's mark. Closing the
  window means changing the runner's persistence ordering and its crash
  semantics, which is **R1-i's** item, not this one — but it is now clearable,
  with the command above;
* **funding** marks and saves, then either the same tick reaches that one writer
  or the runner halts;
* **a liquidation touch** marks and saves and then halts. `halt` keeps the *first*
  reason, so the restart reports the touch rather than the equity; the
  disagreement is underneath it and surfaces if the touch is ever resumed, where
  the command above settles it;
* **`flatten` used to be the fourth**, and it needed no crash at all: it marked
  the ledger, persisted it, and told Aegis nothing, so every ordinary operator
  flatten left the two files disagreeing and halted the campaign on its next
  start. That is fixed — `flatten` now hands the flattened equity to the same
  single writer, under the same condition as the record it writes — so a flatten
  followed by a restart runs rather than disputing.

A disagreement therefore means a crash in that one window, a state directory
restored in pieces, a file copied between hosts, or a truncated ledger — and not
the routine operation of the campaign.

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
path. `resume` refuses too, for the same reason: an operator note cannot make the
file hold what it does not hold, and clearing the halt without repairing the
ledger used to end in a traceback rather than a refusal. (On this build you will
not meet that refusal from the CLI, because `resume` is reached without
`start()` and stops earlier on "the runner is not halted" — section 7. The check
is there for when the CLI adopts the persisted halt.)

**The order of the repair decides whether the campaign survives, and this is the
one place it is written down.** Restore the ledger **before** running `flatten`:

* `restore` → `flatten` saves the campaign's accounting: the ledger survives, and
  the flatten is booked and carries its `ledger_effect`. It does NOT return the
  campaign to READY — the halt is still in `risk.json` and no CLI command clears
  it, which is section 7's blocker, not this procedure's.
  (`flatten` calls `start()` itself, which
  is what makes this the CLI-followable form; there is no `start` subcommand, and
  `resume` cannot be reached from the CLI on this build — section 7.)
* `flatten` → `restore` **ends the campaign**. A flatten while the ledger is
  muted moves both legs to flat and writes neither a ledger nor a
  `ledger_effect` — correctly, that is the whole point — so a copy restored
  afterwards still holds the pre-flatten quantity while the stores hold zero.
  `reconstruct` then disputes `ledger_store_mismatch` — one of the disputes
  section 0 lists (under "a fifth") as clearable by no operator command — and the
  flatten's own exit **slippage** is then recorded nowhere at all. Its fees are
  not lost — the executor stores accumulate `trading_fees`, and
  `_reconcile_ledger` re-derives fees from them. No executor accumulates
  slippage, so nothing on disk states that number. It is not beyond recovery,
  though: each store keeps the exit order with its `average_price` and
  `filled_quantity`, so the figure can be re-derived by hand against the minute's
  close. Recorded nowhere, recoverable by arithmetic — not the same as lost.
  Note what you will actually SEE **on `TEST` or `SOAK`**: the halt reads
  `dispute: ledger disagrees with the stores`, and the ledger's own `disputed`
  field is still `null`, because `start()` never persists that dispute. On
  `CAMPAIGN` you see none of this, for the reason below: `reconstruct` is never
  reached, so the dispute is never raised at all. The literal `ledger_store_mismatch`
  appears only on stderr, so section 6's "inspect the ledger's `disputed`" step
  will show you nothing here.

Exposure is zero either way, so the wrong order is safe before it is
unrecoverable — but it is unrecoverable. That applies to a **restored older copy** exactly as it does to a
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
It is worse than a stale sentence on screen: **each repaired start appends
another HALT record carrying the now-false reason**, so the decision log
accumulates halts for a cause that no longer exists. `run` prints Aegis's carried
sentence and not the current verdict, so **the command to trust here is
`status`**: it reports `ledger_may_speak` and `ledger_complaint` computed from
the ledger as loaded, it needs no `start()`, and after a correct restore it says
`ledger_may_speak: true, ledger_complaint: null` while `run` is still repeating
the old halt text. On a `CAMPAIGN` profile the verdict never reaches `run`'s
output at all — before R1-a because `self_check` returned the `source_identity`
failure of section 0.2 first, and now because no `CAMPAIGN` configuration builds
a runner (section 1, fourth entry). So `status` is the only command you can ask
for it, and on a `CAMPAIGN` not even that. (A muted `flatten` prints the
complaint on stderr too, but that is a side effect of an action, not somewhere
you can go and look.)

**What `status` cannot tell you.** It reports plenty besides the ledger verdict —
`risk_halted`, `halt_reason`, `hedge_state`, `disputed`, the cursor and the
ledger's own totals — but only the two verdict fields are computed fresh. It does
not run `reconstruct`, and that costs more than one missing line: after the wrong
repair order `status` reads `ledger_may_speak: true, ledger_complaint: null` on a
campaign `run` halts immediately, and `disputed` and `hedge_state` are stale
rather than merely silent, because the dispute that would have set them was never
raised. `risk_halted` is your one true signal there, and it does read true.

So a clean ledger verdict means the ledger may speak, not that the campaign is
well; read `risk_halted` beside it, and treat `disputed` as unanswered rather
than answered "no". On a `CAMPAIGN` profile `reconstruct` is not reached on any
CLI path today, so that dispute shows up nowhere at all. This paragraph asked to
be re-checked if the `source_identity` defect of section 0.2 were ever repaired:
R1-a repaired it, and the conclusion survives for a different reason — no
`CAMPAIGN` configuration builds a runner yet (section 1, fourth entry), so no
CLI path on that profile reaches `reconstruct` either. Re-check it again when
that is repaired.

One more halt reason section 6 will show you that nobody asked for: on a
`CAMPAIGN` profile the CLI's `flatten` calls `start(allow_dirty=True)` on your
behalf, so every CAMPAIGN flatten appends a HALT record reading `allow_dirty was
requested for a CAMPAIGN profile`. You did not request it; the CLI did.

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

The service halts on the switch at the next complete minute it decides, writes
`HALT` and `SHUTDOWN`, and exits 3, which the unit does not restart. **Every
command below writes, so the service must not be running:** `flatten`, `resume`
and `resolve` take the state directory's lock and refuse with exit 2 while the
service holds it. Confirm the unit has stopped, and stop it yourself if the
switch has not yet been seen:

```
sudo systemctl stop chimera-demo
```

If the position has to come down:

```
python -m tools.demo_run --config conf/demo/pvc1.json --root data flatten \
    --note "kill switch pulled during the 03:10Z venue incident; removing exposure"
```

To come back, in this order:

```
rm state/demo/KILL_SWITCH
python -m tools.demo_run --config conf/demo/pvc1.json --root data resume \
    --note "venue incident closed; both stores reconciled; feed ages under 30s"
sudo systemctl start chimera-demo
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

Two different signals:

* `RecorderStreamStale` fires when one of the recorder's streams has delivered
  nothing for 3 minutes. Look at `chimera_recorder_up{stream=...}` and
  `chimera_recorder_reconnects_total` to tell a reconnecting stream from a dead
  one.
* `chimera_demo_state{state="FEED_STALLED"} == 1` means the runner's READY gate
  has found that the recorder's heartbeat does not vouch for the perpetual kline
  stream within `max_data_delay_s` of the operational clock (section 0.4): the
  stream is silent or not `up`, or the heartbeat file has stopped being
  rewritten. Read `health/heartbeat.json` (or `python -m tools.recorder status`)
  to tell which. It has no alert rule of its own: section 11.2's table is
  unchanged.

A stale feed is not a halt. The runner decides nothing and holds its position.
The stall tick keeps checking the kill switch, funding and liquidation on the
last recorded minute, and the runner resumes by itself on the first fresh
heartbeat. The decision log shows the stall as a `FEED_STALLED` record and its end
as a `FEED_RESUMED` record. Nothing needs an operator unless the recorder process
itself is down (`RecorderDown`) or the stall does not end. Do not `resume` a
stall: there is no halt to leave, and `resume` refuses outside `HALT`. A
healthy recorder does not stall the runner, whatever its publication cadence
(section 0.4). A stall is a real outage. A minute that waits for its funding
settlement is not a stall either, and writes nothing (section 0.5).

## 11. Process heartbeat

Three different questions, three different series, and they are routinely
confused:

* `chimera_recorder_heartbeat_timestamp` and `chimera_demo_heartbeat_timestamp`
  answer "is the process alive". The runner's is wall clock, stamped at every
  state change, at every record it commits, and at least every 30 seconds while
  it waits for the next minute close — always from the main loop, never from a
  thread of its own, so a loop that wedges stops beating. The recorder also
  writes `health/heartbeat.json` under its storage root every 30 seconds, which
  survives the process and is what to read after a crash.
* `chimera_demo_last_minute_age_seconds` answers "how far behind the data is the
  runner", which is what rises during a catch-up and is not a liveness problem.
* `chimera_demo_feed_age_seconds{market=...}` answers "how old is the newest
  minute available for this market", which is the recorder's problem, not the
  runner's.

Both ages are republished with every heartbeat (R1-f), so during a stall they
keep rising instead of freezing at their last healthy value.

`RunnerDown` is the liveness alert. It fires when the heartbeat is more than
two minutes old while the process is still scraped — wedged — and when there has
been no heartbeat sample at all for two minutes — gone, because a failed scrape
makes Prometheus mark the series stale, so the age test alone would return
nothing for a dead process. A deliberately stopped runner, and one that halted
and exited 3, is also down: silence the alert for a planned stop. Healthy
waiting looks like `chimera_demo_state{state="READY"} == 1` with a heartbeat
under 30 seconds old; whether the runner is halted is `RunnerHalted`, and how far
behind the data it is, is `chimera_demo_last_minute_age_seconds`.

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
* **Edit `risk.json`, delete it, or copy one in from another host or another
  day.** It is compared with the decision log on every start and a document that
  does not continue the log halts the campaign; hand-editing one to get past that
  is forbidden here for the same reason editing the log is, and R1's acceptance
  criterion is "zero manual state edits".
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
