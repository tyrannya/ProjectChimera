"""`DemoRunner`: section 8.1's state machine, over synthetic files only.

No market data is read anywhere in this file, and no economic conclusion is
available from any run in it. What is asserted is that the machinery sequences
correctly, refuses correctly, and writes down what it did.
"""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path

import pytest

from chimera.carry.hedge import HedgeState
from chimera.carry.ledger import LedgerError
from chimera.demo.decision_log import RecordKind, iso_minute
from chimera.demo.fixtures import MinuteShape, SyntheticFeed
from chimera.demo.rules import HedgeTarget, RuleError, RuleRegistry
from chimera.demo.runner import RecoveryCause, RunnerError, RunnerState
from chimera.recorder.contract import load_recorder_contract
from tests.demo_harness import CARRY_PARAMS, DAY, NEXT_DAY, build, campaign_config

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# STARTUP -> SELF_CHECK -> RECOVER -> READY
# ---------------------------------------------------------------------------
def test_a_clean_start_reaches_ready_and_records_startup(tmp_path):
    harness = build(tmp_path)
    assert harness.runner.state is RunnerState.READY
    records = harness.records()
    assert records[0]["kind"] == RecordKind.STARTUP.value
    assert records[0]["prev_hash"].endswith(
        "0" * 64
    ), "a campaign's first record chains to zero"


def test_the_startup_record_says_the_protocol_is_not_frozen(tmp_path):
    """A campaign whose S2 protocol is unfrozen must say so in its own evidence."""
    harness = build(tmp_path)
    startup = harness.records()[0]
    assert startup["protocol_frozen"] is False
    assert "protocol_hash" not in startup


def test_a_dirty_tree_halts_unless_the_run_is_a_soak(tmp_path):
    harness = build(tmp_path, start=False)
    harness.runner.software["dirty"] = True
    assert harness.runner.start() is RunnerState.HALT
    assert "source_identity" in (harness.runner.halt_reason or "")


def test_allow_dirty_is_refused_for_a_campaign_profile(tmp_path):
    state_dir = tmp_path / "state"
    config = campaign_config(state_dir, profile="TEST")
    harness = build(tmp_path, config=config, start=False)
    object.__setattr__(harness.runner.config, "profile", type(config.profile).CAMPAIGN)
    assert harness.runner.start(allow_dirty=True) is RunnerState.HALT
    assert "allow_dirty" in (harness.runner.halt_reason or "")


def test_a_root_with_no_days_refuses_to_start(tmp_path):
    with pytest.raises(RunnerError, match="no minute to start from"):
        build(tmp_path, days=())


def test_a_log_ahead_of_the_runner_state_recovers_rather_than_halting(tmp_path):
    """Section 9.3: the runner appends a RECOVERY record naming it and continues.

    The crash is the one section 9.3's own write order produces: the record was
    committed and the runner-state file naming it was not, so the log is one
    record AHEAD of the state. Before PR-10R the runner halted here and called it
    ``log_behind_state``, which says the opposite of what happened.

    **Driven through `start()`, which is the only entry point there is.** An
    earlier revision of this test set `last_record_hash` by hand after startup
    and called `_recover_log()` directly; it passed while the real path was
    broken, because `_append(STARTUP)` had already overwritten the persisted head
    with the hash of the record the runner itself had just written.
    """
    harness = build(tmp_path)
    harness.run(2)
    config = harness.runner.config
    committed_ms = harness.runner.cursor.last_minute_processed
    records = harness.records()
    state_path = harness.state_dir / "runner_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    # The whole state file rolls back to the PREVIOUS save, which is what a crash
    # between `_append` and `save_state` actually leaves. Rewinding only
    # `last_record_hash` leaves the cursor already past the committed minute --
    # a shape `save_state` cannot produce, and one in which nothing can re-decide
    # that minute however the recovery behaves.
    payload["last_record_hash"] = records[-2]["record_hash"]
    payload["last_minute_processed"] = committed_ms - 60_000
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    resumed = build(tmp_path, config=config)

    written = [r for r in resumed.records() if r["kind"] == RecordKind.RECOVERY.value]
    assert len(written) == 1, "the crash window is recorded, not passed over in silence"
    recovery = written[0]["recovery"]
    assert recovery["cause"] == RecoveryCause.LOG_AHEAD_OF_STATE.value
    assert resumed.runner.state is RunnerState.READY, "and the campaign continues"

    # The committed record's OWN minute, not the last one that completed. This
    # used to be `last_minute_processed`, which `mark_processed`/`save_state`
    # ordering makes the PREVIOUS minute in every crash of this shape.
    committed = records[-1]
    assert recovery["affected_minute"] == committed["minute"]

    # And nothing is excluded: that record is complete, canonical and correctly
    # linked, so a replay is compared against it like any other. This assertion
    # used to be `assert ...["evidence_excluded_minute"]` -- truthiness, which
    # any minute at all satisfies, including the wrong one it was naming.
    assert recovery["evidence_excluded_minute"] is None

    # The cursor adopted the log's minute, so the minute is NOT decided twice.
    # Catching up is what makes this assertion able to fail: it resumes from the
    # CURSOR, so a cursor left behind decides the committed minute a second time.
    resumed.runner.catch_up(now_ms=committed_ms + 60_000)
    decisions = [
        r
        for r in resumed.records()
        if r["kind"] == RecordKind.DECISION.value and r["minute"] == committed["minute"]
    ]
    assert len(decisions) == 1, (
        "the committed minute was decided again after recovery, so the log holds "
        "two DECISION records for one minute"
    )


def test_recovery_names_the_minute_whose_record_was_lost_not_the_one_before(tmp_path):
    """Section 9.3's excluded minute has to be the minute the crash touched.

    `mark_processed` runs before `_append` and `save_state` after it, so the
    persisted cursor always names the PREVIOUS minute when a record is lost.
    Reading the excluded minute off it excluded a minute whose record is
    committed and comparable -- dropping real evidence -- while leaving the
    minute that actually has no record in the comparison, where it diverged.
    """
    harness = build(tmp_path)
    harness.run(3)
    config = harness.runner.config
    completed = harness.runner.cursor.last_minute_processed
    ledger = harness.runner.position.ledger
    # The 9.3 window: the ledger booked, the record never written.
    ledger.state.funding_received += D("24.982411236")
    ledger.state.settled.append(1789804800000000000)
    ledger.save()

    resumed = build(tmp_path, config=config)
    recovery = [r for r in resumed.records() if r["kind"] == RecordKind.RECOVERY.value][0][
        "recovery"
    ]

    excluded = recovery["evidence_excluded_minute"]
    completed_iso = [
        r["minute"] for r in harness.records() if r["kind"] == RecordKind.DECISION.value
    ][-1]
    assert excluded == recovery["affected_minute"]
    assert excluded != completed_iso, (
        "the excluded minute is the last COMPLETED one, whose record is committed "
        "and comparable; excluding it drops real evidence"
    )
    # It is the next minute -- the one whose record the crash lost.
    assert excluded == iso_minute((completed + 60_000) * 1_000_000)


def test_a_lost_record_over_a_moved_ledger_is_recovered(tmp_path):
    """Section 9.3's own window: the state files were written, the record was not.

    A crash between `ledger.save()` and the FUNDING append moves no quantity, so
    a detector that compared only the legs could not see it -- and the carry
    ledger and the decision log would then disagree by one settlement for the
    rest of the campaign, with the daily report summing the log.
    """
    harness = build(tmp_path)
    harness.run(2)
    config = harness.runner.config
    ledger = harness.runner.position.ledger
    # Exactly what the crash leaves: the ledger booked, nothing recorded.
    ledger.state.funding_received += D("24.982411236")
    ledger.state.settled.append(1789804800000000000)
    ledger.save()

    resumed = build(tmp_path, config=config)

    written = [r for r in resumed.records() if r["kind"] == RecordKind.RECOVERY.value]
    assert len(written) == 1
    assert written[0]["recovery"]["cause"] == RecoveryCause.LOG_BEHIND_STATE.value
    assert "funding" in written[0]["recovery"]["detail"]


def test_a_forged_record_is_refused_and_never_repaired(tmp_path):
    """Section 9.3's other half: a COMPLETE record that is wrong is not a crash."""
    harness = build(tmp_path)
    harness.run(2)
    config = harness.runner.config
    day = sorted((harness.state_dir / "decision_log").glob("*.ndjson"))[-1]
    lines = day.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[-1])
    tampered["ledger_effect"]["equity"] = "999999999.00"
    lines[-1] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    day.write_text("\n".join(lines) + "\n", encoding="utf-8")

    resumed = build(tmp_path, config=config, start=False)
    state = resumed.runner.start()

    assert state is RunnerState.HALT
    assert "log_forged" in (resumed.runner.halt_reason or "")
    # And nothing was repaired: the tampered bytes are still on disk.
    assert (
        json.loads(day.read_text(encoding="utf-8").splitlines()[-1])["ledger_effect"]["equity"]
        == "999999999.00"
    )


def test_a_torn_tail_is_repaired_and_the_repair_is_recorded(tmp_path):
    """A crash between the write and the fsync: recoverable, and evidenced.

    Through `start()`. Appending opens the log and `DecisionLog.open` refuses a
    torn tail, so before PR-10R's ordering fix this raised
    `DecisionLogTailError` out of `start()` and the campaign died on a traceback
    -- with `recover_tail`, written for exactly this, never called.
    """
    harness = build(tmp_path)
    harness.run(2)
    config = harness.runner.config
    harness.runner.shutdown("crash drill")
    day = sorted((harness.state_dir / "decision_log").glob("*.ndjson"))[-1]
    with open(day, "ab") as handle:
        handle.write(b'{"schema":"chimera.decision-record/1","seq":99')

    resumed = build(tmp_path, config=config)

    assert resumed.runner.state is RunnerState.READY
    assert (day.with_name(day.name + ".truncated")).is_file()
    written = [r for r in resumed.records() if r["kind"] == RecordKind.RECOVERY.value]
    assert len(written) == 1
    assert written[0]["recovery"]["cause"] == RecoveryCause.TORN_TAIL.value
    assert written[0]["recovery"]["truncated_bytes"] > 0


def test_a_clean_restart_recovers_nothing(tmp_path):
    """The two-sided control: no crash, no RECOVERY record."""
    harness = build(tmp_path)
    harness.run(2)
    config = harness.runner.config
    harness.runner.shutdown("clean stop")

    resumed = build(tmp_path, config=config)

    assert resumed.runner.state is RunnerState.READY
    assert [r for r in resumed.records() if r["kind"] == RecordKind.RECOVERY.value] == []


# ---------------------------------------------------------------------------
# the tick loop
# ---------------------------------------------------------------------------
def test_a_complete_minute_produces_one_decision_record(tmp_path):
    harness = build(tmp_path)
    outcome = harness.run(1)[0]
    assert outcome.kind is RecordKind.DECISION
    assert harness.runner.state is RunnerState.READY
    decision = [r for r in harness.records() if r["kind"] == "DECISION"][0]
    assert decision["rule"]["id"] == "R1_carry"
    assert decision["inputs"]["contract_hash"].startswith("sha256:")
    assert len(decision["inputs"]["um_minute_digest"]) == 64


def test_the_position_is_hedged_with_zero_imbalance_after_a_decision(tmp_path):
    harness = build(tmp_path)
    harness.run(3)
    assert harness.runner.position.state is HedgeState.HEDGED
    assert harness.runner.position.imbalance() == D("0.000")


def test_an_incomplete_minute_logs_and_does_not_evaluate_a_rule(tmp_path):
    harness = build(tmp_path, shapes={f"spot:{DAY}": {0: MinuteShape(book=False)}})
    outcome = harness.run(1)[0]
    assert outcome.kind is RecordKind.INCOMPLETE_STATE
    assert harness.runner.state is RunnerState.READY
    record = [r for r in harness.records() if r["kind"] == "INCOMPLETE_STATE"][0]
    assert record["veto_or_rejection"]["label"] == "incomplete_state"
    assert "rule" not in record, "no rule evaluates on an incomplete minute"


def test_an_incomplete_minute_still_advances_the_cursor(tmp_path):
    """Otherwise a gap in the recorder's files would be replayed for ever."""
    harness = build(tmp_path, shapes={f"spot:{DAY}": {0: MinuteShape(present=False)}})
    first = harness.first_minute_ms()
    harness.run(1)
    assert harness.runner.cursor.last_minute_processed == first


def test_the_hash_chain_is_intact_across_every_record(tmp_path):
    harness = build(tmp_path)
    harness.run(5)
    harness.runner.shutdown("done")
    records = harness.records()
    assert len(records) >= 7
    for earlier, later in zip(records, records[1:]):
        assert later["prev_hash"] == earlier["record_hash"]
        assert later["seq"] == earlier["seq"] + 1


def test_every_record_carries_the_config_hash_and_software(tmp_path):
    harness = build(tmp_path)
    harness.run(2)
    for record in harness.records():
        assert record["config_hash"].startswith("sha256:")
        assert record["software"]["revision"] == "synthetic"


# ---------------------------------------------------------------------------
# shadow rules
# ---------------------------------------------------------------------------
def test_a_shadow_rule_is_recorded_but_never_sizes_the_position(tmp_path):
    harness = build(tmp_path)
    harness.run(1)
    decision = [r for r in harness.records() if r["kind"] == "DECISION"][0]
    assert {s["rule_id"] for s in decision["shadow"]} == {
        "R2_frozen_logistic",
        "R3_daily_momentum",
    }
    assert all(s["kind"] == "signal_only" for s in decision["shadow"])
    assert decision["rule"]["id"] == "R1_carry"


def test_two_actionable_rules_halt_rather_than_choosing_one(tmp_path):
    """A campaign runs exactly one sizing rule; ambiguity is not resolved silently."""
    harness = build(tmp_path, with_shadow=False)
    from chimera.demo.rules_carry import CarryParams, CarryRule

    second = CarryRule(CarryParams.from_config(CARRY_PARAMS))
    second.rule_id = "R1_carry_twin"
    harness.runner.rules.register(second)
    outcome = harness.run(1)[0]
    assert harness.runner.state is RunnerState.HALT
    assert "more than one rule" in (harness.runner.halt_reason or "")
    assert outcome.kind is RecordKind.HALT


# ---------------------------------------------------------------------------
# HALT
# ---------------------------------------------------------------------------
def test_a_rule_exception_halts_and_never_half_decides(tmp_path):
    harness = build(tmp_path, with_shadow=False)

    class Exploding:
        rule_id = "R_boom"
        version = "1.0.0"
        actionable = True

        def rule_hash(self):
            return "sha256:" + "c" * 64

        def params_hash(self):
            return "sha256:" + "d" * 64

        def evaluate(self, state, portfolio):
            raise RuleError("synthetic")

    harness.runner.rules = RuleRegistry([Exploding()])
    outcome = harness.run(1)[0]
    assert harness.runner.state is RunnerState.HALT
    assert "rule_exception" in (harness.runner.halt_reason or "")
    assert outcome.kind is RecordKind.HALT
    assert not [r for r in harness.records() if r["kind"] == "DECISION"]


def test_closing_a_position_returns_its_cash_and_books_no_phantom_drawdown(tmp_path):
    """Section 6.6: after a close, a flat account's equity IS its cash.

    `book_entry` debited `quantity x spot_entry` for the spot inventory and
    `perp_margin` for the margin, and nothing ever credited either back --
    `emergency_reduce` and `flatten_for_correction` moved both legs to flat and
    touched the carry ledger only to stamp leg marks. So `free_cash` kept the
    whole entry debit while the legs it paid for were gone, and
    `mark_to_market` -- free_cash + Q x spot_close + perp_margin + perp_pnl --
    read about a quarter of capital too low on this position.

    That number is not merely displayed. It goes to `risk.update_equity`, so the
    first ordinary exit of a campaign booked a ~25% drawdown against a 5% limit
    and Aegis halted on a loss that never happened -- with the phantom equity
    written into the evidence log as `ledger_effect` and `position_after`.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    assert harness.runner.position.state is HedgeState.HEDGED
    held = harness.runner.position.mark_to_market(
        harness.runner.cursor.state_for(first + 2 * 60_000)
    ).equity

    harness.runner.flatten("operator closed the position")
    ledger = harness.runner.position.ledger.state
    flat = harness.runner.position.mark_to_market(
        harness.runner.cursor.state_for(first + 3 * 60_000)
    ).equity

    # The entry state is unwound, so a later re-open books a fresh entry rather
    # than being mistaken for one already open.
    assert ledger.quantity == D("0")
    assert ledger.perp_margin == D("0")
    assert ledger.spot_entry is None and ledger.perp_entry is None
    assert ledger.entry_basis is None

    # A flat account holds only cash, so those two are the same number.
    assert flat == ledger.free_cash

    # And equity is continuous across the close: it falls by the exit's own
    # costs, not by the position's notional.
    assert held - flat < D("500"), f"closing moved equity by {held - flat}"

    # The identity the ledger's own fields have to satisfy, computed from them
    # rather than from the implementation: what is gone from capital is exactly
    # the frictions paid minus what was realised.
    assert flat == ledger.capital - ledger.fees - ledger.slippage + ledger.realised

    # The realised PnL is the LEGS' own, not a number this ledger computed for
    # itself. Checking it against the carry ledger's copy would be circular --
    # both would read zero if the reduction booked nothing -- so the oracle is
    # the two executors, which realise against their own fills.
    legs_realised = (
        harness.runner.position.spot.ledger.realised_pnl
        + harness.runner.position.perp.ledger.realised_pnl
    )
    assert legs_realised != D("0"), "the legs realised nothing, so this proves nothing"
    assert ledger.realised == legs_realised

    # Aegis was handed the true equity, so no drawdown rule fired.
    assert harness.runner.risk.current_drawdown() < 0.01
    assert not harness.runner.risk.state.halted


def test_a_liquidation_reduction_also_returns_the_position_s_cash(tmp_path):
    """The same accounting on section 6.7's path, which PR-10R made reachable.

    `emergency_reduce` is the one this PR put on the tick loop, so a campaign
    that never sees an operator flatten still reaches it -- and it halts Aegis
    first, which means the phantom drawdown landed on an already-halted engine
    and was written into the HALT record's `position_after` as the campaign's
    closing equity.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    _erode_to_liquidation(harness, first + 3 * 60_000)
    before = harness.runner.position.ledger.state
    cash_before = before.free_cash
    # The entry record, which the reduction code does not write: the oracle.
    principal = before.quantity * before.spot_entry
    margin = before.perp_margin
    realised_before = before.realised
    frictions_before = before.fees + before.slippage

    harness.runner.tick(first + 3 * 60_000)
    ledger = harness.runner.position.ledger.state
    assert harness.runner.position.state is HedgeState.FLAT
    assert ledger.quantity == D("0") and ledger.perp_margin == D("0")

    # What came back is the inventory at cost, plus the margin, plus what the
    # legs realised, less what the exit cost. Every term on the right is read
    # from the ENTRY record or from the executors, never from the reduction.
    returned = ledger.free_cash - cash_before
    exit_frictions = (ledger.fees + ledger.slippage) - frictions_before
    assert (
        returned == principal + margin + (ledger.realised - realised_before) - exit_frictions
    )
    assert returned > margin, "the margin alone was not returned"


def test_a_rule_that_raises_something_other_than_ruleerror_still_halts(tmp_path):
    """The negative control: a rule may raise anything, and none of it decides."""
    harness = build(tmp_path, with_shadow=False)

    class Weird:
        rule_id = "R_weird"
        version = "1.0.0"
        actionable = True

        def rule_hash(self):
            return "sha256:" + "e" * 64

        def params_hash(self):
            return "sha256:" + "f" * 64

        def evaluate(self, state, portfolio):
            raise ZeroDivisionError("synthetic")

    harness.runner.rules = RuleRegistry([Weird()])
    harness.run(1)
    assert harness.runner.state is RunnerState.HALT
    assert "rule_exception" in (harness.runner.halt_reason or "")


def test_the_kill_switch_halts_the_tick(tmp_path):
    harness = build(tmp_path)
    (harness.state_dir / "KILL_SWITCH").write_text("stop", encoding="utf-8")
    outcome = harness.run(1)[0]
    assert harness.runner.state is RunnerState.HALT
    assert outcome.kind is RecordKind.HALT
    assert "kill_switch" in (harness.runner.halt_reason or "")


def test_the_first_halt_reason_is_kept(tmp_path):
    harness = build(tmp_path)
    harness.runner._halt("first")
    harness.runner._halt("second")
    assert harness.runner.halt_reason == "first"


def test_a_halt_record_is_written(tmp_path):
    harness = build(tmp_path)
    harness.runner._halt("synthetic halt")
    halts = [r for r in harness.records() if r["kind"] == "HALT"]
    assert halts and halts[-1]["veto_or_rejection"]["detail"] == "synthetic halt"


# ---------------------------------------------------------------------------
# operator commands
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("note", ["", "   "])
def test_flatten_and_resume_require_a_note(tmp_path, note):
    harness = build(tmp_path)
    harness.run(1)
    with pytest.raises(RunnerError, match="requires an operator note"):
        harness.runner.flatten(note)
    harness.runner._halt("synthetic")
    with pytest.raises(RunnerError, match="requires an operator note"):
        harness.runner.resume(note)


def test_flatten_reduces_to_flat_and_records_the_note(tmp_path):
    harness = build(tmp_path)
    harness.run(2)
    assert harness.runner.position.state is HedgeState.HEDGED
    harness.runner.flatten("operator pulled the position by hand")
    assert harness.runner.position.leg("spot").is_flat
    assert harness.runner.position.leg("perp").is_flat
    record = [r for r in harness.records() if r["kind"] == "OPERATOR"][-1]
    assert record["operator"]["command"] == "flatten"
    assert "by hand" in record["operator"]["note"]


def test_resume_is_refused_when_the_runner_is_not_halted(tmp_path):
    harness = build(tmp_path)
    with pytest.raises(RunnerError, match="not halted"):
        harness.runner.resume("nothing was wrong")


def test_resume_returns_to_ready_and_records_what_was_cleared(tmp_path):
    harness = build(tmp_path)
    harness.run(1)
    harness.runner._halt("synthetic cause")
    harness.runner.resume("checked the feed by hand; the gap was the recorder restarting")
    assert harness.runner.state is RunnerState.READY
    assert harness.runner.halt_reason is None
    record = [r for r in harness.records() if r["kind"] == "RESUME"][-1]
    assert record["operator"]["cleared"] == "synthetic cause"


def test_shutdown_writes_a_record_and_persists(tmp_path):
    harness = build(tmp_path)
    harness.run(1)
    harness.runner.shutdown("SIGTERM")
    assert harness.runner.state is RunnerState.SHUTDOWN
    assert harness.records()[-1]["kind"] == "SHUTDOWN"
    payload = json.loads((harness.state_dir / "runner_state.json").read_text(encoding="utf-8"))
    assert payload["last_record_hash"] == harness.records()[-1]["record_hash"]


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------
def test_a_restart_resumes_at_the_next_unprocessed_minute(tmp_path):
    harness = build(tmp_path)
    harness.run(3)
    harness.runner.shutdown("stop")
    last = harness.runner.cursor.last_minute_processed

    resumed = build(tmp_path, start=False)
    assert resumed.runner.cursor.last_minute_processed == last
    assert resumed.runner.cursor.next_minute_ms() == last + 60_000


def test_a_runner_state_under_a_foreign_schema_is_refused(tmp_path):
    harness = build(tmp_path)
    harness.run(1)
    path = harness.state_dir / "runner_state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = "something.else/9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunnerError, match="declares schema"):
        build(tmp_path, start=False)


def test_an_unreadable_runner_state_is_refused_rather_than_reset(tmp_path):
    """Resetting would reprocess minutes that already have decision records."""
    harness = build(tmp_path)
    harness.run(1)
    (harness.state_dir / "runner_state.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(RunnerError, match="could not be read"):
        build(tmp_path, start=False)


# ---------------------------------------------------------------------------
# catch-up
# ---------------------------------------------------------------------------
def test_catch_up_decides_at_most_max_catchup_minutes(tmp_path):
    """Section 2.2 line 120's first clause: only the recent minutes are decided.

    ``now_ms`` bounds the pending window. Without one the drain runs to the end
    of the fixture's day, and since PR-10R accounts for every pending minute
    rather than abandoning the surplus, that is 1437 SKIPPED_STALE records with
    an fsync each -- minutes of wall clock to assert something about three.
    """
    harness = build(tmp_path, config=None)
    limit = int(harness.runner.config.runner_setting("max_catchup_minutes"))
    first = harness.first_minute_ms()
    outcomes = harness.runner.catch_up(now_ms=first + 9 * 60_000)
    decided = [o for o in outcomes if o.kind is not RecordKind.SKIPPED_STALE]
    assert len(decided) == limit


def test_catch_up_honours_a_configured_limit(tmp_path):
    state_dir = tmp_path / "state"
    config = campaign_config(state_dir, runner={"max_catchup_minutes": 2})
    harness = build(tmp_path, config=config)
    first = harness.first_minute_ms()
    outcomes = harness.runner.catch_up(now_ms=first + 9 * 60_000)
    assert len([o for o in outcomes if o.kind is not RecordKind.SKIPPED_STALE]) == 2


def test_catch_up_accounts_for_every_pending_minute(tmp_path):
    """Section 2.2 line 120's second clause: older minutes are LOGGED as skipped.

    The property is that the campaign's log has no hole. Before PR-10R the
    minutes beyond the limit were abandoned with no record at all, so a restart
    after an outage left a gap nothing in the log named -- and it abandoned the
    NEWEST minutes rather than the stalest, deciding the oldest ones at the
    oldest book.
    """
    state_dir = tmp_path / "state"
    config = campaign_config(state_dir, runner={"max_catchup_minutes": 3})
    harness = build(tmp_path, config=config)
    first = harness.first_minute_ms()
    # Ten pending minutes: the newest three are decided, the oldest seven are not.
    outcomes = harness.runner.catch_up(now_ms=first + 9 * 60_000)

    assert [o.minute_ms for o in outcomes] == [first + i * 60_000 for i in range(10)]
    stale = [o for o in outcomes if o.kind is RecordKind.SKIPPED_STALE]
    decided = [o for o in outcomes if o.kind is not RecordKind.SKIPPED_STALE]
    assert len(stale) == 7 and len(decided) == 3
    assert [o.minute_ms for o in decided] == [first + i * 60_000 for i in (7, 8, 9)]

    records = {r["minute"]: r for r in harness.records()}
    skipped = [r for r in harness.records() if r["kind"] == RecordKind.SKIPPED_STALE.value]
    assert len(skipped) == 7
    assert skipped[0]["catch_up"] is True
    assert skipped[0]["stale"]["max_catchup_minutes"] == 3
    assert skipped[0]["stale"]["age_minutes"] == 9
    assert records  # every processed minute reached the log


def test_a_skipped_stale_minute_changes_no_position(tmp_path):
    """ "No position change is executed" -- asserted, not assumed."""
    state_dir = tmp_path / "state"
    config = campaign_config(state_dir, runner={"max_catchup_minutes": 1})
    harness = build(tmp_path, config=config)
    first = harness.first_minute_ms()
    before = harness.runner._position_block()
    harness.runner.catch_up(now_ms=first + 4 * 60_000)
    stale = [r for r in harness.records() if r["kind"] == RecordKind.SKIPPED_STALE.value]
    assert len(stale) == 4
    # The four stale minutes ran before the one decided minute, so the position
    # at the end of them is still the one the run started with.
    assert before == {
        "hedge_state": "FLAT",
        "spot_qty": "0",
        "perp_qty": "0",
        "imbalance": "0",
    }
    assert all("execution" not in r for r in stale)
    assert all("signal" not in r for r in stale)


# ---------------------------------------------------------------------------
# replay determinism (the property section 10 rests on)
# ---------------------------------------------------------------------------
def test_two_runs_over_the_same_files_write_the_same_decision_bytes(tmp_path):
    """Byte-for-byte, excluding only the fields that are identity rather than content."""

    def run(where):
        harness = build(where)
        harness.run(4)
        harness.runner.shutdown("done")
        return [
            {k: v for k, v in r.items() if k not in ("record_hash", "prev_hash", "seq")}
            for r in harness.records()
            if r["kind"] == "DECISION"
        ]

    first = run(tmp_path / "a")
    second = run(tmp_path / "b")
    assert first == second, "the same files must produce the same decisions"
    assert first, "the comparison is vacuous unless decisions were made"


# ---------------------------------------------------------------------------
# The acceptance shape, shortened so it can run in the suite.
# ---------------------------------------------------------------------------
def test_a_long_run_with_faults_never_lets_hedged_lie_about_balance(tmp_path):
    """Section 10's acceptance shape, at a length the suite can afford.

    The full run is 48 hours (2880 minutes) over two synthetic days with a fault
    schedule; it was executed for the pull request and its result is recorded
    there. This is the same run at 300 minutes, which still crosses no day
    boundary but does meet every injected fault, so the property is exercised
    continuously rather than only at the edges the unit tests poke.

    THE invariant: whenever the state is HEDGED, the two legs hold the same
    quantity. Asserted on every single minute, not at the end -- a check that
    only looked at the final state would pass a run that was briefly one-sided
    in the middle, which is the exact failure section 6.3 exists to prevent.
    """
    from chimera.demo.faults import Fault, FaultSchedule, ScheduledFault

    schedule = FaultSchedule(
        [
            ScheduledFault(37, Fault.MISSING_MINUTE),
            ScheduledFault(120, Fault.MISSING_BOOK),
            ScheduledFault(200, Fault.MISSING_MINUTE),
        ]
    )
    harness = build(
        tmp_path,
        days=(DAY, NEXT_DAY),
        shapes={f"spot:{DAY}": schedule.minute_shapes("spot")},
    )
    first = harness.first_minute_ms()

    kinds: dict[str, int] = {}
    for index in range(540):
        assert harness.runner.state is not RunnerState.HALT, f"halted at minute {index}"
        outcome = harness.tick(first + index * 60_000)
        kinds[outcome.kind.value] = kinds.get(outcome.kind.value, 0) + 1
        if harness.runner.position.state is HedgeState.HEDGED:
            assert harness.runner.position.imbalance() == D(
                "0.000"
            ), f"HEDGED with an imbalance at minute {index}"

    harness.runner.shutdown("acceptance")
    assert kinds["INCOMPLETE_STATE"] == 3, "every injected fault was met"
    assert kinds["DECISION"] == 537

    # The record counts, derived from the plan rather than from the run.
    #
    # DECISION 537: 540 minutes less the three the faults made incomplete.
    # INCOMPLETE_STATE 3: minutes 37, 120 and 200.
    # FUNDING 1: the fixture settles at 00:00, 08:00 and 16:00; the hedge opens
    #   AT 00:00, so section 6.9's `open_instant < settlement` excludes the first,
    #   16:00 is beyond minute 540, and 08:00 is minute 480 -- booked on minute
    #   479, the minute whose CLOSE is the settlement instant.
    # RECONCILIATION 9: minute 0 because the hedge opened there, then every
    #   sixtieth processed minute after the last reconciliation -- 60, then 121
    #   rather than 120 because minute 120 is incomplete and reconciliation lives
    #   in the decision path, then 181, 241, 301, 361, 421, 481.
    # STARTUP 1, SHUTDOWN 1. Nothing halted, nothing was recovered, nothing was
    #   skipped as stale and nothing touched liquidation.
    counts: dict[str, int] = {}
    for record in harness.records():
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    assert counts == {
        "STARTUP": 1,
        "DECISION": 537,
        "INCOMPLETE_STATE": 3,
        "FUNDING": 1,
        "RECONCILIATION": 9,
        "SHUTDOWN": 1,
    }
    reconciled = [
        r["minute"] for r in harness.records() if r["kind"] == RecordKind.RECONCILIATION.value
    ]
    assert [m[11:16] for m in reconciled] == [
        "00:00",
        "01:00",
        "02:01",
        "03:01",
        "04:01",
        "05:01",
        "06:01",
        "07:01",
        "08:01",
    ]

    records = harness.records()
    for earlier, later in zip(records, records[1:]):
        assert later["prev_hash"] == earlier["record_hash"]
        assert later["seq"] == earlier["seq"] + 1


def test_a_held_position_is_not_rehedged_every_minute(tmp_path):
    """Section 6.4: "deliberately no periodic strategy rehedge".

    The rule re-sizes on every minute because the book moves. If the runner
    passed that straight through, it would trade on every tick -- which the
    48-hour acceptance run showed ends with Aegis's rate limiter halting the
    campaign. The runner acts on the entry and the exit and holds in between.
    """
    harness = build(tmp_path)
    harness.run(1)
    assert harness.runner.position.state is HedgeState.HEDGED
    opened = harness.runner.position.leg("spot").quantity
    assert opened > D("0")

    harness.run(20, start=harness.first_minute_ms() + 60_000)
    assert harness.runner.position.leg("spot").quantity == opened, "the size did not drift"
    assert harness.runner.state is RunnerState.READY, "and Aegis never had to halt it"

    executions = [
        record
        for record in harness.records()
        if record["kind"] == "DECISION" and record["execution"]
    ]
    assert len(executions) == 1, "exactly one entry, and no rehedge after it"


# ---------------------------------------------------------------------------
# No rule invents a parameter, and no shadow rule can size a position.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "missing",
    [
        "min_basis",
        "max_notional_fraction",
        "step_size",
        "min_notional",
        "spot_leg_fraction",
        "spot_fee_rate",
        "perp_fee_rate",
        "slippage_rate",
    ],
)
def test_the_carry_rule_refuses_every_missing_parameter_rather_than_defaulting(missing):
    """Section 17's S3 STOP list, at the rule rather than at the config.

    A default written into the rule module would be a parameter chosen by its
    author, before the S2 protocol meant to freeze it and with the recorded data
    already on disk. Each key is checked separately so a refusal that only
    covered the first one could not pass.
    """
    from chimera.demo.rules_carry import CarryParams

    params = {key: value for key, value in CARRY_PARAMS.items() if key != missing}
    with pytest.raises(RuleError, match=missing):
        CarryParams.from_config(params)


def test_the_carry_rule_accepts_a_complete_parameter_set():
    """The negative control: the refusal is about absence, not about the rule."""
    from chimera.demo.rules_carry import CarryParams

    assert CarryParams.from_config(dict(CARRY_PARAMS)).min_basis == D("10")


def test_a_shadow_rule_refuses_a_missing_coefficient_set():
    from chimera.demo.rules_shadow import ShadowParams

    with pytest.raises(RuleError, match="coefficients"):
        ShadowParams.from_config("R2", {"intercept": "0", "threshold": "1"})


def test_a_shadow_rule_with_the_wrong_number_of_coefficients_is_refused():
    from chimera.demo.rules_shadow import FrozenLogisticRule, ShadowParams

    params = ShadowParams.from_config(
        "R2", {"coefficients": ["1"], "intercept": "0", "threshold": "1"}
    )
    with pytest.raises(RuleError, match="coefficients"):
        FrozenLogisticRule(params)


def test_a_rule_that_declares_itself_signal_only_may_not_return_a_target(tmp_path):
    """Defence in depth, in the direction the type system cannot cover.

    A shadow rule returning `SignalOnly` cannot reach an executor because
    `HedgedPosition.plan` takes a `HedgeTarget`. The opposite mistake -- a rule
    that declares ``actionable = False`` and then returns a `HedgeTarget` -- is
    not a type error, and would be a signal-only rule quietly acquiring the
    ability to trade. The runner refuses it.
    """
    from chimera.carry.hedge import HedgeTarget
    from chimera.demo.rules import RuleDecision

    harness = build(tmp_path, with_shadow=False)

    class Liar:
        rule_id = "R_liar"
        version = "1.0.0"
        actionable = False  # says it only observes

        def rule_hash(self):
            return "sha256:" + "1" * 64

        def params_hash(self):
            return "sha256:" + "2" * 64

        def evaluate(self, state, portfolio):
            return RuleDecision(  # ... and then sizes a position anyway
                rule_id=self.rule_id,
                rule_hash=self.rule_hash(),
                target=HedgeTarget(D("0.5")),
                reason="synthetic",
                inputs_hash="sha256:" + "3" * 64,
                params_hash=self.params_hash(),
            )

    harness.runner.rules = RuleRegistry([Liar()])
    harness.run(1)
    assert harness.runner.state is RunnerState.HALT
    assert "may never size a position" in (harness.runner.halt_reason or "")


def test_the_declared_shadow_rules_do_not_trip_that_check(tmp_path):
    """The negative control: the real shadow rules pass it every minute."""
    harness = build(tmp_path, with_shadow=True)
    harness.run(3)
    assert harness.runner.state is RunnerState.READY


def test_each_rule_declares_whether_it_may_size_a_position():
    """The declaration itself, pinned.

    `RuleDecision.is_actionable` is decided by the TYPE of the target, and that
    is the guarantee that matters. This attribute is the second half of the
    cross-check above -- the half that catches a rule declaring one thing and
    returning another -- so it is asserted rather than assumed. Without this,
    flipping a shadow rule's declaration would change nothing any test noticed,
    which is exactly how a defence-in-depth check rots.
    """
    from chimera.demo.rules_carry import CarryRule
    from chimera.demo.rules_shadow import DailyMomentumRule, FrozenLogisticRule

    assert CarryRule.actionable is True, "R1 is the one rule that sizes a position"
    assert FrozenLogisticRule.actionable is False
    assert DailyMomentumRule.actionable is False


# ---------------------------------------------------------------------------
# PR-10R: funding (section 6.5)
# ---------------------------------------------------------------------------
#: The settlement mark every funding witness books against, pinned so the
#: hand-traced arithmetic below does not move with the fixture's price path.
WITNESS_MARK = "30128.33"


def _settlements_at_hour_one(harness, *, rate: str = "0.0001", **kwargs):
    """Put the campaign's settlement one hour in, not eight.

    The cadence is fixture data: what the funding witnesses test is the window,
    the arithmetic and the exactly-once gate, none of which depends on how far
    apart the venue schedules settlements. At the fixture's real 8-hourly cadence
    each of these tests had to tick 485 minutes to reach one, which put
    `tests/test_demo_runner.py` alone past CI's whole job budget.

    ``mark_price`` is pinned to :data:`WITNESS_MARK` so the settlement's own mark
    -- and therefore every hand-computed number in these tests -- is the same
    value it would have had at 08:00.
    """
    harness.feed.write_settlements(
        [DAY],
        hours=(1,),
        rates={(DAY, 1): rate},
        mark_price=WITNESS_MARK,
        **kwargs,
    )
    harness.runner.cursor._settlements = None


def _run_to_first_settlement(harness, *, minutes: int = 65):
    """Tick past the settlement at minute 60, which is booked on minute 59."""
    first = harness.first_minute_ms()
    for index in range(minutes):
        harness.tick(first + index * 60_000)
    return first


def _funding_records(harness):
    return [r for r in harness.records() if r["kind"] == RecordKind.FUNDING.value]


def test_a_settlement_inside_the_window_is_booked_once_and_recorded(tmp_path):
    """Section 6.5 end to end, with the arithmetic checked by hand.

    The settlement carries rate 0.0001 and its own mark 30128.33. The perpetual
    leg is SHORT 8.292 BTC, and amendment A10 gives
    ``-sign(side) * notional * rate`` = ``+1 * 8.292 * 30128.33 * 0.0001`` =
    ``+24.982411236``: a SHORT RECEIVES a positive rate. Every one of those
    numbers is written out rather than read back off the implementation.
    """
    harness = build(tmp_path)
    _settlements_at_hour_one(harness)
    _run_to_first_settlement(harness)

    records = _funding_records(harness)
    assert len(records) == 1
    funding = records[0]["funding"]
    assert funding["settlement_id"] == "1789779600000"
    assert funding["leg"] == "perp" and funding["side"] == "SHORT"
    assert funding["rate"] == "0.0001"
    assert funding["mark_price"] == "30128.33"
    assert funding["quantity"] == "8.292"
    assert funding["notional"] == "249824.11236"
    assert funding["cash_flow"] == "24.982411236"
    assert funding["direction"] == "received"
    assert funding["venue_symbol"] == "BTCUSDT"
    assert funding["symbol"] == "BTC/USDT:USDT"
    # And the ledger agrees with the record, which is the claim that matters.
    assert records[0]["ledger_effect"]["funding"] == "24.982411236"
    assert harness.runner.position.ledger.state.funding_received == D("24.982411236")
    assert harness.runner.position.ledger.state.funding_paid == D("0")


def test_a_settlement_at_the_open_instant_is_not_charged(tmp_path):
    """Section 6.9's lower tie: the position did not hold through it.

    The fixture's first settlement is at 00:00, the same instant the hedge opens,
    and the window is ``open_instant < settlement``. The proof that this is the
    boundary and not an accident is the second settlement, at 08:00, which IS
    charged in the test above.
    """
    harness = build(tmp_path)
    opened_at = harness.first_minute_ms()
    harness.run(30, start=opened_at)
    assert harness.runner.position.state is HedgeState.HEDGED
    # The minute's CLOSE, not its open: a leg fills at `perp_close`, so that is
    # when the position came into existence and when its funding window opens.
    assert harness.runner.position.ledger.state.open_instant_ns == (
        opened_at * 1_000_000 + 60_000_000_000
    )
    assert harness.runner.position.ledger.state.settled == []
    assert _funding_records(harness) == []


def test_a_duplicated_settlement_row_books_nothing_and_writes_no_second_record(tmp_path):
    """The exactly-once gate, against a settlements file holding the row twice."""
    harness = build(tmp_path)
    _settlements_at_hour_one(harness, duplicate=[(DAY, 1)])
    rows = harness.runner.cursor.settlements()
    assert len([r for r in rows if r["funding_time_ms"] == 1789779600000]) == 2

    _run_to_first_settlement(harness)
    assert len(_funding_records(harness)) == 1
    assert harness.runner.position.ledger.state.funding_received == D("24.982411236")


def test_a_settlement_is_not_rebooked_by_a_later_minute(tmp_path):
    """Ticking on past a settlement does not charge it again."""
    harness = build(tmp_path)
    _settlements_at_hour_one(harness)
    _run_to_first_settlement(harness, minutes=100)
    assert len(_funding_records(harness)) == 1
    assert len(harness.runner.position.ledger.state.settled) == 1


def test_funding_survives_a_restart_across_the_settlement_boundary(tmp_path):
    """A second process over the same files books nothing the first already did.

    Both dedup layers are persisted -- the carry ledger's settled instants and
    the perpetual executor's ``applied_funding`` -- so the restart is the real
    test of the claim that neither of them lives only in memory.
    """
    harness = build(tmp_path)
    _settlements_at_hour_one(harness)
    first = _run_to_first_settlement(harness)
    harness.runner.shutdown("restarting")
    booked = harness.runner.position.ledger.state.settled
    received = harness.runner.position.ledger.state.funding_received
    assert len(booked) == 1

    resumed = build(tmp_path, config=harness.runner.config)
    for index in range(65, 80):
        resumed.tick(first + index * 60_000)

    assert resumed.runner.position.ledger.state.settled == booked
    assert resumed.runner.position.ledger.state.funding_received == received
    assert len(_funding_records(resumed)) == 1


def test_a_settlement_with_no_mark_price_halts_rather_than_being_priced(tmp_path):
    """Amendment A4: the recorded mark is not reconstructed from anything."""
    harness = build(tmp_path)
    _settlements_at_hour_one(harness)
    path = harness.feed.normalizer.settlements_path("um")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["mark_price"] = None
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows) + "\n",
        encoding="utf-8",
    )
    harness.runner.cursor._settlements = None

    _run_to_first_settlement(harness)
    assert harness.runner.state is RunnerState.HALT
    assert "funding_unbookable" in (harness.runner.halt_reason or "")
    assert _funding_records(harness) == []


def test_a_settlement_for_another_instrument_is_refused(tmp_path):
    """The venue symbol is checked against the contract, never relabelled."""
    harness = build(tmp_path)
    _settlements_at_hour_one(harness)
    path = harness.feed.normalizer.settlements_path("um")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["symbol"] = "ETHUSDT"
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows) + "\n",
        encoding="utf-8",
    )
    harness.runner.cursor._settlements = None

    _run_to_first_settlement(harness)
    assert harness.runner.state is RunnerState.HALT
    assert "ETHUSDT" in (harness.runner.halt_reason or "")


def test_a_torn_funding_booking_disputes_on_restart(tmp_path):
    """The one crash window funding has, and it fails closed rather than silently.

    `settle_funding` books the perpetual executor's ledger (which persists with
    that leg's store) and then the carry ledger (which persists on the next
    save). A crash between them leaves the money moved and the carry ledger not
    knowing it, and every other check on restart passes.
    """
    harness = build(tmp_path)
    _settlements_at_hour_one(harness)
    _run_to_first_settlement(harness)
    ledger = harness.runner.position.ledger
    assert len(ledger.state.settled) == 1

    # Exactly what the crash leaves: the perpetual leg remembers the settlement
    # and the carry ledger does not.
    ledger.state.settled.clear()
    ledger.save()

    assert harness.runner.position.unbooked_funding_instants() == (1789779600000000000,)
    outcome = harness.runner.position.reconstruct()
    assert outcome.state is HedgeState.DISPUTED
    assert "funding_booking_torn" in (ledger.disputed or "")


# ---------------------------------------------------------------------------
# PR-10R: reconciliation (section 8.1)
# ---------------------------------------------------------------------------
def _reconciliations(harness):
    return [r for r in harness.records() if r["kind"] == RecordKind.RECONCILIATION.value]


def test_reconciliation_runs_after_execution_and_then_hourly(tmp_path):
    """Section 8.1's RECONCILIATION row, both arms, on the same run.

    Minute 0 opens the hedge, so it reconciles because something executed;
    minutes 60, 120 and 180 reconcile because sixty processed minutes have
    passed. The literal minutes are written out rather than derived from the
    schedule under test.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(200, start=first)

    records = _reconciliations(harness)
    assert [r["reconciliation"]["trigger"] for r in records] == [
        "after_execution",
        "periodic",
        "periodic",
        "periodic",
    ]
    minutes = [r["minute"] for r in records]
    assert minutes == [
        "2026-09-19T00:00:00+00:00",
        "2026-09-19T01:00:00+00:00",
        "2026-09-19T02:00:00+00:00",
        "2026-09-19T03:00:00+00:00",
    ]
    first_record = records[0]["reconciliation"]
    assert first_record["outcome"] == "AGREED"
    assert [leg["leg"] for leg in first_record["legs"]] == ["spot", "perp"]
    assert [leg["symbol"] for leg in first_record["legs"]] == [
        "BTC/USDT",
        "BTC/USDT:USDT",
    ]
    assert all(leg["outcome"] == "AGREED" for leg in first_record["legs"])


def test_the_reconciliation_cadence_is_persisted_and_survives_a_restart(tmp_path):
    """A restart every fifty-nine minutes may not postpone the check for ever.

    The anchor is the minute a reconciliation was performed FOR, and it is in the
    runner state file. A build that counted minutes since the process started
    would reconcile at minute 59 of each life and never at an hourly boundary.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(30, start=first)
    assert harness.runner.last_reconcile_minute_ms == first
    harness.runner.shutdown("restart drill")

    persisted = json.loads((harness.state_dir / "runner_state.json").read_text("utf-8"))
    assert persisted["schema"] == "chimera.demo-runner-state/2"
    assert persisted["last_reconcile_minute_ms"] == first

    resumed = build(tmp_path, config=harness.runner.config)
    assert resumed.runner.last_reconcile_minute_ms == first
    # Minute 59 is still inside the hour the first process opened; minute 60 is
    # not, and it reconciles even though this process has only just started.
    before = len(_reconciliations(resumed))
    for index in range(30, 60):
        resumed.tick(first + index * 60_000)
    assert len(_reconciliations(resumed)) == before
    resumed.tick(first + 60 * 60_000)
    assert len(_reconciliations(resumed)) == before + 1


def test_a_version_1_runner_state_reconciles_at_the_next_minute(tmp_path):
    """The documented migration: an absent anchor means "none has been performed"."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(5, start=first)
    harness.runner.shutdown("downgrade drill")

    path = harness.state_dir / "runner_state.json"
    payload = json.loads(path.read_text("utf-8"))
    payload["schema"] = "chimera.demo-runner-state/1"
    payload.pop("last_reconcile_minute_ms")
    path.write_text(json.dumps(payload), encoding="utf-8")

    resumed = build(tmp_path, config=harness.runner.config)
    assert resumed.runner.last_reconcile_minute_ms is None
    before = len(_reconciliations(resumed))
    resumed.tick(first + 5 * 60_000)
    assert len(_reconciliations(resumed)) == before + 1


def test_a_reconciliation_mismatch_fails_closed(tmp_path):
    """Section 8.1: "mismatch -> HALT", and everything that has to come with it.

    The mismatch is produced the only way it can be on a dry-run venue: the
    venue's reported position is made to disagree with the store's. What is
    asserted is the whole fail-closed contract -- the record, the halt, the
    store's dispute, Aegis's own copy of it, and that no increase is planned
    afterwards.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    perp_symbol = harness.runner.position.config.perp_symbol
    local = harness.runner.position.perp.position(perp_symbol)
    assert not local.is_flat

    # The venue now says a different quantity from the one the store holds.
    disagreeing = replace(local, quantity=local.quantity + D("1"))
    harness.runner.position.perp.venue.reported_position = (  # type: ignore[assignment]
        lambda symbol, _p=disagreeing, _s=perp_symbol: (
            _p if symbol == _s else harness.runner.position.perp.position(symbol)
        )
    )
    harness.runner.last_reconcile_minute_ms = None  # force the periodic arm

    harness.tick(first + 3 * 60_000)

    assert harness.runner.state is RunnerState.HALT
    assert "reconciliation_mismatch" in (harness.runner.halt_reason or "")
    records = _reconciliations(harness)
    assert records[-1]["reconciliation"]["outcome"] == "MISMATCH"
    assert records[-1]["veto_or_rejection"]["label"] == "reconciliation_mismatch"
    assert perp_symbol in harness.runner.position.perp.store.state.disputed
    assert perp_symbol in harness.risk.state.reconciliation_disputed
    # A mismatch is evidence, and it is written before the halt that follows it.
    kinds = [r["kind"] for r in harness.records()]
    assert kinds.index(RecordKind.RECONCILIATION.value) < kinds.index(RecordKind.HALT.value)


def test_an_agreement_does_not_clear_a_standing_dispute(tmp_path):
    """Section 7.2: a reconciliation dispute is cleared only by an operator note."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    perp_symbol = harness.runner.position.config.perp_symbol
    harness.risk.note_reconciliation(perp_symbol, "an earlier, unexplained disagreement")

    harness.runner.last_reconcile_minute_ms = None
    harness.tick(first + 3 * 60_000)

    assert _reconciliations(harness)[-1]["reconciliation"]["outcome"] == "AGREED"
    assert perp_symbol in harness.risk.state.reconciliation_disputed


def test_the_operator_resolve_command_clears_a_dispute(tmp_path):
    """The command that could not run at all before PR-10R.

    `DemoRunner.resolve` passed two of the three arguments
    `FuturesExecutor.resolve_reconciliation` requires, so every real invocation
    raised `TypeError` before touching a store.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    perp_symbol = harness.runner.position.config.perp_symbol
    local = harness.runner.position.perp.position(perp_symbol)
    harness.runner.position.perp.reconcile(
        perp_symbol, replace(local, quantity=local.quantity + D("1"))
    )
    harness.risk.note_reconciliation(perp_symbol, "venue disagreed")
    assert perp_symbol in harness.runner.position.perp.store.state.disputed

    outcome = harness.runner.resolve(perp_symbol, "checked both stores by hand")

    assert outcome.kind is RecordKind.OPERATOR
    assert perp_symbol not in harness.runner.position.perp.store.state.disputed
    assert perp_symbol not in harness.risk.state.reconciliation_disputed
    record = harness.records()[-1]
    assert record["operator"]["command"] == "resolve"
    assert record["operator"]["note"] == "checked both stores by hand"
    assert record["operator"]["adopted_qty"] == str(local.quantity)


def test_resolve_refuses_rather_than_clearing_a_dispute_it_cannot_record(tmp_path):
    """Section 8.3: the operator action is the record. No record, no action.

    `tools/demo_run.py` runs `resolve` on a freshly constructed runner and never
    calls `start()`, so the clock had observed nothing. `resolve` saved the
    store, cleared Aegis's copy of the dispute, and only THEN read
    `self.clock.now_ns` -- which raised `RunnerClockError`, a `ValueError` that
    the CLI's `except RunnerError` does not catch. The dispute that freezes
    increases was gone, with nothing in the log to say who cleared it or why.

    Now the clock is read before anything is touched, so an unrecordable resolve
    changes nothing at all -- asserted on all three stores, not just the log.
    """
    harness = build(tmp_path)
    harness.run(2)
    symbol = harness.runner.position.config.perp_symbol
    harness.runner.position.perp.store.state.disputed[symbol] = "reconciliation_mismatch"
    harness.runner.position.perp.store.save()
    harness.runner.risk.note_reconciliation(symbol, "reconciliation_mismatch")

    harness.runner.clock._now_ns = None  # the state a fresh CLI process is in
    records_before = len(harness.records())

    with pytest.raises(RunnerError, match="clock has no observation"):
        harness.runner.resolve(symbol, "operator checked both legs")

    assert symbol in harness.runner.position.perp.store.state.disputed
    assert harness.runner.risk.state.reconciliation_disputed
    assert len(harness.records()) == records_before


def test_a_restart_carries_the_clock_so_the_operator_commands_can_run(tmp_path):
    """The other half: `resolve` must also be able to SUCCEED from a new process.

    `save_state` wrote `clock_now_ns` and nothing read it back, so every new
    process began with a clock that had observed nothing. Refusing safely is
    correct but is not a working command, and section 8.3's resolve is the only
    way out of a reconciliation dispute.
    """
    harness = build(tmp_path)
    harness.run(3)
    symbol = harness.runner.position.config.perp_symbol
    harness.runner.position.perp.store.state.disputed[symbol] = "reconciliation_mismatch"
    harness.runner.position.perp.store.save()
    harness.runner.risk.note_reconciliation(symbol, "reconciliation_mismatch")
    harness.runner.save_state()
    expected = harness.runner.clock.now_ns

    resumed = build(tmp_path, start=False).runner
    assert resumed.clock.started, "a new process began with no instant to record with"
    assert resumed.clock.now_ns == expected

    outcome = resumed.resolve(symbol, "operator checked both legs")
    assert outcome.record_hash
    assert symbol not in resumed.position.perp.store.state.disputed


def test_resolve_neither_clears_nor_overwrites_an_unreadable_carry_ledger(tmp_path):
    """`CarryLedger.open` promises the damaged file is left untouched.

    It kept that promise only while nobody called `save()`. `resolve` cleared
    whatever the ledger was disputing and saved -- and an unreadable ledger is
    represented by a placeholder holding `free_cash == capital` and no position,
    so saving it replaced the only record of the campaign's cash with one saying
    it never traded. The stores still held the position, so the next start
    fail-closed on a mismatch, but the cash was already gone.

    Two guards, both asserted: `resolve` no longer touches a dispute that is not
    this leg's reconciliation, and `save()` refuses an UNREADABLE ledger outright.
    """
    harness = build(tmp_path)
    harness.run(2)
    ledger_path = harness.runner.position.ledger.path
    original = ledger_path.read_bytes()
    assert b'"quantity"' in original

    ledger_path.write_text("{ this is not json", encoding="utf-8")
    resumed = build(tmp_path, start=False).runner
    ledger = resumed.position.ledger
    assert ledger.disputed and ledger.disputed.startswith("ledger_unreadable")

    symbol = resumed.position.config.perp_symbol
    resumed.position.perp.store.state.disputed[symbol] = "reconciliation_mismatch"
    resumed.position.perp.store.save()
    resumed.resolve(symbol, "operator checked both legs")

    # The damaged bytes are still on disk: resolve did not adopt the placeholder.
    assert ledger_path.read_text(encoding="utf-8") == "{ this is not json"
    assert ledger.disputed.startswith("ledger_unreadable")

    # And the guard is in save() itself, not only in resolve's choice not to call it.
    with pytest.raises(LedgerError, match="refusing to overwrite"):
        ledger.save()


def test_resolve_still_refuses_an_empty_note(tmp_path):
    harness = build(tmp_path)
    harness.run(1)
    with pytest.raises(RunnerError, match="operator note"):
        harness.runner.resolve(harness.runner.position.config.perp_symbol, "   ")


# ---------------------------------------------------------------------------
# PR-10R: liquidation touch (section 6.7)
# ---------------------------------------------------------------------------
def _erode_to_liquidation(harness, minute_ms) -> None:
    """Drain the ledger's cash until equity sits just under section 6.7's line.

    Funding-driven equity erosion is what 6.7 says the check exists for, and
    ``free_cash`` is the field funding moves, so draining it is the real
    mechanism. Writing ``last_equity`` directly used to work and no longer does:
    the check marks the position at the minute it is checking, so an injected
    equity is recomputed away -- which is the point of marking there, and which
    means a test that injected one would silently stop forcing a touch.
    """
    position = harness.runner.position
    state = harness.runner.cursor.state_for(minute_ms)
    adverse = state.mark_high or state.mark
    maintenance = (
        position.leg("perp").quantity * adverse * position.config.maintenance_margin_rate
    )
    equity = position.mark_to_market(state).equity
    position.ledger.state.free_cash -= equity - maintenance + D("1")


def _touches(harness):
    return [r for r in harness.records() if r["kind"] == RecordKind.LIQUIDATION_TOUCH.value]


def test_a_liquidation_touch_halts_flattens_and_is_recorded_in_that_order(tmp_path):
    """Section 6.7's consequence, and section 9.3's ordering over it.

    Equity is driven below ``Q * mark * maintenance_margin_rate`` directly, which
    is the funding-driven erosion section 6.7 says the check exists for; no
    threshold and no model is changed to reach it.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    assert harness.runner.position.state is HedgeState.HEDGED
    quantity = harness.runner.position.leg("perp").quantity
    assert quantity > D("0")

    _erode_to_liquidation(harness, first + 3 * 60_000)

    harness.tick(first + 3 * 60_000)

    touches = _touches(harness)
    assert len(touches) == 1
    assert touches[0]["veto_or_rejection"]["label"] == "liquidation_touch"
    assert touches[0]["position_after"]["hedge_state"] == "HEDGED"
    assert touches[0]["position_after"]["perp_qty"] == str(quantity)

    assert harness.risk.state.halted
    assert harness.runner.state is RunnerState.HALT
    assert harness.runner.position.state is HedgeState.FLAT

    kinds = [r["kind"] for r in harness.records()]
    # The touch is written BEFORE the flatten's halt: a crash between them leaves
    # a log that says "touched" over stores that still hold the position, which
    # is true. The other order would leave a log claiming a flatten that never
    # happened.
    assert kinds.index(RecordKind.LIQUIDATION_TOUCH.value) < kinds.index(RecordKind.HALT.value)
    halt = [r for r in harness.records() if r["kind"] == RecordKind.HALT.value][-1]
    assert halt["position_after"]["perp_qty"] == "0"


def test_no_increase_is_possible_after_a_liquidation_touch(tmp_path):
    """ "No new increase can happen after the touch" -- asserted on the next minute."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    _erode_to_liquidation(harness, first + 3 * 60_000)
    harness.tick(first + 3 * 60_000)
    assert harness.runner.position.state is HedgeState.FLAT

    harness.runner.risk.state.halted = True  # it already is; stated for the reader
    intents = harness.runner.position.plan(
        HedgeTarget(D("1")), harness.runner.cursor.state_for(first + 4 * 60_000)
    )
    assert intents == []


def test_an_unknown_liquidation_distance_on_a_held_position_is_refused(tmp_path):
    """Section 7.2: `None` from a non-flat position vetoes, never reads as "far"."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    harness.runner.position.perp.margin = lambda *a, **k: None  # type: ignore[assignment]

    harness.tick(first + 3 * 60_000)

    assert len(_touches(harness)) == 1
    assert harness.runner.state is RunnerState.HALT
    assert harness.runner.position.state is HedgeState.FLAT


def test_a_flat_position_is_never_liquidation_checked(tmp_path):
    """The two-sided control: nothing held, nothing to touch."""
    harness = build(tmp_path)
    harness.runner.position.ledger.state.last_equity = D("1")
    harness.run(1)
    assert _touches(harness) == []
    assert harness.runner.state is not RunnerState.HALT


def test_a_paid_settlement_extends_the_aegis_streak(tmp_path):
    """The other funding direction, end to end, and the Aegis sign with it.

    A10: for a SHORT perpetual leg a NEGATIVE rate is one the position pays, and
    the cost Aegis scores is ``sign(side) * rate`` -- the negation of the cash
    flow -- so a paid settlement extends `funding_adverse_streak`. Without this
    test the sign on the Aegis path could be inverted with the whole suite green,
    and a genuinely adverse streak would reset the counter that is supposed to
    stop the campaign increasing through it.
    """
    harness = build(tmp_path)
    _settlements_at_hour_one(harness, rate="-0.0001")
    _run_to_first_settlement(harness)

    records = _funding_records(harness)
    assert len(records) == 1
    funding = records[0]["funding"]
    assert funding["rate"] == "-0.0001"
    assert funding["cash_flow"] == "-24.982411236"
    assert funding["direction"] == "paid"
    ledger = harness.runner.position.ledger.state
    assert ledger.funding_paid == D("24.982411236")
    assert ledger.funding_received == D("0")
    assert ledger.net_funding == D("-24.982411236")
    # And Aegis scored it as adverse, which is what the funding halt counts.
    assert harness.risk.state.funding_adverse_streak == 1
    assert harness.risk.state.funding_halt is False


def test_three_paid_settlements_raise_the_aegis_funding_halt(tmp_path):
    """Section 7.2's funding halt: after N adverse settlements, no increase."""
    harness = build(tmp_path, days=(DAY, NEXT_DAY))
    harness.feed.write_settlements(
        [DAY, NEXT_DAY],
        rates={(DAY, 8): "-0.0001", (DAY, 16): "-0.0001", (NEXT_DAY, 0): "-0.0001"},
    )
    harness.runner.cursor._settlements = None
    first = harness.first_minute_ms()
    for index in range(1445):
        harness.tick(first + index * 60_000)

    assert len(_funding_records(harness)) == 3
    assert harness.risk.state.funding_adverse_streak == 3
    assert harness.risk.state.funding_halt is True


def test_a_settlement_file_whose_rows_disagree_is_refused(tmp_path):
    """A settlement is published once; two rows that disagree are not resolved."""
    harness = build(tmp_path)
    _settlements_at_hour_one(harness)
    path = harness.feed.normalizer.settlements_path("um")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    contradiction = dict(rows[0])
    contradiction["funding_rate"] = "-0.0009"
    rows.append(contradiction)
    first = harness.first_minute_ms()
    harness.run(2, start=first)  # a good file for the first minutes
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows) + "\n",
        encoding="utf-8",
    )

    # Driven through `DemoRunner.tick` rather than the harness helper, which
    # installs the minute's quote by reading the feed itself and would meet the
    # refusal before the runner did.
    harness.runner.tick(first + 2 * 60_000)
    assert harness.runner.state is RunnerState.HALT
    assert "feed_unreadable" in (harness.runner.halt_reason or "")
    assert "disagree" in (harness.runner.halt_reason or "")


def test_an_unreadable_settlement_row_halts_rather_than_raising(tmp_path):
    """A feed the runner cannot read is a halt, never a traceback out of tick."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(2, start=first)
    path = harness.feed.normalizer.settlements_path("um")
    path.write_text('{"funding_time_ms": "not-a-number", "symbol": "BTCUSDT"}\n', "utf-8")

    outcome = harness.runner.tick(first + 2 * 60_000)

    assert outcome.kind is RecordKind.HALT
    assert harness.runner.state is RunnerState.HALT
    assert "feed_unreadable" in (harness.runner.halt_reason or "")
    assert harness.records()[-1]["kind"] == RecordKind.HALT.value


def test_the_settlements_file_is_reread_when_it_changes(tmp_path):
    """A settlement the recorder writes mid-run is not invisible for the process.

    The cursor used to read the file once for its whole life, so a settlement
    appended after that read was never booked by this process -- and a later one
    would book it at a later minute than the one it belongs to, which a replay of
    the same files would not agree with.
    """
    harness = build(tmp_path)
    path = harness.feed.normalizer.settlements_path("um")
    path.write_text("", encoding="utf-8")
    assert harness.runner.cursor.settlements() == []

    harness.feed.write_settlements([DAY])
    assert len(harness.runner.cursor.settlements()) == 3, "the change was seen"


def test_catch_up_stops_when_no_minute_exists_to_catch_up_to(tmp_path):
    """The cursor answers "one minute later" for ever; the loop must not.

    With the perpetual days gone and a cursor still on disk, an unguarded drain
    walks forward without end writing INCOMPLETE_STATE records for minutes no
    recorder ever wrote -- fabricated evidence, and an unbounded log.
    """
    harness = build(tmp_path)
    harness.run(2)
    before = len(harness.records())
    for parquet in (harness.root / "normalized" / "um" / "1m").glob("*.parquet"):
        parquet.unlink()
    harness.runner.cursor._days.clear()

    assert harness.runner.catch_up() == []
    assert len(harness.records()) == before, "and nothing was written about them"


def test_a_liquidation_price_that_cannot_be_computed_is_refused(tmp_path):
    """Section 7.2: `None` from a non-flat position vetoes, never reads as "far"."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    margin = harness.runner.position.perp.margin

    def no_distance(symbol, mark_price):
        state = margin(symbol, mark_price)
        return None if state is None else replace(state, liquidation_price=None)

    harness.runner.position.perp.margin = no_distance  # type: ignore[assignment]
    harness.tick(first + 3 * 60_000)

    assert len(_touches(harness)) == 1
    assert harness.runner.state is RunnerState.HALT
    assert harness.runner.position.state is HedgeState.FLAT


def test_the_liquidation_check_reads_the_mark_high_and_not_the_close(tmp_path):
    """Section 6.7's own formula: `equity < Q * mark_high * maintenance_margin_rate`.

    Two-sided, on a minute where the two conventions genuinely DISAGREE. The
    synthetic fixture writes ``mark_high == mark_close``, so a test taken from it
    unaltered cannot tell them apart -- and would pass just as happily against
    the close-reading this fixes. The high is therefore raised explicitly, and
    the equity placed between the two thresholds:

    * below ``Q * mark_high * rate``  -> section 6.7 says TOUCHED;
    * at or above ``Q * mark_close * rate`` -> the close-reading says not.

    The close is never above the high, so reading it put the threshold strictly
    below the adopted one -- a deviation in the unsafe direction.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    state = harness.runner.cursor.state_for(first + 3 * 60_000)
    assert state.mark_high is not None and state.mark_high >= state.mark
    assert "mark_high" in state.canonical(), "and it is one of the hashed inputs"

    position = harness.runner.position
    quantity = position.leg("perp").quantity
    rate = position.config.maintenance_margin_rate
    spiked = _MarkState(state, mark_high=state.mark * D("2"))
    between = quantity * state.mark * rate * D("1.5")
    assert between < quantity * spiked.mark_high * rate, "below the high's threshold"
    assert between > quantity * state.mark * rate, "and above the close's"

    assert position.liquidation_touched(spiked, equity=between) is True
    # The control: the same equity on the same minute WITHOUT the spike is not a
    # touch, so the assertion above is about the high and nothing else.
    assert position.liquidation_touched(state, equity=between) is False


def test_a_minute_carrying_neither_mark_is_refused_on_a_held_position(tmp_path):
    """ "There is no third tier": no high and no close on a non-flat position."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    state = harness.runner.cursor.state_for(first + 3 * 60_000)
    blind = _MarkState(state, mark_high=None, mark=None)
    assert harness.runner.position.liquidation_touched(blind, equity=D("1")) is True


class _MarkState:
    """A minute with named fields overridden, for the two-sided liquidation control.

    The synthetic fixture writes one value into every mark column, so a minute
    taken from it cannot distinguish the high from the close. This lets a test
    say which one it means.
    """

    def __init__(self, state, **overrides):
        self._state = state
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._state, name)


def test_the_production_entry_point_actually_fills(tmp_path):
    """The runner installs the minute's book, so a campaign can open a position.

    Driven through `tools.demo_run._load` -- the production construction path --
    rather than through `tests/demo_harness.py`, because the harness installed
    the quote itself and that is exactly what hid this.

    `RecordedQuoteFillModel` names the runner as the caller that must install a
    book and advance the model's clock every cycle. Nothing did: through the CLI
    the model held no quote and a clock of zero, every order was refused
    `no_fresh_quote`, and the campaign stayed FLAT for ever -- so every path this
    PR made reachable was unreachable in production for a second reason, because
    no position was ever opened to fund, reconcile or liquidate.
    """
    import argparse

    from tests.demo_harness import CARRY_PARAMS, MOMENTUM_PARAMS, SHADOW_PARAMS
    from tools.demo_run import _load

    recorder = tmp_path / "recorder"
    state_dir = tmp_path / "state"
    contract = load_recorder_contract("btcusdt-prospective-gen3")
    feed = SyntheticFeed(recorder, contract)
    feed.write_days([DAY])
    feed.write_settlements([DAY])

    payload = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text("utf-8"))
    payload["profile"] = "SOAK"
    payload["runner"] = {**payload.get("runner", {}), "state_dir": str(state_dir)}
    payload["rules"] = {
        "R1_carry": dict(CARRY_PARAMS),
        "R2_frozen_logistic": dict(SHADOW_PARAMS),
        "R3_daily_momentum": dict(MOMENTUM_PARAMS),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    runner = _load(argparse.Namespace(config=config_path, root=recorder, profile="SOAK"))
    runner.start(allow_dirty=True)
    first = runner.cursor.next_minute_ms()
    for index in range(4):
        runner.tick(first + index * 60_000)

    assert runner.position.state is HedgeState.HEDGED
    assert runner.position.leg("spot").quantity > D("0")
    assert runner.position.imbalance() == D("0.000")


def test_each_leg_fills_inside_its_own_book_and_the_basis_has_the_right_sign(tmp_path):
    """Section 6.2: the spot leg buys spot. It must fill at the SPOT book.

    One `RecordedQuoteFillModel` was shared by both legs, and `install_quote`
    can install one book on one model -- it installed the perpetual's. So the
    spot leg bought at the perpetual's ask: on this day 30134.86, against a spot
    ask of 30098.83, a price no spot book ever showed. The ledger then recorded
    `entry_basis -13.06` where the true basis is `perp - spot = +30.00`: wrong in
    magnitude and in SIGN, on the one quantity a carry position exists to
    capture, and the missing 30 was booked as spot slippage.

    Two-sided: each leg's fill is checked against its own book, and the basis
    against the two books rather than against the implementation. A model shared
    between the legs fails the spot assertion whichever book it installs.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    state = harness.runner.cursor.state_for(first)
    spot_book, perp_book = state.book_spot, state.book_perp
    assert perp_book["ask"] > spot_book["ask"], "the day has no basis to get wrong"

    harness.run(2, start=first)
    assert harness.runner.position.state is HedgeState.HEDGED

    spot_fill = harness.runner.position.leg("spot").entry_price
    perp_fill = harness.runner.position.leg("perp").entry_price

    # The spot LONG crosses the SPOT ask and pays slippage above it; it can
    # never reach the perpetual's, which is 30 higher.
    assert spot_book["ask"] <= spot_fill < perp_book["ask"], (
        f"spot filled at {spot_fill}, outside its own book "
        f"({spot_book['bid']}/{spot_book['ask']})"
    )
    # The perp SHORT crosses the PERP bid and pays slippage below it.
    assert spot_book["bid"] < perp_fill <= perp_book["bid"], (
        f"perp filled at {perp_fill}, outside its own book "
        f"({perp_book['bid']}/{perp_book['ask']})"
    )
    # The recorded basis is the two fills' difference and is positive, as the
    # day's own books are: a SHORT perp above a LONG spot is the carry.
    ledger = harness.runner.position.ledger.state
    assert ledger.entry_basis == perp_fill - spot_fill
    assert ledger.entry_basis > D("0"), (
        f"entry basis {ledger.entry_basis} is not positive on a day whose perp "
        f"({perp_book['bid']}) sits above its spot ({spot_book['ask']})"
    )


def test_the_legs_do_not_share_a_fill_model(tmp_path):
    """The structural half of the finding above, asserted directly.

    `chimera/futures/fills.py` states the rule -- "a snapshot carries no symbol,
    so pairing the two is the caller's job: one venue and one fill model per
    leg". A model holds one book and one clock, so two legs sharing one is not a
    style question: whichever book is installed last prices both legs.
    """
    harness = build(tmp_path)
    models = harness.runner.position.fill_models
    assert set(models) == {"spot", "perp"}
    assert models["spot"] is not models["perp"]

    first = harness.first_minute_ms()
    state = harness.runner.cursor.state_for(first)
    harness.runner.position.install_quote(state)

    # Each model holds its own side of the market, not a copy of one side.
    assert models["spot"].quote.ask == state.book_spot["ask"]
    assert models["perp"].quote.ask == state.book_perp["ask"]
    assert models["spot"].quote.ask != models["perp"].quote.ask


def test_a_minute_with_no_book_still_moves_the_fill_model_clock(tmp_path):
    """ "The caller must advance now_ns every decision cycle, whether or not a
    new book arrived" -- otherwise a stale book never ages out and fills for ever.
    """
    from chimera.demo.faults import Fault, FaultSchedule, ScheduledFault

    schedule = FaultSchedule([ScheduledFault(2, Fault.MISSING_BOOK)])
    harness = build(
        tmp_path,
        shapes={
            f"spot:{DAY}": schedule.minute_shapes("spot"),
            f"um:{DAY}": schedule.minute_shapes("um"),
        },
    )
    first = harness.first_minute_ms()
    harness.run(2, start=first)
    models = harness.runner.position.fill_models
    before = {leg: model.now_ns for leg, model in models.items()}

    harness.runner.tick(first + 2 * 60_000)

    expected = (first + 2 * 60_000) * 1_000_000 + 60_000_000_000
    # Every leg's clock, not one shared one. Each model holds its own book, so a
    # leg whose clock stopped keeps filling against a snapshot of unbounded age
    # no matter what the other leg's clock did.
    for leg, model in models.items():
        assert model.now_ns > before[leg], f"the {leg} clock stopped on a bookless minute"
        assert model.now_ns == expected


def test_a_restart_reconciles_against_a_venue_that_still_knows_the_position(tmp_path):
    """Section 8.1's hourly check must survive a restart, and this is why it can.

    The dry-run venue holds the simulated account's positions in memory, so a new
    process starts with an empty one while the stores still hold what the account
    was left holding. `FuturesExecutor.reconcile` asks the venue -- so before
    PR-10R restored the venue's view from each leg's store, the FIRST periodic
    reconciliation after ANY restart compared a held position against an empty
    simulator, reported MISMATCH on both legs, disputed them and halted the
    campaign. A restart is a normal event this design expects (section 2.1), so
    that made a campaign unable to survive one.

    Asserted on the OUTCOME, not on the cadence: a test that only checked when
    the reconciliation happened passed throughout.
    """
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    harness.run(3, start=first)
    assert harness.runner.position.state is HedgeState.HEDGED
    held = harness.runner.position.leg("perp").quantity
    config = harness.runner.config
    harness.runner.shutdown("restart drill")

    resumed = build(tmp_path, config=config)
    assert resumed.runner.position.state is HedgeState.HEDGED
    resumed.runner.last_reconcile_minute_ms = None  # force the periodic arm now
    resumed.runner.tick(first + 3 * 60_000)

    assert resumed.runner.state is not RunnerState.HALT, resumed.runner.halt_reason
    record = _reconciliations(resumed)[-1]["reconciliation"]
    assert record["outcome"] == "AGREED"
    for leg in record["legs"]:
        assert leg["outcome"] == "AGREED", leg
        assert leg["reported_qty"] == str(held), "the venue still knows the position"
    assert resumed.risk.state.reconciliation_disputed == {}
