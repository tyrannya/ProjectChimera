"""`DemoRunner`: section 8.1's state machine, over synthetic files only.

No market data is read anywhere in this file, and no economic conclusion is
available from any run in it. What is asserted is that the machinery sequences
correctly, refuses correctly, and writes down what it did.
"""

from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from chimera.carry.hedge import HedgeState
from chimera.demo.decision_log import RecordKind
from chimera.demo.fixtures import MinuteShape
from chimera.demo.rules import RuleError, RuleRegistry
from chimera.demo.runner import RunnerError, RunnerState
from tests.demo_harness import CARRY_PARAMS, DAY, build, campaign_config


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


def test_a_log_ahead_of_the_runner_state_halts(tmp_path):
    """Section 8.1's SELF_CHECK: the tail hash must equal the persisted head."""
    harness = build(tmp_path)
    harness.run(1)
    harness.runner.last_record_hash = "sha256:" + "b" * 64
    assert harness.runner.self_check() is not None
    assert "log_behind_state" in harness.runner.self_check()


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
def test_catch_up_processes_at_most_max_catchup_minutes(tmp_path):
    harness = build(tmp_path, config=None)
    limit = int(harness.runner.config.runner_setting("max_catchup_minutes"))
    outcomes = harness.runner.catch_up()
    assert len(outcomes) <= limit


def test_catch_up_honours_a_configured_limit(tmp_path):
    state_dir = tmp_path / "state"
    config = campaign_config(state_dir, runner={"max_catchup_minutes": 2})
    harness = build(tmp_path, config=config)
    assert len(harness.runner.catch_up()) == 2


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
    harness = build(tmp_path, shapes={f"spot:{DAY}": schedule.minute_shapes("spot")})
    first = harness.first_minute_ms()

    kinds: dict[str, int] = {}
    for index in range(300):
        assert harness.runner.state is not RunnerState.HALT, f"halted at minute {index}"
        outcome = harness.tick(first + index * 60_000)
        kinds[outcome.kind.value] = kinds.get(outcome.kind.value, 0) + 1
        if harness.runner.position.state is HedgeState.HEDGED:
            assert harness.runner.position.imbalance() == D(
                "0.000"
            ), f"HEDGED with an imbalance at minute {index}"

    harness.runner.shutdown("acceptance")
    assert kinds["INCOMPLETE_STATE"] == 3, "every injected fault was met"
    assert kinds["DECISION"] == 297

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
