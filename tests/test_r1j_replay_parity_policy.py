"""R1-j: the replay parity policy -- ``seq``, ``OPERATOR`` and the risk hash.

The roadmap's item, verbatim (``docs/master_roadmap_r0_r18.md``):

    **R1-j Replay parity policy:** decide `seq` (per-kind sequence or exclusion
    of operational kinds) and `OPERATOR` (deterministic replay counterpart from a
    committed operator-action file); re-include deterministic risk fields in the
    hash; the 48 h synthetic fixture reaches PARITY with a restart and an
    operator action.

The 48 h witness is ``tests/test_r1j_parity_48h.py``. This module holds the
two-sided tests of each decision. Everything is synthetic
(``chimera.demo.fixtures``); nothing here is market data or evidence.

A "restart" here is a fresh runner built by ``tests.demo_harness.build`` over the
same directories, which goes through ``start()`` from the persisted state, as a
new process does. A live operator action is issued the way the CLI issues it:
from such a fresh runner, through the same ``flatten`` / ``resume`` method.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from chimera.demo import risk_continuity
from chimera.demo.operator_actions import (
    OPERATOR_ACTIONS_SCHEMA,
    OperatorActionError,
    canonical_text,
    load_operator_actions,
    parse_operator_actions,
)
from chimera.demo.risk_continuity import (
    RISK_HASH_POLICIES,
    RISK_HASH_POLICY,
    RISK_HASH_POLICY_LEGACY,
    RiskContinuityFault,
    RiskHashPolicyError,
    assess_risk_continuity,
    hash_policy_of,
    read_log_risk_history,
    risk_state_hash,
)
from chimera.demo.rules_carry import CarryRule
from chimera.demo.runner import RecoveryCause, RunnerState
from chimera.risk import RiskState, RiskStateLoad
from tests.demo_harness import build, campaign_config
from tools.replay_parity import (
    OPERATIONAL_KINDS,
    OPERATOR_KINDS,
    OperatorActionRefused,
    compare_logs,
    replay_with_actions,
)

NOTE = "R1-j test: reduce the hedge to flat after a restart"
N = 24


# --------------------------------------------------------------------------- #
# shared runs
# --------------------------------------------------------------------------- #
def _actions(*entries: dict[str, Any]):
    return parse_operator_actions(
        canonical_text({"schema": OPERATOR_ACTIONS_SCHEMA, "actions": list(entries)})
    )


def _flatten_action(minute: str, note: str = NOTE, **extra: Any) -> dict[str, Any]:
    return {
        "id": "flatten-1",
        "after_minute": minute,
        "command": "flatten",
        "note": note,
        **extra,
    }


def _live_with_restart_and_flatten(where: Path, minutes: int = N):
    """Live: minutes/2 decided, shutdown; a fresh process flattens; a fresh
    process decides the rest. Returns (records, first_ms, anchor)."""
    live = build(where)
    first = live.first_minute_ms()
    half = minutes // 2
    for i in range(half):
        live.tick(first + i * 60_000)
    live.runner.shutdown("planned restart")
    operator = build(where)
    operator.runner.flatten(NOTE)
    resumed = build(where)
    for i in range(half, minutes):
        resumed.tick(first + i * 60_000)
    resumed.runner.shutdown("done")
    records = resumed.records()
    anchor = next(r["minute"] for r in records if r["kind"] == "OPERATOR")
    return records, first, anchor


def _replay(where: Path, first: int, minutes: int, actions=None):
    replay = build(where)
    applied = replay_with_actions(
        replay.runner, first, first + (minutes - 1) * 60_000, actions
    )
    replay.runner.shutdown("replay")
    return replay.records(), applied


@pytest.fixture(scope="module")
def flatten_run(tmp_path_factory):
    where = tmp_path_factory.mktemp("r1j_flatten")
    records, first, anchor = _live_with_restart_and_flatten(where / "live")
    return {"root": where, "records": records, "first": first, "anchor": anchor}


def _renumber(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give a hand-edited log a fresh strictly rising seq, so only the edit
    under test differs (the chronology check is tested on its own)."""
    out = [dict(r) for r in records]
    for seq, record in enumerate(out, start=1):
        record["seq"] = seq
    return out


def _record(minute: str, kind: str, seq: int, **extra: Any) -> dict[str, Any]:
    return {"minute": minute, "kind": kind, "seq": seq, "runner_now_ns": 1, **extra}


M0 = "2026-09-19T00:00:00+00:00"
M1 = "2026-09-19T00:01:00+00:00"
M2 = "2026-09-19T00:02:00+00:00"


# --------------------------------------------------------------------------- #
# seq: exclusion of operational kinds, compared as parity_seq
# --------------------------------------------------------------------------- #
def test_a_clean_restart_reaches_parity_where_the_raw_seq_cannot(tmp_path):
    live = build(tmp_path / "live")
    first = live.first_minute_ms()
    for i in range(N // 2):
        live.tick(first + i * 60_000)
    live.runner.shutdown("planned restart")
    again = build(tmp_path / "live")
    for i in range(N // 2, N):
        again.tick(first + i * 60_000)
    again.runner.shutdown("done")
    records = again.records()
    replayed, _ = _replay(tmp_path / "replay", first, N)

    report = compare_logs(records, replayed)

    assert report.ok, report.divergences[:1] and report.divergences[0].render()
    assert report.compared >= N and report.explained_exclusions == []
    # The control: the raw seq of every record after the restart moved, which
    # is exactly what the pre-R1-j comparison demanded be equal.
    live_last = next(r for r in reversed(records) if r["kind"] == "DECISION")
    replay_last = next(r for r in reversed(replayed) if r["kind"] == "DECISION")
    assert live_last["seq"] != replay_last["seq"]


def test_many_operational_records_do_not_shift_parity_seq(tmp_path):
    live = build(tmp_path / "live")
    live.run(8)
    live.runner.shutdown("done")
    records = live.records()
    noisy: list[dict[str, Any]] = []
    for record in records:
        noisy.append(record)
        if record["kind"] == "DECISION":
            for kind in sorted(OPERATIONAL_KINDS - {"RECOVERY"}):
                noisy.append(_record(record["minute"], kind, 0, operator={"note": "noise"}))
            noisy.append(
                _record(
                    record["minute"],
                    "RECOVERY",
                    0,
                    recovery={"cause": "LOG_AHEAD_OF_STATE", "minute_finished": True},
                )
            )
    noisy = _renumber(noisy)

    report = compare_logs(noisy, records)

    assert report.ok and report.explained_exclusions == []
    assert report.compared == len([r for r in records if r["kind"] not in OPERATIONAL_KINDS])


def test_two_comparable_records_swapped_fail_on_parity_seq(tmp_path):
    live = build(tmp_path / "live")
    live.run(6)
    live.runner.shutdown("done")
    records = live.records()
    swapped = [dict(r) for r in records]
    decisions = [i for i, r in enumerate(swapped) if r["kind"] == "DECISION"]
    a, b = decisions[1], decisions[3]
    swapped[a], swapped[b] = swapped[b], swapped[a]

    report = compare_logs(records, _renumber(swapped))

    assert not report.ok
    assert any(d.field_name == "parity_seq" for d in report.divergences)
    assert not report.live_only and not report.replay_only, "the keys still align"


def test_a_dropped_comparable_record_fails(tmp_path):
    live = build(tmp_path / "live")
    live.run(6)
    live.runner.shutdown("done")
    records = live.records()
    index = [i for i, r in enumerate(records) if r["kind"] == "DECISION"][2]
    dropped = _renumber(records[:index] + records[index + 1 :])

    report = compare_logs(records, dropped)

    assert not report.ok and report.live_only
    assert any(d.field_name == "parity_seq" for d in report.divergences)


def test_an_added_comparable_record_fails(tmp_path):
    live = build(tmp_path / "live")
    live.run(6)
    live.runner.shutdown("done")
    records = live.records()
    index = [i for i, r in enumerate(records) if r["kind"] == "DECISION"][2]
    ghost = dict(records[index], minute="2026-09-19T12:00:00+00:00")
    added = _renumber(records[: index + 1] + [ghost] + records[index + 1 :])

    report = compare_logs(records, added)

    assert not report.ok and report.replay_only
    assert any(d.field_name == "parity_seq" for d in report.divergences)


def test_repeated_same_minute_same_kind_records_keep_their_order():
    live = [
        _record(M0, "FUNDING", 1, ledger_effect={"funding": "1"}),
        _record(M0, "FUNDING", 2, ledger_effect={"funding": "2"}),
        _record(M0, "DECISION", 3, ledger_effect={"equity": "10"}),
    ]
    reordered = [dict(live[1], seq=1), dict(live[0], seq=2), dict(live[2])]

    report = compare_logs(live, reordered)

    assert not report.ok
    assert {d.field_name for d in report.divergences} == {"ledger_effect"}
    assert compare_logs(live, [dict(r) for r in live]).ok, "the control"


def test_a_record_moved_across_another_kind_in_its_minute_fails():
    """A settlement booked after the minute's decision instead of before it
    aligns on the same keys, so only the order can catch it."""
    live = [
        _record(M0, "FUNDING", 1, ledger_effect={"funding": "1"}),
        _record(M0, "DECISION", 2, ledger_effect={"equity": "10"}),
        _record(M1, "DECISION", 3, ledger_effect={"equity": "11"}),
    ]
    moved = [dict(live[1], seq=1), dict(live[0], seq=2), dict(live[2])]

    report = compare_logs(live, moved)

    assert not report.ok
    assert {d.field_name for d in report.divergences} == {"parity_seq"}


def test_the_earlier_of_two_same_minute_records_is_compared_too():
    """Each repeated record is compared with its own counterpart, not only the
    last one: here only the first settlement differs."""
    live = [
        _record(M0, "FUNDING", 1, ledger_effect={"funding": "1"}),
        _record(M0, "FUNDING", 2, ledger_effect={"funding": "2"}),
    ]
    replay = [dict(live[0], ledger_effect={"funding": "9"}), dict(live[1])]

    report = compare_logs(live, replay)

    assert not report.ok
    assert [d.field_name for d in report.divergences] == ["ledger_effect"]


def test_a_raw_seq_that_does_not_rise_is_reported_on_its_side():
    live = [_record(M0, "DECISION", 1), _record(M1, "DECISION", 2)]
    broken = [_record(M0, "DECISION", 2), _record(M1, "DECISION", 2)]

    report = compare_logs(live, broken)

    assert not report.ok
    [problem] = [d for d in report.divergences if d.field_name == "seq"]
    assert problem.live is None and problem.replay == {"seq": 2, "previous_seq": 2}
    assert compare_logs(broken, live).divergences[0].replay is None


def test_request_seq_still_names_the_persisted_request(flatten_run):
    """R1-i's link is untouched: `request_seq` is the request's persisted seq,
    the restart's triage finds nothing unfinished, and parity reads the link as
    the request's parity_seq."""
    records = flatten_run["records"]
    operator = [r for r in records if r["kind"] == "OPERATOR"]
    by_seq = {r["seq"]: r for r in records}
    requested = by_seq[operator[1]["operator"]["request_seq"]]
    assert requested is operator[0] and requested["operator"]["phase"] == "requested"
    assert not [r for r in records if r["kind"] == "RECOVERY"], "no OPERATOR_INCOMPLETE"
    assert operator[0]["seq"] > len([r for r in records if r["kind"] == "DECISION"]) // 2


def test_a_request_seq_that_names_the_wrong_record_diverges(flatten_run, tmp_path):
    records = flatten_run["records"]
    actions = _actions(_flatten_action(flatten_run["anchor"]))
    replayed, _ = _replay(tmp_path / "replay", flatten_run["first"], N, actions)
    assert compare_logs(records, replayed, operator_file_hash=actions.file_hash).ok
    wrong = [dict(r) for r in records]
    completed = next(
        r for r in wrong if r["kind"] == "OPERATOR" and r["operator"]["phase"] == "completed"
    )
    completed["operator"] = {**completed["operator"], "request_seq": completed["seq"] - 3}

    report = compare_logs(wrong, replayed, operator_file_hash=actions.file_hash)

    assert not report.ok
    [bad] = [d for d in report.divergences if d.field_name == "operator"]
    assert "unresolved" in json.dumps(bad.live)


# --------------------------------------------------------------------------- #
# the operator-action file
# --------------------------------------------------------------------------- #
def test_a_canonical_file_is_read_and_identified_by_its_bytes(tmp_path):
    text = canonical_text(
        {"schema": OPERATOR_ACTIONS_SCHEMA, "actions": [_flatten_action(M1)]}
    )
    path = tmp_path / "actions.json"
    path.write_bytes(text.encode("ascii"))

    actions = load_operator_actions(path)

    [action] = actions.actions
    assert (action.id, action.command, action.note, action.after_minute) == (
        "flatten-1",
        "flatten",
        NOTE,
        M1,
    )
    assert actions.file_hash == "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _document(*entries: dict[str, Any], schema: str = OPERATOR_ACTIONS_SCHEMA) -> str:
    return canonical_text({"schema": schema, "actions": list(entries)})


REFUSALS: dict[str, tuple[str, str]] = {
    "unknown schema": (
        _document(_flatten_action(M1), schema="chimera.operator-actions/0"),
        "schema",
    ),
    "unknown action": (
        _document({**_flatten_action(M1), "command": "liquidate"}),
        "liquidate",
    ),
    "blank note": (_document(_flatten_action(M1, note="   ")), "no note"),
    "missing note": (
        _document({k: v for k, v in _flatten_action(M1).items() if k != "note"}),
        "missing",
    ),
    "padded note": (_document(_flatten_action(M1, note=f" {NOTE} ")), "whitespace"),
    "ambiguous selector": (
        _document(
            {"id": "r", "after_minute": M1, "command": "resolve", "note": NOTE,
             "symbol": "BTCUSDT", "equity": True}
        ),
        "ambiguous",
    ),  # fmt: skip
    "no selector": (
        _document({"id": "r", "after_minute": M1, "command": "resolve", "note": NOTE}),
        "ambiguous",
    ),
    "resolve is not replayable": (
        _document(
            {"id": "r", "after_minute": M1, "command": "resolve", "note": NOTE, "equity": True}
        ),
        "no deterministic replay counterpart",
    ),
    "impossible selector": (_document(_flatten_action(M1, symbol="BTCUSDT")), "impossible"),
    "unknown key": (_document(_flatten_action(M1, at_wall_clock="now")), "unknown key"),
    "duplicate id": (
        _document(_flatten_action(M1), {**_flatten_action(M2), "command": "resume"}),
        "twice",
    ),
    "duplicate at an anchor": (
        _document(_flatten_action(M1), {**_flatten_action(M1), "id": "flatten-2"}),
        "twice after",
    ),
    "out of order": (
        _document(_flatten_action(M2), {**_flatten_action(M1), "id": "flatten-2"}),
        "file order",
    ),
    "Z suffix": (_document(_flatten_action("2026-09-19T00:01:00Z")), "canonical spelling"),
    "naive minute": (_document(_flatten_action("2026-09-19T00:01:00")), "not UTC"),
    "off the minute": (_document(_flatten_action("2026-09-19T00:01:30+00:00")), "boundary"),
    "not a minute": (_document(_flatten_action("yesterday")), "ISO-8601"),
    "bad id": (_document({**_flatten_action(M1), "id": "a b"}), "identifier"),
    "not canonical": (
        json.dumps({"schema": OPERATOR_ACTIONS_SCHEMA, "actions": [_flatten_action(M1)]}),
        "canonical form",
    ),
    "not json": ("{", "not JSON"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_malformed_file_is_refused_by_name(case):
    text, expected = REFUSALS[case]
    with pytest.raises(OperatorActionError, match=expected):
        parse_operator_actions(text)


def test_the_tool_refuses_a_malformed_file_before_replaying(tmp_path, capsys, flatten_run):
    from tools import replay_parity

    bad = tmp_path / "bad.json"
    bad.write_bytes(REFUSALS["blank note"][0].encode("ascii"))
    argv = [
        "--config", str(tmp_path / "unused.json"), "--root", str(tmp_path),
        "--live-log", str(tmp_path), "--days", "2026-09-19",
        "--operator-actions", str(bad),
    ]  # fmt: skip
    assert (
        replay_parity.main(argv, operational_clock=lambda: 0.0) == replay_parity.EXIT_REFUSED
    )
    assert "no note" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# live OPERATOR vs its replay counterpart
# --------------------------------------------------------------------------- #
def test_the_exact_committed_counterpart_reaches_parity(flatten_run, tmp_path):
    records = flatten_run["records"]
    actions = _actions(_flatten_action(flatten_run["anchor"]))
    replayed, applied = _replay(tmp_path / "replay", flatten_run["first"], N, actions)

    report = compare_logs(records, replayed, operator_file_hash=actions.file_hash)

    assert applied == ["flatten-1"]
    assert report.ok, report.divergences[:1] and report.divergences[0].render()
    assert report.explained_exclusions == []
    compared_operator = [r for r in replayed if r["kind"] in OPERATOR_KINDS]
    assert len(compared_operator) == 2
    # The deterministic effect is the same flatten, and what followed matches.
    after = [r for r in replayed if r["kind"] == "OPERATOR"][1]["position_after"]
    assert Decimal(after["perp_qty"]) == 0 and Decimal(after["spot_qty"]) == 0


def test_a_live_operator_action_missing_from_the_file_fails(flatten_run, tmp_path):
    replayed, _ = _replay(tmp_path / "replay", flatten_run["first"], N, None)

    report = compare_logs(flatten_run["records"], replayed)

    assert not report.ok
    assert "OPERATOR at" in " ".join(report.live_only)
    assert any("flatten requested" in entry for entry in report.live_only)


def test_an_extra_action_in_the_file_fails(tmp_path):
    live = build(tmp_path / "live")
    first = live.first_minute_ms()
    live.run(N)
    live.runner.shutdown("done")
    anchor = next(r["minute"] for r in live.records() if r["kind"] == "DECISION")
    actions = _actions(_flatten_action(anchor))
    replayed, _ = _replay(tmp_path / "replay", first, N, actions)

    report = compare_logs(live.records(), replayed, operator_file_hash=actions.file_hash)

    assert not report.ok
    assert any("flatten" in entry for entry in report.replay_only)


def test_a_wrong_note_fails(flatten_run, tmp_path):
    actions = _actions(_flatten_action(flatten_run["anchor"], note="some other reason"))
    replayed, _ = _replay(tmp_path / "replay", flatten_run["first"], N, actions)

    report = compare_logs(
        flatten_run["records"], replayed, operator_file_hash=actions.file_hash
    )

    assert not report.ok
    assert {d.field_name for d in report.divergences} == {"operator"}
    assert "some other reason" in report.divergences[0].render()


def test_a_wrong_action_is_refused_rather_than_performed(flatten_run, tmp_path):
    actions = _actions({**_flatten_action(flatten_run["anchor"]), "command": "resume"})
    with pytest.raises(OperatorActionRefused, match="never reached|not halted"):
        _replay(tmp_path / "replay", flatten_run["first"], N, actions)


def test_a_wrong_anchor_fails(flatten_run, tmp_path):
    late = flatten_run["first"] + (N // 2 + 2) * 60_000
    from chimera.demo.decision_log import iso_minute

    actions = _actions(_flatten_action(iso_minute(late * 1_000_000)))
    replayed, _ = _replay(tmp_path / "replay", flatten_run["first"], N, actions)

    report = compare_logs(
        flatten_run["records"], replayed, operator_file_hash=actions.file_hash
    )

    assert not report.ok and report.live_only and report.replay_only


def test_an_action_outside_the_range_or_never_reached_is_refused(flatten_run, tmp_path):
    early = _actions(_flatten_action("2026-09-18T23:00:00+00:00"))
    with pytest.raises(OperatorActionRefused, match="outside the replayed range"):
        _replay(tmp_path / "a", flatten_run["first"], N, early)


def test_the_replay_attribution_is_required_and_checked(flatten_run, tmp_path):
    records = flatten_run["records"]
    actions = _actions(_flatten_action(flatten_run["anchor"]))
    replayed, _ = _replay(tmp_path / "replay", flatten_run["first"], N, actions)

    other = compare_logs(records, replayed, operator_file_hash="sha256:" + "0" * 64)
    assert not other.ok and "operator.replay_action" in {
        d.field_name for d in other.divergences
    }

    stripped = [dict(r) for r in replayed]
    for record in stripped:
        if record["kind"] in OPERATOR_KINDS:
            record["operator"] = {
                k: v for k, v in record["operator"].items() if k != "replay_action"
            }
    unattributed = compare_logs(records, stripped, operator_file_hash=actions.file_hash)
    assert not unattributed.ok

    # A "live" log that is really a replay's.
    assert not compare_logs(replayed, replayed, operator_file_hash=actions.file_hash).ok


def test_a_runner_that_refuses_the_action_is_a_refusal_not_a_skip():
    """Whatever `apply_operator_action` refuses propagates as a refusal."""
    from chimera.demo.runner import RunnerError

    class Cursor:
        last_minute_processed = 60_000

        def next_minute_ms(self):
            return 120_000

    class Refusing:
        cursor = Cursor()
        state = RunnerState.READY
        calls: list[str] = []

        def replay(self, start, end):
            self.calls.append(f"replay {start}-{end}")

        def apply_operator_action(self, command, note, *, replay_action):
            raise RunnerError("the runner refuses")

    from chimera.demo.decision_log import iso_minute

    actions = _actions(_flatten_action(iso_minute(60_000 * 1_000_000)))
    runner = Refusing()
    with pytest.raises(OperatorActionRefused, match="the runner refuses"):
        replay_with_actions(runner, 0, 600_000, actions)
    assert runner.calls == [], "nothing is replayed past a refused action"


def test_operator_is_not_an_operational_kind():
    assert not OPERATOR_KINDS & OPERATIONAL_KINDS
    assert {"OPERATOR", "RESUME"} == set(OPERATOR_KINDS)


# --------------------------------------------------------------------------- #
# resume: replayed only where the replay halted too
# --------------------------------------------------------------------------- #
@pytest.fixture
def one_shot_rule_fault(monkeypatch):
    """A rule that raises the first time each campaign decides one minute.

    A deterministic, runner-level halt a replay meets in the same minute (the
    runner halts on a rule exception), which an operator's resume then clears.
    Keyed by campaign so the live campaign and its replay each meet it once.
    """
    state: dict[str, Any] = {"minute_ns": None, "campaign": None, "met": set()}
    real = CarryRule.evaluate

    def evaluate(self, market, portfolio):
        key = (state["campaign"], market.minute_ns)
        if market.minute_ns == state["minute_ns"] and key not in state["met"]:
            state["met"].add(key)
            raise RuntimeError("synthetic one-shot rule fault")
        return real(self, market, portfolio)

    monkeypatch.setattr(CarryRule, "evaluate", evaluate)
    return state


def test_a_resume_after_a_deterministic_halt_has_its_counterpart(
    tmp_path, one_shot_rule_fault
):
    one_shot_rule_fault["campaign"] = "live"
    live = build(tmp_path / "live")
    first = live.first_minute_ms()
    one_shot_rule_fault["minute_ns"] = (first + 10 * 60_000) * 1_000_000
    for i in range(N):
        if live.runner.state is RunnerState.HALT:
            break
        live.tick(first + i * 60_000)
    assert live.runner.state is RunnerState.HALT
    live.runner.shutdown("halted")
    operator = build(tmp_path / "live")
    assert operator.runner.state is RunnerState.HALT
    operator.runner.resume("rule fault understood; resuming")
    resumed = build(tmp_path / "live")
    while resumed.runner.cursor.last_minute_processed < first + (N - 1) * 60_000:
        resumed.tick(resumed.runner.cursor.next_minute_ms())
    resumed.runner.shutdown("done")
    records = resumed.records()
    anchor = next(r["minute"] for r in records if r["kind"] == "OPERATOR")
    actions = _actions(
        {"id": "resume-1", "after_minute": anchor, "command": "resume",
         "note": "rule fault understood; resuming"}
    )  # fmt: skip

    one_shot_rule_fault["campaign"] = "replay"
    replayed, applied = _replay(tmp_path / "replay", first, N, actions)
    report = compare_logs(records, replayed, operator_file_hash=actions.file_hash)

    assert applied == ["resume-1"]
    assert report.ok, report.divergences[:1] and report.divergences[0].render()
    kinds = [r["kind"] for r in replayed if r["kind"] in OPERATOR_KINDS]
    assert kinds == ["OPERATOR", "RESUME"], "RESUME is compared, not ignored"
    cleared = next(r for r in replayed if r["kind"] == "OPERATOR")["operator"]["cleared"]
    assert cleared.startswith("rule_exception")


def test_a_resume_whose_halt_the_replay_never_met_is_refused(tmp_path):
    """Aegis halted on the order-rate limit mid-run, which only a restart's
    `start()` turns into a runner HALT. The replay never restarts, so it is not
    halted where the operator resumed, and the action is refused, not faked."""

    def config(where: Path):
        return campaign_config(where / "state", limits={"max_orders_per_minute": 1})

    live = build(tmp_path / "live", config=config(tmp_path / "live"))
    first = live.first_minute_ms()
    live.run(12)
    assert live.runner.risk.state.halted and live.runner.state is not RunnerState.HALT
    live.runner.shutdown("stopped")
    operator = build(tmp_path / "live", config=config(tmp_path / "live"))
    operator.runner.resume("rate limit understood")
    anchor = next(r["minute"] for r in operator.records() if r["kind"] == "OPERATOR")
    actions = _actions(
        {"id": "resume-1", "after_minute": anchor, "command": "resume",
         "note": "rate limit understood"}
    )  # fmt: skip
    replay = build(tmp_path / "replay", config=config(tmp_path / "replay"))
    with pytest.raises(OperatorActionRefused):
        replay_with_actions(replay.runner, first, first + 20 * 60_000, actions)


# --------------------------------------------------------------------------- #
# the risk hash: policy, re-included fields, compatibility
# --------------------------------------------------------------------------- #
def _snapshot(**changes: Any) -> dict[str, Any]:
    state = RiskState(
        equity=1000.0, peak_equity=1000.0, day_start_equity=1000.0, day="2026-09-19"
    )
    snap = state.snapshot()
    snap.update(changes)
    return snap


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("order_times", [1_789_776_060.0]),
        ("cooldown_until", 1_789_779_600.0),
        ("day", "2026-09-20"),
    ],
)
def test_each_re_included_field_moves_the_new_hash_and_not_the_legacy_one(field_name, value):
    base = _snapshot()
    moved = _snapshot(**{field_name: value})
    assert risk_state_hash(moved) != risk_state_hash(base)
    assert risk_state_hash(moved, RISK_HASH_POLICY_LEGACY) == risk_state_hash(
        base, RISK_HASH_POLICY_LEGACY
    )


def test_the_new_policy_hashes_every_snapshot_field():
    base = _snapshot()
    assert RISK_HASH_POLICIES[RISK_HASH_POLICY] == frozenset()
    for name in base:
        without = {k: v for k, v in base.items() if k != name}
        assert risk_state_hash(without) != risk_state_hash(base), f"{name} is not hashed"


def test_the_policy_is_read_and_an_unknown_one_refused():
    assert hash_policy_of({"state_hash": "x"}) == RISK_HASH_POLICY_LEGACY
    assert (
        hash_policy_of({"state_hash": "x", "hash_policy": RISK_HASH_POLICY})
        == RISK_HASH_POLICY
    )
    for bad in ("chimera.risk-hash/3", "", None):
        with pytest.raises(RiskHashPolicyError):
            hash_policy_of({"state_hash": "x", "hash_policy": bad})
    with pytest.raises(RiskHashPolicyError):
        risk_state_hash(_snapshot(), "chimera.risk-hash/3")


def test_every_restated_hash_names_the_new_policy_and_matches_the_file(tmp_path):
    harness = build(tmp_path)
    harness.run(6)
    harness.runner.shutdown("done")
    blocks = [r["risk"] for r in harness.records() if isinstance(r.get("risk"), dict)]
    assert blocks and {b["hash_policy"] for b in blocks} == {RISK_HASH_POLICY}
    history = read_log_risk_history(harness.state_dir)
    assert history.hash_policy == RISK_HASH_POLICY
    on_disk = RiskState.from_dict(
        json.loads((harness.state_dir / "risk.json").read_text(encoding="utf-8"))
    )
    assert risk_state_hash(on_disk.snapshot()) == history.state_hash


def test_the_re_included_fields_ignore_a_hostile_host_clock(tmp_path, monkeypatch):
    """Aegis's order window and day are the recorded decision clock's, whatever
    the host says, so the replay under an honest clock matches."""
    import time as time_module

    live_dir, replay_dir = tmp_path / "live", tmp_path / "replay"
    monkeypatch.setattr(time_module, "time", lambda: 4_102_444_800.0)
    live = build(live_dir)
    first = live.first_minute_ms()
    live.run(6)
    live.runner.shutdown("done")
    risk = json.loads((live_dir / "state" / "risk.json").read_text(encoding="utf-8"))
    assert risk["day"] == "2026-09-19"
    assert all(t < 1_900_000_000 for t in risk["order_times"])
    monkeypatch.undo()
    replayed, _ = _replay(replay_dir, first, 6)
    assert compare_logs(live.records(), replayed).ok


def _legacy_writer(monkeypatch):
    """The risk block exactly as an R1-i build wrote it: no policy, legacy hash."""
    from chimera.demo import runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "_risk_statement",
        lambda risk: {"state_hash": risk_state_hash(risk.snapshot(), RISK_HASH_POLICY_LEGACY)},
    )


def test_an_r1i_state_upgrades_without_a_false_dispute(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        _legacy_writer(patch)
        old = build(tmp_path)
        old.run(8)
        old.runner.shutdown("the R1-i build stops")
    legacy = [r for r in old.records() if isinstance(r.get("risk"), dict)]
    assert legacy and all("hash_policy" not in r["risk"] for r in legacy)
    on_disk = RiskState.from_dict(
        json.loads((tmp_path / "state" / "risk.json").read_text(encoding="utf-8"))
    ).snapshot()
    assert legacy[-1]["risk"]["state_hash"] == risk_state_hash(
        on_disk, RISK_HASH_POLICY_LEGACY
    )
    assert legacy[-1]["risk"]["state_hash"] != risk_state_hash(
        on_disk
    ), "read under the new policy, this very file would be a false dispute"

    upgraded = build(tmp_path)

    assert upgraded.runner.state is RunnerState.READY
    assert not [r for r in upgraded.records() if r["kind"] == "RECOVERY"]
    upgraded.run(2, start=upgraded.runner.cursor.next_minute_ms())
    newest = [r for r in upgraded.records() if isinstance(r.get("risk"), dict)][-1]
    assert newest["risk"]["hash_policy"] == RISK_HASH_POLICY


def test_a_real_mismatch_under_the_legacy_policy_still_fails_closed(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        _legacy_writer(patch)
        old = build(tmp_path)
        old.run(8)
        old.runner.shutdown("the R1-i build stops")
    path = tmp_path / "state" / "risk.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["peak_equity"] = document["peak_equity"] * 2
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    restarted = build(tmp_path)

    assert restarted.runner.state is RunnerState.HALT
    causes = [r["recovery"]["cause"] for r in restarted.records() if r["kind"] == "RECOVERY"]
    assert causes == [RecoveryCause.RISK_STATE_MISMATCH.value]


def test_a_legacy_hash_is_never_read_under_the_new_policy(tmp_path, monkeypatch):
    """The field the legacy hash left out may differ; the new policy's may not."""
    with monkeypatch.context() as patch:
        _legacy_writer(patch)
        old = build(tmp_path / "old")
        old.run(8)
        old.runner.shutdown("stop")
    new = build(tmp_path / "new")
    new.run(8)
    new.runner.shutdown("stop")

    def tamper(state_dir: Path) -> dict[str, Any]:
        document = json.loads((state_dir / "risk.json").read_text(encoding="utf-8"))
        document["order_times"] = []
        document["day"] = "2026-09-01"
        return RiskState.from_dict(document).snapshot()

    for where, disputed in (
        (tmp_path / "old" / "state", False),
        (tmp_path / "new" / "state", True),
    ):
        verdict = assess_risk_continuity(
            load=RiskStateLoad.LOADED, snapshot=tamper(where), state_dir=where
        )
        assert verdict.disputed is disputed, where


def test_an_unknown_policy_in_the_log_is_a_dispute(tmp_path):
    harness = build(tmp_path)
    harness.run(4)
    harness.runner.shutdown("stop")
    history = read_log_risk_history(harness.state_dir)
    from dataclasses import replace

    unknown = replace(history, hash_policy="chimera.risk-hash/9")
    snapshot = RiskState.from_dict(
        json.loads((harness.state_dir / "risk.json").read_text(encoding="utf-8"))
    )

    verdict = assess_risk_continuity(
        load=RiskStateLoad.LOADED,
        snapshot=snapshot.snapshot(),
        state_dir=harness.state_dir,
        history=unknown,
    )

    assert verdict.fault is RiskContinuityFault.RISK_STATE_MISMATCH
    assert "chimera.risk-hash/9" in verdict.detail
    ok = assess_risk_continuity(
        load=RiskStateLoad.LOADED,
        snapshot=snapshot.snapshot(),
        state_dir=harness.state_dir,
        history=history,
    )
    assert not ok.disputed, "the control"


# --------------------------------------------------------------------------- #
# the order window in R1-c's crash proofs
# --------------------------------------------------------------------------- #
def _orders_minute(tmp_path: Path):
    """A campaign whose newest statement hashed a non-empty order window."""
    harness = build(tmp_path)
    first = harness.first_minute_ms()
    for i in range(12):
        harness.tick(first + i * 60_000)
        risk = json.loads((harness.state_dir / "risk.json").read_text(encoding="utf-8"))
        if risk["order_times"]:
            break
    else:  # pragma: no cover - the fixture always opens in its first minutes
        raise AssertionError("the fixture never traded")
    harness.runner.shutdown("stop")
    return harness, json.loads((harness.state_dir / "risk.json").read_text(encoding="utf-8"))


def _write_risk(state_dir: Path, document: dict[str, Any]) -> None:
    (state_dir / "risk.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verdict(state_dir: Path, document: dict[str, Any], harness):
    """R1-c's verdict with the campaign's limits, as the runner asks for it."""
    from chimera.demo.risk_wiring import risk_limits

    return assess_risk_continuity(
        load=RiskStateLoad.LOADED,
        snapshot=RiskState.from_dict(document).snapshot(),
        state_dir=state_dir,
        limits=risk_limits(harness.runner.config.limits),
    )


def test_an_order_persisted_before_its_record_is_a_proved_window(tmp_path):
    harness, document = _orders_minute(tmp_path)
    # One more approval at the statement's own clock, persisted before any
    # record restated the hash.
    stamp = document["order_times"][-1]
    appended = {**document, "order_times": document["order_times"] + [stamp]}
    from datetime import datetime, timezone

    appended["updated_at"] = datetime.fromtimestamp(stamp, timezone.utc).isoformat()
    _write_risk(harness.state_dir, appended)

    verdict = _verdict(harness.state_dir, appended, harness)

    assert verdict.fault is RiskContinuityFault.RISK_STATE_MISMATCH
    assert verdict.crash_transition == "order"


def test_a_prune_is_proved_only_when_the_file_says_the_entries_aged_out(tmp_path):
    from datetime import datetime, timezone

    harness, document = _orders_minute(tmp_path)
    newest = document["order_times"][-1]
    for written_at, proved in ((newest + 60.0, "order"), (newest + 30.0, "")):
        pruned = {
            **document,
            "order_times": [],
            "updated_at": datetime.fromtimestamp(written_at, timezone.utc).isoformat(),
        }
        _write_risk(harness.state_dir, pruned)
        verdict = _verdict(harness.state_dir, pruned, harness)
        assert verdict.disputed
        assert verdict.crash_transition == proved, written_at


def test_an_order_window_outside_the_logs_instants_is_not_proved(tmp_path):
    harness, document = _orders_minute(tmp_path)
    forged = {**document, "order_times": [document["order_times"][-1] - 3600.0]}
    _write_risk(harness.state_dir, forged)
    verdict = _verdict(harness.state_dir, forged, harness)
    assert verdict.disputed and verdict.crash_transition == ""


def test_the_proof_uses_the_same_window_aegis_does():
    from chimera.risk import _ORDER_WINDOW_S

    assert risk_continuity._ORDER_WINDOW_NS == int(_ORDER_WINDOW_S * 1_000_000_000)
