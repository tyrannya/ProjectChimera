"""Section 11.4's reports: the arithmetic, the refusals, and the purity.

Three failures are guarded against here, and they are not variations of one.

*A number that is wrong.* Every arithmetic expectation below is computed IN THIS
FILE from the raw record strings, or written out as a literal the test author
worked out by hand from amendment A10's table. Nothing is compared against the
function that produced it, and no expectation is read out of
``chimera/demo/reports.py``: a report that reformatted a fee, netted a funding
split, or reported a cumulative total as a daily delta has to fail a comparison
against a number computed some other way.

*A zero that means two things.* The runner at this revision writes seven of the
twelve record kinds. A campaign run by it produces no ``FUNDING``,
``RECONCILIATION``, ``LIQUIDATION_TOUCH``, ``RECOVERY`` or ``SKIPPED_STALE``
record at all, so those counts are zero for a reason that has nothing to do with
the campaign — and presented as a bare zero they read exactly like "none
occurred". Two families of test hold the line: ``input_coverage`` must say which
of the two a zero is, checked against the runner's own ``_append`` call sites
walked out of its AST; and the arithmetic must be *correct* on synthetic logs
that DO hold those kinds, so the report is already right the day the runner
learns to write them.

*A report that changed what it reported on.* Proven mechanically rather than
promised: every file under the state directory is hashed before and after a
report run and must not have moved, no path may appear or vanish, and an AST walk
asserts that the module calls nothing that can write. The last one is the
load-bearing guard for a mutation the digests cannot see — a store opened
``"r+"`` and left alone changes no byte.

Everything synthetic here is written through the real
:class:`~chimera.demo.decision_log.DecisionLog`, so the chain, the canonical
bytes and the hashes are the writer's own; only the payload numbers are the
test's. Where a campaign is needed rather than a hand-built log,
``tests/demo_harness.py`` runs one over ``chimera.demo.fixtures``' invented
prices. No market data is read anywhere in this file and no economic claim is
available from any number in it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from chimera.demo import reports
from chimera.demo.decision_log import DecisionLog
from chimera.demo.reports import (
    ProtocolBinding,
    ReportRefused,
    daily_report,
    monthly_report,
    render_daily_markdown,
    render_monthly_status,
)
from tests.demo_harness import CARRY_PARAMS, DAY, NEXT_DAY, build
from tools.replay_parity import compare_logs

REPO = Path(__file__).resolve().parents[1]
MINUTE_NS = 60 * 1_000_000_000

#: Identities for the hand-built logs. Explicitly synthetic: they identify no
#: configuration, no contract and no protocol that exists.
CONFIG_HASH = "sha256:" + "a" * 64
CONTRACT_HASH = "sha256:" + "b" * 64
SOFTWARE = {
    "revision": "SYNTHETIC",
    "source_digest": "SYNTHETIC",
    "dirty": False,
    "python": "3.11",
}

#: The twelve record kinds of section 9.1, written out. Compared against the
#: report's own key set, so a kind renamed in the enum fails here rather than
#: agreeing with itself.
TWELVE_KINDS = (
    "DECISION",
    "FUNDING",
    "HALT",
    "INCOMPLETE_STATE",
    "LIQUIDATION_TOUCH",
    "OPERATOR",
    "RECONCILIATION",
    "RECOVERY",
    "RESUME",
    "SHUTDOWN",
    "SKIPPED_STALE",
    "STARTUP",
)

#: Section 9.4's classes for five kinds, written out from the adopted plan rather
#: than derived from the sets the module imports.
CLASS_ORACLE = {
    "DECISION": "evidence",
    "STARTUP": "operational",
    "HALT": "unclassified",
    "RESUME": "unclassified",
    "RECOVERY": "unclassified",
}

#: The top-level field names a daily report promises. A literal contract: adding
#: a field is a deliberate change to this tuple, and dropping one is a failure.
DAILY_FIELDS = (
    "campaign",
    "chain",
    "day",
    "funding",
    "generated_from",
    "halts",
    "input_coverage",
    "ledger",
    "liquidation",
    "minutes",
    "operator",
    "orders",
    "parity",
    "parity_note",
    "position_close",
    "purity",
    "reconciliation",
    "record_class_by_kind",
    "records_by_kind",
    "report_class",
    "rules",
    "schema",
    "vetoes",
)

#: ``generated_from``'s names. Written out because one of them --
#: ``records_without_minute`` -- is the only place a record the log admits but no
#: day can claim is visible at all.
GENERATED_FROM_FIELDS = (
    "day_file_present",
    "day_files",
    "log_dir",
    "records_scanned",
    "records_selected",
    "records_without_minute",
    "selection_rule",
    "state_dir",
)

#: The funding block's names. Written out because the paid/received/net split is
#: the field set amendment A10 governs, and a report that emitted only `net`
#: would be a report that netted funding.
FUNDING_FIELDS = (
    "net",
    "paid",
    "received",
    "records_of_kind_FUNDING",
    "settlement_minutes",
    "sign_convention",
    "source",
)

#: Words a daily report may not use as a key. It counts; it does not score.
SCORE_SHAPED_KEYS = frozenset(
    {
        "score",
        "pass",
        "fail",
        "verdict",
        "significance",
        "p_value",
        "sharpe",
        "alpha",
        "edge",
        "hypothesis",
        "preregistered",
    }
)

#: Calls that can change a file. ``chimera/demo/reports.py`` may contain none.
WRITING_CALLS = frozenset(
    {
        "open",
        "write",
        "writelines",
        "write_text",
        "write_bytes",
        "mkdir",
        "makedirs",
        "touch",
        "unlink",
        "remove",
        "rename",
        "replace",
        "rmtree",
        "truncate",
        "fsync",
        "recover_tail",
    }
)


# ---------------------------------------------------------------------------
# hand-built logs: the writer is real, the numbers are the test's
# ---------------------------------------------------------------------------
def _ns(minute: str) -> int:
    """The nanosecond instant of a canonical ISO minute."""
    return int(datetime.fromisoformat(minute).timestamp()) * 1_000_000_000


def _base(kind: str, minute: str, *, now_ns: int | None = None) -> dict:
    return {
        "kind": kind,
        "minute": minute,
        "runner_now_ns": _ns(minute) + MINUTE_NS if now_ns is None else now_ns,
        "config_hash": CONFIG_HASH,
        "software": dict(SOFTWARE),
    }


def _ledger(fees="0", slippage="0", funding="0", realised="0", equity="1000000") -> dict:
    return {
        "fees": fees,
        "slippage": slippage,
        "funding": funding,
        "realised": realised,
        "equity": equity,
    }


def _decision(minute: str, *, now_ns: int | None = None, execution=None, **ledger) -> dict:
    record = _base("DECISION", minute, now_ns=now_ns)
    record["inputs"] = {"contract_hash": CONTRACT_HASH}
    record["rule"] = {"id": "R1_carry", "version": "1.0.0"}
    record["signal"] = {"target_qty": "0", "reason": "hold", "kind": "target"}
    record["shadow"] = []
    record["execution"] = list(execution or [])
    record["position_after"] = {
        "hedge_state": "FLAT",
        "spot_qty": "0",
        "perp_qty": "0",
        "imbalance": "0",
    }
    record["ledger_effect"] = _ledger(**ledger)
    record["veto_or_rejection"] = None
    return record


def _write(state_dir: Path, payloads) -> Path:
    """Append hand-written payloads through the real chained writer."""
    state_dir.mkdir(parents=True, exist_ok=True)
    log = DecisionLog.open(state_dir)
    try:
        for payload in payloads:
            log.append(payload)
    finally:
        log.close()
    return state_dir / "decision_log"


def _digests(root: Path) -> dict[str, str]:
    """``{relative posix path: sha256}`` for every file under ``root``."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _campaign(tmp_path: Path, minutes: int = 6):
    """One real campaign over the synthetic fixture, shut down cleanly."""
    harness = build(tmp_path, days=(DAY, NEXT_DAY))
    first = harness.first_minute_ms()
    for index in range(minutes):
        harness.tick(first + index * 60_000)
    harness.runner.shutdown("report tests")
    return harness


# ---------------------------------------------------------------------------
# what the day held
# ---------------------------------------------------------------------------
def test_the_daily_report_counts_the_records_the_log_holds(tmp_path):
    """The oracle is the NDJSON, read again here and tallied independently."""
    harness = _campaign(tmp_path)
    report = daily_report(harness.state_dir, DAY)

    expected = Counter()
    for path in sorted((harness.state_dir / "decision_log").glob("*.ndjson")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["minute"][:10] == DAY:
                expected[record["kind"]] += 1

    assert {k: v for k, v in report["records_by_kind"].items() if v} == dict(expected)
    assert report["generated_from"]["records_selected"] == sum(expected.values())


def test_a_daily_report_over_an_absent_log_refuses_rather_than_reporting_zeros(tmp_path):
    """A directory with no log is not a day on which nothing happened.

    Before this, a report over a state directory that held no `decision_log/` at
    all returned `chain.ok = True`, `"0 record(s), chain intact"`, every count
    zero and exit 0 -- indistinguishable from a day the runner sat idle. That is
    the same silent zero `input_coverage` exists to prevent, and the likeliest
    way to meet it is the likeliest operator mistake: the wrong `state_dir`.
    """
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(ReportRefused, match="no decision log"):
        daily_report(empty, DAY)

    # A log directory that exists but holds no day file is the same fact, said
    # differently: the runner writes STARTUP the first time it starts, so an
    # empty directory means it never ran here.
    (empty / "decision_log").mkdir()
    with pytest.raises(ReportRefused, match="never ran"):
        daily_report(empty, DAY)


def test_a_day_with_no_file_of_its_own_is_reported_and_flagged(tmp_path):
    """The other side: a log that exists, and a day inside it that has no file.

    That IS a real zero -- the campaign is running and this day has no records --
    so it is reported rather than refused, with `day_file_present` false so the
    two cases cannot be confused by a reader of the JSON.
    """
    harness = _campaign(tmp_path)
    report = daily_report(harness.state_dir, "2020-01-01")
    assert report["generated_from"]["day_file_present"] is False
    assert report["minutes"]["processed"] == 0
    assert daily_report(harness.state_dir, DAY)["generated_from"]["day_file_present"] is True


def test_all_twelve_kinds_are_present_and_zero_filled(tmp_path):
    harness = _campaign(tmp_path)
    report = daily_report(harness.state_dir, DAY)
    assert sorted(report["records_by_kind"]) == sorted(TWELVE_KINDS)
    assert sorted(report["record_class_by_kind"]) == sorted(TWELVE_KINDS)
    assert sorted(report["input_coverage"]["by_kind"]) == sorted(TWELVE_KINDS)


def test_the_daily_report_field_names_are_the_ones_it_promises(tmp_path):
    harness = _campaign(tmp_path)
    report = daily_report(harness.state_dir, DAY)
    assert tuple(sorted(report)) == DAILY_FIELDS
    assert tuple(sorted(report["generated_from"])) == GENERATED_FROM_FIELDS
    assert tuple(sorted(report["funding"])) == FUNDING_FIELDS
    assert tuple(sorted(report["ledger"])) == (
        "baseline",
        "equity",
        "fees",
        "realised",
        "slippage",
    )
    assert tuple(sorted(report["ledger"]["equity"])) == (
        "close",
        "open",
        "worst",
        "worst_minute",
    )


def test_the_last_minute_of_a_day_is_reported_on_its_own_day(tmp_path):
    """The 23:59 record is filed into the NEXT day's file and belongs to this one.

    ``DecisionLog.append`` files by ``utc_day(runner_now_ns)`` and the runner sets
    that clock to ``minute + 1``, so the last minute of every day lands in the
    next day's file. A report keyed on the file name would move it, silently, for
    every day of every campaign.
    """
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-19T23:58:00+00:00"),
            _decision("2026-09-19T23:59:00+00:00"),
            _decision("2026-09-20T00:00:00+00:00"),
        ],
    )
    assert sorted(p.name for p in (state_dir / "decision_log").glob("*.ndjson")) == [
        "2026-09-19.ndjson",
        "2026-09-20.ndjson",
    ]

    today = daily_report(state_dir, "2026-09-19")
    tomorrow = daily_report(state_dir, "2026-09-20")

    assert today["minutes"]["last"] == "2026-09-19T23:59:00+00:00"
    assert today["generated_from"]["records_selected"] == 2
    assert tomorrow["minutes"]["first"] == "2026-09-20T00:00:00+00:00"
    assert tomorrow["generated_from"]["records_selected"] == 1
    assert today["generated_from"]["day_files"] == [
        "2026-09-19.ndjson",
        "2026-09-20.ndjson",
    ]
    assert today["generated_from"]["records_scanned"] == 3


# ---------------------------------------------------------------------------
# a zero that means "unwritable" is not a zero that means "none happened"
# ---------------------------------------------------------------------------
def test_the_runner_written_kinds_are_the_runners_own_append_sites(tmp_path):
    """The oracle is ``chimera/demo/runner.py``'s AST, not this module's constant.

    Without this, the disclosure the daily report carries would be a comment that
    could drift: the day the runner learns to write a ``FUNDING`` record, every
    report would go on saying it cannot. That day was PR-10R, and this test is
    what made the stale disclosure impossible to leave behind.
    """
    tree = ast.parse((REPO / "chimera" / "demo" / "runner.py").read_text(encoding="utf-8"))
    written = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "_append"):
            continue
        first = node.args[0]
        assert isinstance(first, ast.Attribute), ast.dump(first)
        assert isinstance(first.value, ast.Name) and first.value.id == "RecordKind"
        written.add(first.attr)
    assert sorted(written) == sorted(reports.RUNNER_WRITTEN_KINDS)
    # The second, independent oracle. Every one of section 9.1's kinds now has a
    # writer: PR-10R closed the D1 gap, and this equality is what says so from
    # the runner's own source rather than from a constant beside it.
    assert set(TWELVE_KINDS) - written == set()


def test_input_coverage_separates_absent_from_unwritable(tmp_path):
    """A campaign log: absent kinds, and every one of them writable.

    The block's whole job is to keep those two facts apart. On this build the
    second is uniformly true, so every zero below is an observation -- which is
    the claim the block has to be able to make, and could not before PR-10R.
    """
    harness = _campaign(tmp_path)
    coverage = daily_report(harness.state_dir, DAY)["input_coverage"]["by_kind"]

    assert coverage["FUNDING"] == {"present": False, "records": 0, "runner_can_write": True}
    for kind in ("RECONCILIATION", "LIQUIDATION_TOUCH", "RECOVERY", "SKIPPED_STALE"):
        assert coverage[kind]["runner_can_write"] is True, kind
    assert coverage["HALT"] == {"present": False, "records": 0, "runner_can_write": True}
    assert coverage["DECISION"]["present"] is True
    assert coverage["DECISION"]["runner_can_write"] is True


def test_input_coverage_marks_a_kind_present_when_the_log_holds_one(tmp_path):
    """The two-sided control: on a log that DOES hold them, present goes true."""
    state_dir = tmp_path / "state"
    minute = "2026-09-19T00:00:00+00:00"
    _write(
        state_dir,
        [
            _decision(minute),
            {**_base("FUNDING", minute), "ledger_effect": _ledger(funding="-4.20")},
            {**_base("RECONCILIATION", minute), "veto_or_rejection": None},
            {**_base("LIQUIDATION_TOUCH", minute)},
            {**_base("RECOVERY", minute)},
            {**_base("SKIPPED_STALE", minute)},
        ],
    )
    coverage = daily_report(state_dir, DAY)["input_coverage"]["by_kind"]
    for kind in (
        "FUNDING",
        "RECONCILIATION",
        "LIQUIDATION_TOUCH",
        "RECOVERY",
        "SKIPPED_STALE",
    ):
        assert coverage[kind]["present"] is True, kind
        assert coverage[kind]["records"] == 1, kind
        assert coverage[kind]["runner_can_write"] is True, kind


def test_a_record_with_no_minute_is_counted_rather_than_dropped(tmp_path):
    """The log admits ``minute: null`` for a kind that has no decision minute.

    Such a record belongs to no UTC day, so it can be in no day's report -- and
    silently vanishing between ``records_scanned`` and ``records_selected`` would
    make a report that had skipped records look exactly like one that had not.
    """
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00"),
            {
                "kind": "SHUTDOWN",
                "minute": None,
                "runner_now_ns": _ns("2026-09-19T00:01:00+00:00"),
                "config_hash": CONFIG_HASH,
                "software": dict(SOFTWARE),
            },
        ],
    )
    generated = daily_report(state_dir, DAY)["generated_from"]
    assert generated["records_scanned"] == 2
    assert generated["records_selected"] == 1
    assert generated["records_without_minute"] == 1
    assert daily_report(state_dir, DAY)["records_by_kind"]["SHUTDOWN"] == 0


# ---------------------------------------------------------------------------
# purity, proven mechanically
# ---------------------------------------------------------------------------
def test_the_daily_report_changes_no_byte_under_the_state_directory(tmp_path):
    harness = _campaign(tmp_path)
    before = _digests(harness.state_dir)
    assert "carry_ledger.json" in before and "risk.json" in before

    report = daily_report(harness.state_dir, DAY)

    after = _digests(harness.state_dir)
    assert after == before
    assert set(after) == set(before)
    assert report["purity"]["written"] == []
    assert report["purity"]["read"] == [
        path.as_posix()
        for path in sorted((harness.state_dir / "decision_log").glob("*.ndjson"))
    ]


def test_the_report_module_calls_nothing_that_can_write(tmp_path):
    """The guard the digests cannot provide: ``open(store, "r+")`` moves no byte."""
    source = (REPO / "chimera" / "demo" / "reports.py").read_text(encoding="utf-8")
    offenders = _writing_calls(ast.parse(source))
    assert offenders == [], offenders


def test_the_writing_call_guard_catches_a_planted_write(tmp_path):
    """The negative control. A matcher that has stopped matching proves nothing."""
    planted = ast.parse(
        "from pathlib import Path\n"
        "def daily_report(state_dir, day):\n"
        "    Path(state_dir, 'carry_ledger.json').write_text('{}')\n"
    )
    assert _writing_calls(planted) == ["write_text"]


def _writing_calls(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name in WRITING_CALLS:
            found.append(name)
    return sorted(found)


def test_the_report_module_imports_no_tool_and_no_second_kind_set(tmp_path):
    """``tools/replay_parity.py``'s OPERATIONAL_KINDS is a DIFFERENT set.

    Section 10's alignment rule holds ``HALT`` and ``RESUME`` and does not hold
    ``INCOMPLETE_STATE`` or ``SKIPPED_STALE``; section 9.4's operational set is
    the other way round. Importing one where the other belongs would move the
    evidence boundary without changing a line of the boundary's own definition.
    """
    tree = ast.parse((REPO / "chimera" / "demo" / "reports.py").read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert [m for m in modules if m.startswith("tools")] == []
    assert [m for m in modules if m.startswith("chimera")] == ["chimera.demo.decision_log"]

    from chimera.demo.decision_log import OPERATIONAL_KINDS
    from tools import replay_parity

    # Both sets written out. They differ in four of the twelve kinds, and the
    # difference is exactly where a mix-up would move the evidence boundary.
    assert {kind.value for kind in OPERATIONAL_KINDS} == {
        "STARTUP",
        "SHUTDOWN",
        "INCOMPLETE_STATE",
        "SKIPPED_STALE",
    }
    assert set(replay_parity.OPERATIONAL_KINDS) == {
        "STARTUP",
        "SHUTDOWN",
        "RECOVERY",
        "HALT",
        "RESUME",
    }
    assert reports.kind_class("HALT") == "unclassified"
    assert reports.kind_class("INCOMPLETE_STATE") == "operational"


def test_a_day_report_through_the_cli_creates_no_file(tmp_path, monkeypatch, capsys):
    from tools import demo_report

    harness = _campaign(tmp_path)
    config_path = _write_config(tmp_path, harness)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    before = _digests(harness.state_dir)
    assert (
        demo_report.main(["--config", str(config_path), "--profile", "TEST", "--day", DAY])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["day"] == DAY
    assert list(workdir.iterdir()) == []
    assert _digests(harness.state_dir) == before


def _write_config(tmp_path: Path, harness) -> Path:
    """The harness's own campaign configuration, on disk for the CLI to read."""
    payload = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text(encoding="utf-8"))
    payload["profile"] = harness.runner.config.profile.value
    payload["runner"] = {"state_dir": str(harness.state_dir)}
    payload["rules"] = {"R1_carry": dict(CARRY_PARAMS)}
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_daily_report_repairs_no_torn_tail(tmp_path):
    harness = _campaign(tmp_path)
    last = sorted((harness.state_dir / "decision_log").glob("*.ndjson"))[-1]
    size = last.stat().st_size
    with open(last, "ab") as handle:
        handle.write(b"x" * 40)

    report = daily_report(harness.state_dir, DAY)

    assert report["chain"]["ok"] is False
    assert "TORN_TAIL" in report["chain"]["faults"]
    assert report["chain"]["torn_tail_bytes"] == 40
    assert last.stat().st_size == size + 40
    assert not list((harness.state_dir / "decision_log").glob("*.truncated"))


def test_a_forged_record_is_reported_not_hidden(tmp_path):
    harness = _campaign(tmp_path)
    path = sorted((harness.state_dir / "decision_log").glob("*.ndjson"))[0]
    lines = path.read_bytes().decode("utf-8").splitlines()
    index = max(i for i, line in enumerate(lines) if json.loads(line)["kind"] == "DECISION")
    forged = json.loads(lines[index])
    forged["ledger_effect"]["fees"] = "999999"
    lines[index] = json.dumps(forged, sort_keys=True, separators=(",", ":"))
    # Bytes: the log is byte-canonical, and `write_text` turns every "\n"
    # into "\r\n" on Windows -- which makes EVERY record non-canonical, so
    # the runner reports NON_CANONICAL_BYTES instead of the forgery this
    # test is about and the assertion below passes for the wrong reason.
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))

    report = daily_report(harness.state_dir, DAY)

    assert report["chain"]["ok"] is False
    assert "RECORD_HASH_MISMATCH" in report["chain"]["faults"]
    assert report["records_by_kind"]["DECISION"] >= 1


# ---------------------------------------------------------------------------
# the arithmetic, against numbers computed here
# ---------------------------------------------------------------------------
def test_the_ledger_deltas_are_close_minus_open(tmp_path):
    """A cumulative total reported as a daily delta is the classic error here."""
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-18T23:59:00+00:00", fees="1.5", slippage="0.25", realised="-1"),
            _decision(
                "2026-09-19T00:00:00+00:00", fees="2.25", slippage="0.50", realised="-1"
            ),
            _decision(
                "2026-09-19T00:01:00+00:00", fees="4.00", slippage="0.75", realised="-3"
            ),
            _decision(
                "2026-09-19T00:02:00+00:00", fees="7.75", slippage="1.25", realised="-4"
            ),
        ],
    )
    report = daily_report(state_dir, DAY)

    raw = []
    for path in sorted((state_dir / "decision_log").glob("*.ndjson")):
        raw += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    opening = raw[0]["ledger_effect"]
    closing = raw[-1]["ledger_effect"]

    assert report["ledger"]["baseline"]["source"] == "previous_record"
    assert report["ledger"]["baseline"]["minute"] == "2026-09-18T23:59:00+00:00"
    for name, literal in (("fees", "6.25"), ("slippage", "1.00"), ("realised", "-3")):
        block = report["ledger"][name]
        assert block["open"] == opening[name]
        assert block["close"] == closing[name]
        assert Decimal(block["delta"]) == Decimal(closing[name]) - Decimal(opening[name])
        assert block["delta"] == literal, name


def test_a_campaign_start_baseline_opens_the_cost_fields_at_zero(tmp_path):
    """The two-sided control for the baseline: no earlier record, so open is 0."""
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00", fees="2.25", equity="900"),
            _decision("2026-09-19T00:01:00+00:00", fees="7.75", equity="800"),
        ],
    )
    report = daily_report(state_dir, DAY)
    assert report["ledger"]["baseline"]["source"] == "campaign_start"
    assert report["ledger"]["fees"] == {"open": "0", "close": "7.75", "delta": "7.75"}
    assert report["ledger"]["equity"]["open"] == "900"


def test_a_paid_settlement_is_reported_as_paid_not_received(tmp_path):
    """`0 -> -4.20 -> -4.20`: one settlement, paid, and no second one."""
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00", funding="0"),
            _decision("2026-09-19T00:01:00+00:00", funding="-4.20"),
            _decision("2026-09-19T00:02:00+00:00", funding="-4.20"),
        ],
    )
    funding = daily_report(state_dir, DAY)["funding"]
    assert funding["paid"] == "4.20"
    assert funding["received"] == "0"
    assert funding["net"] == "-4.20"
    assert funding["settlement_minutes"] == 1


def test_a_received_settlement_is_reported_as_received(tmp_path):
    """The other side of the sign. `0 -> +1.75`."""
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00", funding="0"),
            _decision("2026-09-19T00:01:00+00:00", funding="1.75"),
        ],
    )
    funding = daily_report(state_dir, DAY)["funding"]
    assert funding["received"] == "1.75"
    assert funding["paid"] == "0"
    assert funding["net"] == "1.75"
    assert funding["settlement_minutes"] == 1


def test_funding_paid_and_received_are_never_netted(tmp_path):
    """Three separate fields. -4.20 then +1.75 is not "-2.45 and nothing else"."""
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00", funding="0"),
            _decision("2026-09-19T00:01:00+00:00", funding="-4.20"),
            _decision("2026-09-19T00:02:00+00:00", funding="-2.45"),
        ],
    )
    funding = daily_report(state_dir, DAY)["funding"]
    assert funding["paid"] == "4.20"
    assert funding["received"] == "1.75"
    assert funding["net"] == "-2.45"
    assert funding["settlement_minutes"] == 2
    # Computed here from the same two magnitudes, so the identity is checked and
    # not assumed: net is received minus paid, which is CarryLedgerState's own.
    assert Decimal(funding["net"]) == Decimal(funding["received"]) - Decimal(funding["paid"])


#: Amendment A10's four cases, written out from `docs/amendment_a10_funding_sign.md`.
#: `funding_cash_flow = -sign(side) * notional * rate`, and a negative cash flow
#: means the position paid.
A10_TABLE = (
    ("LONG", 1, "0.0001", "paid"),
    ("LONG", 1, "-0.0001", "received"),
    ("SHORT", -1, "0.0001", "received"),
    ("SHORT", -1, "-0.0001", "paid"),
)


@pytest.mark.parametrize("side,sign,rate,column", A10_TABLE)
def test_amendment_a10s_four_cases_land_in_the_column_a10_names(
    tmp_path, side, sign, rate, column
):
    notional = Decimal("30000")
    cash_flow = -Decimal(sign) * notional * Decimal(rate)
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00", funding="0"),
            _decision("2026-09-19T00:01:00+00:00", funding=format(cash_flow, "f")),
        ],
    )
    funding = daily_report(state_dir, DAY)["funding"]
    other = "received" if column == "paid" else "paid"

    assert Decimal(funding[column]) == abs(cash_flow), (side, rate)
    assert Decimal(funding[other]) == 0, (side, rate)
    assert Decimal(funding["net"]) == cash_flow


def test_a_funding_record_moves_the_split_like_a_decision_does(tmp_path):
    """D1's requirement: right on a log holding the kinds the runner cannot write.

    The settlement is the day's LAST record, so a report that iterated only the
    DECISION records would report no funding at all rather than a smaller number,
    which is the failure this discriminates.
    """
    state_dir = tmp_path / "state"
    minute = "2026-09-19T00:01:00+00:00"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00", funding="0", equity="1000"),
            {
                **_base("FUNDING", minute),
                "ledger_effect": _ledger(funding="-4.20", equity="1000"),
            },
        ],
    )
    report = daily_report(state_dir, DAY)
    assert report["funding"]["paid"] == "4.20"
    assert report["funding"]["net"] == "-4.20"
    assert report["funding"]["settlement_minutes"] == 1
    assert report["funding"]["records_of_kind_FUNDING"] == 1
    assert report["input_coverage"]["by_kind"]["FUNDING"]["present"] is True


def test_worst_equity_is_the_days_minimum_and_names_its_minute(tmp_path):
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00", equity="100"),
            _decision("2026-09-19T00:01:00+00:00", equity="90"),
            _decision("2026-09-19T00:02:00+00:00", equity="95"),
        ],
    )
    equity = daily_report(state_dir, DAY)["ledger"]["equity"]
    assert equity == {
        "open": "100",
        "close": "95",
        "worst": "90",
        "worst_minute": "2026-09-19T00:01:00+00:00",
    }


def test_orders_are_counted_and_summed_per_leg(tmp_path):
    """Fees and quantities summed here, from the strings the records carry."""
    state_dir = tmp_path / "state"
    execution = [
        {
            "leg": "spot",
            "order_id": "S-1",
            "state": "FILLED",
            "filled_qty": "0.500",
            "fees": "1.25",
        },
        {
            "leg": "perp",
            "order_id": "P-1",
            "state": "FILLED",
            "filled_qty": "0.500",
            "fees": "0.75",
        },
    ]
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00", execution=execution),
            _decision(
                "2026-09-19T00:01:00+00:00",
                execution=[
                    {
                        "leg": "spot",
                        "order_id": "S-2",
                        "state": "PARTIALLY_FILLED",
                        "filled_qty": "0.125",
                        "fees": "0.25",
                    }
                ],
            ),
        ],
    )
    orders = daily_report(state_dir, DAY)["orders"]
    assert orders["spot"]["orders"] == 2
    assert orders["spot"]["entries"] == 2
    assert Decimal(orders["spot"]["filled_quantity"]) == Decimal("0.500") + Decimal("0.125")
    assert Decimal(orders["spot"]["fees"]) == Decimal("1.25") + Decimal("0.25")
    assert orders["spot"]["by_state"] == {"FILLED": 1, "PARTIALLY_FILLED": 1}
    assert orders["perp"]["orders"] == 1
    assert Decimal(orders["perp"]["fees"]) == Decimal("0.75")


# ---------------------------------------------------------------------------
# counted, and not classified
# ---------------------------------------------------------------------------
def test_halt_resume_and_recovery_are_counted_and_left_unclassified(tmp_path):
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00"),
            {
                **_base("HALT", "2026-09-19T00:01:00+00:00"),
                "veto_or_rejection": {
                    "stage": "runner",
                    "label": "halt",
                    "detail": "dispute: the two legs disagree",
                },
            },
            {
                **_base("RESUME", "2026-09-19T00:02:00+00:00"),
                "operator": {
                    "command": "resume",
                    "note": "stores reconciled by hand",
                    "cleared": "dispute: the two legs disagree",
                },
            },
            {**_base("RECOVERY", "2026-09-19T00:03:00+00:00")},
        ],
    )
    report = daily_report(state_dir, DAY)
    classes = report["record_class_by_kind"]

    for kind, expected in CLASS_ORACLE.items():
        assert classes[kind] == expected, kind
    assert report["halts"]["count"] == 1
    assert report["halts"]["causes"] == {"dispute": 1}
    assert report["halts"]["events"][0]["detail"] == "dispute: the two legs disagree"
    assert report["halts"]["recoveries"] == 1
    assert report["halts"]["resumes"][0]["cleared"] == "dispute: the two legs disagree"
    assert report["halts"]["classification"] == "unclassified"
    assert report["reconciliation"]["dispute_halts"] == 1


def test_the_daily_report_carries_no_score_shaped_field(tmp_path):
    harness = _campaign(tmp_path)
    report = daily_report(harness.state_dir, DAY)
    assert report["report_class"] == "operational"

    offenders = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in SCORE_SHAPED_KEYS:
                    offenders.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(report, "report")
    assert offenders == [], offenders
    # And no total aggregated by section 9.4 class, which is what a score would
    # be built out of.
    assert not [key for key in report if key.endswith("_records")]


#: `halt_cause`'s table, written out from the `halt(...)` sites it quotes.
#: The last row is the two-sided control: an unanticipated reason becomes
#: `other` and is never renamed into one of the labels above it.
HALT_CAUSE_ORACLE = (
    ("kill_switch", "kill_switch"),
    ("kill_switch: the switch path could not be examined", "kill_switch"),
    ("dispute: the two legs disagree", "dispute"),
    ("carry_dispute: identity gap 12", "dispute"),
    ("rule_exception: R1_carry: boom", "rule_exception"),
    ("identity_violation", "identity_violation"),
    (
        "more than one rule produced an actionable target: ['R1', 'R2']",
        "multiple_actionable_rules",
    ),
    ("source_identity: the working tree is dirty", "source_identity"),
    # PR-10R: `log_behind_state:` is no longer a halt reason. Section 9.3's
    # LOG_BEHIND_STATE is a RECOVERY cause the runner continues from, so a halt
    # carrying that text would be a reason nobody wrote a prefix for -- and the
    # oracle says exactly that rather than pretending the label still exists.
    ("log_behind_state: the decision log's tail is x", "other"),
    ("log_forged: a complete record does not verify", "log_forged"),
    ("liquidation_touch: emergency reduce: risk_halt", "liquidation_touch"),
    ("liquidation_unknown: the minute has no mark", "liquidation_unknown"),
    ("funding_window_unknown: no open instant", "funding_unbookable"),
    ("funding_unbookable: no mark_price", "funding_unbookable"),
    ("funding_not_booked: settlement 1 fell in the window", "funding_unbookable"),
    ("funding_source_unreadable: bad row", "funding_unbookable"),
    ("reconciliation_error: perp BTC/USDT:USDT: boom", "reconciliation_error"),
    ("store_error: no such file", "store_error"),
    ("max daily loss breached: 3.00% >= 2.00%", "daily_loss"),
    ("max drawdown breached: 6.00% >= 5.00%", "drawdown"),
    ("order rate limit exceeded: 5 orders/min > 4", "order_rate"),
    ("equity is non-positive: -1.0", "non_positive_equity"),
    ("risk halted", "risk_halted"),
    ("something nobody wrote a prefix for", "other"),
)


#: The three modules that construct a halt reason. Read out of the tree rather
#: than restated, because the point of the test below is that the table in
#: `reports.py` and these files cannot drift apart silently.
HALT_SITE_SOURCES: tuple[str, ...] = (
    "chimera/demo/runner.py",
    "chimera/risk.py",
    "chimera/carry/hedge.py",
)


def test_every_halt_cause_prefix_still_exists_at_a_halt_site():
    """`HALT_CAUSES` says every prefix is quoted from a live `halt(...)` call.

    Nothing checked that. The prefixes are matched against a free-text reason, so
    a reason that is reworded -- `carry_dispute: ` to `carry-dispute: `, say --
    stops matching, silently collapses to `other`, and the daily report's
    `reconciliation.dispute_halts` quietly becomes 0 while every test stays
    green. A count that goes to zero because a message was reworded is worse
    than no count.

    Read as text across the three modules that build a reason, because the
    reasons are f-strings and the prefix is the literal part; what matters is
    that the literal still occurs where a halt is raised.
    """
    from chimera.demo.reports import HALT_CAUSE_OTHER, HALT_CAUSES

    corpus = "\n".join(
        (REPO / relative).read_text(encoding="utf-8") for relative in HALT_SITE_SOURCES
    )
    missing = [prefix for prefix, _label in HALT_CAUSES if prefix not in corpus]
    assert not missing, (
        "HALT_CAUSES names prefixes that no longer appear at any halt site, so a "
        f"real halt would now collapse to {HALT_CAUSE_OTHER!r}: {missing}"
    )


def test_the_halt_prefix_guard_catches_a_reworded_reason():
    """The negative control, without editing the tree: a prefix nothing raises."""
    from chimera.demo.reports import HALT_CAUSES

    corpus = "\n".join(
        (REPO / relative).read_text(encoding="utf-8") for relative in HALT_SITE_SOURCES
    )
    assert "carry-dispute: " not in corpus, "the reworded spelling must not exist"
    assert any(prefix == "carry_dispute:" for prefix, _ in HALT_CAUSES)


@pytest.mark.parametrize("detail,label", HALT_CAUSE_ORACLE)
def test_a_halt_reason_collapses_to_the_label_its_call_site_implies(detail, label):
    assert reports.halt_cause(detail) == label


def test_a_reconciliation_record_is_quoted_and_never_re_derived(tmp_path):
    """D1's requirement again: correct on a log holding RECONCILIATION records."""
    state_dir = tmp_path / "state"
    minute = "2026-09-19T00:01:00+00:00"
    detail = "spot store says 0.5, venue says 0.4"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00"),
            {
                **_base("RECONCILIATION", minute),
                "veto_or_rejection": {
                    "stage": "venue",
                    "label": "reconciliation",
                    "detail": detail,
                },
                "position_after": {"hedge_state": "DISPUTED", "spot_qty": "0.5"},
            },
            {
                **_base("OPERATOR", "2026-09-19T00:02:00+00:00"),
                "operator": {
                    "command": "resolve",
                    "symbol": "BTCUSDT",
                    "note": "venue confirmed 0.4",
                },
            },
        ],
    )
    block = daily_report(state_dir, DAY)["reconciliation"]

    assert block["records_of_kind_RECONCILIATION"] == 1
    assert block["outcomes"] == [
        {
            "minute": minute,
            "seq": 2,
            "detail": detail,
            "position_after": {"hedge_state": "DISPUTED", "spot_qty": "0.5"},
        }
    ]
    assert block["operator_resolutions"] == [
        {
            "minute": "2026-09-19T00:02:00+00:00",
            "seq": 3,
            "symbol": "BTCUSDT",
            "note": "venue confirmed 0.4",
        }
    ]


def test_reconciliation_outcomes_are_null_rather_than_an_empty_clean_bill(tmp_path):
    """The two-sided control. Null says "no outcome recorded", not "all fine".

    Built on a log holding only a decision, because a real campaign now
    reconciles: PR-10R wired section 8.1's post-execution and hourly checks, so
    ``_campaign`` produces the records this case exists to be the absence of.
    """
    state_dir = tmp_path / "state"
    _write(state_dir, [_decision("2026-09-19T00:00:00+00:00")])
    block = daily_report(state_dir, DAY)["reconciliation"]
    assert block["records_of_kind_RECONCILIATION"] == 0
    assert block["outcomes"] is None


def test_a_campaign_reconciles_and_the_report_quotes_the_outcome(tmp_path):
    """The other side: a real campaign's reconciliation reaches the daily report.

    Before PR-10R this was unwritable -- the runner never called
    ``FuturesExecutor.reconcile`` -- so the block could only ever be null on a
    real log, and null was indistinguishable from "the check is not wired".
    """
    harness = _campaign(tmp_path)
    block = daily_report(harness.state_dir, DAY)["reconciliation"]
    assert block["records_of_kind_RECONCILIATION"] >= 1
    assert block["outcomes"] is not None and block["outcomes"]
    first = block["outcomes"][0]
    assert first["minute"] and first["seq"]
    # An agreement carries no veto_or_rejection, and the report says so rather
    # than inventing a "detail" for a check that found nothing wrong.
    assert first["detail"] is None


def test_a_liquidation_touch_is_reported_when_the_log_holds_one(tmp_path):
    state_dir = tmp_path / "state"
    minute = "2026-09-19T00:01:00+00:00"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00"),
            {
                **_base("LIQUIDATION_TOUCH", minute),
                "position_after": {"hedge_state": "HEDGED", "perp_qty": "0.5"},
                "veto_or_rejection": {
                    "stage": "venue",
                    "label": "liquidation",
                    "detail": "perp within 0.2% of liquidation",
                },
            },
        ],
    )
    block = daily_report(state_dir, DAY)["liquidation"]
    assert block["records_of_kind_LIQUIDATION_TOUCH"] == 1
    assert block["events"][0]["detail"] == "perp within 0.2% of liquidation"
    assert block["events"][0]["position_after"] == {"hedge_state": "HEDGED", "perp_qty": "0.5"}


# ---------------------------------------------------------------------------
# parity, Markdown, determinism, replay
# ---------------------------------------------------------------------------
def test_a_parity_verdict_is_embedded_verbatim_when_present(tmp_path):
    harness = _campaign(tmp_path)
    parity_dir = tmp_path / "parity"
    parity_dir.mkdir()
    verdict = {
        "label": "PARITY",
        "parity": True,
        "records_compared": 6,
        "divergences": 0,
        "live_only": [],
        "replay_only": [],
        "environment_note": "",
        "first_divergence": None,
    }
    (parity_dir / f"{DAY}.json").write_text(json.dumps(verdict), encoding="utf-8")

    report = daily_report(harness.state_dir, DAY, parity_dir=parity_dir)

    assert report["parity"]["present"] is True
    assert report["parity"]["report"] == json.loads(
        (parity_dir / f"{DAY}.json").read_text(encoding="utf-8")
    )


def test_a_missing_parity_verdict_is_null_and_not_a_pass(tmp_path):
    harness = _campaign(tmp_path)
    report = daily_report(harness.state_dir, DAY, parity_dir=tmp_path / "nowhere")
    assert report["parity"] is None
    markdown = render_daily_markdown(report)
    assert report["parity_note"] in markdown
    assert "PARITY" not in markdown


def test_the_markdown_quotes_the_json_numbers_verbatim(tmp_path):
    """Every money string in the Markdown is a WHOLE token of the JSON's own.

    Substring matching is not enough and was not enough here: a renderer that
    formatted `"0"` as `"0.00"` still contains `"0"`, so the check has to be on
    delimited tokens. The values below are chosen so a reformat changes one.
    """
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision(
                "2026-09-18T23:59:00+00:00",
                fees="1.50",
                slippage="0.25",
                funding="0",
                realised="-1",
                equity="1000000",
            ),
            _decision(
                "2026-09-19T00:00:00+00:00",
                fees="4.00",
                slippage="0.75",
                funding="-4.20",
                realised="-3",
                equity="999000.125",
            ),
            _decision(
                "2026-09-19T00:01:00+00:00",
                fees="7.75",
                slippage="1.25",
                funding="-2.45",
                realised="-4",
                equity="999500.5",
            ),
        ],
    )
    report = daily_report(state_dir, DAY)
    markdown = render_daily_markdown(report)
    tokens = set(re.split(r"[`|\s,]+", markdown))

    quoted = [
        report["ledger"][name][field]
        for name in ("fees", "slippage", "realised")
        for field in ("open", "close", "delta")
    ]
    quoted += [report["ledger"]["equity"][field] for field in ("open", "close", "worst")]
    quoted += [report["funding"][field] for field in ("paid", "received", "net")]
    for value in quoted:
        assert value is not None
        assert value in tokens, value

    # And the two lines the numbers actually appear on, written out by hand from
    # the records above: fees 7.75 - 1.50, funding -4.20 then +1.75.
    assert "| fees | 1.50 | 7.75 | 6.25 |" in markdown
    assert "Paid `4.20`, received `1.75`, net `-2.45`, over 2 settlement(s)." in markdown
    assert markdown.endswith("\n")
    assert not markdown.endswith("\n\n")


def test_the_daily_report_is_the_same_when_every_store_is_replaced(tmp_path):
    """The report is a function of the LOG, and of nothing else in the directory.

    ``carry_ledger.json`` and ``risk.json`` hold the CURRENT snapshot, so a
    report that consulted either would give a past day today's totals and would
    give a replayed log different answers from the run it replayed. Overwriting
    all five state files and re-running is what proves none of them was read;
    an AST guard would not, because reading a file is not a call this module
    could not otherwise make.
    """
    harness = _campaign(tmp_path)
    before = json.dumps(daily_report(harness.state_dir, DAY), indent=2, sort_keys=True)

    replaced = []
    for name in (
        "carry_ledger.json",
        "risk.json",
        "spot_store.json",
        "perp_store.json",
        "runner_state.json",
    ):
        path = harness.state_dir / name
        if path.is_file():
            path.write_text("{}", encoding="utf-8")
            replaced.append(name)
    assert len(replaced) >= 4, replaced

    after = json.dumps(daily_report(harness.state_dir, DAY), indent=2, sort_keys=True)
    assert after == before


def test_the_same_inputs_produce_a_byte_identical_report_twice(tmp_path):
    harness = _campaign(tmp_path)
    first = json.dumps(daily_report(harness.state_dir, DAY), indent=2, sort_keys=True)
    second = json.dumps(daily_report(harness.state_dir, DAY), indent=2, sort_keys=True)
    assert first == second
    assert render_daily_markdown(json.loads(first)) == render_daily_markdown(
        json.loads(second)
    )


def test_the_daily_report_reproduces_from_a_replayed_log(tmp_path):
    """Section 14 row 12's second acceptance criterion.

    ``tools.replay_parity.main`` is not driven end to end here because
    ``tools/demo_run.py``'s loader installs a ``RecordedQuoteFillModel`` with no
    quote, so a replay through the CLI fills nothing and diverges from any live
    run that did — a gap in the runner's wiring, not in the report. What matters
    for this criterion is the comparison itself, so the two runs go through the
    harness and section 10's own ``compare_logs`` establishes that the second
    reproduced the first before the two reports are compared.
    """
    live = _campaign(tmp_path / "live", minutes=8)
    replay = _campaign(tmp_path / "replay", minutes=8)

    parity = compare_logs(live.records(), replay.records())
    assert parity.ok, parity.to_dict()

    first = daily_report(live.state_dir, DAY)
    second = daily_report(replay.state_dir, DAY)
    for report in (first, second):
        # Both name their own directory and their own files; everything the
        # report SAYS about the day has to agree.
        report.pop("generated_from")
        report.pop("purity")
    assert first == second


# ---------------------------------------------------------------------------
# the monthly report, which refuses
# ---------------------------------------------------------------------------
#: An explicitly SYNTHETIC protocol identity. It names no protocol that exists,
#: no contract that exists and no revision that exists, and it is here only so
#: the success path of the interface can be exercised at all. Nothing computed
#: under it is evidence for anything.
SYNTHETIC_BINDING = ProtocolBinding(
    protocol_hash="sha256:" + "5" * 64,
    contract_hash="sha256:" + "6" * 64,
    prospective_from="2026-09-01T00:00:00+00:00",
    campaign_start="2026-09-08T00:00:00+00:00",
    source_identity={"revision": "SYNTHETIC", "source_digest": "SYNTHETIC", "dirty": "false"},
    quantities=("minutes_processed",),
    excluded_minutes=(),
    unclassified_treatment={
        "HALT": "operational",
        "RESUME": "operational",
        "RECOVERY": "operational",
    },
)

#: The thirteen research-question ids `artifacts/README.md` carries. Written out,
#: so a fourteenth row -- a PVC-1 row above all -- fails here.
THIRTEEN_CHECKPOINTS = (
    "v4",
    "P2a",
    "P2b",
    "P2c",
    "P3",
    "P4",
    "P5",
    "P6",
    "P6-EXT",
    "P7",
    "P8",
    "P13",
    "P14",
)


def _month_log(tmp_path: Path, extra=()) -> Path:
    state_dir = tmp_path / "state"
    _write(
        state_dir,
        [
            _decision("2026-09-19T00:00:00+00:00"),
            _decision("2026-09-19T00:01:00+00:00"),
            _decision("2026-09-19T00:02:00+00:00"),
            *extra,
        ],
    )
    return state_dir


def test_the_monthly_report_refuses_on_the_committed_campaign_config(tmp_path):
    """The repository's own state, read from the committed file."""
    from chimera.demo.config import ConfigProfile, load_demo_config

    config = load_demo_config(
        REPO / "conf" / "demo" / "pvc1.json", expected_profile=ConfigProfile.CAMPAIGN
    )
    assert config.protocol_hash is None
    assert config.protocol_frozen is False

    with pytest.raises(ReportRefused) as excinfo:
        monthly_report(_month_log(tmp_path), "2026-09", binding=None)
    message = str(excinfo.value)
    assert "protocol_hash=null" in message
    assert "Nothing was written." in message


def test_a_refused_monthly_freeze_writes_nothing(tmp_path, monkeypatch, capsys):
    from tools import demo_report

    monkeypatch.chdir(tmp_path)
    code = demo_report.main(
        [
            "--config",
            str(REPO / "conf" / "demo" / "pvc1.json"),
            "--month",
            "2026-10",
            "--freeze",
        ]
    )
    assert code == 2
    message = capsys.readouterr().err
    assert "protocol_hash=null" in message
    assert "Nothing was written." in message
    assert not (tmp_path / "artifacts").exists()
    assert list(tmp_path.iterdir()) == []


def test_the_cli_builds_no_protocol_binding_from_the_committed_config():
    """The refusal is reached through the ordinary path, not a special case."""
    from chimera.demo.config import ConfigProfile, load_demo_config
    from tools import demo_report

    config = load_demo_config(
        REPO / "conf" / "demo" / "pvc1.json", expected_profile=ConfigProfile.CAMPAIGN
    )
    assert demo_report._binding_from(config) is None


def _config_with_protocol_hash(tmp_path: Path, digest: str) -> Path:
    """The committed campaign config, with a protocol hash written into it.

    A file exactly like the one a well-meaning operator would produce after
    reading that `protocol_hash` is what says the protocol is frozen: the hash
    is filled in and nothing else is, because nothing else has anywhere to go
    in a campaign configuration.
    """
    payload = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text(encoding="utf-8"))
    payload["protocol_hash"] = digest
    path = tmp_path / "pvc1_with_hash.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_protocol_hash_alone_is_refused_rather_than_completed_from_defaults(tmp_path):
    """The other half of the gate, and the half that could fabricate evidence.

    `protocol_hash=null` refusing is the easy half: nothing has been claimed. The
    dangerous half is a hash that IS present, because a binding needs the
    contract hash, the prospective boundary, the campaign start, the source
    identity, the protocol-named quantities, the excluded minutes and a treatment
    for each unclassified kind -- none of which a campaign configuration holds.
    Filling those in from defaults would be this tool choosing the terms of the
    evidence and then stamping `evidence_class: prospective` on the result.

    Written because gutting the refusal -- returning a binding instead of
    raising -- passed every other test in this file.
    """
    from chimera.demo.config import ConfigProfile, load_demo_config
    from tools import demo_report

    path = _config_with_protocol_hash(tmp_path, "sha256:" + "a" * 64)
    config = load_demo_config(path, expected_profile=ConfigProfile.CAMPAIGN)
    assert config.protocol_hash == "sha256:" + "a" * 64
    assert config.protocol_frozen is True

    with pytest.raises(ReportRefused) as excinfo:
        demo_report._binding_from(config)
    message = str(excinfo.value)
    assert "Nothing was written." in message


def test_a_freeze_under_a_bare_protocol_hash_writes_nothing(tmp_path, monkeypatch, capsys):
    """End to end, through the CLI, with the filesystem as the witness."""
    from tools import demo_report

    path = _config_with_protocol_hash(tmp_path, "sha256:" + "a" * 64)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    code = demo_report.main(["--config", str(path), "--month", "2026-10", "--freeze"])

    assert code == 2
    assert "Nothing was written." in capsys.readouterr().err
    # The filesystem, not the exit code, is what says nothing was written: a
    # refusal that had already created the directory would still exit 2.
    assert list(workdir.iterdir()) == []
    assert not (workdir / "artifacts").exists()
    assert not (workdir / "artifacts" / "prospective").exists()


def test_a_synthetic_protocol_binding_reaches_the_prospective_payload(tmp_path):
    report = monthly_report(_month_log(tmp_path), "2026-09", binding=SYNTHETIC_BINDING)
    assert report["evidence_class"] == "prospective"
    assert report["quantities"] == {"minutes_processed": 3}
    assert report["protocol_hash"] == "sha256:" + "5" * 64
    assert report["days"] == ["2026-09-19"]
    assert report["month"] == "2026-09"
    status = render_monthly_status(report)
    assert "prospective" in status
    assert "answers no research question" in status


def _freeze_with(monkeypatch, tmp_path, binding, month="2026-09"):
    """Drive the real monthly write block under an explicitly SYNTHETIC protocol.

    The block is unreachable through the committed configuration, because the
    protocol is not frozen and the gate refuses -- which is exactly why nothing
    executed `mkdir`, either `write_text`, or `freeze` until this helper existed.
    The binding is patched in at the gate, so everything downstream of it is the
    shipped code running verbatim.
    """
    from tools import demo_report

    workdir = tmp_path / "cwd"
    (workdir / "state").mkdir(parents=True)
    state_dir = _month_log(workdir, ())
    payload = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text(encoding="utf-8"))
    payload.setdefault("runner", {})["state_dir"] = str(state_dir)
    config_path = workdir / "synthetic.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    # `freeze_evidence.ROOT` is derived from its own `__file__`, so every path
    # under tmp_path is "outside the repository" to it. Pointing it at the
    # synthetic tree is what lets the write block run at all without putting a
    # file carrying `evidence_class: prospective` into the real artifacts/.
    from tools import freeze_evidence

    monkeypatch.setattr(freeze_evidence, "ROOT", workdir.resolve())
    monkeypatch.setattr(demo_report, "_binding_from", lambda config: binding)
    monkeypatch.chdir(workdir)
    code = demo_report.main(["--config", str(config_path), "--month", month, "--freeze"])
    return demo_report, workdir, config_path, code


def test_a_synthetic_freeze_writes_the_artifact_and_its_manifest(tmp_path, monkeypatch):
    """The write block, executed. Nothing had ever run it."""
    _, workdir, _, code = _freeze_with(monkeypatch, tmp_path, SYNTHETIC_BINDING)
    assert code == 0
    out_dir = workdir / "artifacts" / "prospective" / "pvc1" / "2026-09"
    assert (out_dir / "report.json").is_file()
    assert (out_dir / "STATUS.md").is_file()
    assert (workdir / "artifacts" / "pvc1_2026-09_SHA256SUMS.txt").is_file()
    written = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert written["evidence_class"] == "prospective"


def test_refreezing_a_frozen_month_refuses_before_it_touches_the_frozen_bytes(
    tmp_path, monkeypatch
):
    """The ordering defect, and the reason the refusal moved ahead of the writes.

    `freeze` refuses to overwrite a manifest, but it was called AFTER both
    `write_text` calls: a second run replaced the frozen report.json with today's
    bytes and only then raised, leaving a committed manifest naming a digest no
    file matches. A frozen result had quietly become whatever the code does
    today, which is the one thing that refusal exists to prevent.
    """
    from dataclasses import replace

    demo_report, workdir, config_path, code = _freeze_with(
        monkeypatch, tmp_path, SYNTHETIC_BINDING
    )
    assert code == 0
    frozen = workdir / "artifacts" / "prospective" / "pvc1" / "2026-09" / "report.json"
    manifest = workdir / "artifacts" / "pvc1_2026-09_SHA256SUMS.txt"
    before = frozen.read_bytes()
    manifest_before = manifest.read_bytes()

    # A second run whose report would differ, so an overwrite is detectable.
    other = replace(SYNTHETIC_BINDING, quantities=("minutes_processed", "decisions"))
    monkeypatch.setattr(demo_report, "_binding_from", lambda config: other)
    with pytest.raises(SystemExit) as excinfo:
        demo_report.main(["--config", str(config_path), "--month", "2026-09", "--freeze"])

    assert "already exists" in str(excinfo.value)
    assert frozen.read_bytes() == before, "the frozen report was overwritten"
    assert manifest.read_bytes() == manifest_before

    from tools.freeze_evidence import check

    monkeypatch.chdir(workdir)
    assert check(manifest) == [], "the manifest no longer describes the bytes"


def test_a_freeze_from_outside_the_repository_refuses_before_writing(tmp_path, monkeypatch):
    """`relative()` refuses a path outside the repository -- but only once
    `freeze` reached it, which was after both writes, so an artifact carrying
    `evidence_class: prospective` was already on disk with no manifest at all."""
    from tools import demo_report

    outside = tmp_path / "outside"
    (outside / "state").mkdir(parents=True)
    state_dir = _month_log(outside, ())
    payload = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text(encoding="utf-8"))
    payload.setdefault("runner", {})["state_dir"] = str(state_dir)
    config_path = outside / "synthetic.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(demo_report, "_binding_from", lambda config: SYNTHETIC_BINDING)
    monkeypatch.chdir(outside)
    with pytest.raises(SystemExit) as excinfo:
        demo_report.main(["--config", str(config_path), "--month", "2026-09", "--freeze"])

    assert "outside the repository" in str(excinfo.value)
    assert not (outside / "artifacts").exists(), "an unmanifested artifact was left behind"


def test_the_monthly_report_computes_only_the_quantities_the_protocol_names(tmp_path):
    from dataclasses import replace

    binding = replace(SYNTHETIC_BINDING, quantities=("decisions", "minutes_processed"))
    report = monthly_report(_month_log(tmp_path), "2026-09", binding=binding)
    assert sorted(report["quantities"]) == ["decisions", "minutes_processed"]
    assert "records" not in report["quantities"]
    assert "incomplete_minutes" not in report["quantities"]


def test_a_quantity_the_protocol_does_not_name_is_refused(tmp_path):
    from dataclasses import replace

    binding = replace(SYNTHETIC_BINDING, quantities=("sharpe",))
    with pytest.raises(ReportRefused, match="sharpe"):
        monthly_report(_month_log(tmp_path), "2026-09", binding=binding)


def test_an_unclassified_kind_without_a_treatment_is_refused(tmp_path):
    from dataclasses import replace

    halt = {
        **_base("HALT", "2026-09-19T00:03:00+00:00"),
        "veto_or_rejection": {"stage": "runner", "label": "halt", "detail": "kill_switch"},
    }
    state_dir = _month_log(tmp_path, extra=[halt])
    binding = replace(SYNTHETIC_BINDING, unclassified_treatment={})
    with pytest.raises(ReportRefused, match="HALT"):
        monthly_report(state_dir, "2026-09", binding=binding)
    # The two-sided control: with all three named, the same month computes.
    assert monthly_report(state_dir, "2026-09", binding=SYNTHETIC_BINDING)["quantities"]


def test_a_partial_treatment_is_refused_as_firmly_as_none(tmp_path):
    from dataclasses import replace

    halt = {
        **_base("HALT", "2026-09-19T00:03:00+00:00"),
        "veto_or_rejection": {"stage": "runner", "label": "halt", "detail": "kill_switch"},
    }
    binding = replace(SYNTHETIC_BINDING, unclassified_treatment={"HALT": "operational"})
    with pytest.raises(ReportRefused, match="HALT"):
        monthly_report(_month_log(tmp_path, extra=[halt]), "2026-09", binding=binding)


def test_a_blank_or_non_string_treatment_is_a_silence_and_is_refused(tmp_path):
    """The gate read `sorted(mapping)`, which sorts KEYS and never looks at values.

    A binding naming all three kinds with `""`, `"  "`, `None`, `17` or `["x"]`
    therefore passed, and the value went verbatim into the frozen payload -- the
    silence this gate exists to refuse, wearing the shape of an answer. What a
    legal treatment IS stays deliberately unnamed: `"banana"` still passes,
    because fixing the vocabulary is the protocol's job and not this report's.
    """
    from dataclasses import replace

    halt = {
        **_base("HALT", "2026-09-19T00:03:00+00:00"),
        "veto_or_rejection": {"stage": "runner", "label": "halt", "detail": "kill_switch"},
    }
    state_dir = _month_log(tmp_path, extra=[halt])

    for treatment in (
        {"HALT": "", "RESUME": "operational", "RECOVERY": "operational"},
        {"HALT": "  ", "RESUME": "operational", "RECOVERY": "operational"},
        {"HALT": None, "RESUME": 17, "RECOVERY": ["x"]},
    ):
        binding = replace(SYNTHETIC_BINDING, unclassified_treatment=treatment)
        with pytest.raises(ReportRefused, match="HALT"):
            monthly_report(state_dir, "2026-09", binding=binding)
    # The two-sided control: the same month, with three real treatments, computes.
    assert monthly_report(state_dir, "2026-09", binding=SYNTHETIC_BINDING)["quantities"]


def test_a_treatment_this_report_does_not_recognise_still_passes(tmp_path):
    """The other side, and it is deliberate rather than an oversight.

    Naming the legal values would be PR-12 deciding what HALT, RESUME and
    RECOVERY mean for scoring, which is exactly the decision the module refuses
    to make and the S2 protocol exists to freeze.
    """
    from dataclasses import replace

    binding = replace(
        SYNTHETIC_BINDING,
        unclassified_treatment={k: "banana" for k in ("HALT", "RESUME", "RECOVERY")},
    )
    halt = {
        **_base("HALT", "2026-09-19T00:03:00+00:00"),
        "veto_or_rejection": {"stage": "runner", "label": "halt", "detail": "kill_switch"},
    }
    report = monthly_report(_month_log(tmp_path, extra=[halt]), "2026-09", binding=binding)
    assert report["unclassified_treatment"] == {
        "HALT": "banana",
        "RESUME": "banana",
        "RECOVERY": "banana",
    }


def test_an_excluded_minute_in_a_spelling_the_log_never_writes_is_refused(tmp_path):
    """An exclusion that silently applies to nothing is worse than a wrong number.

    The filter is plain set membership on raw strings. `decision_log` refuses a
    `Z` suffix, a naive stamp and a local offset, and writes `...+00:00` only, so
    a protocol excluding `2026-09-19T00:01:00Z` would match no record, remove no
    minute, and still be copied into the frozen artifact as an exclusion that had
    been applied -- reporting MORE minutes than the protocol authorises under a
    record saying otherwise.
    """
    from dataclasses import replace

    state_dir = _month_log(tmp_path)
    for spelling in (
        "2026-09-19T00:01:00Z",
        "2026-09-19 00:01:00+00:00",
        "2026-09-19T00:01:00",
        "2026-09-19T00:01:30+00:00",
        "not-a-minute",
    ):
        binding = replace(SYNTHETIC_BINDING, excluded_minutes=(spelling,))
        with pytest.raises(ReportRefused, match="spelling"):
            monthly_report(state_dir, "2026-09", binding=binding)

    # The two-sided control: the canonical spelling is accepted AND applied.
    canonical = replace(
        SYNTHETIC_BINDING,
        excluded_minutes=("2026-09-19T00:01:00+00:00",),
        quantities=("minutes_processed",),
    )
    report = monthly_report(state_dir, "2026-09", binding=canonical)
    assert report["quantities"]["minutes_processed"] == 2, "the exclusion did not apply"


def test_a_null_prospective_boundary_is_refused(tmp_path):
    from dataclasses import replace

    binding = replace(SYNTHETIC_BINDING, prospective_from=None)
    with pytest.raises(ReportRefused, match="prospective_from"):
        monthly_report(_month_log(tmp_path), "2026-09", binding=binding)


def test_a_campaign_start_before_the_boundary_is_refused(tmp_path):
    from dataclasses import replace

    binding = replace(SYNTHETIC_BINDING, campaign_start="2026-08-30T00:00:00+00:00")
    with pytest.raises(ReportRefused, match="precedes the prospective boundary"):
        monthly_report(_month_log(tmp_path), "2026-09", binding=binding)


def test_a_malformed_protocol_hash_is_refused(tmp_path):
    from dataclasses import replace

    binding = replace(SYNTHETIC_BINDING, protocol_hash="not-a-hash")
    with pytest.raises(ReportRefused, match="not sha256"):
        monthly_report(_month_log(tmp_path), "2026-09", binding=binding)


def test_a_month_whose_log_does_not_verify_is_refused(tmp_path):
    state_dir = _month_log(tmp_path)
    path = sorted((state_dir / "decision_log").glob("*.ndjson"))[0]
    with open(path, "ab") as handle:
        handle.write(b"not a record")
    with pytest.raises(ReportRefused, match="does not verify"):
        monthly_report(state_dir, "2026-09", binding=SYNTHETIC_BINDING)


def test_an_empty_month_is_refused_rather_than_reported_as_zeros(tmp_path):
    with pytest.raises(ReportRefused, match="holds no record"):
        monthly_report(_month_log(tmp_path), "2026-08", binding=SYNTHETIC_BINDING)


# ---------------------------------------------------------------------------
# nothing prospective is committed, and no research question is claimed
# ---------------------------------------------------------------------------
def test_no_prospective_artifact_is_committed():
    prospective = REPO / "artifacts" / "prospective"
    assert not prospective.exists() or not any(p.is_file() for p in prospective.rglob("*"))


def test_the_research_state_block_has_no_pvc1_row():
    text = (REPO / "artifacts" / "README.md").read_text(encoding="utf-8")
    block = text.split("<!-- research-state:begin -->")[1].split(
        "<!-- research-state:end -->"
    )[0]
    assert "PVC-1" not in block
    assert "prospective" not in block
    ids = [
        line.split("`")[1]
        for line in block.splitlines()
        if line.startswith("| `") and "`" in line
    ]
    assert tuple(ids) == THIRTEEN_CHECKPOINTS


# ---------------------------------------------------------------------------
# PR-10R follow-up: the disclosures have to describe the report that exists
# ---------------------------------------------------------------------------
#: Every file that tells an operator where to look when the funding panel reads
#: a flat zero. Each is a place a claim about the daily report can go stale.
FUNDING_DISCLOSURE_SOURCES = (
    "docs/demo_runbook.md",
    "conf/alerts_demo.yml",
    "grafana/provisioning/dashboards/demo.json",
)


def test_no_disclosure_claims_the_daily_report_carries_per_settlement_detail():
    """The report's funding block holds totals and counts. Three files said otherwise.

    The runbook, the alert file and the dashboard panel all told an operator that
    "the daily report's funding block carries one record per settlement with its
    rate, mark, notional and signed cash flow", and that reading it distinguishes
    "no settlement has fallen inside this position's window yet" from "the
    position was flat across every settlement". It carries neither: `net`,
    `paid`, `received`, `settlement_minutes`, `sign_convention`, `source` and
    `records_of_kind_FUNDING`, all of which are zero in both of those cases.

    An operator sent to evidence that does not exist is worse off than one told
    plainly that it does not: PR-10R exists because stale disclosures outlived
    the code they described, so this is the executable version of the fix.
    """
    for relative in FUNDING_DISCLOSURE_SOURCES:
        text = (REPO / relative).read_text(encoding="utf-8")
        for claim in (
            "funding block, which carries one record per settlement",
            "funding block carries one record per settlement",
        ):
            assert claim not in text, (
                f"{relative} claims the daily report's funding block holds one record "
                f"per settlement; it holds exactly {FUNDING_FIELDS}, which "
                "test_the_daily_report_has_exactly_the_adopted_shape pins"
            )
