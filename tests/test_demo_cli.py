"""`tools/demo_run.py`: section 8.3's six subcommands and the mandatory notes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from chimera.demo.config import config_hash
from chimera.demo.runner import RunnerState
from tests.demo_harness import CARRY_PARAMS, DAY, build, campaign_config
from tools import demo_run

SIX = ("run", "status", "flatten", "resume", "resolve", "report")


def written_config(tmp_path: Path, harness) -> Path:
    """The harness's own config, on disk, so the CLI parses what the tests ran."""
    config = harness.runner.config
    payload = json.loads(
        (Path(__file__).parents[1] / "conf/demo/pvc1.json").read_text("utf-8")
    )
    payload.update(
        {
            "profile": config.profile.value,
            "runner": {"state_dir": str(harness.state_dir)},
            "rules": {"R1_carry": dict(CARRY_PARAMS)},
        }
    )
    path = tmp_path / "cli_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exactly_the_six_subcommands_the_plan_names():
    parser = demo_run.build_parser()
    actions = [
        a for a in parser._actions if getattr(a, "choices", None) and a.dest == "command"
    ]
    assert actions, "the parser must declare subcommands"
    assert tuple(sorted(actions[0].choices)) == tuple(sorted(SIX))


@pytest.mark.parametrize("command", ["flatten", "resume", "resolve"])
def test_the_note_is_required_by_the_parser(command):
    """argparse refuses the command outright, before anything is touched."""
    argv = ["--config", "x", "--root", "y", command]
    if command == "resolve":
        argv += ["--symbol", "BTC/USDT"]
    with pytest.raises(SystemExit):
        demo_run.build_parser().parse_args(argv)


@pytest.mark.parametrize("note", ["", "   ", "\t\n"])
def test_a_whitespace_only_note_is_refused(note):
    """Required is not the same as meaningful; a blank note explains nothing."""
    with pytest.raises(SystemExit, match="non-empty"):
        demo_run._require_note(note, "flatten")


def test_a_real_note_survives_stripped():
    assert demo_run._require_note("  checked both legs  ", "flatten") == "checked both legs"


RUNBOOK = Path(__file__).resolve().parents[1] / "docs" / "demo_runbook.md"

_FENCED_BLOCK = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
_BACKTICKED_SPAN = re.compile(r"`([^`\n]+)`")
#: ``ledger_x: value`` -- how the runbook quotes what `status` prints.
_FIELD_IN_OUTPUT_POSITION = re.compile(r"\b(ledger_[a-z_]+)\s*:")
#: The prose form: "it reports `ledger_may_speak` and `ledger_complaint`". The
#: colon form alone misses it, and it is one of the two lines round 7 had to
#: rename -- drift confined to it would have stayed green. Harvested ONLY from
#: the PARAGRAPH that introduces the fields: the runbook backticks plenty of
#: other `ledger_*` names (halt reasons, dispute ids, record keys) that `status`
#: neither emits nor should, and every one of them lives in a paragraph that
#: does not introduce anything.
_INTRODUCES_THE_FIELDS = "it reports"
_FIELD_AS_BARE_NAME = re.compile(r"\A(ledger_[a-z_]+)\Z")
#: Paragraphs are split on blank lines, which is how this document separates
#: them throughout. A paragraph that swallowed a heading would over-harvest and
#: fail RED here, which is the direction that gets looked at. The first
#: version of this scanned LINES, which made the check a hostage to where the
#: author happened to wrap: the identical rename, with `ledger_complaint` pushed
#: onto the following line by an ordinary re-flow, stopped being harvested at all
#: and the contract went green on the drift it exists to catch. A paragraph is
#: the unit a Markdown author edits; a line is an artefact of filling it.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


def _paragraphs(text: str) -> list[str]:
    """The document's paragraphs, each unwrapped onto one logical line.

    Fenced blocks go first: they carry their own backticks, so every inline span
    after the first fence otherwise pairs off by one and the extractor matches
    nothing at all -- silently, which is how this check was once vacuous.
    """
    prose = _FENCED_BLOCK.sub("", text)
    return [" ".join(block.split()) for block in _PARAGRAPH_BREAK.split(prose)]


def _status_fields_named_by(text: str) -> set[str]:
    """Every `ledger_*` field the text shows as `status` output.

    Scans each inline backticked span separately: one span can quote several
    fields (`ledger_may_speak: true, ledger_complaint: null`), and a single
    pattern over the whole string finds only the first of them. Fenced code
    blocks are stripped first -- their backticks otherwise pair with the inline
    ones and every span after the first fence is read off by one.
    """
    named: set[str] = set()
    for paragraph in _paragraphs(text):
        introduces = _INTRODUCES_THE_FIELDS in paragraph
        for span in _BACKTICKED_SPAN.findall(paragraph):
            named.update(_FIELD_IN_OUTPUT_POSITION.findall(span))
            if introduces:
                named.update(_FIELD_AS_BARE_NAME.findall(span.strip()))
    return named


def test_status_reports_the_runner_without_changing_it(tmp_path, capsys):
    harness = build(tmp_path)
    harness.run(2)
    before = harness.runner.cursor.last_minute_processed
    payload = demo_run._status(harness.runner)
    assert payload["hedge_state"] == harness.runner.position.state.value
    assert payload["protocol_frozen"] is False
    assert payload["imbalance"] == "0.000"
    assert harness.runner.cursor.last_minute_processed == before


def _persisted_bytes(root: Path) -> dict[str, bytes]:
    """Every file below the runner state directory, keyed without host paths."""
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_status_production_load_is_strictly_read_only(tmp_path):
    """No active runtime, clock seed, directory, or persisted write for status."""
    import argparse

    from chimera.demo.runner import RunnerError
    from chimera.demo.telemetry import NullTelemetry

    harness = build(tmp_path, start=False, telemetry=NullTelemetry())
    config_path = written_config(tmp_path, harness)
    assert not harness.state_dir.exists()

    runner = demo_run._load(
        argparse.Namespace(
            config=config_path,
            root=harness.root,
            profile="TEST",
            command="status",
        )
    )
    payload = demo_run._status(runner)

    assert payload["runner_state"] == RunnerState.STARTUP.value
    assert payload["ledger_may_speak"] is True
    assert not runner.clock.started
    assert not harness.state_dir.exists(), "status created runner state"
    for attribute in ("risk", "position"):
        with pytest.raises(RunnerError, match="before the clock starts"):
            getattr(runner, attribute)


def test_status_does_not_change_any_existing_persisted_byte(tmp_path):
    """The production status path reads a complete prior run without mutation."""
    import argparse

    harness = build(tmp_path)
    harness.run(2)
    harness.runner.shutdown("status inspection fixture")
    config_path = written_config(tmp_path, harness)
    before = _persisted_bytes(harness.state_dir)

    runner = demo_run._load(
        argparse.Namespace(
            config=config_path,
            root=harness.root,
            profile="TEST",
            command="status",
        )
    )
    payload = demo_run._status(runner)

    assert payload["last_minute_processed"] is not None
    assert payload["ledger_may_speak"] is True
    assert not runner.clock.started
    assert _persisted_bytes(harness.state_dir) == before


def test_status_names_the_ledger_verdict_fields_the_runbook_names(tmp_path):
    """The keys are an operator-facing contract, and the runbook quotes them.

    A round-7 reviewer found the runbook telling operators to look for
    `ledger_regression` in `status` output, in the same commit that renamed the
    key to `ledger_complaint` -- so the documented string appeared nowhere and
    nothing in CI could notice. `tests/test_demo_runbook.py` checks the commands
    section 6 names, not the fields it quotes; this closes that gap.
    """
    harness = build(tmp_path)
    harness.run(2)
    payload = demo_run._status(harness.runner)

    # A healthy campaign: the ledger may speak and has nothing to complain of.
    assert payload["ledger_may_speak"] is True
    assert payload["ledger_complaint"] is None

    # And a ledger that may not speak says so, without start() having been called.
    harness.runner.shutdown("stop")
    harness.runner.position.ledger.path.unlink()
    refused = build(tmp_path, config=harness.runner.config, start=False).runner
    payload = demo_run._status(refused)
    assert refused.state is RunnerState.STARTUP, "status must not need start()"
    assert payload["ledger_may_speak"] is False
    assert "ledger_behind_log" in payload["ledger_complaint"]


def test_status_asks_the_whole_ledger_predicate_not_half_of_it(tmp_path):
    """`status` must ask `_ledger_may_speak()`, not `_ledger_regression is None`.

    This is the defect the branch already shipped and fixed in `resume`: an
    unreadable ledger on a campaign that never traded has NO regression to find,
    so half the predicate calls it healthy. `status` had the same hole and no
    test saw it -- reducing `_status` to the regression term alone left the whole
    file green while `status` printed the documented all-clear
    (`ledger_may_speak: true, ledger_complaint: null`) for a ledger that may not
    speak.

    The other status tests cannot catch it: every ledger they build is behind the
    log as well as unreadable, so both halves agree by accident.
    """
    config = campaign_config(
        tmp_path / "state",
        rules={"R1_carry": {**CARRY_PARAMS, "min_basis": "1000000"}},
    )
    harness = build(tmp_path, config=config, with_shadow=False)
    harness.run(3)
    assert harness.runner.position.leg("spot").is_flat, "the fixture was supposed to hold off"
    harness.runner.shutdown("stop")
    harness.runner.position.ledger.path.write_text("{ this is not json", encoding="utf-8")

    runner = build(tmp_path, config=config, with_shadow=False, start=False).runner
    # Nothing was ever committed, so the regression term alone finds nothing...
    assert runner._ledger_regression is None
    # ...but the ledger is unreadable, so it may not speak, and status must say so.
    payload = demo_run._status(runner)
    assert payload["ledger_may_speak"] is False
    assert "UNREADABLE" in payload["ledger_complaint"]


def test_the_runbook_only_names_status_fields_that_status_emits(tmp_path):
    """The runbook/CLI agreement itself, not one dead field name.

    Round 7 found section 6 telling operators to look for `ledger_regression` in
    `status` output, in the same commit that renamed the key to
    `ledger_complaint`. The first test written for it asserted what `_status`
    emits and that one historical name was absent -- which pins the
    IMPLEMENTATION plus a tombstone. Rename the field in the CLI and in that
    test while forgetting the runbook and the same defect returns green.

    This reads the runbook instead. Every `ledger_*` name the document shows in
    an output position -- ``name: value`` inside backticks, which is how section
    6 quotes what an operator will see -- must be a key `_status` actually
    emits. No allow-list to rot, and it fails on the historical text (see
    `test_the_runbook_contract_catches_the_defect_it_was_written_for`).
    """
    harness = build(tmp_path)
    harness.run(2)
    emitted = set(demo_run._status(harness.runner))

    text = RUNBOOK.read_text(encoding="utf-8")
    named = _status_fields_named_by(text)
    assert named, "the runbook must keep quoting status output, or this test guards nothing"
    assert any(_INTRODUCES_THE_FIELDS in p for p in _paragraphs(text)), (
        "the runbook no longer introduces the verdict fields in prose, so the bare-name "
        "half of this check is guarding nothing -- the colon form alone is what round 8 "
        "found insufficient"
    )
    assert named <= emitted, (
        f"the runbook shows status fields that status does not emit: {sorted(named - emitted)}; "
        f"status emits {sorted(emitted)}"
    )


def test_the_runbook_contract_catches_the_defect_it_was_written_for(tmp_path):
    """A control: the check above must FAIL on the text that actually shipped.

    Without this, the runbook check could be vacuous -- passing because it
    matches nothing rather than because the document is right.
    """
    harness = build(tmp_path)
    harness.run(2)
    emitted = set(demo_run._status(harness.runner))

    # The verbatim shape of the round-7 defect: a field named in an output
    # position that `_status` has never emitted.
    shipped = "it says `ledger_may_speak: true, ledger_regression: null` while `run` repeats"
    named = _status_fields_named_by(shipped)

    assert "ledger_regression" in named, "the extractor must see the name that shipped"
    assert not (named <= emitted), "the contract check must reject the text that shipped"


def test_the_contract_survives_the_introducing_prose_being_re_wrapped(tmp_path):
    """A control: ordinary Markdown re-wrapping must not defeat the check.

    Round 8 added the bare-name harvest because the colon form alone misses the
    one sentence that tells an operator which two fields exist -- and then
    harvested it per LINE. So the identical drift, with the second name pushed
    onto the following line by a re-flow that changed no words, went green: the
    guard was defeated by where the author pressed return. Both spellings of the
    same drift are pinned here, so neither can regress alone.
    """
    harness = build(tmp_path)
    harness.run(2)
    emitted = set(demo_run._status(harness.runner))

    one_line = (
        "the command to trust here is\n"
        "`status`**: it reports `ledger_may_speak` and `ledger_verdict` computed from\n"
        "the ledger as loaded, it needs no `start()`.\n"
    )
    wrapped = (
        "the command to trust here is\n"
        "`status`**: it reports `ledger_may_speak` and\n"
        "`ledger_verdict` computed from\n"
        "the ledger as loaded, it needs no `start()`.\n"
    )
    for label, drift in (("unwrapped", one_line), ("wrapped", wrapped)):
        named = _status_fields_named_by(drift)
        assert "ledger_verdict" in named, f"the {label} drift was not even read"
        assert not (named <= emitted), f"the {label} drift must fail the contract"


def test_the_contract_does_not_demand_status_emit_halt_reasons(tmp_path):
    """The extractor stays narrow, and that narrowness is pinned rather than trusted.

    The runbook backticks plenty of `ledger_*` names that are halt reasons,
    dispute ids or record keys -- `status` neither emits them nor should.
    Harvesting bare names document-wide was tried and reverted for exactly that,
    so the reverted direction is a test now: each of these is in the document,
    none is harvested, and none is a `status` key.
    """
    harness = build(tmp_path)
    harness.run(2)
    emitted = set(demo_run._status(harness.runner))

    text = RUNBOOK.read_text(encoding="utf-8")
    named = _status_fields_named_by(text)
    for name in (
        "ledger_behind_log",
        "ledger_effect",
        "ledger_store_mismatch",
        "ledger_unreadable",
        "ledger_capital_mismatch",
    ):
        assert f"`{name}" in text, f"{name} left the runbook; this control now proves nothing"
        assert name not in named, f"{name} is a halt reason, not a field `status` shows"
        assert name not in emitted


def test_report_counts_a_days_records(tmp_path, capsys):
    harness = build(tmp_path)
    harness.run(3)
    harness.runner.shutdown("done")
    config_path = written_config(tmp_path, harness)

    code = demo_run.main(
        ["--config", str(config_path), "--profile", "TEST", "report", "--day", DAY]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["day"] == DAY
    assert payload["records"] >= 5
    assert payload["kinds"]["DECISION"] == 3


def test_report_on_a_day_with_no_log_says_so_rather_than_failing(tmp_path, capsys):
    harness = build(tmp_path)
    config_path = written_config(tmp_path, harness)
    assert (
        demo_run.main(
            [
                "--config",
                str(config_path),
                "--profile",
                "TEST",
                "report",
                "--day",
                "2019-01-01",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["records"] == 0


def test_resolve_refuses_a_symbol_that_is_not_a_leg(tmp_path):
    from chimera.demo.runner import RunnerError

    harness = build(tmp_path)
    harness.run(1)
    with pytest.raises(RunnerError, match="not one of this position's legs"):
        harness.runner.resolve("DOGE/USDT", "a note")


def test_the_cli_opens_no_socket_and_reads_no_credential():
    """Checked over the AST, not the text.

    The module's own docstring discusses sockets and credentials, and a guard
    that could not tell prose from code would forbid saying so -- the same trap
    as `tests/test_demo_no_live_path.py`'s shadow-rule check. The one socket the
    CLI may open is the Prometheus scrape endpoint `run --metrics-port` asks
    for, which `prometheus_client` opens and which none of the modules named
    below is imported to reach.
    """
    import ast

    tree = ast.parse((Path(__file__).parents[1] / "tools" / "demo_run.py").read_text("utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"socket", "ssl", "requests", "urllib", "ccxt", "http"}

    # A credential read is a call or an attribute access, never a docstring.
    reads = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert "environ" not in reads and "getenv" not in reads
    constants = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    code_strings = [c for c in constants if "\n" not in c]  # docstrings are multi-line
    for marker in ("API_KEY", "API_SECRET"):
        assert not [c for c in code_strings if marker in c]


def test_the_config_hash_does_not_move_with_the_state_directory(tmp_path):
    """A path may not enter a campaign's identity; two hosts must hash alike."""
    a = campaign_config(tmp_path / "host_a")
    b = campaign_config(tmp_path / "host_b")
    assert config_hash(a) == config_hash(b)


def test_the_config_hash_does_move_with_a_rule_parameter(tmp_path):
    """The negative control: identity ignores paths, never parameters."""
    a = campaign_config(tmp_path / "s")
    b = campaign_config(
        tmp_path / "s", rules={"R1_carry": {**CARRY_PARAMS, "min_basis": "999"}}
    )
    assert config_hash(a) != config_hash(b)
