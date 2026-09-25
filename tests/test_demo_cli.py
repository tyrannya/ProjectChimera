"""`tools/demo_run.py`: section 8.3's six subcommands and the mandatory notes."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from chimera.demo.config import config_hash
from chimera.demo.runner import RunnerState
from nn.source_identity import SOURCE_ROOTS
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
            # The harness's limits, not the committed file's: they are equal
            # unless a test changed one on purpose (R1-f's service tests hold
            # `max_data_delay_s` out of reach), and then the CLI must run it too.
            "limits": config.limits.to_dict(),
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


# --- R1-a: the software block, against a real checkout -----------------------
#
# `_software` called `source_identity()` with no argument -- it REQUIRES a root
# -- so every call raised `TypeError`, fell into the fail-closed branch, and
# reported `dirty: True`. No CAMPAIGN could start. The second half is why this
# section exists at all: the identity is a MAPPING and was read with `getattr`,
# so repairing only the call signature would have yielded `revision: ""` and
# `dirty: False` and let a campaign run on a tree nobody could reconstruct.
#
# The witnesses below run against REAL git checkouts built in a temporary
# directory, not against a stubbed identity: a test that stubs `source_identity`
# passes happily while the real call signature is still wrong, which is exactly
# how this survived.
#: A git object id, as `git rev-parse HEAD` prints it: 40 lowercase hex under
#: the SHA-1 object format and 64 under SHA-256. Git supports both through
#: `git init --object-format`, and `source_identity` passes `rev-parse HEAD`
#: through verbatim without assuming a length, so a witness that required
#: SHA-1 would fail on a SHA-256 checkout the runner had in fact identified
#: correctly. Both lengths exactly -- not "hex of any length", which would
#: accept a truncated or abbreviated id that names no object.
GIT_OBJECT_ID = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")

#: `source_digest` is NOT a git object id: it is `hashlib.sha256` over the
#: source material, computed by `nn.source_identity` itself, so it is 64 hex
#: whatever object format the checkout uses. It stays pinned to exactly that.
HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A real git checkout shaped from `SOURCE_ROOTS`, not a stub.

    Built here rather than imported from `tests/test_source_identity.py`, whose
    module imports pull the whole ML stack in behind `nn.p2b_compare`; this
    module is the demo CLI's and needs none of it. The roots come from
    `nn.source_identity` so the fixture follows what the digest actually covers.
    """
    return build_checkout(tmp_path / "checkout")


def build_checkout(root: Path, *, object_format: str | None = None) -> Path:
    """Build the fixture's checkout, optionally under a named object format.

    ``object_format`` is the one knob, and it exists because the revision the
    runner records is whatever `git rev-parse HEAD` prints, which is 40 hex
    under SHA-1 and 64 under SHA-256.
    """
    for name in SOURCE_ROOTS:
        (root / name).mkdir(parents=True)
        (root / name / "__init__.py").write_text(f'"""{name}"""\n', encoding="utf-8")
    (root / "nn" / "engine.py").write_text("VALUE = 1\n", encoding="utf-8")
    init = ["git", "init", "-q"]
    if object_format is not None:
        init += ["--object-format", object_format]
    subprocess.run(init + [str(root)], check=True, capture_output=True)
    run = ["git", "-C", str(root)]
    for args in (
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "test"],
        ["add", "-A"],
        ["commit", "-qm", "initial"],
    ):
        subprocess.run(run + args, check=True, capture_output=True)
    return root


def test_a_genuinely_clean_checkout_is_identified_and_reported_clean(checkout):
    """The positive half: a real clean checkout names itself and is not dirty."""
    block = demo_run._software(checkout)
    assert block["dirty"] is False
    assert GIT_OBJECT_ID.match(block["revision"]), block["revision"]
    assert HEX64.match(block["source_digest"]), block["source_digest"]


@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
def test_a_checkout_is_identified_under_either_git_object_format(
    tmp_path: Path, object_format: str
):
    """The revision is whatever git prints, and git prints two lengths.

    Requiring SHA-1 failed a SHA-256 checkout that `_software` had identified
    perfectly well -- the assertion was pinning git's object format rather than
    anything the runner does. Both formats are real: `git init --object-format`
    supports each, and a host may default to either.
    """
    root = build_checkout(tmp_path / object_format, object_format=object_format)
    block = demo_run._software(root)

    assert block["dirty"] is False
    assert GIT_OBJECT_ID.match(block["revision"]), block["revision"]
    # The digest is the repository's own sha256 over the source, not a git
    # object id, so it is 64 hex under both formats.
    assert HEX64.match(block["source_digest"]), block["source_digest"]

    expected = 40 if object_format == "sha1" else 64
    assert len(block["revision"]) == expected, block["revision"]

    # And the negative half still holds under this format.
    (root / "nn" / "engine.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert demo_run._software(root)["dirty"] is True


@pytest.mark.parametrize(
    "revision",
    [
        "",
        "a" * 39,
        "a" * 41,
        "a" * 63,
        "a" * 65,
        "A" * 40,
        "g" * 40,
        "a" * 39 + "z",
        "HEAD",
        "sha256:" + "a" * 40,
        " " + "a" * 40,
    ],
)
def test_a_malformed_revision_is_not_a_git_object_id(revision: str):
    """The widening admits two lengths, not arbitrary hex.

    The fix for the SHA-256 case must not become "any hex string": an
    abbreviated or truncated id names no object, and neither does an uppercase
    or prefixed one. This is the control that keeps the matcher honest.
    """
    assert GIT_OBJECT_ID.match(revision) is None, revision


def test_a_genuinely_dirty_checkout_is_reported_dirty(checkout):
    """The negative half, and the `getattr` witness.

    `getattr(mapping, "dirty", False)` is False for every mapping, so this is
    the assertion that fails if the identity is ever read as an attribute again.
    """
    clean = demo_run._software(checkout)
    assert clean["dirty"] is False

    (checkout / "nn" / "engine.py").write_text("VALUE = 2\n", encoding="utf-8")

    dirty = demo_run._software(checkout)
    assert dirty["dirty"] is True
    # Still identified: a dirty tree names the revision it departed from.
    assert GIT_OBJECT_ID.match(dirty["revision"])


def test_an_untracked_python_file_under_a_source_root_is_dirty(checkout):
    """Modification is not the only way to become unreconstructible."""
    (checkout / "nn" / "smuggled.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert demo_run._software(checkout)["dirty"] is True


def test_a_tree_that_is_not_a_checkout_fails_closed(tmp_path):
    """The fail-closed branch, still reached, and still dirty rather than clean."""
    block = demo_run._software(tmp_path)
    assert block == {
        "revision": "",
        "source_digest": "",
        "dirty": True,
        "python": sys.version.split()[0],
    }


def test_a_checkout_git_cannot_name_a_revision_for_is_dirty(tmp_path):
    """An identity git only half-answered is refused, not averaged out.

    `source_identity` reports `revision: None` with `dirty: False` for a tree
    git can list but has no commit to name -- `bool(None)` is False, so a block
    built field by field would claim a CLEAN tree while naming no source at all,
    and the campaign whose records exist to say what produced them would start.
    """
    root = tmp_path / "uncommitted"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    assert demo_run._software(root)["dirty"] is True
    assert demo_run._software(root)["revision"] == ""


def test_an_identity_whose_dirty_is_unanswered_is_refused(monkeypatch):
    """The `dirty is not None` conjunct, which no real-checkout witness reaches.

    `source_identity` answers `dirty: None` when `git status` could not be run
    while `ls-files` and `rev-parse` both succeeded -- a complete-looking
    identity whose one load-bearing field is the unanswered one. `bool(None)`
    is False, so without that conjunct this returns `dirty: False` and a
    campaign starts on a tree nobody asked git about.

    Stubbed rather than built from git, because that combination cannot be
    produced by a real checkout on demand. It is safe to stub here only
    because the real-checkout witnesses above already pin the call signature
    and the by-key reads; this test pins neither and is not a substitute.
    """
    import nn.source_identity

    monkeypatch.setattr(
        nn.source_identity,
        "source_identity",
        lambda root: {"revision": "a" * 40, "source_digest": "b" * 64, "dirty": None},
    )
    block = demo_run._software()
    assert block["dirty"] is True
    assert block["revision"] == ""
    assert block["source_digest"] == ""


def test_this_checkout_identifies_itself_with_no_argument():
    """The call-signature witness: the default root is a real checkout.

    `source_identity()` without its `root` raises `TypeError`, which the
    fail-closed branch swallows into an empty revision. Asserting a real
    revision here is what makes that regression visible instead of silent.
    """
    block = demo_run._software()
    assert GIT_OBJECT_ID.match(block["revision"]), block["revision"]
    assert HEX64.match(block["source_digest"]), block["source_digest"]


def _campaign_runner(tmp_path, checkout):
    """A runner on the CAMPAIGN profile whose software block is a real identity.

    The profile is forced the way `test_allow_dirty_is_refused_for_a_campaign_profile`
    already forces it: the committed CAMPAIGN configuration cannot build a runner
    while `protocol_hash` is null (`chimera/demo/config.py` refuses a CAMPAIGN
    carrying `rules` until the protocol that froze them is named), and that is a
    separate R1 item.
    """
    harness = build(tmp_path, start=False)
    config = harness.runner.config
    object.__setattr__(config, "profile", type(config.profile).CAMPAIGN)
    harness.runner.software = demo_run._software(checkout)
    return harness


def test_a_campaign_passes_self_check_on_a_genuinely_clean_tree(tmp_path, checkout):
    """The behaviour R1-a restores: a clean campaign starts."""
    harness = _campaign_runner(tmp_path, checkout)
    assert harness.runner.software["dirty"] is False
    assert harness.runner.start() is RunnerState.READY
    assert harness.runner.halt_reason is None


def test_a_campaign_is_refused_on_a_genuinely_dirty_tree(tmp_path, checkout):
    """The behaviour R1-a must not lose: a dirty campaign is refused.

    Driven by real `git status` rather than by setting the flag by hand, so it
    fails if the identity is ever read as an attribute or built from a root the
    runner does not actually run from.
    """
    (checkout / "chimera" / "__init__.py").write_text("# edited\n", encoding="utf-8")
    harness = _campaign_runner(tmp_path, checkout)
    assert harness.runner.software["dirty"] is True
    assert harness.runner.start() is RunnerState.HALT
    assert "source_identity" in (harness.runner.halt_reason or "")


# ---------------------------------------------------------------------------
# `resolve --equity`: R1-b's clearing path, driven through main()
# ---------------------------------------------------------------------------
# The parser is checked by `tests/test_demo_runbook.py` and the semantics by
# `tests/test_r1b_persisted_equity.py`. What is checked HERE is the thing both of
# those take on trust: that the operator can actually reach it. The dispute this
# settles is re-raised at every construction, so a command that parses but does
# not run would leave a campaign with no permitted way out.


class _SimulatedKill(Exception):
    """Stands in for a SIGKILL arriving at one exact statement."""


def _diverged(tmp_path: Path):
    """A real disagreement, made the way the runner really makes one.

    The kill lands between `_save_ledger()` and `update_equity()` in `tick` --
    R1-b's known window. Nothing is hand-edited.
    """
    harness = build(tmp_path, days=(DAY,))
    first = harness.first_minute_ms()
    harness.run(479, start=first)

    def killed(*args, **kwargs):
        raise _SimulatedKill()

    harness.runner.risk.update_equity = killed
    with pytest.raises(_SimulatedKill):
        harness.tick(first + 479 * 60_000)
    return harness


def test_resolve_equity_settles_the_dispute_and_the_campaign_runs_again(tmp_path, capsys):
    """The whole operator recovery, through the CLI, as separate invocations."""
    harness = _diverged(tmp_path)
    config_path = written_config(tmp_path, harness)
    argv = ["--config", str(config_path), "--root", str(harness.root), "--profile", "TEST"]

    assert demo_run.main(argv + ["run"]) == demo_run.EXIT_HALTED
    assert "equity_dispute" in json.loads(capsys.readouterr().out)["reason"]

    assert (
        demo_run.main(argv + ["resolve", "--equity", "--note", "ledger is right"])
        == demo_run.EXIT_OK
    )
    settled = json.loads(capsys.readouterr().out)
    assert settled["resolved"] == "equity"
    assert settled["record"]

    assert demo_run.main(argv + ["run", "--once"]) == demo_run.EXIT_OK
    assert json.loads(capsys.readouterr().out)["state"] != RunnerState.HALT.value


def test_resolve_equity_refuses_rather_than_clearing_nothing(tmp_path, capsys):
    """No dispute, so the command changes nothing and says so on stderr."""
    harness = build(tmp_path, days=(DAY,))
    harness.run(3)
    config_path = written_config(tmp_path, harness)
    argv = ["--config", str(config_path), "--root", str(harness.root), "--profile", "TEST"]

    assert demo_run.main(argv + ["resolve", "--equity", "--note", "nothing is wrong"]) == (
        demo_run.EXIT_REFUSED
    )
    assert "no equity dispute to settle" in capsys.readouterr().err


def test_resolve_names_which_dispute_it_is_clearing(tmp_path):
    """One or the other, never both and never neither."""
    parser = demo_run.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["resolve", "--note", "n"])
    with pytest.raises(SystemExit):
        parser.parse_args(["resolve", "--symbol", "BTC/USDT", "--equity", "--note", "n"])
    assert parser.parse_args(["resolve", "--equity", "--note", "n"]).equity is True
    assert (
        parser.parse_args(["resolve", "--symbol", "BTC/USDT", "--note", "n"]).equity is False
    )


# ---------------------------------------------------------------------------
# what a halt tells the operator to type
# ---------------------------------------------------------------------------
# The halt reason is where an operator meets the equity dispute: section 7 of the
# runbook makes `risk.json`'s `halt_reason` the authoritative text, and `run`
# prints the same string. It shipped advertising `demo_run resolve-equity
# --note`, which no subcommand matches -- argparse exits 2 on "invalid choice".
# Nothing caught it: `tests/test_demo_runbook.py` parses the commands in the
# runbook and never the ones inside a Python string, and every assertion on this
# reason elsewhere stops at its `equity_dispute:` prefix.
#
# The oracle is the real parser and never a second copy of the sentence. A test
# asserting the message equals some expected text would agree with whatever the
# message happened to say, which is the tautology `tests/test_demo_runbook.py`
# refuses in its own module docstring.

#: A command a message tells a human to type. Backticks are how this repository
#: marks one, in prose and in these strings alike. Distinct from
#: `_BACKTICKED_SPAN` above on purpose: that one harvests `status` FIELD names
#: out of the runbook, and widening either to serve both would break the other.
_ADVERTISED = re.compile(r"`demo_run ([^`]+)`")

#: Where a message an operator reads can be raised: Aegis, the demo runtime that
#: wires it, and the CLI that prints what they say. Bounded on purpose -- a scan
#: of the whole tree would read hundreds of files to check a handful of strings.
_MESSAGE_SOURCES = (
    Path(__file__).resolve().parents[1] / "chimera",
    Path(__file__).resolve().parents[1] / "chimera" / "demo",
    Path(__file__).resolve().parents[1] / "tools",
)

#: Which of those files must carry an advertised command, written out here by
#: hand. Equality, not containment, and for `tests/test_demo_runbook.py`'s
#: reason: a set harvested from the sources would agree with whatever the
#: sources happened to say, and an empty harvest would agree with silence. A new
#: message in a new file is a deliberate edit here, by a reviewer.
EXPECTED_ADVERTISERS = frozenset({"risk_wiring.py"})


def advertised_commands(text: str) -> list[str]:
    """Every `demo_run ...` span in ``text``, each fed to the real parser.

    Returns what it found rather than asserting on the count, so each caller can
    refuse its own empty scan. A checker that silently matched nothing would keep
    passing for as long as the thing it checks stayed broken.

    Parseable is what this can prove, and it is not the same as runnable:
    `--config` and `--root` are optional to argparse and required by
    `demo_run._load`, so a command that parses here can still exit on a missing
    one. That gap is what the runbook pointer in the message is for.
    """
    parser = demo_run.build_parser()
    found = _ADVERTISED.findall(text)
    for command in found:
        parser.parse_args(shlex.split(command))
    return found


def test_the_command_the_equity_dispute_advertises_is_one_the_parser_accepts(tmp_path, capsys):
    """The string read at 03:00, against the parser it will be typed at.

    Read out of `risk.json`, which is where the runbook sends an operator and
    not merely where it is convenient to look: `run` prints the RUNNER's halt
    reason, and `_halt` keeps the FIRST one, so any earlier refusal -- a dirty
    working tree, a regressed ledger -- stands in front of this dispute there
    while Aegis still holds it underneath. The authoritative text is the one
    Aegis persisted, and that is the one an operator acts on.
    """
    harness = _diverged(tmp_path)
    config_path = written_config(tmp_path, harness)
    argv = ["--config", str(config_path), "--root", str(harness.root), "--profile", "TEST"]

    assert demo_run.main(argv + ["run"]) == demo_run.EXIT_HALTED
    capsys.readouterr()
    persisted = json.loads((harness.state_dir / "risk.json").read_text(encoding="utf-8"))

    assert persisted["halted"] is True
    assert persisted["halt_reason"].startswith("equity_dispute:")
    assert advertised_commands(persisted["halt_reason"]), (
        "the dispute must tell the operator how to settle it: `resume` does not "
        "clear it and hand-editing a state file is forbidden, so the command named "
        "here is the only way out of this halt"
    )


def test_no_operator_message_advertises_a_command_the_parser_would_refuse():
    """The whole class, not just the one string that was wrong.

    A message is the only documentation an operator has in front of them at the
    moment they need it, and it is checked by nothing else: the runbook's
    commands are parsed by `tests/test_demo_runbook.py`, the alert annotations by
    `promtool`, and a command quoted in a Python string by neither.
    """
    advertisers: set[str] = set()
    for directory in _MESSAGE_SOURCES:
        assert directory.is_dir(), f"{directory} is not where the messages live"
        for source in sorted(directory.glob("*.py")):
            if advertised_commands(source.read_text(encoding="utf-8")):
                advertisers.add(source.name)

    assert advertisers == set(EXPECTED_ADVERTISERS), (
        "a file started or stopped telling an operator what to type, and the list "
        "above is where a reviewer says so"
    )


def test_the_advertised_command_check_catches_the_string_that_shipped():
    """Negative control: the parser must be the thing doing the work."""
    with pytest.raises(SystemExit):
        advertised_commands('`demo_run resolve-equity --note "n"`')

    assert advertised_commands("an operator decides which is right") == []
