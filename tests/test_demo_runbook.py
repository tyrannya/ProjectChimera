"""`docs/demo_runbook.md`: every command in it is one this repository supports.

A runbook is read once, at 03:00, by somebody who is already having a bad night.
A command in it that no longer parses is worse than no runbook: the operator
concludes the tool is broken, or worse, guesses at the flag it should have been.
So this file does not check that the document exists or that it mentions the
right words in prose -- it takes every command out of every fenced block and
feeds it to the argparse parser of the tool it names, and it checks every `make`
target against the `Makefile` and every systemd unit against `deploy/systemd/`.

Argparse is the right depth here. It validates flags, subcommands, choices and
`nargs` against the real parser, and it touches no disk, opens no socket and
runs no campaign: `parse_args` on `--config conf/demo/pvc1.json` builds a `Path`
and stops. What it cannot check is whether the command would then do the right
thing, which is what the rest of the demo test suite is for.

Every oracle below -- the tool names, the make targets, the unit names, the
required phrases -- is written out in this file by hand. None of them is
collected from the document under test: a set read out of the runbook would
agree with whatever the runbook happened to say, which is the shape of tautology
this repository has shipped once already and does not intend to ship again.
"""

from __future__ import annotations

import re
import shlex
from importlib import import_module
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "demo_runbook.md"
MAKEFILE = ROOT / "Makefile"
COMPOSE = ROOT / "docker-compose.yml"
SYSTEMD = ROOT / "deploy" / "systemd"

#: How each tool's parser is reached. Two spellings exist in `tools/` and the
#: map is literal so a tool that quietly renamed its factory is a failure here
#: rather than a silently skipped command.
PARSER_FACTORIES: dict[str, str] = {
    "tools.recorder": "build_parser",
    "tools.demo_run": "build_parser",
    "tools.demo_report": "build_parser",
    "tools.replay_parity": "build_parser",
    "tools.freeze_evidence": "build_argparser",
}

#: `python -m` targets that are not repository tools. `json.tool` is the
#: standard library's pretty-printer, used in section 6 to read the two futures
#: stores. Anything else must be justified by being added here.
STDLIB_MODULES: frozenset[str] = frozenset({"json.tool"})

#: Shell verbs a runbook command may start with. Deliberately short: these are
#: the ones whose behaviour is the operating system's rather than this
#: repository's, so there is no parser here to check them against. A command
#: starting with anything else is either a tool invocation, which is checked, or
#: a mistake.
SHELL_VERBS: frozenset[str] = frozenset({"sudo", "docker", "touch", "rm", "mkdir", "df"})

#: The only thing `sudo` is allowed to run. A runbook that told an operator to
#: `sudo rm` something would be a runbook that had stopped being reviewed.
SUDO_ALLOWS: frozenset[str] = frozenset({"systemctl"})

#: The make targets the runbook uses, written out here. Compared for equality
#: against what is actually found, so a target that vanishes from the document
#: fails just as loudly as one that appears in it unannounced.
EXPECTED_MAKE_TARGETS: frozenset[str] = frozenset(
    {"recorder-preflight", "demo-report", "demo-replay-parity"}
)

#: The tools the runbook drives, written out for the same reason.
EXPECTED_TOOLS: frozenset[str] = frozenset(
    {
        "tools.recorder",
        "tools.demo_run",
        "tools.demo_report",
        "tools.replay_parity",
        "tools.freeze_evidence",
        "json.tool",
    }
)

#: The systemd units the runbook names. Each must exist under `deploy/systemd/`.
EXPECTED_UNITS: frozenset[str] = frozenset({"chimera-recorder", "chimera-demo"})

#: The two adopted metrics ports. 9102 is the recorder's and is already in
#: `deploy/systemd/chimera-recorder.service`; 9103 is the runner's.
EXPECTED_PORTS: tuple[str, str] = ("9102", "9103")

#: The compose profile the demo deployment starts under.
EXPECTED_PROFILE = "demo"

_FENCE = re.compile(r"^```", re.MULTILINE)


def code_blocks(text: str) -> list[str]:
    """The contents of every fenced block, in document order."""
    parts = _FENCE.split(text)
    # split() alternates outside/inside starting outside, so the odd indices are
    # the block bodies. An unclosed fence would leave a trailing body, which is
    # a malformed document and is asserted against below.
    return [part.lstrip("\n") for part in parts[1::2]]


def prose(text: str) -> str:
    """The document with every run of whitespace collapsed to one space.

    The phrase checks below are about what the runbook SAYS, and a sentence that
    was rewrapped by an editor is the same sentence. Matching against the raw
    text would make every prose oracle depend on where the line breaks fell,
    which is how a guard ends up being loosened to make a reflow pass.
    """
    return " ".join(text.split())


def commands(text: str) -> list[str]:
    """Every command in every block, with backslash continuations joined."""
    found: list[str] = []
    for block in code_blocks(text):
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            found.append(line)
    return found


def check_command(command: str) -> tuple[str, str]:
    """Validate one command and say what it was.

    Returns ``(kind, name)`` -- ``("make", target)``, ``("tool", module)`` or
    ``("shell", verb)`` -- and raises for anything it cannot account for. The
    return value is what the coverage test below counts, so a command that
    passed validation cannot also be invisible to it.

    One command, not a pipeline. The runbook writes each step on its own line
    and redirects in prose, and a checker that split on ``|`` would have to
    decide which half of a pipeline the line "is" before it could be counted.
    A pipeline is refused here rather than half-checked.
    """
    assert "|" not in command, f"one command per line, not a pipeline: {command!r}"
    tokens = shlex.split(command.strip())
    assert tokens, f"empty command in {command!r}"
    head, rest = tokens[0], tokens[1:]

    if head == "make":
        targets = [token for token in rest if "=" not in token]
        assert len(targets) == 1, f"expected one make target in {command!r}"
        _assert_make_target(targets[0])
        return ("make", targets[0])

    if head == "python":
        assert rest[:1] == ["-m"], f"only `python -m` is documented: {command!r}"
        module, args = rest[1], rest[2:]
        if module in STDLIB_MODULES:
            return ("tool", module)
        factory = PARSER_FACTORIES.get(module)
        assert factory is not None, f"{module} has no parser factory in this test"
        parser = getattr(import_module(module), factory)()
        parser.parse_args(args)
        return ("tool", module)

    if head in SHELL_VERBS:
        if head == "sudo":
            assert rest[0] in SUDO_ALLOWS, f"sudo may not run {rest[0]!r}"
            _assert_units(rest)
        return ("shell", head)

    raise AssertionError(f"unrecognised command in the runbook: {command!r}")


def _assert_make_target(target: str) -> None:
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert re.search(
        rf"^{re.escape(target)}:", makefile, re.MULTILINE
    ), f"`make {target}` is documented but the Makefile has no such rule"
    phony = re.search(r"^\.PHONY:(.*?)(?=\n[^\t\s])", makefile, re.MULTILINE | re.DOTALL)
    assert phony is not None, "the Makefile has no .PHONY block"
    assert target in phony.group(1).split(), f"{target} is missing from .PHONY"


def _assert_units(tokens: list[str]) -> None:
    """`sudo systemctl <verb> <unit> [<unit>...]` names units that exist."""
    for unit in tokens[2:]:
        assert (SYSTEMD / f"{unit}.service").is_file(), f"no unit file for {unit}"


# --- the document itself -------------------------------------------------


def test_the_runbook_exists_and_its_fences_are_balanced():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert len(_FENCE.findall(text)) % 2 == 0, "an unclosed code fence"
    assert code_blocks(text), "the runbook has no commands at all"


def test_every_documented_command_parses():
    """The whole point of the file. Every command, against the real parser."""
    for command in commands(RUNBOOK.read_text(encoding="utf-8")):
        check_command(command)


def test_the_checker_rejects_a_flag_the_tool_does_not_have():
    """Negative control: the parser must be the thing doing the work."""
    with pytest.raises(SystemExit):
        check_command(
            "python -m tools.demo_run --config c.json --root data resolve "
            '--leg "BTC/USDT:USDT" --note "n"'
        )


def test_the_checker_rejects_a_make_target_that_does_not_exist():
    with pytest.raises(AssertionError, match="no such rule"):
        check_command("make demo-report-that-was-renamed DEMO_DAY=2026-09-19")


def test_the_checker_rejects_a_unit_that_has_no_file():
    with pytest.raises(AssertionError, match="no unit file"):
        check_command("sudo systemctl start chimera-nonexistent")


def test_the_checker_rejects_an_unrecognised_verb():
    with pytest.raises(AssertionError, match="unrecognised command"):
        check_command("curl -s http://example.invalid/")


def test_the_runbook_drives_exactly_the_tools_and_targets_named_here():
    """A literal coverage oracle, so a silently empty scan cannot pass.

    Equality rather than containment: a command that disappears from the runbook
    fails this, and so does a tool that appears in it without a reviewer having
    put it in the list above.
    """
    targets: set[str] = set()
    tools: set[str] = set()
    for command in commands(RUNBOOK.read_text(encoding="utf-8")):
        kind, name = check_command(command)
        if kind == "make":
            targets.add(name)
        elif kind == "tool":
            tools.add(name)
    assert targets == set(EXPECTED_MAKE_TARGETS)
    assert tools == set(EXPECTED_TOOLS)


def test_the_runbook_names_the_systemd_units_that_exist():
    text = prose(RUNBOOK.read_text(encoding="utf-8"))
    named = {unit for unit in EXPECTED_UNITS if unit in text}
    assert named == set(EXPECTED_UNITS)
    for unit in EXPECTED_UNITS:
        assert (SYSTEMD / f"{unit}.service").is_file()


def test_the_runbook_uses_the_demo_kill_switch_path():
    """Section 11.5's path is `chimera/risk.py`'s default, which is Freqtrade's.

    The demo's `RiskEngine` is constructed with the runner's state directory, so
    an operator who touched `user_data/KILL_SWITCH` would watch a campaign keep
    running while believing it had been stopped.
    """
    text = prose(RUNBOOK.read_text(encoding="utf-8"))
    assert "state/demo/KILL_SWITCH" in text
    assert "user_data/KILL_SWITCH" in text, "the wrong path must be named as wrong"
    assert "touch state/demo/KILL_SWITCH" in text
    assert "touch user_data/KILL_SWITCH" not in text


def test_the_runbook_names_the_adopted_ports():
    text = prose(RUNBOOK.read_text(encoding="utf-8"))
    for port in EXPECTED_PORTS:
        assert port in text, f"the runbook never mentions port {port}"
    assert "--metrics-port 9102" in text
    assert "--metrics-port 9103" in text


def test_the_runbook_starts_the_compose_profile_the_new_services_carry():
    """The document and the compose file must name the same profile."""
    text = prose(RUNBOOK.read_text(encoding="utf-8"))
    profiles = set(re.findall(r"--profile (\w+)", text))
    assert profiles == {EXPECTED_PROFILE}
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    for service in ("recorder", "demo"):
        assert compose["services"][service]["profiles"] == [EXPECTED_PROFILE]


def test_the_runbook_records_that_a_campaign_cannot_start_today():
    """Two blockers that would otherwise be discovered as an unexplained halt."""
    text = prose(RUNBOOK.read_text(encoding="utf-8"))
    assert "protocol_hash" in text
    assert "SELF_CHECK" in text
    assert "--allow-dirty" in text


def test_the_runbook_states_the_operator_prohibitions():
    """The section that exists to be read before a campaign, not after one."""
    text = prose(RUNBOOK.read_text(encoding="utf-8"))
    heading = "## 16. What an operator may never do during a campaign"
    assert heading in text
    section = text.split(heading, 1)[1]
    for phrase in (
        "config_hash",
        "Restart the recorder",
        "frozen manifest",
        "--allow-dirty",
        "decision-log file",
        "reconciliation dispute",
        "docs/p*_preregistration.md",
        "leverage above 1x",
    ):
        assert phrase in section, f"the prohibitions no longer mention {phrase!r}"


def test_the_runbook_asserts_no_research_result():
    """`tools.verify_research_state` runs in CI; a doc must not contradict it."""
    text = prose(RUNBOOK.read_text(encoding="utf-8"))
    assert "P8 was withdrawn and has no result" in text
    assert "P13 is preregistered and unrun" in text
    assert "answers no research question" in text


def test_the_runbook_does_not_claim_the_future_work_has_happened():
    """It may describe the monthly freeze; it may not report one."""
    text = prose(RUNBOOK.read_text(encoding="utf-8"))
    assert "Nothing in this section has happened" in text
    assert "No prospective artifact exists in this repository" in text
    assert not (ROOT / "artifacts" / "prospective").exists()
