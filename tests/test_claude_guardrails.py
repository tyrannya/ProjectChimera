"""The Claude Code guardrail layer: both tiers, tested in both directions.

`.claude/settings.json` carries two independent barriers against rewriting
published Git chronology and reading credentials. `permissions.deny` is the
native floor that Claude Code enforces itself; `.claude/hooks/chimera-guardrails.mjs`
is the enhanced PreToolUse hook that sees the invocation forms prefix rules
cannot (`git -C`, quoting, `bash -c`, flag clusters, chaining, Windows
spellings). docs/claude_extension_layer.md explains the threat model.

Every fixture in `claude_guardrails_cases.json` is a PreToolUse call wrapped
in a real hook envelope and run through the hook with Node, so both CI legs
exercise it. A deny must be for the named rule, not merely a deny; an allow
must be silent, because the hook never grants permission. Fixtures tagged
`native` must also be covered by a native deny rule under the documented rule
semantics, and `native_negative` ones by none: that is the coverage that is
left when the hook cannot start or times out. The live probes that confirmed
those semantics on Claude Code 2.1.278 are recorded in the doc.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / ".claude" / "settings.json"
HOOK = ROOT / ".claude" / "hooks" / "chimera-guardrails.mjs"
FIXTURES = json.loads(
    (Path(__file__).parent / "claude_guardrails_cases.json").read_text("utf-8")
)
CASES = FIXTURES["cases"]
NODE = shutil.which("node")
MATCHER = "Bash|PowerShell|Monitor|Read|Write|Edit|NotebookEdit|Grep"
SKILLS = (
    "chimera-author",
    "chimera-remediate",
    "chimera-review",
    "chimera-delta-review",
    "chimera-reproduce",
    "chimera-verify",
    "chimera-mutation-audit",
)
OWNER_ONLY = {
    "chimera-review",
    "chimera-delta-review",
    "chimera-reproduce",
    "chimera-mutation-audit",
}

# Locally a missing Node skips; in CI it must fail, or both legs would pass vacuously.
needs_node = pytest.mark.skipif(
    NODE is None and not os.environ.get("CI"), reason="node is not installed"
)


def run_hook(stdin: str, script: Path = HOOK) -> subprocess.CompletedProcess:
    return subprocess.run(
        [NODE or "node", str(script)],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )


def payload(case: dict) -> str:
    return json.dumps(
        {**FIXTURES["envelope"], "tool_name": case["tool"], "tool_input": case["input"]}
    )


def assert_deny_json(result: subprocess.CompletedProcess, rule: str) -> str:
    assert result.returncode == 2, result
    out = json.loads(result.stdout)
    assert list(out) == ["hookSpecificOutput"]
    spec = out["hookSpecificOutput"]
    assert set(spec) == {"hookEventName", "permissionDecision", "permissionDecisionReason"}
    assert spec["hookEventName"] == "PreToolUse"
    assert spec["permissionDecision"] == "deny"
    reason = spec["permissionDecisionReason"]
    assert f"[{rule}]" in reason, reason
    assert reason in result.stderr
    return reason


_REASONS: dict[str, str] = {}


@needs_node
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_fixture(case):
    result = run_hook(payload(case))
    if case["expect"] == "allow":
        assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
        return
    reason = assert_deny_json(result, case["rule"])
    # The reason is a function of the rule alone, so no command text (and no
    # secret inlined in one) can ever reach the transcript through it.
    assert _REASONS.setdefault(case["rule"], reason) == reason


def test_fixtures_are_two_sided_for_every_rule():
    denied = {c["rule"] for c in CASES if c["expect"] == "deny"}
    allowed = sum(c["expect"] == "allow" for c in CASES)
    assert allowed >= 50 and len(denied) >= 15
    assert all(c["expect"] in {"allow", "deny"} for c in CASES)
    assert all(("rule" in c) == (c["expect"] == "deny") for c in CASES)


# --- degraded modes ------------------------------------------------------------


@needs_node
@pytest.mark.parametrize("stdin", ["this is not json", "", "null", "[]", "42"])
def test_malformed_stdin_fails_closed(stdin):
    assert_deny_json(run_hook(stdin), "internal-error")


@needs_node
def test_oversized_command_fails_closed():
    big = {
        **FIXTURES["envelope"],
        "tool_name": "Bash",
        "tool_input": {"command": "x" * 2_000_001},
    }
    assert_deny_json(run_hook(json.dumps(big)), "internal-error")


@needs_node
def test_a_hook_that_cannot_start_is_not_a_block(tmp_path):
    """Characterises the degraded mode rather than hiding it.

    Claude Code blocks a PreToolUse call only on exit 2 (or a deny decision). A
    missing script exits with Node's own error code, which Claude Code treats
    as a non-blocking hook error: the call proceeds and only the native deny
    rules stand. `test_native_floor_covers_canonical_forms` is that floor.
    """
    result = run_hook(payload(CASES[0]), script=tmp_path / "missing.mjs")
    assert result.returncode not in (0, 2)
    assert result.stdout == ""


@needs_node
@pytest.mark.parametrize(
    "command",
    [
        "a " * 500_000,
        "'a'" * 10_000 + " git push --force",
        'bash -c "' * 50 + "git push --force",
        "$(" * 5_000 + "git rebase",
        "cat <<EOF\n" + "line\n" * 100_000,
        "git " * 30_000,
        "git status; " * 20_000 + "git -C x reset --hard",
        "bash -c 'git status' " * 5_000,
        "bash -c env " + "-c env " * 140_000,
        "bash -c compgen " + "-c compgen " * 90_000,
        "env " + "> x " * 250_000,
        "{" * 1_000_000 + "git status",
        "& (Get-Command git) " * 50_000 + "status",
    ],
    ids=[
        "1MB-words",
        "10k-quotes",
        "nested-launchers",
        "nested-substitutions",
        "long-heredoc",
        "30k-git-words",
        "20k-git-segments",
        "5k-launchers",
        "140k-env-launch-flags",
        "90k-compgen-launch-flags",
        "250k-redirects",
        "1M-braces",
        "50k-get-command",
    ],
)
def test_pathological_input_finishes_far_inside_the_timeout(command):
    """A real timeout (10 s) would let the call through under the native rules only."""
    started = time.monotonic()
    run_hook(
        json.dumps(
            {**FIXTURES["envelope"], "tool_name": "Bash", "tool_input": {"command": command}}
        )
    )
    assert time.monotonic() - started < 5.0


# --- the native floor ----------------------------------------------------------


def _glob(pattern: str, star: str) -> str:
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + star, i + 1
        elif pattern[i] == "?" and star != ".*":
            out, i = out + "[^/]", i + 1
        elif pattern[i] == "[":
            end = pattern.index("]", i)
            out, i = out + pattern[i : end + 1], end + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return out


def native_denies(tool: str, value: str, rules: list[str]) -> list[str]:
    """Which deny rules match, under the documented rule semantics.

    Bash/PowerShell: `*` is any sequence, a trailing ` *` also matches no
    arguments, PowerShell is case-insensitive. Read/Edit: gitignore paths, `//`
    the filesystem root, `~/` home; `Edit` governs Write and NotebookEdit, and a
    Read deny also blocks Edit and Write.
    """
    hits = []
    for rule in rules:
        kind, spec = rule[: rule.index("(")], rule[rule.index("(") + 1 : -1]
        if tool in {"Bash", "PowerShell"} and kind == tool:
            body = (
                _glob(spec[:-2], ".*") + "( .*)?" if spec.endswith(" *") else _glob(spec, ".*")
            )
            flags = re.IGNORECASE if kind == "PowerShell" else 0
            if re.fullmatch(body, value.strip(), flags):
                hits.append(rule)
        elif kind in {"Read", "Edit"}:
            if not (kind == "Read" and tool in {"Read", "Grep", "Edit", "Write"}) and not (
                kind == "Edit" and tool in {"Edit", "Write", "NotebookEdit"}
            ):
                continue
            path = value.replace("\\", "/")
            if spec.startswith("//"):
                ok = (
                    re.fullmatch("(.*/)?" + _glob(spec[5:], "[^/]*"), path)
                    if spec.startswith("//**/")
                    else None
                )
            else:
                ok = re.fullmatch(_glob(spec, "[^/]*"), path)
            if ok:
                hits.append(rule)
    return hits


def _native_value(case: dict) -> str:
    inp = case["input"]
    return (
        inp.get("command") or inp.get("file_path") or inp.get("notebook_path") or inp["path"]
    )


def test_native_floor_covers_canonical_forms():
    rules = json.loads(SETTINGS.read_text("utf-8"))["permissions"]["deny"]
    tagged = [c for c in CASES if c.get("native")]
    assert len(tagged) >= 40
    for case in tagged:
        assert case["expect"] == "deny"
        assert native_denies(case["tool"], _native_value(case), rules), case["id"]


def test_native_floor_leaves_ordinary_work_alone():
    rules = json.loads(SETTINGS.read_text("utf-8"))["permissions"]["deny"]
    for case in (c for c in CASES if c.get("native_negative")):
        assert case["expect"] == "allow"
        assert not native_denies(case["tool"], _native_value(case), rules), case["id"]


def test_native_env_rules_spare_only_the_committed_template():
    rules = json.loads(SETTINGS.read_text("utf-8"))["permissions"]["deny"]
    spared = [".env.example"]
    denied = [
        ".env",
        ".env.local",
        ".env.e",
        ".env.ex",
        ".env.exa",
        ".env.exampl",
        ".env.examples",
        ".env.example.bak",
        ".envrc",
        ".env-prod",
        ".env_local",
        ".env.1",
        ".env._x",
        ".env.Production",
    ]
    for name in spared:
        assert not native_denies("Read", "/r/" + name, rules), name
    for name in denied:
        assert native_denies("Read", "/r/" + name, rules), name


# --- structure -----------------------------------------------------------------


def test_settings_keep_plan_mode_and_register_the_hook_explicitly():
    settings = json.loads(SETTINGS.read_text("utf-8"))
    assert settings["permissions"]["defaultMode"] == "plan"
    [entry] = settings["hooks"]["PreToolUse"]
    assert entry["matcher"] == MATCHER
    [hook] = entry["hooks"]
    assert hook["type"] == "command" and hook["command"] == "node"
    assert hook["args"] == ["${CLAUDE_PROJECT_DIR}/.claude/hooks/chimera-guardrails.mjs"]
    assert 0 < hook["timeout"] <= 30
    assert HOOK.is_file()


def test_no_path_rule_that_claude_code_would_silently_ignore():
    for rule in json.loads(SETTINGS.read_text("utf-8"))["permissions"]["deny"]:
        assert not re.match(r"(Write|MultiEdit|NotebookEdit|Glob)\(", rule), rule


def _frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text("utf-8")
    assert text.startswith("---\n"), path
    head, body = text[4:].split("\n---\n", 1)
    fields = dict(line.split(": ", 1) for line in head.splitlines() if ": " in line)
    return fields, body


@pytest.mark.parametrize("name", SKILLS)
def test_skill_contract(name):
    fields, body = _frontmatter(ROOT / ".claude" / "skills" / name / "SKILL.md")
    assert fields["name"] == name
    assert fields["description"].strip()
    assert "model" not in fields, "a skill must not select a foundation model"
    assert (fields.get("disable-model-invocation") == "true") == (name in OWNER_ONLY)
    assert "gh pr merge" not in body and "CLAUDE.md" in body
    if name in {"chimera-review", "chimera-delta-review"}:
        assert "READ-ONLY" in body
        assert "BLOCKED / CANNOT VERIFY" in body
        assert "never independent acceptance" in body
    if name == "chimera-delta-review":
        assert "DELTA APPROVED" in body and "REQUEST CHANGES" in body
    if name == "chimera-review":
        assert "REQUEST CHANGES — PR #" in body and "NOT READY" in body


def test_agents_inherit_the_serving_model():
    for agent in (ROOT / ".claude" / "agents").glob("*.md"):
        fields, _ = _frontmatter(agent)
        assert fields["model"] == "inherit", agent


def test_the_reminder_rule_is_short_and_unconditional():
    text = (ROOT / ".claude" / "rules" / "guardrail-reminder.md").read_text("utf-8")
    assert not text.startswith("---"), "a paths: frontmatter would make it conditional"
    assert len(text.splitlines()) <= 15
