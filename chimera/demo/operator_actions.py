"""R1-j: the committed operator-action file a replay uses as its OPERATOR counterpart.

The roadmap's item, verbatim:

    **R1-j Replay parity policy:** decide `seq` (per-kind sequence or exclusion
    of operational kinds) and `OPERATOR` (deterministic replay counterpart from a
    committed operator-action file); re-include deterministic risk fields in the
    hash; the 48 h synthetic fixture reaches PARITY with a restart and an
    operator action.

An operator command is intentional state mutation. A replay reads finished
recorder files and has no operator, so without a counterpart it could never
produce the ``OPERATOR`` records a live campaign wrote, and a campaign day with a
flatten could never reach PARITY. Making ``OPERATOR`` invisible to the
comparison would exempt a safety-significant action from it. This file is the
other answer: the operator's actions are written down, committed, and replayed
through the same runner methods the CLI calls.

**The format.** One JSON document, schema :data:`OPERATOR_ACTIONS_SCHEMA`::

    {
      "actions": [
        {
          "after_minute": "2026-09-20T05:59:00+00:00",
          "command": "flatten",
          "id": "flatten-after-restart",
          "note": "why the operator did it"
        }
      ],
      "schema": "chimera.operator-actions/1"
    }

The bytes must be exactly ``json.dumps(document, indent=2, sort_keys=True,
ensure_ascii=True) + "\\n"`` (:func:`canonical_text`). A file in any other
spelling is refused, so one set of actions has one byte form and one hash
(:attr:`OperatorActions.file_hash`), which the parity report and every replayed
``OPERATOR`` record name.

**When an action happens.** ``after_minute`` is the minute the runner had
processed last when the operator ran the command: the runner's
``cursor.last_minute_processed``. It is the ``minute`` the live command's own
``OPERATOR`` records carry, so the action is anchored on a recorded minute, not
on a ``seq`` (which restarts shift) and not on a wall-clock time. The replay
applies it after deciding that minute and before the next one. Several actions
at one anchor are applied in file order; ``after_minute`` may not decrease
through the file; a command may appear only once at one anchor.

**What an action is.** ``flatten`` and ``resume``, each with a non-blank note
and no selector: both are whole-campaign commands in the CLI. ``resolve`` is
recognised and refused. Every dispute it clears is a fact about files that a
crash, a disk or an operator damaged, and a replay from recorder inputs starts
from an empty state directory where no such dispute can exist. A file that says
a ``resolve`` happened therefore cannot be replayed, and saying so is the
honest answer. Its selector is still checked first, so an ambiguous one is
named as such.

**What it is not.** Engineering replay input. It creates no evidence and no
decision, and nothing on a live path reads it. No environment variable, prompt,
clock or uncommitted state can add an action: the only input is the file.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "OPERATOR_ACTIONS_SCHEMA",
    "REPLAYABLE_COMMANDS",
    "REFUSED_COMMANDS",
    "RESOLVE_SELECTORS",
    "OperatorAction",
    "OperatorActions",
    "OperatorActionError",
    "canonical_text",
    "load_operator_actions",
    "parse_operator_actions",
]

OPERATOR_ACTIONS_SCHEMA = "chimera.operator-actions/1"

#: The commands a replay performs, through `DemoRunner.flatten` and
#: `DemoRunner.resume`, the methods the CLI calls.
REPLAYABLE_COMMANDS: frozenset[str] = frozenset({"flatten", "resume"})

#: Commands this schema knows and refuses, with the reason.
REFUSED_COMMANDS: Mapping[str, str] = {
    "resolve": (
        "a `resolve` clears a dispute, and every dispute is a fact about files a "
        "crash, a disk or an operator damaged. A replay starts from an empty state "
        "directory and reads finished recorder files, so the dispute the action "
        "cleared does not exist there. The action has no deterministic replay "
        "counterpart, and a file claiming one is refused rather than skipped"
    ),
}

#: `resolve`'s selectors, exactly the CLI's mutually exclusive group.
RESOLVE_SELECTORS: tuple[str, ...] = ("symbol", "equity", "ledger", "risk_state", "emergency")

_ACTION_KEYS: frozenset[str] = frozenset({"id", "after_minute", "command", "note"})
_DOCUMENT_KEYS: frozenset[str] = frozenset({"schema", "actions"})
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class OperatorActionError(ValueError):
    """An operator-action file that cannot be replayed. Never skipped."""


@dataclass(frozen=True)
class OperatorAction:
    """One committed action: what the operator ran, and after which minute."""

    id: str
    after_minute: str
    command: str
    note: str

    @property
    def after_minute_ms(self) -> int:
        return int(datetime.fromisoformat(self.after_minute).timestamp() * 1000)


@dataclass(frozen=True)
class OperatorActions:
    """A validated file: its actions in application order, and its identity."""

    actions: tuple[OperatorAction, ...]
    file_hash: str

    def attribution(self, action: OperatorAction) -> dict[str, str]:
        """What a replayed OPERATOR record says about where it came from."""
        return {"id": action.id, "file_hash": self.file_hash}

    def to_report(self, path: str | None = None) -> dict[str, Any]:
        return {
            "schema": OPERATOR_ACTIONS_SCHEMA,
            "path": path,
            "file_hash": self.file_hash,
            "actions": [
                {"id": a.id, "after_minute": a.after_minute, "command": a.command}
                for a in self.actions
            ],
        }


def canonical_text(document: Mapping[str, Any]) -> str:
    """The one byte form an operator-action file may have."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _refuse(message: str) -> OperatorActionError:
    return OperatorActionError(f"operator-action file refused: {message}")


def _minute(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise _refuse(f"{where}.after_minute must be an ISO-8601 UTC minute, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise _refuse(f"{where}.after_minute {value!r} is not ISO-8601") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise _refuse(f"{where}.after_minute {value!r} is not UTC")
    if not value.endswith("+00:00") or parsed.isoformat() != value:
        raise _refuse(
            f"{where}.after_minute {value!r} is not the canonical spelling "
            f"{parsed.isoformat()!r}: the decision log writes minutes one way only"
        )
    if parsed.second or parsed.microsecond:
        raise _refuse(f"{where}.after_minute {value!r} is not on a minute boundary")
    return value


def _action(raw: Any, index: int) -> OperatorAction:
    where = f"actions[{index}]"
    if not isinstance(raw, Mapping):
        raise _refuse(f"{where} is not an object")
    command = raw.get("command")
    if not isinstance(command, str):
        raise _refuse(f"{where}.command must be a string, got {command!r}")
    if command == "resolve":
        chosen = [key for key in RESOLVE_SELECTORS if key in raw]
        if len(chosen) != 1:
            raise _refuse(
                f"{where} is a resolve with {len(chosen)} selectors ({chosen}); the CLI "
                f"takes exactly one of {list(RESOLVE_SELECTORS)}, so this one is "
                "ambiguous"
            )
        raise _refuse(f"{where} ({raw.get('id')!r}): {REFUSED_COMMANDS['resolve']}")
    if command not in REPLAYABLE_COMMANDS:
        raise _refuse(
            f"{where}.command {command!r} is not one of {sorted(REPLAYABLE_COMMANDS)}"
        )
    selectors = [key for key in RESOLVE_SELECTORS if key in raw]
    if selectors:
        raise _refuse(
            f"{where} is a {command} with the selector(s) {selectors}; {command} acts on "
            "the whole campaign and takes none, so the combination is impossible"
        )
    unknown = sorted(set(raw) - _ACTION_KEYS)
    missing = sorted(_ACTION_KEYS - set(raw))
    if unknown:
        raise _refuse(f"{where} carries unknown key(s) {unknown}")
    if missing:
        raise _refuse(f"{where} is missing {missing}")
    action_id = raw["id"]
    if not isinstance(action_id, str) or not _ID.match(action_id):
        raise _refuse(f"{where}.id {action_id!r} is not an identifier ({_ID.pattern})")
    note = raw["note"]
    if not isinstance(note, str) or not note.strip():
        raise _refuse(
            f"{where} ({action_id}) has no note. The CLI refuses an operator action "
            "with no stated reason, and so does its replay"
        )
    if note != note.strip():
        raise _refuse(
            f"{where} ({action_id}) has a note with surrounding whitespace; the runner "
            "records the stripped note, so the file must hold exactly that"
        )
    return OperatorAction(
        id=action_id,
        after_minute=_minute(raw["after_minute"], where),
        command=command,
        note=note,
    )


def parse_operator_actions(text: str) -> OperatorActions:
    """Validate an operator-action file's text, fail-closed."""
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise _refuse(f"not JSON ({exc})") from None
    if not isinstance(document, Mapping):
        raise _refuse("the document is not an object")
    schema = document.get("schema")
    if schema != OPERATOR_ACTIONS_SCHEMA:
        raise _refuse(f"schema {schema!r} is not {OPERATOR_ACTIONS_SCHEMA!r}")
    unknown = sorted(set(document) - _DOCUMENT_KEYS)
    if unknown:
        raise _refuse(f"unknown top-level key(s) {unknown}")
    raw_actions = document.get("actions")
    if not isinstance(raw_actions, list):
        raise _refuse("actions must be a list")
    if text != canonical_text(document):
        raise _refuse(
            "the file is not in canonical form (json.dumps(indent=2, sort_keys=True, "
            "ensure_ascii=True) plus one newline). One set of actions has one byte "
            "form, so its hash identifies it"
        )
    actions = tuple(_action(raw, index) for index, raw in enumerate(raw_actions))
    seen_ids: set[str] = set()
    seen_anchor: set[tuple[str, str]] = set()
    previous: OperatorAction | None = None
    for action in actions:
        if action.id in seen_ids:
            raise _refuse(f"the action id {action.id!r} appears twice")
        seen_ids.add(action.id)
        if (action.after_minute, action.command) in seen_anchor:
            raise _refuse(
                f"{action.command} appears twice after {action.after_minute}; one "
                "command is run once per anchor"
            )
        seen_anchor.add((action.after_minute, action.command))
        if previous is not None and action.after_minute_ms < previous.after_minute_ms:
            raise _refuse(
                f"{action.id} ({action.after_minute}) comes after {previous.id} "
                f"({previous.after_minute}); actions are listed in the order they "
                "were applied, and that order is the file order"
            )
        previous = action
    digest = hashlib.sha256(text.encode("ascii")).hexdigest()
    return OperatorActions(actions=actions, file_hash=f"sha256:{digest}")


def load_operator_actions(path: str | Path) -> OperatorActions:
    """Read and validate a committed operator-action file."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise _refuse(f"cannot read {path}: {exc.strerror or exc}") from None
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        raise _refuse("the file is not ASCII; its canonical form is") from None
    return parse_operator_actions(text)
