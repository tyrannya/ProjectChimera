"""Section 10's replay parity tool: does a replay decide what the live run decided?

Given a date range, a campaign config and a storage root, this
  1. copies the recorder's normalized and funding files for the range into a
     scratch directory, so the replay reads exactly what the live run read and
     cannot be affected by anything written since;
  2. runs `DemoRunner` in replay mode with an EMPTY state directory and a
     `RunnerClock` fed from the recorded minutes, not the wall clock;
  3. compares the replay's decision log with the live log for the same range.

Section 10 fixes what must match and what may not:

    must match byte-for-byte
        inputs, rule, signal, requested_action, risk.decisions, execution
        (order ids, event ids, quantities, prices, fees), position_after,
        ledger_effect, veto_or_rejection, minute, kind, seq

    compared, and excluded only under a declared difference
        runner_now_ns -- "compared with tolerance zero because it is derived
        from recorded receipt stamps, not the wall clock". Tolerance zero means
        it is compared exactly; it is listed in the plan's right-hand column but
        the text is what governs, and treating it as free would drop the very
        field that proves the clock is not the wall clock.

        software.python / software.libs -- may differ ONLY if the parity run
        declares a different environment, in which case the whole run is
        labelled ENVIRONMENT_PARITY and reported separately rather than being
        quietly counted as a pass.

    semantic rather than byte-for-byte
        STARTUP, SHUTDOWN and RECOVERY: "the replay may have fewer restarts;
        the comparison aligns by minute and kind and ignores operational kinds".

R1-j decides the two questions section 10 left open (see ``docs/replay_parity.md``):

    ``seq`` -- exclusion of operational kinds. The persisted ``seq`` keeps its
        meaning: one counter over every record of the log, which ``request_seq``,
        the hash chain, R1-c and the reports read. What parity compares is a
        DERIVED ``parity_seq``: a record's 1-based position among the comparable
        records of its own log, in log order. Operational records do not count,
        so a restart shifts no ``parity_seq``, while a reordered, dropped or added
        comparable record still does. Each side's raw ``seq`` must still rise
        strictly through its records.

    ``OPERATOR`` -- a deterministic replay counterpart from a committed
        operator-action file (``chimera.demo.operator_actions``). ``OPERATOR``
        and ``RESUME`` are operator-semantic: compared like every other
        comparable kind, and their ``operator`` block too.

**A parity failure is never repaired here.** The tool prints the first divergent
record and both versions and exits non-zero. Section 10's failure criterion is
"any mismatch in a must-match field", and a tool that could paper over one would
be worse than no tool: the whole point is that the decision log is reproducible
from the files it names.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "MUST_MATCH",
    "OPERATIONAL_KINDS",
    "OPERATOR_KINDS",
    "SEQ_POLICY",
    "OperatorActionRefused",
    "replay_with_actions",
    "EXCLUDING_KINDS",
    "EXCLUDING_RECOVERY_CAUSES",
    "ParityReport",
    "Divergence",
    "compare_logs",
    "read_log",
    "main",
]

#: Section 10's must-match list, verbatim. `seq` and `minute` and `kind` are
#: included: a replay that decided the same things in a different order, or
#: skipped a minute, is not a parity pass. Since R1-j ``seq`` is compared under
#: :data:`SEQ_POLICY`, as the derived ``parity_seq``, and reported under that
#: name; the persisted ``seq`` is never renumbered.
MUST_MATCH: tuple[str, ...] = (
    "kind",
    "minute",
    "seq",
    "runner_now_ns",
    "inputs",
    "rule",
    "signal",
    "requested_action",
    "risk",
    "execution",
    "position_after",
    "ledger_effect",
    "veto_or_rejection",
)

#: Left out of the comparison: neither their presence nor their contents is
#: compared, and they do not count towards ``parity_seq``. Section 10: "the
#: replay may have fewer restarts".
#:
#: R1-j's reading of each, and what still has to be reproduced:
#:
#: ``STARTUP`` / ``SHUTDOWN`` / ``RECOVERY`` -- how often the process started,
#:     stopped or recovered. A replay starts once. A restart moves no decision
#:     state, which the next comparable record's ``risk`` block proves. A
#:     crash's damaged minute is still named, by ``EXCLUDING_KINDS``.
#: ``HALT`` -- a halted campaign writes one again at every restart, so the
#:     count is operational. The halt itself is not invisible: no minute is
#:     decided while it stands (a missing ``DECISION`` diverges), ``halted`` and
#:     ``halt_reason`` are in the next ``risk`` hash, and a ``resume`` names what
#:     it cleared in its compared ``operator`` block.
#: ``FEED_STALLED`` / ``FEED_RESUMED`` -- when the LIVE service found the feed
#:     stale, a fact about the running of the campaign. A replay reads finished
#:     files and is never stale, so it writes neither.
#:
#: ``RESUME`` left this set in R1-j: it is the completion of an operator
#: ``resume`` and is compared with ``OPERATOR`` (:data:`OPERATOR_KINDS`).
OPERATIONAL_KINDS: frozenset[str] = frozenset(
    {"STARTUP", "SHUTDOWN", "RECOVERY", "HALT", "FEED_STALLED", "FEED_RESUMED"}
)

#: R1-j: intentional state mutation by an operator. Compared like every other
#: comparable kind, and their ``operator`` block as well (command, note, phase,
#: what the command saw or cleared, and the request/completion link). A replay
#: produces them only from a committed operator-action file, so a live action
#: the file omits is ``live_only`` and an extra one is ``replay_only``.
OPERATOR_KINDS: frozenset[str] = frozenset({"OPERATOR", "RESUME"})

#: R1-j's ``seq`` policy, by name, as the report states it.
SEQ_POLICY = (
    "chimera.parity-seq/1: operational kinds excluded; seq compared as parity_seq, "
    "the record's 1-based position among its log's comparable records"
)

#: The key a replayed operator record carries naming its committed action
#: (`chimera.demo.runner.REPLAY_ACTION_KEY`). Required on the replay side,
#: forbidden on the live side, and otherwise the only key of an ``operator``
#: block not compared.
REPLAY_ACTION_KEY = "replay_action"

#: Kinds whose presence in the LIVE log can exclude their minute from the
#: comparison, with the reason reported. This is not a third kind-set with a
#: third meaning: it is section 10's own "explained and excluded once", made
#: executable. :func:`_excluded_minutes` READS it, so widening or narrowing this
#: set changes what the tool does rather than only what it says it does.
EXCLUDING_KINDS: frozenset[str] = frozenset({"SKIPPED_STALE", "RECOVERY"})

#: The RECOVERY causes whose affected minute has no comparable record in the live
#: log, and only those.
#:
#: ``LOG_BEHIND_STATE`` is section 9.3's own -- the record was never written, so
#: there is nothing to compare -- and ``TORN_TAIL``'s record was never committed.
#:
#: ``LOG_AHEAD_OF_STATE`` is deliberately absent, and is handled by the extra
#: condition in :func:`_excluded_minutes` instead. There the tail record WAS
#: committed, but that only settles the minute when the tail is the record that
#: FINISHED it. A tail of FUNDING, RECONCILIATION or LIQUIDATION_TOUCH is written
#: for its minute before the minute is decided, so such a minute holds part of
#: its evidence and no decision, and the live runner does not decide it again.
#: The unconditional rule dropped real evidence for a finished minute; no rule at
#: all left an unfinished one to diverge as an unexplainable `replay_only`.
EXCLUDING_RECOVERY_CAUSES: frozenset[str] = frozenset({"LOG_BEHIND_STATE", "TORN_TAIL"})


class OperatorActionRefused(RuntimeError):
    """A committed operator action the replay could not perform. Never skipped."""


EXIT_PARITY = 0
EXIT_DIVERGED = 1
EXIT_REFUSED = 2

#: Section 10's label for a run whose environment differs. Reported separately;
#: never silently counted as a pass.
ENVIRONMENT_PARITY = "ENVIRONMENT_PARITY"


@dataclass(frozen=True)
class Divergence:
    """One difference, in one field, of one record."""

    index: int
    minute: str
    kind: str
    field_name: str
    live: Any
    replay: Any

    def render(self) -> str:
        return (
            f"record {self.index} ({self.kind} at {self.minute}) field {self.field_name!r}\n"
            f"  live  : {json.dumps(self.live, sort_keys=True, default=str)[:600]}\n"
            f"  replay: {json.dumps(self.replay, sort_keys=True, default=str)[:600]}"
        )


@dataclass
class ParityReport:
    """What the comparison found. Never a bare boolean."""

    compared: int = 0
    divergences: list[Divergence] = field(default_factory=list)
    live_only: list[str] = field(default_factory=list)
    replay_only: list[str] = field(default_factory=list)
    label: str = "PARITY"
    environment_note: str = ""
    #: Minutes section 10 allows a replay to decide differently, each with the
    #: reason. Never empty silently: a run with no exclusions reports none, and a
    #: run with any lists every one. See :func:`_excluded_minutes`.
    explained_exclusions: list[str] = field(default_factory=list)
    #: R1-j: the committed operator-action file the replay applied, by hash, or
    #: None when the run was given none.
    operator_actions: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.divergences and not self.live_only and not self.replay_only

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "parity": self.ok,
            "records_compared": self.compared,
            "divergences": len(self.divergences),
            "live_only": self.live_only,
            "replay_only": self.replay_only,
            "environment_note": self.environment_note,
            "explained_exclusions": list(self.explained_exclusions),
            "first_divergence": (self.divergences[0].render() if self.divergences else None),
            "seq_policy": SEQ_POLICY,
            "operator_actions": self.operator_actions,
        }


def read_log(log_dir: Path, days: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Every record in a decision log directory, in file and line order."""
    records: list[dict[str, Any]] = []
    paths = sorted(Path(log_dir).glob("*.ndjson"))
    for path in paths:
        if days is not None and path.stem not in days:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _keys(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, int]]:
    """Alignment keys: ``(minute, kind, ordinal)``, in log order.

    The ordinal is how many records of that kind this minute has already
    produced, and it exists because a minute may produce more than one record of
    a kind. Catching up over a settlement boundary can book two FUNDING
    settlements in one minute; a mismatch can produce a RECONCILIATION and then
    a HALT.

    Without it the comparison keyed on ``(minute, kind)`` into a dict, so the
    LAST record of a repeated key overwrote the earlier ones on both sides. The
    consequences were all silent: the earlier records were never compared to
    anything, so a replay could differ on them freely; a replay that emitted
    FEWER of them produced no ``replay_only`` entry, because the key was still
    present; and ``compared`` counted the duplicates, so the "the comparison must
    not be vacuous" guard went up rather than down. A parity proof that cannot
    see a dropped settlement is not a proof.

    With one record per ``(minute, kind)`` -- which is every case before
    PR-10R -- every ordinal is 0 and the keys are exactly the old ones.
    """
    seen: dict[tuple[str, str], int] = {}
    keys: list[tuple[str, str, int]] = []
    for record in records:
        base = (str(record.get("minute", "")), str(record.get("kind", "")))
        ordinal = seen.get(base, 0)
        seen[base] = ordinal + 1
        keys.append((base[0], base[1], ordinal))
    return keys


def _render_key(key: tuple[str, str, int], record: Mapping[str, Any] | None = None) -> str:
    """One alignment key, for a human. The ordinal shows only when it matters.

    An operator record also names its command and phase, so a missing or extra
    operator action reads as one.
    """
    minute, kind, ordinal = key
    text = f"{kind} at {minute}" if ordinal == 0 else f"{kind} #{ordinal + 1} at {minute}"
    operator = (record or {}).get("operator")
    if kind in OPERATOR_KINDS and isinstance(operator, Mapping):
        text += f" (operator {operator.get('command')} {operator.get('phase')})"
    return text


def _environment_differs(
    live: Sequence[Mapping[str, Any]], replay: Sequence[Mapping[str, Any]]
) -> str:
    """Whether the two runs declare different Python or library versions."""

    def declared(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        for record in records:
            software = record.get("software")
            if isinstance(software, Mapping):
                return {
                    "python": software.get("python"),
                    "libs": software.get("libs"),
                }
        return {}

    a, b = declared(live), declared(replay)
    if a and b and a != b:
        return (
            f"live {json.dumps(a, sort_keys=True)} vs replay {json.dumps(b, sort_keys=True)}"
        )
    return ""


def compare_logs(
    live: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    *,
    must_match: Iterable[str] = MUST_MATCH,
    operator_file_hash: str | None = None,
) -> ParityReport:
    """Compare two decision logs under section 10's rules and R1-j's policy.

    Operational kinds are left out, because "the replay may have fewer
    restarts". Everything else is compared field by field over the must-match
    list, with ``seq`` compared as ``parity_seq`` (:data:`SEQ_POLICY`), and the
    FIRST divergence is what the tool reports -- a list of hundreds of downstream
    differences from one upstream cause is noise.

    Operator records (:data:`OPERATOR_KINDS`) are compared on their ``operator``
    block too, with ``request_seq`` read as the ``parity_seq`` of the request it
    names in its own log. ``operator_file_hash`` is the hash of the committed
    operator-action file the replay applied; every replayed operator record must
    name it, and no live one may carry a replay attribution at all.
    """
    report = ParityReport()
    note = _environment_differs(live, replay)
    if note:
        report.label = ENVIRONMENT_PARITY
        report.environment_note = note

    excluded = _excluded_minutes(live)
    report.explained_exclusions = [
        f"{minute}: {reason}" for minute, reason in sorted(excluded.items())
    ]

    report.divergences.extend(_chronology(live, "live"))
    report.divergences.extend(_chronology(replay, "replay"))

    def comparable(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [
            r
            for r in records
            if str(r.get("kind")) not in OPERATIONAL_KINDS
            and str(r.get("minute", "")) not in excluded
        ]

    live_decisions = comparable(live)
    replay_decisions = comparable(replay)
    live_pseq = {id(r): n for n, r in enumerate(live_decisions, start=1)}
    replay_pseq = {id(r): n for n, r in enumerate(replay_decisions, start=1)}

    live_keys = _keys(live_decisions)
    replay_keys = _keys(replay_decisions)
    replay_by_key = dict(zip(replay_keys, replay_decisions))
    live_by_key = dict(zip(live_keys, live_decisions))

    for key in live_keys:
        if key not in replay_by_key:
            report.live_only.append(_render_key(key, live_by_key[key]))
    for key in replay_keys:
        if key not in live_by_key:
            report.replay_only.append(_render_key(key, replay_by_key[key]))

    live_operator = _OperatorView(live, live_pseq, side="live", file_hash=None)
    replay_operator = _OperatorView(
        replay, replay_pseq, side="replay", file_hash=operator_file_hash
    )

    fields = tuple(must_match)
    for index, key in enumerate(live_keys):
        replayed = replay_by_key.get(key)
        if replayed is None:
            continue
        original = live_by_key[key]
        report.compared += 1
        for name in fields:
            if name in ("software",):  # handled by the environment label
                continue
            if name == "seq":
                a = {"parity_seq": live_pseq[id(original)], "seq": original.get("seq")}
                b = {"parity_seq": replay_pseq[id(replayed)], "seq": replayed.get("seq")}
                if a["parity_seq"] != b["parity_seq"]:
                    report.divergences.append(
                        Divergence(index, key[0], key[1], "parity_seq", a, b)
                    )
                continue
            a, b = original.get(name), replayed.get(name)
            if a != b:
                report.divergences.append(Divergence(index, key[0], key[1], name, a, b))
        if key[1] in OPERATOR_KINDS:
            a = live_operator.normalised(original)
            b = replay_operator.normalised(replayed)
            if a != b:
                report.divergences.append(Divergence(index, key[0], key[1], "operator", a, b))
    for side_view, records in (
        (live_operator, live_decisions),
        (replay_operator, replay_decisions),
    ):
        for record in records:
            problem = side_view.attribution_problem(record)
            if problem is not None:
                report.divergences.append(
                    Divergence(
                        -1,
                        str(record.get("minute", "")),
                        str(record.get("kind", "")),
                        f"operator.{REPLAY_ACTION_KEY}",
                        problem if side_view.side == "live" else None,
                        problem if side_view.side == "replay" else None,
                    )
                )
    return report


def _chronology(records: Sequence[Mapping[str, Any]], side: str) -> list[Divergence]:
    """R1-j: the persisted ``seq`` must rise strictly through a log's records.

    ``parity_seq`` is derived from log order, so log order has to be the chain's
    order: a record out of place, or a ``seq`` that is not an integer, is
    reported rather than re-sorted. (The chain itself is the decision log's to
    verify; this is the one property the derived sequence depends on.)
    """
    found: list[Divergence] = []
    previous: int | None = None
    for index, record in enumerate(records):
        seq = record.get("seq")
        if (
            not isinstance(seq, int)
            or isinstance(seq, bool)
            or (previous is not None and seq <= previous)
        ):
            detail = {"seq": seq, "previous_seq": previous}
            found.append(
                Divergence(
                    index,
                    str(record.get("minute", "")),
                    str(record.get("kind", "")),
                    "seq",
                    detail if side == "live" else None,
                    detail if side == "replay" else None,
                )
            )
        if isinstance(seq, int) and not isinstance(seq, bool):
            previous = seq
    return found


class _OperatorView:
    """One log's operator records, read under R1-j's correspondence rules."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        pseq: Mapping[int, int],
        *,
        side: str,
        file_hash: str | None,
    ) -> None:
        self.side = side
        self.file_hash = file_hash
        self.pseq = pseq
        self.by_seq = {
            int(r["seq"]): r
            for r in records
            if isinstance(r.get("seq"), int) and not isinstance(r.get("seq"), bool)
        }

    def normalised(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """The ``operator`` block as compared.

        ``request_seq`` names a record by its PERSISTED ``seq``, which a restart
        shifts, so it is compared as what it resolves to in its own log: the
        ``parity_seq`` of a ``requested`` OPERATOR record, or ``unresolved`` with
        the raw value when it names nothing of the kind. The replay attribution
        is set aside here and checked by :meth:`attribution_problem`.
        """
        block = record.get("operator")
        if not isinstance(block, Mapping):
            return {"operator": block}
        out = {k: v for k, v in block.items() if k != REPLAY_ACTION_KEY}
        if "request_seq" in out:
            raw = out["request_seq"]
            target = self.by_seq.get(raw) if isinstance(raw, int) else None
            requested = (
                target is not None
                and str(target.get("kind")) == "OPERATOR"
                and isinstance(target.get("operator"), Mapping)
                and target["operator"].get("phase") == "requested"
                and id(target) in self.pseq
            )
            out["request_seq"] = (
                {"parity_seq": self.pseq[id(target)]} if requested else {"unresolved": raw}
            )
        return out

    def attribution_problem(self, record: Mapping[str, Any]) -> str | None:
        """Why a record's replay attribution is wrong for its side, or None."""
        if str(record.get("kind")) not in OPERATOR_KINDS:
            return None
        block = record.get("operator")
        attribution = block.get(REPLAY_ACTION_KEY) if isinstance(block, Mapping) else None
        if self.side == "live":
            if attribution is not None:
                return (
                    "a live operator record carries a replay attribution: this log was "
                    "written by a replay, not by the campaign"
                )
            return None
        if attribution is None:
            return (
                "a replayed operator record names no committed operator action; a replay "
                "performs operator actions only from the committed file"
            )
        if (
            self.file_hash is None
            or not isinstance(attribution, Mapping)
            or (attribution.get("file_hash") != self.file_hash)
        ):
            return (
                f"a replayed operator record names {attribution!r}, not the committed "
                f"operator-action file {self.file_hash!r} this run applied"
            )
        return None


def _excluded_minutes(live: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Minutes section 10 lets a replay decide differently, and why, by name.

    Section 10's failure criterion ends "a divergence caused by a
    ``LOG_BEHIND_STATE`` minute is explained and excluded once". Two conditions
    produce a live minute a replay cannot be asked to reproduce, and both are
    facts about the RUNNING of the campaign rather than about the files it read:

    ``SKIPPED_STALE``
        Written only by builds before R1-g, which retired the catch-up cap; kept
        so their logs still compare. The live process came back from an outage
        and the minute was already too old to decide. A replay reads the whole
        range at once and is never late, so it decides that minute. Which
        minutes were stale depends on when the process restarted, and nothing
        in the recorded files records it.

    ``RECOVERY``
        Section 9.3's own: the affected minute "is excluded from the campaign's
        evidence and counted in the monthly report". The record names it.

    Each exclusion is a MINUTE and it is REPORTED. It is not a record kind
    quietly dropped from the comparison: an excluded minute is listed in
    ``explained_exclusions`` with the reason, appears in the tool's output and in
    the JSON, and a run with none has an empty list. A parity pass that excluded
    something silently would be worth nothing, which is why this returns the
    reasons and not a filter.
    """
    excluded: dict[str, str] = {}
    for record in live:
        kind = str(record.get("kind"))
        if kind not in EXCLUDING_KINDS:
            continue
        minute = str(record.get("minute", ""))
        if not minute:
            continue
        if kind == "SKIPPED_STALE":
            stale = record.get("stale") or {}
            excluded[minute] = (
                "the live run skipped this minute as stale on catch-up "
                f"({stale.get('age_minutes')} minute(s) old, limit "
                f"{stale.get('max_catchup_minutes')}); a replay is never late"
            )
        else:  # RECOVERY
            recovery = record.get("recovery") or {}
            cause = str(recovery.get("cause", ""))
            # `minute_finished` is False only when the runner says the tail record
            # did not finish its minute. The tool still decides for itself: a
            # LOG_AHEAD_OF_STATE record that claims the minute WAS finished can
            # never exclude anything, whatever else it says, so a live log cannot
            # talk this comparison out of reading evidence that exists.
            unfinished = recovery.get("minute_finished") is False
            if cause not in EXCLUDING_RECOVERY_CAUSES and not (
                cause == "LOG_AHEAD_OF_STATE" and unfinished
            ):
                # The record for this minute was committed and it finished the
                # minute, so it is compared like any other.
                continue
            affected = str(recovery.get("evidence_excluded_minute") or minute)
            reason = (
                "the crash left this minute part-recorded and undecided "
                f"({cause}); the live run does not decide it again"
                if unfinished
                else "section 9.3 excludes the minute a crash left inconsistent " f"({cause})"
            )
            excluded[affected] = reason
    return excluded


# ---------------------------------------------------------------------------
# the scratch copy and the replay
# ---------------------------------------------------------------------------
def copy_range(root: Path, scratch: Path, days: Sequence[str]) -> Path:
    """Copy the normalized and funding files for ``days`` into ``scratch``.

    A copy rather than a read of the live tree, so a recorder still writing
    cannot change what the replay sees halfway through -- and so a parity run
    can never write anywhere near the recorder's own files.
    """
    scratch = Path(scratch)
    for market in ("um", "spot"):
        source = Path(root) / "normalized" / market / "1m"
        if not source.is_dir():
            continue
        target = scratch / "normalized" / market / "1m"
        target.mkdir(parents=True, exist_ok=True)
        for day in days:
            for suffix in (".parquet", ".meta.json", ".sha256"):
                candidate = source / f"{day}{suffix}"
                if candidate.is_file():
                    shutil.copy2(candidate, target / candidate.name)
    funding = Path(root) / "funding"
    if funding.is_dir():
        shutil.copytree(funding, scratch / "funding", dirs_exist_ok=True)
    return scratch


def replay_with_actions(
    runner: Any, first_ms: int, last_ms: int, actions: Any = None
) -> list[str]:
    """Replay ``first_ms``..``last_ms`` and apply the committed operator actions.

    Section 10's replay (``DemoRunner.replay``), cut at each action's anchor:
    the minutes up to and including ``after_minute`` are decided, the action is
    performed through `DemoRunner.apply_operator_action` -- the CLI's own
    `flatten` / `resume` -- and the replay continues from the runner's cursor.
    With no actions this is exactly one ``runner.replay(first_ms, last_ms)``.

    Returns the ids of the actions applied, in order. Raises
    :class:`OperatorActionRefused` when an action lies outside the replay's
    range, when the replay never reached its anchor (it halted, or a minute was
    deferred), or when the runner refused to perform it. A committed action is
    never skipped.
    """
    from chimera.demo.runner import RunnerError, RunnerState

    applied: list[str] = []
    listed = list(actions.actions) if actions is not None else []
    for action in listed:
        anchor = action.after_minute_ms
        if not first_ms <= anchor <= last_ms:
            raise OperatorActionRefused(
                f"operator action {action.id!r} is anchored after {action.after_minute}, "
                "outside the replayed range; an action for a minute this replay does not "
                "decide cannot be applied, and it is not skipped"
            )
        upcoming = runner.cursor.next_minute_ms()
        if upcoming is not None and upcoming <= anchor:
            runner.replay(upcoming, anchor)
        reached = runner.cursor.last_minute_processed
        if reached != anchor:
            raise OperatorActionRefused(
                f"operator action {action.id!r} ({action.command}) is anchored after "
                f"{action.after_minute}, and the replay's last decided minute is "
                f"{reached} (state {runner.state.value}): the replay never reached the "
                "anchor, so the action cannot be applied where the file says it was"
            )
        try:
            runner.apply_operator_action(
                action.command, action.note, replay_action=actions.attribution(action)
            )
        except RunnerError as exc:
            raise OperatorActionRefused(
                f"operator action {action.id!r} ({action.command} after "
                f"{action.after_minute}) cannot be performed by the replay: {exc}"
            ) from exc
        applied.append(action.id)
    upcoming = runner.cursor.next_minute_ms()
    if (
        upcoming is not None
        and upcoming <= last_ms
        and runner.state not in (RunnerState.HALT, RunnerState.FEED_STALLED)
    ):
        runner.replay(upcoming, last_ms)
    return applied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay_parity",
        description="Section 10: compare a replay's decision log with the live one.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="the recorder's root")
    parser.add_argument(
        "--live-log", type=Path, required=True, help="the live decision_log dir"
    )
    parser.add_argument("--days", nargs="+", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument(
        "--profile", default="CAMPAIGN", help="which profile the config must declare"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--operator-actions",
        type=Path,
        default=None,
        help=(
            "R1-j: the committed operator-action file (chimera.operator-actions/1) "
            "the replay applies as its OPERATOR counterpart; its sha256 is reported "
            "and stamped on every replayed operator record. Without it the replay "
            "performs no operator action"
        ),
    )
    return parser


def _last_live_minute_ms(records: Sequence[Mapping[str, Any]]) -> int | None:
    """The newest minute the live log names, in epoch milliseconds.

    Read from the records rather than from a count of them, and from every kind:
    a SHUTDOWN at the end of the range still names the minute the campaign
    reached.
    """
    from datetime import datetime

    newest: int | None = None
    for record in records:
        minute = record.get("minute")
        if not isinstance(minute, str) or not minute:
            continue
        try:
            stamp = int(datetime.fromisoformat(minute).timestamp() * 1000)
        except ValueError:
            continue
        if newest is None or stamp > newest:
            newest = stamp
    return newest


def main(
    argv: Sequence[str] | None = None,
    *,
    operational_clock: Callable[[], float] = time.time,
) -> int:
    """The tool's entry point, and so the one place its operational clock enters.

    Like `tools.demo_run.main`, this is a process's composition boundary (R1-e):
    host wall time is the operational clock by definition, and it reaches only
    the replay runner's telemetry. The replay DECIDES on its own `RunnerClock`,
    fed from the recorded minutes -- which is what the comparison of
    ``runner_now_ns`` with tolerance zero checks.
    """
    args = build_parser().parse_args(argv)

    actions = None
    if args.operator_actions is not None:
        from chimera.demo.operator_actions import OperatorActionError, load_operator_actions

        try:
            actions = load_operator_actions(args.operator_actions)
        except OperatorActionError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_REFUSED

    live_records = read_log(args.live_log, args.days)
    if not live_records:
        print(
            f"no live records for {args.days} under {args.live_log}. A parity run over "
            "nothing would report a pass, which is worse than reporting nothing",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    import tempfile

    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="replay-parity-"))
    copy_range(args.root, scratch, args.days)

    replay_state = Path(scratch) / "replay_state"
    replay_state.mkdir(parents=True, exist_ok=True)

    from chimera.demo.runner import RunnerState
    from tools.demo_run import _load, _software  # noqa: F401 - reused, not duplicated

    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload.setdefault("runner", {})
    payload["runner"] = {**payload["runner"], "state_dir": str(replay_state)}
    patched = Path(scratch) / "replay_config.json"
    patched.write_text(json.dumps(payload), encoding="utf-8")

    namespace = argparse.Namespace(config=patched, root=scratch, profile=args.profile)
    runner = _load(namespace, operational_clock=operational_clock)
    # `--allow-dirty` only where SELF_CHECK admits it. It refuses the flag on a
    # CAMPAIGN profile outright ("allow_dirty is for soak runs, whose records are
    # operational rather than evidence"), so passing it unconditionally made the
    # replay leg HALT before its first minute on the tool's own default profile
    # -- and every live record was then reported as `live_only`, a DIVERGED
    # verdict with nothing in the output saying the replay never started.
    #
    # Refusing a dirty tree here is the right answer rather than a limitation: a
    # replay is a reproduction of evidence, and it has to run from the revision
    # that produced it.
    campaign = runner.config.profile.value == "CAMPAIGN"
    state = runner.start(allow_dirty=not campaign)
    if state is RunnerState.HALT:
        print(
            f"the replay runner halted before its first minute: {runner.halt_reason}. "
            "A parity verdict from a replay that never ran would be meaningless",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    first = runner.cursor.next_minute_ms()
    if first is None:
        print("the scratch copy carries no minutes", file=sys.stderr)
        return EXIT_REFUSED
    # The window is the live run's MINUTE RANGE, not a count of its records.
    # Counting records over-runs the range whenever a minute produced more than
    # one -- an incomplete minute, an operator command, and since PR-10R a
    # funding settlement or a reconciliation -- and `DemoRunner.replay` expands
    # whatever it is given into every minute between the two ends, so the replay
    # ticked minutes the live run never saw and reported them as `replay_only`.
    last = _last_live_minute_ms(live_records)
    if last is None:
        print("the live log names no minute to replay", file=sys.stderr)
        return EXIT_REFUSED
    try:
        replay_with_actions(runner, first, max(first, last), actions)
    except OperatorActionRefused as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED
    runner.shutdown("replay parity")

    replay_records = read_log(replay_state / "decision_log", args.days)
    report = compare_logs(
        live_records,
        replay_records,
        operator_file_hash=None if actions is None else actions.file_hash,
    )
    if actions is not None:
        report.operator_actions = actions.to_report(str(args.operator_actions))

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"{report.label}: {'PARITY' if report.ok else 'DIVERGED'}")
        print(f"  records compared : {report.compared}")
        print(f"  divergences      : {len(report.divergences)}")
        # Printed always, including the zero: an exclusion a reader has to go
        # looking for is an exclusion that will be missed.
        print(f"  explained exclusions: {len(report.explained_exclusions)}")
        for line in report.explained_exclusions[:10]:
            print(f"    - {line}")
        if report.environment_note:
            print(f"  environment      : {report.environment_note}")
        print(f"  seq policy       : {SEQ_POLICY}")
        if report.operator_actions is not None:
            print(
                f"  operator actions : {len(report.operator_actions['actions'])} from "
                f"{report.operator_actions['path']} ({report.operator_actions['file_hash']})"
            )
        else:
            print("  operator actions : none supplied")
        if report.live_only:
            print(f"  live only        : {report.live_only[:5]}")
        if report.replay_only:
            print(f"  replay only      : {report.replay_only[:5]}")
        if report.divergences:
            print("\nfirst divergence:")
            print(report.divergences[0].render())
    return EXIT_PARITY if report.ok else EXIT_DIVERGED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
