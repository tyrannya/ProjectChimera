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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "MUST_MATCH",
    "OPERATIONAL_KINDS",
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
#: skipped a minute, is not a parity pass.
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

#: Aligned by minute and kind, and otherwise not compared field by field.
#: Section 10: "the replay may have fewer restarts".
OPERATIONAL_KINDS: frozenset[str] = frozenset(
    {"STARTUP", "SHUTDOWN", "RECOVERY", "HALT", "RESUME"}
)

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


def _render_key(key: tuple[str, str, int]) -> str:
    """One alignment key, for a human. The ordinal shows only when it matters."""
    minute, kind, ordinal = key
    return f"{kind} at {minute}" if ordinal == 0 else f"{kind} #{ordinal + 1} at {minute}"


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
) -> ParityReport:
    """Compare two decision logs under section 10's rules.

    Operational kinds are aligned by ``(minute, kind)`` and their presence is
    checked rather than their contents, because "the replay may have fewer
    restarts". Everything else is compared field by field over the must-match
    list, and the FIRST divergence is what the tool reports -- a list of
    hundreds of downstream differences from one upstream cause is noise.
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

    def comparable(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [
            r
            for r in records
            if str(r.get("kind")) not in OPERATIONAL_KINDS
            and str(r.get("minute", "")) not in excluded
        ]

    live_decisions = comparable(live)
    replay_decisions = comparable(replay)

    live_keys = _keys(live_decisions)
    replay_keys = _keys(replay_decisions)
    replay_by_key = dict(zip(replay_keys, replay_decisions))
    live_by_key = dict(zip(live_keys, live_decisions))

    for key in live_keys:
        if key not in replay_by_key:
            report.live_only.append(_render_key(key))
    for key in replay_keys:
        if key not in live_by_key:
            report.replay_only.append(_render_key(key))

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
            a, b = original.get(name), replayed.get(name)
            if a != b:
                report.divergences.append(Divergence(index, key[0], key[1], name, a, b))
    return report


def _excluded_minutes(live: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Minutes section 10 lets a replay decide differently, and why, by name.

    Section 10's failure criterion ends "a divergence caused by a
    ``LOG_BEHIND_STATE`` minute is explained and excluded once". Two conditions
    produce a live minute a replay cannot be asked to reproduce, and both are
    facts about the RUNNING of the campaign rather than about the files it read:

    ``SKIPPED_STALE``
        The live process came back from an outage and the minute was already too
        old to decide. A replay reads the whole range at once and is never late,
        so it decides that minute. Which minutes were stale depends on when the
        process restarted, and nothing in the recorded files records it.

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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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
    runner = _load(namespace)
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
    runner.replay(first, max(first, last))
    runner.shutdown("replay parity")

    replay_records = read_log(replay_state / "decision_log", args.days)
    report = compare_logs(live_records, replay_records)

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
