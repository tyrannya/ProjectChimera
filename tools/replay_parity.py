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


def _key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("minute", "")), str(record.get("kind", ""))


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

    live_decisions = [r for r in live if str(r.get("kind")) not in OPERATIONAL_KINDS]
    replay_decisions = [r for r in replay if str(r.get("kind")) not in OPERATIONAL_KINDS]

    live_keys = [_key(r) for r in live_decisions]
    replay_keys = [_key(r) for r in replay_decisions]
    replay_by_key = {k: r for k, r in zip(replay_keys, replay_decisions)}
    live_by_key = {k: r for k, r in zip(live_keys, live_decisions)}

    for key in live_keys:
        if key not in replay_by_key:
            report.live_only.append(f"{key[1]} at {key[0]}")
    for key in replay_keys:
        if key not in live_by_key:
            report.replay_only.append(f"{key[1]} at {key[0]}")

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

    from tools.demo_run import _load, _software  # noqa: F401 - reused, not duplicated

    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload.setdefault("runner", {})
    payload["runner"] = {**payload["runner"], "state_dir": str(replay_state)}
    patched = Path(scratch) / "replay_config.json"
    patched.write_text(json.dumps(payload), encoding="utf-8")

    namespace = argparse.Namespace(config=patched, root=scratch, profile=args.profile)
    runner = _load(namespace)
    runner.start(allow_dirty=True)
    first = runner.cursor.next_minute_ms()
    if first is None:
        print("the scratch copy carries no minutes", file=sys.stderr)
        return EXIT_REFUSED
    minutes = [r for r in live_records if r.get("kind") not in OPERATIONAL_KINDS]
    runner.replay(first, first + (len(minutes) + 1) * 60_000)
    runner.shutdown("replay parity")

    replay_records = read_log(replay_state / "decision_log", args.days)
    report = compare_logs(live_records, replay_records)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"{report.label}: {'PARITY' if report.ok else 'DIVERGED'}")
        print(f"  records compared : {report.compared}")
        print(f"  divergences      : {len(report.divergences)}")
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
