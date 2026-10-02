"""R1-n's off-host dead-man check: confirm life, and treat silence as death.

    python -m tools.deadman_check \
        --recorder http://recorder-host:9102/metrics \
        --runner   http://runner-host:9103/metrics

WHY THIS EXISTS AND WHY IT RUNS SOMEWHERE ELSE. ``conf/alerts_demo.yml`` already
has ``RecorderDown`` and ``RunnerDown``, and they are good alerts. They are also
evaluated by the Prometheus in ``docker-compose.yml``, which runs on the same
host as the two processes it watches. That arrangement reports a dead process
and cannot report a dead host: when the machine goes, the recorder, the runner,
Prometheus and Alertmanager go with it, and the silence that follows is
indistinguishable from a quiet night. Section 11 of the runbook is explicit that
a failed scrape makes Prometheus mark a series stale -- which is exactly why the
evaluator must not share the host's fate.

So this check is deliberately small, has no dependency outside the standard
library, and is meant to run from a different machine on a timer. It asks one
question -- "did both processes beat recently?" -- and it FAILS CLOSED: every
answer other than a fresh heartbeat is reported as dead, including the answers
that mean "I do not know". A dead-man switch that stayed quiet when it could not
reach the host would be a switch that only works when nothing is wrong.

THE FIVE VERDICTS, per target:

* ``alive`` -- the heartbeat is present and younger than the threshold;
* ``stale`` -- present, older than the threshold: the process is wedged;
* ``unreachable`` -- the endpoint did not answer: the process, the port or the
  whole host is gone;
* ``no_such_series`` -- the endpoint answered but published no heartbeat: this
  is a metrics endpoint that is not the one we think it is, or a process that
  stopped publishing. Not evidence of life, so not treated as life;
* ``clock_disagreement`` -- the heartbeat is stamped in the checker's future by
  more than the tolerance. Freshness is a subtraction between two hosts' clocks,
  so if they disagree this check cannot judge age at all, and saying ``alive``
  would be saying something it does not know. This is the executable half of
  R1-n's chrony requirement: the recorder's own
  ``chimera_recorder_clock_skew_ms`` compares the recorder with the exchange,
  and nothing compared the recorder with the observer.

WHAT IT IS NOT. Not a replacement for the Prometheus alerts: they carry
staleness, disk, gap and halt detail this check has no opinion about. Not a
liveness probe for anything that decides -- it touches no state directory, no
ledger and no decision log, and it cannot resume, flatten or resolve anything.
It reads two HTTP endpoints and exits.

EXIT CODES: ``0`` both targets alive; ``1`` at least one not demonstrably alive;
``2`` usage error. Each target prints one line, so the cron mail or the alert
body says which host and which reason.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

#: The two series this check reads, by the names :mod:`chimera.metrics` exports.
#: Pinned here rather than passed in, so a rename breaks a test instead of
#: quietly turning every verdict into ``no_such_series``.
RECORDER_SERIES = "chimera_recorder_heartbeat_timestamp"
RUNNER_SERIES = "chimera_demo_heartbeat_timestamp"

#: ``RecorderDown`` and ``RunnerDown`` both fire at 120 s. The same number here,
#: so an operator does not have to hold two thresholds in their head.
DEFAULT_MAX_AGE_S = 120.0

#: How far into the checker's future a heartbeat may be stamped before the two
#: clocks are called irreconcilable. Generous on purpose: a second of drift
#: between two NTP-synced hosts is normal, half a minute is not.
DEFAULT_FUTURE_TOLERANCE_S = 30.0

#: Seconds to wait for an endpoint. Short: an endpoint that cannot answer in
#: five seconds is not answering, and a dead-man that hangs is not a dead-man.
DEFAULT_TIMEOUT_S = 5.0

ALIVE = "alive"
STALE = "stale"
UNREACHABLE = "unreachable"
NO_SUCH_SERIES = "no_such_series"
CLOCK_DISAGREEMENT = "clock_disagreement"

#: Verdicts that are not "this process beat recently". Named rather than
#: computed as "not ALIVE" so that adding a verdict forces a decision about
#: which side of the switch it falls on.
NOT_ALIVE: tuple[str, ...] = (STALE, UNREACHABLE, NO_SUCH_SERIES, CLOCK_DISAGREEMENT)


@dataclass(frozen=True)
class Target:
    """One process to ask about: a human name, a URL, and the series to read."""

    name: str
    url: str
    series: str


@dataclass(frozen=True)
class Reading:
    """What one target answered, and what that means."""

    target: Target
    verdict: str
    age_s: float | None = None
    detail: str | None = None

    @property
    def alive(self) -> bool:
        return self.verdict == ALIVE

    def line(self) -> str:
        """One line for a cron mail or an alert body."""
        parts = [f"{self.target.name}: {self.verdict.upper()}"]
        if self.age_s is not None:
            parts.append(f"heartbeat {self.age_s:.1f}s old")
        if self.detail:
            parts.append(self.detail)
        parts.append(self.target.url)
        return " -- ".join(parts)


def fetch_text(url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    """GET a metrics endpoint. Raises on anything that is not a body."""
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def parse_gauge(exposition: str, series: str) -> float | None:
    """The value of an unlabelled gauge in Prometheus text exposition.

    ``None`` when the series is absent. Comment lines are skipped, so the
    ``# HELP`` and ``# TYPE`` lines that mention the name do not count as a
    sample -- a subtlety worth stating, because an endpoint that declares a
    series it never sets would otherwise read as a value.
    """
    pattern = re.compile(
        r"^" + re.escape(series) + r"(?:\{\})?\s+([0-9eE+\-.]+|NaN|\+Inf|-Inf)\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(exposition):
        raw = match.group(1)
        try:
            value = float(raw)
        except ValueError:
            return None
        if value != value:  # NaN: the client's way of saying "never set"
            return None
        return value
    return None


def read_target(
    target: Target,
    now_s: float,
    *,
    fetch: Callable[[str], str],
    max_age_s: float = DEFAULT_MAX_AGE_S,
    future_tolerance_s: float = DEFAULT_FUTURE_TOLERANCE_S,
) -> Reading:
    """Ask one target and classify the answer.

    ``now_s`` and ``fetch`` are injected rather than taken from the module, for
    the same reason R1-e injects the runner's clocks: a test that cannot choose
    the time and the response can only assert what today happens to produce.
    """
    try:
        body = fetch(target.url)
    except Exception as exc:  # urllib raises a small zoo; all of them mean "no"
        return Reading(target, UNREACHABLE, detail=f"{type(exc).__name__}: {exc}")

    stamp = parse_gauge(body, target.series)
    if stamp is None:
        return Reading(
            target,
            NO_SUCH_SERIES,
            detail=f"{target.series} is not published by this endpoint",
        )

    age = now_s - stamp
    if age < -future_tolerance_s:
        return Reading(
            target,
            CLOCK_DISAGREEMENT,
            age_s=age,
            detail=(
                f"the heartbeat is {-age:.1f}s in this host's future, past the "
                f"{future_tolerance_s:.0f}s tolerance; age cannot be judged until "
                "both hosts agree on the time (chrony on both)"
            ),
        )
    if age > max_age_s:
        return Reading(target, STALE, age_s=age, detail=f"threshold {max_age_s:.0f}s")
    return Reading(target, ALIVE, age_s=age)


def assess(
    targets: Iterable[Target],
    now_s: float,
    *,
    fetch: Callable[[str], str],
    max_age_s: float = DEFAULT_MAX_AGE_S,
    future_tolerance_s: float = DEFAULT_FUTURE_TOLERANCE_S,
) -> list[Reading]:
    """Every target's reading, in the order given. Asks all of them.

    No short-circuit on the first dead target: an operator woken at 03:00 wants
    to know whether one process died or the whole host did, and that difference
    is the second reading.
    """
    return [
        read_target(
            target,
            now_s,
            fetch=fetch,
            max_age_s=max_age_s,
            future_tolerance_s=future_tolerance_s,
        )
        for target in targets
    ]


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a factory.

    Separate from :func:`main` by this repository's convention: the runbook test
    reaches every documented command's parser by name and feeds the command to
    it, so a flag that drifted apart from the runbook fails in CI rather than at
    03:00 on the host that cannot be reached.
    """
    parser = argparse.ArgumentParser(
        description="Off-host dead-man check for the recorder and the demo runner.",
    )
    parser.add_argument(
        "--recorder",
        metavar="URL",
        help="the recorder's metrics endpoint, e.g. http://host:9102/metrics",
    )
    parser.add_argument(
        "--runner",
        metavar="URL",
        help="the demo runner's metrics endpoint, e.g. http://host:9103/metrics",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=float,
        default=DEFAULT_MAX_AGE_S,
        help=f"a heartbeat older than this is stale (default {DEFAULT_MAX_AGE_S:.0f})",
    )
    parser.add_argument(
        "--future-tolerance-seconds",
        type=float,
        default=DEFAULT_FUTURE_TOLERANCE_S,
        help=(
            "a heartbeat stamped further than this into the future means the two "
            f"hosts' clocks disagree (default {DEFAULT_FUTURE_TOLERANCE_S:.0f})"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"per-endpoint HTTP timeout (default {DEFAULT_TIMEOUT_S:.0f})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    targets: list[Target] = []
    if args.recorder:
        targets.append(Target("recorder", args.recorder, RECORDER_SERIES))
    if args.runner:
        targets.append(Target("runner", args.runner, RUNNER_SERIES))
    if not targets:
        # Not a silent pass: a dead-man invoked with nothing to watch is a
        # misconfigured dead-man, and reporting "all alive" would be the worst
        # possible answer.
        parser.error("give at least one of --recorder or --runner")

    readings = assess(
        targets,
        time.time(),
        fetch=lambda url: fetch_text(url, timeout_s=args.timeout_seconds),
        max_age_s=args.max_age_seconds,
        future_tolerance_s=args.future_tolerance_seconds,
    )
    dead = [reading for reading in readings if not reading.alive]
    for reading in readings:
        print(reading.line(), file=sys.stderr if not reading.alive else sys.stdout)
    if dead:
        print(
            f"DEAD-MAN TRIPPED: {len(dead)} of {len(readings)} target(s) not "
            "demonstrably alive",
            file=sys.stderr,
        )
        return 1
    print(f"dead-man clear: {len(readings)} target(s) alive")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
