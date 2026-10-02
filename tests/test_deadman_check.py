"""R1-n's dead-man check, both sides of every verdict.

The switch has one job and one direction of error that matters: it must not say
``alive`` when it does not know. So every test here comes in a pair -- a healthy
input that must be allowed through, and the matching unhealthy one that must
trip it -- and the four unhealthy shapes are deliberately different in kind:
old, unreachable, silent, and stamped in the future.

Nothing here opens a socket. ``read_target`` takes ``fetch`` and ``now_s`` as
arguments precisely so that a test can choose the response and the clock; a test
that could not would be asserting what one machine happened to do on one day.
The exposition bodies below are written out in the Prometheus text format the
``prometheus_client`` endpoint actually serves, including the ``# HELP`` and
``# TYPE`` lines, because one of the parser's jobs is to not mistake a comment
that mentions the series for a sample of it.
"""

from __future__ import annotations

import pytest

from tools.deadman_check import (
    ALIVE,
    CLOCK_DISAGREEMENT,
    NO_SUCH_SERIES,
    NOT_ALIVE,
    RECORDER_SERIES,
    RUNNER_SERIES,
    STALE,
    UNREACHABLE,
    Target,
    assess,
    main,
    parse_gauge,
    read_target,
)

NOW = 1_759_000_000.0

RECORDER = Target("recorder", "http://recorder:9102/metrics", RECORDER_SERIES)
RUNNER = Target("runner", "http://runner:9103/metrics", RUNNER_SERIES)


#: The ``# HELP`` text carries a number because real help strings do. It is NOT
#: what makes the comment-line control below bite, and a mutation campaign is
#: what established that: a parser loosened to match the series name anywhere on
#: a line still returns ``None`` on a comment, because ``[0-9eE+-.]`` matches the
#: ``e`` in "timestamp" long before any digit and ``float("e")`` raises. The
#: implementation fails safe there by construction, so no input distinguishes the
#: two -- which is worth knowing, and is not the same as the tests catching it.
#: What the tests DO catch is the dangerous loosening: a pattern that lets a
#: longer series name match this one. See the prefix test below.
HELP_TEXT = "Unix timestamp (seconds since 1970) of the last heartbeat"


def exposition(series: str, value: str) -> str:
    """One gauge, as the real endpoint serves it."""
    return f"# HELP {series} {HELP_TEXT}\n" f"# TYPE {series} gauge\n" f"{series} {value}\n"


def serving(body: str):
    """A ``fetch`` that answers every URL with ``body``."""
    return lambda _url: body


def refusing(exc: Exception):
    """A ``fetch`` that fails the way a dead host fails."""

    def fetch(_url: str) -> str:
        raise exc

    return fetch


# --------------------------------------------------------------------------- #
# the parser
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1759000000", 1_759_000_000.0),
        ("1759000000.5", 1_759_000_000.5),
        # prometheus_client renders large floats in scientific notation, which
        # is the form this parser will actually meet on the wire.
        ("1.759e+09", 1.759e9),
    ],
)
def test_a_gauge_sample_is_read_in_every_form_the_client_writes(
    value: str, expected: float
) -> None:
    assert parse_gauge(exposition(RECORDER_SERIES, value), RECORDER_SERIES) == expected


def test_an_empty_label_set_is_still_the_same_series() -> None:
    body = f"{RECORDER_SERIES}{{}} 1759000000\n"
    assert parse_gauge(body, RECORDER_SERIES) == 1_759_000_000.0


def test_a_declared_but_never_set_series_is_not_a_value() -> None:
    """The negative control for the parser: comments are not samples.

    An endpoint can declare a gauge it never sets. Reading the ``# HELP`` line
    as a sample would turn a process that publishes nothing into one that
    publishes a heartbeat, which is the exact inversion this switch exists to
    prevent.
    """
    declared_only = (
        f"# HELP {RECORDER_SERIES} {HELP_TEXT}\n" f"# TYPE {RECORDER_SERIES} gauge\n"
    )
    assert parse_gauge(declared_only, RECORDER_SERIES) is None
    assert parse_gauge(exposition(RECORDER_SERIES, "NaN"), RECORDER_SERIES) is None


def test_another_series_is_not_this_one() -> None:
    body = exposition(RUNNER_SERIES, "1759000000")
    assert parse_gauge(body, RECORDER_SERIES) is None
    assert parse_gauge(body, RUNNER_SERIES) == 1_759_000_000.0


def test_a_series_whose_name_is_a_prefix_of_another_is_not_confused() -> None:
    """``..._heartbeat_timestamp`` must not match ``..._heartbeat_timestamp_x``."""
    body = exposition(RECORDER_SERIES + "_extra", "1759000000")
    assert parse_gauge(body, RECORDER_SERIES) is None


# --------------------------------------------------------------------------- #
# the four unhealthy shapes, each against its healthy twin
# --------------------------------------------------------------------------- #


def test_a_fresh_heartbeat_is_alive() -> None:
    body = exposition(RECORDER_SERIES, str(NOW - 10))
    reading = read_target(RECORDER, NOW, fetch=serving(body), max_age_s=120)
    assert reading.verdict == ALIVE
    assert reading.alive
    assert reading.age_s == pytest.approx(10.0)


def test_an_old_heartbeat_is_stale() -> None:
    body = exposition(RECORDER_SERIES, str(NOW - 121))
    reading = read_target(RECORDER, NOW, fetch=serving(body), max_age_s=120)
    assert reading.verdict == STALE
    assert not reading.alive
    assert reading.age_s == pytest.approx(121.0)


def test_the_threshold_is_the_boundary_and_not_a_guess() -> None:
    """Exactly at the threshold is still alive; a hair past it is not.

    Stated as a test because "older than" and "at least as old as" differ by one
    page of an incident review.
    """
    at = exposition(RECORDER_SERIES, str(NOW - 120))
    past = exposition(RECORDER_SERIES, str(NOW - 120.001))
    assert read_target(RECORDER, NOW, fetch=serving(at), max_age_s=120).verdict == ALIVE
    assert read_target(RECORDER, NOW, fetch=serving(past), max_age_s=120).verdict == STALE


def test_an_endpoint_that_does_not_answer_is_unreachable_not_unknown() -> None:
    """Fail closed. This is the whole reason the check runs off-host."""
    reading = read_target(RECORDER, NOW, fetch=refusing(OSError("connection refused")))
    assert reading.verdict == UNREACHABLE
    assert not reading.alive
    assert "connection refused" in (reading.detail or "")


def test_an_endpoint_that_publishes_no_heartbeat_is_not_alive() -> None:
    """An answer is not the same as evidence of life."""
    body = "# HELP something_else A different metric\nsomething_else 1\n"
    reading = read_target(RECORDER, NOW, fetch=serving(body))
    assert reading.verdict == NO_SUCH_SERIES
    assert not reading.alive


def test_a_heartbeat_in_the_future_is_a_clock_disagreement_not_a_fresh_beat() -> None:
    """The chrony half: freshness is a subtraction across two hosts' clocks.

    A naive implementation computes a negative age, finds it under the
    threshold, and reports ``alive`` -- which is the one answer it must not give
    when the inputs cannot be compared.
    """
    body = exposition(RECORDER_SERIES, str(NOW + 600))
    reading = read_target(RECORDER, NOW, fetch=serving(body), future_tolerance_s=30)
    assert reading.verdict == CLOCK_DISAGREEMENT
    assert not reading.alive
    assert "chrony" in (reading.detail or "")


def test_a_second_of_drift_is_not_a_clock_disagreement() -> None:
    """The other side: ordinary drift between NTP-synced hosts stays alive."""
    body = exposition(RECORDER_SERIES, str(NOW + 1))
    reading = read_target(RECORDER, NOW, fetch=serving(body), future_tolerance_s=30)
    assert reading.verdict == ALIVE


def test_every_unhealthy_verdict_is_registered_as_not_alive() -> None:
    """A verdict added without deciding which side it falls on fails here.

    ``NOT_ALIVE`` is written out rather than computed as "everything but
    ``ALIVE``" so that the decision is explicit; this asserts the two agree.
    """
    assert ALIVE not in NOT_ALIVE
    assert set(NOT_ALIVE) == {STALE, UNREACHABLE, NO_SUCH_SERIES, CLOCK_DISAGREEMENT}


# --------------------------------------------------------------------------- #
# both targets together
# --------------------------------------------------------------------------- #


def test_both_targets_are_asked_even_after_the_first_is_dead() -> None:
    """One dead process and a dead host look different, and that difference is
    the second reading. A short-circuit would hide it."""
    asked: list[str] = []

    def fetch(url: str) -> str:
        asked.append(url)
        if url == RECORDER.url:
            raise OSError("connection refused")
        return exposition(RUNNER_SERIES, str(NOW - 5))

    readings = assess([RECORDER, RUNNER], NOW, fetch=fetch)
    assert asked == [RECORDER.url, RUNNER.url]
    assert [r.verdict for r in readings] == [UNREACHABLE, ALIVE]


def test_each_target_reads_its_own_series() -> None:
    """The runner's endpoint does not publish the recorder's gauge.

    Swapping the two series would make each target read the other's heartbeat
    and report ``no_such_series`` on both -- or, worse, report the recorder alive
    from the runner's beat if they ever shared a name.
    """

    def fetch(url: str) -> str:
        if url == RECORDER.url:
            return exposition(RECORDER_SERIES, str(NOW - 5))
        return exposition(RUNNER_SERIES, str(NOW - 5))

    readings = assess([RECORDER, RUNNER], NOW, fetch=fetch)
    assert [r.verdict for r in readings] == [ALIVE, ALIVE]

    swapped = [
        Target("recorder", RECORDER.url, RUNNER_SERIES),
        Target("runner", RUNNER.url, RECORDER_SERIES),
    ]
    assert [r.verdict for r in assess(swapped, NOW, fetch=fetch)] == [
        NO_SUCH_SERIES,
        NO_SUCH_SERIES,
    ]


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #


def test_the_cli_refuses_to_watch_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    """A dead-man with no targets must not report "all clear".

    argparse's ``error`` exits 2, which is the usage code this tool documents.
    """
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2
    assert "at least one of" in capsys.readouterr().err


def test_the_cli_exit_code_follows_the_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """0 when alive, 1 when tripped, and the reason is printed either way."""
    monkeypatch.setattr("tools.deadman_check.time.time", lambda: NOW)

    monkeypatch.setattr(
        "tools.deadman_check.fetch_text",
        lambda url, timeout_s=5.0: exposition(RECORDER_SERIES, str(NOW - 5)),
    )
    assert main(["--recorder", RECORDER.url]) == 0
    assert "clear" in capsys.readouterr().out

    monkeypatch.setattr(
        "tools.deadman_check.fetch_text",
        lambda url, timeout_s=5.0: exposition(RECORDER_SERIES, str(NOW - 9999)),
    )
    assert main(["--recorder", RECORDER.url]) == 1
    captured = capsys.readouterr()
    assert "TRIPPED" in captured.err
    assert "STALE" in captured.err


def test_the_default_threshold_matches_the_prometheus_alerts() -> None:
    """120 s here and 120 s in conf/alerts_demo.yml, asserted rather than hoped.

    Two thresholds for the same question that drift apart produce an incident
    where one system says the runner is down and the other says it is fine.
    """
    from pathlib import Path

    import yaml

    from tools.deadman_check import DEFAULT_MAX_AGE_S

    rules = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "conf/alerts_demo.yml").read_text(
            encoding="utf-8"
        )
    )
    expressions = [
        rule["expr"]
        for group in rules["groups"]
        for rule in group["rules"]
        if rule.get("alert") in {"RecorderDown", "RunnerDown"}
    ]
    assert expressions, "neither RecorderDown nor RunnerDown is defined any more"
    for expression in expressions:
        assert f"> {DEFAULT_MAX_AGE_S:.0f}" in expression, expression
