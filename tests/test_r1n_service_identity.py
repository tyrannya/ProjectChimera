"""R1-n's separate identities: neither service can write the other's state.

Before this, both units ran as one ``chimera`` user. Nothing in the code made
that safe -- it meant the runner was *able* to rewrite the recorder's parquet
files, and the recorder was *able* to rewrite the runner's ledger and decision
log. Neither needs the capability, and a capability that exists is one an
incident can use: a confused path, a stale configuration, a bug in a cleanup
routine. The split is the only kind of fix that removes the ability rather than
the intention.

The property asserted here is the one that matters operationally and is
checkable from the files: **two accounts, two writable roots, no overlap**, with
the runner holding read access to the recorder's output through a shared group
and the recorder holding nothing of the runner's.

These are assertions about unit FILES. They cannot prove what systemd does on a
host -- no test in this repository can, and claiming otherwise would be the kind
of overstatement R1-n exists to remove. What they catch is the regression that
is actually likely: someone editing one unit and putting the two services back
on one account, or adding a ``ReadWritePaths`` that reaches into the other's
root. Each guard has a negative control that plants exactly that onto the real
parsed unit, and a positive one asserting the current files pass.

The host commands that create the two accounts live in the header of
``chimera-recorder.service``, where an operator installing the unit will read
them; a test cannot run ``useradd``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

RECORDER_UNIT = REPO / "deploy/systemd/chimera-recorder.service"
DEMO_UNIT = REPO / "deploy/systemd/chimera-demo.service"

#: Directives that add to a unit's writable set. ``StateDirectory`` is relative
#: to /var/lib, which is why it is expanded rather than compared as written.
WRITABLE_DIRECTIVES = ("ReadWritePaths", "StateDirectory")

STATE_DIRECTORY_BASE = "/var/lib"


def parse_unit(text: str) -> dict[str, list[str]]:
    """A systemd unit as directive -> every value given for it, in order.

    Deliberately not a full systemd parser, but it does two things a naive one
    gets wrong, and both were found by the tests below rather than by reading:

    * it keeps duplicates, because ``ReadWritePaths`` is additive and a parser
      that kept only the last value would silently drop half of what a unit can
      write -- the exact fact this file is about;
    * it joins trailing-backslash continuations, because both ``ExecStart``
      directives here span several lines. A line-based parser reads the first
      line, finds no ``=`` on the rest, and reports an ``ExecStart`` that
      contains neither ``--base-dir`` nor ``--root``.
    """
    joined: list[str] = []
    pending = ""
    for raw in text.splitlines():
        line = pending + raw.strip()
        if line.endswith("\\"):
            pending = line[:-1].rstrip() + " "
            continue
        pending = ""
        joined.append(line)

    directives: dict[str, list[str]] = {}
    for line in joined:
        if not line or line.startswith(("#", ";", "[")):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        directives.setdefault(key.strip(), []).append(value.strip())
    return directives


def writable_paths(directives: dict[str, list[str]]) -> set[str]:
    """Every path the unit may write, normalised to an absolute path.

    A leading ``-`` on ``ReadWritePaths`` means "skip if absent"; it is part of
    the syntax, not of the path, and a guard that compared the raw string would
    miss ``-/var/lib/chimera/data``.
    """
    paths: set[str] = set()
    for value in directives.get("ReadWritePaths", []):
        for item in value.split():
            paths.add(item.lstrip("-+!"))
    for value in directives.get("StateDirectory", []):
        for item in value.split():
            paths.add(f"{STATE_DIRECTORY_BASE}/{item}")
    return paths


def is_under(child: str, parent: str) -> bool:
    """Whether ``child`` is ``parent`` or sits inside it.

    String containment would call ``/var/lib/chimera-demo`` a child of
    ``/var/lib/chimera``, which it is not: that is a sibling whose name happens
    to start with the same letters, and conflating the two would make this
    file's central assertion fail for a reason that is not a real overlap.
    """
    return child == parent or child.startswith(parent.rstrip("/") + "/")


def overlapping(left: set[str], right: set[str]) -> list[tuple[str, str]]:
    """Every pair where one unit's writable path reaches into the other's."""
    return [
        (a, b) for a in sorted(left) for b in sorted(right) if is_under(a, b) or is_under(b, a)
    ]


@pytest.fixture
def recorder() -> dict[str, list[str]]:
    return parse_unit(RECORDER_UNIT.read_text(encoding="utf-8"))


@pytest.fixture
def demo() -> dict[str, list[str]]:
    return parse_unit(DEMO_UNIT.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# positive control
# --------------------------------------------------------------------------- #


def test_the_parser_reads_both_units(
    recorder: dict[str, list[str]], demo: dict[str, list[str]]
) -> None:
    """A parser that returned nothing would make every guard below vacuous."""
    for name, directives in (("recorder", recorder), ("demo", demo)):
        assert directives, f"{name}: parsed no directives at all"
        assert directives.get("User"), f"{name}: declares no User"
        assert directives.get("ExecStart"), f"{name}: declares no ExecStart"
        assert writable_paths(directives), f"{name}: declares no writable path"


# --------------------------------------------------------------------------- #
# the guards
# --------------------------------------------------------------------------- #


def test_the_two_services_run_as_different_users(
    recorder: dict[str, list[str]], demo: dict[str, list[str]]
) -> None:
    assert recorder["User"] == ["chimera-recorder"]
    assert demo["User"] == ["chimera-runner"]
    assert recorder["User"] != demo["User"], (
        "the recorder and the runner share a Unix account, so each can rewrite "
        "the other's files. R1-n splits them."
    )


def test_neither_writable_root_reaches_into_the_other(
    recorder: dict[str, list[str]], demo: dict[str, list[str]]
) -> None:
    clashes = overlapping(writable_paths(recorder), writable_paths(demo))
    assert not clashes, (
        "a writable path of one service reaches into the other's state root: "
        f"{clashes}. The recorder owns the market data, the runner owns the "
        "decisions, and neither may write the other's."
    )


def test_the_runner_reads_the_recorder_through_a_shared_group(
    recorder: dict[str, list[str]], demo: dict[str, list[str]]
) -> None:
    """Read access is granted by group membership, not by sharing an account.

    The asymmetry is the point and it is asserted in both directions: the runner
    is in the recorder's data group, and the recorder is in nothing of the
    runner's.
    """
    assert recorder["Group"] == ["chimera-data"]
    assert demo.get("SupplementaryGroups") == ["chimera-data"]
    assert demo["Group"] == ["chimera-runner"]
    assert "chimera-runner" not in recorder.get("SupplementaryGroups", [])


def test_the_recorder_root_the_runner_reads_is_the_one_the_recorder_writes(
    recorder: dict[str, list[str]], demo: dict[str, list[str]]
) -> None:
    """The split is only meaningful if the two still agree on the data path.

    Two accounts pointed at two different directories would not be a hardened
    runtime, it would be a runner reading an empty tree.
    """
    recorder_root = next(
        value.split("--base-dir")[1].split()[0]
        for value in recorder["ExecStart"]
        if "--base-dir" in value
    )
    runner_reads = next(
        value.split("--root")[1].split()[0] for value in demo["ExecStart"] if "--root" in value
    )
    assert recorder_root == runner_reads == "/var/lib/chimera/data"
    assert recorder_root in writable_paths(recorder)
    assert recorder_root not in writable_paths(demo)


def test_the_runner_state_root_is_outside_the_checkout(demo: dict[str, list[str]]) -> None:
    """With a checkout neither account can write, a state root inside it is a lie.

    This is the half of the change that is easy to leave half-done: the users get
    split, and a ``ReadWritePaths`` pointing at ``/opt/chimera/state`` stays
    behind naming a path the service user cannot write.
    """
    paths = writable_paths(demo)
    assert paths == {"/var/lib/chimera-demo"}
    assert not any(path.startswith("/opt/chimera") for path in paths)


# --------------------------------------------------------------------------- #
# negative controls -- planted onto the real parsed units
# --------------------------------------------------------------------------- #


def test_putting_the_services_back_on_one_account_is_caught(
    recorder: dict[str, list[str]], demo: dict[str, list[str]]
) -> None:
    dirty = copy.deepcopy(demo)
    dirty["User"] = recorder["User"]
    assert dirty["User"] == recorder["User"], "the control did not plant the clash"
    # The guard's own comparison, re-run against the planted unit.
    assert not (recorder["User"] != dirty["User"])


@pytest.mark.parametrize(
    "planted",
    [
        "/var/lib/chimera/data",  # the recorder's root, exactly
        "/var/lib/chimera/data/prospective",  # inside it
        "/var/lib",  # a parent that contains both
        "-/var/lib/chimera/data",  # the skip-if-absent form
    ],
)
def test_a_writable_path_reaching_into_the_recorder_is_caught(
    recorder: dict[str, list[str]], demo: dict[str, list[str]], planted: str
) -> None:
    dirty = copy.deepcopy(demo)
    dirty.setdefault("ReadWritePaths", []).append(planted)
    assert overlapping(
        writable_paths(recorder), writable_paths(dirty)
    ), f"planting {planted!r} did not register as an overlap"


def test_a_sibling_directory_is_not_an_overlap(
    recorder: dict[str, list[str]], demo: dict[str, list[str]]
) -> None:
    """The other side, and it is load-bearing.

    ``/var/lib/chimera-demo`` starts with the same nine characters as
    ``/var/lib/chimera``. A guard that compared prefixes as strings would call
    the real, correct layout an overlap and would then have to be weakened to
    pass -- which is how a guard stops guarding.
    """
    assert not overlapping(writable_paths(recorder), writable_paths(demo))
    assert not is_under("/var/lib/chimera-demo", "/var/lib/chimera")
    assert is_under("/var/lib/chimera/data", "/var/lib/chimera")
