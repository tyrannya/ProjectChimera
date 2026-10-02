"""R1-n's supply-chain pins: nothing this repository executes is named by a tag.

A tag is a movable pointer. ``actions/checkout@v4`` is whatever the ``v4`` ref
points at the moment a job starts, and ``python:3.11-slim`` is whatever the
upstream image library last published under that name. Both are resolved at run
time, by someone else, after review. A compromised action runs with this
repository's token; a rebuilt base image changes what a soak was running on.
Neither event leaves a mark in any diff, which is the whole problem: the change
is real, it is executed, and the repository's history says nothing happened.

So every reference is pinned to an immutable object -- a 40-hex commit SHA for
an action, a ``sha256:`` manifest digest for an image -- and this file refuses
anything else. The version comments beside the pins are labels for the person
reading them and carry no authority; the matchers below ignore them entirely.

THREE SURFACES, because a pin can hide in any of them:

* ``uses:`` in a workflow or a composite action;
* ``FROM`` in a Dockerfile anywhere in the tree;
* ``image:`` in compose, for a service this repository does not build itself.

The collectors are globbed over the whole tree rather than listed by name. That
is deliberate and it is the lesson of R1-m, where a guard written to close
exactly this kind of gap read one hard-coded file (``pyproject.toml``) and would
have reported a completed retirement while Freqtrade was still installed on
every CI run from a second file it never opened. A new Dockerfile or a second
workflow must be covered the day it lands, not the day someone remembers to add
it here.

The first test is a POSITIVE CONTROL and it is the load-bearing one: a collector
that silently found nothing would satisfy every "no unpinned reference"
assertion below while proving nothing. Each matcher then ships with a negative
control that plants the forbidden form onto the real file's own content and
asserts the matcher catches it, and a positive one that plants the pinned form
and asserts it does not. Nothing is written outside ``tmp_path``.

What this file does NOT claim: that a pinned SHA or digest is the *right* one.
It cannot -- that is a review question about what the object contains. It claims
only that what runs is an object and not a name.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

#: A pinned action reference: ``owner/repo[/subdir]@<40 lowercase hex>``. GitHub
#: resolves nothing else to an immutable object -- a tag and a branch are both
#: refs, and a short SHA is ambiguous by construction.
ACTION_PIN = re.compile(r"^[^/\s]+/[^@\s]+@[0-9a-f]{40}$")

#: A pinned image reference: anything, then ``@sha256:<64 lowercase hex>``. The
#: tag may stay in front of the digest (``python:3.11-slim@sha256:...``); docker
#: resolves the digest and the tag is then a comment for the reader.
IMAGE_PIN = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")

#: ``uses:`` values that name no third party and so pin nothing: an action in
#: this repository, resolved from the same commit the workflow came from.
LOCAL_USES_PREFIXES = ("./", "../")

#: ``FROM`` operands that are not images: the empty base, and a reference back
#: to a stage declared earlier in the same Dockerfile.
NOT_AN_IMAGE = ("scratch",)


# --------------------------------------------------------------------------- #
# collectors
# --------------------------------------------------------------------------- #


def workflow_files() -> list[Path]:
    """Every workflow and composite action definition in the checkout."""
    found: list[Path] = []
    for pattern in (".github/workflows/*.yml", ".github/workflows/*.yaml"):
        found.extend(sorted(REPO.glob(pattern)))
    for pattern in (".github/actions/**/action.yml", ".github/actions/**/action.yaml"):
        found.extend(sorted(REPO.glob(pattern)))
    return found


def dockerfiles() -> list[Path]:
    """Every Dockerfile in the checkout, wherever it lives.

    Globbed, not listed: ``deploy/docker/`` and ``nn/`` each hold one today and
    neither location is privileged.
    """
    return sorted(
        p for p in REPO.rglob("Dockerfile*") if p.is_file() and ".git" not in p.parts
    )


def compose_files() -> list[Path]:
    """Every compose file in the checkout."""
    return sorted(p for p in REPO.glob("docker-compose*.y*ml") if p.is_file())


# --------------------------------------------------------------------------- #
# matchers -- pure functions over text, so a control can plant into a copy
# --------------------------------------------------------------------------- #


def action_uses(text: str) -> list[str]:
    """Every ``uses:`` value in a workflow, read as text.

    Text and not parsed YAML on purpose: a ``uses:`` inside a commented-out or
    not-yet-valid block is still a line someone will uncomment, and the question
    here is what the file says, not what one loader makes of it today.
    """
    return [
        m.group(1).strip().strip("'\"")
        for m in re.finditer(r"^\s*(?:-\s*)?uses:\s*(\S+)", text, re.MULTILINE)
    ]


def unpinned_action_uses(text: str) -> list[str]:
    """The ``uses:`` values that name something movable."""
    bad: list[str] = []
    for ref in action_uses(text):
        if ref.startswith(LOCAL_USES_PREFIXES):
            continue
        if ref.startswith("docker://"):
            if not IMAGE_PIN.match(ref[len("docker://") :]):
                bad.append(ref)
            continue
        if not ACTION_PIN.match(ref):
            bad.append(ref)
    return bad


def dockerfile_froms(text: str) -> list[str]:
    """Every ``FROM`` operand in a Dockerfile, with flags and ``AS`` stripped."""
    operands: list[str] = []
    for m in re.finditer(r"^\s*FROM\s+(.*)$", text, re.MULTILINE | re.IGNORECASE):
        words = [w for w in m.group(1).split() if not w.startswith("--")]
        if words:
            operands.append(words[0])
    return operands


def unpinned_dockerfile_froms(text: str) -> list[str]:
    """The ``FROM`` operands that name a movable image.

    A stage name declared earlier in the same file (``FROM x AS build`` then
    ``FROM build``) is not an image and is skipped; so is ``scratch``.
    """
    stages = {
        m.group(1).lower()
        for m in re.finditer(r"^\s*FROM\s+.*\bAS\s+(\S+)", text, re.MULTILINE | re.IGNORECASE)
    }
    bad: list[str] = []
    for ref in dockerfile_froms(text):
        if ref.lower() in stages or ref.lower() in NOT_AN_IMAGE:
            continue
        if ref.startswith("$"):  # an ARG-driven base: pinned where the ARG is
            continue
        if not IMAGE_PIN.match(ref):
            bad.append(ref)
    return bad


def unpinned_compose_images(document: Any) -> list[str]:
    """The compose services pulled by a movable name.

    A service carrying ``build:`` is built from this checkout, so its ``image:``
    is the local tag the build is written to -- there is no digest to pin and
    demanding one would be nonsense. A service with only ``image:`` is pulled
    from a registry, and the name is all that stands between the stack and a
    different image.
    """
    services = (document or {}).get("services") or {}
    bad: list[str] = []
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        if "build" in service:
            continue
        ref = service.get("image")
        if ref is None:
            continue
        if not IMAGE_PIN.match(str(ref)):
            bad.append(f"{name}: {ref}")
    return bad


# --------------------------------------------------------------------------- #
# positive control
# --------------------------------------------------------------------------- #


def test_the_collectors_find_the_three_surfaces() -> None:
    """A collector that found nothing would make every guard below vacuous.

    This is the load-bearing test in the file. The counts are lower bounds, not
    the current numbers: adding a workflow or a Dockerfile must not fail here,
    only removing the last one must.
    """
    workflows = workflow_files()
    assert workflows, "no workflow files found -- the uses: guard proves nothing"
    assert any(
        action_uses(p.read_text(encoding="utf-8")) for p in workflows
    ), "no uses: values found in any workflow -- the matcher found nothing to judge"

    files = dockerfiles()
    assert files, "no Dockerfiles found -- the FROM guard proves nothing"
    assert any(
        dockerfile_froms(p.read_text(encoding="utf-8")) for p in files
    ), "no FROM operands found -- the matcher found nothing to judge"

    composes = compose_files()
    assert composes, "no compose files found -- the image: guard proves nothing"
    pulled = 0
    for path in composes:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for service in (document.get("services") or {}).values():
            if isinstance(service, dict) and "build" not in service and "image" in service:
                pulled += 1
    assert pulled, "no pulled compose service found -- the image: guard proves nothing"


# --------------------------------------------------------------------------- #
# the guards
# --------------------------------------------------------------------------- #


def test_every_action_is_pinned_to_a_commit_sha() -> None:
    offenders: dict[str, list[str]] = {}
    for path in workflow_files():
        bad = unpinned_action_uses(path.read_text(encoding="utf-8"))
        if bad:
            offenders[str(path.relative_to(REPO))] = bad
    assert not offenders, (
        "a `uses:` names a movable ref instead of a 40-hex commit SHA. A tag is "
        "resolved when the job starts, so what runs is not what was reviewed. "
        "Resolve it with `git ls-remote https://github.com/<owner>/<repo> "
        f"refs/tags/<tag>` and pin the SHA: {offenders}"
    )


def test_every_base_image_is_pinned_to_a_digest() -> None:
    offenders: dict[str, list[str]] = {}
    for path in dockerfiles():
        bad = unpinned_dockerfile_froms(path.read_text(encoding="utf-8"))
        if bad:
            offenders[str(path.relative_to(REPO))] = bad
    assert not offenders, (
        "a `FROM` names an image by tag. The tag is rebuilt upstream, so two "
        "builds of the same Dockerfile are two different images. Read the digest "
        "with `docker buildx imagetools inspect <ref>` and pin it: "
        f"{offenders}"
    )


def test_every_pulled_compose_image_is_pinned_to_a_digest() -> None:
    offenders: dict[str, list[str]] = {}
    for path in compose_files():
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        bad = unpinned_compose_images(document)
        if bad:
            offenders[str(path.relative_to(REPO))] = bad
    assert not offenders, (
        "a compose service is pulled by tag. Services this repository builds are "
        "exempt (their `image:` is the local build tag); these are not: "
        f"{offenders}"
    )


# --------------------------------------------------------------------------- #
# negative controls -- planted onto the real inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "planted",
    [
        "      - uses: actions/checkout@v4",
        "      - uses: actions/checkout@main",
        "      - uses: actions/checkout@11d5960",  # a short SHA is ambiguous
        "      - uses: docker://python:3.11-slim",
    ],
)
def test_an_unpinned_uses_is_caught(planted: str) -> None:
    """Graft the forbidden form onto the real workflow and re-run the matcher."""
    text = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert not unpinned_action_uses(text), "the real workflow is already dirty"
    assert unpinned_action_uses(text + "\n" + planted) == [planted.split("uses:")[1].strip()]


def test_a_pinned_uses_is_allowed() -> None:
    """The other side: the matcher must not reject the pinned form it demands."""
    text = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    pinned = "      - uses: actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830"
    assert unpinned_action_uses(text + "\n" + pinned) == []
    local = "      - uses: ./.github/actions/setup"
    assert unpinned_action_uses(text + "\n" + local) == []


@pytest.mark.parametrize(
    "planted, expected",
    [
        ("FROM python:3.11-slim", "python:3.11-slim"),
        ("FROM python:latest", "python:latest"),
        ("FROM ubuntu", "ubuntu"),
        ("FROM --platform=linux/amd64 python:3.11-slim", "python:3.11-slim"),
    ],
)
def test_an_unpinned_from_is_caught(planted: str, expected: str) -> None:
    """Graft the forbidden form onto the real Dockerfile and re-run the matcher."""
    text = (REPO / "deploy/docker/Dockerfile.recorder").read_text(encoding="utf-8")
    assert not unpinned_dockerfile_froms(text), "the real Dockerfile is already dirty"
    assert unpinned_dockerfile_froms(text + "\n" + planted) == [expected]


def test_a_pinned_or_staged_from_is_allowed() -> None:
    """The other side: a digest, a build stage and ``scratch`` all pass."""
    text = (REPO / "deploy/docker/Dockerfile.recorder").read_text(encoding="utf-8")
    digest = "sha256:" + "0" * 64
    staged = f"\nFROM python:3.11-slim@{digest} AS build\nFROM build\nFROM scratch\n"
    assert unpinned_dockerfile_froms(text + staged) == []


@pytest.mark.parametrize("planted", ["prom/prometheus:v2.53.0", "prom/prometheus"])
def test_an_unpinned_pulled_compose_image_is_caught(planted: str) -> None:
    """Graft the forbidden form onto the real compose document."""
    path = REPO / "docker-compose.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert not unpinned_compose_images(document), "the real compose file is already dirty"

    dirty = copy.deepcopy(document)
    dirty["services"]["planted"] = {"image": planted}
    assert unpinned_compose_images(dirty) == [f"planted: {planted}"]


def test_a_built_compose_service_is_exempt() -> None:
    """The other side: a service built from this checkout has no digest to pin.

    Demanding one would make the guard unsatisfiable for every image this
    repository produces, so the exemption is load-bearing and is asserted here
    rather than left to be rediscovered.
    """
    path = REPO / "docker-compose.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    built = copy.deepcopy(document)
    built["services"]["planted"] = {
        "build": {"context": ".", "dockerfile": "deploy/docker/Dockerfile.recorder"},
        "image": "chimera/planted:local",
    }
    assert unpinned_compose_images(built) == []
