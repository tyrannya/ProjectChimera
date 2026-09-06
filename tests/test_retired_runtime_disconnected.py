"""PR-13's structural proof: the demo runtime cannot reach the retired layers.

Section 3.3 of the master plan disconnects Freqtrade, the inference service and
its registry, the inference client, the trading-mode controller and the
consensus function. Disconnection is a claim about REACHABILITY, so it is
asserted over the import graph rather than by running anything. An
``import chimera.demo`` that happened to succeed would prove only that one
interpreter, on one day, with one set of installed packages, did not load
``freqtrade`` -- and it would pass for the wrong reason on every machine where
Freqtrade is simply not installed, which is every machine after PR-16 and every
developer who ran ``pip install -e .`` without ``[all]``.

Everything here is static. This module imports none of the packages it makes
claims about: not ``freqtrade``, not ``chimera.demo``, not ``chimera.modes``.
It reads source files with :func:`ast.parse` and resolves dotted names against
the checkout's own file tree. The second reason for that, beyond the one above,
is that the retired edges which exist today are LAZY --
``strategies/nn_predictor_strategy.py`` imports ``nn.registry`` inside a
function body -- so a check that watched one execution would see nothing.

The first test in the file is a POSITIVE CONTROL, and it is the load-bearing
one: a walker that silently returned an empty set would satisfy every "reaches
nothing retired" assertion below while proving nothing at all. Every other
guard ships with a negative control that plants the exact forbidden construct
-- onto the real module map, or onto a parsed copy of the real compose file or
workflow -- and asserts the guard's own matcher catches it. The controls graft
onto the real inputs rather than onto an isolated toy one because two of the
resolution rules only fire against a map that knows the target, so a control
run against a map holding nothing but the planted file would pass or fail for
reasons unrelated to the guard the real tests use. Nothing is written outside
``tmp_path``.

PR-13 DELETES NOTHING. ``grep -r freqtrade`` is deliberately still non-empty;
the retired modules keep their files, their tests and their history, and the
retired compose services and CI jobs keep every step they had. What this file
forbids is an EDGE from the active runtime into them, a place for them in the
default compose topology, and a step that names them in an automatically
triggered CI job.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

#: The import roots this repository ships. A dotted name whose walk starts
#: outside these is a third-party leaf: recorded so it can be matched against
#: the retired set, never opened.
FIRST_PARTY_ROOTS = ("chimera", "nn", "strategies", "tools")

#: Section 3.3's disconnect column, as import prefixes, plus Freqtrade itself.
#: A module matches when it IS one of these or lives under it. Written out as a
#: literal rather than derived from anything in the tree: a guard that computed
#: its own forbidden set from the source it guards would agree with whatever
#: that source became.
RETIRED = (
    "strategies",
    "tools.run_bot",
    "nn.infer_service",
    "nn.registry",
    "chimera.inference_client",
    "chimera.modes",
    "chimera.consensus",
    "freqtrade",
)

#: The active entrypoints, as repository-relative paths. ``tools/demo_report.py``
#: is PR-12's and does not exist on this branch; the parametrized guard skips a
#: path that is absent so that the check strengthens by itself the day PR-12
#: merges, and :func:`test_the_active_cli_list_has_not_quietly_emptied` is what
#: stops "skip what is absent" from degenerating into "skip everything".
ACTIVE_CLIS = (
    "tools/demo_run.py",
    "tools/replay_parity.py",
    "tools/demo_report.py",
    "tools/recorder.py",
)

#: The subset whose whole job is to read files that already exist.
#: ``tools/recorder.py`` is deliberately NOT here: recording public market data
#: over a websocket is what it is for, and forbidding ``wss://`` in it would be
#: a guard nobody could keep.
DEMO_CLIS = ("tools/demo_run.py", "tools/replay_parity.py", "tools/demo_report.py")

#: Copied verbatim from tests/test_futures_no_live_path.py so the two guards
#: cannot drift into disagreeing about what a credential looks like. Matched
#: case-insensitively against the raw source, comments and docstrings included:
#: naming a credential is the step before holding one.
CREDENTIAL_TOKENS = (
    "api_key",
    "apikey",
    "api_secret",
    "secret_key",
    "x-mbx-apikey",
    "signature=",
    "private_key",
)

#: Also verbatim from that file. Scoped to exchange hosts and the two API path
#: prefixes rather than to URLs in general, because a link to a specification
#: is not an order path and a guard that failed on any URL is one people learn
#: to route around.
ENDPOINT_TOKENS = (
    "fapi.binance.com",
    "api.binance.com",
    "binance.com",
    "/fapi/",
    "/api/v3/",
    "wss://",
    "ws://",
)

LEGACY_PROFILE = "legacy"
RETIRED_SERVICES = ("freqtrade", "nn_infer")

#: The job-level condition PR-13 puts on the two historical jobs. GitHub
#: Actions has no per-job ``on:``; triggers are declared once for the whole
#: workflow, so a job-level ``if:`` evaluated against the event context is the
#: only mechanism that keeps a job runnable on demand while taking it off every
#: automatic trigger.
MANUAL_ONLY = "github.event_name == 'workflow_dispatch'"

#: A step that names any of these is a step for the retired runtime: the
#: Freqtrade library, the ``trade`` extra that installs it, the retired
#: launcher, or one of the two retired images. ``Dockerfile`` is deliberately
#: broad -- both Dockerfiles in this repository are retired ones, and an
#: automatic job that starts building an image should have to justify itself
#: here rather than slip past a narrower literal.
RETIRED_CI_TOKENS = ("freqtrade", "[trade]", "tools.run_bot", "Dockerfile")


# ---------------------------------------------------------------------------
# the closure walker
# ---------------------------------------------------------------------------
def module_name(path: Path, root: Path) -> str:
    """Dotted module name for a source file, with a trailing ``__init__`` dropped."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def first_party_modules(root: Path = REPO) -> dict[str, Path]:
    """Every first-party module in the checkout, keyed by dotted name.

    Built from the file tree rather than from an import, so the map is the same
    whether or not the packages it names would import cleanly in this
    interpreter -- which is the whole point of a static guard.
    """
    found: dict[str, Path] = {}
    for top in FIRST_PARTY_ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found[module_name(path, root)] = path
    return found


def resolve_dotted(dotted: str, known: set[str]) -> str:
    """Truncate a dotted name segment by segment until it names a known module.

    ``chimera.futures.executor.OrderIntent`` is an import of
    ``chimera.futures.executor``; ``numpy.typing`` is an import of a package
    this repository does not ship. A name that never matches is kept verbatim
    rather than dropped, because ``freqtrade.strategy`` must still be
    recognisable as a retired edge when Freqtrade is not installed.
    """
    parts = dotted.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in known:
            return candidate
        parts.pop()
    return dotted


def imported_dotted_names(tree: ast.AST, package: str, known: set[str]) -> set[str]:
    """Every module an AST imports, as dotted names resolved against ``known``.

    Three resolutions matter and none of them is what a naive reader of
    ``node.module`` does. ``from chimera import modes`` names its target in
    ``node.names``, not in ``node.module``, so the submodule is recorded
    whenever the map knows it. A relative ``from .b import thing`` carries no
    package at all and is resolved against the importing module's own package.
    And a deep attribute import is truncated to the module that actually
    exists.

    Deliberately NOT reused: ``imported_top_level_modules`` from
    tests/test_futures_no_live_path.py, which truncates to the first segment.
    That is right for its question -- is any network library reachable -- and
    useless for this one, because it makes ``chimera.modes`` indistinguishable
    from ``chimera.risk``.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(resolve_dotted(alias.name, known))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                base = ".".join(parts[: len(parts) - node.level + 1])
                if base and node.module:
                    module = f"{base}.{node.module}"
                else:
                    module = node.module or base
            else:
                module = node.module or ""
            if not module:
                continue
            found.add(resolve_dotted(module, known))
            for alias in node.names:
                child = f"{module}.{alias.name}"
                if child in known:
                    found.add(child)
    return found


def import_closure(seeds: list[str], modules: dict[str, Path]) -> set[str]:
    """Transitive import closure of ``seeds``, including unwalkable leaves.

    A worklist rather than recursion, because the graph has cycles and the
    question is only which nodes are reachable. Third-party names are added to
    the answer and not opened: they cannot be parsed from this tree, and their
    presence in the closure is exactly what the retired-set match needs.
    """
    known = set(modules)
    seen: set[str] = set()
    work = list(seeds)
    while work:
        name = work.pop()
        if name in seen:
            continue
        seen.add(name)
        path = modules.get(name)
        if path is None:
            continue
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        work.extend(imported_dotted_names(tree, package, known) - seen)
    return seen


def retired_hits(closure: set[str]) -> list[str]:
    """The members of a closure that are, or live under, a retired module."""
    return sorted(m for m in closure if any(m == b or m.startswith(b + ".") for b in RETIRED))


def package_seeds(prefix: str, modules: dict[str, Path]) -> list[str]:
    """Every module that is, or lives under, ``prefix``.

    Seeding from every module of a package rather than from its ``__init__``
    matters here: ``chimera/__init__.py`` imports nothing, so a closure seeded
    at the package root alone would be almost empty and would prove almost
    nothing.
    """
    return sorted(m for m in modules if m == prefix or m.startswith(prefix + "."))


def synthetic_modules(root: Path, sources: dict[str, str]) -> dict[str, Path]:
    """Write a throwaway module map under ``root`` for the negative controls.

    The layout is derived from the dotted names so that the controls exercise
    :func:`module_name` and the walker's own package resolution rather than a
    second, hand-maintained convention: ``pkg`` becomes ``pkg/__init__.py`` and
    ``pkg.a`` becomes ``pkg/a.py``.
    """
    modules: dict[str, Path] = {}
    for dotted, text in sources.items():
        parts = dotted.split(".")
        if any(other.startswith(dotted + ".") for other in sources):
            path = root.joinpath(*parts, "__init__.py")
        else:
            path = root.joinpath(*parts[:-1], f"{parts[-1]}.py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        modules[module_name(path, root)] = path
    return modules


def planted_modules(root: Path, sources: dict[str, str]) -> dict[str, Path]:
    """The real module map with the synthetic modules grafted onto it.

    The negative controls need the REAL map, not an isolated one. Two of the
    resolution rules only fire against a map that knows the target -- ``from
    chimera import modes`` is recorded because ``chimera.modes`` is a module in
    this checkout, and a deep attribute import is truncated to the deepest name
    the map contains -- so a control run against a map holding nothing but the
    planted file would pass or fail for reasons that have nothing to do with
    the guard the real tests use.
    """
    return {**first_party_modules(), **synthetic_modules(root, sources)}


# ---------------------------------------------------------------------------
# compose and workflow readers
# ---------------------------------------------------------------------------
def compose_spec() -> dict:
    return yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))


def workflow_spec() -> dict:
    path = REPO / ".github" / "workflows" / "ci.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def prometheus_spec() -> dict:
    return yaml.safe_load((REPO / "conf" / "prometheus.yml").read_text(encoding="utf-8"))


def workflow_triggers(spec: dict) -> dict:
    """The workflow's ``on:`` mapping, looked up under both possible keys.

    PyYAML resolves the YAML 1.1 plain scalar ``on`` to the boolean ``True``,
    so ``spec["on"]`` is ``None`` and the triggers live under ``spec[True]``.
    Both keys are tried so the guard does not become a test of the YAML
    library's version.
    """
    return spec.get(True) if spec.get(True) is not None else spec.get("on")


def dependency_names(service: dict) -> list[str]:
    """``depends_on`` in either of the two forms Compose accepts.

    The long form is a mapping of service name to condition and the short form
    is a list of names; sorting either yields the names, which is all the
    reachability question needs.
    """
    return sorted(service.get("depends_on") or [])


def default_profile_services(spec: dict) -> set[str]:
    """The services ``docker compose up`` creates with no ``--profile`` flag."""
    return {name for name, s in spec["services"].items() if not (s or {}).get("profiles")}


def dangling_default_dependencies(spec: dict) -> list[str]:
    """Default-profile services that depend on something outside the default profile.

    Compose refuses to render such a topology, so this is a syntax error
    waiting to happen as well as a reachability leak: it is the shape that
    would put a retired service back into ``docker compose up``.
    """
    default = default_profile_services(spec)
    offenders = []
    for name in sorted(default):
        for dependency in dependency_names(spec["services"][name] or {}):
            if dependency not in default:
                offenders.append(f"{name} -> {dependency}")
    return offenders


def scraped_hosts(prometheus: dict) -> set[str]:
    """Every host Prometheus is configured to scrape, port stripped."""
    return {
        target.split(":")[0]
        for job in prometheus["scrape_configs"]
        for static in job.get("static_configs") or []
        for target in static.get("targets") or []
    }


def scrape_targets_outside(prometheus: dict, allowed: set[str]) -> list[str]:
    """Scraped hosts that are not in ``allowed``. ``localhost`` is Prometheus itself."""
    permitted = allowed | {"localhost"}
    return sorted(host for host in scraped_hosts(prometheus) if host not in permitted)


def automatic_jobs(spec: dict) -> dict[str, dict]:
    """The jobs that run on ``push`` and ``pull_request`` without being asked."""
    jobs = spec["jobs"].items()
    return {n: j for n, j in jobs if str(j.get("if", "")).strip() != MANUAL_ONLY}


def retired_ci_steps(spec: dict) -> list[str]:
    """``job: token`` for every retired-runtime token named in an automatic job.

    The steps are rendered to JSON and scanned as text rather than walked
    field by field, because a Freqtrade dependency can appear in a ``run:``
    heredoc, a ``with.file:``, a step name or an ``env:`` value, and a scan
    that only knew about ``run:`` would miss the image builds entirely.
    """
    offenders = []
    for job_name, job in automatic_jobs(spec).items():
        rendered = json.dumps(job.get("steps", []), sort_keys=True)
        named = [token for token in RETIRED_CI_TOKENS if token in rendered]
        offenders.extend(f"{job_name}: {token}" for token in named)
    return sorted(offenders)


def token_hits(text: str, tokens: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [token for token in tokens if token in lowered]


# ---------------------------------------------------------------------------
# (0) positive control -- without this, an empty closure passes everything
# ---------------------------------------------------------------------------
def test_the_closure_walk_actually_reaches_the_modules_the_demo_path_uses():
    """The single most important test in this file.

    Every guard below asserts that something is ABSENT from a closure. A
    walker that returned ``set()`` -- because the module map was empty, because
    ``rglob`` found nothing after a directory rename, because an exception was
    swallowed -- would satisfy all of them and prove nothing whatsoever. So the
    walk is first required to find the modules the demo runner is known to use:
    the risk engine, the futures executor, the hedged position, the recorder's
    normaliser and the metrics module. The size floor is deliberately far below
    the real answer, so that the test fails on a broken walker rather than on a
    refactor that moves a module.
    """
    modules = first_party_modules()
    closure = import_closure(package_seeds("chimera.demo", modules), modules)
    for expected in (
        "chimera.risk",
        "chimera.futures.executor",
        "chimera.carry.hedge",
        "chimera.recorder.normalize",
        "chimera.metrics",
    ):
        assert expected in closure, f"the walk did not reach {expected}; it is not walking"
    assert len(closure) > 20, f"closure of {len(closure)} is too small to be the real graph"


def test_the_walk_returns_nothing_extra_for_a_module_that_imports_nothing(tmp_path):
    """The other half of the positive control: the walk is not returning the universe.

    A closure that answered "everything" would also satisfy the membership
    assertions above, so the walker is pinned from both sides.
    """
    modules = synthetic_modules(tmp_path, {"lonely": "x = 1\n"})
    assert import_closure(["lonely"], modules) == {"lonely"}


# ---------------------------------------------------------------------------
# (1)(2) the active packages reach nothing retired
# ---------------------------------------------------------------------------
def test_the_demo_package_reaches_nothing_retired():
    modules = first_party_modules()
    seeds = package_seeds("chimera.demo", modules)
    assert seeds, "chimera/demo is missing; the guard would be vacuous"
    hits = retired_hits(import_closure(seeds, modules))
    assert hits == [], (
        "section 3.3 disconnects the retired runtime from the demo path; "
        f"chimera.demo can reach {hits}"
    )


def test_the_carry_package_reaches_nothing_retired():
    modules = first_party_modules()
    seeds = package_seeds("chimera.carry", modules)
    assert seeds, "chimera/carry is missing; the guard would be vacuous"
    hits = retired_hits(import_closure(seeds, modules))
    assert hits == [], f"chimera.carry can reach {hits}"


# ---------------------------------------------------------------------------
# (3) the active CLIs reach nothing retired
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("relative", ACTIVE_CLIS)
def test_an_active_cli_reaches_nothing_retired(relative):
    path = REPO / relative
    if not path.is_file():
        pytest.skip(f"{relative} has not landed on this branch yet")
    modules = first_party_modules()
    hits = retired_hits(import_closure([module_name(path, REPO)], modules))
    assert hits == [], f"{relative} can reach {hits}"


def test_the_active_cli_list_has_not_quietly_emptied():
    """The guard against the skip above turning the parametrized test into a no-op.

    ``tools/demo_run.py`` and ``tools/replay_parity.py`` landed with PR-10 and
    PR-11 and are not allowed to be absent. ``tools/demo_report.py`` is PR-12's
    and legitimately is, which is why the list is checked rather than the skip
    being removed.
    """
    landed = {c for c in ACTIVE_CLIS if (REPO / c).is_file()}
    assert {"tools/demo_run.py", "tools/replay_parity.py", "tools/recorder.py"} <= landed


# ---------------------------------------------------------------------------
# negative controls for the closure walk
# ---------------------------------------------------------------------------
def test_an_inert_planted_module_is_clean(tmp_path):
    """The baseline the four controls below are measured against.

    Each of them grafts one forbidden line onto the real module map. This
    asserts that the graft itself is what makes them fail: the same map, with a
    module that imports nothing retired, reports nothing.
    """
    modules = planted_modules(tmp_path, {"planted": "from chimera.risk import RiskEngine\n"})
    assert retired_hits(import_closure(["planted"], modules)) == []


def test_the_closure_walk_catches_a_planted_direct_edge(tmp_path):
    modules = planted_modules(tmp_path, {"planted": "from chimera.modes import TradingMode\n"})
    assert "chimera.modes" in retired_hits(import_closure(["planted"], modules))


def test_the_closure_walk_catches_a_planted_edge_two_hops_away(tmp_path):
    """Transitivity, which a one-hop guard would miss entirely."""
    modules = planted_modules(
        tmp_path,
        {
            "planted": "import middle\n",
            "middle": "from strategies.nn_predictor_strategy import NNPredictorStrategy\n",
        },
    )
    hits = retired_hits(import_closure(["planted"], modules))
    assert "strategies.nn_predictor_strategy" in hits


def test_the_closure_walk_catches_a_from_package_import_submodule_edge(tmp_path):
    """``from chimera import modes`` -- the form a ``node.module`` reader records
    as a harmless import of ``chimera`` and then misses."""
    modules = planted_modules(tmp_path, {"planted": "from chimera import modes\n"})
    assert "chimera.modes" in retired_hits(import_closure(["planted"], modules))


def test_the_closure_walk_catches_a_relative_edge(tmp_path):
    """A relative import names no package at all until it is resolved."""
    modules = planted_modules(
        tmp_path,
        {
            "pkg": "",
            "pkg.a": "from .b import thing\n",
            "pkg.b": "import freqtrade\nthing = 1\n",
        },
    )
    assert retired_hits(import_closure(["pkg.a"], modules)) == ["freqtrade"]


def test_the_closure_walk_catches_a_deep_attribute_import(tmp_path):
    """``import nn.registry.something`` still resolves to the retired module."""
    modules = planted_modules(tmp_path, {"planted": "import nn.registry\n"})
    assert "nn.registry" in retired_hits(import_closure(["planted"], modules))


# ---------------------------------------------------------------------------
# (4) the default compose topology
# ---------------------------------------------------------------------------
def test_the_default_compose_stack_excludes_the_retired_services():
    """Disconnected from ``docker compose up``, and still present in the file.

    Both halves are asserted. PR-13 moves the two retired services behind the
    ``legacy`` profile; it does not delete them, and a later change that
    deleted them here would be PR-16's work, not this guard's silence.
    """
    spec = compose_spec()
    for name in RETIRED_SERVICES:
        assert name in spec["services"], "PR-13 disconnects the service; PR-16 deletes it"
        assert spec["services"][name].get("profiles") == [LEGACY_PROFILE]

    default = default_profile_services(spec)
    assert default, "the default profile is empty; the guard would be vacuous"
    assert not default & set(RETIRED_SERVICES)
    assert dangling_default_dependencies(spec) == []


def test_the_compose_guard_catches_a_retired_service_left_in_the_default_profile():
    spec = copy.deepcopy(compose_spec())
    del spec["services"]["freqtrade"]["profiles"]
    assert default_profile_services(spec) & set(RETIRED_SERVICES) == {"freqtrade"}


def test_the_compose_guard_catches_a_default_service_depending_on_a_legacy_one():
    spec = copy.deepcopy(compose_spec())
    legacy_dependency = {"nn_infer": {"condition": "service_healthy"}}
    spec["services"]["prometheus"]["depends_on"] = legacy_dependency
    assert dangling_default_dependencies(spec) == ["prometheus -> nn_infer"]


# ---------------------------------------------------------------------------
# (4b) Prometheus follows the compose topology
# ---------------------------------------------------------------------------
def test_prometheus_scrapes_nothing_outside_the_default_compose_stack():
    """The scrape list and the default stack are one decision, not two.

    Ruling D3: the `freqtrade` and `nn_infer` scrape jobs go in the same change
    that profiles those two services out, so main never holds a scrape job
    pointing at a service `docker compose up` does not create.

    tests/test_observability.py:112 asks the weaker question -- is the host a
    KEY in docker-compose.yml -- and PR-13 keeps both services as keys, so that
    test would have stayed green with the jobs left in place. Two independent
    assertions here rather than one: the structural one against the rendered
    default profile, and a literal one naming the two services, so the guard
    still says something if the profile mechanism itself is what changes.
    """
    prometheus = prometheus_spec()
    default = default_profile_services(compose_spec())
    assert scrape_targets_outside(prometheus, default) == []
    job_names = {job["job_name"] for job in prometheus["scrape_configs"]}
    assert not job_names & set(RETIRED_SERVICES), f"{job_names} still names a retired service"


def test_the_legacy_alert_rules_are_unloaded_and_still_on_disk():
    """Disconnected, not deleted -- the whole shape of this PR in one file.

    Every expression in conf/alerts.yml queries a series only the two retired
    services export, so it is dropped from `rule_files`: a loaded rule whose
    job nothing scrapes is an alert that can never fire and can never be
    trusted. The file stays in the repository as the legacy rule set, stays
    mounted into the container, and stays covered by
    tests/test_observability.py:104.
    """
    assert (prometheus_spec().get("rule_files") or []) == []
    assert (REPO / "conf" / "alerts.yml").is_file(), "PR-13 deletes nothing"
    mounts = compose_spec()["services"]["prometheus"]["volumes"]
    assert any("conf/alerts.yml" in mount for mount in mounts)


def test_the_prometheus_guard_catches_a_reinstated_legacy_scrape_job():
    prometheus = copy.deepcopy(prometheus_spec())
    prometheus["scrape_configs"].append(
        {"job_name": "nn_infer", "static_configs": [{"targets": ["nn_infer:3000"]}]}
    )
    default = default_profile_services(compose_spec())
    assert scrape_targets_outside(prometheus, default) == ["nn_infer"]


# ---------------------------------------------------------------------------
# (5) CI
# ---------------------------------------------------------------------------
def test_no_automatically_triggered_ci_job_has_a_freqtrade_or_retired_image_step():
    """Exactly what it says, and deliberately not more.

    The claim is about STEPS: no job that runs on ``push`` or
    ``pull_request`` names Freqtrade, the ``trade`` extra, the retired launcher
    or a retired image. It is NOT the claim that no automatic job requires
    Freqtrade. The library remains INSTALLED transitively on both legs of the
    ``test`` job -- through ``requirements-lock.txt`` on Linux and ``.[all]``
    on Windows -- because tests/test_strategies.py,
    tests/test_config_and_cli.py and tests/test_risk_regressions.py import it
    at module scope and PR-13's acceptance is that the existing tests keep
    passing. Removing the library and those imports together is PR-16's work
    after S4. Nothing here should be read as having achieved it.

    Also narrower than it may look in the other direction: ``python -m
    tools.smoke`` still runs in the ``test`` job and still exercises
    ``nn.registry`` and ``nn.infer_service``. That is deliberate --
    ``nn/registry.py`` is dual-role, ``nn/train.py`` writes through it, and
    gating the smoke step would remove CI coverage of historical code the plan
    requires to keep running.
    """
    spec = workflow_spec()
    assert set(automatic_jobs(spec)) == {"lint", "test"}
    assert retired_ci_steps(spec) == [], (
        "the Freqtrade schema job and the retired image builds are manual "
        f"(if: {MANUAL_ONLY}) after PR-13; found {retired_ci_steps(spec)}"
    )

    # And the manual jobs are still there, with every step they had.
    for job_name in ("config", "docker"):
        job = spec["jobs"][job_name]
        assert job["if"].strip() == MANUAL_ONLY
        assert job["steps"], f"{job_name} lost its steps; PR-13 deletes nothing"

    # The workflow's own triggers are untouched, so the historical validation
    # stays runnable on demand and lint/test still run on every push and PR.
    assert workflow_triggers(spec) == {
        "push": {"branches": ["main"]},
        "pull_request": None,
        "workflow_dispatch": None,
    }


def test_compose_validation_still_runs_on_every_push_and_pull_request():
    """The coverage that gating the ``config`` job would otherwise have dropped.

    ``docker compose config --quiet`` used to live inside ``Config
    validation``. PR-13 is the change that edits docker-compose.yml, so losing
    compose validation from automatic CI in the same commit is not acceptable;
    the step was relocated into ``lint``, and the profile variant was added so
    the newly profiled services are still parsed by something.
    """
    spec = workflow_spec()
    rendered = json.dumps(automatic_jobs(spec)["lint"]["steps"], sort_keys=True)
    assert "docker compose config --quiet" in rendered
    assert f"docker compose --profile {LEGACY_PROFILE} config --quiet" in rendered


def test_the_ci_guard_catches_a_freqtrade_step_in_an_automatic_job():
    """The exact regression: a retired step moved back onto an automatic trigger."""
    spec = copy.deepcopy(workflow_spec())
    step = next(s for s in spec["jobs"]["config"]["steps"] if "[trade]" in json.dumps(s))
    spec["jobs"]["lint"]["steps"].append(step)
    assert retired_ci_steps(spec) == ["lint: [trade]"]


def test_the_ci_guard_catches_a_job_whose_if_was_removed():
    spec = copy.deepcopy(workflow_spec())
    del spec["jobs"]["config"]["if"]
    assert "config" in automatic_jobs(spec)
    assert retired_ci_steps(spec), "an ungated config job must report its Freqtrade steps"


# ---------------------------------------------------------------------------
# (6) the demo CLIs name no venue
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("relative", DEMO_CLIS)
def test_a_demo_cli_names_no_exchange_endpoint_and_no_credential(relative):
    """Textual on purpose, exactly as the futures guard is.

    tests/test_demo_cli.py already asserts the AST half for
    ``tools/demo_run.py``. This is the half an AST cannot see: a host named in
    a comment or a docstring is a host somebody can reach for, and an
    environment variable spelled ``API_SECRET`` in a help string is the first
    half of a live client.
    """
    path = REPO / relative
    if not path.is_file():
        pytest.skip(f"{relative} has not landed on this branch yet")
    text = path.read_text(encoding="utf-8")
    assert token_hits(text, ENDPOINT_TOKENS) == [], f"{relative} names an exchange endpoint"
    assert token_hits(text, CREDENTIAL_TOKENS) == [], f"{relative} names a credential"


def test_the_endpoint_guard_catches_a_planted_endpoint(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        'URL = "https://fapi.binance.com/fapi/v1/order"\nKEY = "X-MBX-APIKEY"\n',
        encoding="utf-8",
    )
    text = planted.read_text(encoding="utf-8")
    assert token_hits(text, ENDPOINT_TOKENS)
    assert token_hits(text, CREDENTIAL_TOKENS)


# ---------------------------------------------------------------------------
# (7) anti-rot on the guards PR-13 leans on
# ---------------------------------------------------------------------------
def test_the_existing_no_live_path_guards_still_forbid_freqtrade():
    """PR-13 modifies none of these files; this is what makes that safe.

    The disconnect argument rests on three existing guards continuing to name
    Freqtrade in their forbidden sets. A later edit that quietly dropped the
    string from one of them would weaken this PR's claim in a file nobody would
    think to re-read, and would otherwise fail nowhere.
    """
    from tests import test_demo_no_live_path as demo_guard

    assert "freqtrade" in demo_guard.NETWORK_ROOTS

    for relative, constant in (
        ("tests/test_futures_no_live_path.py", "FORBIDDEN_IMPORTS"),
        ("tests/test_recorder_no_network.py", "FORBIDDEN_IMPORTS"),
        ("tests/test_recorder_no_network.py", "LIVE_FORBIDDEN_IMPORTS"),
    ):
        text = (REPO / relative).read_text(encoding="utf-8")
        assert '"freqtrade"' in text, f"{relative}:{constant} no longer forbids freqtrade"
