"""Versioned Research Contracts: the seal is selected, never constructed.

PR #33 replaced a boundary derived from dataset length with one immutable UTC
instant. That was right for one BTC cycle and wrong as a long-term shape: a
single module constant cannot describe several instruments, timeframes or
research generations, and "edit the constant" is exactly the move that
contaminates a holdout.

So the instant now lives in a committed contract, and this suite holds the
properties that make that worth having:

* production research derives its seal from a **selected committed contract**,
  and nothing on a command line, in the environment or in the data can name a
  different instant;
* a contract's identity is a hash of its **research-defining content**, so
  reformatting leaves it alone and changing the research question does not;
* a contract **fails closed** outside its scope;
* the resolved row is dataset metadata: appending after the anchor cannot move
  it, a repair before the anchor may, and neither changes the contract;
* every new artifact records the exact contract it ran under, and two runs are
  not comparable merely for landing on the same row;
* artifacts that predate contracts stay readable and are never given one.

Everything here is functional. The one test that reads the source text is
labelled as such and is a backstop, not the guarantee.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec
from nn import experiment, train, walkforward, wf_diagnostics
from nn.data_pipeline import build_dataset, save_dataset
from nn.dataset import resolve_sealed_boundary
from nn.research_contract import (
    CONTRACT_SCHEMA,
    CONTRACTS_DIR,
    DEFAULT_CONTRACT_ID,
    ContractScopeError,
    DatasetScope,
    ResearchContract,
    ResearchContractError,
    available_contract_ids,
    canonical_material,
    load_contract,
    parse_contract,
    read_contract_file,
)
from tools.make_sample_data import generate_candles

REPO = Path(__file__).resolve().parents[1]

CONTRACT = load_contract(DEFAULT_CONTRACT_ID)

#: The semantic identity of the first BTC research contract, pinned.
#:
#: Not a golden file for its own sake: this hash is written into every artifact
#: produced under the contract, so a change to it silently orphans the
#: provenance of everything already on disk. If this assertion fails, either the
#: contract's research-defining content changed — which must be a *new*
#: generation, a new file — or the canonicalisation did, which is a schema
#: change and must bump :data:`nn.research_contract.CONTRACT_SCHEMA`.
BTC_GEN1_HASH = "dca6d0a891a257197a7c8aecec04fdba0a0b3009cfe93c5f3a397d458ab4a1de"

TINY = [
    "--epochs",
    "1",
    "--seq-len",
    "16",
    "--d-model",
    "16",
    "--n-heads",
    "2",
    "--num-layers",
    "1",
    "--device",
    "cpu",
]


def contract_payload(**overrides) -> dict:
    """A well-formed contract document, before any deliberate damage."""
    payload = {
        "contract_id": "synthetic-gen1",
        "research_generation": 1,
        "domain": "directional-classification",
        "scope": {"exchange": "binance", "pair": "BTC/USDT", "timeframe": "1h"},
        "sealed_test_start": "2025-08-27T23:00:00+00:00",
        "description": "A fixture.",
    }
    payload.update(overrides)
    return payload


def synthetic(**overrides) -> ResearchContract:
    """A contract object that no CLI can produce, for invariant tests."""
    return parse_contract(contract_payload(**overrides))


# --- A. the first contract is exactly the boundary PR #33 established ---------
def test_the_committed_btc_contract_says_what_it_should():
    assert CONTRACT.contract_id == "btc-usdt-1h-gen1"
    assert CONTRACT.research_generation == 1
    assert CONTRACT.domain == "directional-classification"
    assert CONTRACT.scope == DatasetScope(exchange="binance", pair="BTC/USDT", timeframe="1h")
    assert CONTRACT.sealed_test_start == pd.Timestamp("2025-08-27T23:00:00+00:00")
    assert CONTRACT.sealed_test_start.isoformat() == "2025-08-27T23:00:00+00:00"


def test_the_first_contract_preserves_the_seal_the_committed_artifacts_ran_under():
    """The contract does not move the boundary; it names the one already in force.

    Each committed walk-forward run recorded a sealed period starting at this
    instant. Row 48217 is what that instant resolved to in a 56,726-row dataset
    — evidence about that dataset, checked here, and never the contract itself.
    """
    artifacts = sorted((REPO / "artifacts" / "walkforward").glob("*/walkforward.json"))
    assert artifacts, "the committed artifacts must be present to check against"
    for artifact in artifacts:
        sealed = json.loads(artifact.read_text())["sealed_test"]
        assert pd.Timestamp(sealed["period"]["start"]) == CONTRACT.sealed_test_start
        assert sealed["start_row"] == 48217


def test_the_contract_identity_is_pinned():
    assert CONTRACT.contract_hash == BTC_GEN1_HASH
    assert len(CONTRACT.contract_hash) == 64
    assert CONTRACT.short_hash == BTC_GEN1_HASH[:16]
    assert CONTRACT.label == f"btc-usdt-1h-gen1@{BTC_GEN1_HASH[:16]}"


def test_only_committed_contracts_are_selectable():
    assert available_contract_ids() == sorted(
        path.stem for path in CONTRACTS_DIR.glob("*.json")
    )
    assert DEFAULT_CONTRACT_ID in available_contract_ids()
    with pytest.raises(ResearchContractError, match="unknown research contract"):
        load_contract("btc-usdt-1h-gen2")


# --- B. semantic identity: stable under formatting, not under meaning ---------
def test_identity_is_deterministic():
    assert synthetic().contract_hash == synthetic().contract_hash
    assert canonical_material(synthetic()) == canonical_material(synthetic())
    # And it is a pure function of content, not of the file it came from.
    assert synthetic().contract_hash == parse_contract(contract_payload()).contract_hash


@pytest.mark.parametrize(
    "cosmetic",
    [
        pytest.param({"sealed_test_start": "2025-08-27T23:00:00Z"}, id="anchor-as-Z"),
        pytest.param(
            {"sealed_test_start": "2025-08-28T01:00:00+02:00"}, id="anchor-in-another-zone"
        ),
        pytest.param({"description": "completely rewritten prose"}, id="description"),
        pytest.param({"description": ""}, id="description-removed"),
        pytest.param(
            {"scope": {"exchange": "Binance", "pair": "btc/usdt", "timeframe": "1H"}},
            id="scope-casing",
        ),
        pytest.param({"contract_id": "  synthetic-gen1  "}, id="id-padding"),
    ],
)
def test_cosmetic_changes_do_not_create_a_new_generation(cosmetic):
    assert synthetic(**cosmetic).contract_hash == synthetic().contract_hash


def test_reformatting_the_file_does_not_change_its_identity(tmp_path):
    """Whitespace and key order are how a file is written, not what it says."""
    source = CONTRACTS_DIR / f"{DEFAULT_CONTRACT_ID}.json"
    payload = json.loads(source.read_text())
    reordered = dict(reversed(list(payload.items())))

    reformatted = tmp_path / f"{DEFAULT_CONTRACT_ID}.json"
    reformatted.write_text(json.dumps(reordered, separators=(",", ":")))

    assert read_contract_file(reformatted).contract_hash == CONTRACT.contract_hash


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"contract_id": "synthetic-gen2"}, id="id"),
        pytest.param({"research_generation": 2}, id="generation"),
        pytest.param({"domain": "funding-basis"}, id="domain"),
        pytest.param(
            {"scope": {"exchange": "bybit", "pair": "BTC/USDT", "timeframe": "1h"}},
            id="exchange",
        ),
        pytest.param(
            {"scope": {"exchange": "binance", "pair": "ETH/USDT", "timeframe": "1h"}},
            id="pair",
        ),
        pytest.param(
            {"scope": {"exchange": "binance", "pair": "BTC/USDT", "timeframe": "4h"}},
            id="timeframe",
        ),
        pytest.param({"sealed_test_start": "2025-08-27T22:00:00+00:00"}, id="anchor-one-hour"),
    ],
)
def test_a_changed_research_question_is_a_changed_identity(change):
    assert synthetic(**change).contract_hash != synthetic().contract_hash


def test_the_hashed_material_names_the_schema_it_was_computed_under():
    """A future change to *what* is hashed must not be mistaken for the same hash."""
    material = json.loads(canonical_material(synthetic()))
    assert material["contract_schema"] == CONTRACT_SCHEMA
    assert set(material) == {
        "contract_schema",
        "contract_id",
        "research_generation",
        "domain",
        "exchange",
        "pair",
        "timeframe",
        "sealed_test_start",
    }


# --- C. a contract file that cannot be trusted is refused ---------------------
@pytest.mark.parametrize(
    "damage, message",
    [
        pytest.param({"sealed_test_start": None}, "must be an ISO-8601 string", id="null"),
        pytest.param(
            {"sealed_test_start": "2025-08-27T23:00:00"}, "no UTC offset", id="naive"
        ),
        pytest.param(
            {"sealed_test_start": "not a timestamp"}, "not a valid timestamp", id="garbage"
        ),
        pytest.param({"research_generation": 0}, "integer >= 1", id="generation-zero"),
        pytest.param({"research_generation": "1"}, "integer >= 1", id="generation-string"),
        pytest.param({"contract_id": ""}, "non-empty string", id="empty-id"),
        pytest.param({"domain": " "}, "non-empty string", id="blank-domain"),
        pytest.param(
            {"scope": {"exchange": "binance"}}, "must carry exactly", id="part-scope"
        ),
        pytest.param(
            {"scope": {"exchange": "", "pair": "BTC/USDT", "timeframe": "1h"}},
            "scope.exchange must be a non-empty string",
            id="blank-exchange",
        ),
        pytest.param({"sealed_date": "2020-01-01T00:00:00Z"}, "unknown field", id="stray-key"),
    ],
)
def test_an_unclear_contract_is_refused(damage, message):
    with pytest.raises(ResearchContractError, match=message):
        parse_contract(contract_payload(**damage))


def test_a_missing_field_is_refused():
    payload = contract_payload()
    del payload["sealed_test_start"]
    with pytest.raises(ResearchContractError, match="missing required field"):
        parse_contract(payload)


def test_a_file_must_be_named_after_the_contract_it_holds(tmp_path):
    path = tmp_path / "something-else.json"
    path.write_text(json.dumps(contract_payload()))
    with pytest.raises(ResearchContractError, match="is named"):
        read_contract_file(path)


def test_every_committed_contract_loads_and_is_uniquely_named():
    files = sorted(CONTRACTS_DIR.glob("*.json"))
    assert files, "the registry must not be empty"
    ids = [read_contract_file(path).contract_id for path in files]
    assert len(set(ids)) == len(ids)


# --- D. nothing at runtime can name a seal -----------------------------------
ENTRYPOINTS = (train, experiment, walkforward)


@pytest.mark.parametrize("module", ENTRYPOINTS, ids=lambda m: m.__name__)
@pytest.mark.parametrize(
    "flag",
    [
        "--sealed-date",
        "--sealed-test-start",
        "--sealed-start",
        "--anchor",
        "--contract-file",
        "--contract-json",
        "--research-contract-path",
    ],
)
def test_no_entrypoint_accepts_a_seal_of_its_own(module, flag):
    """The escape hatches this design exists to not have."""
    parser = module.build_argparser()
    options = {
        option for action in parser._actions for option in (action.option_strings or [])
    }
    assert flag not in options

    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "x", flag, "2020-01-01T00:00:00Z"])


@pytest.mark.parametrize("module", ENTRYPOINTS, ids=lambda m: m.__name__)
def test_the_selector_only_offers_committed_contracts(module):
    parser = module.build_argparser()
    action = next(a for a in parser._actions if a.dest == "research_contract")

    assert action.default == DEFAULT_CONTRACT_ID
    assert sorted(action.choices) == available_contract_ids()
    assert parser.parse_args(["--dataset", "x"]).research_contract == DEFAULT_CONTRACT_ID
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "x", "--research-contract", "btc-usdt-1h-gen9"])


def test_the_resolver_will_not_run_without_a_contract():
    """There is no default anchor left to fall back to."""
    dates = pd.date_range(end=CONTRACT.sealed_test_start, periods=10, freq="1h")
    with pytest.raises(TypeError):
        resolve_sealed_boundary(dates)


def test_no_environment_variable_moves_the_seal(monkeypatch):
    dates = pd.date_range(
        start=CONTRACT.sealed_test_start - pd.Timedelta(hours=100), periods=200, freq="1h"
    )
    before = resolve_sealed_boundary(dates, contract=CONTRACT)

    for name in (
        "SEALED_TEST_START_UTC",
        "CHIMERA_SEALED_TEST_START",
        "CHIMERA_RESEARCH_CONTRACT",
        "RESEARCH_CONTRACT_SEALED_TEST_START",
    ):
        monkeypatch.setenv(name, "2024-01-01T00:00:00+00:00")

    after = resolve_sealed_boundary(dates, contract=load_contract(DEFAULT_CONTRACT_ID))
    assert after == before
    assert after.anchor == CONTRACT.sealed_test_start


def test_the_source_carries_no_second_copy_of_the_anchor():
    """A backstop for the property the tests above establish functionally.

    One source of truth means the instant is written down in exactly one place:
    the contract file. A stray copy in a module would be the constant this
    change replaced, reintroduced.
    """
    anchor = "2025-08-27T23:00:00"
    for path in sorted((REPO / "nn").rglob("*.py")) + sorted((REPO / "chimera").rglob("*.py")):
        assert anchor not in path.read_text(), f"{path} restates the sealed instant"


# --- E. scope fails closed ----------------------------------------------------
@pytest.mark.parametrize(
    "identity, expected",
    [
        pytest.param(
            {"exchange": "bybit", "pair": "BTC/USDT", "timeframe": "1h"},
            "exchange is 'bybit'",
            id="exchange",
        ),
        pytest.param(
            {"exchange": "binance", "pair": "ETH/USDT", "timeframe": "1h"},
            "pair is 'ETH/USDT'",
            id="pair",
        ),
        pytest.param(
            {"exchange": "binance", "pair": "BTC/USDT", "timeframe": "4h"},
            "timeframe is '4h'",
            id="timeframe",
        ),
        pytest.param(
            {"exchange": "", "pair": "", "timeframe": ""},
            "exchange is unset",
            id="undeclared",
        ),
    ],
)
def test_a_contract_refuses_a_dataset_it_does_not_describe(identity, expected):
    with pytest.raises(ContractScopeError, match=expected):
        CONTRACT.require_scope(**identity)


def test_scope_matching_ignores_case_and_padding():
    CONTRACT.require_scope(exchange=" BINANCE ", pair="btc/usdt", timeframe=" 1H ")


# --- F. the row is dataset metadata, the contract is the boundary -------------
def hours(count: int, *, end_before_anchor: int = 0) -> pd.DatetimeIndex:
    last = CONTRACT.sealed_test_start - pd.Timedelta(hours=end_before_anchor)
    return pd.date_range(end=last, periods=count, freq="1h")


def straddling(before: int, after: int) -> pd.DatetimeIndex:
    return hours(before + after + 1, end_before_anchor=-after)


def test_appending_sealed_rows_changes_neither_the_contract_nor_the_row():
    original = straddling(before=500, after=50)
    grown = original.append(
        pd.date_range(start=original[-1] + pd.Timedelta(hours=1), periods=5_000, freq="1h")
    )
    base = resolve_sealed_boundary(original, contract=CONTRACT)
    after = resolve_sealed_boundary(grown, contract=CONTRACT)

    assert after.start_row == base.start_row
    assert after.contract.contract_hash == base.contract.contract_hash == BTC_GEN1_HASH
    assert after.to_dict()["research_contract"] == base.to_dict()["research_contract"]


def test_a_repair_before_the_anchor_moves_the_row_and_not_the_contract():
    """A legitimately recovered historical candle is not a new generation."""
    original = straddling(before=500, after=50)
    repaired = original.insert(10, original[10] - pd.Timedelta(minutes=30)).sort_values()

    base = resolve_sealed_boundary(original, contract=CONTRACT)
    after = resolve_sealed_boundary(repaired, contract=CONTRACT)

    assert after.start_row == base.start_row + 1
    assert after.contract == base.contract
    assert after.contract.contract_hash == base.contract.contract_hash
    assert after.anchor == base.anchor


def test_the_same_row_under_two_contracts_is_two_generations():
    """Landing on the same index says nothing about sealing the same data."""
    other = synthetic(contract_id="synthetic-gen2", research_generation=2)
    dates = straddling(before=500, after=50)

    mine = resolve_sealed_boundary(dates, contract=CONTRACT)
    theirs = resolve_sealed_boundary(dates, contract=other)

    assert mine.start_row == theirs.start_row
    assert mine.anchor == theirs.anchor
    assert mine.contract.contract_hash != theirs.contract.contract_hash
    assert mine.to_dict() != theirs.to_dict()


def test_the_boundary_records_both_halves():
    boundary = resolve_sealed_boundary(straddling(before=100, after=10), contract=CONTRACT)
    recorded = boundary.to_dict()

    assert recorded["anchor_timestamp"] == CONTRACT.sealed_test_start.isoformat()
    assert recorded["start_row"] == 100
    assert recorded["research_contract"] == {
        "contract_schema": CONTRACT_SCHEMA,
        "contract_id": "btc-usdt-1h-gen1",
        "contract_hash": BTC_GEN1_HASH,
        "research_generation": 1,
        "domain": "directional-classification",
        "scope": {"exchange": "binance", "pair": "BTC/USDT", "timeframe": "1h"},
        "sealed_test_start": "2025-08-27T23:00:00+00:00",
    }


# --- G. the entrypoints, run for real ----------------------------------------
def build_dataset_at(
    path: Path,
    *,
    rows: int = 900,
    exchange: str = "binance",
    pair: str = "BTC/USDT",
    timeframe: str = "1h",
) -> Path:
    """A processed dataset straddling the contract's anchor."""
    candles = generate_candles(rows=rows, seed=11, timeframe_minutes=60)
    frame, meta = build_dataset(
        candles,
        FeatureSpec(),
        TargetSpec(),
        exchange=exchange,
        pair=pair,
        timeframe=timeframe,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(path, frame, meta)
    return path


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    return build_dataset_at(tmp_path_factory.mktemp("data") / "btc.parquet")


@pytest.fixture(scope="module")
def out_of_scope_dataset(tmp_path_factory) -> Path:
    return build_dataset_at(tmp_path_factory.mktemp("eth") / "eth.parquet", pair="ETH/USDT")


def test_training_records_the_contract_it_ran_under(dataset, tmp_path):
    models_dir = tmp_path / "models"
    assert (
        train.main(
            [
                "--dataset",
                str(dataset),
                "--models-dir",
                str(models_dir),
                "--validation-only",
                *TINY,
            ]
        )
        == 0
    )
    version = next(p for p in models_dir.iterdir() if p.is_dir())

    report = json.loads((version / "report.json").read_text())
    recorded = report["sealed_test"]["research_contract"]
    assert recorded["contract_id"] == DEFAULT_CONTRACT_ID
    assert recorded["contract_hash"] == BTC_GEN1_HASH
    assert report["sealed_test"]["evaluated"] is False
    assert report["test_evaluated"] is False
    assert report["test"] is None

    metadata = json.loads((version / "metadata.json").read_text())
    assert metadata["research_contract_id"] == DEFAULT_CONTRACT_ID
    assert metadata["research_contract_hash"] == BTC_GEN1_HASH


def test_the_experiment_manifest_and_results_carry_the_contract(dataset, tmp_path):
    out = tmp_path / "exp"
    assert experiment.main(["--dataset", str(dataset), "--out", str(out), *TINY]) == 0

    manifest = json.loads((out / "experiment_plan.json").read_text())
    results = json.loads((out / "experiments.json").read_text())
    for payload in (manifest, results):
        recorded = payload["sealed_test"]["research_contract"]
        assert recorded["contract_id"] == DEFAULT_CONTRACT_ID
        assert recorded["contract_hash"] == BTC_GEN1_HASH
    assert results["plan_hash"] == manifest["plan_hash"]
    assert results["test_evaluated"] is False


def test_experiment_identity_changes_with_the_contract(dataset):
    """Same grid, same dataset, different generation: a different experiment."""
    data = train.load_research_data(dataset)
    args = experiment.build_argparser().parse_args(["--dataset", str(dataset)])
    configs = experiment.build_grid(
        {name: [getattr(experiment.RunConfig(), name)] for name in experiment.GRID_DIMENSIONS},
        experiment.RunConfig(),
    )

    def plan_hash_for(contract):
        boundary = data.sealed_boundary(contract)
        from nn.dataset import chronological_split

        plan = chronological_split(data.n_rows, boundary.start_row)
        return experiment.build_plan(args, configs, data, plan, boundary, [])["plan_hash"]

    gen1 = plan_hash_for(CONTRACT)
    # A second generation that seals at the same instant, so the two resolve to
    # the same row. Only the contract differs, and that must be enough.
    gen2 = plan_hash_for(
        replace(
            synthetic(contract_id="btc-usdt-1h-gen2", research_generation=2),
            source=None,
        )
    )
    assert gen1 != gen2


def test_walk_forward_records_the_contract_and_names_it_in_the_summary(dataset, tmp_path):
    out = tmp_path / "wf"
    assert (
        walkforward.main(["--dataset", str(dataset), "--out", str(out), "--folds", "2", *TINY])
        == 0
    )

    payload = json.loads((out / "walkforward.json").read_text())
    recorded = payload["sealed_test"]["research_contract"]
    assert recorded["contract_id"] == DEFAULT_CONTRACT_ID
    assert recorded["contract_hash"] == BTC_GEN1_HASH
    assert payload["sealed_test"]["evaluated"] is False
    assert payload["test_evaluated"] is False

    markdown = (out / "walkforward.md").read_text()
    assert DEFAULT_CONTRACT_ID in markdown
    assert BTC_GEN1_HASH in markdown


@pytest.mark.parametrize(
    "module, extra",
    [
        (train, ["--validation-only"]),
        (experiment, []),
        (walkforward, ["--folds", "2"]),
    ],
    ids=["train", "experiment", "walkforward"],
)
def test_an_out_of_scope_dataset_is_refused_by_every_entrypoint(
    module, extra, out_of_scope_dataset, tmp_path
):
    argv = ["--dataset", str(out_of_scope_dataset), *extra, *TINY]
    if module is train:
        argv += ["--models-dir", str(tmp_path / "models")]
    else:
        argv += ["--out", str(tmp_path / "out")]

    with pytest.raises(SystemExit, match="does not match"):
        module.main(argv)


# --- H. diagnostics: comparability follows the contract, history stays honest -
HISTORICAL = REPO / "artifacts" / "walkforward" / "btc_nested_v1" / "walkforward.json"


def historical_copy(directory: Path, *, contract: ResearchContract | None = None) -> Path:
    """A copy of a committed run, optionally re-issued under a contract.

    The committed file is only ever read. Writing the contract block into a
    *copy* is how the contract-era shape gets tested against a real artifact
    without touching the historical record.
    """
    payload = json.loads(HISTORICAL.read_text())
    if contract is not None:
        payload["sealed_test"]["anchor_timestamp"] = contract.sealed_test_start.isoformat()
        payload["sealed_test"]["research_contract"] = contract.provenance()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "walkforward.json").write_text(json.dumps(payload, indent=2))
    return directory


def test_the_committed_artifacts_keep_their_row_only_provenance():
    """They predate contracts. Reading them must not invent one."""
    for artifact in sorted((REPO / "artifacts" / "walkforward").glob("*/walkforward.json")):
        payload = json.loads(artifact.read_text())
        assert "research_contract" not in payload["sealed_test"]
        assert "anchor_timestamp" not in payload["sealed_test"]

        run = wf_diagnostics.load_run(artifact)
        assert run.contract is None
        assert run.contract_id is None
        assert run.contract_hash is None
        assert run.sealed_anchor is None
        assert wf_diagnostics.audit_run(run) == []


def test_row_only_runs_stay_comparable_with_each_other(tmp_path):
    runs = [
        wf_diagnostics.load_run(historical_copy(tmp_path / name))
        for name in ("run_a", "run_b")
    ]
    assert wf_diagnostics.compare_runs(runs) == []
    assert all(run.contract is None for run in runs)


def test_a_contract_run_is_not_comparable_with_a_row_only_run(tmp_path):
    runs = [
        wf_diagnostics.load_run(historical_copy(tmp_path / "historical")),
        wf_diagnostics.load_run(historical_copy(tmp_path / "current", contract=CONTRACT)),
    ]
    problems = wf_diagnostics.compare_runs(runs)
    assert any("no research contract (row-only provenance)" in p for p in problems)
    assert any("different research generations" in p for p in problems)


def test_two_contracts_on_identical_geometry_are_not_aggregated(tmp_path):
    """Same rows, same folds, same seal instant — still two generations."""
    gen2 = synthetic(contract_id="btc-usdt-1h-gen2", research_generation=2)
    runs = [
        wf_diagnostics.load_run(historical_copy(tmp_path / "gen1", contract=CONTRACT)),
        wf_diagnostics.load_run(historical_copy(tmp_path / "gen2", contract=gen2)),
    ]

    assert runs[0].sealed_test_start == runs[1].sealed_test_start
    assert runs[0].sealed_anchor == runs[1].sealed_anchor
    assert runs[0].geometry == runs[1].geometry

    problems = wf_diagnostics.compare_runs(runs)
    assert any("different research generations" in p for p in problems)
    assert any("does not make them comparable" in p for p in problems)


def test_an_edited_contract_orphans_the_artifacts_produced_under_it(tmp_path):
    """The id is committed, but its content has moved. That is a fault, not a read."""
    stale = copy.deepcopy(CONTRACT.provenance())
    stale["contract_hash"] = "0" * 64
    directory = historical_copy(tmp_path / "stale", contract=CONTRACT)
    payload = json.loads((directory / "walkforward.json").read_text())
    payload["sealed_test"]["research_contract"] = stale
    (directory / "walkforward.json").write_text(json.dumps(payload, indent=2))

    problems = wf_diagnostics.audit_run(wf_diagnostics.load_run(directory))
    assert any("has been edited since the run" in p for p in problems)


def test_a_half_written_contract_block_is_a_fault(tmp_path):
    directory = historical_copy(tmp_path / "partial", contract=CONTRACT)
    payload = json.loads((directory / "walkforward.json").read_text())
    del payload["sealed_test"]["research_contract"]["contract_hash"]
    (directory / "walkforward.json").write_text(json.dumps(payload, indent=2))

    problems = wf_diagnostics.audit_run(wf_diagnostics.load_run(directory))
    assert any("missing ['contract_hash']" in p for p in problems)


def test_a_contract_that_disagrees_with_its_own_anchor_is_a_fault(tmp_path):
    directory = historical_copy(tmp_path / "mixed", contract=CONTRACT)
    payload = json.loads((directory / "walkforward.json").read_text())
    payload["sealed_test"]["anchor_timestamp"] = "2024-01-01T00:00:00+00:00"
    (directory / "walkforward.json").write_text(json.dumps(payload, indent=2))

    problems = wf_diagnostics.audit_run(wf_diagnostics.load_run(directory))
    assert any("cannot have been planned under both" in p for p in problems)


def test_the_cli_withholds_the_aggregate_across_generations(tmp_path):
    """End to end: incompatible contracts produce problems and no headline."""
    gen2 = synthetic(contract_id="btc-usdt-1h-gen2", research_generation=2)
    out = tmp_path / "report"
    exit_code = wf_diagnostics.main(
        [
            str(historical_copy(tmp_path / "gen1", contract=CONTRACT)),
            str(historical_copy(tmp_path / "gen2", contract=gen2)),
            "--out",
            str(out),
        ]
    )

    assert exit_code == 1
    payload = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())
    assert payload["summary"] is None
    assert payload["analysis"] is None
    assert any(
        "different research generations" in problem
        for problem in payload["comparability_problems"]
    )
    assert "Runs are not comparable" in (out / wf_diagnostics.REPORT_MD).read_text()


def test_the_diagnostics_report_never_fills_in_a_missing_contract(tmp_path):
    out = tmp_path / "report"
    assert (
        wf_diagnostics.main([str(historical_copy(tmp_path / "run_a")), "--out", str(out)]) == 0
    )
    payload = json.loads((out / wf_diagnostics.REPORT_JSON).read_text())

    assert payload["runs"][0]["research_contract"] is None
    assert payload["sealed_test_evaluated"] is False
    assert (
        "no research contract (row-only provenance)"
        in (out / wf_diagnostics.REPORT_MD).read_text()
    )
