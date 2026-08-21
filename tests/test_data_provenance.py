"""Research-data provenance: which candles a run actually saw.

A Research Contract says what a generation is *allowed* to look at. Every
artifact in this repository already records that. What none of them could record
before this suite existed is what the generation *did* look at: two datasets can
agree on exchange, pair, timeframe, row count, first and last timestamp, feature
contract and target spec while differing in a historical candle, and every
recorded field would be identical.

The properties held here, in the order they are asserted below:

* research-data identity is **semantic** — compression, timestamp resolution,
  column order in the file, a stray column, a reformatted sidecar and the path
  on disk all leave it alone;
* it describes **exactly the research-visible region**: appending candles after
  the seal cannot change it, and a correction before the seal must;
* every research-visible value participates — a timestamp, a price, a feature, a
  future return, a target, a segment id, or a row that is no longer there;
* the **specification** participates: identical numbers under a different
  feature or target definition are a different research input;
* contract identity, resolved row and data identity are **three** things that
  move for three different reasons;
* new train / experiment / walk-forward artifacts carry it, and it participates
  in experiment identity;
* ``nn.wf_diagnostics`` refuses to aggregate runs whose research data differs
  even when contract, geometry and dates all match;
* artifacts that predate fingerprints stay readable and are never given one;
* computing any of this reads no sealed label, and the whole-table digest that
  does span sealed rows is kept out of comparability on purpose.

Everything here runs on synthetic fixtures. Nothing in this file is evidence
about BTC, and nothing in it opens a sealed test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chimera.contracts import TargetSpec
from chimera.features import FeatureSpec
from nn import experiment, train, walkforward, wf_diagnostics
from nn.data_fingerprint import (
    RAW_INPUT_SCHEMA,
    RESEARCH_INPUT_SCHEMA,
    DataFingerprintError,
    fingerprint_raw_input,
    research_input_digest,
)
from nn.data_pipeline import build_dataset, load_dataset, save_dataset
from nn.dataset import chronological_split
from nn.regime import RegimeDataError, load_raw_ohlcv, load_research_frame
from nn.research_contract import (
    SYNTHETIC_CONTRACT_ID,
    load_contract,
    parse_contract,
)
from tools.make_sample_data import generate_candles

REPO = Path(__file__).resolve().parents[1]

CONTRACT = load_contract(SYNTHETIC_CONTRACT_ID)

#: The synthetic dataset's scope, which is the only scope its contract accepts.
SCOPE = {"exchange": "synthetic", "pair": "SYNTH/USDT", "timeframe": "1h"}

UNDER_SYNTHETIC = ["--research-contract", SYNTHETIC_CONTRACT_ID]

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
    *UNDER_SYNTHETIC,
]


# --- fixtures ----------------------------------------------------------------
def make_candles(rows: int = 1200, *, appended: int = 0, seed: int = 17) -> pd.DataFrame:
    """Synthetic candles straddling the seal, optionally extended past it.

    Appended candles continue the same series after its last row, which is
    already past the anchor, so the shared prefix — the whole research region —
    is untouched. Generating one longer series instead would start it earlier
    and change every row.
    """
    candles = generate_candles(rows=rows, seed=seed)
    if not appended:
        return candles
    longer = generate_candles(rows=rows + appended, seed=seed)
    tail = longer.iloc[len(longer) - appended :].copy()
    tail["date"] = pd.date_range(
        start=candles["date"].iloc[-1] + pd.Timedelta(hours=1), periods=appended, freq="1h"
    )
    return pd.concat([candles, tail], ignore_index=True)


def build_at(
    path: Path,
    candles: pd.DataFrame | None = None,
    *,
    target_spec: TargetSpec | None = None,
    **scope,
) -> Path:
    """Build and save a processed dataset from candles."""
    frame, metadata = build_dataset(
        make_candles() if candles is None else candles,
        FeatureSpec(),
        target_spec or TargetSpec(horizon=4),
        **{**SCOPE, **scope},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(path, frame, metadata)
    return path


def fingerprint_of(path: Path, contract=CONTRACT):
    """The research-input fingerprint one dataset carries under one contract."""
    data = train.load_research_data(path)
    return data.input_fingerprint(data.sealed_boundary(contract))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    return build_at(tmp_path_factory.mktemp("provenance") / "base.parquet")


@pytest.fixture(scope="module")
def base_fingerprint(dataset):
    return fingerprint_of(dataset)


def corrupted_copy(directory: Path, source: Path, *, row: int, column: str, value) -> Path:
    """A copy of a built dataset with one research-visible value replaced.

    The sidecar is copied verbatim, so the rebuilt dataset still *describes*
    itself exactly as the original does. That is the case that matters: a change
    every metadata field agrees is not a change.
    """
    frame, _ = load_dataset(source)
    frame = frame.copy()
    before = frame.loc[row, column]
    frame.loc[row, column] = value(before) if callable(value) else value
    # Otherwise a "corruption" that happened to write back the value already
    # there would assert that an unchanged dataset has an unchanged identity.
    assert frame.loc[row, column] != before
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.name
    frame.to_parquet(target, index=False)
    sidecar = str(source) + ".meta.json"
    (Path(str(target) + ".meta.json")).write_text(Path(sidecar).read_text())
    return target


# --- A. identity is semantic, not a hash of the file -------------------------
def test_the_same_dataset_hashes_the_same_twice(dataset, base_fingerprint):
    again = fingerprint_of(dataset)
    assert again.research_input_hash == base_fingerprint.research_input_hash
    assert again.full_table_hash == base_fingerprint.full_table_hash
    assert len(base_fingerprint.research_input_hash) == 64
    assert int(base_fingerprint.research_input_hash, 16) >= 0


@pytest.mark.parametrize("compression", ["snappy", "gzip", None], ids=str)
def test_parquet_compression_does_not_define_identity(
    dataset, base_fingerprint, tmp_path, compression
):
    """Different bytes, same table, same research input."""
    frame, _ = load_dataset(dataset)
    rewritten = tmp_path / f"{compression}" / "same.parquet"
    rewritten.parent.mkdir(parents=True)
    frame.to_parquet(rewritten, index=False, compression=compression)
    (Path(str(rewritten) + ".meta.json")).write_text(
        Path(str(dataset) + ".meta.json").read_text()
    )
    assert (
        fingerprint_of(rewritten).research_input_hash == base_fingerprint.research_input_hash
    )


def test_column_order_and_a_stored_index_do_not_define_identity(
    dataset, base_fingerprint, tmp_path
):
    """The file's layout is not the research input; the named values are."""
    frame, _ = load_dataset(dataset)
    reordered = frame[list(reversed(frame.columns))]
    rewritten = tmp_path / "reordered.parquet"
    # index=True adds a stored index column that research never reads.
    reordered.to_parquet(rewritten, index=True)
    (Path(str(rewritten) + ".meta.json")).write_text(
        Path(str(dataset) + ".meta.json").read_text()
    )
    assert (
        fingerprint_of(rewritten).research_input_hash == base_fingerprint.research_input_hash
    )


def test_timestamp_resolution_does_not_define_identity(dataset, base_fingerprint, tmp_path):
    """Microsecond and nanosecond storage name the same instants."""
    frame, _ = load_dataset(dataset)
    frame = frame.copy()
    frame["date"] = frame["date"].astype("datetime64[us, UTC]")
    rewritten = tmp_path / "microseconds.parquet"
    frame.to_parquet(rewritten, index=False)
    (Path(str(rewritten) + ".meta.json")).write_text(
        Path(str(dataset) + ".meta.json").read_text()
    )
    assert (
        fingerprint_of(rewritten).research_input_hash == base_fingerprint.research_input_hash
    )


def test_a_reformatted_sidecar_does_not_define_identity(dataset, base_fingerprint, tmp_path):
    """The header is hashed as canonical JSON, not as the file's text."""
    frame, _ = load_dataset(dataset)
    rewritten = tmp_path / "reformatted.parquet"
    frame.to_parquet(rewritten, index=False)
    payload = json.loads(Path(str(dataset) + ".meta.json").read_text())
    reordered = dict(reversed(list(payload.items())))
    (Path(str(rewritten) + ".meta.json")).write_text(json.dumps(reordered, indent=8))
    assert (
        fingerprint_of(rewritten).research_input_hash == base_fingerprint.research_input_hash
    )


def test_the_market_name_is_case_normalised(dataset, base_fingerprint, tmp_path):
    """``Binance`` and ``binance`` are one market, not two research inputs."""
    frame, _ = load_dataset(dataset)
    rewritten = tmp_path / "recased.parquet"
    frame.to_parquet(rewritten, index=False)
    payload = json.loads(Path(str(dataset) + ".meta.json").read_text())
    payload["exchange"] = "  SYNTHETIC  "
    payload["pair"] = "synth/usdt"
    payload["timeframe"] = "1H"
    (Path(str(rewritten) + ".meta.json")).write_text(json.dumps(payload))
    assert (
        fingerprint_of(rewritten).research_input_hash == base_fingerprint.research_input_hash
    )


def test_the_chunk_size_does_not_define_identity(dataset, base_fingerprint, monkeypatch):
    """Streaming is an implementation detail; the digest is not.

    The columns are hashed a chunk at a time so a dataset far larger than this
    one costs a fixed amount of memory. If a chunk boundary could reach the
    digest, the identity would depend on how much memory the machine had.
    """
    monkeypatch.setattr("nn.data_fingerprint.CHUNK_ROWS", 7)
    assert fingerprint_of(dataset).research_input_hash == base_fingerprint.research_input_hash


def test_negative_zero_and_zero_are_one_number(dataset, tmp_path):
    """IEEE keeps their sign bits apart; arithmetic does not, so nor does this."""
    frame, _ = load_dataset(dataset)
    zeroed = frame.copy()
    zeroed.loc[0, "future_return"] = 0.0
    signed = frame.copy()
    signed.loc[0, "future_return"] = -0.0

    hashes = set()
    for name, variant in (("zero", zeroed), ("negative-zero", signed)):
        path = tmp_path / name / "d.parquet"
        path.parent.mkdir(parents=True)
        variant.to_parquet(path, index=False)
        (Path(str(path) + ".meta.json")).write_text(
            Path(str(dataset) + ".meta.json").read_text()
        )
        hashes.add(fingerprint_of(path).research_input_hash)
    assert len(hashes) == 1


# --- B. the research-visible region, and only it ------------------------------
@pytest.fixture(scope="module")
def grown(tmp_path_factory, dataset) -> Path:
    directory = tmp_path_factory.mktemp("appended")
    return build_at(directory / "grown.parquet", make_candles(appended=400))


def test_the_fixture_really_appends_only_sealed_rows(dataset, grown):
    """A control: without it every assertion below could pass vacuously."""
    base = train.load_research_data(dataset)
    later = train.load_research_data(grown)
    assert later.n_rows > base.n_rows
    appended = pd.to_datetime(pd.Index(later.dates[base.n_rows :]), utc=True)
    assert (appended >= CONTRACT.sealed_test_start).all()


def test_appending_after_the_seal_leaves_the_research_input_alone(
    dataset, grown, base_fingerprint
):
    """Growth after the seal does not change what research was allowed to see."""
    later = fingerprint_of(grown)
    assert later.research_input_hash == base_fingerprint.research_input_hash
    assert later.research_rows == base_fingerprint.research_rows
    assert later.research_end == base_fingerprint.research_end


def test_appending_after_the_seal_does_change_the_whole_table_digest(
    dataset, grown, base_fingerprint
):
    """The two digests answer different questions, and must be free to disagree.

    "Is this the same file" is an audit question. "Is this the same research
    input" is the comparability question. A run whose dataset has since grown
    past the seal is still the same research input, and tying comparability to
    the file would break that for no reason.
    """
    later = fingerprint_of(grown)
    assert later.full_table_hash != base_fingerprint.full_table_hash
    assert later.total_rows > base_fingerprint.total_rows


def test_scrambling_the_sealed_block_leaves_the_research_input_alone(
    dataset, base_fingerprint, tmp_path
):
    """The research digest does not depend on a single sealed value.

    Asserted rather than argued: every label and future return at or after the
    seal is replaced, and the research-visible identity does not move. The
    whole-table digest does, which is exactly why it is audit metadata and never
    a comparability input.
    """
    frame, _ = load_dataset(dataset)
    frame = frame.copy()
    sealed = frame.index[pd.to_datetime(frame["date"], utc=True) >= CONTRACT.sealed_test_start]
    assert len(sealed) > 0
    frame.loc[sealed, "target"] = 0
    frame.loc[sealed, "future_return"] = -0.5

    path = tmp_path / "scrambled.parquet"
    frame.to_parquet(path, index=False)
    (Path(str(path) + ".meta.json")).write_text(Path(str(dataset) + ".meta.json").read_text())

    scrambled = fingerprint_of(path)
    assert scrambled.research_input_hash == base_fingerprint.research_input_hash
    assert scrambled.full_table_hash != base_fingerprint.full_table_hash


# --- C. any meaningful change inside the research region is visible -----------
@pytest.mark.parametrize(
    "column, value",
    [
        # A candle timestamped half an hour late: still sorted, still unique,
        # and still a different observation than the one recorded.
        ("date", lambda before: before + pd.Timedelta(minutes=30)),
        ("close", 12345.678),
        ("ema_cross", 0.123456),
        ("atr_norm", 0.9),
        ("future_return", 0.5),
        ("target", lambda before: (int(before) + 1) % 3),
        ("segment_id", 99),
    ],
)
def test_a_changed_research_visible_value_changes_the_identity(
    dataset, base_fingerprint, tmp_path, column, value
):
    """One value, one row, before the seal. The metadata still agrees."""
    corrupted = corrupted_copy(tmp_path / column, dataset, row=5, column=column, value=value)
    assert (
        fingerprint_of(corrupted).research_input_hash != base_fingerprint.research_input_hash
    )


def test_removing_a_research_visible_row_changes_the_identity(
    dataset, base_fingerprint, tmp_path
):
    """Row membership is part of the input, not merely the row count."""
    frame, _ = load_dataset(dataset)
    shortened = frame.drop(index=5).reset_index(drop=True)
    path = tmp_path / "dropped.parquet"
    shortened.to_parquet(path, index=False)
    (Path(str(path) + ".meta.json")).write_text(Path(str(dataset) + ".meta.json").read_text())
    assert fingerprint_of(path).research_input_hash != base_fingerprint.research_input_hash


def test_a_column_research_never_reads_does_not_change_the_identity(
    dataset, base_fingerprint, tmp_path
):
    """The identity is over what research reads, not over what the file holds."""
    frame, _ = load_dataset(dataset)
    frame = frame.copy()
    frame["scratch_note"] = np.arange(len(frame), dtype=np.float64)
    path = tmp_path / "extra_column.parquet"
    frame.to_parquet(path, index=False)
    (Path(str(path) + ".meta.json")).write_text(Path(str(dataset) + ".meta.json").read_text())
    assert fingerprint_of(path).research_input_hash == base_fingerprint.research_input_hash


def test_dropping_the_segment_ids_changes_the_identity(dataset, base_fingerprint, tmp_path):
    """A dataset that cannot say where its gaps are is a different input.

    Segment ids decide which windows are allowed to exist at all, so a build
    without them is not the same research input with a column missing — it is
    one whose gap handling nobody can reconstruct.
    """
    frame, _ = load_dataset(dataset)
    path = tmp_path / "no_segments.parquet"
    frame.drop(columns=["segment_id"]).to_parquet(path, index=False)
    (Path(str(path) + ".meta.json")).write_text(Path(str(dataset) + ".meta.json").read_text())

    without = fingerprint_of(path)
    assert without.research_input_hash != base_fingerprint.research_input_hash
    assert "segment_id" not in without.columns


# --- D. research specification participates ----------------------------------
def specification_digest(dataset: Path, **overrides) -> str:
    """The digest of one dataset's rows under a stated specification."""
    data = train.load_research_data(dataset)
    boundary = data.sealed_boundary(CONTRACT)
    arguments = {
        "feature_names": data.feature_names,
        "exchange": data.ds_meta.exchange,
        "pair": data.ds_meta.pair,
        "timeframe": data.ds_meta.timeframe,
        "feature_spec": data.ds_meta.feature_spec,
        "target_spec": data.ds_meta.target_spec,
        "rows": boundary.start_row,
    }
    arguments.update(overrides)
    return research_input_digest(data.research_columns(), **arguments)


def test_the_target_spec_participates(dataset, base_fingerprint):
    """Identical numbers, a different definition of what they mean."""
    assert specification_digest(dataset) == base_fingerprint.research_input_hash
    relabelled = specification_digest(
        dataset, target_spec=TargetSpec(horizon=12, fee_rate=0.002).to_dict()
    )
    assert relabelled != base_fingerprint.research_input_hash


def test_the_feature_spec_participates(dataset, base_fingerprint):
    assert (
        specification_digest(dataset, feature_spec=FeatureSpec(ema_fast=3).to_dict())
        != base_fingerprint.research_input_hash
    )


def test_the_feature_order_participates(dataset, base_fingerprint):
    """The model's input width is ordered; a reordered contract is another one."""
    data = train.load_research_data(dataset)
    reordered = list(reversed(data.feature_names))
    assert (
        specification_digest(dataset, feature_names=reordered)
        != base_fingerprint.research_input_hash
    )


def test_the_market_participates(dataset, base_fingerprint):
    assert (
        specification_digest(dataset, pair="ETH/USDT") != base_fingerprint.research_input_hash
    )


def test_a_dataset_missing_a_research_column_has_no_identity(dataset):
    """Refused rather than hashed over whatever survived."""
    data = train.load_research_data(dataset)
    columns = data.research_columns()
    del columns["future_return"]
    with pytest.raises(DataFingerprintError, match="future_return"):
        research_input_digest(
            columns,
            feature_names=data.feature_names,
            exchange=data.ds_meta.exchange,
            pair=data.ds_meta.pair,
            timeframe=data.ds_meta.timeframe,
            feature_spec=data.ds_meta.feature_spec,
            target_spec=data.ds_meta.target_spec,
            rows=10,
        )


# --- E. three identities, three reasons to change ----------------------------
def test_a_correction_before_the_seal_moves_the_data_not_the_contract(
    dataset, base_fingerprint, tmp_path
):
    """The legitimate case invariant 5 names, represented honestly.

    A historical candle is corrected. The research data is different — every row
    index now addresses a different observation — while the research question,
    and therefore the contract, is untouched.
    """
    corrected = corrupted_copy(tmp_path / "fix", dataset, row=9, column="close", value=31000.0)
    data = train.load_research_data(corrected)
    boundary = data.sealed_boundary(CONTRACT)

    assert boundary.contract.contract_hash == CONTRACT.contract_hash
    assert data.input_fingerprint(boundary).research_input_hash != (
        base_fingerprint.research_input_hash
    )


def test_a_repair_that_inserts_a_row_moves_the_resolved_row_too(dataset, base_fingerprint):
    """The resolved row is dataset metadata and may move with a repair.

    Inserting a missing candle before the seal shifts every later index by one.
    The contract does not move, the data identity does, and the row does too —
    three separate things, and none of them implies the others.
    """
    frame, meta = load_dataset(dataset)
    inserted = pd.concat([frame.iloc[:3], frame.iloc[[3]], frame.iloc[3:]], ignore_index=True)
    inserted.loc[3, "date"] = inserted.loc[3, "date"] - pd.Timedelta(minutes=30)

    data = train.load_research_data(dataset)
    base_row = data.sealed_boundary(CONTRACT).start_row
    stamps = pd.to_datetime(inserted["date"], utc=True)
    moved_row = int(stamps.searchsorted(CONTRACT.sealed_test_start, side="left"))

    assert moved_row == base_row + 1
    assert meta.exchange == SCOPE["exchange"]


def test_a_second_contract_over_the_same_rows_shares_the_data_identity(dataset):
    """Contract identity and data identity are independent, in both directions.

    Two generations sealing at the same instant over the same dataset saw the
    same data. That is one fact; which generation they belong to is another, and
    ``nn.wf_diagnostics`` checks both.
    """
    other = parse_contract(
        {
            "contract_id": "synth-usdt-1h-gen2",
            "research_generation": 2,
            "domain": "pipeline-smoke",
            "scope": dict(SCOPE),
            "sealed_test_start": CONTRACT.sealed_test_start.isoformat(),
        }
    )
    data = train.load_research_data(dataset)
    first = data.input_fingerprint(data.sealed_boundary(CONTRACT))
    second = data.input_fingerprint(data.sealed_boundary(other))

    assert other.contract_hash != CONTRACT.contract_hash
    assert first.research_input_hash == second.research_input_hash


# --- F. new artifacts carry it -----------------------------------------------
def test_training_records_the_research_input(dataset, tmp_path, base_fingerprint):
    models_dir = tmp_path / "models"
    assert (
        train.main(
            ["--dataset", str(dataset), "--models-dir", str(models_dir), "--validation-only"]
            + TINY
        )
        == 0
    )
    version = next(p for p in models_dir.iterdir() if p.is_dir())

    report = json.loads((version / "report.json").read_text())
    recorded = report["research_input"]
    assert recorded["fingerprint_schema"] == RESEARCH_INPUT_SCHEMA
    assert recorded["research_input_hash"] == base_fingerprint.research_input_hash
    assert recorded["research_rows"] == report["sealed_test"]["start_row"]
    # The contract block is still there, and is still a different field.
    assert report["sealed_test"]["research_contract"]["contract_id"] == SYNTHETIC_CONTRACT_ID
    assert report["sealed_test"]["evaluated"] is False

    metadata = json.loads((version / "metadata.json").read_text())
    assert metadata["research_input_hash"] == base_fingerprint.research_input_hash
    assert metadata["research_contract_hash"] == CONTRACT.contract_hash


def test_experiment_records_it_and_hashes_it_into_the_plan(dataset, tmp_path):
    """A grid re-searched over corrected data is a new plan, not a second opinion."""
    corrected = corrupted_copy(
        tmp_path / "corrected", dataset, row=11, column="close", value=29500.0
    )

    def plan_for(path: Path) -> dict:
        args = experiment.build_argparser().parse_args(
            ["--dataset", str(path), *UNDER_SYNTHETIC]
        )
        data = train.load_research_data(path)
        boundary = data.sealed_boundary(CONTRACT)
        configs = experiment.build_grid(
            {
                name: [getattr(experiment.RunConfig(), name)]
                for name in experiment.GRID_DIMENSIONS
            },
            experiment.RunConfig(),
        )
        plan = chronological_split(data.n_rows, boundary.start_row)
        return experiment.build_plan(
            args, configs, data, plan, boundary, data.input_fingerprint(boundary), []
        )

    original, rebuilt = plan_for(dataset), plan_for(corrected)
    assert original["research_input"]["fingerprint_schema"] == RESEARCH_INPUT_SCHEMA
    assert (
        original["research_input"]["research_input_hash"]
        != rebuilt["research_input"]["research_input_hash"]
    )
    assert original["plan_hash"] != rebuilt["plan_hash"]


@pytest.fixture(scope="module")
def walk_forward_run(tmp_path_factory, dataset) -> Path:
    out = tmp_path_factory.mktemp("wf") / "run"
    assert (
        walkforward.main(["--dataset", str(dataset), "--out", str(out), "--folds", "2", *TINY])
        == 0
    )
    return out


def test_walk_forward_records_it_and_names_it_in_the_summary(
    walk_forward_run, base_fingerprint
):
    payload = json.loads((walk_forward_run / "walkforward.json").read_text())
    recorded = payload["research_input"]
    assert recorded["research_input_hash"] == base_fingerprint.research_input_hash
    assert recorded["research_rows"] == payload["sealed_test"]["start_row"]
    assert payload["test_evaluated"] is False

    markdown = (walk_forward_run / "walkforward.md").read_text()
    assert base_fingerprint.research_input_hash in markdown


def test_the_recorded_identity_needs_no_local_path(walk_forward_run):
    """Provenance travels with the artifact, not with the machine it ran on."""
    payload = json.loads((walk_forward_run / "walkforward.json").read_text())
    recorded = payload["research_input"]
    assert set(wf_diagnostics.RESEARCH_INPUT_FIELDS) <= set(recorded)
    assert not any(
        isinstance(value, str) and value.startswith("/") for value in recorded.values()
    )


# --- G. diagnostics: comparability follows the data, history stays honest -----
HISTORICAL = REPO / "artifacts" / "walkforward" / "btc_nested_v1" / "walkforward.json"


def input_block(digest: str, rows: int, **overrides) -> dict:
    """A well-formed research-input block, before any deliberate damage."""
    block = {
        "fingerprint_schema": RESEARCH_INPUT_SCHEMA,
        "research_input_hash": digest,
        "full_table_hash": "f" * 64,
        "research_rows": rows,
        "total_rows": rows + 100,
        "research_start": "2020-01-01T00:00:00+00:00",
        "research_end": "2025-08-27T22:00:00+00:00",
        "columns": ["date", "close", "target"],
    }
    block.update(overrides)
    return block


def run_copy(directory: Path, *, research_input: dict | None = None) -> Path:
    """A copy of a committed run, optionally re-issued with provenance.

    The committed artifact is only ever read. Everything below writes into
    copies, which is how the fingerprint-era shape gets tested against a real
    artifact without touching the historical record.
    """
    payload = json.loads(HISTORICAL.read_text())
    payload["sealed_test"]["anchor_timestamp"] = CONTRACT.sealed_test_start.isoformat()
    payload["sealed_test"]["research_contract"] = CONTRACT.provenance()
    if research_input is not None:
        payload["research_input"] = research_input
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "walkforward.json").write_text(json.dumps(payload, indent=2))
    return directory


@pytest.fixture
def sealed_row() -> int:
    return int(json.loads(HISTORICAL.read_text())["sealed_test"]["start_row"])


def test_runs_reading_the_same_data_stay_comparable(tmp_path, sealed_row):
    block = input_block("a" * 64, sealed_row)
    runs = [
        wf_diagnostics.load_run(run_copy(tmp_path / name, research_input=dict(block)))
        for name in ("seed_a", "seed_b")
    ]
    assert wf_diagnostics.compare_runs(runs) == []
    assert all(not wf_diagnostics.audit_run(run) for run in runs)


def test_runs_reading_different_data_are_refused(tmp_path, sealed_row):
    """Same contract, same geometry, same dates. Different candles."""
    runs = [
        wf_diagnostics.load_run(
            run_copy(tmp_path / name, research_input=input_block(digest * 64, sealed_row))
        )
        for name, digest in (("seed_a", "a"), ("seed_b", "b"))
    ]
    assert runs[0].contract_hash == runs[1].contract_hash
    assert runs[0].geometry == runs[1].geometry
    assert runs[0].sealed_test_start == runs[1].sealed_test_start

    problems = wf_diagnostics.compare_runs(runs)
    assert any("did not see the same research data" in problem for problem in problems)


def test_a_fingerprinted_run_is_not_comparable_with_one_that_recorded_none(
    tmp_path, sealed_row
):
    """Absence is a real difference, and is reported as absence."""
    runs = [
        wf_diagnostics.load_run(
            run_copy(tmp_path / "with", research_input=input_block("a" * 64, sealed_row))
        ),
        wf_diagnostics.load_run(run_copy(tmp_path / "without")),
    ]
    problems = wf_diagnostics.compare_runs(runs)
    assert any("no research-input fingerprint (dataset metadata only)" in p for p in problems)


@pytest.mark.parametrize(
    "damage, expected",
    [
        ({"fingerprint_schema": "something/else"}, "fingerprint_schema"),
        ({"research_input_hash": "short"}, "research_input_hash"),
        ({"full_table_hash": 17}, "full_table_hash"),
        ({"research_rows": 1}, "research_rows"),
    ],
)
def test_an_untrustworthy_fingerprint_block_is_an_integrity_fault(
    tmp_path, sealed_row, damage, expected
):
    """Present but unreliable is a fault; absent is not."""
    block = input_block("a" * 64, sealed_row, **damage)
    run = wf_diagnostics.load_run(run_copy(tmp_path / "damaged", research_input=block))
    problems = wf_diagnostics.audit_run(run)
    assert any(expected in problem for problem in problems)


def test_a_partial_fingerprint_block_is_a_fault(tmp_path, sealed_row):
    block = input_block("a" * 64, sealed_row)
    del block["research_input_hash"]
    run = wf_diagnostics.load_run(run_copy(tmp_path / "partial", research_input=block))
    assert any("missing" in problem for problem in wf_diagnostics.audit_run(run))


def test_the_committed_artifacts_have_no_fingerprint_and_are_never_given_one():
    """They predate this work. Reading them must not invent provenance."""
    all_artifacts = sorted((REPO / "artifacts" / "walkforward").glob("*/walkforward.json"))
    legacy_artifacts = [a for a in all_artifacts if "_v4_" not in a.parent.name]
    assert legacy_artifacts, "expected at least one pre-fingerprint committed artifact"

    for artifact in legacy_artifacts:
        payload = json.loads(artifact.read_text())
        assert "research_input" not in payload

        run = wf_diagnostics.load_run(artifact)
        assert run.research_input is None
        assert run.research_input_hash is None
        # Absent provenance is not an integrity problem, and is described as
        # what it is rather than as somebody else's identity.
        assert not any(
            "research_input" in problem for problem in wf_diagnostics.audit_run(run)
        )
        assert (
            wf_diagnostics._research_input_label(run)
            == "no research-input fingerprint (dataset metadata only)"
        )


def test_the_committed_v4_artifacts_carry_current_research_input_provenance():
    """v4 is current-generation work. Its provenance is real and must be read as such."""
    v4_artifacts = sorted((REPO / "artifacts" / "walkforward").glob("*_v4_*/walkforward.json"))
    assert v4_artifacts, "expected the committed v4 walkforward artifacts"

    for artifact in v4_artifacts:
        payload = json.loads(artifact.read_text())
        assert "research_input" in payload

        run = wf_diagnostics.load_run(artifact)
        assert run.research_input is not None
        assert run.research_input_hash
        assert not any(
            "research_input" in problem for problem in wf_diagnostics.audit_run(run)
        )
        assert wf_diagnostics._research_input_label(run) == (
            f"research input sha256:{str(run.research_input_hash)[:16]}"
        )


# --- H. the dataset a reader supplies must be the data the run read -----------
def test_the_regime_loader_accepts_the_dataset_the_run_was_produced_against(
    dataset, base_fingerprint
):
    data = train.load_research_data(dataset)
    boundary = data.sealed_boundary(CONTRACT)
    frame = load_research_frame(
        dataset,
        sealed_test_start=boundary.start_row,
        identity={**data.ds_meta.to_dict(), "rows": data.n_rows},
        research_input=base_fingerprint.to_dict(),
    )
    assert len(frame.frame) == boundary.start_row


def test_the_regime_loader_refuses_a_dataset_that_only_describes_itself_the_same(
    dataset, base_fingerprint, tmp_path
):
    """The metadata checks pass. The values do not, and that is the whole point."""
    corrected = corrupted_copy(
        tmp_path / "rebuilt", dataset, row=7, column="close", value=28000.0
    )
    data = train.load_research_data(dataset)
    boundary = data.sealed_boundary(CONTRACT)
    with pytest.raises(RegimeDataError, match="not the research input"):
        load_research_frame(
            corrected,
            sealed_test_start=boundary.start_row,
            identity={**data.ds_meta.to_dict(), "rows": data.n_rows},
            research_input=base_fingerprint.to_dict(),
        )


def test_the_regime_loader_refuses_a_fingerprint_from_another_schema(
    dataset, base_fingerprint
):
    data = train.load_research_data(dataset)
    recorded = {**base_fingerprint.to_dict(), "fingerprint_schema": "chimera.something/9"}
    with pytest.raises(RegimeDataError, match="different definitions"):
        load_research_frame(
            dataset,
            sealed_test_start=data.sealed_boundary(CONTRACT).start_row,
            research_input=recorded,
        )


def test_the_regime_loader_still_reads_a_dataset_for_a_run_that_recorded_none(dataset):
    """Historical runs stay analysable; they simply cannot be verified."""
    data = train.load_research_data(dataset)
    frame = load_research_frame(
        dataset,
        sealed_test_start=data.sealed_boundary(CONTRACT).start_row,
        research_input=None,
    )
    assert len(frame.frame) > 0


# --- I. raw candles are identified too ---------------------------------------
@pytest.fixture(scope="module")
def raw_candles(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("raw") / "SYNTH_USDT_1h.parquet"
    make_candles().to_parquet(path, index=False)
    return path


def raw_fingerprint(path: Path, through: pd.Timestamp):
    return fingerprint_raw_input(load_raw_ohlcv(path), **SCOPE, visible_through=through)


def test_raw_identity_is_bounded_by_the_research_region(tmp_path, raw_candles):
    """Candles the diagnostics never looked at do not change what it looked at."""
    through = pd.Timestamp("2025-08-01T00:00:00+00:00")
    extended = tmp_path / "extended.parquet"
    make_candles(appended=400).to_parquet(extended, index=False)

    assert (
        raw_fingerprint(raw_candles, through).raw_input_hash
        == raw_fingerprint(extended, through).raw_input_hash
    )


def test_raw_identity_changes_when_a_covered_candle_changes(tmp_path, raw_candles):
    through = pd.Timestamp("2025-08-01T00:00:00+00:00")
    candles = make_candles()
    candles.loc[4, "volume"] = candles.loc[4, "volume"] + 1.0
    altered = tmp_path / "altered.parquet"
    candles.to_parquet(altered, index=False)

    assert (
        raw_fingerprint(raw_candles, through).raw_input_hash
        != raw_fingerprint(altered, through).raw_input_hash
    )


def test_raw_identity_refuses_unsorted_candles(raw_candles):
    """Rows before the bound are a prefix only in a sorted table."""
    shuffled = load_raw_ohlcv(raw_candles).iloc[::-1]
    with pytest.raises(DataFingerprintError, match="not sorted ascending"):
        fingerprint_raw_input(
            shuffled, **SCOPE, visible_through=pd.Timestamp("2025-08-01T00:00:00+00:00")
        )


def test_raw_identity_refuses_a_bound_before_the_first_candle(raw_candles):
    with pytest.raises(DataFingerprintError, match="no raw candle"):
        raw_fingerprint(raw_candles, pd.Timestamp("1999-01-01T00:00:00+00:00"))


def test_diagnostics_records_the_raw_input_it_read(
    walk_forward_run, dataset, raw_candles, tmp_path
):
    """When raw candles inform the statistics, the artifact says which ones."""
    out = tmp_path / "diagnostics"
    assert (
        wf_diagnostics.main(
            [
                str(walk_forward_run),
                "--dataset",
                str(dataset),
                "--raw",
                str(raw_candles),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    report = json.loads((out / "regime_diagnostics.json").read_text())
    analysis = report["analysis"]
    assert analysis["research_input_verified"] is True
    assert analysis["raw_input"]["fingerprint_schema"] == RAW_INPUT_SCHEMA
    assert len(analysis["raw_input"]["raw_input_hash"]) == 64
    assert report["sealed_test_evaluated"] is False
    assert (out / "regime_diagnostics.md").read_text().count("Raw input:") == 1


def test_diagnostics_refuses_a_dataset_that_is_not_what_the_run_read(
    walk_forward_run, dataset, tmp_path
):
    """The end-to-end version of the loader check, through the real CLI."""
    rebuilt = corrupted_copy(
        tmp_path / "rebuilt", dataset, row=6, column="close", value=27000.0
    )
    with pytest.raises(SystemExit, match="not the research input"):
        wf_diagnostics.main([str(walk_forward_run), "--dataset", str(rebuilt)])


def test_the_regime_frame_never_sees_a_sealed_row(walk_forward_run, dataset):
    """The verification path reads the research half only, by construction."""
    payload = json.loads((walk_forward_run / "walkforward.json").read_text())
    frame = load_research_frame(
        dataset,
        sealed_test_start=payload["sealed_test"]["start_row"],
        research_input=payload["research_input"],
    )
    stamps = pd.to_datetime(frame.frame["date"], utc=True)
    assert (stamps < CONTRACT.sealed_test_start).all()


# --- timestamp resolution is not part of a research input's identity ---------
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_the_same_candles_hash_alike_at_every_timestamp_resolution(unit):
    """A Parquet writer's choice of time unit is an encoding, not a value.

    This is the failure that reached CI: `asi8` reports the index's *own*
    resolution, pandas 2 coerced everything to nanoseconds on the way in, and
    pandas 3 preserves what the file stored. The same committed candles then
    hashed to two different identities depending on which pandas read them —
    so a research input's identity depended on the reader's library version,
    which is the one thing a semantic fingerprint exists to rule out.
    """
    dates = pd.date_range("2021-03-04", periods=64, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.linspace(100.0, 110.0, 64),
            "high": np.linspace(101.0, 111.0, 64),
            "low": np.linspace(99.0, 109.0, 64),
            "close": np.linspace(100.5, 110.5, 64),
            "volume": np.linspace(10.0, 20.0, 64),
        },
        index=dates.as_unit(unit),
    )
    fingerprint = fingerprint_raw_input(
        frame,
        exchange="binance",
        pair="BTC/USDT",
        timeframe="1h",
        visible_through=frame.index[-1],
    )
    # The nanosecond digest is the one every committed artifact was produced
    # under, so it is the value the other three resolutions have to agree with.
    reference = fingerprint_raw_input(
        frame.set_index(pd.DatetimeIndex(frame.index).as_unit("ns")),
        exchange="binance",
        pair="BTC/USDT",
        timeframe="1h",
        visible_through=frame.index[-1],
    )
    assert fingerprint.raw_input_hash == reference.raw_input_hash


def test_a_changed_price_still_changes_the_identity_after_the_resolution_fold():
    """The fold must hide encoding differences only, never a value difference."""
    dates = pd.date_range("2021-03-04", periods=16, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.full(16, 100.0),
            "high": np.full(16, 101.0),
            "low": np.full(16, 99.0),
            "close": np.full(16, 100.5),
            "volume": np.full(16, 10.0),
        },
        index=dates,
    )
    moved = frame.copy()
    moved.iloc[7, moved.columns.get_loc("close")] = 100.5000001

    def digest(f):
        return fingerprint_raw_input(
            f,
            exchange="binance",
            pair="BTC/USDT",
            timeframe="1h",
            visible_through=f.index[-1],
        ).raw_input_hash

    assert digest(frame) != digest(moved)
