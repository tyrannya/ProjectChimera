"""Shared fixtures.

Two things are deliberately global here:

* the repository root goes on ``sys.path`` so ``chimera``/``nn``/``strategies``
  import the same way they do at runtime;
* nothing stubs out a third-party module. The previous suite replaced
  ``freqtrade`` with a hand-written ``types.ModuleType`` that defined a
  ``TemporaryStopException`` the real library does not have, so the tests
  asserted against a fictional dependency and stayed green while the code was
  unimportable. Tests that need Freqtrade import the real Freqtrade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.export_trade_snapshot import MANIFEST_NAME as MANIFEST_FILENAME  # noqa: E402
from tools.make_sample_data import generate_candles  # noqa: E402


@pytest.fixture
def candles():
    """1000 deterministic synthetic candles satisfying the OHLC invariants."""
    return generate_candles(rows=1000, seed=1234)


@pytest.fixture
def small_candles():
    """400 candles: enough for a short window, small enough to stay fast."""
    return generate_candles(rows=400, seed=99)


# --------------------------------------------------------------------------- #
# P3 fixtures: a complete, verifiable pair of research snapshots on disk
#
# Everything the P3 machinery reads is verified before it is used, and every
# verification is a claim about files. So the only way to exercise a rejection
# is to build a snapshot that would otherwise pass and then break exactly one
# thing about it — which the public Binance archive cannot be asked to do.
#
# These fixtures build that pair from `tools.make_sample_data`: a research
# snapshot of synthetic candles under the committed *synthetic* contract, and a
# trade snapshot of synthetic trades aligned to it. They are fixtures, not
# research data. Nothing measured on them is a result, no committed P3 evidence
# came from them, and they only ever live under `tmp_path`.
#
# The trade snapshot binds to the OHLCV snapshot's semantic hash, so a fixture
# built here cannot be mistaken for the committed BTC source by the verifier
# even if someone copied it into `data/research/`: it would be rejected for
# being aligned to candles that are not the committed ones.
# --------------------------------------------------------------------------- #
SYNTHETIC_HORIZON = 6


@pytest.fixture(scope="session")
def synthetic_research_snapshot(tmp_path_factory):
    """An OHLCV research snapshot that passes `verify_research_snapshot` in full."""
    import hashlib
    import json

    import pandas as pd

    from chimera.contracts import HOLD_IDX, LONG_IDX, SHORT_IDX, TargetSpec
    from chimera.features import FeatureSpec
    from nn.data_fingerprint import fingerprint_raw_input, research_input_digest
    from nn.data_pipeline import DatasetMetadata, build_dataset, save_dataset
    from nn.regime import load_raw_ohlcv
    from nn.research_contract import SYNTHETIC_CONTRACT_ID, load_contract
    from nn.walkforward import plan_nested_folds
    from tools.export_research_snapshot import SNAPSHOT_SCHEMA

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    contract = load_contract(SYNTHETIC_CONTRACT_ID)
    root = tmp_path_factory.mktemp("chimera-research")
    research = root / "data" / "research"
    research.mkdir(parents=True)

    candles = generate_candles(rows=1600, seed=5, start="2021-01-01")
    frame, ds_meta = build_dataset(
        candles,
        FeatureSpec(),
        TargetSpec(horizon=SYNTHETIC_HORIZON),
        exchange="synthetic",
        pair="SYNTH/USDT",
        timeframe="1h",
    )
    research_rows = len(frame)
    outer = max(int(research_rows * 0.10), 1)
    folds = plan_nested_folds(
        research_rows, 4, max(int(research_rows * 0.45), 1), outer, outer, outer
    )
    outer_end = folds[-1].outer.end
    assert outer_end + SYNTHETIC_HORIZON <= research_rows, "fixture is too short"

    processed = frame.iloc[:outer_end].copy()
    processed_dates = pd.DatetimeIndex(pd.to_datetime(processed["date"], utc=True))
    counts = processed["target"].value_counts().to_dict()
    processed_path = research / "synthetic_outer_coverage.parquet"
    save_dataset(
        processed_path,
        processed,
        DatasetMetadata(
            exchange=ds_meta.exchange,
            pair=ds_meta.pair,
            timeframe=ds_meta.timeframe,
            rows=len(processed),
            start=processed_dates[0].isoformat(),
            end=processed_dates[-1].isoformat(),
            feature_names=list(ds_meta.feature_names),
            feature_spec=dict(ds_meta.feature_spec),
            target_spec=dict(ds_meta.target_spec),
            class_balance={
                "SHORT": int(counts.get(SHORT_IDX, 0)),
                "HOLD": int(counts.get(HOLD_IDX, 0)),
                "LONG": int(counts.get(LONG_IDX, 0)),
            },
            validation=dict(ds_meta.validation),
            created_at=ds_meta.created_at,
        ),
    )
    metadata_path = processed_path.with_suffix(processed_path.suffix + ".meta.json")

    raw_path = research / "synthetic_raw_pre_styx.parquet"
    candles.to_parquet(raw_path, index=False)
    raw = load_raw_ohlcv(raw_path)

    manifest_path = research / "synthetic_snapshot_manifest.json"
    manifest = {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "contract": contract.provenance(),
        "contains_styx": False,
        "canonical_reference": {
            "artifact": "synthetic fixture; no frozen walk-forward artifact exists",
            "research_input_hash": research_input_digest(
                {name: frame[name] for name in frame.columns},
                feature_names=ds_meta.feature_names,
                exchange=ds_meta.exchange,
                pair=ds_meta.pair,
                timeframe=ds_meta.timeframe,
                feature_spec=ds_meta.feature_spec,
                target_spec=ds_meta.target_spec,
                rows=research_rows,
            ),
            "research_rows": research_rows,
            "max_outer_end_row": outer_end,
            "outer_to_styx_margin_rows": research_rows - outer_end,
            "target_horizon": SYNTHETIC_HORIZON,
        },
        "raw_pre_styx": {
            "path": str(raw_path.relative_to(root)),
            "rows": len(raw),
            "start": raw.index[0].isoformat(),
            "end": raw.index[-1].isoformat(),
            "semantic_hash": fingerprint_raw_input(
                raw,
                exchange=ds_meta.exchange,
                pair=ds_meta.pair,
                timeframe=ds_meta.timeframe,
                visible_through=raw.index[-1],
            ).raw_input_hash,
            "sha256": sha256(raw_path),
            "columns": ["date", "open", "high", "low", "close", "volume"],
        },
        "processed_outer_coverage": {
            "path": str(processed_path.relative_to(root)),
            "rows": len(processed),
            "row_range": [0, outer_end],
            "start": processed_dates[0].isoformat(),
            "end": processed_dates[-1].isoformat(),
            "semantic_prefix_hash": research_input_digest(
                {name: processed[name] for name in processed.columns},
                feature_names=ds_meta.feature_names,
                exchange=ds_meta.exchange,
                pair=ds_meta.pair,
                timeframe=ds_meta.timeframe,
                feature_spec=ds_meta.feature_spec,
                target_spec=ds_meta.target_spec,
                rows=len(processed),
            ),
            "sha256": sha256(processed_path),
            "metadata_path": str(metadata_path.relative_to(root)),
            "metadata_sha256": sha256(metadata_path),
        },
        "safety_checks": {
            "canonical_research_input_verified": True,
            "raw_strictly_before_styx": True,
            "processed_strictly_before_styx": True,
            "outer_horizon_within_research": True,
            "styx_rows_exported": 0,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "root": root,
        "manifest": manifest_path,
        "contract": contract,
        "candles": candles,
        "spine": processed,
        "ds_meta": ds_meta,
        "folds": folds,
        "research_rows": research_rows,
        "outer_end": outer_end,
    }


@pytest.fixture(scope="session")
def synthetic_trade_snapshot(synthetic_research_snapshot):
    """A trade snapshot aligned to `synthetic_research_snapshot`, verifiable in full."""
    import pandas as pd

    from nn.trade_aggregates import HourlyTradeAggregator
    from tools.export_trade_snapshot import acquisition_window, write_snapshot
    from tools.make_sample_data import trades_reproducing

    fixture = synthetic_research_snapshot
    contract = fixture["contract"]
    candles = fixture["candles"]
    start = pd.Timestamp(candles["date"].iloc[0]).tz_convert("UTC")

    # `trades_reproducing`, not `generate_trades`: the two generators draw
    # independent price paths, so a fixture built from both holds two recordings
    # of two different markets and cannot exercise the verifier's
    # `cross_source_values` check at all.
    ts, price, qty, maker, ids = trades_reproducing(candles, seed=23, trades_per_hour=6)
    aggregator = HourlyTradeAggregator(styx_ns=int(contract.sealed_test_start.value))
    aggregator.add_chunk(ts, price, qty, maker, ids)
    aggregates = aggregator.finish()

    _, window_end, alignment_source = acquisition_window(fixture["manifest"], contract)
    manifest = write_snapshot(
        aggregates,
        aggregator.stats,
        [
            {
                "name": "SYNTHETIC-FIXTURE-aggTrades.zip",
                "url": "fixture://synthetic",
                "bytes": int(len(ts)),
                "sha256": "0" * 64,
                "published_sha256": None,
                "checksum_verified": False,
                "epoch_units": ["ns"],
            }
        ],
        contract=contract,
        out_dir=fixture["root"] / "data" / "research",
        ohlcv_manifest=fixture["manifest"],
        alignment_source=alignment_source,
        requested_from=start,
        requested_through=window_end,
        acquired_at="1970-01-01T00:00:00Z",
    )
    return {
        "root": fixture["root"],
        "manifest": fixture["root"] / "data" / "research" / MANIFEST_FILENAME,
        "aggregates": aggregates,
        "payload": manifest,
        "trades": (ts, price, qty, maker, ids),
    }


# --------------------------------------------------------------------------- #
# P4 fixtures: the committed research spine, plus a *synthetic* derivatives source
#
# Two things are deliberate and neither is an accident of convenience.
#
# The **spine is the real one**. P4's guards are about the committed geometry —
# the snapshot stops at row 45802, P4-HOLD begins at the next hour, the four
# outer blocks are the burned ones — and a fixture on synthetic candles would
# exercise none of that: `check_holdout_boundary` compares the ledger's declared
# instant against the committed snapshot's own last hour, and on a synthetic
# spine it would either be meaningless or wrong. So the tree below is a copy of
# `data/research/`, and the tests run against the geometry P4 will actually run
# against.
#
# The **derivatives source is synthetic**, and it is generated here rather than
# fetched. It has to be: every rejection the acquisition and the verifier can
# raise is a claim about a file, and the public Binance archive cannot be asked
# to serve a corrupt one. Nothing measured on it is a result, no P4 evidence
# comes from it, and it only ever lives under `tmp_path`.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def p4_tree(tmp_path_factory):
    """A repository root holding a copy of the committed research snapshot."""
    import shutil

    root = tmp_path_factory.mktemp("chimera-p4")
    research = root / "data" / "research"
    research.mkdir(parents=True)
    for name in (
        "btc_usdt_1h_gen1_snapshot_manifest.json",
        "btc_usdt_1h_gen1_raw_pre_styx.parquet",
        "btc_usdt_1h_gen1_ohlcv14_outer_coverage.parquet",
        "btc_usdt_1h_gen1_ohlcv14_outer_coverage.parquet.meta.json",
        "p4_holdout_ledger.json",
        "p4_stage1_authorisation.json",
    ):
        shutil.copy(ROOT / "data" / "research" / name, research / name)
    return root


@pytest.fixture(scope="session")
def p4_spine(p4_tree):
    """The committed spine and raw candles, read once."""
    import pandas as pd

    research = p4_tree / "data" / "research"
    return {
        "manifest": research / "btc_usdt_1h_gen1_snapshot_manifest.json",
        "spine": pd.read_parquet(research / "btc_usdt_1h_gen1_ohlcv14_outer_coverage.parquet"),
        "raw": pd.read_parquet(research / "btc_usdt_1h_gen1_raw_pre_styx.parquet"),
        "research": research,
    }


def synthetic_derivatives_hourly(
    start, end, raw_candles, *, oi_from="2020-09-01", missing_days=(), seed=7
):
    """A derivatives source in the committed schema, over a contiguous hourly grid.

    Deterministic, and shaped like the real thing rather than like noise: funding
    settles at 00:00/08:00/16:00 UTC, open interest is observed every hour from
    ``oi_from`` onwards, and the perpetual close sits a few basis points above the
    spot candle so that the verifier's cross-source check has something true to
    verify. ``missing_days`` removes whole UTC days of open interest, which is
    exactly what a 404 on the daily metrics archive does.
    """
    import numpy as np
    import pandas as pd

    from nn.derivatives_sources import (
        DERIVATIVES_COLUMNS,
        HOUR_NS,
        UNAVAILABLE_AGE_NS,
        staleness_bound_ns,
    )

    hours = np.arange(int(pd.Timestamp(start).value), int(pd.Timestamp(end).value), HOUR_NS)
    dates = pd.DatetimeIndex(pd.to_datetime(hours, unit="ns", utc=True))
    n = len(hours)
    rng = np.random.default_rng(seed)

    settled = (dates.hour % 8 == 0).astype(np.int64)
    rates = np.where(settled == 1, rng.normal(0.0001, 0.00015, n), 0.0)
    last_rate = np.zeros(n)
    age = np.full(n, UNAVAILABLE_AGE_NS, dtype=np.int64)
    current, last_index, seen = 0.0, None, 0
    for i in range(n):
        if settled[i]:
            current, last_index, seen = float(rates[i]), i, seen + 1
        if last_index is not None:
            age[i] = (i - last_index) * HOUR_NS
        last_rate[i] = current
    funding_available = (age >= 0) & (age <= staleness_bound_ns("funding_rate"))

    spot = pd.Series(
        np.asarray(raw_candles["close"], dtype=np.float64),
        index=pd.DatetimeIndex(pd.to_datetime(raw_candles["date"], utc=True)),
    ).reindex(dates)
    close = spot.to_numpy(dtype=np.float64)
    filled = pd.Series(close).ffill().bfill().to_numpy(dtype=np.float64)
    basis = rng.normal(0.0004, 0.0006, n)
    perp_close = filled * (1.0 + basis)

    oi_start = pd.Timestamp(oi_from, tz="UTC")
    oi_available = np.asarray(dates >= oi_start, dtype=bool)
    for day in missing_days:
        stamp = pd.Timestamp(day, tz="UTC").normalize()
        inside = np.asarray(
            (dates >= stamp) & (dates < stamp + pd.Timedelta(days=1)), dtype=bool
        )
        oi_available &= ~inside
    oi_contracts = 40_000.0 + np.cumsum(rng.normal(0.0, 40.0, n))
    oi_notional = oi_contracts * filled

    columns = {
        "date": dates,
        "funding_settled": settled,
        "funding_settled_rate": np.where(settled == 1, rates, 0.0),
        "funding_visible_count": np.cumsum(settled),
        "funding_available": funding_available.astype(np.int64),
        "funding_last_rate": np.where(funding_available, last_rate, 0.0),
        "funding_age_ns": np.where(funding_available, age, UNAVAILABLE_AGE_NS),
        "oi_available": oi_available.astype(np.int64),
        "oi_contracts": np.where(oi_available, oi_contracts, 0.0),
        "oi_notional": np.where(oi_available, oi_notional, 0.0),
        "oi_age_ns": np.where(oi_available, 0, UNAVAILABLE_AGE_NS),
        "perp_available": np.ones(n, dtype=np.int64),
        "perp_close": perp_close,
        "perp_age_ns": np.zeros(n, dtype=np.int64),
    }
    return pd.DataFrame({name: columns[name] for name in DERIVATIVES_COLUMNS})


@pytest.fixture(scope="session")
def p4_hourly(p4_spine):
    """A synthetic derivatives source covering the committed spine and its warm-up."""
    from nn.research_contract import load_contract
    from tools.export_derivatives_snapshot import acquisition_window

    start, end, _ = acquisition_window(p4_spine["manifest"], load_contract("btc-usdt-1h-gen1"))
    return synthetic_derivatives_hourly(start, end, p4_spine["raw"])


@pytest.fixture(scope="session")
def p4_universe(p4_spine, p4_hourly):
    """The sample universe the synthetic source produces on the committed spine."""
    from nn.p4_universe import build_universe

    return build_universe(p4_spine["spine"], p4_hourly, p4_spine["raw"])
