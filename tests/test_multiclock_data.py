"""The multi-clock foundation's integrity checks, and a positive control for each.

Every property below is checked twice: once on the committed source, where it
must hold, and once on a deliberately broken copy, where the check must fire.
The second half is the load-bearing one. **A check that has never failed is not
evidence** — it is a line that has always happened to be true, and a resampler
is exactly the kind of code whose bugs are invisible in the output: a grid
shifted by one minute, a bar completed from the period after it, or a boundary
crossed by a single row all produce a frame that sorts, plots and trains without
complaint.

The five mutations `docs/current_development_plan.md` requires are each here:
a higher-timeframe bar shifted one period into the future, an incomplete bar
injected, the research boundary crossed, a constituent candle corrupted, and an
alignment boundary moved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nn.multiclock import (
    ALL_CLOCKS,
    CLOCKS,
    RESEARCH_VISIBLE_END,
    STYX_START,
    MulticlockError,
    assert_manifest_clock,
    assert_minute_grid,
    bar_availability,
    candle_digest,
    constituent_count,
    minute_gaps,
    parity_against,
    resample_from_minutes,
)
from tools import verify_multiclock_snapshot as verifier
from tools.acquire_multiclock_source import MANIFEST_NAME

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "research" / MANIFEST_NAME


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def minutes(manifest) -> pd.DataFrame:
    return pd.read_parquet(REPO_ROOT / manifest["minutes"]["path"])


@pytest.fixture(scope="module")
def synthetic_minutes() -> pd.DataFrame:
    """Four clean hours of minute candles, well inside the research region.

    Synthetic on purpose: the mutations below have to break exactly one property
    at a time, and the committed archive cannot be asked to do that.
    """
    start = pd.Timestamp("2021-06-01T00:00:00+00:00")
    index = pd.date_range(start, periods=240, freq="1min", tz="UTC")
    close = pd.Series(range(len(index)), dtype="float64") + 100.0
    return pd.DataFrame(
        {
            "date": index,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": pd.Series(range(len(index)), dtype="float64") + 1.0,
        }
    )


# --------------------------------------------------------------------------- #
# The committed source
# --------------------------------------------------------------------------- #


def test_committed_source_is_a_clean_minute_grid(minutes):
    dates = assert_minute_grid(minutes)
    assert dates.is_monotonic_increasing
    assert not dates.duplicated().any()
    assert (dates.dt.second == 0).all()


def test_committed_source_stops_before_the_research_boundary(minutes):
    dates = pd.to_datetime(minutes["date"], utc=True)
    assert dates.max() < RESEARCH_VISIBLE_END
    assert int((dates >= RESEARCH_VISIBLE_END).sum()) == 0


def test_committed_source_never_reaches_styx(minutes):
    dates = pd.to_datetime(minutes["date"], utc=True)
    assert dates.max() < STYX_START
    assert RESEARCH_VISIBLE_END < STYX_START


def test_p4_hold_region_is_not_present(minutes):
    """The retired holdout begins at the research boundary and stays unread."""
    dates = pd.to_datetime(minutes["date"], utc=True)
    ledger = json.loads((REPO_ROOT / "data/research/p4_holdout_ledger.json").read_text())
    assert ledger["state"] == "retired"
    assert ledger["checkpoint"] is None
    start = pd.Timestamp(ledger["region_span"].split(" .. ")[0])
    assert start == RESEARCH_VISIBLE_END
    assert int((dates >= start).sum()) == 0


def test_manifest_verification_passes():
    report = verifier.verify(MANIFEST_PATH)
    assert report["boundaries"]["rows_at_or_after_research_boundary"] == 0
    assert report["boundaries"]["rows_at_or_after_styx"] == 0
    assert set(report["clocks"]) == set(ALL_CLOCKS)


@pytest.mark.parametrize("timeframe", ALL_CLOCKS)
def test_every_clock_reproduces_its_manifest_record(minutes, manifest, timeframe):
    frame = resample_from_minutes(minutes, timeframe)
    assert_manifest_clock(frame, manifest["clocks"][timeframe])


@pytest.mark.parametrize("timeframe", ALL_CLOCKS)
def test_every_bar_opens_on_a_strict_utc_boundary(minutes, timeframe):
    frame = resample_from_minutes(minutes, timeframe)
    period = pd.Timedelta(minutes=constituent_count(timeframe))
    dates = pd.to_datetime(frame["date"], utc=True)
    epoch = pd.Timestamp("1970-01-01T00:00:00+00:00")
    assert ((dates - epoch) % period == pd.Timedelta(0)).all()


@pytest.mark.parametrize("timeframe", ALL_CLOCKS)
def test_every_bar_closes_before_the_research_boundary(minutes, timeframe):
    frame = resample_from_minutes(minutes, timeframe)
    period = pd.Timedelta(minutes=constituent_count(timeframe))
    closes = pd.to_datetime(frame["date"], utc=True) + period
    assert closes.max() <= RESEARCH_VISIBLE_END


@pytest.mark.parametrize("timeframe", CLOCKS)
def test_incomplete_bars_are_dropped_not_completed(minutes, timeframe):
    """Bars the exchange never finished printing are absent, not filled."""
    availability = bar_availability(minutes, timeframe)
    frame = resample_from_minutes(minutes, timeframe)
    assert availability["complete_bars"] == len(frame)
    assert availability["buckets_touched"] == (
        availability["complete_bars"] + availability["incomplete_bars_dropped"]
    )
    if timeframe != "1m":
        assert availability["incomplete_bars_dropped"] > 0


def test_gap_structure_is_the_exchange_and_is_recorded(minutes, manifest):
    gaps = minute_gaps(minutes)
    assert gaps == manifest["minutes"]["gaps"]
    assert len(gaps) == 15


def test_faster_clocks_carry_more_bars(minutes, manifest):
    counts = [manifest["clocks"][clock]["rows"] for clock in ALL_CLOCKS]
    assert counts == sorted(counts, reverse=True)


def test_resampling_is_deterministic(minutes):
    first = resample_from_minutes(minutes, "15m")
    second = resample_from_minutes(minutes, "15m")
    assert candle_digest(first) == candle_digest(second)


def test_derived_1h_agrees_with_the_committed_history(minutes, manifest):
    """The parity claim that licenses fitting on this source, recomputed."""
    reference = pd.read_parquet(
        REPO_ROOT / "data/research/btc_usdt_1h_gen1_raw_pre_styx.parquet"
    )
    reference = reference.loc[reference["date"] < RESEARCH_VISIBLE_END].reset_index(drop=True)
    result = parity_against(resample_from_minutes(minutes, "1h"), reference, timeframe="1h")

    declared = manifest["parity_1h"]
    assert result.overlapping_bars == declared["overlapping_bars"]
    assert result.mismatching_bars == declared["mismatching_bars"]
    assert list(result.mismatching_timestamps) == declared["mismatching_timestamps"]
    # Open is identical on every overlapping hour: the two series agree about
    # where each hour begins, which is what makes the rest a value comparison
    # rather than an alignment comparison.
    assert result.max_relative_difference["open"] == 0.0
    assert result.overlapping_bars - result.mismatching_bars == 47_094


def test_the_disagreeing_hours_are_confined_to_the_early_history(manifest):
    """Where the two Binance series disagree, and where they do not.

    All 29 disagreements sit between 2020-04 and 2022-05. Nothing after
    2022-05-01 differs by more than a float64 summation-order difference. The
    checkpoint-specific consequence — that none of them reaches a reported
    outer block — is asserted in `test_p6_preregistration.py`, where the blocks
    are defined.
    """
    stamps = [pd.Timestamp(value) for value in manifest["parity_1h"]["mismatching_timestamps"]]
    assert len(stamps) == 29
    assert min(stamps) >= pd.Timestamp("2020-04-01T00:00:00+00:00")
    assert max(stamps) <= pd.Timestamp("2022-05-01T23:00:00+00:00")


# --------------------------------------------------------------------------- #
# Positive controls: each mutation must be caught
# --------------------------------------------------------------------------- #


def test_control_shifted_higher_timeframe_bar_is_caught(synthetic_minutes):
    """A 1h bar moved one period into the future stops matching its constituents.

    Two halves, and the second is the one that says something about the
    resampler. The first mutates the resampler's *output* and shows the detector
    catches it. The second takes the unmutated output and checks each bar against
    the minutes it claims to summarise — that a bar labelled `t` is built from
    `[t, t+1h)` and from nothing later — which is the property the mutation
    violates, asserted directly rather than by proxy.
    """
    clean = resample_from_minutes(synthetic_minutes, "1h", boundary=None)
    leaked = clean.copy()
    leaked["date"] = leaked["date"] - pd.Timedelta(hours=1)

    honest = candle_digest(clean)
    assert candle_digest(leaked) != honest
    # The leak is that row t now carries the bar that closes at t+1h. Re-cutting
    # from the minutes it claims to summarise disagrees on every value.
    result = parity_against(leaked, clean, timeframe="1h")
    assert result.mismatching_bars > 0

    # And the resampler itself does not produce such a bar: every OHLCV value of
    # every bar is the aggregate of exactly the minutes inside its own window.
    minutes = synthetic_minutes.sort_values("date")
    for row in clean.itertuples():
        inside = (minutes["date"] >= row.date) & (
            minutes["date"] < row.date + pd.Timedelta(hours=1)
        )
        window = minutes.loc[inside]
        assert len(window) == 60
        assert row.open == pytest.approx(window["open"].iloc[0])
        assert row.close == pytest.approx(window["close"].iloc[-1])
        assert row.high == pytest.approx(window["high"].max())
        assert row.low == pytest.approx(window["low"].min())
        assert row.volume == pytest.approx(window["volume"].sum())


def test_control_incomplete_bar_is_caught(synthetic_minutes):
    """Deleting one minute makes its hour unavailable rather than short."""
    mutated = synthetic_minutes.drop(index=7).reset_index(drop=True)
    frame = resample_from_minutes(mutated, "1h", boundary=None)
    assert len(frame) == 3
    assert pd.Timestamp("2021-06-01T00:00:00+00:00") not in set(frame["date"])

    availability = bar_availability(mutated, "1h")
    assert availability["incomplete_bars_dropped"] == 1
    assert availability["complete_bars"] == 3


def test_control_crossing_the_research_boundary_is_caught(synthetic_minutes):
    """A minute at the boundary is refused, and the count is reported, not the row."""
    mutated = synthetic_minutes.copy()
    mutated.loc[mutated.index[-1], "date"] = RESEARCH_VISIBLE_END
    with pytest.raises(MulticlockError, match="research-visible boundary"):
        resample_from_minutes(mutated, "5m")
    with pytest.raises(MulticlockError, match="research-visible boundary"):
        resample_from_minutes(mutated, "1m")


def test_control_bar_closing_after_the_boundary_is_caught(synthetic_minutes):
    """Constituents may all precede the boundary while the bar still closes past it."""
    start = RESEARCH_VISIBLE_END - pd.Timedelta(minutes=30)
    index = pd.date_range(start, periods=30, freq="1min", tz="UTC")
    frame = synthetic_minutes.iloc[: len(index)].copy().reset_index(drop=True)
    frame["date"] = index
    # Thirty minutes, all strictly before the boundary: a 30m bar is complete and
    # legal, and the 1h bar that would contain them is not, because it needs
    # minutes the boundary forbids.
    assert len(resample_from_minutes(frame, "30m")) == 1
    with pytest.raises(MulticlockError):
        resample_from_minutes(frame, "1h")


def test_control_corrupted_constituent_changes_the_derived_clock(synthetic_minutes):
    """One altered 1m candle must move the bar that contains it."""
    clean = resample_from_minutes(synthetic_minutes, "15m", boundary=None)
    mutated = synthetic_minutes.copy()
    mutated.loc[3, "high"] = float(mutated.loc[3, "high"]) + 1_000.0

    corrupted = resample_from_minutes(mutated, "15m", boundary=None)
    assert candle_digest(corrupted) != candle_digest(clean)
    assert corrupted.loc[0, "high"] == pytest.approx(float(mutated.loc[3, "high"]))

    record = {
        "timeframe": "15m",
        "rows": len(clean),
        "start": pd.to_datetime(clean["date"], utc=True).iloc[0].isoformat(),
        "end": pd.to_datetime(clean["date"], utc=True).iloc[-1].isoformat(),
        "digest": candle_digest(clean),
    }
    with pytest.raises(MulticlockError, match="disagrees with the manifest on digest"):
        assert_manifest_clock(corrupted, record)


def test_control_moved_alignment_boundary_is_caught(synthetic_minutes):
    """Offsetting the grid by one minute must not silently produce full bars."""
    shifted = synthetic_minutes.copy()
    shifted["date"] = shifted["date"] + pd.Timedelta(minutes=1)
    frame = resample_from_minutes(shifted, "1h", boundary=None)
    # Every bucket is now short by one minute at each end, so the strict rule
    # leaves only the interior hours and never an off-grid one.
    epoch = pd.Timestamp("1970-01-01T00:00:00+00:00")
    dates = pd.to_datetime(frame["date"], utc=True)
    assert ((dates - epoch) % pd.Timedelta(hours=1) == pd.Timedelta(0)).all()
    assert len(frame) < 4


def test_control_duplicate_minute_is_caught(synthetic_minutes):
    """Sixty rows in an hour is only a complete hour if they are sixty minutes."""
    mutated = pd.concat(
        [synthetic_minutes, synthetic_minutes.iloc[[5]]], ignore_index=True
    ).sort_values("date")
    with pytest.raises(MulticlockError, match="duplicate timestamp"):
        resample_from_minutes(mutated, "1h", boundary=None)


def test_control_unordered_source_is_caught(synthetic_minutes):
    mutated = synthetic_minutes.iloc[::-1].reset_index(drop=True)
    with pytest.raises(MulticlockError, match="increasing timestamp order"):
        resample_from_minutes(mutated, "1h", boundary=None)


def test_control_sub_minute_timestamp_is_caught(synthetic_minutes):
    mutated = synthetic_minutes.copy()
    mutated.loc[2, "date"] = mutated.loc[2, "date"] + pd.Timedelta(seconds=30)
    with pytest.raises(MulticlockError, match="whole minutes"):
        resample_from_minutes(mutated, "1h", boundary=None)


def test_control_manifest_row_count_lie_is_caught(minutes, manifest):
    record = dict(manifest["clocks"]["1h"])
    record["rows"] = record["rows"] + 1
    with pytest.raises(MulticlockError, match="disagrees with the manifest on rows"):
        assert_manifest_clock(resample_from_minutes(minutes, "1h"), record)


def test_control_verifier_rejects_a_moved_boundary(tmp_path, manifest):
    mutated = json.loads(json.dumps(manifest))
    mutated["boundaries"]["research_visible_end"] = "2025-06-01T00:00:00+00:00"
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps(mutated))
    with pytest.raises(verifier.SnapshotError, match="does not move"):
        verifier.verify(path)


def test_control_verifier_rejects_an_opened_boundary_claim(tmp_path, manifest):
    mutated = json.loads(json.dumps(manifest))
    mutated["boundaries"]["styx_opened"] = True
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps(mutated))
    with pytest.raises(verifier.SnapshotError, match="nothing may open it"):
        verifier.verify(path)


def test_control_verifier_rejects_an_unenumerated_parity_claim(tmp_path, manifest):
    mutated = json.loads(json.dumps(manifest))
    mutated["parity_1h"]["mismatching_bars"] = 0
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps(mutated))
    with pytest.raises(verifier.SnapshotError, match="disagree and enumerates"):
        verifier.verify(path)


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("close", 0.0, "non-positive price"),
        ("open", -1.0, "non-positive price"),
        ("high", 1.0, "does not contain its own open and close"),
        ("low", 10_000.0, "does not contain its own open and close"),
        ("volume", -1.0, "negative volume"),
    ],
)
def test_control_a_broken_candle_is_refused_rather_than_aggregated(
    synthetic_minutes, column, value, message
):
    """One impossible row inside an hour must not become that hour's wick.

    `high` and `low` are a max and a min over the constituents, so a single row
    with a high below its own body widens — or narrows — the bar it lands in, and
    nothing downstream can tell the result from a real wick.
    """
    broken = synthetic_minutes.copy()
    broken.loc[7, column] = value
    with pytest.raises(MulticlockError, match=message):
        resample_from_minutes(broken, "1h", boundary=None)


def test_the_committed_source_carries_no_broken_candle(minutes):
    """Not a property of the checker: a property of what is committed."""
    dates = assert_minute_grid(minutes, what="the committed 1m source")
    assert len(dates) == len(minutes)
