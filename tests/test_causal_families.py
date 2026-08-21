"""The contract every causal feature family owes, checked over the registry.

``tests/test_smc_features.py`` and ``tests/test_chart_structure.py`` derive their
expected numbers from the specs by hand and are the reason we believe either
engine computes what its document says. This file believes nothing about any
particular column. It asserts only what every causal family must satisfy
whatever it measures, once per entry in
:data:`nn.causal_families.CAUSAL_FEATURE_FAMILIES`, so a family added to that
table inherits the whole set and a family whose author forgot to write these
tests is red rather than untested.

The gap reset is the one property that cannot be stated in a family's own
vocabulary here. What resets at a segment boundary is family-specific —
``smc_v1`` §3 drops swing levels, pending sweeps and unfilled gaps;
``chart_structure_v1`` §1 restarts the trailing windows and abandons the pending
breakout — and only that family's own test can name the columns and the numbers.
The general form is structural: the segmented entry point's post-gap rows must be
exactly what the post-gap candles produce on their own, and must differ from what
those same rows produce when the hole is ignored. The second half is what gives
the first half teeth — it shows the family really does carry state that a
boundary has to stop, so the equality is a reset rather than there having been
nothing to bridge.

Every frame comparison is ``check_exact=True``. The default tolerance of
:func:`pandas.testing.assert_frame_equal` is 1e-5 relative, which would accept an
engine that re-dates history by a rounding error — and a leak that small is still
a leak, because the model is scored on a level drawn after the move.

The real candles are the committed pre-Styx research history; nothing here reads
market data at or after the Styx boundary.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nn.causal_families import CAUSAL_FEATURE_FAMILIES, CausalFeatureFamily

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CANDLES = REPO_ROOT / "data" / "research" / "btc_usdt_1h_gen1_raw_pre_styx.parquet"
STYX = pd.Timestamp("2025-08-27T23:00:00+00:00")

START = pd.Timestamp("2024-01-01T00:00:00+00:00")
HOUR = pd.Timedelta(hours=1)

#: Every test below runs once per registered family.
FAMILY = pytest.mark.parametrize(
    "family",
    list(CAUSAL_FEATURE_FAMILIES.values()),
    ids=list(CAUSAL_FEATURE_FAMILIES),
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_candles() -> pd.DataFrame:
    """The committed pre-Styx BTC research candles."""
    frame = pd.read_parquet(REAL_CANDLES).reset_index(drop=True)
    assert pd.to_datetime(frame["date"], utc=True).max() < STYX
    return frame


@pytest.fixture(scope="module")
def real_contiguous(real_candles: pd.DataFrame) -> pd.DataFrame:
    """A gap-free block of the real candles, for the one-segment entry point."""
    block = real_candles.iloc[4289:8010].reset_index(drop=True)
    assert (pd.to_datetime(block["date"], utc=True).diff().iloc[1:] == HOUR).all()
    return block


GAP_SEGMENT_ONE = 90
GAP_SEGMENT_TWO = 60


def _gapped_candles() -> pd.DataFrame:
    """Two runs of candles separated by a five-hour market-data hole.

    Both runs are long enough that every registered engine has live state to
    carry over: the longest trailing window in either spec is 60 candles and both
    confirm a pivot three candles late.
    """
    rows = GAP_SEGMENT_ONE + GAP_SEGMENT_TWO
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, rows))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.3, rows))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.3, rows))
    dates = [START + i * HOUR for i in range(GAP_SEGMENT_ONE)] + [
        START + (GAP_SEGMENT_ONE + 5 + i) * HOUR for i in range(GAP_SEGMENT_TWO)
    ]
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates, tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 10.0,
        }
    )


GAPPED = _gapped_candles()


def _with_scaled_candle(frame: pd.DataFrame, i: int) -> pd.DataFrame:
    """``frame`` with candle ``i`` 5% higher — scaled as a whole, so o/h/l/c stay ordered."""
    out = frame.copy()
    out.loc[out.index[i], ["open", "high", "low", "close"]] *= 1.05
    return out


def _perturbed(value: object) -> object:
    """A different value of the same kind, for the spec-hash sensitivity check."""
    if isinstance(value, (int, float)):
        return value + 1
    raise TypeError(f"extend _perturbed for spec constants of type {type(value).__name__}")


# --------------------------------------------------------------------------- #
# 0. the registry itself
# --------------------------------------------------------------------------- #
def test_the_written_families_are_registered() -> None:
    """An empty or mistyped table would make every check below vacuously green."""
    assert {"smc_v1", "chart_structure_v1"} <= set(CAUSAL_FEATURE_FAMILIES)
    for name, family in CAUSAL_FEATURE_FAMILIES.items():
        assert family.name == name


# --------------------------------------------------------------------------- #
# 1. append invariance: F(raw[:N+K])[:N] == F(raw[:N])
# --------------------------------------------------------------------------- #
@FAMILY
@pytest.mark.parametrize("n,k", [(1000, 1), (1800, 7), (2500, 600)])
def test_append_invariance_real(
    family: CausalFeatureFamily, real_contiguous: pd.DataFrame, n: int, k: int
) -> None:
    long = family.compute(real_contiguous.iloc[: n + k])
    short = family.compute(real_contiguous.iloc[:n])
    assert_frame_equal(long.iloc[:n], short, check_exact=True)


@FAMILY
@pytest.mark.parametrize("n,k", [(1200, 1), (2000, 300), (3000, 50)])
def test_append_invariance_real_segmented(
    family: CausalFeatureFamily, real_candles: pd.DataFrame, n: int, k: int
) -> None:
    """The first 3000 real candles hold four market-data gaps, so the longer
    slice has to re-derive the segmentation rather than extend one segment."""
    long = family.compute_segmented(real_candles.iloc[: n + k], "1h")
    short = family.compute_segmented(real_candles.iloc[:n], "1h")
    assert_frame_equal(long.iloc[:n], short, check_exact=True)


# --------------------------------------------------------------------------- #
# 2. changing candle i leaves every row before i alone
# --------------------------------------------------------------------------- #
@FAMILY
@pytest.mark.parametrize("i", [50, 700, 1400])
def test_changing_one_candle_leaves_every_earlier_row_identical(
    family: CausalFeatureFamily, real_contiguous: pd.DataFrame, i: int
) -> None:
    """The same property as append invariance, stated where it is cheap to check.

    Appending is the special case in which the changed candles are the ones that
    did not exist yet; this covers a revision to a candle that did.
    """
    original = real_contiguous.iloc[:1500]
    base = family.compute(original)
    after = family.compute(_with_scaled_candle(original, i))

    assert_frame_equal(after.iloc[:i], base.iloc[:i], check_exact=True)
    # ...and the changed candle mattered, so the equality above is a result
    # rather than a family that ignored it.
    assert not after.iloc[i:].equals(base.iloc[i:])


# --------------------------------------------------------------------------- #
# 3. no NaN, no infinity, on the real history
# --------------------------------------------------------------------------- #
@FAMILY
def test_no_nan_and_no_infinity_on_the_real_history(
    family: CausalFeatureFamily, real_candles: pd.DataFrame, real_contiguous: pd.DataFrame
) -> None:
    """``nn.data_pipeline.build_dataset`` drops any row with a NaN feature, so a
    NaN here would evaluate two information sets on different samples and
    confound the comparison with the sample universe."""
    segmented = family.compute_segmented(real_candles, "1h")
    single = family.compute(real_contiguous)

    for out in (segmented, single):
        assert (out.dtypes == np.float64).all()
        assert not out.isna().to_numpy().any()
        assert np.isfinite(out.to_numpy()).all()

    assert len(segmented) == len(real_candles)
    assert len(single) == len(real_contiguous)


# --------------------------------------------------------------------------- #
# 4. the declared columns, in the declared order
# --------------------------------------------------------------------------- #
@FAMILY
def test_columns_are_the_declared_names_in_order(
    family: CausalFeatureFamily, real_contiguous: pd.DataFrame
) -> None:
    """Order, not just membership: the spec hash and every persisted prediction
    record depend on the position of a column, not only on its presence."""
    block = real_contiguous.iloc[:400]
    for out in (family.compute(block), family.compute_segmented(block, "1h")):
        assert tuple(out.columns) == family.columns
        assert out.index.equals(block.index)


# --------------------------------------------------------------------------- #
# 5. the families partition the columns
# --------------------------------------------------------------------------- #
@FAMILY
def test_families_partition_the_columns(family: CausalFeatureFamily) -> None:
    """A leave-one-family-out ablation only means something if the families cover
    every column and share none."""
    seen: set[str] = set()
    total = 0
    for name, columns in family.families.items():
        assert len(set(columns)) == len(columns), name
        assert seen.isdisjoint(columns), name
        seen.update(columns)
        total += len(columns)

    assert total == len(family.columns) == len(set(family.columns))
    assert seen == set(family.columns)


# --------------------------------------------------------------------------- #
# 6. state does not bridge a market-data gap
# --------------------------------------------------------------------------- #
@FAMILY
def test_state_does_not_bridge_a_missing_candle_gap(family: CausalFeatureFamily) -> None:
    """See the module docstring for why this is the family-generic form."""
    segmented = family.compute_segmented(GAPPED, "1h")
    ignoring_the_hole = family.compute(GAPPED)
    assert len(segmented) == len(GAPPED)

    before = GAPPED.iloc[:GAP_SEGMENT_ONE]
    after = GAPPED.iloc[GAP_SEGMENT_ONE:]
    assert_frame_equal(
        segmented.iloc[:GAP_SEGMENT_ONE], family.compute(before), check_exact=True
    )
    assert_frame_equal(
        segmented.iloc[GAP_SEGMENT_ONE:], family.compute(after), check_exact=True
    )

    # Teeth: there really was state on the other side of the hole to carry.
    assert not segmented.iloc[GAP_SEGMENT_ONE:].equals(
        ignoring_the_hole.iloc[GAP_SEGMENT_ONE:]
    )


# --------------------------------------------------------------------------- #
# 7. the spec hash
# --------------------------------------------------------------------------- #
@FAMILY
def test_spec_hash_is_stable_and_moves_with_every_constant(
    family: CausalFeatureFamily,
) -> None:
    """A hash that ignored a constant would let it be tuned against an outer
    result while the recorded artifact still claimed the predeclared spec."""
    digest = family.spec_hash()
    assert digest == family.spec_hash()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")

    fields = dataclasses.fields(family.spec)
    assert fields, "a family's spec must declare its constants as dataclass fields"
    for field in fields:
        current = getattr(family.spec, field.name)
        other = dataclasses.replace(family.spec, **{field.name: _perturbed(current)})
        assert other.spec_hash() != digest, field.name


# --------------------------------------------------------------------------- #
# 8. the spec document
# --------------------------------------------------------------------------- #
@FAMILY
def test_spec_document_exists_and_names_the_version(family: CausalFeatureFamily) -> None:
    document = REPO_ROOT / family.spec_document
    assert document.is_file(), family.spec_document
    assert family.version in document.read_text(encoding="utf-8")
