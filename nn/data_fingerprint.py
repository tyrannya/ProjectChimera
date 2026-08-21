"""Content identity for research inputs: what a run actually saw.

A :class:`nn.research_contract.ResearchContract` says what a research
generation is *allowed* to see. It cannot say what it *did* see. Two datasets
can agree on exchange, pair, timeframe, row count, first and last timestamp,
feature contract and target spec while differing in a historical candle — and
every artifact this repository writes would record them identically. This
module closes that gap by giving the research-visible data a deterministic
identity of its own.

Three identities, deliberately kept apart::

    research contract   what this generation may look at   (an instant)
    research input      what it actually looked at         (this module)
    resolved row        where the instant falls in a table (dataset metadata)

Conflating them is how provenance rots. A legitimate correction to a candle
before the seal changes the research input and the resolved row while leaving
the contract untouched, and that has to be representable — so the contract hash
and the input hash are separate fields that change for separate reasons.

**Identity is semantic, not a hash of the file.** Nothing here reads Parquet
bytes. The digest is taken over the *values* the research code reads, each
normalised to a fixed width and byte order: timestamps as UTC nanoseconds since
the epoch, integers as little-endian ``int64``, floats as little-endian
``float64`` with ``-0.0`` folded to ``0.0`` and any NaN folded to one quiet NaN.
Rewriting the same table with different compression, a different row-group
layout, different key/value metadata, a different column order in the file or a
different path on disk therefore produces the same identity. Changing one price
does not.

**The research-visible digest stops at the seal.** :func:`research_input_digest`
takes a row count and hashes ``[0, rows)``. Callers pass the row the contract's
sealed instant resolved to, so appending candles after the seal — the ordinary
way this dataset grows — leaves the identity of the research input untouched.
That is the point: append-only growth after the seal does not change what
research was allowed to see, and it must not make two otherwise identical runs
incomparable.

:attr:`ResearchInputFingerprint.full_table_hash` covers every row instead, for
the separate question "is this the same file?". It is audit metadata and is
never a comparability input; a run that appended sealed candles is still the
same research input. Computing it reads sealed *values* into a one-way digest
and nothing else — no label is inspected, no metric is computed, and no decision
anywhere in this repository branches on it.

**Specification is part of identity.** The digest is prefixed by a canonical
header carrying the market, the ordered feature names, the feature spec and the
target spec, so two tables of identical numbers built under different research
definitions do not collide.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nn.research_contract import normalise_scope_value

#: Names the meaning of a research-input digest. Hashed with the data, so a
#: future change to *what counts as* research-visible content is a change of
#: identity rather than a silent reinterpretation of the digests already
#: recorded in committed artifacts.
RESEARCH_INPUT_SCHEMA = "chimera.research-input/1"

#: The same, for the raw candles the optional diagnostics statistics read.
RAW_INPUT_SCHEMA = "chimera.raw-input/1"

#: Columns of a built dataset that research actually reads, in the order they
#: are hashed. ``segment_id`` is conditional — a dataset built before it existed
#: has none, and hashing a fabricated one would claim gap handling it never had.
#: The feature columns sit between, in the dataset's own recorded order.
LEADING_COLUMNS = (("date", "t8"), ("close", "f8"))
TRAILING_COLUMNS = (("future_return", "f8"), ("target", "i8"))
SEGMENT_COLUMN = ("segment_id", "i8")

#: Raw OHLCV columns, in the order they are hashed.
RAW_COLUMN_KINDS = (
    ("date", "t8"),
    ("open", "f8"),
    ("high", "f8"),
    ("low", "f8"),
    ("close", "f8"),
    ("volume", "f8"),
)

#: Rows canonicalised and fed to the digest at a time.
#:
#: The whole column is never materialised in its canonical form, so hashing a
#: dataset several times the size of the current one costs a fixed few hundred
#: kilobytes above the frame itself.
CHUNK_ROWS = 65_536


class DataFingerprintError(ValueError):
    """A research input cannot be given an honest identity."""


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """The header form the digest is taken over: sorted, tight, UTF-8."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_chunk(chunk: Any, kind: str) -> np.ndarray:
    """One slice of one column, in its fixed-width little-endian canonical form.

    Every normalisation here exists to make an *encoding* difference invisible
    while leaving a *value* difference visible:

    * ``t8`` — timestamps become UTC nanoseconds since the epoch, so a column
      stored as microseconds, as nanoseconds, tz-aware or naive-UTC all hash
      alike, and a column in another zone is converted rather than reinterpreted;
    * ``i8`` — integers become ``int64``, so a column narrowed to ``int32`` by a
      writer still names the same values;
    * ``f8`` — floats become ``float64``. ``-0.0`` and ``0.0`` are the same
      number and are folded together; NaN has many bit patterns and no ordering,
      so every NaN folds to one. Widening a ``float32`` column is *not* a fold:
      the stored values genuinely differ from their ``float64`` counterparts,
      and pretending otherwise would hide a real precision change.
    """
    if kind == "t8":
        stamps = pd.to_datetime(pd.Index(np.asarray(chunk)), utc=True)
        if stamps.hasnans:
            raise DataFingerprintError(
                "the date column contains missing or unparseable timestamps, so the "
                "rows a research input covers cannot be stated"
            )
        # `.as_unit("ns")` is what makes the nanosecond claim above true rather
        # than incidental. `asi8` reports the index's *own* resolution, and
        # pandas 2 happened to coerce everything to nanoseconds while pandas 3
        # preserves whatever the Parquet file stored — so the same candles read
        # by two pandas versions hashed to two different identities, which is
        # precisely the failure a semantic fingerprint exists to rule out.
        return np.asarray(stamps.as_unit("ns").asi8, dtype=np.int64).astype("<i8", copy=False)
    if kind == "i8":
        return np.asarray(chunk, dtype=np.int64).astype("<i8", copy=False)
    # ``+ 0.0`` both copies (so the NaN fold below cannot touch the caller's
    # array) and maps -0.0 onto 0.0.
    values = np.asarray(chunk, dtype=np.float64) + 0.0
    values[np.isnan(values)] = np.nan
    return values.astype("<f8", copy=False)


def _digest(
    schema: str,
    header: Mapping[str, Any],
    columns: Sequence[tuple[str, str]],
    values: Mapping[str, Any],
    rows: int,
) -> str:
    """SHA-256 over the canonical header and then ``rows`` rows of each column.

    Hashed column by column rather than row by row, which is what makes the
    streaming cheap. It loses nothing: a value differs in exactly one column at
    exactly one offset, and reordering rows permutes every column at once, so
    both are visible. The per-column preamble names the column and its length,
    so two different column layouts cannot produce the same byte stream.
    """
    sha = hashlib.sha256()
    sha.update(_canonical_json({**header, "fingerprint_schema": schema}).encode("utf-8"))
    for name, kind in columns:
        column = values[name]
        if len(column) < rows:
            raise DataFingerprintError(
                f"column {name!r} has {len(column)} rows, fewer than the {rows} rows "
                "being fingerprinted"
            )
        sha.update(f"\x1e{name}\x1f{kind}\x1f{rows}\x1e".encode("utf-8"))
        for start in range(0, rows, CHUNK_ROWS):
            chunk = column[start : min(start + CHUNK_ROWS, rows)]
            sha.update(_canonical_chunk(chunk, kind).tobytes())
    return sha.hexdigest()


def research_column_kinds(
    feature_names: Sequence[str], *, has_segment_ids: bool
) -> tuple[tuple[str, str], ...]:
    """The columns a research input is hashed over, in order, with their kinds.

    Exactly what :class:`nn.train.ResearchData` reads and nothing else. A column
    the research code never looks at — a scratch column, an index left over from
    a rebuild — is not part of the research input and does not change its
    identity; a column it does look at cannot be dropped without changing it.
    """
    segment = (SEGMENT_COLUMN,) if has_segment_ids else ()
    features = tuple((name, "f8") for name in feature_names)
    return LEADING_COLUMNS + segment + features + TRAILING_COLUMNS


def _scope_value(field: str, value: Any) -> str:
    """One market field, normalised as research contracts normalise it."""
    text = value.strip() if isinstance(value, str) else ""
    return normalise_scope_value(field, text) if text else ""


def _research_header(
    columns: Sequence[tuple[str, str]],
    *,
    exchange: str,
    pair: str,
    timeframe: str,
    feature_names: Sequence[str],
    feature_spec: Mapping[str, Any],
    target_spec: Mapping[str, Any],
    rows: int,
) -> dict[str, Any]:
    """The specification half of a research input's identity.

    Scope values are normalised the way research contracts normalise them, so
    ``Binance`` and ``binance`` are one market rather than two research inputs.
    The feature and target specs are here because they are what the numbers
    *mean*: a table relabelled at a different horizon or cost threshold is a
    different research input even where the feature values are untouched.

    A dataset that declares no market keeps the empty string rather than a
    stand-in name, which cannot collide with a real one. Such a dataset is
    already refused by ``ResearchContract.require_scope`` before any entrypoint
    fingerprints it; this only keeps the digest well defined for a direct caller.
    """
    return {
        "exchange": _scope_value("exchange", exchange),
        "pair": _scope_value("pair", pair),
        "timeframe": _scope_value("timeframe", timeframe),
        "feature_names": list(feature_names),
        "feature_spec": dict(feature_spec),
        "target_spec": dict(target_spec),
        "columns": [list(entry) for entry in columns],
        "rows": rows,
    }


def research_input_digest(
    values: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    exchange: str,
    pair: str,
    timeframe: str,
    feature_spec: Mapping[str, Any],
    target_spec: Mapping[str, Any],
    rows: int,
) -> str:
    """Identity of the first ``rows`` rows of one built dataset.

    ``values`` maps column name to an array-like: the canonical research columns
    (:func:`research_column_kinds`) must all be present and at least ``rows``
    long. Callers pass ``rows`` = the row the research contract's sealed instant
    resolved to, which is what makes the result the identity of the *research
    input* rather than of the file.
    """
    if rows <= 0:
        raise DataFingerprintError(
            f"a research input covers {rows} rows; there is nothing to identify"
        )
    columns = research_column_kinds(feature_names, has_segment_ids="segment_id" in values)
    missing = [name for name, _ in columns if name not in values]
    if missing:
        raise DataFingerprintError(
            f"cannot fingerprint the research input: column(s) {missing} are absent. "
            "The identity is defined over the columns research reads, so a dataset "
            "missing one of them has no identity under this schema"
        )
    header = _research_header(
        columns,
        exchange=exchange,
        pair=pair,
        timeframe=timeframe,
        feature_names=feature_names,
        feature_spec=feature_spec,
        target_spec=target_spec,
        rows=rows,
    )
    return _digest(RESEARCH_INPUT_SCHEMA, header, columns, values, rows)


@dataclass(frozen=True)
class ResearchInputFingerprint:
    """What one run's research-visible data was, and what the whole table was.

    ``research_input_hash`` is the identity that decides comparability:
    everything strictly before the sealed boundary, which appending later
    candles cannot alter. ``full_table_hash`` answers the different, weaker
    question of whether two runs read the same file, and is recorded for audit
    only — two runs of the same research input on a table that has since grown
    past the seal remain comparable, and must.
    """

    research_input_hash: str
    full_table_hash: str
    research_rows: int
    total_rows: int
    research_start: str
    research_end: str
    columns: tuple[str, ...]

    @property
    def short_hash(self) -> str:
        """First 16 hex digits, for logs and console lines. Never for identity."""
        return self.research_input_hash[:16]

    def to_dict(self) -> dict[str, Any]:
        """The block an artifact records so a later reader needs no local file.

        Everything the identity was taken over is named here — the schema, the
        row count it stops at, the columns and the span — so a reader holding
        the dataset can recompute the digest, and a reader without it can still
        tell two runs' inputs apart.
        """
        return {
            "fingerprint_schema": RESEARCH_INPUT_SCHEMA,
            "research_input_hash": self.research_input_hash,
            "full_table_hash": self.full_table_hash,
            "research_rows": self.research_rows,
            "total_rows": self.total_rows,
            "research_start": self.research_start,
            "research_end": self.research_end,
            "columns": list(self.columns),
        }


def fingerprint_research_input(
    values: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    exchange: str,
    pair: str,
    timeframe: str,
    feature_spec: Mapping[str, Any],
    target_spec: Mapping[str, Any],
    research_rows: int,
    total_rows: int,
) -> ResearchInputFingerprint:
    """Both digests plus the span they cover, for one dataset and one boundary."""
    if not 0 < research_rows <= total_rows:
        raise DataFingerprintError(
            f"the research region is rows [0, {research_rows}) of a {total_rows}-row "
            "table, which is not a region this dataset has"
        )
    dates = pd.to_datetime(pd.Index(np.asarray(values["date"])[:research_rows]), utc=True)
    columns = research_column_kinds(feature_names, has_segment_ids="segment_id" in values)

    def digest(rows: int) -> str:
        return research_input_digest(
            values,
            feature_names=feature_names,
            exchange=exchange,
            pair=pair,
            timeframe=timeframe,
            feature_spec=feature_spec,
            target_spec=target_spec,
            rows=rows,
        )

    return ResearchInputFingerprint(
        research_input_hash=digest(research_rows),
        full_table_hash=digest(total_rows),
        research_rows=research_rows,
        total_rows=total_rows,
        research_start=dates[0].isoformat(),
        research_end=dates[-1].isoformat(),
        columns=tuple(name for name, _ in columns),
    )


@dataclass(frozen=True)
class RawInputFingerprint:
    """Identity of the raw candles a diagnostics run actually read.

    Bounded at ``visible_through`` — the last timestamp the processed research
    region covers — for the same reason the research input is bounded at the
    seal: the raw file keeps growing, and candles the diagnostics never looked
    at must not change what it says it looked at.
    """

    raw_input_hash: str
    rows: int
    start: str
    end: str
    visible_through: str
    columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint_schema": RAW_INPUT_SCHEMA,
            "raw_input_hash": self.raw_input_hash,
            "rows": self.rows,
            "start": self.start,
            "end": self.end,
            "visible_through": self.visible_through,
            "columns": list(self.columns),
        }


def fingerprint_raw_input(
    raw: pd.DataFrame,
    *,
    exchange: str,
    pair: str,
    timeframe: str,
    visible_through: Any,
) -> RawInputFingerprint:
    """Identify the raw candles at or before ``visible_through``.

    ``raw`` is the timestamp-indexed table :func:`nn.regime.load_raw_ohlcv`
    returns: sorted, unique, UTC. Rows after the bound are excluded rather than
    hashed, so a raw file that has been extended since the run still identifies
    the same input.
    """
    bound = pd.Timestamp(visible_through)
    bound = bound.tz_localize("UTC") if bound.tzinfo is None else bound.tz_convert("UTC")
    dates = pd.DatetimeIndex(raw.index)
    # The loader guarantees this; checked anyway, because "the rows at or before
    # the bound" is a prefix only in a sorted table, and silently hashing the
    # wrong prefix would produce a confident identity for the wrong candles.
    if not dates.is_monotonic_increasing:
        raise DataFingerprintError(
            "raw candles are not sorted ascending, so the rows the research region "
            "covers are not a prefix of them"
        )
    rows = int(dates.searchsorted(bound, side="right"))
    if rows == 0:
        raise DataFingerprintError(
            f"no raw candle at or before {bound.isoformat()}, so the raw input the "
            "research region covers is empty and cannot be identified"
        )

    values: dict[str, Any] = {"date": dates.to_numpy()}
    for name, _ in RAW_COLUMN_KINDS[1:]:
        if name not in raw.columns:
            raise DataFingerprintError(f"raw candles are missing the {name!r} column")
        values[name] = raw[name].to_numpy()

    header = {
        "exchange": _scope_value("exchange", exchange),
        "pair": _scope_value("pair", pair),
        "timeframe": _scope_value("timeframe", timeframe),
        "columns": [list(entry) for entry in RAW_COLUMN_KINDS],
        "rows": rows,
    }
    return RawInputFingerprint(
        raw_input_hash=_digest(RAW_INPUT_SCHEMA, header, RAW_COLUMN_KINDS, values, rows),
        rows=rows,
        start=dates[0].isoformat(),
        end=dates[rows - 1].isoformat(),
        visible_through=bound.isoformat(),
        columns=tuple(name for name, _ in RAW_COLUMN_KINDS),
    )
