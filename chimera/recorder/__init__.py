"""The prospective recorder: offline core.

This package is the data model of the forward-only recorder the project's next
deciding evidence depends on — the contract that says what will be recorded, the
event types that say what one observation is, the append-only sink that stores
them, and the normalizer that turns them into one row per minute the recorder
captured a closed kline for.

**It is offline, and offline is a property rather than an intention.** Nothing
here opens a socket, makes a request, holds a credential or reads a clock. The
websocket clients, the REST pollers, the service and the command line that will
drive it are a separate, later change; the archive reconciliation and the
coverage gate are a separate, later change again. What is here can be exercised
end to end with network access denied, and ``tests/test_recorder_no_network.py``
denies it and does exactly that.

**It computes nothing.** No return, no signal, no funding profit, no basis, no
PnL, no accuracy, no statistic. Every number that comes out was published by an
exchange and carried across unchanged. The recorder's whole job is to be a
faithful witness, and a witness that computes is a witness that can be wrong in
a way nobody can check.

**Missing data is missing.** No forward fill, no backward fill, no
interpolation, no invented candle, no value borrowed from a neighbouring minute
or a different stream. A minute this recorder holds no usable closed kline for
has no row and is named in the day's ``missing`` list.

**And missing means missing here, not missing at the venue.** Absence from a
normalized day says that the recorder has no usable closed kline for that
minute, and says nothing about whether the exchange published one — the two are
the coverage gate's ``captured_minutes`` and ``published_minutes``, and only the
first is knowable offline. The second is established by the archive
reconciliation, later, from the venue's own archive. Nothing in this package
infers one from the other, because a recorder outage and a venue publication gap
are indistinguishable from here.

The prospective boundary — ``prospective_from`` — is unset in the committed
contract, and until it is written no minute recorded under it is scientific
evidence.
"""

from chimera.recorder.contract import (
    CONTRACT_SCHEMA,
    CONTRACTS_DIR,
    GEN3_CONTRACT_ID,
    STORAGE_LAYOUT_VERSION,
    Market,
    ProspectiveBoundaryError,
    RecorderContract,
    RecorderContractError,
    available_recorder_contract_ids,
    canonical_material,
    contract_hash,
    load_recorder_contract,
    parse_recorder_contract,
    read_recorder_contract_file,
)
from chimera.recorder.events import (
    MINUTES_PER_DAY,
    RAW_EVENT_SCHEMA,
    SPOT_BOOK_TICKER,
    SPOT_KLINE_1M,
    UM_BOOK_TICKER,
    UM_FUNDING,
    UM_KLINE_1M,
    UM_MARK_PRICE,
    BookTickerEvent,
    EventSource,
    FundingSettlement,
    KlineEvent,
    MarkPriceEvent,
    RawEvent,
    RecorderEventError,
    TimeBasis,
    canonical_json,
    day_start_ns,
    iso_utc,
    minute_open_ms,
    sort_events,
    utc_day,
)
from chimera.recorder.normalize import (
    CLOCK,
    MARKET_COLUMNS,
    NORMALIZED_META_SCHEMA,
    NORMALIZED_SCHEMA,
    BookMinute,
    ColumnSpec,
    DayReport,
    KlineMinute,
    MarkMinute,
    MinuteNormalizer,
    MinuteRecord,
    RecorderNormalizeError,
    SettlementsReport,
    build_minutes,
    columns_for,
    digest,
    gaps_of,
    meta,
    minute_frame,
)
from chimera.recorder.sink import (
    DAY_MANIFEST_SCHEMA,
    DEFAULT_DEDUP_WINDOW,
    AppendOutcome,
    AppendResult,
    DayCounters,
    RawSink,
    RecorderSinkError,
    TailRecovery,
    available_days,
    read_raw_events,
    require_day,
    require_stream_id,
)

__all__ = [
    "AppendOutcome",
    "AppendResult",
    "BookMinute",
    "BookTickerEvent",
    "CLOCK",
    "CONTRACTS_DIR",
    "CONTRACT_SCHEMA",
    "ColumnSpec",
    "DAY_MANIFEST_SCHEMA",
    "DEFAULT_DEDUP_WINDOW",
    "DayCounters",
    "DayReport",
    "EventSource",
    "FundingSettlement",
    "GEN3_CONTRACT_ID",
    "KlineEvent",
    "KlineMinute",
    "MARKET_COLUMNS",
    "MINUTES_PER_DAY",
    "Market",
    "MarkMinute",
    "MarkPriceEvent",
    "MinuteNormalizer",
    "MinuteRecord",
    "NORMALIZED_META_SCHEMA",
    "NORMALIZED_SCHEMA",
    "ProspectiveBoundaryError",
    "RAW_EVENT_SCHEMA",
    "RawEvent",
    "RawSink",
    "RecorderContract",
    "RecorderContractError",
    "RecorderEventError",
    "RecorderNormalizeError",
    "RecorderSinkError",
    "SPOT_BOOK_TICKER",
    "SPOT_KLINE_1M",
    "STORAGE_LAYOUT_VERSION",
    "SettlementsReport",
    "TailRecovery",
    "TimeBasis",
    "UM_BOOK_TICKER",
    "UM_FUNDING",
    "UM_KLINE_1M",
    "UM_MARK_PRICE",
    "available_days",
    "available_recorder_contract_ids",
    "build_minutes",
    "canonical_json",
    "canonical_material",
    "columns_for",
    "contract_hash",
    "day_start_ns",
    "digest",
    "gaps_of",
    "iso_utc",
    "load_recorder_contract",
    "meta",
    "minute_frame",
    "minute_open_ms",
    "parse_recorder_contract",
    "read_raw_events",
    "read_recorder_contract_file",
    "require_day",
    "require_stream_id",
    "sort_events",
    "utc_day",
]
