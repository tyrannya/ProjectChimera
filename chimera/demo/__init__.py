"""The demo runtime: the parts of Minimum Viable Chimera that are not the market.

The plumbing every other part depends on, and, since PR-10, the runner that runs
on it:

``chimera.demo.clock``
    :class:`~chimera.demo.clock.RunnerClock`. The decision clock is
    ``max(receipt_ns)`` over what has been read from the recorder, never the wall
    clock, so the same recorded minutes replayed produce the same decisions and
    the same stale-feed vetoes for ever (section 2.4).

``chimera.demo.config``
    the campaign configuration: parsed strictly — an unrecognised key is refused
    rather than ignored — hashed semantically, and forbidden to configure fault
    injection when it is a campaign (section 5.3).

``chimera.demo.decision_log``
    the append-only, hash-chained NDJSON log the campaign's evidence is made of,
    with section 9.2's canonical serialization and a verifier that tells a torn
    tail apart from a forged chain (section 9).

``chimera.demo.feed``
    :class:`~chimera.demo.feed.FeedCursor` and :class:`MarketState`: the
    recorder's normalized minutes, read one at a time and read-only, with the
    recorder's own digest function narrowed to the single minute a decision was
    made from (section 2.2's D -> E edge).

``chimera.demo.rules``, ``rules_carry``, ``rules_shadow``
    the rule contract and the three rules. A rule sees a `MarketState` and a
    read-only portfolio view and nothing else; a shadow rule returns a
    `SignalOnly`, which no executor can accept. **No rule carries a default
    parameter**: the S2 protocol freezes those and it is PR-14's, so a rule
    refuses to be constructed without explicit values (section 17's S3 STOP).

``chimera.demo.runner``
    :class:`~chimera.demo.runner.DemoRunner`: section 8.1's state machine. It
    owns sequencing and evidence and no arithmetic -- sizing is the rule's,
    execution the position's, permission Aegis's, cash the ledger's.

``chimera.demo.telemetry``
    section 11.1's Prometheus series, and the only module on this path permitted
    to import :mod:`chimera.metrics`. It writes; it reads nothing back and
    returns nothing, so no campaign decision can be a function of its own
    monitoring -- which is asserted structurally and by a byte comparison of two
    identical campaigns run with and without it.

``chimera.demo.reports``
    section 11.4's daily operational report, and the monthly frozen report that
    refuses to exist. Both are pure functions of the persisted log: they open it
    read-only, repair no torn tail, reinterpret no failed reconciliation and infer
    no record that is not there. The daily report counts HALT, RESUME and RECOVERY
    without classifying them, and states per record kind whether the runner in
    this build can write one at all -- so a zero that means "unreachable" is never
    printed as a zero that means "nothing happened".

``chimera.demo.fixtures``, ``chimera.demo.faults``
    synthetic days and fault schedules, for tests and soak drills only. Nothing
    on the production path imports either, and a test asserts that.

**What is deliberately not here yet.** The prospective protocol itself (PR-14),
which is why ``chimera.demo.reports.monthly_report`` refuses every month it is
given: with no frozen protocol there is nothing a monthly number could be
evidence for.

**And nothing here is a scientific authorisation.** The demo's prospective
protocol is preregistered by PR-14, not by this package: ``protocol_hash`` in
``conf/demo/pvc1.json`` is ``null``, no campaign has started, no evidence exists,
and no real money is authorised by any of it.
"""

from chimera.demo.clock import RunnerClock, RunnerClockError
from chimera.demo.config import (
    CONFIG_SCHEMA,
    LIMIT_FIELDS,
    ConfigProfile,
    DemoConfig,
    DemoConfigError,
    DemoLimits,
    config_hash,
    load_demo_config,
    parse_demo_config,
)
from chimera.demo.feed import FeedCursor, FeedError, MarketState, MinuteRecord
from chimera.demo.reports import (
    DAILY_REPORT_SCHEMA,
    MONTHLY_REPORT_SCHEMA,
    ProtocolBinding,
    ReportError,
    ReportRefused,
    daily_report,
    halt_cause,
    kind_class,
    monthly_report,
    render_daily_markdown,
    render_monthly_status,
)
from chimera.demo.rules import (
    HedgeTarget,
    Rule,
    RuleDecision,
    RuleError,
    RuleRegistry,
    SignalOnly,
)
from chimera.demo.runner import DemoRunner, RunnerError, RunnerState, TickOutcome
from chimera.demo.decision_log import (
    DECISION_RECORD_SCHEMA,
    EVIDENCE_KINDS,
    OPERATIONAL_KINDS,
    UNCLASSIFIED_KINDS,
    ZERO_PREV_HASH,
    AppendedRecord,
    ChainDefect,
    ChainFault,
    ChainVerification,
    DecisionLog,
    DecisionLogError,
    DecisionLogTailError,
    RecordKind,
    TailRepair,
    canonical_json,
    canonical_line,
    compute_record_hash,
    decimal_str,
    is_evidence,
    iso_minute,
    recover_tail,
    require_iso_minute,
    verify_chain,
    verify_log,
)

__all__ = [
    "AppendedRecord",
    "CONFIG_SCHEMA",
    "ChainDefect",
    "ChainFault",
    "ChainVerification",
    "ConfigProfile",
    "DAILY_REPORT_SCHEMA",
    "DECISION_RECORD_SCHEMA",
    "DecisionLog",
    "DecisionLogError",
    "DecisionLogTailError",
    "DemoConfig",
    "DemoConfigError",
    "DemoLimits",
    "DemoRunner",
    "EVIDENCE_KINDS",
    "FeedCursor",
    "FeedError",
    "HedgeTarget",
    "LIMIT_FIELDS",
    "MONTHLY_REPORT_SCHEMA",
    "MarketState",
    "MinuteRecord",
    "OPERATIONAL_KINDS",
    "ProtocolBinding",
    "RecordKind",
    "ReportError",
    "ReportRefused",
    "Rule",
    "RuleDecision",
    "RuleError",
    "RuleRegistry",
    "RunnerClock",
    "RunnerClockError",
    "RunnerError",
    "RunnerState",
    "SignalOnly",
    "TailRepair",
    "TickOutcome",
    "UNCLASSIFIED_KINDS",
    "ZERO_PREV_HASH",
    "canonical_json",
    "canonical_line",
    "compute_record_hash",
    "config_hash",
    "daily_report",
    "decimal_str",
    "halt_cause",
    "is_evidence",
    "iso_minute",
    "kind_class",
    "load_demo_config",
    "monthly_report",
    "parse_demo_config",
    "recover_tail",
    "render_daily_markdown",
    "render_monthly_status",
    "require_iso_minute",
    "verify_chain",
    "verify_log",
]
