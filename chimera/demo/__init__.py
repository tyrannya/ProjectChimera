"""The demo runtime: the parts of Minimum Viable Chimera that are not the market.

Three modules today, and they are the three things every later part of the demo
has to be able to depend on before it can be written at all:

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

**What is deliberately not here yet.** The feed, the rules, the runner state
machine, replay, fault injection, observability and the reports are later
packages. None of what is here decides anything, trades anything or computes an
economic quantity; it is the plumbing those parts will run on.

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
    "CONFIG_SCHEMA",
    "DECISION_RECORD_SCHEMA",
    "EVIDENCE_KINDS",
    "LIMIT_FIELDS",
    "OPERATIONAL_KINDS",
    "UNCLASSIFIED_KINDS",
    "ZERO_PREV_HASH",
    "AppendedRecord",
    "ChainDefect",
    "ChainFault",
    "ChainVerification",
    "ConfigProfile",
    "DecisionLog",
    "DecisionLogError",
    "DecisionLogTailError",
    "DemoConfig",
    "DemoConfigError",
    "DemoLimits",
    "RecordKind",
    "RunnerClock",
    "RunnerClockError",
    "TailRepair",
    "canonical_json",
    "canonical_line",
    "compute_record_hash",
    "config_hash",
    "decimal_str",
    "is_evidence",
    "iso_minute",
    "load_demo_config",
    "parse_demo_config",
    "recover_tail",
    "require_iso_minute",
    "verify_chain",
    "verify_log",
]
