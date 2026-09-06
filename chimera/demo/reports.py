"""Section 11.4's reports: what a day of the campaign held, and nothing more.

Two reports live here and they are not the same kind of object.

**The daily report is operational.** It counts what the decision log holds and
does arithmetic on the numbers the log already carries. It scores nothing, it
answers no research question, and its ``report_class`` says so in the payload
rather than only in this paragraph. Section 11.4 asks for minutes processed,
incomplete minutes, decisions by kind, orders and fills per leg, fees, slippage,
funding paid and received, equity open/close/worst, halts and their causes,
reconciliation outcomes and the replay-parity verdict, and that is the whole of
what it produces.

**The monthly report is the prospective evidence of a preregistered protocol**,
and this repository has no such protocol: ``conf/demo/pvc1.json`` carries
``protocol_hash: null`` and the recorder contract carries
``prospective_from: null``. So :func:`monthly_report` refuses, and the refusal is
the feature. A monthly number computed with nothing frozen would be a choice of
what to report made after seeing the data, which is the one failure the whole
campaign design exists to prevent. :class:`ProtocolBinding` states exactly what a
future protocol has to hand this module; every field is required and none has a
default, because a field this module could fill in for itself is a field it would
be choosing after the fact.

**A report is a pure function of persisted inputs, and it repairs nothing.**
Every file is opened read-only and nothing under the state directory moves. In
particular this module does not call
:func:`~chimera.demo.decision_log.recover_tail`, does not open a
:class:`~chimera.demo.decision_log.DecisionLog` (whose ``open`` verifies and can
raise on a torn tail), and never re-runs a replay. A torn tail, a forged record
and a failed reconciliation are *reported* — as a chain verdict and as records
quoted verbatim — never repaired, re-derived into a success, or smoothed over
by inferring the record that should have been there.

**Absence is reported as absence.** The runner at this revision writes only seven
of the twelve record kinds: :data:`RUNNER_WRITTEN_KINDS`. A campaign run by it
therefore produces no ``FUNDING``, ``RECONCILIATION``, ``LIQUIDATION_TOUCH``,
``RECOVERY`` or ``SKIPPED_STALE`` record at all — and a bare ``0`` beside those
names would read exactly like "nothing happened" when it means "this build cannot
say". So every daily report carries :data:`INPUT_COVERAGE_NOTE`'s block: per
kind, whether the day's log held one, and whether the runner in this build can
write one. The arithmetic below nevertheless handles all twelve kinds, because
the day PR-10's gap is closed the report has to be right without being rewritten.

**HALT, RESUME and RECOVERY are counted and not classified.** Section 9.4 puts
them in neither the evidence set nor the operational set, and
``chimera/demo/decision_log.py`` says in as many words that answering it is a
scientific decision belonging to the reports (this PR) and to the PVC-1 protocol
(PR-14). This module answers the half it can: the daily report *counts* them and
their causes, and it publishes ``unclassified`` as a third class rather than
folding them into ``operational``. It computes no total aggregated by class, so
nothing in a daily report can be read as a score. The other half — whether a
campaign's halts are scored — is refused rather than assumed: a monthly report
over a month holding any of the three is refused unless the protocol names a
treatment for all three by name.

**Two kind-sets exist in this repository and they are not interchangeable.**
``chimera.demo.decision_log``'s ``EVIDENCE_KINDS`` / ``OPERATIONAL_KINDS`` /
``UNCLASSIFIED_KINDS`` are section 9.4's *evidence boundary* — what an audit
scores. ``tools/replay_parity.py``'s ``OPERATIONAL_KINDS`` is section 10's
*alignment rule* — which records a parity comparison aligns by ``(minute, kind)``
instead of comparing field by field — and it is a different set, holding ``HALT``
and ``RESUME`` and not holding ``INCOMPLETE_STATE`` or ``SKIPPED_STALE``. Using
either where the other belongs would silently move the evidence boundary, so this
module imports the first, never imports ``tools`` at all, and says here why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from chimera.demo.decision_log import (
    EVIDENCE_KINDS,
    LOG_DIR_NAME,
    OPERATIONAL_KINDS,
    DecisionLogError,
    RecordKind,
    day_files,
    is_hash,
    read_records,
    require_iso_minute,
    require_kind,
    verify_log,
)

#: The payload identities. Bumped when a field changes meaning, never when one is
#: added, so a reader that pinned the schema can tell the two apart.
DAILY_REPORT_SCHEMA = "chimera.demo-daily-report/1"
MONTHLY_REPORT_SCHEMA = "chimera.demo-monthly-report/1"

#: Section 9.4's three classes, as the strings the payload carries.
CLASS_EVIDENCE = "evidence"
CLASS_OPERATIONAL = "operational"
CLASS_UNCLASSIFIED = "unclassified"

#: What a daily report *is*, stated in the payload. Not a computed field: there
#: is no arithmetic in this module that could make a daily report anything else,
#: and a reader who greps for a verdict should find this instead.
DAILY_REPORT_CLASS = "operational"

#: The evidence class ``tools/freeze_evidence.py`` reads out of a frozen monthly
#: artifact. It is neither ``derived`` (which that tool refuses to freeze) nor
#: absent (which it notes and freezes as primary), so a prospective campaign
#: month freezes as primary evidence with no note -- which is what section 11.4
#: relies on and the reason the string is written out here rather than left to a
#: caller.
EVIDENCE_CLASS_PROSPECTIVE = "prospective"

#: The two legs of a carry position, in the order ``chimera.carry.hedge`` names
#: them. Both keys are always present in the ``orders`` block, so "this leg
#: traded nothing" and "this report forgot this leg" cannot look the same.
LEGS: tuple[str, ...] = ("spot", "perp")

#: The kinds ``chimera/demo/runner.py`` can write at this revision, read off its
#: ``self._append(RecordKind.X, ...)`` call sites and pinned here because the
#: consequence is scientific rather than cosmetic: five of the twelve kinds have
#: no writer, ``HedgedPosition.settle_funding`` and ``.liquidation_touched`` have
#: no caller, and a campaign run today would therefore produce no funding
#: evidence and no reconciliation evidence at all. Emitting those records is
#: runner behaviour and belongs to the change that owns the runner; what belongs
#: here is refusing to present the resulting zeros as observations.
#: ``tests/test_demo_reports.py`` walks the runner's AST and fails if this tuple
#: and its call sites disagree, so the disclosure cannot quietly go stale.
RUNNER_WRITTEN_KINDS: tuple[str, ...] = (
    "DECISION",
    "HALT",
    "INCOMPLETE_STATE",
    "OPERATOR",
    "RESUME",
    "SHUTDOWN",
    "STARTUP",
)

#: Why every daily report carries ``input_coverage``. Written into the payload,
#: because the distinction it draws is the whole reason the block exists and a
#: reader of the JSON has no access to this module's docstring.
INPUT_COVERAGE_NOTE = (
    "per record kind: whether this day's log held one, and whether the runner in "
    "this build can write one at all. The two are different facts and a bare zero "
    "conflates them: five kinds -- FUNDING, RECONCILIATION, LIQUIDATION_TOUCH, "
    "RECOVERY and SKIPPED_STALE -- have no writer in chimera/demo/runner.py at "
    "this revision, so their counts below say 'this build cannot record it' and "
    "not 'it did not happen'. The arithmetic in this report handles all twelve "
    "kinds regardless, so it is already correct on a log that holds them."
)

#: How a free-text halt reason is collapsed into a bounded label, longest and
#: most specific prefix first. Every prefix is quoted from a live ``halt(...)``
#: or ``_halt(...)`` site -- ``chimera/demo/runner.py`` lines 303-313, 373, 397,
#: 402, 419, 456, 463, 473 and ``self_check``; ``chimera/risk.py`` 467, 479, 771,
#: 787, 797, 814; ``chimera/carry/hedge.py`` 275, 589 -- so the table describes
#: this repository rather than a vocabulary invented for a dashboard. The
#: fallback is ``other``: a reason nobody anticipated is grouped under a name
#: that says so, never renamed into one of these.
HALT_CAUSES: tuple[tuple[str, str], ...] = (
    ("kill_switch", "kill_switch"),
    ("dispute:", "dispute"),
    ("carry_dispute:", "dispute"),
    ("rule_exception:", "rule_exception"),
    ("identity_violation", "identity_violation"),
    ("more than one rule", "multiple_actionable_rules"),
    ("source_identity:", "source_identity"),
    ("log_behind_state:", "log_behind_state"),
    ("store_error:", "store_error"),
    ("max daily loss", "daily_loss"),
    ("max drawdown", "drawdown"),
    ("order rate limit", "order_rate"),
    ("equity is non-positive", "non_positive_equity"),
    ("risk halted", "risk_halted"),
)

#: The collapse target for a halt reason no prefix above matches.
HALT_CAUSE_OTHER = "other"

#: Amendment A10's convention, quoted into every report that reports funding.
#: The split is derived rather than recorded, so the payload states the rule it
#: was derived under and a reader can check the arithmetic against A10's table
#: without reading this module.
FUNDING_SIGN_CONVENTION = (
    "amendment A10: funding_cash_flow = -sign(side) * notional * rate, and a "
    "negative cash flow means the position paid. LONG pays a positive rate, SHORT "
    "receives one. `paid` and `received` are magnitudes and are never netted; "
    "`net` is received minus paid, which is CarryLedgerState.net_funding."
)

#: Where the paid/received split comes from, stated in the payload because it is
#: a derivation and not a recorded field.
FUNDING_SOURCE = (
    "the per-minute increment of ledger_effect.funding, which is "
    "CarryLedgerState.net_funding = funding_received - funding_paid. A negative "
    "increment is a settlement the position paid and a positive one is a "
    "settlement it received. The split is exact while at most one settlement is "
    "booked per record, which the contract's 8-hourly settlement schedule "
    "guarantees; a record carrying two would understate both magnitudes by the "
    "smaller of them, and no such record exists at this revision because no "
    "settlement is booked at all."
)

#: The quantities :func:`monthly_report` knows how to compute. Counting only, on
#: purpose: a monthly frozen report is prospective evidence, and a performance
#: statistic in this table would be a scoring rule chosen by an engineering
#: change rather than by a preregistered protocol. A protocol that needs another
#: quantity extends this table in the change that freezes the protocol, where the
#: choice is preregistered rather than incidental.
MONTHLY_QUANTITIES: tuple[str, ...] = (
    "records",
    "minutes_processed",
    "decisions",
    "incomplete_minutes",
)


class ReportError(RuntimeError):
    """A report cannot be produced from what is on disk."""


class ReportRefused(ReportError):
    """A report *could* have been produced and must not be.

    Separate from :class:`ReportError` because the two need different handling:
    an error is a defect to fix, and a refusal is the correct outcome of a
    correct program on inputs that do not authorise a result. Every refusal
    message ends by saying that nothing was written, because the caller acts on
    that and the exit code alone does not say it.
    """


def kind_class(kind: RecordKind | str) -> str:
    """Section 9.4's class of one record kind, with a third value for neither.

    Falling back to :data:`CLASS_OPERATIONAL` would classify ``HALT``, ``RESUME``
    and ``RECOVERY`` by omission, and section 9.4 says the operational kinds are
    not scored -- so the fallback would rule, without any adopted document saying
    so, that a campaign which halted repeatedly scores identically to one that
    never halted. The third value keeps the question visible in every payload.
    """
    resolved = require_kind(kind)
    if resolved in EVIDENCE_KINDS:
        return CLASS_EVIDENCE
    if resolved in OPERATIONAL_KINDS:
        return CLASS_OPERATIONAL
    return CLASS_UNCLASSIFIED


def halt_cause(detail: str) -> str:
    """A halt's free-text reason as one of :data:`HALT_CAUSES`' bounded labels.

    The verbatim reason is kept beside the label in every payload. This is a
    grouping key so that "seven halts, all disputes" can be said at all; it is
    not a replacement for the text, and a reason no prefix matches becomes
    ``other`` rather than a label derived from its own wording.
    """
    text = detail if isinstance(detail, str) else ""
    for prefix, label in HALT_CAUSES:
        if text.startswith(prefix):
            return label
    return HALT_CAUSE_OTHER


# ---------------------------------------------------------------------------
# rendering and reading primitives
# ---------------------------------------------------------------------------
def _plain(value: Decimal) -> str:
    """``format(value, "f")``, with a negative zero normalised, and never ``str``.

    ``str(Decimal("1E-8"))`` is ``"1E-8"``, and scientific notation in a fee
    field is a number two readers can disagree about.
    :func:`chimera.demo.decision_log.decimal_str` is not used because it requires
    a scale, and these totals are sums of quantities whose scales this module has
    no business inventing: the log states what the ledger held, and a report that
    re-quantised it would be reporting a different number.
    """
    if value == 0:
        value = abs(value)
    return format(value, "f")


def _decimal(value: Any, *, where: str) -> Decimal:
    """One string from the log as a ``Decimal``, or an explanation.

    A missing or unparseable money field is refused rather than defaulted to
    zero. A zero this module invented would be indistinguishable from a zero the
    campaign actually recorded, and every arithmetic result downstream would be
    quietly wrong by whatever the field held.
    """
    if isinstance(value, Decimal):
        return value
    if not isinstance(value, str) or not value:
        raise ReportError(
            f"{where} is {value!r}; a money or quantity field of a decision record is a "
            "decimal string, and this report will not substitute a zero for one it "
            "cannot read"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ReportError(f"{where} is {value!r}, which is not a decimal: {exc}") from exc
    if not parsed.is_finite():
        raise ReportError(f"{where} is {value!r}, which is not a finite decimal")
    return parsed


def _mapping(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """One record's sub-block, or an empty mapping when it carries none."""
    value = record.get(name)
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str | None:
    """A string field as itself, and anything else as ``None``."""
    return value if isinstance(value, str) else None


def _day_of(record: Mapping[str, Any]) -> str | None:
    """The UTC date of a record's own ``minute``, or ``None`` if it has none.

    **Not the day file it sits in.** ``DecisionLog.append`` files a record by
    ``utc_day(runner_now_ns)`` and ``DemoRunner.tick`` sets the clock to
    ``minute + 1``, so the 23:59 minute of every day is written into the *next*
    day's file, and catching up after downtime widens the gap arbitrarily. A
    report keyed on the file name would attribute the last minute of every day to
    the day after it, for ever, and the error would be invisible because both
    days would still look complete.

    ``minute`` may legitimately be ``null`` -- the log admits it for a kind with
    no decision minute -- and such a record belongs to no day. It is counted in
    ``generated_from.records_without_minute`` rather than being dropped in
    silence or attributed to a day it does not name.
    """
    minute = record.get("minute")
    if minute is None:
        return None
    try:
        stamp = require_iso_minute(minute, field_name="minute")
    except DecisionLogError as exc:
        raise ReportError(
            f"a record carries minute {minute!r}, which is not a canonical UTC minute: "
            f"{exc}. The report groups on this field and will not bucket a value it "
            "cannot read"
        ) from exc
    return stamp[:10]


def _sorted_distinct(values: Sequence[Any]) -> list[str]:
    """The distinct strings among ``values``, sorted. Empty stays empty."""
    return sorted({value for value in values if isinstance(value, str) and value})


def _first_seen(blocks: Sequence[Any]) -> list[Any]:
    """The distinct JSON blocks among ``blocks``, in the order first seen."""
    seen: list[str] = []
    out: list[Any] = []
    for block in blocks:
        key = json.dumps(block, sort_keys=True, separators=(",", ":"), default=str)
        if key in seen:
            continue
        seen.append(key)
        out.append(block)
    return out


def _log_dir(state_dir: str | Path) -> Path:
    """``<state_dir>/decision_log``, resolved the one way the writer resolves it."""
    return Path(state_dir) / LOG_DIR_NAME


def _scan(log_dir: Path) -> tuple[list[dict[str, Any]], list[Path], int, int]:
    """Every record of every day file, in chain order, and what was read.

    All of them, not a window: name order is chain order, a campaign month is a
    few tens of thousands of records, and any window heuristic would have to
    guess how far a catch-up moved a record from the day it describes. Reading
    the lot removes the guess, and ``generated_from`` names every file so a
    reader can check that it was read.
    """
    files = day_files(log_dir)
    records: list[dict[str, Any]] = []
    scanned = 0
    without_minute = 0
    for path in files:
        for record in read_records(path):
            scanned += 1
            if record.get("minute") is None:
                without_minute += 1
                continue
            records.append(record)
    return records, files, scanned, without_minute


def _chain_block(log_dir: Path) -> dict[str, Any]:
    """``verify_log``'s verdict, reported and never acted on.

    ``verify_log`` is read-only, and this is the only verification the report
    performs. A torn tail stays torn and a forged record stays in the file: both
    are findings this report states, and repairing either would be a report
    changing the evidence it was asked to describe.
    """
    try:
        verification = verify_log(log_dir)
    except DecisionLogError as exc:
        raise ReportError(
            f"the decision log at {log_dir} could not be verified: {exc}"
        ) from exc
    return {
        "ok": bool(verification.ok),
        "records": int(verification.records),
        "first_seq": verification.first_seq,
        "last_seq": verification.last_seq,
        "faults": [fault.value for fault in verification.faults],
        "torn_tail_bytes": int(verification.torn_tail_bytes),
        "summary": verification.summary(),
    }


# ---------------------------------------------------------------------------
# the daily report, block by block
# ---------------------------------------------------------------------------
def _require_day(day: Any) -> str:
    """A ``YYYY-MM-DD`` UTC day, or an explanation of why that string is not one."""
    if not isinstance(day, str):
        raise ReportError(f"day must be a YYYY-MM-DD string, got {type(day).__name__}")
    try:
        parsed = datetime.strptime(day, "%Y-%m-%d")
    except ValueError as exc:
        raise ReportError(f"day {day!r} is not a YYYY-MM-DD UTC date: {exc}") from exc
    return parsed.strftime("%Y-%m-%d")


def _event(record: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """Where a record is, so a reader can find the line this row came from."""
    return {"minute": record.get("minute"), "seq": record.get("seq"), **extra}


def _campaign_block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which campaign, which configuration and which build wrote this day.

    ``protocol_frozen`` is read from the ``STARTUP`` records and is ``None`` when
    the day holds none -- a day in the middle of a run has no STARTUP record, and
    reporting ``false`` there would be this module asserting something it did not
    observe.
    """
    startups = [r for r in selected if r.get("kind") == RecordKind.STARTUP.value]
    frozen: Any = None
    for record in startups:
        value = record.get("protocol_frozen")
        if isinstance(value, bool):
            frozen = value
    return {
        "campaign_ids": _sorted_distinct([r.get("campaign_id") for r in startups]),
        "config_hashes": _sorted_distinct([r.get("config_hash") for r in selected]),
        "software": _first_seen(
            [r["software"] for r in selected if isinstance(r.get("software"), Mapping)]
        ),
        "contract_hashes": _sorted_distinct(
            [_mapping(r, "inputs").get("contract_hash") for r in selected]
        ),
        "protocol_frozen": frozen,
    }


def _minutes_block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the day covered, and what it did not.

    ``minutes_absent`` is ``1440 - distinct_minutes_present`` and is nothing more
    than that. It is not a gap count and carries no cause: a minute with no
    record may be one the runner never reached, one it was not running for, or
    one the recorder never wrote, and the log cannot tell those apart. Naming a
    cause here would be inferring a record that is not there.
    """
    processed_kinds = {
        RecordKind.DECISION.value,
        RecordKind.INCOMPLETE_STATE.value,
        RecordKind.SKIPPED_STALE.value,
    }
    minutes = sorted({str(r.get("minute")) for r in selected})
    incomplete = [r for r in selected if r.get("kind") == RecordKind.INCOMPLETE_STATE.value]
    return {
        "first": minutes[0] if minutes else None,
        "last": minutes[-1] if minutes else None,
        "processed": sum(1 for r in selected if r.get("kind") in processed_kinds),
        "decisions": sum(1 for r in selected if r.get("kind") == RecordKind.DECISION.value),
        "incomplete": len(incomplete),
        "skipped_stale": sum(
            1 for r in selected if r.get("kind") == RecordKind.SKIPPED_STALE.value
        ),
        "incomplete_detail": [
            _event(r, missing=_text(_mapping(r, "veto_or_rejection").get("detail")))
            for r in incomplete
        ],
        "distinct_minutes_present": len(minutes),
        "minutes_in_a_utc_day": 1440,
        "minutes_absent": 1440 - len(minutes),
    }


def _records_by_kind(selected: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """All twelve kinds, zero-filled. A kind absent from the payload would read
    as a kind this report does not know about."""
    counts = {kind.value: 0 for kind in RecordKind}
    for record in selected:
        counts[str(record.get("kind"))] += 1
    return counts


def _input_coverage(counts: Mapping[str, int]) -> dict[str, Any]:
    """Section 9.4's twelve kinds against what this build can actually write."""
    return {
        "note": INPUT_COVERAGE_NOTE,
        "kinds_the_runner_can_write": list(RUNNER_WRITTEN_KINDS),
        "by_kind": {
            kind.value: {
                "present": counts[kind.value] > 0,
                "records": counts[kind.value],
                "runner_can_write": kind.value in RUNNER_WRITTEN_KINDS,
            }
            for kind in RecordKind
        },
    }


def _rules_block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per rule: how many decisions it contributed to, and of which signal kind.

    A record names its primary rule in ``rule``/``signal`` and every other rule
    of the minute in ``shadow``. When no rule produced an actionable target the
    primary IS one of the shadow rules, so the same rule appears twice in one
    record; each rule is therefore counted once per record rather than once per
    mention, which is what makes the totals add up to the decision count.
    """
    tally: dict[str, dict[str, int]] = {}
    for record in selected:
        if record.get("kind") != RecordKind.DECISION.value:
            continue
        per_record: dict[str, str] = {}
        rule_id = _text(_mapping(record, "rule").get("id"))
        if rule_id:
            per_record[rule_id] = _text(_mapping(record, "signal").get("kind")) or "unknown"
        shadow = record.get("shadow")
        for entry in shadow if isinstance(shadow, list) else []:
            if not isinstance(entry, Mapping):
                continue
            name = _text(entry.get("rule_id"))
            if name:
                per_record[name] = _text(entry.get("kind")) or "unknown"
        for name, signal_kind in per_record.items():
            seen = tally.setdefault(name, {"decisions": 0, "signal_kinds": {}})
            seen["decisions"] += 1
            kinds = seen["signal_kinds"]
            kinds[signal_kind] = kinds.get(signal_kind, 0) + 1
    return {name: tally[name] for name in sorted(tally)}


def _orders_block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Orders, filled quantity and fees per leg, from the records' ``execution``.

    A record's ``execution`` list holds only the orders that minute ADDED --
    ``DemoRunner`` diffs the executor's store against the ids it held before the
    tick -- so summing over the day counts each order once without any
    de-duplication here. ``entries`` is reported beside ``orders`` anyway: if the
    two ever disagree, an order was named in two minutes and the sums below
    double-count it, and a reader can see that rather than having to trust it.

    No fill count and no average fill price: the record carries per-order
    aggregates as of the minute the order appeared and carries no per-fill event,
    so a "fills" number here would be an order count wearing another name.
    """
    legs: dict[str, dict[str, Any]] = {
        leg: {
            "orders": [],
            "entries": 0,
            "by_state": {},
            "filled": Decimal("0"),
            "fees": Decimal("0"),
        }
        for leg in LEGS
    }
    for record in selected:
        execution = record.get("execution")
        for index, entry in enumerate(execution if isinstance(execution, list) else []):
            if not isinstance(entry, Mapping):
                continue
            leg = _text(entry.get("leg")) or "unknown"
            where = f"record seq {record.get('seq')} execution[{index}]"
            block = legs.setdefault(
                leg,
                {
                    "orders": [],
                    "entries": 0,
                    "by_state": {},
                    "filled": Decimal("0"),
                    "fees": Decimal("0"),
                },
            )
            block["entries"] += 1
            order_id = _text(entry.get("order_id"))
            if order_id and order_id not in block["orders"]:
                block["orders"].append(order_id)
            state = _text(entry.get("state")) or "unknown"
            block["by_state"][state] = block["by_state"].get(state, 0) + 1
            block["filled"] += _decimal(entry.get("filled_qty"), where=f"{where}.filled_qty")
            block["fees"] += _decimal(entry.get("fees"), where=f"{where}.fees")
    return {
        leg: {
            "orders": len(block["orders"]),
            "entries": block["entries"],
            "by_state": {
                state: block["by_state"][state] for state in sorted(block["by_state"])
            },
            "filled_quantity": _plain(block["filled"]),
            "fees": _plain(block["fees"]),
        }
        for leg, block in sorted(legs.items())
    }


_LEDGER_COSTS: tuple[str, ...] = ("fees", "slippage", "realised")

#: What ``ledger.baseline.source`` means, spelled out in the payload.
_BASELINE_NOTES: Mapping[str, str] = {
    "previous_record": (
        "the last record before this day that carried a ledger_effect, in chain order. "
        "That is the state at 00:00 UTC taken from the evidence itself. It is NOT read "
        "from carry_ledger.json or risk.json: those hold the CURRENT snapshot, so a past "
        "day read from them would be attributed today's totals, and a replayed log -- "
        "whose stores live in a scratch directory -- would report different numbers for "
        "the same evidence."
    ),
    "campaign_start": (
        "no record before this day carries a ledger_effect, so the cumulative cost fields "
        "open at zero and the opening equity is the day's own first marked equity."
    ),
}


def _ledger_and_funding(
    selected: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The day's cost, equity and funding movement, from ``ledger_effect`` alone.

    Every record that carries a ``ledger_effect`` is used, not only the
    ``DECISION`` records: section 9.1 gives one record shape to all twelve kinds,
    so a ``FUNDING`` record -- which the runner cannot write yet and one day will
    -- moves ``ledger_effect.funding`` exactly as a decision minute does. Reading
    only decisions would drop a settlement's whole cash flow the day funding is
    booked at all, which is precisely the kind of silent zero this report exists
    not to produce.

    ``equity.worst`` is the minimum over THIS DAY's marked records, not
    ``CarryLedgerState.worst_equity``: that field is campaign-to-date, it is in a
    mutable store rather than in the log, and reading it would make a daily
    report depend on when it was run.
    """
    marked = [r for r in selected if _mapping(r, "ledger_effect")]
    if baseline is None:
        opens = {name: Decimal("0") for name in _LEDGER_COSTS}
        prev_funding = Decimal("0")
        open_equity: Decimal | None = None
        if marked:
            first = _mapping(marked[0], "ledger_effect")
            open_equity = _decimal(first.get("equity"), where="the day's first equity")
        baseline_block = {
            "source": "campaign_start",
            "minute": None,
            "seq": None,
            "note": _BASELINE_NOTES["campaign_start"],
        }
    else:
        effect = _mapping(baseline, "ledger_effect")
        where = f"baseline record seq {baseline.get('seq')} ledger_effect"
        opens = {
            name: _decimal(effect.get(name), where=f"{where}.{name}") for name in _LEDGER_COSTS
        }
        prev_funding = _decimal(effect.get("funding"), where=f"{where}.funding")
        open_equity = _decimal(effect.get("equity"), where=f"{where}.equity")
        baseline_block = {
            "source": "previous_record",
            "minute": baseline.get("minute"),
            "seq": baseline.get("seq"),
            "note": _BASELINE_NOTES["previous_record"],
        }

    closes: dict[str, Decimal | None] = {name: None for name in _LEDGER_COSTS}
    close_equity: Decimal | None = None
    worst: Decimal | None = None
    worst_minute: str | None = None
    paid = Decimal("0")
    received = Decimal("0")
    settlements = 0
    funding_close = prev_funding

    for record in marked:
        effect = _mapping(record, "ledger_effect")
        where = f"record seq {record.get('seq')} ledger_effect"
        for name in _LEDGER_COSTS:
            closes[name] = _decimal(effect.get(name), where=f"{where}.{name}")
        equity = _decimal(effect.get("equity"), where=f"{where}.equity")
        close_equity = equity
        if worst is None or equity < worst:
            worst, worst_minute = equity, _text(record.get("minute"))
        funding = _decimal(effect.get("funding"), where=f"{where}.funding")
        delta = funding - funding_close
        funding_close = funding
        if delta < 0:
            paid += -delta
            settlements += 1
        elif delta > 0:
            received += delta
            settlements += 1

    ledger = {
        "baseline": baseline_block,
        **{
            name: {
                "open": _plain(opens[name]),
                "close": None if closes[name] is None else _plain(closes[name]),
                "delta": (
                    None if closes[name] is None else _plain(closes[name] - opens[name])
                ),
            }
            for name in _LEDGER_COSTS
        },
        "equity": {
            "open": None if open_equity is None else _plain(open_equity),
            "close": None if close_equity is None else _plain(close_equity),
            "worst": None if worst is None else _plain(worst),
            "worst_minute": worst_minute,
        },
    }
    funding_block = {
        "net": _plain(funding_close - prev_funding),
        "paid": _plain(paid),
        "received": _plain(received),
        "settlement_minutes": settlements,
        "sign_convention": FUNDING_SIGN_CONVENTION,
        "source": FUNDING_SOURCE,
        "records_of_kind_FUNDING": sum(
            1 for r in selected if r.get("kind") == RecordKind.FUNDING.value
        ),
    }
    return ledger, funding_block


#: Why the daily report counts the three unclassified kinds without classifying
#: them. Carried in the payload so the boundary travels with the number.
HALT_CLASSIFICATION_NOTE = (
    "HALT, RESUME and RECOVERY are the three kinds section 9.4 puts in neither the "
    "evidence set nor the operational set (chimera/demo/decision_log.py). This "
    "report counts them and their causes and classifies them as 'unclassified'. "
    "Whether a campaign's halts are scored is the PVC-1 protocol's answer, and "
    "this report neither gives it nor persists one."
)


def _halts_block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Halts, resumes and recoveries: counted, caused, and left unclassified."""
    halts = [r for r in selected if r.get("kind") == RecordKind.HALT.value]
    resumes = [r for r in selected if r.get("kind") == RecordKind.RESUME.value]
    recoveries = [r for r in selected if r.get("kind") == RecordKind.RECOVERY.value]
    causes: dict[str, int] = {}
    events = []
    for record in halts:
        detail = _text(_mapping(record, "veto_or_rejection").get("detail")) or ""
        cause = halt_cause(detail)
        causes[cause] = causes.get(cause, 0) + 1
        events.append(_event(record, cause=cause, detail=detail or None))
    return {
        "count": len(halts),
        "events": events,
        "resumes": [
            _event(
                record,
                note=_text(_mapping(record, "operator").get("note")),
                cleared=_text(_mapping(record, "operator").get("cleared")),
            )
            for record in resumes
        ],
        "recoveries": len(recoveries),
        "recovery_events": [
            _event(record, detail=_text(_mapping(record, "veto_or_rejection").get("detail")))
            for record in recoveries
        ],
        "causes": {cause: causes[cause] for cause in sorted(causes)},
        "classification": CLASS_UNCLASSIFIED,
        "classification_note": HALT_CLASSIFICATION_NOTE,
    }


#: Why ``reconciliation.outcomes`` is null on a log holding no RECONCILIATION
#: record, rather than an empty list or a clean bill of health.
RECONCILIATION_NOTE = (
    "outcomes is null when the day's log holds no RECONCILIATION record: this "
    "report has no reconciliation outcome to state, which is not the same as "
    "stating that every reconciliation succeeded. Where such records exist their "
    "veto_or_rejection and position_after are quoted verbatim and are never "
    "re-derived: a reconciliation the log records as failed stays failed here. "
    "dispute_halts counts HALT records whose reason collapses to 'dispute' and "
    "operator_resolutions lists OPERATOR records whose command is 'resolve'; "
    "neither is a judgement about whether the disagreement was real."
)

#: Why a LIQUIDATION_TOUCH block exists on a build that cannot write one.
LIQUIDATION_NOTE = (
    "HedgedPosition.liquidation_touched has no caller at this revision, so no run "
    "of this build produces a LIQUIDATION_TOUCH record. The block is computed "
    "from the log anyway, so it is already right on a log that holds them; "
    "input_coverage is where the difference between 'none happened' and 'none "
    "can be written' is stated."
)

#: The six operator commands this report groups by, plus the collapse target.
#: ``startup`` and ``shutdown`` come from their own kinds rather than from an
#: OPERATOR record; ``resume`` is the RESUME kind. A command outside this set is
#: reported under ``other`` with its text, never renamed into one of these.
_OPERATOR_COMMANDS: tuple[str, ...] = ("flatten", "resolve")
_OPERATOR_OTHER = "other"


def _reconciliation_block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the log says about disagreements between the two legs' stories."""
    records = [r for r in selected if r.get("kind") == RecordKind.RECONCILIATION.value]
    resolutions = [
        r
        for r in selected
        if r.get("kind") == RecordKind.OPERATOR.value
        and _mapping(r, "operator").get("command") == "resolve"
    ]
    dispute_halts = sum(
        1
        for r in selected
        if r.get("kind") == RecordKind.HALT.value
        and halt_cause(_text(_mapping(r, "veto_or_rejection").get("detail")) or "")
        == "dispute"
    )
    return {
        "records_of_kind_RECONCILIATION": len(records),
        "outcomes": (
            None
            if not records
            else [
                _event(
                    record,
                    detail=_text(_mapping(record, "veto_or_rejection").get("detail")),
                    position_after=record.get("position_after"),
                )
                for record in records
            ]
        ),
        "dispute_halts": dispute_halts,
        "operator_resolutions": [
            _event(
                record,
                symbol=_text(_mapping(record, "operator").get("symbol")),
                note=_text(_mapping(record, "operator").get("note")),
            )
            for record in resolutions
        ],
        "note": RECONCILIATION_NOTE,
    }


def _liquidation_block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every LIQUIDATION_TOUCH the day recorded, quoted rather than interpreted."""
    records = [r for r in selected if r.get("kind") == RecordKind.LIQUIDATION_TOUCH.value]
    return {
        "records_of_kind_LIQUIDATION_TOUCH": len(records),
        "events": [
            _event(
                record,
                detail=_text(_mapping(record, "veto_or_rejection").get("detail")),
                position_after=record.get("position_after"),
            )
            for record in records
        ],
        "note": LIQUIDATION_NOTE,
    }


def _operator_block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every operator action and lifecycle record of the day, with its note."""
    block: dict[str, list[dict[str, Any]]] = {
        "startup": [],
        "shutdown": [],
        "resume": [],
        _OPERATOR_OTHER: [],
    }
    for name in _OPERATOR_COMMANDS:
        block[name] = []
    for record in selected:
        kind = record.get("kind")
        operator = _mapping(record, "operator")
        if kind == RecordKind.STARTUP.value:
            block["startup"].append(_event(record))
        elif kind == RecordKind.SHUTDOWN.value:
            block["shutdown"].append(_event(record, note=_text(operator.get("note"))))
        elif kind == RecordKind.RESUME.value:
            block["resume"].append(
                _event(
                    record,
                    note=_text(operator.get("note")),
                    cleared=_text(operator.get("cleared")),
                )
            )
        elif kind == RecordKind.OPERATOR.value:
            command = _text(operator.get("command"))
            where = command if command in _OPERATOR_COMMANDS else _OPERATOR_OTHER
            block[where].append(
                _event(
                    record,
                    command=command,
                    symbol=_text(operator.get("symbol")),
                    note=_text(operator.get("note")),
                )
            )
    return {name: block[name] for name in sorted(block)}


#: Why the veto tally holds the halts too.
VETO_NOTE = (
    "every non-null veto_or_rejection of the day, grouped by 'stage/label'. That "
    "includes the runner/halt entry each HALT record carries, because this counts "
    "the field rather than a subset of it chosen here; halts are also counted, "
    "with their causes, in the halts block. risk.decisions is not counted "
    "separately: it restates the same veto."
)


def _vetoes_block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Vetoes and rejections, grouped by the two bounded fields they carry."""
    tally: dict[str, int] = {}
    total = 0
    for record in selected:
        veto = record.get("veto_or_rejection")
        if not isinstance(veto, Mapping):
            continue
        total += 1
        key = f"{_text(veto.get('stage')) or '?'}/{_text(veto.get('label')) or '?'}"
        tally[key] = tally.get(key, 0) + 1
    return {
        "count": total,
        "by_stage_and_label": {key: tally[key] for key in sorted(tally)},
        "note": VETO_NOTE,
    }


def _position_close(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The last ``position_after`` of the day, verbatim, and which record it is.

    The last record that carries one rather than the last DECISION: an operator
    ``flatten`` writes an OPERATOR record with a ``position_after`` and no
    decision, so keying on DECISION would report the position the campaign held
    before the flatten as the position it closed the day on.
    """
    for record in reversed(selected):
        block = record.get("position_after")
        if isinstance(block, Mapping):
            return {
                "source": {
                    "minute": record.get("minute"),
                    "seq": record.get("seq"),
                    "kind": record.get("kind"),
                },
                "position_after": dict(block),
            }
    return None


#: Why ``parity`` is null rather than a pass when no verdict file is there.
PARITY_NOTE = (
    "a replay-parity verdict is read from <parity_dir>/<day>.json when one is "
    "present and embedded verbatim; null means no verdict was found, which is not "
    "a pass. This report never runs a replay: doing so would make it a function "
    "of a second execution rather than of the persisted log. The runbook's "
    "replay-parity step is what writes the file, by redirecting "
    "`python -m tools.replay_parity --json` into it."
)


def _parity_block(parity_dir: str | Path | None, day: str) -> tuple[Any, list[Path]]:
    """The day's parity verdict as written, and the file it was read from."""
    if parity_dir is None:
        return None, []
    path = Path(parity_dir) / f"{day}.json"
    if not path.is_file():
        return None, []
    try:
        verdict = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportError(
            f"the parity verdict at {path} could not be read: {exc}. It is reported as "
            "unreadable rather than as absent, because the two mean different things"
        ) from exc
    return {"present": True, "path": path.as_posix(), "report": verdict}, [path]


#: How the day was chosen, stated in the payload because getting it wrong is
#: invisible: both days would still look complete.
SELECTION_RULE = (
    "the UTC date of each record's own `minute` field, over every day file the log "
    "holds. Not the day file the record sits in: DecisionLog.append files a record "
    "by utc_day(runner_now_ns) and DemoRunner.tick sets that clock to minute + 1, "
    "so the 23:59 minute of every day is written into the next day's file, and a "
    "catch-up after downtime widens the gap arbitrarily."
)

#: What this report did to the state directory, stated in the payload.
PURITY_NOTE = (
    "this report opened the files listed under `read`, read-only, and wrote none "
    "of them. It repaired no torn tail, mutated no store, re-ran no replay, "
    "reinterpreted no failed reconciliation and inferred no record that is not in "
    "the log. Where the log is torn or a record does not verify, the chain block "
    "says so and the bytes are left exactly as they are."
)


def daily_report(
    state_dir: str | Path, day: str, *, parity_dir: str | Path | None = None
) -> dict[str, Any]:
    """Section 11.4's daily operational report over one UTC day of the log.

    ``state_dir`` is the runner's state directory -- the decision log lives at
    ``<state_dir>/decision_log`` -- and the name differs from section 12.3's
    ``root`` only because that is where the log actually is. Nothing else is
    read: no carry ledger, no futures store, no risk state, no runner state. Each
    of those holds the CURRENT snapshot rather than the day's, so a report that
    consulted them would give a past day today's totals and would give different
    answers for a replayed log, whose stores live in a scratch directory.
    """
    day = _require_day(day)
    log_dir = _log_dir(state_dir)
    if not log_dir.is_dir():
        raise ReportRefused(
            f"{log_dir} is not a directory, so there is no decision log to report on. "
            "A report over an absent log would say 0 minutes processed and 'chain "
            "intact', which is what a day the runner sat idle also says -- and the "
            "difference between them is the whole question. Check the state_dir in "
            "the configuration."
        )
    records, files, scanned, without_minute = _scan(log_dir)
    if not files:
        raise ReportRefused(
            f"{log_dir} holds no day file at all. The runner writes a STARTUP record "
            "the first time it starts, so an empty log directory means it never ran "
            "against this state directory rather than that it ran and did nothing."
        )
    dated = [(_day_of(record), record) for record in records]

    selected = [record for date, record in dated if date == day]
    baseline: Mapping[str, Any] | None = None
    for date, record in dated:
        if date is not None and date < day and _mapping(record, "ledger_effect"):
            baseline = record

    counts = _records_by_kind(selected)
    ledger, funding = _ledger_and_funding(selected, baseline)
    parity, parity_paths = _parity_block(parity_dir, day)

    return {
        "schema": DAILY_REPORT_SCHEMA,
        "report_class": DAILY_REPORT_CLASS,
        "day": day,
        "generated_from": {
            "state_dir": Path(state_dir).as_posix(),
            "log_dir": log_dir.as_posix(),
            "day_files": [path.name for path in files],
            "records_scanned": scanned,
            "records_selected": len(selected),
            "records_without_minute": without_minute,
            "day_file_present": f"{day}.ndjson" in {path.name for path in files},
            "selection_rule": SELECTION_RULE,
        },
        "campaign": _campaign_block(selected),
        "chain": _chain_block(log_dir),
        "minutes": _minutes_block(selected),
        "records_by_kind": counts,
        "record_class_by_kind": {kind.value: kind_class(kind) for kind in RecordKind},
        "input_coverage": _input_coverage(counts),
        "rules": _rules_block(selected),
        "orders": _orders_block(selected),
        "ledger": ledger,
        "funding": funding,
        "halts": _halts_block(selected),
        "reconciliation": _reconciliation_block(selected),
        "liquidation": _liquidation_block(selected),
        "operator": _operator_block(selected),
        "vetoes": _vetoes_block(selected),
        "position_close": _position_close(selected),
        "parity": parity,
        "parity_note": PARITY_NOTE,
        "purity": {
            "read": [path.as_posix() for path in (*files, *parity_paths)],
            "written": [],
            "note": PURITY_NOTE,
        },
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _cell(value: Any) -> str:
    """One table cell. ``null`` is shown as an em dash, never as ``0`` or blank."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def render_daily_markdown(report: Mapping[str, Any]) -> str:
    """The daily report as Markdown, quoting the JSON's own strings.

    The renderer performs no arithmetic: every money and quantity string below is
    the exact string :func:`daily_report` put in the payload. Two renderings of
    one number that disagreed in their last digit would be two numbers, and the
    one an operator pasted into a review would be whichever they had open.
    """
    minutes = report["minutes"]
    ledger = report["ledger"]
    funding = report["funding"]
    halts = report["halts"]
    lines: list[str] = [
        f"# Demo campaign — {report['day']}",
        "",
        "**Operational report.** Counting and arithmetic over the decision log. It "
        "scores nothing, and it is not evidence for or against any research question.",
        "",
        "| | |",
        "| --- | --- |",
        f"| campaign | `{', '.join(report['campaign']['campaign_ids']) or '—'}` |",
        f"| config hash | `{', '.join(report['campaign']['config_hashes']) or '—'}` |",
        f"| chain | {_cell(report['chain']['summary'])} |",
        f"| minutes processed | {minutes['processed']} of "
        f"{minutes['minutes_in_a_utc_day']} |",
        f"| incomplete minutes | {minutes['incomplete']} |",
        f"| decisions | {minutes['decisions']} |",
        "",
        "## Records by kind",
        "",
        "| kind | count | section 9.4 class | in this log | writable by this build |",
        "| --- | ---: | --- | --- | --- |",
    ]
    coverage = report["input_coverage"]["by_kind"]
    for kind in RecordKind:
        name = kind.value
        entry = coverage[name]
        lines.append(
            f"| {name} | {report['records_by_kind'][name]} | "
            f"{report['record_class_by_kind'][name]} | {_cell(entry['present'])} | "
            f"{_cell(entry['runner_can_write'])} |"
        )
    lines += ["", report["input_coverage"]["note"], "", "## Orders and fills, per leg", ""]
    lines += [
        "| leg | orders | filled quantity | fees | states |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for leg, block in report["orders"].items():
        states = ", ".join(f"{state}={count}" for state, count in block["by_state"].items())
        lines.append(
            f"| {leg} | {block['orders']} | {block['filled_quantity']} | "
            f"{block['fees']} | {states or '—'} |"
        )
    lines += [
        "",
        "## Ledger",
        "",
        "| quantity | open | close | day |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in _LEDGER_COSTS:
        row = ledger[name]
        lines.append(
            f"| {name} | {_cell(row['open'])} | {_cell(row['close'])} | "
            f"{_cell(row['delta'])} |"
        )
    equity = ledger["equity"]
    lines += [
        "",
        f"Equity opened at `{_cell(equity['open'])}`, closed at "
        f"`{_cell(equity['close'])}`, worst `{_cell(equity['worst'])}` at "
        f"`{_cell(equity['worst_minute'])}`.",
        "",
        f"Baseline: {ledger['baseline']['source']}. {ledger['baseline']['note']}",
        "",
        "## Funding",
        "",
        f"Paid `{funding['paid']}`, received `{funding['received']}`, net "
        f"`{funding['net']}`, over {funding['settlement_minutes']} settlement(s).",
        "",
        funding["sign_convention"],
        "",
        funding["source"],
        "",
        "## Halts",
        "",
    ]
    if halts["events"]:
        lines += [
            f"- `{event['minute']}` — `{event['cause']}` — {_cell(event['detail'])}"
            for event in halts["events"]
        ]
    else:
        lines.append("No HALT record on this day.")
    lines += [
        "",
        f"Resumes {len(halts['resumes'])}, recoveries {halts['recoveries']}.",
        "",
        halts["classification_note"],
        "",
        "## Reconciliation",
        "",
        report["reconciliation"]["note"],
        "",
    ]
    for entry in report["reconciliation"]["operator_resolutions"]:
        lines.append(
            f"- `{entry['minute']}` — resolve `{_cell(entry['symbol'])}` — "
            f"{_cell(entry['note'])}"
        )
    lines += ["", "## Replay parity", ""]
    if report["parity"] is None:
        lines.append(report["parity_note"])
    else:
        lines += [
            f"Verdict read from `{report['parity']['path']}`:",
            "",
            "```json",
            json.dumps(report["parity"]["report"], indent=2, sort_keys=True),
            "```",
        ]
    lines += ["", "---", "", report["purity"]["note"], ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the monthly report, which refuses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProtocolBinding:
    """What a future PVC-1 protocol must hand this module before a month is computed.

    Every field is required and none has a default. A monthly frozen report is
    the prospective evidence of a preregistered protocol, so a field this module
    could fill in for itself would be a parameter of the result chosen after
    seeing the data -- which is the failure the whole preregistration discipline
    exists to prevent. There is deliberately no constructor that reads
    ``conf/demo/pvc1.json``: today that file carries ``protocol_hash: null``, and
    a helper that turned a null into a default would make the refusal below
    unreachable through exactly the path an operator would use.

    ``quantities`` is a whitelist and not a hint: the payload holds these names
    and no others, and a name this module cannot compute is refused rather than
    omitted.
    """

    protocol_hash: str
    contract_hash: str
    prospective_from: str | None
    campaign_start: str
    source_identity: Mapping[str, str]
    quantities: tuple[str, ...]
    excluded_minutes: tuple[str, ...]
    unclassified_treatment: Mapping[str, str]


#: The three kinds a protocol must rule on before a month holding any of them can
#: be computed. Sorted, because the refusal compares a sorted list.
_UNCLASSIFIED_NAMES: tuple[str, ...] = ("HALT", "RECOVERY", "RESUME")

_NOTHING_WRITTEN = "Nothing was written."


def _require_month(month: Any) -> str:
    """A ``YYYY-MM`` UTC month, or an explanation of why that string is not one."""
    if not isinstance(month, str):
        raise ReportError(f"month must be a YYYY-MM string, got {type(month).__name__}")
    try:
        parsed = datetime.strptime(month, "%Y-%m")
    except ValueError as exc:
        raise ReportError(f"month {month!r} is not a YYYY-MM UTC month: {exc}") from exc
    return parsed.strftime("%Y-%m")


def monthly_report(
    state_dir: str | Path, month: str, *, binding: ProtocolBinding | None
) -> dict[str, Any]:
    """One month as prospective evidence, or a refusal saying why it is not.

    Nine conditions are checked and every one of them refuses. Six are checked
    before a file is opened and the other three the moment the log has been read,
    because they are questions about the log; nothing is computed until all nine
    have passed, and this function never writes anywhere at all -- the caller
    that would write an artifact does so only after this returns.

    On the repository as it stands, refusals 1 and 4 both hold:
    ``conf/demo/pvc1.json`` carries ``protocol_hash: null`` so no binding can be
    built, and the recorder contract carries ``prospective_from: null`` so no
    minute recorded under it is prospective evidence in the first place. That is
    the intended state, not a gap to work around: PR-14 freezes the PVC-1
    protocol and writes that hash, and until it has, any monthly number would be
    a choice of what to report made after seeing the data.
    """
    month = _require_month(month)

    if binding is None:
        raise ReportRefused(
            "a monthly frozen report is the prospective evidence of a preregistered "
            "protocol, and this repository has none: conf/demo/pvc1.json carries "
            "protocol_hash=null, so nothing has been preregistered for the minutes this "
            "month holds. PR-14 freezes the PVC-1 protocol and writes that hash; until "
            f"it has, computing a monthly number would be choosing what to report after "
            f"seeing the data. {_NOTHING_WRITTEN}"
        )
    if not is_hash(binding.protocol_hash):
        raise ReportRefused(
            f"protocol_hash {binding.protocol_hash!r} is not sha256:<64 hex>; a protocol "
            "identified by a malformed hash is a protocol nobody can check the report "
            f"against. {_NOTHING_WRITTEN}"
        )
    if not is_hash(binding.contract_hash):
        raise ReportRefused(
            f"contract_hash {binding.contract_hash!r} is not sha256:<64 hex>; the "
            "recorder contract is what says which minutes exist and under what rules, "
            f"and an unidentifiable one identifies nothing. {_NOTHING_WRITTEN}"
        )
    if binding.prospective_from is None:
        raise ReportRefused(
            "the recorder contract carries prospective_from=null, so no minute recorded "
            "under it is scientific evidence (the contract's boundary rule). A monthly "
            "frozen report over engineering minutes would present them as prospective "
            f"evidence. {_NOTHING_WRITTEN}"
        )
    if binding.campaign_start < binding.prospective_from:
        raise ReportRefused(
            f"the campaign start {binding.campaign_start} precedes the prospective "
            f"boundary {binding.prospective_from}; minutes before the boundary are "
            f"engineering data permanently. {_NOTHING_WRITTEN}"
        )
    if not binding.quantities:
        raise ReportRefused(
            "the protocol names no quantity, so there is nothing this report is "
            "authorised to compute. A frozen artifact holding no protocol-named "
            f"quantity would be an empty claim wearing a manifest. {_NOTHING_WRITTEN}"
        )
    unknown = [name for name in binding.quantities if name not in MONTHLY_QUANTITIES]
    if unknown:
        raise ReportRefused(
            f"{unknown!r} is not among the quantities this report computes "
            f"({list(MONTHLY_QUANTITIES)}); the monthly report computes only the "
            "quantities the S2 protocol names and only the ones whose definition is "
            "already committed here, so a protocol asking for another needs that "
            f"definition frozen in the same change. {_NOTHING_WRITTEN}"
        )

    log_dir = _log_dir(state_dir)
    records, _files, _scanned, _no_minute = _scan(log_dir)
    excluded = set(binding.excluded_minutes)
    dated = [(_day_of(record), record) for record in records]
    in_month = [
        record
        for date, record in dated
        if date is not None and date[:7] == month and record.get("minute") not in excluded
    ]

    if not in_month:
        raise ReportRefused(
            f"the log holds no record for {month} that the protocol does not exclude. A "
            "monthly frozen report over nothing would present zeros as prospective "
            f"evidence. {_NOTHING_WRITTEN}"
        )
    chain = _chain_block(log_dir)
    if not chain["ok"]:
        raise ReportRefused(
            f"the decision log does not verify ({chain['summary']}); records that cannot "
            "be shown to be unaltered are not evidence, and freezing them would put a "
            f"checksum around a claim nobody can stand behind. {_NOTHING_WRITTEN}"
        )
    present = sorted(
        {
            str(record.get("kind"))
            for record in in_month
            if str(record.get("kind")) in _UNCLASSIFIED_NAMES
        }
    )
    # The VALUES, not just the keys. `sorted(mapping)` sorts keys, so a binding
    # naming all three with `""`, `None` or `17` satisfied a key-only check and
    # wrote that straight into the frozen payload -- which is the silence this
    # gate exists to refuse, wearing the shape of an answer. What a legal value
    # IS stays unnamed here on purpose: the vocabulary of treatments is the
    # protocol's to fix, and inventing one would be this report deciding the
    # scoring rule that PR-14 owns.
    named = sorted(
        name
        for name, value in binding.unclassified_treatment.items()
        if isinstance(value, str) and value.strip()
    )
    if present and named != list(_UNCLASSIFIED_NAMES):
        raise ReportRefused(
            f"the month holds {present} records, and section 9.4 classifies none of "
            "HALT, RESUME or RECOVERY. The protocol must say whether each of the three "
            "is scored -- all three by name, so that a silence about one cannot be read "
            f"as a treatment -- and this report will not decide it. {_NOTHING_WRITTEN}"
        )

    quantities = _monthly_quantities(in_month, binding.quantities)
    return {
        "schema": MONTHLY_REPORT_SCHEMA,
        "evidence_class": EVIDENCE_CLASS_PROSPECTIVE,
        "month": month,
        "campaign_ids": _sorted_distinct(
            [
                record.get("campaign_id")
                for record in in_month
                if record.get("kind") == RecordKind.STARTUP.value
            ]
        ),
        "protocol_hash": binding.protocol_hash,
        "contract_hash": binding.contract_hash,
        "prospective_from": binding.prospective_from,
        "campaign_start": binding.campaign_start,
        "source_identity": {key: str(value) for key, value in binding.source_identity.items()},
        "config_hashes": _sorted_distinct([record.get("config_hash") for record in in_month]),
        "excluded_minutes": list(binding.excluded_minutes),
        "unclassified_treatment": dict(binding.unclassified_treatment),
        "chain": chain,
        "quantities": quantities,
        "days": sorted({str(record.get("minute"))[:10] for record in in_month}),
    }


def _monthly_quantities(
    records: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> dict[str, int]:
    """Exactly the named quantities, computed by counting and nothing else."""
    processed_kinds = {
        RecordKind.DECISION.value,
        RecordKind.INCOMPLETE_STATE.value,
        RecordKind.SKIPPED_STALE.value,
    }
    available = {
        "records": len(records),
        "minutes_processed": sum(1 for r in records if r.get("kind") in processed_kinds),
        "decisions": sum(1 for r in records if r.get("kind") == RecordKind.DECISION.value),
        "incomplete_minutes": sum(
            1 for r in records if r.get("kind") == RecordKind.INCOMPLETE_STATE.value
        ),
    }
    return {name: available[name] for name in sorted(set(names))}


def render_monthly_status(report: Mapping[str, Any]) -> str:
    """``STATUS.md`` for a frozen month: what the directory is, and what it is not."""
    quantities = report["quantities"]
    lines = [
        f"# PVC-1 prospective campaign — {report['month']}",
        "",
        "| | |",
        "| --- | --- |",
        f"| evidence class | `{report['evidence_class']}` |",
        f"| protocol hash | `{report['protocol_hash']}` |",
        f"| recorder contract | `{report['contract_hash']}` |",
        f"| prospective from | `{report['prospective_from']}` |",
        f"| campaign start | `{report['campaign_start']}` |",
        f"| days | {len(report['days'])} |",
        f"| excluded minutes | {len(report['excluded_minutes'])} |",
        f"| chain | {report['chain']['summary']} |",
        "",
        "## Protocol-named quantities",
        "",
        "| quantity | value |",
        "| --- | ---: |",
    ]
    lines += [f"| {name} | {quantities[name]} |" for name in sorted(quantities)]
    lines += [
        "",
        "This directory is prospective campaign evidence recorded under the protocol "
        "named above. It answers no research question, adds no row to the research "
        "table in `artifacts/README.md`, and carries only the quantities that protocol "
        "names: nothing here was chosen after the minutes it covers were seen.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "CLASS_EVIDENCE",
    "CLASS_OPERATIONAL",
    "CLASS_UNCLASSIFIED",
    "DAILY_REPORT_CLASS",
    "DAILY_REPORT_SCHEMA",
    "EVIDENCE_CLASS_PROSPECTIVE",
    "FUNDING_SIGN_CONVENTION",
    "HALT_CAUSES",
    "HALT_CAUSE_OTHER",
    "INPUT_COVERAGE_NOTE",
    "LEGS",
    "MONTHLY_QUANTITIES",
    "MONTHLY_REPORT_SCHEMA",
    "RUNNER_WRITTEN_KINDS",
    "ProtocolBinding",
    "ReportError",
    "ReportRefused",
    "daily_report",
    "halt_cause",
    "kind_class",
    "monthly_report",
    "render_daily_markdown",
    "render_monthly_status",
]
