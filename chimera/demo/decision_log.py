"""The decision log: an append-only, hash-chained record of what the runner did.

This is the campaign's evidence. Everything the monthly report is computed from
and everything an S6 audit reads comes out of these files, so the properties
that matter are not "it logged something" but "nobody, including the process
that wrote it, can change what it says without that being visible".

**One file per UTC day, appended and never rewritten**, under
``<state_dir>/decision_log/YYYY-MM-DD.ndjson``. The day is taken from the
record's ``runner_now_ns`` — the :class:`chimera.demo.clock.RunnerClock` instant
at the moment of the write — and not from ``minute``. The two agree in normal
operation, and where they do not, the clock is the one that also produced
``seq``: keying the file by it is what makes "read the day files in name order"
reproduce the chain order exactly, including for the kinds that have no decision
minute at all.

**Hash-chained.** Each record carries ``prev_hash``, the ``record_hash`` of the
record before it, and the first record of a campaign carries
:data:`ZERO_PREV_HASH`. A record's ``record_hash`` is SHA-256 over its own
canonical bytes with ``record_hash`` removed. Editing any field of any record
therefore breaks that record's hash; removing or reordering records breaks the
link of the record that followed. Neither can be repaired without rewriting
every record after the change, and the runner's persisted ``last_record_hash``
(section 9.3) pins the end of the chain from outside the file.

**Canonical serialization, section 9.2, exactly.**
``json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)``.
The same function serializes the record live and in replay, so the byte
comparison section 10 performs is meaningful. Three consequences are enforced
rather than hoped for:

*Numbers.* A ``Decimal`` is a semantic quantity — a quantity, a price, a fee —
and it has no canonical JSON form, so this module refuses to serialize one and
:func:`decimal_str` renders it to a string at the scale of its constraint
instead. Routing it through a binary float would make ``0.1 + 0.2`` appear in
the evidence, and letting each caller pick its own ``str()`` would put ``1E-8``
in one record and ``0.00000001`` in another for the same fee. ``NaN`` and the
infinities are refused everywhere, on the way in and on the way back out: they
are not JSON, Python's encoder emits them anyway, and Python's decoder accepts
them, so a single unguarded write would produce a file that only Python can read
and that no two readers need agree about.

*Time.* Every timestamp is ISO-8601 UTC with an explicit ``+00:00``.
:func:`require_iso_utc` refuses a naive stamp, a ``Z`` suffix and any other
offset, because a record whose instant depends on the reader's locale is not
evidence.

*Bytes.* ``ensure_ascii=True`` means a record is pure ASCII whatever text a
detail string carries, the file is opened in binary append mode so no newline is
ever translated, and the record separator is one ``\\n``. A record written on
Windows and the same record written on Linux are the same bytes.

**Torn is not forged, and neither is a tail the state file disagrees with.**
:func:`verify_chain` separates three things it never treats as valid evidence: a
final line with no terminating newline, which is the writer dying mid-append and
is recoverable; a line that is complete and wrong, which is tampering or
corruption and is not; and a log whose end disagrees with the ``last_record_hash``
the runner persisted beside it, which section 9.3's write order produces on any
crash in that window and which says nothing about any record. What to *do* about
the first is the runner's decision (section 9.3's ``LOG_BEHIND_STATE`` and its
``RECOVERY`` record); this module provides the primitive that detects it and
:func:`recover_tail`, which removes the unfinished bytes after preserving them —
and which removes nothing else, because a complete record that does not verify is
evidence of a problem and not a mess to tidy away.

**This module opens no socket, makes no request and reads no clock.** Every
instant it writes was handed to it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping

from chimera.recorder.events import (
    NS_PER_MINUTE,
    RecorderEventError,
    iso_utc,
    require_canonical_ns,
    utc_day,
)

#: Names the meaning of one decision record. A record from a schema this build
#: does not know is refused rather than read with today's field meanings.
DECISION_RECORD_SCHEMA = "chimera.decision-record/1"

#: The prefix every hash in a record carries. Written out rather than implied so
#: that a bare digest and a prefixed one can never be compared as equal strings.
HASH_PREFIX = "sha256:"

#: ``prev_hash`` of the first record of a campaign: the prefix and 64 zeros. It
#: is a value rather than an absent key, because "there was no previous record"
#: and "the previous record's hash was not written" must not look the same.
ZERO_PREV_HASH = HASH_PREFIX + "0" * 64

#: The directory the day files live in, under the runner's state directory.
LOG_DIR_NAME = "decision_log"

#: The suffix of one day's file.
LOG_SUFFIX = ".ndjson"

#: Where :func:`recover_tail` preserves the bytes it removes. The same name the
#: recorder's :meth:`chimera.recorder.sink.RawSink.recover_tail` uses, because
#: an operator looking at a state directory after a crash should find one
#: convention and not two.
TRUNCATED_SUFFIX = ".truncated"

#: How far back :func:`torn_tail_bytes` looks for the last newline. A torn tail
#: is at most one unfinished record; a megabyte without a line break is damage.
TAIL_SCAN_BYTES = 1 << 20

#: Fields this module owns and stamps itself. A caller offering one of them is
#: refused: a payload that could set its own ``seq`` or ``prev_hash`` could
#: forge a chain through the very function that exists to make forging visible.
RESERVED_FIELDS: tuple[str, ...] = ("schema", "seq", "prev_hash", "record_hash")

#: Fields every record carries whatever its kind, and which the caller supplies.
#: ``minute`` is required to be *present* and may be ``null``: the operational
#: kinds have no decision minute, and an absent key would be indistinguishable
#: from one that was forgotten.
#:
#: Section 9.1 also puts ``config_hash`` and ``software`` at the top level of
#: every record, and they are deliberately **not** here. Both are stamped by the
#: runner, which is PR-10: ``software`` is a provenance block whose shape — the
#: revision, the source digest, the dirty flag, the interpreter and the library
#: versions — nothing in PR-09 can settle, and requiring half of it would freeze
#: the wrong half. ``config_hash`` is produced by :mod:`chimera.demo.config` in
#: this package but is supplied by whoever holds the parsed configuration, and
#: making it structurally mandatory here would oblige every operational record —
#: including the ``STARTUP`` record written before a configuration has been
#: read — to carry one. What this module guarantees about both is the *form*:
#: :func:`_validate_hashes` holds ``config_hash``, and any other ``*_hash`` at
#: any depth, to ``sha256:<hex>`` whenever it is present. Requiring their
#: presence is PR-10's to add once it owns the writer.
CORE_FIELDS: tuple[str, ...] = ("kind", "minute", "runner_now_ns")

#: What a record may hold. A whitelist rather than a blacklist, because the
#: failure this prevents is a type nobody thought about — a ``Path``, a
#: ``datetime``, a ``set`` — reaching ``json.dumps`` and either raising in the
#: middle of a write or being coerced by a ``default=`` hook nobody reviewed.
_SCALARS = (str, int, float, bool, type(None))


class DecisionLogError(ValueError):
    """A record cannot be written, or a chain cannot be trusted."""


class DecisionLogTailError(DecisionLogError):
    """The end of a log is not in a state that can be appended to.

    Carries the :class:`ChainVerification` that found it rather than only its
    formatted summary. The distinction the caller has to make — section 9.3's
    torn tail, which is a crash it recovers from and continues past, against a
    complete record that is wrong, which is not something continuing repairs —
    is :attr:`ChainVerification.is_torn` against
    :attr:`ChainVerification.is_forged`, and a caller that had to recover it by
    matching substrings of a message would be one message rewrite away from
    treating a forgery as a crash.
    """

    def __init__(self, message: str, verification: "ChainVerification | None" = None):
        super().__init__(message)
        self.verification = verification


class RecordKind(str, Enum):
    """The kinds of record section 9.1 admits. A bounded vocabulary.

    ``RECOVERY`` is here and the choice is deliberate. Section 9.3 has the runner
    append one after a ``LOG_BEHIND_STATE`` start, so the *name* has to be part
    of the log's schema — a log that refused the kind would make the recovery
    unrecordable. Deciding *when* one is written is runner recovery logic and is
    not in this module.
    """

    DECISION = "DECISION"
    FUNDING = "FUNDING"
    RECONCILIATION = "RECONCILIATION"
    OPERATOR = "OPERATOR"
    HALT = "HALT"
    RESUME = "RESUME"
    STARTUP = "STARTUP"
    SHUTDOWN = "SHUTDOWN"
    INCOMPLETE_STATE = "INCOMPLETE_STATE"
    SKIPPED_STALE = "SKIPPED_STALE"
    LIQUIDATION_TOUCH = "LIQUIDATION_TOUCH"
    RECOVERY = "RECOVERY"


#: The kinds section 9.4 names as evidence: what the monthly frozen report is
#: computed from and what an audit scores. Listed literally, because it is a
#: frozen scientific boundary and not a derivable property of a name.
EVIDENCE_KINDS: frozenset[RecordKind] = frozenset(
    {
        RecordKind.DECISION,
        RecordKind.FUNDING,
        RecordKind.RECONCILIATION,
        RecordKind.OPERATOR,
        RecordKind.LIQUIDATION_TOUCH,
    }
)

#: The kinds section 9.4 names as operational: kept and reported, never scored.
#: Listed literally for the same reason the evidence set is — it is half of a
#: scientific boundary and not a property derivable from a name.
OPERATIONAL_KINDS: frozenset[RecordKind] = frozenset(
    {
        RecordKind.STARTUP,
        RecordKind.SHUTDOWN,
        RecordKind.INCOMPLETE_STATE,
        RecordKind.SKIPPED_STALE,
    }
)

#: The kinds section 9.4 puts in **neither** list — ``HALT``, ``RESUME`` and
#: ``RECOVERY`` — which this module therefore declines to classify.
#:
#: Deriving the operational set as "everything that is not evidence" would put
#: them there silently, and that is a scientific decision rather than an
#: engineering one. Section 9.4 says the operational kinds are *not scored*, so
#: the derivation would rule — without the adopted plan saying so — that a
#: campaign's Aegis halts are excluded from the monthly frozen report and from
#: what S6 audits, and a campaign that halted repeatedly would then score
#: identically to one that never halted. Section 10 pulls the other way for one
#: of the three: its parity comparison ignores ``STARTUP``, ``SHUTDOWN`` and
#: ``RECOVERY`` and pointedly does not name ``HALT`` or ``RESUME`` — but that
#: settles how the parity *tool* aligns records, not what the report counts.
#: PR-12 (reports) and PR-14 (the PVC-1 protocol) own the answer; PR-09 keeps
#: the question visible rather than closing it by default. The three sets still
#: partition :class:`RecordKind`, so a kind added later cannot be absent from
#: all of them.
UNCLASSIFIED_KINDS: frozenset[RecordKind] = (
    frozenset(RecordKind) - EVIDENCE_KINDS - OPERATIONAL_KINDS
)


def is_evidence(kind: RecordKind | str) -> bool:
    """Whether a kind is section 9.4 evidence rather than operational record."""
    return require_kind(kind) in EVIDENCE_KINDS


def require_kind(kind: RecordKind | str) -> RecordKind:
    """The record kind, or an explanation of why that string is not one."""
    if isinstance(kind, RecordKind):
        return kind
    try:
        return RecordKind(kind)
    except ValueError:
        raise DecisionLogError(
            f"kind {kind!r} is not one of {sorted(k.value for k in RecordKind)}. The "
            "vocabulary is closed on purpose: a report that groups by kind cannot "
            "group a kind it has never heard of"
        ) from None


# --- canonical form ---------------------------------------------------------
def _refuse_constant(token: str) -> Any:
    """Reader-side refusal of ``NaN``/``Infinity``. Never returns."""
    raise ValueError(
        f"{token} is not JSON and cannot appear in a decision record; Python's decoder "
        "accepts it and no other reader does"
    )


def _non_finite_path(node: Any, path: str = "record") -> str | None:
    """Where a decoded record holds a float that is not finite, if anywhere.

    ``parse_constant`` is the only hook :mod:`json` offers for the literal
    ``NaN`` and ``Infinity`` tokens, and it is not enough: an ordinary *numeric*
    token can overflow to an infinity without ever being one of them, because
    ``float("1e999")`` is ``inf`` and raises nothing. A line carrying one parses,
    passes every field check, and then makes :func:`canonical_json` raise in the
    middle of :func:`verify_chain` — so the verifier, whose whole contract is to
    report what it found on exactly this kind of corruption-controlled input,
    would instead propagate an exception and audit nothing else in the campaign.
    Found here, it is classified as a malformed record like the literal tokens.
    """
    if isinstance(node, float):
        if node != node or node in (float("inf"), float("-inf")):
            return path
        return None
    if isinstance(node, Mapping):
        for key, value in node.items():
            found = _non_finite_path(value, f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(node, list):
        for index, value in enumerate(node):
            found = _non_finite_path(value, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def canonical_json(record: Mapping[str, Any]) -> str:
    """Section 9.2's serialization, and the only one this package uses.

    ``allow_nan=False`` is the one addition to the literal call in the plan, and
    it changes no byte of any record that can legally be written: it only turns
    a record carrying a non-finite float from a file no conforming reader can
    parse into an explicit failure at the point of the write.
    """
    try:
        return json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DecisionLogError(
            f"record cannot be serialised canonically: {exc}. A decision record holds "
            "JSON scalars, lists and objects with string keys, and nothing else"
        ) from exc


def canonical_bytes(record: Mapping[str, Any]) -> bytes:
    """The exact bytes one line of the log holds, without the newline.

    ASCII, because ``ensure_ascii=True`` guarantees it; encoding it as ASCII
    rather than as UTF-8 makes that guarantee an assertion rather than a claim.
    """
    return canonical_json(record).encode("ascii")


def canonical_line(record: Mapping[str, Any]) -> bytes:
    """One complete line, newline included.

    Always ``\\n``, never ``\\r\\n``. The file is opened in binary append mode so
    that a record written on Windows and one written on Linux are the same bytes
    and the replay parity comparison in section 10 means what it says.
    """
    return canonical_bytes(record) + b"\n"


def compute_record_hash(record: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical bytes of a record *without* ``record_hash``.

    Refuses a mapping that still carries the field rather than dropping it
    quietly: hashing over the wrong material is the one mistake that makes every
    subsequent verification pass while proving nothing.
    """
    if "record_hash" in record:
        raise DecisionLogError(
            "compute_record_hash is taken over the record without record_hash; remove "
            "the field before hashing rather than hashing a record that contains its "
            "own digest"
        )
    return HASH_PREFIX + hashlib.sha256(canonical_bytes(record)).hexdigest()


def is_hash(value: Any) -> bool:
    """Whether a value is a ``sha256:<64 lowercase hex>`` string."""
    if not isinstance(value, str) or not value.startswith(HASH_PREFIX):
        return False
    digest = value[len(HASH_PREFIX) :]
    return len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def require_hash(value: Any, *, field_name: str) -> str:
    """A ``sha256:<hex>`` string, or an explanation of why it is not one."""
    if not is_hash(value):
        raise DecisionLogError(
            f"{field_name} must be {HASH_PREFIX}<64 lowercase hex digits>, got {value!r}. "
            "An unprefixed digest and a prefixed one compare unequal, so the prefix is "
            "part of the value and not decoration"
        )
    return value


def require_iso_utc(value: Any, *, field_name: str) -> str:
    """An ISO-8601 UTC timestamp written with an explicit ``+00:00`` offset.

    A ``Z`` suffix, a naive stamp and any non-zero offset are all refused. They
    are refused even though two of them denote the same instant, because the
    records are compared byte for byte: two spellings of one instant would make
    a live record and its replay differ for a reason that has nothing to do with
    the decision.
    """
    if not isinstance(value, str):
        raise DecisionLogError(
            f"{field_name} must be an ISO-8601 UTC string, got {type(value).__name__}"
        )
    if not value.endswith("+00:00"):
        raise DecisionLogError(
            f"{field_name} is {value!r}; every timestamp in a decision record ends in "
            "'+00:00'. 'Z', a naive stamp and a local offset are all refused: the "
            "records are compared as bytes, so one instant must have one spelling"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DecisionLogError(f"{field_name} {value!r} is not ISO-8601: {exc}") from exc
    if parsed.isoformat() != value:
        raise DecisionLogError(
            f"{field_name} {value!r} is not the canonical ISO-8601 spelling of "
            f"{parsed.isoformat()!r}"
        )
    return value


def iso_minute(canonical_ns: int) -> str:
    """A decision minute as the canonical timestamp a record carries.

    Renders through :func:`chimera.recorder.events.iso_utc`, so the runner's
    minute and the recorder's manifests spell an instant the same way and a
    reader never has to ask which of two formats a file is in.

    An instant that is not on a minute boundary is **refused, not floored**.
    Section 9.1's ``minute`` is the kline open time, section 10 puts it in the
    must-match-byte-for-byte column and aligns operational records by it, so a
    value like ``00:02:00.123456+00:00`` is not a minute and a field that
    silently rounded one would hide a caller's mistake exactly where the report
    groups on it. This is :func:`decimal_str`'s rule applied to time: a record
    states the value that was used, and quietly adjusting it here would conceal
    the disagreement rather than report it. The recorder already owns the
    flooring primitive — :func:`chimera.recorder.events.minute_open_ms` — for a
    caller that has an arbitrary instant and wants the minute it falls in.
    """
    try:
        instant = require_canonical_ns(canonical_ns, field="canonical_ns")
    except RecorderEventError as exc:
        raise DecisionLogError(str(exc)) from exc
    if instant % NS_PER_MINUTE:
        raise DecisionLogError(
            f"{instant} is not a minute boundary, and a decision minute is the kline "
            "open time (section 9.1). Rounding it here would put a value that is not a "
            "minute in the field the parity comparison and the reports group on; use "
            "chimera.recorder.events.minute_open_ms to choose the minute deliberately"
        )
    try:
        return iso_utc(instant)
    except RecorderEventError as exc:  # pragma: no cover - bounds already checked
        raise DecisionLogError(str(exc)) from exc


def require_iso_minute(value: Any, *, field_name: str = "minute") -> str:
    """A canonical UTC timestamp that is also on a minute boundary.

    The record-level half of :func:`iso_minute`'s rule. It is checked on the
    string rather than only at the point one is produced, because a runner is
    free to build the field any way it likes and the invariant belongs to the
    record: two records carrying the same decision minute have to carry the same
    bytes, and ``00:01:00+00:00`` and ``00:01:00.5+00:00`` do not.
    """
    stamp = require_iso_utc(value, field_name=field_name)
    parsed = datetime.fromisoformat(stamp)
    if parsed.second or parsed.microsecond:
        raise DecisionLogError(
            f"{field_name} is {stamp!r}, which is an instant inside a minute rather than "
            "a minute. Section 9.1's minute is the kline open time, and section 10 "
            "compares it byte for byte and aligns records by it"
        )
    return stamp


def decimal_str(value: Decimal, *, scale: int) -> str:
    """A semantic ``Decimal`` as the string a record holds, at a fixed scale.

    ``scale`` is the number of digits after the point that the value's own
    constraint fixes — a quantity step, a price tick, the ``1e-8`` fee
    quantum — and it is required rather than inferred, because ``Decimal`` keeps
    the scale it was constructed with and ``Decimal("0.5")`` and
    ``Decimal("0.500")`` would otherwise write two different bytes for one
    quantity.

    Three refusals, each preventing a specific way the evidence could lie:

    * a non-finite value is refused outright — ``NaN`` is not a quantity;
    * a value that does not *fit* the scale is refused rather than rounded,
      because silently dropping a digit of a fill quantity turns a mismatch the
      reconciliation would have caught into a record that agrees with nothing;
    * negative zero is rendered as zero, so a ledger effect of ``-0.00`` and one
      of ``0.00`` are the same bytes.

    ``format`` rather than ``str``: ``str(Decimal("0.00000001"))`` is ``1E-8``,
    and scientific notation in a fee field is a number two readers can disagree
    about.
    """
    if not isinstance(value, Decimal):
        raise DecisionLogError(
            f"decimal_str renders a Decimal, got {type(value).__name__}. A float is not "
            "promoted to one here: the promotion is where the binary rounding enters"
        )
    if not isinstance(scale, int) or isinstance(scale, bool) or scale < 0:
        raise DecisionLogError(f"scale must be a non-negative integer, got {scale!r}")
    if not value.is_finite():
        raise DecisionLogError(
            f"{value!r} is not a finite quantity and cannot appear in a decision record"
        )
    exponent = Decimal(1).scaleb(-scale)
    try:
        quantized = value.quantize(exponent)
    except InvalidOperation as exc:
        raise DecisionLogError(
            f"{value!r} cannot be represented at scale {scale}: {exc}"
        ) from exc
    if quantized != value:
        raise DecisionLogError(
            f"{value!r} does not fit scale {scale}; rendering it would round it to "
            f"{quantized!r}. A record states the quantity that was used, and rounding "
            "one here would hide the disagreement rather than report it"
        )
    if quantized.is_zero():
        quantized = abs(quantized)
    return format(quantized, f".{scale}f")


# --- payload validation -----------------------------------------------------
def _validate_value(value: Any, path: str) -> None:
    """Refuse anything that cannot be written canonically, naming where it is."""
    if isinstance(value, Decimal):
        raise DecisionLogError(
            f"{path} is a Decimal. Render it with decimal_str(value, scale=...) at the "
            "scale its constraint fixes; a Decimal has no canonical JSON form and "
            "serialising one through a float is how binary rounding enters the evidence"
        )
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise DecisionLogError(
                f"{path} is {value!r}. NaN and the infinities are not JSON, and a record "
                "carrying one is a file no conforming reader can parse"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DecisionLogError(
                    f"{path} has a {type(key).__name__} key {key!r}; a record's object "
                    "keys are strings, because that is what sorting them deterministically "
                    "requires"
                )
            _validate_value(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_value(item, f"{path}[{index}]")
        return
    raise DecisionLogError(
        f"{path} is a {type(value).__name__}, which is not a JSON value. A decision "
        f"record holds {', '.join(t.__name__ for t in _SCALARS)}, lists and objects"
    )


def _validate_hashes(value: Any, path: str) -> None:
    """Hold every ``*_hash`` field, at any depth, to the one hash form.

    The rule is on the *name* rather than on a list of known fields because the
    nested blocks — ``inputs``, ``rule``, ``risk``, ``execution`` — are the
    runner's to shape, and a hash written bare in one of them would compare
    unequal to the same hash written prefixed elsewhere. Digests that are
    deliberately unprefixed, the recorder's minute digests among them, are named
    ``*_digest`` and are not caught by this.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            where = f"{path}.{key}"
            if key.endswith("_hash"):
                require_hash(item, field_name=where)
            _validate_hashes(item, where)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_hashes(item, f"{path}[{index}]")


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Everything checked before a byte is written, and why each check is here.

    The write is the irreversible step: the file is append-only, so a record that
    should not have been written cannot be taken back, only annotated. So the
    whole payload is walked first.
    """
    if not isinstance(payload, Mapping):
        raise DecisionLogError(
            f"a decision record is a JSON object, got {type(payload).__name__}"
        )
    offered = [name for name in RESERVED_FIELDS if name in payload]
    if offered:
        raise DecisionLogError(
            f"the log stamps {sorted(RESERVED_FIELDS)} itself and was offered "
            f"{sorted(offered)}. A caller that could set its own seq or prev_hash could "
            "forge a chain through the function whose whole purpose is to make forging "
            "visible"
        )
    missing = [name for name in CORE_FIELDS if name not in payload]
    if missing:
        raise DecisionLogError(
            f"missing required field(s) {sorted(missing)}. Every record carries "
            f"{sorted(CORE_FIELDS)} whatever its kind; minute may be null for a kind "
            "that has no decision minute, but the key is never absent"
        )

    record = dict(payload)
    record["kind"] = require_kind(record["kind"]).value
    if record["minute"] is not None:
        require_iso_minute(record["minute"], field_name="minute")
    try:
        require_canonical_ns(record["runner_now_ns"], field="runner_now_ns")
    except RecorderEventError as exc:
        raise DecisionLogError(str(exc)) from exc

    _validate_value(record, "record")
    _validate_hashes(record, "record")
    return record


# --- the writer -------------------------------------------------------------
@dataclass(frozen=True)
class AppendedRecord:
    """What one committed append was, for the caller that has to persist it.

    ``record_hash`` is what section 9.3 has the runner write into its state file
    as ``last_record_hash``, which is what pins the end of the chain from outside
    the log.
    """

    seq: int
    kind: RecordKind
    day: str
    path: Path
    prev_hash: str
    record_hash: str
    record: Mapping[str, Any]
    line: bytes


class DecisionLog:
    """The append-only writer. One per campaign, one open day file at a time.

    Not thread-safe, deliberately: the chain is a sequence, and two threads
    appending to it would produce two records claiming the same ``prev_hash``.
    Section 8.2 gives the runner one tick-loop thread for exactly this kind of
    reason.

    **The constructor takes the log directory, and :meth:`open` takes the state
    directory.** They are one level apart — ``<state_dir>/decision_log`` is the
    log directory — and the parameter is named for which one it is in both
    places, because getting it wrong is silent: a log constructed on the state
    directory writes its day files one level too high, every method here agrees
    with itself about the wrong place, and a later ``verify_log(state_dir /
    "decision_log")`` reports an empty log that is intact. :meth:`open` is the
    entry point to prefer; the constructor exists for a caller that has already
    resolved the directory and is carrying a chain head with it.
    """

    def __init__(
        self,
        log_dir: str | Path,
        *,
        seq: int = 0,
        prev_hash: str = ZERO_PREV_HASH,
        last_runner_now_ns: int | None = None,
    ) -> None:
        """Open a log whose chain continues from ``seq`` and ``prev_hash``.

        ``log_dir`` is the directory the ``YYYY-MM-DD.ndjson`` files live in —
        that is ``<state_dir>/decision_log``, **not** the state directory itself;
        see the class docstring.

        The defaults are the start of a campaign: nothing written, so the next
        record is ``seq`` 1 and chains to :data:`ZERO_PREV_HASH`. Resuming an
        existing campaign goes through :meth:`open`, which reads the tail rather
        than trusting a caller's memory of it.
        """
        self.log_dir = Path(log_dir)
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise DecisionLogError(f"seq must be a non-negative integer, got {seq!r}")
        self._seq = seq
        self._prev_hash = require_hash(prev_hash, field_name="prev_hash")
        self._last_runner_now_ns = last_runner_now_ns
        self._day: str | None = None
        self._handle: BinaryIO | None = None
        #: Set by a failed commit, and never cleared. See :meth:`append`.
        self._failure: str | None = None

    # --- construction -----------------------------------------------------
    @classmethod
    def open(cls, state_dir: str | Path) -> "DecisionLog":
        """Continue the campaign's log under ``<state_dir>/decision_log``.

        ``state_dir`` is the runner's state directory; the day files go under
        ``<state_dir>/decision_log``, which this resolves. The constructor takes
        that resolved directory instead — see the class docstring.

        The tail is read rather than assumed, and it is *verified* before it is
        adopted: appending onto a chain that does not verify would extend a
        forgery or paper over a torn tail, and either would make every record
        after it worthless as evidence. A log that does not verify raises
        :class:`DecisionLogTailError`, which carries the
        :class:`ChainVerification` itself so the caller can branch on
        :attr:`~ChainVerification.is_torn` against
        :attr:`~ChainVerification.is_forged` rather than on the wording of a
        message. What to do about it — section 9.3's ``LOG_BEHIND_STATE``, its
        ``RECOVERY`` record, and :func:`recover_tail` for the torn case — is the
        runner's decision and not this module's.

        Only the most recent non-empty day file is verified, which is exactly
        what section 9.3 asks of ``SELF_CHECK``: the last line parses, and the
        chain is intact for the current day. When that file is also the
        campaign's *first*, its opening record is held to :data:`ZERO_PREV_HASH`
        as well, so a campaign cannot be resumed on a log whose head has been
        removed. "First" here means the first file that holds a record, not the
        first file that exists: an empty day file is what a crash between opening
        a day and committing its first record leaves behind, it is also the
        cheapest thing for an editor to leave in place of a day it emptied, and
        counting one would put the anchor on a file that carries nothing and
        leave the campaign's real opening record unchecked. :func:`verify_log`
        walks every day and checks the links between them.
        """
        root = Path(state_dir) / LOG_DIR_NAME
        root.mkdir(parents=True, exist_ok=True)
        files = day_files(root)
        holding = [path for path in files if path.stat().st_size > 0]
        for path in reversed(holding):
            first = path == holding[0]
            verification = verify_chain(
                path,
                expected_prev_hash=ZERO_PREV_HASH if first else None,
                start_seq=0 if first else None,
            )
            if not verification.ok:
                raise DecisionLogTailError(
                    f"the decision log tail at {path} does not verify "
                    f"({verification.summary()}). Appending to it would extend a chain "
                    "nothing can stand behind",
                    verification,
                )
            return cls(
                root,
                seq=verification.last_seq or 0,
                prev_hash=verification.last_record_hash or ZERO_PREV_HASH,
                last_runner_now_ns=verification.last_runner_now_ns,
            )
        return cls(root)

    # --- state ------------------------------------------------------------
    @property
    def next_seq(self) -> int:
        """The ``seq`` the next appended record will carry."""
        return self._seq + 1

    @property
    def last_record_hash(self) -> str:
        """The chain head: what the next record's ``prev_hash`` will be."""
        return self._prev_hash

    @property
    def last_runner_now_ns(self) -> int | None:
        """The clock of the last committed record, or ``None`` if there is none."""
        return self._last_runner_now_ns

    def path_for_day(self, day: str) -> Path:
        """The file one UTC day's records live in."""
        return self.log_dir / f"{day}{LOG_SUFFIX}"

    # --- writing ----------------------------------------------------------
    def append(self, payload: Mapping[str, Any]) -> AppendedRecord:
        """Commit one record: validate, chain, hash, append, flush, ``fsync``.

        The order is the guarantee. Everything that can be refused is refused
        before the file is touched, because an append-only file cannot take a
        record back. The ``fsync`` is what makes the record survive a power loss
        rather than merely a process death, and it is per record rather than
        batched because a batched chain would lose its own end.

        A commit that fails closes the log for writing. After a failed write it
        is unknown whether the bytes reached the file, so the chain head this
        object holds may already be one record behind what is on disk, and the
        next append would then write a second record claiming the same ``seq``.
        The recoverable state is the one on disk: stop, and let a restart read
        the tail back through :meth:`open`.
        """
        if self._failure is not None:
            raise DecisionLogError(
                f"this log stopped at {self._failure} and accepts no further records. "
                "What is on disk is the chain; reopen the log so the tail is read and "
                "verified rather than continued from memory"
            )
        record = validate_payload(payload)
        now_ns = int(record["runner_now_ns"])
        if self._last_runner_now_ns is not None and now_ns < self._last_runner_now_ns:
            raise DecisionLogError(
                f"runner_now_ns {now_ns} precedes the last committed record's "
                f"{self._last_runner_now_ns}. The runner clock never moves backwards "
                "(section 2.4), so a record that claims it did is a defect in the caller "
                "and not a record to preserve"
            )
        day = utc_day(now_ns)

        record["schema"] = DECISION_RECORD_SCHEMA
        record["seq"] = self._seq + 1
        record["prev_hash"] = self._prev_hash
        record_hash = compute_record_hash(record)
        record["record_hash"] = record_hash
        line = canonical_line(record)

        handle = self._ensure_day(day)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        except OSError as exc:
            self._failure = f"record {record['seq']} ({exc})"
            self._handle, self._day = None, None
            handle.close()
            raise DecisionLogError(
                f"could not commit decision record {record['seq']} to "
                f"{self.path_for_day(day)}: {exc}. The campaign's evidence is the log, so "
                "a write that failed is a halt and never a warning"
            ) from exc

        appended = AppendedRecord(
            seq=int(record["seq"]),
            kind=RecordKind(record["kind"]),
            day=day,
            path=self.path_for_day(day),
            prev_hash=self._prev_hash,
            record_hash=record_hash,
            record=record,
            line=line,
        )
        self._seq = appended.seq
        self._prev_hash = record_hash
        self._last_runner_now_ns = now_ns
        return appended

    def _ensure_day(self, day: str) -> BinaryIO:
        """Open ``day``'s file for binary append, rotating if another is open.

        Refuses a day earlier than one already on disk. Within one process the
        non-decreasing ``runner_now_ns`` rule already implies it, but a log
        resumed through the constructor rather than through :meth:`open` carries
        no memory of the last instant, and appending into an older file after a
        newer one exists would break the property :func:`day_files` relies on:
        that reading the day files in name order reproduces the chain order.

        Refuses a file whose last byte is not a newline, which is the one case
        where continuing does active damage. A crash mid-append leaves a torn
        fragment with no newline; appending after it glues the next record onto
        that fragment, producing one line that is neither record. The file then
        *ends* in a newline again, so it no longer looks torn — a recoverable
        crash has been turned into an unparseable line that reads as a forgery,
        and every record written after it stops being counted, because
        :func:`verify_chain` cannot chain past a line it cannot read.
        :meth:`open` already refuses such a tail; this closes the constructor
        path, and :func:`recover_tail` is how a caller gets past it deliberately.
        """
        if self._day == day and self._handle is not None:
            return self._handle
        self.close()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        existing = day_files(self.log_dir)
        if existing:
            newest = existing[-1].name[: -len(LOG_SUFFIX)]
            if day < newest:
                raise DecisionLogError(
                    f"a record for {day} would be appended while {newest} already exists. "
                    "Day files read in name order have to reproduce the chain order, and a "
                    "later seq in an earlier file breaks that"
                )
        path = self.path_for_day(day)
        torn = torn_tail_bytes(path)
        if torn:
            raise DecisionLogTailError(
                f"{path} ends in {torn} byte(s) with no terminating newline, so the "
                "writer did not finish its last record. Appending after them would glue "
                "the next record onto the fragment and make one unreadable line of the "
                "two; recover_tail() is what removes them, and it preserves them first"
            )
        try:
            self._handle = open(path, "ab")
        except OSError as exc:
            raise DecisionLogError(
                f"could not open the decision log at {path}: {exc}"
            ) from exc
        self._day = day
        return self._handle

    def close(self) -> None:
        """Flush, ``fsync`` and close the open day. Reopened by the next append."""
        handle, self._handle, self._day = self._handle, None, None
        if handle is None:
            return
        try:
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()

    def __enter__(self) -> "DecisionLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def day_files(root: str | Path) -> list[Path]:
    """The day files of a log, in chain order.

    Name order is chain order because the names are ``YYYY-MM-DD``, which sorts
    lexicographically exactly as it sorts chronologically, and because
    :meth:`DecisionLog.append` refuses to write a record into a day earlier than
    the one it has open.
    """
    directory = Path(root)
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.glob(f"*{LOG_SUFFIX}")
        if path.is_file() and _is_day_name(path.name[: -len(LOG_SUFFIX)])
    )


def _is_day_name(name: str) -> bool:
    """Whether a stem is a ``YYYY-MM-DD`` UTC day."""
    try:
        datetime.strptime(name, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def torn_tail_bytes(path: str | Path) -> int:
    """Bytes at the end of a file that have no terminating newline.

    Zero for a file that does not exist, is empty, or ends properly. The scan is
    bounded: a complete record is a few kilobytes and a torn tail is at most one
    record, so a file whose last :data:`TAIL_SCAN_BYTES` hold no newline at all
    is damage rather than a tail a crash cut short, and it is reported as such
    instead of being silently treated as one enormous fragment to discard.
    """
    location = Path(path)
    try:
        size = location.stat().st_size
    except OSError:
        return 0
    if size == 0:
        return 0
    window = min(size, TAIL_SCAN_BYTES)
    try:
        with open(location, "rb") as handle:
            handle.seek(size - window)
            chunk = handle.read(window)
    except OSError as exc:
        raise DecisionLogError(f"could not read the tail of {location}: {exc}") from exc
    cut = chunk.rfind(b"\n")
    if cut == -1:
        if window < size:
            raise DecisionLogError(
                f"the last {TAIL_SCAN_BYTES} bytes of {location} hold no line break, so "
                "the end of the file cannot be bounded. That is damage, not a tail a "
                "crash cut short, and it is left exactly as it is"
            )
        return size
    return window - cut - 1


# --- verification -----------------------------------------------------------
class ChainFault(str, Enum):
    """What a verification found. A bounded label, so an alert can group on it.

    Two members are kept apart from the rest on purpose, because neither is a
    claim that any record is wrong.

    :attr:`TORN_TAIL` is the signature of a crash between the write and the
    ``fsync`` — the writer never finished the line — and it is recoverable: the
    runner names the affected minute, excludes it from the evidence and
    continues (section 9.3). :attr:`TAIL_HASH_MISMATCH` says the log and the
    ``last_record_hash`` the runner persisted beside it disagree about where the
    chain ends, which section 9.3's write order (state, then record, then hash)
    produces on any crash in that window; every record can still be complete,
    canonical and correctly linked.

    Every other member says a *complete* record is wrong, which is corruption or
    tampering, and no amount of continuing repairs it. That split is what
    :attr:`ChainVerification.is_torn`, :attr:`ChainVerification.is_forged` and
    :attr:`ChainVerification.is_tail_disagreement` report separately. All of
    them, including the two recoverable ones, make :attr:`ChainVerification.ok`
    false: nothing here is a warning.
    """

    TORN_TAIL = "TORN_TAIL"
    MALFORMED_RECORD = "MALFORMED_RECORD"
    NON_CANONICAL_BYTES = "NON_CANONICAL_BYTES"
    RECORD_HASH_MISMATCH = "RECORD_HASH_MISMATCH"
    PREV_HASH_MISMATCH = "PREV_HASH_MISMATCH"
    DUPLICATE_SEQ = "DUPLICATE_SEQ"
    SEQ_NOT_MONOTONE = "SEQ_NOT_MONOTONE"
    TAIL_HASH_MISMATCH = "TAIL_HASH_MISMATCH"


#: Faults that are not a claim that a complete record is wrong, and so are not
#: forgery: a tail the writer never finished, and a disagreement between the log
#: and the hash the runner persisted beside it.
_NOT_FORGERY = frozenset({ChainFault.TORN_TAIL, ChainFault.TAIL_HASH_MISMATCH})


@dataclass(frozen=True)
class ChainDefect:
    """One finding, located precisely enough to look at."""

    fault: ChainFault
    path: Path
    line: int
    detail: str

    def __str__(self) -> str:
        where = f"{self.path.name}:{self.line}" if self.line else self.path.name
        return f"{self.fault.value} at {where}: {self.detail}"


@dataclass(frozen=True)
class ChainVerification:
    """The verdict on a log or on one of its day files.

    ``ok`` is the only thing that makes the records evidence. Everything else is
    here so a caller can tell a torn tail from a forged chain and act
    differently, which is the whole reason the two are distinguished.
    """

    files: tuple[Path, ...]
    records: int = 0
    first_seq: int | None = None
    last_seq: int | None = None
    last_record_hash: str | None = None
    last_runner_now_ns: int | None = None
    torn_tail_bytes: int = 0
    defects: tuple[ChainDefect, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """No defect of any kind. A torn tail is a defect."""
        return not self.defects

    @property
    def is_torn(self) -> bool:
        """The tail is incomplete: a crash mid-append, and nothing else."""
        return any(d.fault is ChainFault.TORN_TAIL for d in self.defects)

    @property
    def is_forged(self) -> bool:
        """A complete record is wrong: corruption or tampering, not a crash."""
        return any(d.fault not in _NOT_FORGERY for d in self.defects)

    @property
    def is_tail_disagreement(self) -> bool:
        """The records are sound and the runner's persisted head disagrees.

        Section 9.3 writes the state files, then the log record, then
        ``last_record_hash``, so a crash between the second step and the third
        leaves the log one record *ahead* of the state by construction. Every
        record in the file is then complete, canonical and correctly chained, and
        the only thing wrong is which of the two sides is behind — which is the
        runner's question and not a claim about any record. Grouping it under
        :attr:`is_forged` would have a recovery routine that branches on forgery
        declare an ordinary power cut to be tampering and halt the campaign.
        """
        return any(d.fault is ChainFault.TAIL_HASH_MISMATCH for d in self.defects)

    @property
    def faults(self) -> tuple[ChainFault, ...]:
        """The distinct faults found, in the order they were first seen."""
        seen: list[ChainFault] = []
        for defect in self.defects:
            if defect.fault not in seen:
                seen.append(defect.fault)
        return tuple(seen)

    def summary(self) -> str:
        """One line naming every defect, for a log message or an exception."""
        if self.ok:
            return f"{self.records} record(s), chain intact"
        return "; ".join(str(defect) for defect in self.defects)


def _parse_record(raw: bytes) -> tuple[dict[str, Any] | None, str]:
    """One line to a validated record, or ``None`` and why it is not one."""
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        return None, f"the line is not ASCII ({exc}); canonical records are ensure_ascii"
    try:
        record = json.loads(text, parse_constant=_refuse_constant)
    except ValueError as exc:
        return None, f"the line is not JSON: {exc}"
    if not isinstance(record, dict):
        return None, f"a record is a JSON object, got {type(record).__name__}"
    non_finite = _non_finite_path(record)
    if non_finite is not None:
        return None, (
            f"{non_finite} is a non-finite number. A numeric token can overflow to an "
            "infinity without being the literal NaN or Infinity, and a record holding "
            "one is a file no conforming reader can parse"
        )
    if record.get("schema") != DECISION_RECORD_SCHEMA:
        return None, (
            f"schema is {record.get('schema')!r}, not {DECISION_RECORD_SCHEMA!r}; a record "
            "from another schema is refused rather than read with today's field meanings"
        )
    for name in ("seq", "prev_hash", "record_hash", "kind", "runner_now_ns"):
        if name not in record:
            return None, f"missing field {name!r}"
    if isinstance(record["seq"], bool) or not isinstance(record["seq"], int):
        return None, f"seq must be an integer, got {record['seq']!r}"
    if isinstance(record["runner_now_ns"], bool) or not isinstance(
        record["runner_now_ns"], int
    ):
        return None, f"runner_now_ns must be an integer, got {record['runner_now_ns']!r}"
    for name in ("prev_hash", "record_hash"):
        if not is_hash(record[name]):
            return None, f"{name} is {record[name]!r}, not {HASH_PREFIX}<64 hex>"
    try:
        require_kind(record["kind"])
    except DecisionLogError as exc:
        return None, str(exc)
    return record, ""


def verify_chain(
    path: str | Path,
    *,
    expected_prev_hash: str | None = None,
    expected_last_hash: str | None = None,
    start_seq: int | None = None,
) -> ChainVerification:
    """Verify one day file, reporting every defect it can locate.

    ``expected_prev_hash`` is what this file's first record must chain to: the
    previous day's last ``record_hash``, or :data:`ZERO_PREV_HASH` when the file
    is the campaign's first. Passing ``None`` leaves that record unchecked, which
    is right only for a caller verifying a file in isolation with no idea what
    precedes it — :func:`verify_log`, which does know, always supplies it.
    ``expected_last_hash`` is section 9.3's persisted ``last_record_hash``, and a
    disagreement is reported as :attr:`ChainFault.TAIL_HASH_MISMATCH` rather than
    resolved: which of the two is behind is the runner's question.

    Parsing failures stop the scan, because a record that cannot be read cannot
    be chained from and every later report would be an artefact of that. Hash,
    link and sequence faults do not stop it: they are all locatable and a reader
    is better served by the whole list.
    """
    location = Path(path)
    try:
        data = location.read_bytes()
    except OSError as exc:
        raise DecisionLogError(
            f"could not read the decision log at {location}: {exc}"
        ) from exc

    defects: list[ChainDefect] = []
    torn = b""
    if data and not data.endswith(b"\n"):
        # No terminating newline means the writer did not finish. The bytes may
        # happen to parse; that is not evidence the record was complete, and
        # `RawSink.recover_tail` takes the same view of a raw file's tail.
        body, newline, torn = data.rpartition(b"\n")
        data = body + newline

    lines = data.split(b"\n")[:-1] if data else []
    prev_hash = expected_prev_hash
    prev_seq = start_seq
    records = 0
    first_seq: int | None = None
    last_seq: int | None = None
    last_hash: str | None = None
    last_now_ns: int | None = None

    for index, raw in enumerate(lines, start=1):
        record, why = _parse_record(raw)
        if record is None:
            defects.append(ChainDefect(ChainFault.MALFORMED_RECORD, location, index, why))
            break

        hashed = {k: v for k, v in record.items() if k != "record_hash"}
        computed = compute_record_hash(hashed)
        declared = str(record["record_hash"])
        if computed != declared:
            defects.append(
                ChainDefect(
                    ChainFault.RECORD_HASH_MISMATCH,
                    location,
                    index,
                    f"record {record['seq']} declares {declared} and hashes to {computed}",
                )
            )
        elif canonical_line(record) != raw + b"\n":
            # Only meaningful once the content itself verifies: a record whose
            # hash is already wrong tells us nothing extra by also being
            # differently formatted.
            defects.append(
                ChainDefect(
                    ChainFault.NON_CANONICAL_BYTES,
                    location,
                    index,
                    f"record {record['seq']} is not written in section 9.2's canonical form",
                )
            )

        if prev_hash is not None and record["prev_hash"] != prev_hash:
            defects.append(
                ChainDefect(
                    ChainFault.PREV_HASH_MISMATCH,
                    location,
                    index,
                    f"record {record['seq']} chains to {record['prev_hash']}, but the "
                    f"record before it hashes to {prev_hash}",
                )
            )
        if prev_seq is not None:
            if record["seq"] == prev_seq:
                defects.append(
                    ChainDefect(
                        ChainFault.DUPLICATE_SEQ,
                        location,
                        index,
                        f"seq {record['seq']} appears twice",
                    )
                )
            elif record["seq"] < prev_seq:
                defects.append(
                    ChainDefect(
                        ChainFault.SEQ_NOT_MONOTONE,
                        location,
                        index,
                        f"seq {record['seq']} follows {prev_seq}",
                    )
                )

        records += 1
        if first_seq is None:
            first_seq = int(record["seq"])
        prev_seq = last_seq = int(record["seq"])
        prev_hash = last_hash = declared
        last_now_ns = int(record["runner_now_ns"])

    if torn:
        defects.append(
            ChainDefect(
                ChainFault.TORN_TAIL,
                location,
                len(lines) + 1,
                f"{len(torn)} byte(s) after the last complete record have no terminating "
                "newline, so the writer did not finish them",
            )
        )

    if expected_last_hash is not None:
        observed = last_hash if last_hash is not None else ZERO_PREV_HASH
        if observed != expected_last_hash:
            defects.append(
                ChainDefect(
                    ChainFault.TAIL_HASH_MISMATCH,
                    location,
                    len(lines),
                    f"the log ends at {observed} and {expected_last_hash} was expected",
                )
            )

    return ChainVerification(
        files=(location,),
        records=records,
        first_seq=first_seq,
        last_seq=last_seq,
        last_record_hash=last_hash,
        last_runner_now_ns=last_now_ns,
        torn_tail_bytes=len(torn),
        defects=tuple(defects),
    )


def verify_log(
    root: str | Path, *, expected_last_hash: str | None = None
) -> ChainVerification:
    """Verify a whole campaign: every day file, and the links between them.

    The day-to-day link is the reason this exists rather than a loop at the call
    site: a chain that verifies inside each file but whose second day does not
    continue the first is exactly what removing a whole day would leave behind.
    The campaign's *first* record is held to :data:`ZERO_PREV_HASH` here for the
    same reason — it is the one record a per-file walk has nothing to compare
    against, and it is the record an editor would remove first.
    """
    directory = Path(root)
    files = day_files(directory)
    if not files:
        if expected_last_hash is not None and expected_last_hash != ZERO_PREV_HASH:
            return ChainVerification(
                files=(),
                defects=(
                    ChainDefect(
                        ChainFault.TAIL_HASH_MISMATCH,
                        directory,
                        0,
                        f"the log holds no record and {expected_last_hash} was expected",
                    ),
                ),
            )
        return ChainVerification(files=())

    defects: list[ChainDefect] = []
    records = 0
    first_seq: int | None = None
    last_seq: int | None = None
    last_hash: str | None = None
    last_now_ns: int | None = None
    torn_bytes = 0
    #: The campaign's first record is anchored, not unconstrained: section 9.2
    #: fixes its ``prev_hash`` at :data:`ZERO_PREV_HASH`, and the writer numbers
    #: it ``seq`` 1. Seeding the walk with those is what makes deleting the head
    #: of the evidence visible. Without them the first record is the one record
    #: nothing checks, and dropping the campaign's opening records — or its whole
    #: first day — leaves a log whose every surviving record still hashes and
    #: still links, so it verifies clean; section 9.3's persisted
    #: ``last_record_hash`` does not catch it either, because the tail it names
    #: is untouched. These are kept apart from ``last_hash``/``last_seq`` so that
    #: a log holding no record still reports ``last_record_hash`` as ``None``
    #: rather than as the zero hash.
    chain_hash: str = ZERO_PREV_HASH
    chain_seq: int = 0
    for path in files:
        one = verify_chain(path, expected_prev_hash=chain_hash, start_seq=chain_seq)
        defects.extend(one.defects)
        records += one.records
        torn_bytes += one.torn_tail_bytes
        if first_seq is None:
            first_seq = one.first_seq
        if one.last_seq is not None:
            chain_seq = last_seq = one.last_seq
        if one.last_record_hash is not None:
            chain_hash = last_hash = one.last_record_hash
        if one.last_runner_now_ns is not None:
            last_now_ns = one.last_runner_now_ns

    if expected_last_hash is not None:
        observed = last_hash if last_hash is not None else ZERO_PREV_HASH
        if observed != expected_last_hash:
            defects.append(
                ChainDefect(
                    ChainFault.TAIL_HASH_MISMATCH,
                    files[-1],
                    0,
                    f"the log ends at {observed} and {expected_last_hash} was expected",
                )
            )

    return ChainVerification(
        files=tuple(files),
        records=records,
        first_seq=first_seq,
        last_seq=last_seq,
        last_record_hash=last_hash,
        last_runner_now_ns=last_now_ns,
        torn_tail_bytes=torn_bytes,
        defects=tuple(defects),
    )


@dataclass(frozen=True)
class TailRepair:
    """What :func:`recover_tail` removed, and where it put it."""

    path: Path | None
    truncated_bytes: int = 0
    truncated_path: Path | None = None

    @property
    def repaired(self) -> bool:
        """Whether anything was removed. ``False`` for a log that was intact."""
        return self.truncated_bytes > 0


def recover_tail(state_dir: str | Path) -> TailRepair:
    """Remove an unfinished trailing fragment from the newest day file.

    The one repair this module performs, and it is deliberately the smallest one
    that exists. Section 9.3 has the runner continue after a crash between the
    state write and the log write, and :meth:`DecisionLog.open` refuses to append
    onto a torn tail, so without a primitive here the only way forward would be
    file surgery in PR-10 on a file this module owns.

    **Only bytes with no terminating newline are removed**, and they are copied
    into a ``<day>.ndjson.truncated`` companion before the file is shortened, so
    a crash between the two loses nothing. This is where it parts company with
    :meth:`chimera.recorder.sink.RawSink.recover_tail`, which also drops
    *complete* trailing records it cannot read: here a complete record that does
    not verify is a forgery or corruption, and quietly deleting it would be the
    log repairing away the very evidence it exists to preserve. Such a file is
    left exactly as it is and :func:`verify_chain` keeps reporting it.

    Returns what was done. A log with nothing to repair is not an error: the
    caller asks whether the tail is torn by verifying, and asks for it to be
    removed by calling this, and the second question has a truthful answer even
    when the first was no.
    """
    root = Path(state_dir) / LOG_DIR_NAME
    files = day_files(root)
    if not files:
        return TailRepair(None)
    path = files[-1]
    torn = torn_tail_bytes(path)
    if not torn:
        return TailRepair(path)

    size = path.stat().st_size
    keep = size - torn
    companion = path.with_name(path.name + TRUNCATED_SUFFIX)
    try:
        with open(path, "rb") as reader:
            reader.seek(keep)
            salvage = reader.read()
        with open(companion, "ab") as writer:
            writer.write(salvage)
            writer.flush()
            os.fsync(writer.fileno())
        os.truncate(path, keep)
    except OSError as exc:
        raise DecisionLogError(
            f"could not preserve the torn tail of {path} into {companion}: {exc}. The "
            "bytes are left in place rather than removed unrecorded"
        ) from exc
    return TailRepair(path, truncated_bytes=torn, truncated_path=companion)


def read_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Every complete, well-formed record of one day file, in order.

    Stops at the first line it cannot read, because continuing past one would
    present the records after a corruption as though nothing had happened.
    :func:`verify_chain` is what says whether the file is trustworthy at all;
    this is for a caller that has already asked.
    """
    location = Path(path)
    with open(location, "rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                return
            record, _ = _parse_record(raw[:-1])
            if record is None:
                return
            yield record


__all__ = [
    "CORE_FIELDS",
    "DECISION_RECORD_SCHEMA",
    "EVIDENCE_KINDS",
    "HASH_PREFIX",
    "LOG_DIR_NAME",
    "LOG_SUFFIX",
    "OPERATIONAL_KINDS",
    "RESERVED_FIELDS",
    "TAIL_SCAN_BYTES",
    "TRUNCATED_SUFFIX",
    "UNCLASSIFIED_KINDS",
    "ZERO_PREV_HASH",
    "AppendedRecord",
    "ChainDefect",
    "ChainFault",
    "ChainVerification",
    "DecisionLog",
    "DecisionLogError",
    "DecisionLogTailError",
    "RecordKind",
    "TailRepair",
    "canonical_bytes",
    "canonical_json",
    "canonical_line",
    "compute_record_hash",
    "day_files",
    "decimal_str",
    "is_evidence",
    "is_hash",
    "iso_minute",
    "read_records",
    "recover_tail",
    "require_hash",
    "require_iso_minute",
    "require_iso_utc",
    "require_kind",
    "torn_tail_bytes",
    "validate_payload",
    "verify_chain",
    "verify_log",
]
