"""The prospective recorder's contract: what is recorded, and where the
prospective boundary will live.

A recorder contract is the forward-looking counterpart of
:mod:`nn.research_contract`. A research contract answers "what may this
generation look at", and it does so by naming a ``sealed_test_start`` — an
instant in the *past* of the data it describes. A prospective generation has no
seal, because none of its data exists yet; what it needs frozen instead is the
acquisition itself:

    what will be recorded, how it will be identified, how it will be stored,
    what timestamp semantics apply, and where the prospective boundary will
    live — all committed **before** the first byte is acquired.

That is the whole of this module. It is a separate schema rather than an edit to
:mod:`nn.research_contract`, whose parser requires ``sealed_test_start`` and
refuses unknown keys; a prospective contract carrying a seal would be a claim
about history it does not have.

**The boundary starts unset, and that is the honest state.**
:attr:`RecorderContract.prospective_from` is ``null`` in the committed file. It
is written once, by the recorder's first run, and committed in a follow-up pull
request; after that the file is immutable. Until then *no minute recorded under
this contract is scientific evidence* — which is why the key is required to be
present rather than optional. A missing key would be a contract that forgot to
say where its boundary is; an explicit ``null`` is a contract that says it does
not have one yet. :meth:`RecorderContract.with_prospective_from` is the only way
one is set, it is pure, it writes nothing, and it refuses to move a boundary
that already exists.

**Identity is semantic, not textual.** :func:`canonical_material` reduces a
contract to the fields that define the acquisition, normalised, and
:attr:`RecorderContract.contract_hash` is SHA-256 over that, following
:func:`nn.research_contract.canonical_material`. Reformatting the file,
reordering its keys, reordering ``streams``, recasing a symbol or rewriting
``description`` all leave the hash alone. Changing a stream, a market, the
timestamp semantics, the storage layout, the coverage rule, the reconciliation
rule — or writing the prospective boundary — all change it. The boundary is
inside the identity deliberately: the one edit the file is ever allowed is the
edit that finalises what the contract means, and it must be visible as a new
hash rather than as the same contract with a quietly different meaning.

**Nothing machine-specific enters the identity.** No absolute path, no user
name, no hostname, no temporary directory. The storage *root* is derived from an
injected base directory by :meth:`RecorderContract.storage_root`, so the same
contract has the same hash on every host and under every checkout.

This module opens no socket, makes no request and reads no clock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

#: Directory holding the committed recorder contracts. One JSON file per
#: contract, named after its ``contract_id``, exactly as
#: :data:`nn.research_contract.CONTRACTS_DIR` works: adding or activating a
#: prospective generation is a reviewed commit, never a runtime decision.
CONTRACTS_DIR = Path(__file__).resolve().parent / "contracts"

#: Names the meaning of :func:`canonical_material`. Hashed with the contract, so
#: a future change to *what counts as* acquisition-defining content is itself a
#: change of identity rather than a silent reinterpretation of old hashes.
CONTRACT_SCHEMA = "chimera.recorder-contract/1"

#: The prospective generation the demo records. The only committed contract.
GEN3_CONTRACT_ID = "btcusdt-prospective-gen3"

#: The only timezone a recorder contract may declare. Every instant in this
#: package is UTC and every day boundary is a UTC midnight; a contract naming
#: anything else would describe a recorder that does not exist.
CONTRACT_TIMEZONE = "UTC"

#: The storage layout this build writes. The layout itself — which directories
#: hold raw events, normalized days, funding settlements and their checksums —
#: is defined by :mod:`chimera.recorder.sink` and
#: :mod:`chimera.recorder.normalize`; the contract carries the *version* so a
#: tree written under a layout this build does not understand is refused rather
#: than read with today's assumptions.
STORAGE_LAYOUT_VERSION = 1

#: Storage layouts this build knows how to write and read.
SUPPORTED_STORAGE_LAYOUT_VERSIONS = frozenset({STORAGE_LAYOUT_VERSION})

#: Keys a contract file must carry. A file missing one, or carrying anything
#: else, is refused rather than read with defaults. ``prospective_from`` is in
#: this list and may be ``null``: the key is mandatory, and the *value* is what
#: says whether the boundary has been fixed.
REQUIRED_FIELDS: tuple[str, ...] = (
    "contract_schema",
    "contract_id",
    "generation",
    "exchange",
    "markets",
    "streams",
    "required_for_coverage",
    "timezone",
    "minute_key",
    "prospective_from",
    "boundary_rule",
    "sealed_regions_inherited",
    "storage_layout_version",
    "checksum_scheme",
    "coverage_rule",
    "reconciliation_rule",
    "recorder_version_policy",
)

#: Keys a contract file may also carry, and which say nothing about what is
#: acquired. Excluded from :func:`canonical_material`, so editing the prose of a
#: committed contract does not invent a new prospective generation.
DOCUMENTARY_FIELDS: tuple[str, ...] = ("description",)

#: The three fields that identify one market inside a contract.
MARKET_FIELDS: tuple[str, ...] = ("symbol", "instrument", "quote")

#: Characters a market key or stream name may use. A stream id is also a
#: directory name, so the alphabet is the one that survives every filesystem
#: this project runs on.
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class RecorderContractError(ValueError):
    """A recorder contract cannot be read, or the registry is inconsistent."""


class ProspectiveBoundaryError(RecorderContractError):
    """The prospective boundary cannot be set, or something tried to move it."""


@dataclass(frozen=True)
class Market:
    """One instrument a recorder contract covers.

    ``key`` is the prefix every one of that market's stream ids carries, which
    is what makes ``um.kline_1m`` resolvable to a symbol without a lookup table
    living somewhere else.
    """

    key: str
    symbol: str
    instrument: str
    quote: str

    def to_dict(self) -> dict[str, str]:
        return {"symbol": self.symbol, "instrument": self.instrument, "quote": self.quote}


@dataclass(frozen=True)
class RecorderContract:
    """One committed prospective generation.

    Every field except :attr:`description` and :attr:`source` is part of
    :attr:`contract_hash`. ``streams``, ``required_for_coverage``, ``markets``
    and ``sealed_regions_inherited`` are normalised at parse time — sorted,
    deduplicated, case-normalised — so a file's key order and stream order are
    presentation rather than identity.
    """

    contract_id: str
    generation: int
    exchange: str
    markets: tuple[Market, ...]
    streams: tuple[str, ...]
    required_for_coverage: tuple[str, ...]
    minute_key: str
    prospective_from: datetime | None
    boundary_rule: str
    sealed_regions_inherited: tuple[tuple[str, str], ...]
    storage_layout_version: int
    checksum_scheme: str
    coverage_rule: str
    reconciliation_rule: str
    recorder_version_policy: str
    description: str = ""
    #: Where this contract was read from, for error messages. Deliberately
    #: outside equality and outside the hash: the same contract is the same
    #: contract from any checkout, and a path is a fact about a machine.
    source: Path | None = field(default=None, compare=False, repr=False)

    @property
    def contract_hash(self) -> str:
        """SHA-256 over :func:`canonical_material`, as 64 lowercase hex digits."""
        return hashlib.sha256(canonical_material(self).encode("utf-8")).hexdigest()

    @property
    def short_hash(self) -> str:
        """First 16 hex digits, for logs and file names. Never for identity."""
        return self.contract_hash[:16]

    @property
    def label(self) -> str:
        """``id@shorthash``: how a contract names itself in a log line."""
        return f"{self.contract_id}@{self.short_hash}"

    @property
    def activated(self) -> bool:
        """Whether the prospective boundary has been written.

        ``False`` means every minute recorded under this contract is engineering
        data. It is not a defect and it is not a pending task: it is the state
        the contract is committed in, and the state PR-04 leaves it in.
        """
        return self.prospective_from is not None

    def market(self, key: str) -> Market:
        """The market a stream prefix names. An unknown key is refused."""
        for entry in self.markets:
            if entry.key == key:
                return entry
        raise RecorderContractError(
            f"recorder contract {self.label} declares markets "
            f"{[m.key for m in self.markets]}, not {key!r}"
        )

    def market_keys(self) -> tuple[str, ...]:
        """Declared market keys, sorted."""
        return tuple(entry.key for entry in self.markets)

    def streams_for(self, market: str) -> tuple[str, ...]:
        """Every declared stream belonging to one market, in canonical order."""
        self.market(market)
        return tuple(name for name in self.streams if name.split(".", 1)[0] == market)

    def storage_root(self, base: str | Path) -> Path:
        """Where this generation's data lives under ``base``.

        Injected rather than declared. A contract that carried an absolute local
        path in its identity would hash differently on the host that records the
        data and the host that reviews it, which is the exact failure the hash
        exists to make impossible.
        """
        return Path(base) / "prospective" / f"gen{self.generation}"

    def with_prospective_from(self, instant: datetime) -> "RecorderContract":
        """Fix the prospective boundary. Pure: returns a new contract, writes nothing.

        Refuses to move a boundary that is already set, which is what "immutable"
        means here, and refuses anything that is not an exact UTC midnight,
        because the boundary is defined as *the first UTC midnight after the
        recorder has run one complete hour* and an arbitrary instant would be a
        boundary chosen after the fact rather than derived.

        Calling this does not start a clock, does not record anything and does
        not make any minute scientific evidence. The recorder's first run
        supplies the instant, and committing the resulting file is what makes it
        real.
        """
        if self.prospective_from is not None:
            raise ProspectiveBoundaryError(
                f"recorder contract {self.label} already fixes prospective_from at "
                f"{self.prospective_from.isoformat()}. The boundary is written once and is "
                "immutable; a different boundary is a different generation, which is a new "
                "contract file with a supersedes field, never an edit to this one."
            )
        return replace(self, prospective_from=require_utc_midnight(instant))

    def to_dict(self) -> dict[str, Any]:
        """The contract as a JSON document, in the shape the committed file has."""
        return {
            "contract_schema": CONTRACT_SCHEMA,
            "contract_id": self.contract_id,
            "generation": self.generation,
            "exchange": self.exchange,
            "markets": {entry.key: entry.to_dict() for entry in self.markets},
            "streams": list(self.streams),
            "required_for_coverage": list(self.required_for_coverage),
            "timezone": CONTRACT_TIMEZONE,
            "minute_key": self.minute_key,
            "prospective_from": self._boundary_text(),
            "boundary_rule": self.boundary_rule,
            "sealed_regions_inherited": dict(self.sealed_regions_inherited),
            "storage_layout_version": self.storage_layout_version,
            "checksum_scheme": self.checksum_scheme,
            "coverage_rule": self.coverage_rule,
            "reconciliation_rule": self.reconciliation_rule,
            "recorder_version_policy": self.recorder_version_policy,
            "description": self.description,
        }

    def provenance(self) -> dict[str, Any]:
        """The block every day manifest records so it can name the exact contract.

        The id alone is not enough — an id can be reused while its contents
        change — so the hash travels with it, together with the boundary state
        the day was recorded under.
        """
        return {
            "contract_schema": CONTRACT_SCHEMA,
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "generation": self.generation,
            "storage_layout_version": self.storage_layout_version,
            "prospective_from": self._boundary_text(),
        }

    def _boundary_text(self) -> str | None:
        if self.prospective_from is None:
            return None
        return self.prospective_from.isoformat()


def canonical_material(contract: RecorderContract) -> str:
    """The exact bytes :attr:`RecorderContract.contract_hash` is taken over.

    Everything acquisition-defining and nothing else, normalised so that two
    files describing the same acquisition hash the same however they are
    written: keys sorted, no insignificant whitespace, streams in canonical
    order, the boundary as a UTC ISO-8601 instant or ``null``, and the
    documentary fields (:data:`DOCUMENTARY_FIELDS`) absent.

    ``ensure_ascii=False`` with an explicit UTF-8 encode at the call site: the
    bytes hashed are the same on a machine whose default encoding is cp1251 as
    on one whose default is UTF-8.
    """
    material = {
        "contract_schema": CONTRACT_SCHEMA,
        "contract_id": contract.contract_id,
        "generation": contract.generation,
        "exchange": contract.exchange,
        "markets": {entry.key: entry.to_dict() for entry in contract.markets},
        "streams": list(contract.streams),
        "required_for_coverage": list(contract.required_for_coverage),
        "timezone": CONTRACT_TIMEZONE,
        "minute_key": contract.minute_key,
        "prospective_from": contract._boundary_text(),
        "boundary_rule": contract.boundary_rule,
        "sealed_regions_inherited": dict(contract.sealed_regions_inherited),
        "storage_layout_version": contract.storage_layout_version,
        "checksum_scheme": contract.checksum_scheme,
        "coverage_rule": contract.coverage_rule,
        "reconciliation_rule": contract.reconciliation_rule,
        "recorder_version_policy": contract.recorder_version_policy,
    }
    return json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def contract_hash(contract: RecorderContract) -> str:
    """SHA-256 over :func:`canonical_material`. A contract's identity."""
    return contract.contract_hash


def require_utc_midnight(instant: Any) -> datetime:
    """The instant, in UTC, or an explanation of why it cannot be a boundary."""
    if not isinstance(instant, datetime):
        raise ProspectiveBoundaryError(
            f"prospective_from must be a datetime, got {type(instant).__name__}"
        )
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ProspectiveBoundaryError(
            f"prospective_from {instant!r} has no UTC offset. The boundary is an instant, "
            "and a naive timestamp names a different instant in every time zone"
        )
    moment = instant.astimezone(timezone.utc)
    midnight = moment.hour == moment.minute == moment.second == moment.microsecond == 0
    if not midnight:
        raise ProspectiveBoundaryError(
            f"prospective_from {moment.isoformat()} is not a UTC midnight. The boundary is "
            "the first UTC midnight after the recorder has run one complete hour; an "
            "instant chosen anywhere else is a boundary chosen rather than derived"
        )
    return moment


def _text(payload: Mapping[str, Any], key: str, where: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise RecorderContractError(f"{where}{key} must be a non-empty string, got {value!r}")
    return value.strip()


def _positive_int(payload: Mapping[str, Any], key: str, where: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RecorderContractError(f"{where}{key} must be an integer >= 1, got {value!r}")
    return value


def _stream_list(value: Any, key: str, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RecorderContractError(f"{where}{key} must be a non-empty list of stream ids")
    names: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise RecorderContractError(f"{where}{key} holds a non-string entry {entry!r}")
        name = entry.strip()
        market, _, rest = name.partition(".")
        if not market or not rest or "." in rest:
            raise RecorderContractError(
                f"{where}{key} entry {name!r} is not a <market>.<stream> id"
            )
        if set(market) - _NAME_CHARS or set(rest) - _NAME_CHARS:
            raise RecorderContractError(
                f"{where}{key} entry {name!r} carries a character outside [A-Za-z0-9_]; a "
                "stream id is also a directory name"
            )
        names.append(name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RecorderContractError(
            f"{where}{key} lists {duplicates} more than once. A stream is recorded once; a "
            "repeated entry is a file that would mean two different things"
        )
    return tuple(sorted(names))


def _parse_markets(payload: Mapping[str, Any], where: str) -> tuple[Market, ...]:
    raw = payload["markets"]
    if not isinstance(raw, Mapping) or not raw:
        raise RecorderContractError(f"{where}markets must be a non-empty object")
    markets: list[Market] = []
    for key in sorted(raw):
        if not isinstance(key, str) or not key or set(key) - _NAME_CHARS:
            raise RecorderContractError(f"{where}markets key {key!r} is not a stream prefix")
        entry = raw[key]
        if not isinstance(entry, Mapping):
            raise RecorderContractError(f"{where}markets.{key} must be an object")
        unknown = sorted(set(entry) - set(MARKET_FIELDS))
        missing = sorted(set(MARKET_FIELDS) - set(entry))
        if unknown or missing:
            raise RecorderContractError(
                f"{where}markets.{key} must carry exactly {list(MARKET_FIELDS)}: "
                f"unexpected {unknown}, missing {missing}"
            )
        prefix = f"{where}markets.{key}."
        markets.append(
            Market(
                key=key,
                symbol=_text(entry, "symbol", prefix).upper(),
                instrument=_text(entry, "instrument", prefix).lower(),
                quote=_text(entry, "quote", prefix).upper(),
            )
        )
    return tuple(markets)


def _parse_boundary(payload: Mapping[str, Any], where: str) -> datetime | None:
    boundary = payload["prospective_from"]
    if boundary is None:
        return None
    if not isinstance(boundary, str):
        raise RecorderContractError(
            f"{where}prospective_from must be null or an ISO-8601 string, got "
            f"{type(boundary).__name__}. Null means the boundary has not been fixed and "
            "nothing recorded so far is scientific evidence; it is a state, not an omission"
        )
    try:
        parsed = datetime.fromisoformat(boundary)
    except ValueError as exc:
        raise RecorderContractError(
            f"{where}prospective_from {boundary!r} is not an ISO-8601 instant: {exc}"
        ) from exc
    try:
        return require_utc_midnight(parsed)
    except ProspectiveBoundaryError as exc:
        raise RecorderContractError(f"{where}{exc}") from exc


def parse_recorder_contract(
    payload: Mapping[str, Any], *, source: Path | None = None
) -> RecorderContract:
    """Build a contract from a parsed JSON document, refusing anything unclear.

    Every failure here is a case where reading on would mean guessing at what
    the recorder will acquire: an unknown key could be a misspelled
    ``prospective_from`` silently ignored, a naive boundary could be read in two
    time zones an hour apart, and a ``required_for_coverage`` entry that is not
    a recorded stream would make the coverage gate measure something that is
    never written.
    """
    where = f"{source}: " if source is not None else ""
    if not isinstance(payload, Mapping):
        raise RecorderContractError(f"{where}a recorder contract must be a JSON object")

    missing = [name for name in REQUIRED_FIELDS if name not in payload]
    if missing:
        raise RecorderContractError(f"{where}missing required field(s): {sorted(missing)}")
    unknown = sorted(set(payload) - set(REQUIRED_FIELDS) - set(DOCUMENTARY_FIELDS))
    if unknown:
        raise RecorderContractError(
            f"{where}unknown field(s) {unknown}. A recorder contract carries exactly "
            f"{sorted(REQUIRED_FIELDS)} plus {sorted(DOCUMENTARY_FIELDS)}; an unrecognised "
            "key is refused rather than ignored, because an ignored key can be a "
            "misspelling of one that defines the acquisition"
        )

    schema = payload["contract_schema"]
    if schema != CONTRACT_SCHEMA:
        raise RecorderContractError(
            f"{where}contract_schema is {schema!r}, not {CONTRACT_SCHEMA!r}. A contract "
            "written under another schema is refused rather than best-effort parsed"
        )

    zone = payload["timezone"]
    if zone != CONTRACT_TIMEZONE:
        raise RecorderContractError(
            f"{where}timezone is {zone!r}, not {CONTRACT_TIMEZONE!r}. Every instant in the "
            "recorder is UTC and every day boundary is a UTC midnight"
        )

    layout = _positive_int(payload, "storage_layout_version", where)
    if layout not in SUPPORTED_STORAGE_LAYOUT_VERSIONS:
        raise RecorderContractError(
            f"{where}storage_layout_version {layout} is not one this build writes "
            f"({sorted(SUPPORTED_STORAGE_LAYOUT_VERSIONS)}). Reading a layout this code "
            "does not know with today's assumptions is how a directory silently changes "
            "meaning"
        )

    markets = _parse_markets(payload, where)
    streams = _stream_list(payload["streams"], "streams", where)
    required = _stream_list(payload["required_for_coverage"], "required_for_coverage", where)
    market_keys = {entry.key for entry in markets}
    orphans = sorted({name for name in streams if name.split(".", 1)[0] not in market_keys})
    if orphans:
        raise RecorderContractError(
            f"{where}streams {orphans} name a market this contract does not declare "
            f"({sorted(market_keys)}). A stream's market prefix is what resolves it to a "
            "symbol, so a stream without one records an unnamed instrument"
        )
    unrecorded = sorted(set(required) - set(streams))
    if unrecorded:
        raise RecorderContractError(
            f"{where}required_for_coverage names {unrecorded}, which is not in streams. The "
            "coverage gate would then measure a stream the recorder never writes"
        )

    sealed = payload["sealed_regions_inherited"]
    if not isinstance(sealed, Mapping):
        raise RecorderContractError(f"{where}sealed_regions_inherited must be an object")
    for key, value in sealed.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            raise RecorderContractError(
                f"{where}sealed_regions_inherited.{key!r} must map a name to a non-empty note"
            )

    description = payload.get("description", "")
    if not isinstance(description, str):
        raise RecorderContractError(f"{where}description must be a string")

    return RecorderContract(
        contract_id=_text(payload, "contract_id", where),
        generation=_positive_int(payload, "generation", where),
        exchange=_text(payload, "exchange", where).lower(),
        markets=markets,
        streams=streams,
        required_for_coverage=required,
        minute_key=_text(payload, "minute_key", where),
        prospective_from=_parse_boundary(payload, where),
        boundary_rule=_text(payload, "boundary_rule", where),
        sealed_regions_inherited=tuple(sorted((str(k), str(v)) for k, v in sealed.items())),
        storage_layout_version=layout,
        checksum_scheme=_text(payload, "checksum_scheme", where),
        coverage_rule=_text(payload, "coverage_rule", where),
        reconciliation_rule=_text(payload, "reconciliation_rule", where),
        recorder_version_policy=_text(payload, "recorder_version_policy", where),
        description=description,
        source=source,
    )


def read_recorder_contract_file(path: str | Path) -> RecorderContract:
    """Parse one contract file. Used by the registry; not a runtime entrypoint.

    Deliberately not wired to any CLI, for the reason
    :func:`nn.research_contract.read_contract_file` is not: a contract loaded
    from an arbitrary path would be an acquisition chosen at runtime. Selection
    goes through :func:`load_recorder_contract`.
    """
    location = Path(path)
    try:
        text = location.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecorderContractError(f"{location} could not be read: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RecorderContractError(f"{location} is not readable JSON: {exc}") from exc
    contract = parse_recorder_contract(payload, source=location)
    if contract.contract_id != location.stem:
        raise RecorderContractError(
            f"{location} declares contract_id {contract.contract_id!r} but is named "
            f"{location.stem!r}. One file, one id, and the file name is how a reviewer "
            "finds the contract a manifest names"
        )
    return contract


@lru_cache(maxsize=1)
def _registry() -> dict[str, RecorderContract]:
    """Every committed recorder contract, keyed by id. Read once per process."""
    if not CONTRACTS_DIR.is_dir():
        raise RecorderContractError(
            f"no recorder contract directory at {CONTRACTS_DIR}; the committed contracts "
            "are part of the package and no acquisition can be planned without them"
        )
    contracts: dict[str, RecorderContract] = {}
    for path in sorted(CONTRACTS_DIR.glob("*.json")):
        contract = read_recorder_contract_file(path)
        if contract.contract_id in contracts:
            raise RecorderContractError(
                f"two committed contracts share the id {contract.contract_id!r}"
            )
        contracts[contract.contract_id] = contract
    if not contracts:
        raise RecorderContractError(f"{CONTRACTS_DIR} holds no recorder contracts")
    return contracts


def available_recorder_contract_ids() -> list[str]:
    """The committed contract ids, sorted. These are the only selectable values."""
    return sorted(_registry())


def load_recorder_contract(contract_id: str = GEN3_CONTRACT_ID) -> RecorderContract:
    """Select one committed recorder contract by id.

    An unknown id fails rather than falling back to the default: silently
    recording under a different generation than the one asked for is exactly the
    confusion contracts exist to remove.
    """
    registry = _registry()
    try:
        return registry[contract_id]
    except KeyError:
        raise RecorderContractError(
            f"unknown recorder contract {contract_id!r}. Committed contracts: "
            f"{available_recorder_contract_ids()}. A new prospective generation is added by "
            f"committing a contract to {CONTRACTS_DIR}, never by editing an existing one."
        ) from None
