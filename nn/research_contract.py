"""Versioned Research Contracts: what a research generation may look at.

A Research Contract is a committed, immutable JSON document that says which
market a research generation studies and the instant at which its test data is
sealed. It is the successor to the single module-level
``SEALED_TEST_START_UTC`` constant: that constant was correct for one BTC cycle,
but a project with several instruments, timeframes and research generations
needs the boundary to be *selected* from a versioned set rather than edited in
place.

The rule that gives the whole thing its point::

    a new research generation is a new contract file,
    never a new boundary inside an existing one.

Editing ``sealed_test_start`` in a committed contract does not produce "the same
contract, updated" — it produces a different semantic identity
(:attr:`ResearchContract.contract_hash`), which every artifact records and
``nn.wf_diagnostics`` refuses to aggregate across. That is deliberate: once
research has begun, moving the seal after seeing results contaminates the
holdout, and the only honest way to change it is to declare a new generation.

**Nothing at runtime can construct a research boundary.** Contracts are read
from :data:`CONTRACTS_DIR` inside this package and nowhere else — there is no
``--sealed-date``, no external contract path, no inline JSON and no environment
variable. The research entrypoints expose a *selector* (``--research-contract``)
whose accepted values are exactly the committed contract ids, so the worst a
command line can do is pick a different committed generation, which it must then
also record in every artifact it writes.

**Identity is semantic, not textual.** :func:`canonical_material` reduces a
contract to the fields that define the research question, normalised, and
:attr:`ResearchContract.contract_hash` is SHA-256 over that. Reformatting the
file, reordering its keys, respelling the anchor as ``Z`` instead of ``+00:00``,
recasing the exchange, or rewriting ``description`` all leave the hash alone.
Changing the pair, the timeframe, the generation, the domain, the id or the
sealed instant all change it.

**Scope fails closed.** A contract for Binance BTC/USDT 1h applied to any other
exchange, pair or timeframe raises :class:`ContractScopeError` rather than
resolving a boundary against data it does not describe — including when the
dataset declares no identity at all, which is a dataset that cannot be shown to
be in scope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

#: Directory holding the committed contracts. One JSON file per contract, named
#: after its ``contract_id``. Repository-controlled: adding a research
#: generation is a reviewed commit, not a runtime decision.
CONTRACTS_DIR = Path(__file__).resolve().parent / "research_contracts"

#: Names the meaning of :func:`canonical_material`. Hashed with the contract, so
#: a future change to *what counts as* research-defining content is itself a
#: change of identity rather than a silent reinterpretation of old hashes.
CONTRACT_SCHEMA = "chimera.research-contract/1"

#: The contract every research entrypoint selects unless told otherwise: the
#: Binance BTC/USDT 1h generation the repository's committed research was
#: produced under.
DEFAULT_CONTRACT_ID = "btc-usdt-1h-gen1"

#: The contract the synthetic series from ``tools.make_sample_data`` runs under.
#:
#: Committed for the same reason as any other: the smoke path and the test suite
#: go through the real entrypoints, which resolve a real contract, and a dataset
#: outside every committed scope is refused. Its scope is the synthetic series,
#: so it can never be applied to market data and no run under it is comparable
#: with a run under a market contract.
SYNTHETIC_CONTRACT_ID = "synth-usdt-1h-gen1"

#: Keys a contract file must carry. A file missing one, or carrying anything
#: else, is refused rather than read with defaults: a contract that is quietly
#: half-specified is worse than no contract.
REQUIRED_FIELDS = (
    "contract_id",
    "research_generation",
    "domain",
    "scope",
    "sealed_test_start",
)

#: Keys a contract file may also carry, and which say nothing about the research
#: question. They are excluded from :func:`canonical_material`, so editing the
#: prose of a committed contract does not invent a new research generation.
DOCUMENTARY_FIELDS = ("description",)

#: The three fields that identify the market a contract describes.
SCOPE_FIELDS = ("exchange", "pair", "timeframe")


class ResearchContractError(ValueError):
    """A contract file cannot be read, or the registry is inconsistent."""


class ContractScopeError(ValueError):
    """A contract was applied to a dataset it does not describe."""


def normalise_scope_value(field: str, value: Any) -> str:
    """Case- and whitespace-insensitive form of one scope field.

    Exchanges and timeframes are written lower case and pairs upper case
    throughout this repository, so normalising here means ``Binance`` and
    ``binance`` name the same market rather than two incompatible ones — both
    for identity and for the compatibility check.
    """
    if not isinstance(value, str) or not value.strip():
        raise ResearchContractError(f"scope.{field} must be a non-empty string, got {value!r}")
    text = value.strip()
    return text.upper() if field == "pair" else text.lower()


@dataclass(frozen=True)
class DatasetScope:
    """The market a contract describes, normalised."""

    exchange: str
    pair: str
    timeframe: str

    def to_dict(self) -> dict[str, str]:
        return {"exchange": self.exchange, "pair": self.pair, "timeframe": self.timeframe}


@dataclass(frozen=True)
class ResearchContract:
    """One committed research generation.

    ``sealed_test_start`` is the immutable instant this generation seals at:
    everything strictly before it is research data, everything at or after it is
    sealed. It is not a default, not a heuristic, and not derived from any
    dataset — resolving it against a particular dataset yields a row index,
    which is metadata about that dataset and never the contract itself.
    """

    contract_id: str
    research_generation: int
    domain: str
    scope: DatasetScope
    sealed_test_start: pd.Timestamp
    description: str = ""
    #: Where this contract was read from, for error messages. Not part of
    #: identity: the same contract has the same hash from any checkout.
    source: Path | None = None

    @property
    def contract_hash(self) -> str:
        """SHA-256 over :func:`canonical_material`, as 64 lowercase hex digits."""
        return hashlib.sha256(canonical_material(self).encode("utf-8")).hexdigest()

    @property
    def short_hash(self) -> str:
        """First 16 hex digits, for logs and console lines. Never for identity."""
        return self.contract_hash[:16]

    @property
    def label(self) -> str:
        """``id@shorthash``: how a contract names itself in a log line."""
        return f"{self.contract_id}@{self.short_hash}"

    def provenance(self) -> dict[str, Any]:
        """The block an artifact records so it can name the exact contract used.

        The human-readable id alone is not enough — an id can be reused while
        its contents change — so the hash travels with it, together with the
        fields the hash is taken over. A reader with this block can recompute
        the identity without the contract file.
        """
        return {
            "contract_schema": CONTRACT_SCHEMA,
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "research_generation": self.research_generation,
            "domain": self.domain,
            "scope": self.scope.to_dict(),
            "sealed_test_start": self.sealed_test_start.isoformat(),
        }

    def require_scope(
        self, *, exchange: str, pair: str, timeframe: str, source: str = "the dataset"
    ) -> None:
        """Fail closed unless the dataset is the market this contract describes.

        A sealed instant only means something against the instrument it was
        declared for: resolving the BTC/USDT 1h seal against an ETH dataset
        would produce a row index that partitions data the contract never
        described, and an artifact claiming a research generation it does not
        belong to. Missing identity fails the same way as wrong identity — a
        dataset that does not say what it is cannot be shown to be in scope.
        """
        given = {"exchange": exchange, "pair": pair, "timeframe": timeframe}
        expected = self.scope.to_dict()
        problems = []
        for field in SCOPE_FIELDS:
            value = given[field]
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{field} is unset ({value!r}), expected {expected[field]!r}")
                continue
            if normalise_scope_value(field, value) != expected[field]:
                problems.append(f"{field} is {value!r}, expected {expected[field]!r}")
        if problems:
            raise ContractScopeError(
                f"research contract {self.label} describes "
                f"{expected['exchange']} {expected['pair']} {expected['timeframe']}, but "
                f"{source} does not match: {'; '.join(problems)}. A sealed instant is "
                "only meaningful for the market it was declared for; select the "
                "contract for this dataset, or add one."
            )


def canonical_material(contract: ResearchContract) -> str:
    """The exact bytes :attr:`ResearchContract.contract_hash` is taken over.

    Everything research-defining and nothing else, normalised so that two files
    describing the same research question hash the same however they are
    written: keys sorted, no insignificant whitespace, the anchor as a UTC
    ISO-8601 instant, scope values case-normalised, and the documentary fields
    (:data:`DOCUMENTARY_FIELDS`) absent.
    """
    material = {
        "contract_schema": CONTRACT_SCHEMA,
        "contract_id": contract.contract_id,
        "research_generation": contract.research_generation,
        "domain": contract.domain,
        "exchange": contract.scope.exchange,
        "pair": contract.scope.pair,
        "timeframe": contract.scope.timeframe,
        "sealed_test_start": contract.sealed_test_start.isoformat(),
    }
    return json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_contract(
    payload: Mapping[str, Any], *, source: Path | None = None
) -> ResearchContract:
    """Build a contract from a parsed JSON document, refusing anything unclear.

    Every failure here is a case where reading on would mean guessing at the
    research boundary: an unknown key could be a misspelled ``sealed_test_start``
    silently ignored, and a naive timestamp could be read in two time zones an
    hour apart.
    """
    where = f"{source}: " if source is not None else ""
    if not isinstance(payload, Mapping):
        raise ResearchContractError(f"{where}a research contract must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ResearchContractError(f"{where}missing required field(s): {sorted(missing)}")
    unknown = sorted(set(payload) - set(REQUIRED_FIELDS) - set(DOCUMENTARY_FIELDS))
    if unknown:
        raise ResearchContractError(
            f"{where}unknown field(s) {unknown}. A contract carries exactly "
            f"{sorted(REQUIRED_FIELDS)} plus {sorted(DOCUMENTARY_FIELDS)}; an unrecognised "
            "key is refused rather than ignored, because an ignored key can be a "
            "misspelling of one that defines the seal"
        )

    contract_id = payload["contract_id"]
    if not isinstance(contract_id, str) or not contract_id.strip():
        raise ResearchContractError(f"{where}contract_id must be a non-empty string")

    generation = payload["research_generation"]
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise ResearchContractError(
            f"{where}research_generation must be an integer >= 1, got {generation!r}"
        )

    domain = payload["domain"]
    if not isinstance(domain, str) or not domain.strip():
        raise ResearchContractError(f"{where}domain must be a non-empty string")

    scope = payload["scope"]
    if not isinstance(scope, Mapping):
        raise ResearchContractError(f"{where}scope must be an object")
    scope_unknown = sorted(set(scope) - set(SCOPE_FIELDS))
    scope_missing = sorted(set(SCOPE_FIELDS) - set(scope))
    if scope_unknown or scope_missing:
        raise ResearchContractError(
            f"{where}scope must carry exactly {list(SCOPE_FIELDS)}: unexpected "
            f"{scope_unknown}, missing {scope_missing}"
        )

    anchor = payload["sealed_test_start"]
    if not isinstance(anchor, str):
        raise ResearchContractError(
            f"{where}sealed_test_start must be an ISO-8601 string, got {type(anchor).__name__}"
        )
    try:
        stamp = pd.Timestamp(anchor)
    except (TypeError, ValueError) as exc:
        raise ResearchContractError(
            f"{where}sealed_test_start {anchor!r} is not a valid timestamp: {exc}"
        ) from exc
    if stamp.tzinfo is None:
        raise ResearchContractError(
            f"{where}sealed_test_start {anchor!r} has no UTC offset. The seal is an "
            "instant, and a naive timestamp names a different instant in every time zone"
        )

    description = payload.get("description", "")
    if not isinstance(description, str):
        raise ResearchContractError(f"{where}description must be a string")

    return ResearchContract(
        contract_id=contract_id.strip(),
        research_generation=generation,
        domain=domain.strip(),
        scope=DatasetScope(
            **{field: normalise_scope_value(field, scope[field]) for field in SCOPE_FIELDS}
        ),
        sealed_test_start=stamp.tz_convert("UTC"),
        description=description,
        source=source,
    )


def read_contract_file(path: str | Path) -> ResearchContract:
    """Parse one contract file. Used by the registry; not a runtime entrypoint.

    Deliberately *not* wired to any CLI: a contract loaded from an arbitrary
    path would be a research boundary chosen at runtime, which is the thing this
    module exists to prevent. Selection goes through :func:`load_contract`.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ResearchContractError(f"{path} is not readable JSON: {exc}") from exc
    contract = parse_contract(payload, source=path)
    if contract.contract_id != path.stem:
        raise ResearchContractError(
            f"{path} declares contract_id {contract.contract_id!r} but is named "
            f"{path.stem!r}. One file, one id, and the file name is how a reviewer "
            "finds the contract an artifact names"
        )
    return contract


@lru_cache(maxsize=1)
def _registry() -> dict[str, ResearchContract]:
    """Every committed contract, keyed by id. Read once per process."""
    if not CONTRACTS_DIR.is_dir():
        raise ResearchContractError(
            f"no research contract directory at {CONTRACTS_DIR}; the committed "
            "contracts are part of the package and research cannot be planned without them"
        )
    contracts: dict[str, ResearchContract] = {}
    for path in sorted(CONTRACTS_DIR.glob("*.json")):
        contract = read_contract_file(path)
        if contract.contract_id in contracts:
            raise ResearchContractError(
                f"two committed contracts share the id {contract.contract_id!r}"
            )
        contracts[contract.contract_id] = contract
    if not contracts:
        raise ResearchContractError(f"{CONTRACTS_DIR} holds no research contracts")
    return contracts


def available_contract_ids() -> list[str]:
    """The committed contract ids, sorted. These are the only selectable values."""
    return sorted(_registry())


def load_contract(contract_id: str = DEFAULT_CONTRACT_ID) -> ResearchContract:
    """Select one committed contract by id.

    The only way a research run obtains a boundary. An unknown id fails rather
    than falling back to the default: silently researching under a different
    generation than the one asked for is exactly the confusion contracts exist
    to remove.
    """
    registry = _registry()
    try:
        return registry[contract_id]
    except KeyError:
        raise ResearchContractError(
            f"unknown research contract {contract_id!r}. Committed contracts: "
            f"{available_contract_ids()}. A new research generation is added by "
            f"committing a contract to {CONTRACTS_DIR}, never by moving the boundary "
            "of an existing one."
        ) from None


def add_contract_argument(parser: Any) -> None:
    """Add the ``--research-contract`` selector to a research entrypoint.

    A selector, not a boundary: ``choices`` is the committed registry, so the
    flag can pick a different research generation but cannot invent one. There
    is deliberately no flag that takes a date, a file path or inline JSON.
    """
    parser.add_argument(
        "--research-contract",
        default=DEFAULT_CONTRACT_ID,
        choices=available_contract_ids(),
        help=(
            "Committed research contract to run under. Selects an existing "
            "generation; it cannot define a new sealed boundary. Default: "
            f"{DEFAULT_CONTRACT_ID}."
        ),
    )
