"""The demo campaign's configuration: parsed strictly, hashed semantically.

A campaign's ``config_hash`` is stamped on every decision record (section 9.1),
so it has to answer one question exactly: *were these two records produced under
the same configuration?* That makes two properties load-bearing and they pull in
opposite directions, which is why both are enforced here rather than left to a
convention.

**Identity is semantic, not textual.** Reformatting the file, reordering its
keys, changing its indentation, rewriting its ``description``, saving it with
CRLF line endings, or reading it from a different directory on a different host
all leave the hash alone. Changing a limit, the campaign id, the profile, or
freezing the protocol hash all move it. Nothing machine-specific is in the
material: no absolute path, no user name, no hostname, so the host that runs the
campaign and the host that audits it compute the same value.

**An unrecognised key is refused, never ignored.** ``chimera.risk``'s
``RiskLimits.from_dict`` drops keys it does not know, which is right for a
tolerant runtime config and wrong here: a misspelled ``max_drawdown_pct`` would
silently leave the *default* limit in force while the file on disk, and the
reviewer reading it, said something else. Every key of a demo config is
decision-relevant except :data:`DOCUMENTARY_FIELDS`, and anything else is an
error naming the key.

**A campaign configuration may not carry ``faults``.** Section 5.3 puts fault
injection — dropped quotes, duplicated fill events, forced restarts, partial
fills — under test and soak configurations only, and says the campaign config
schema forbids the key. The rule implemented here is deliberately total: for
:attr:`ConfigProfile.CAMPAIGN` the key ``faults`` may not appear **anywhere in
the document, at any depth, in any object, under any capitalisation**. A rule
scoped to the top level would be satisfied by nesting it one block deeper, and a
case-sensitive rule would be satisfied by ``Faults``; neither would be an honest
reading of "the campaign config schema forbids the key". The refusal names the
path it was found at. The same rule is enforced a second time by
:class:`DemoConfig` itself, because a configuration that never went through the
parser — built by a fixture, a CLI override merge, or a ``dataclasses.replace``
that changes one field of a soak config — would otherwise reach PR-10 carrying a
fault schedule and hash it into the campaign's identity. Soak and test profiles
accept the block and hash it like any other field, so a soak run's identity
records that faults were configured. The block's own schema belongs to
``chimera/demo/faults.py``, which is PR-10's.

**The profile is declared in the file and demanded by the caller.** Both, not
either: the file says what it is so that its identity records it, and
:func:`parse_demo_config` requires :attr:`ConfigProfile.CAMPAIGN` unless the
caller explicitly asks for something else, so reaching the fault-injection
schema is a deliberate act in two places at once.

**PR-09 freezes no science.** The limits below are section 7.4's *proposed*
values, which that section marks "subject to S2". This module and
``conf/demo/pvc1.json`` are the plumbing that will carry a protocol, not the
protocol: ``protocol_hash`` is ``null`` today, and PR-14 — the PVC-1
preregistration — is what freezes the prospective protocol, computes its hash
and writes it here. Until then no campaign is authorised, no evidence exists,
and a config that parses is a config that parses and nothing more.

This module opens no socket, makes no request and reads no clock.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from chimera.demo.decision_log import (
    HASH_PREFIX,
    DecisionLogError,
    canonical_json,
    is_hash,
)

#: Names the meaning of :func:`canonical_material`. Part of the hashed material,
#: so a future change to *what counts as* decision-relevant content is itself a
#: change of identity rather than a silent reinterpretation of old hashes.
CONFIG_SCHEMA = "chimera.demo-campaign-config/1"

#: Keys a demo config must carry. A file missing one, or carrying anything not
#: listed here or in :data:`DOCUMENTARY_FIELDS`, is refused rather than read with
#: defaults. ``protocol_hash`` is in this list and may be ``null``: the key is
#: mandatory and the *value* is what says whether the protocol has been frozen,
#: exactly as ``prospective_from`` works in the recorder contract.
REQUIRED_FIELDS: tuple[str, ...] = (
    "config_schema",
    "campaign_id",
    "profile",
    "protocol_hash",
    "limits",
)

#: Keys a config may also carry and which say nothing about what is decided.
#: Excluded from :func:`canonical_material`, so editing the prose of a committed
#: config does not invent a new campaign identity.
DOCUMENTARY_FIELDS: tuple[str, ...] = ("description",)

#: The fault-injection block. Permitted only outside a campaign profile, and
#: refused at every depth inside one. See the module docstring.
FAULTS_FIELD = "faults"

#: Section 7.4's proposed demo limits, by name. Every one is required: a limit
#: that could be omitted would fall back to a default nobody reviewed, and the
#: reviewer of a campaign config has to be able to read every bound off the file
#: in front of them. The integer-valued ones are counts and are refused as
#: floats, so ``2`` and ``2.0`` cannot become two identities for one campaign.
COUNT_LIMITS: tuple[str, ...] = (
    "funding_adverse_streak_limit",
    "loss_streak_limit",
    "max_open_positions",
    "max_orders_per_minute",
)

#: The limits carried as real numbers. Normalised to ``float`` before hashing,
#: so ``1`` and ``1.0`` are one value and one identity.
RATIO_LIMITS: tuple[str, ...] = (
    "cooldown_seconds",
    "max_daily_loss_pct",
    "max_data_delay_s",
    "max_drawdown_pct",
    "max_exposure_per_asset_pct",
    "max_funding_cost_rate",
    "max_leverage",
    "max_total_exposure_pct",
    "min_liquidation_distance_pct",
)

#: Every limit name, sorted. The two kinds above partition it.
LIMIT_FIELDS: tuple[str, ...] = tuple(sorted(COUNT_LIMITS + RATIO_LIMITS))

#: Characters a campaign id may use. It names a directory of evidence and a row
#: in a report, so the alphabet is the one that survives every filesystem this
#: project runs on and every place the id is printed.
_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")


class DemoConfigError(ValueError):
    """A demo configuration cannot be read, and reading it anyway would be worse."""


class ConfigProfile(str, Enum):
    """What a configuration is for, and therefore what its schema admits.

    The distinction is not cosmetic. :attr:`CAMPAIGN` is the profile whose
    decision records are evidence, and it is the profile that may not configure
    a fault. :attr:`SOAK` and :attr:`TEST` exist to inject faults deliberately
    and their records are operational.
    """

    CAMPAIGN = "CAMPAIGN"
    SOAK = "SOAK"
    TEST = "TEST"


#: Profiles that may configure fault injection (section 5.3).
FAULT_PROFILES: frozenset[ConfigProfile] = frozenset({ConfigProfile.SOAK, ConfigProfile.TEST})


@dataclass(frozen=True)
class DemoLimits:
    """Section 7.4's demo limits, as the config carries them.

    A separate type from :class:`chimera.risk.RiskLimits`, deliberately, for
    three reasons that all point the same way. The names differ — 7.4 asks for
    ``max_funding_cost_rate`` where ``RiskLimits`` has ``max_funding_rate`` — and
    7.4's ``funding_adverse_streak_limit`` does not exist there at all; it is
    PR-03's to add, and PR-09 does not depend on PR-03. ``RiskLimits.from_dict``
    silently drops unknown keys, which is the one behaviour this module exists to
    refuse. And importing ``chimera.risk`` here would make the demo config depend
    on a module that is being changed concurrently, for the sake of a dataclass
    that would then have to be filtered on the way in and on the way out. The
    runner (PR-10) is where the two meet: it is the thing that builds a
    ``RiskEngine`` from a parsed campaign config, and mapping one to the other in
    one visible place is better than sharing a type that fits neither.
    """

    funding_adverse_streak_limit: int
    loss_streak_limit: int
    max_open_positions: int
    max_orders_per_minute: int
    cooldown_seconds: float
    max_daily_loss_pct: float
    max_data_delay_s: float
    max_drawdown_pct: float
    max_exposure_per_asset_pct: float
    max_funding_cost_rate: float
    max_leverage: float
    max_total_exposure_pct: float
    min_liquidation_distance_pct: float

    def to_dict(self) -> dict[str, Any]:
        """The limits as the hashed material carries them: counts, then reals."""
        material: dict[str, Any] = {name: int(getattr(self, name)) for name in COUNT_LIMITS}
        material.update({name: float(getattr(self, name)) for name in RATIO_LIMITS})
        return material


@dataclass(frozen=True)
class DemoConfig:
    """One parsed, validated campaign configuration and its identity."""

    campaign_id: str
    profile: ConfigProfile
    protocol_hash: str | None
    limits: DemoLimits
    faults: Mapping[str, Any] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """A campaign configuration carries no fault schedule, whoever built it.

        The same rule :func:`_refuse_faults` applies to the document, applied to
        the object. Both are needed and neither replaces the other: the document
        check is the only one that can name the JSON path a stray key was found
        at, and this one is the only one that holds for a configuration that
        never went through :func:`parse_demo_config` at all — a builder, a test
        fixture, a CLI override merge, or the ``dataclasses.replace`` that turns
        a soak configuration into a campaign by changing one field. Without it
        the single most safety-relevant rule in this module would live in one
        function rather than in the type PR-10 passes around, and a campaign
        whose evidence was produced under an injected fault schedule would have
        a well-formed ``config_hash`` that nothing flagged.
        """
        if self.profile is ConfigProfile.CAMPAIGN and self.faults is not None:
            raise DemoConfigError(
                f"a {ConfigProfile.CAMPAIGN.value} configuration carries no "
                f"{FAULTS_FIELD!r}, and this one was built with "
                f"{sorted(self.faults)}. Fault injection is enabled by a test or soak "
                "configuration and never by a campaign configuration (section 5.3): a "
                "campaign whose evidence was produced with injected faults would be "
                "evidence about the fault schedule"
            )

    @property
    def config_hash(self) -> str:
        """``sha256:<hex>`` over :func:`canonical_material`. The config's identity."""
        return config_hash(self)

    @property
    def protocol_frozen(self) -> bool:
        """Whether PR-14's prospective protocol hash has been written yet.

        ``False`` in every committed configuration today. A campaign whose
        protocol is not frozen has nothing preregistered to be evidence *for*;
        acting on that is the runner's decision and not this module's, so the
        fact is exposed rather than enforced here.
        """
        return self.protocol_hash is not None


def canonical_material(config: DemoConfig) -> str:
    """The exact text :func:`config_hash` is taken over.

    Everything decision-relevant and nothing else, normalised so that two files
    describing the same campaign hash the same however they are written: keys
    sorted, no insignificant whitespace, counts as integers, real limits as
    floats, the documentary fields absent, and no path anywhere.

    It reuses the decision log's serializer rather than defining a second one.
    Section 9.2 fixes that function for the record stream, this package has
    exactly one canonical JSON form, and a config hash computed with a slightly
    different one would be a second convention nobody would remember to keep in
    step.
    """
    material: dict[str, Any] = {
        "config_schema": CONFIG_SCHEMA,
        "campaign_id": config.campaign_id,
        "profile": config.profile.value,
        "protocol_hash": config.protocol_hash,
        "limits": config.limits.to_dict(),
    }
    if config.faults is not None:
        material[FAULTS_FIELD] = dict(config.faults)
    return canonical_json(material)


def config_hash(config: DemoConfig) -> str:
    """SHA-256 over :func:`canonical_material`, in the ``sha256:<hex>`` form."""
    digest = hashlib.sha256(canonical_material(config).encode("ascii")).hexdigest()
    return HASH_PREFIX + digest


# --- parsing ----------------------------------------------------------------
def _refuse_faults(node: Any, path: str) -> None:
    """Refuse a ``faults`` key at any depth, naming where it was found.

    Case-insensitive on purpose: a schema that forbids ``faults`` and admits
    ``Faults`` forbids nothing. Applied to the *raw document* before anything is
    interpreted, so the refusal cannot be reached around by a block the schema
    would otherwise have rejected for a different reason.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            where = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key.casefold() == FAULTS_FIELD:
                raise DemoConfigError(
                    f"a campaign configuration carries no {FAULTS_FIELD!r} key and this one "
                    f"has it at {where}. Fault injection is enabled by a test or soak "
                    "configuration and never by a campaign configuration (section 5.3): a "
                    "campaign whose evidence was produced with injected faults would be "
                    "evidence about the fault schedule"
                )
            _refuse_faults(value, where)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _refuse_faults(value, f"{path}[{index}]")


def _require_number(value: Any, *, name: str, where: str) -> float:
    """A finite, strictly positive real limit."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DemoConfigError(
            f"{where}limits.{name} must be a number, got {type(value).__name__} {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise DemoConfigError(
            f"{where}limits.{name} is {value!r}, which is not a finite limit"
        )
    if number <= 0.0:
        raise DemoConfigError(
            f"{where}limits.{name} is {value!r}. Every demo limit is a strictly positive "
            "bound; a zero or negative one would either block everything or bound nothing, "
            "and neither is a limit somebody meant to write"
        )
    return number


def _require_count(value: Any, *, name: str, where: str) -> int:
    """A whole count of at least one."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DemoConfigError(
            f"{where}limits.{name} counts things and must be an integer, got "
            f"{type(value).__name__} {value!r}. A float here would give one campaign two "
            "identities, because 2 and 2.0 do not serialise the same way"
        )
    if value < 1:
        raise DemoConfigError(
            f"{where}limits.{name} is {value!r}; a count of zero or less is not a limit"
        )
    return value


def parse_demo_limits(payload: Any, *, where: str = "") -> DemoLimits:
    """Build :class:`DemoLimits`, refusing a missing or unrecognised limit."""
    if not isinstance(payload, Mapping):
        raise DemoConfigError(f"{where}limits must be a JSON object")
    missing = [name for name in LIMIT_FIELDS if name not in payload]
    if missing:
        raise DemoConfigError(f"{where}limits is missing {sorted(missing)}")
    unknown = sorted(set(payload) - set(LIMIT_FIELDS))
    if unknown:
        raise DemoConfigError(
            f"{where}limits carries unknown entr(ies) {unknown}. The demo limits are "
            f"exactly {list(LIMIT_FIELDS)}; an unrecognised name is refused rather than "
            "ignored, because an ignored name is usually a misspelling of a bound that is "
            "then silently not in force"
        )
    values: dict[str, Any] = {
        name: _require_count(payload[name], name=name, where=where) for name in COUNT_LIMITS
    }
    values.update(
        {name: _require_number(payload[name], name=name, where=where) for name in RATIO_LIMITS}
    )
    return DemoLimits(**values)


def parse_demo_config(
    payload: Mapping[str, Any],
    *,
    expected_profile: ConfigProfile = ConfigProfile.CAMPAIGN,
    source: Path | None = None,
) -> DemoConfig:
    """Build a configuration from a parsed JSON document, refusing anything unclear.

    ``expected_profile`` defaults to :attr:`ConfigProfile.CAMPAIGN` so that the
    unqualified call is the safe one: reading a soak or test configuration — the
    only kinds that may configure a fault — takes an explicit argument at the
    call site, where a reviewer can see it.
    """
    where = f"{source}: " if source is not None else ""
    if not isinstance(payload, Mapping):
        raise DemoConfigError(f"{where}a demo configuration must be a JSON object")
    if not isinstance(expected_profile, ConfigProfile):
        raise DemoConfigError(
            f"expected_profile must be a ConfigProfile, got {expected_profile!r}"
        )
    if expected_profile not in FAULT_PROFILES:
        _refuse_faults(payload, "")

    missing = [name for name in REQUIRED_FIELDS if name not in payload]
    if missing:
        raise DemoConfigError(f"{where}missing required field(s) {sorted(missing)}")
    permitted = set(REQUIRED_FIELDS) | set(DOCUMENTARY_FIELDS)
    if expected_profile in FAULT_PROFILES:
        permitted.add(FAULTS_FIELD)
    unknown = sorted(set(payload) - permitted)
    if unknown:
        raise DemoConfigError(
            f"{where}unknown field(s) {unknown}. A demo configuration carries exactly "
            f"{sorted(REQUIRED_FIELDS)} plus {sorted(DOCUMENTARY_FIELDS)}; an unrecognised "
            "key is refused rather than ignored, because an ignored key can be a "
            "misspelling of one that governs a decision"
        )

    schema = payload["config_schema"]
    if schema != CONFIG_SCHEMA:
        raise DemoConfigError(
            f"{where}config_schema is {schema!r}, not {CONFIG_SCHEMA!r}. A configuration "
            "written under another schema is refused rather than best-effort parsed"
        )

    declared = payload["profile"]
    try:
        profile = ConfigProfile(declared)
    except ValueError:
        raise DemoConfigError(
            f"{where}profile is {declared!r}, not one of "
            f"{sorted(p.value for p in ConfigProfile)}"
        ) from None
    if profile is not expected_profile:
        raise DemoConfigError(
            f"{where}this configuration declares profile {profile.value} and a "
            f"{expected_profile.value} configuration was asked for. The profile is in the "
            "hashed identity and the caller states which one it wants, so neither side "
            "alone can turn a soak configuration into a campaign"
        )

    campaign_id = payload["campaign_id"]
    if not isinstance(campaign_id, str) or not campaign_id:
        raise DemoConfigError(f"{where}campaign_id must be a non-empty string")
    if len(campaign_id) > 64 or set(campaign_id) - _ID_CHARS:
        raise DemoConfigError(
            f"{where}campaign_id {campaign_id!r} must be at most 64 characters of "
            f"{''.join(sorted(_ID_CHARS))}. It names a directory of evidence, so it is "
            "held to the alphabet every filesystem this project runs on accepts"
        )

    protocol_hash = payload["protocol_hash"]
    if protocol_hash is not None and not is_hash(protocol_hash):
        raise DemoConfigError(
            f"{where}protocol_hash is {protocol_hash!r}; it is null until the prospective "
            f"protocol is frozen, and {HASH_PREFIX}<64 hex digits> afterwards. PR-14 is "
            "what writes it, and PR-09 freezes no scientific protocol"
        )

    description = payload.get("description", "")
    if not isinstance(description, str):
        raise DemoConfigError(f"{where}description must be a string")

    faults = payload.get(FAULTS_FIELD)
    if faults is not None and not isinstance(faults, Mapping):
        raise DemoConfigError(
            f"{where}{FAULTS_FIELD} must be a JSON object. Its contents are the fault "
            "schedule's own schema and belong to chimera/demo/faults.py; this parser only "
            "records that a schedule was configured, and carries it into the hash"
        )
    if faults is not None:
        # The block goes into the hashed material unread, so the one thing this
        # parser does have to establish is that it *can* be hashed. Checked here
        # rather than at the first `config_hash` call, so that a block holding a
        # non-string key or a value JSON has no form for is a configuration
        # error at the point the file is read, and not a DecisionLogError
        # surfacing later out of a property of a parsed config.
        try:
            canonical_json(dict(faults))
        except DecisionLogError as exc:
            raise DemoConfigError(
                f"{where}{FAULTS_FIELD} cannot be serialised canonically: {exc}. It is "
                "part of the configuration's identity, so a block that cannot be hashed "
                "is a configuration that has none"
            ) from exc

    return DemoConfig(
        campaign_id=campaign_id,
        profile=profile,
        protocol_hash=protocol_hash,
        limits=parse_demo_limits(payload["limits"], where=where),
        faults=None if faults is None else dict(faults),
        description=description,
    )


def _refuse_json_constant(token: str) -> Any:
    """Refuse ``NaN``/``Infinity`` in a configuration file. Never returns."""
    raise ValueError(
        f"{token} is not JSON. Python's decoder accepts the bare token and no other "
        "reader does, so a configuration carrying one is a file that means one thing "
        "here and nothing anywhere else"
    )


def _refuse_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object, refusing a key that appears twice.

    Python's decoder keeps the *last* of a repeated key, so a file listing
    ``max_leverage`` twice would put the second value in force while a reviewer
    reading the file from the top sees the first. That is the reviewer-reads-one
    thing-and-the-machine-reads-another failure this whole module exists to
    prevent, so it is refused rather than resolved by a rule nobody looked up.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(
                f"the key {key!r} appears more than once. A repeated key would leave the "
                "value a reader sees first not the one in force"
            )
        seen[key] = value
    return seen


def read_demo_config_file(path: str | Path) -> Mapping[str, Any]:
    """The parsed JSON of a config file, read as UTF-8 whatever the host default.

    Explicit encoding, because a file holding a non-ASCII description would be
    read differently on a host whose default is cp1251, and a configuration that
    means two things is worse than one that fails to load. For the same reason
    the decoder is given no latitude either: Python accepts the non-JSON
    ``NaN``/``Infinity`` tokens and silently keeps the last of a duplicated key,
    and both are refused here so that "this file parsed" means the same thing to
    every reader of it.
    """
    location = Path(path)
    try:
        text = location.read_text(encoding="utf-8")
    except OSError as exc:
        raise DemoConfigError(
            f"could not read the demo configuration {location}: {exc}"
        ) from exc
    try:
        payload = json.loads(
            text,
            parse_constant=_refuse_json_constant,
            object_pairs_hook=_refuse_duplicate_keys,
        )
    except ValueError as exc:
        raise DemoConfigError(f"{location} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DemoConfigError(f"{location} must hold a JSON object")
    return payload


def load_demo_config(
    path: str | Path, *, expected_profile: ConfigProfile = ConfigProfile.CAMPAIGN
) -> DemoConfig:
    """Read and parse one configuration file.

    The path is used to open the file and to write better error messages, and it
    enters neither :func:`canonical_material` nor the hash: the same bytes are
    the same campaign whether they are read from a checkout, from a container's
    ``/etc`` or from a reviewer's temporary directory.
    """
    location = Path(path)
    return parse_demo_config(
        read_demo_config_file(location), expected_profile=expected_profile, source=location
    )


__all__ = [
    "CONFIG_SCHEMA",
    "COUNT_LIMITS",
    "DOCUMENTARY_FIELDS",
    "FAULTS_FIELD",
    "FAULT_PROFILES",
    "LIMIT_FIELDS",
    "RATIO_LIMITS",
    "REQUIRED_FIELDS",
    "ConfigProfile",
    "DemoConfig",
    "DemoConfigError",
    "DemoLimits",
    "canonical_material",
    "config_hash",
    "load_demo_config",
    "parse_demo_config",
    "parse_demo_limits",
    "read_demo_config_file",
]
