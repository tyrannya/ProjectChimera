"""The ENGINEERING acquisition schema: an acquisition that cannot become evidence.

R2's gen4 preflight has to record ~20 perpetuals for 30 days in order to measure
what a year of them costs and which of their streams can be verified at all.
None of that may ever be scored. The coordination record classifies this schema
work as ``NOT owned by anyone yet`` and as something that ``must exist before R2
can instantiate its contract``; this is that work.

The property under test throughout is one sentence: **the boundary fields are
absent by schema rather than null**, so there is no value an edit could fill in.
Every test below is one way of trying to get a boundary onto an engineering
acquisition, with the prospective control beside it.

ENGINEERING only. No scientific evidence, no boundary is activated, no contract
file naming a symbol universe is created -- the universe is measured in R2 and
frozen in R4, and choosing one here would be choosing it before the measurement.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from chimera.recorder.contract import (
    BOUNDARY_FIELDS,
    CONTRACT_SCHEMA,
    ENGINEERING_CONTRACT_SCHEMA,
    ENGINEERING_REQUIRED_FIELDS,
    REQUIRED_FIELDS,
    ProspectiveBoundaryError,
    RecorderContractError,
    canonical_material,
    load_recorder_contract,
    parse_recorder_contract,
)

MIDNIGHT = datetime(2026, 10, 1, tzinfo=timezone.utc)


def prospective_payload(**overrides):
    """The committed gen3 contract as a document, so the shapes cannot drift."""
    document = load_recorder_contract("btcusdt-prospective-gen3").to_dict()
    document.update(overrides)
    return document


def engineering_payload(**overrides):
    """The same acquisition, written as an engineering contract."""
    document = prospective_payload()
    document["contract_schema"] = ENGINEERING_CONTRACT_SCHEMA
    document["contract_id"] = "gen4-preflight"
    for field in BOUNDARY_FIELDS:
        document.pop(field, None)
    document.update(overrides)
    return document


# ---------------------------------------------------------------------------
# the boundary is absent, not null
# ---------------------------------------------------------------------------
def test_an_engineering_contract_parses_without_the_boundary_fields():
    contract = parse_recorder_contract(engineering_payload())

    assert contract.is_engineering is True
    assert contract.activated is False
    assert contract.prospective_from is None


def test_a_prospective_contract_still_requires_them():
    """The control: nothing about this change relaxes the prospective schema.

    A prospective contract without ``prospective_from`` is a contract that
    forgot to say where its boundary goes, and it is still refused.
    """
    for field in BOUNDARY_FIELDS:
        document = prospective_payload()
        document.pop(field)
        with pytest.raises(RecorderContractError, match="missing required field"):
            parse_recorder_contract(document)


def test_an_engineering_contract_carrying_a_boundary_is_refused_by_name():
    """Not "unknown field". The author is told the schema has no boundary.

    Someone reaching for ``prospective_from`` here is trying to give an
    engineering acquisition a boundary; "unrecognised key" would read as a typo
    and send them looking for the right spelling of a field that must not exist.
    """
    for field, value in (("prospective_from", None), ("boundary_rule", "anything")):
        with pytest.raises(RecorderContractError, match="carries no") as raised:
            parse_recorder_contract(engineering_payload(**{field: value}))
        assert field in str(raised.value)
        assert "absent by schema rather than null" in str(raised.value)


def test_a_null_boundary_is_refused_just_as_firmly():
    """The tempting middle case, and the one the schema exists to exclude.

    ``prospective_from: null`` is what a prospective contract looks like before
    activation, and it is one reviewed commit away from an instant. An
    engineering contract that accepted the key -- even as null -- would be that
    same contract with a promise nobody had noticed.
    """
    with pytest.raises(RecorderContractError, match="carries no"):
        parse_recorder_contract(engineering_payload(prospective_from=None))


def test_the_boundary_cannot_be_written_onto_one_afterwards():
    """The code path that writes a boundary refuses this contract outright."""
    contract = parse_recorder_contract(engineering_payload())

    with pytest.raises(ProspectiveBoundaryError, match="no prospective boundary to fix"):
        contract.with_prospective_from(MIDNIGHT)


def test_a_prospective_contract_can_still_be_activated():
    """The control: the activation path is untouched for the schema that has one."""
    contract = parse_recorder_contract(prospective_payload())
    assert contract.prospective_from is None

    activated = contract.with_prospective_from(MIDNIGHT)

    assert activated.prospective_from == MIDNIGHT
    assert activated.contract_hash != contract.contract_hash


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
def test_the_two_schemas_cannot_hash_alike():
    """The collision that would undo the whole distinction.

    An engineering contract's material has no boundary terms at all. If they
    were merely null, its material would be byte-identical to that of a
    prospective contract awaiting activation over the same streams -- and one
    storage root could then hold both identities without any file disagreeing.
    """
    engineering = parse_recorder_contract(engineering_payload(contract_id="same-id"))
    prospective = parse_recorder_contract(prospective_payload(contract_id="same-id"))

    assert engineering.contract_hash != prospective.contract_hash
    # The KEY, not the word: the prose of `coverage_rule` names the boundary
    # repeatedly, and a substring check would pass for the wrong reason.
    for field in BOUNDARY_FIELDS:
        key = f'"{field}":'
        assert key not in canonical_material(engineering)
        assert key in canonical_material(prospective)


def test_the_committed_gen3_contract_hash_is_unchanged():
    """The one number this change may not move.

    gen3 is engineering-only and unactivated, and its hash is stamped on every
    day manifest already written. A schema-dispatching parser that moved it
    would orphan that material.
    """
    contract = load_recorder_contract("btcusdt-prospective-gen3")

    assert contract.contract_hash == (
        "55268a0fe805f937824e1ae45ee3a32e6db00f0714910b31c1cc3266d63cde7c"
    )
    assert contract.is_engineering is False
    assert contract.prospective_from is None


def test_the_schema_is_part_of_the_identity():
    """Two contracts that acquire the same streams are not the same contract.

    One of them can become evidence and the other cannot, and that difference
    has to live in the hash rather than in a reader's memory.
    """
    engineering = parse_recorder_contract(engineering_payload())
    same_again = parse_recorder_contract(engineering_payload())

    assert engineering.contract_hash == same_again.contract_hash
    assert engineering.schema == ENGINEERING_CONTRACT_SCHEMA
    assert same_again.to_dict()["contract_schema"] == ENGINEERING_CONTRACT_SCHEMA


def test_the_required_sets_differ_by_exactly_the_boundary_fields():
    """Stated as an equality rather than trusted, so neither set can drift."""
    assert set(REQUIRED_FIELDS) - set(ENGINEERING_REQUIRED_FIELDS) == set(BOUNDARY_FIELDS)
    assert set(ENGINEERING_REQUIRED_FIELDS) - set(REQUIRED_FIELDS) == set()


# ---------------------------------------------------------------------------
# where the data lands, and what a manifest says about it
# ---------------------------------------------------------------------------
def test_engineering_data_never_lands_under_prospective():
    """The directory name is the first thing a reader sees.

    A path claiming a standing the data does not have is how an engineering
    recording gets cited as evidence by somebody who never opened the contract.
    """
    engineering = parse_recorder_contract(engineering_payload())
    prospective = parse_recorder_contract(prospective_payload())

    assert engineering.storage_root("/data") == (
        __import__("pathlib").Path("/data/engineering/gen4-preflight")
    )
    assert "prospective" not in engineering.storage_root("/data").parts
    assert "prospective" in prospective.storage_root("/data").parts


def test_a_manifest_tells_no_boundary_yet_from_no_boundary_ever():
    """``prospective_from: null`` and an absent key are different claims.

    A day manifest written under a prospective contract awaiting activation says
    "the boundary is not fixed". One written under an engineering contract must
    not say the same thing, because for it the boundary is never coming.
    """
    engineering = parse_recorder_contract(engineering_payload()).provenance()
    prospective = parse_recorder_contract(prospective_payload()).provenance()

    assert "prospective_from" not in engineering
    assert engineering["evidence_class"] == "DIAGNOSTIC"
    assert prospective["prospective_from"] is None
    assert "evidence_class" not in prospective


# ---------------------------------------------------------------------------
# everything else about an acquisition is still required
# ---------------------------------------------------------------------------
def test_an_engineering_contract_is_not_a_relaxed_one():
    """Having no boundary buys no other licence.

    Storage layout, checksum scheme, coverage and reconciliation rules, the
    inherited seals and the version policy are all still mandatory: what this
    schema drops is the claim to evidence, not the description of the
    acquisition.
    """
    for field in ENGINEERING_REQUIRED_FIELDS:
        if field == "contract_schema":
            continue
        document = engineering_payload()
        document.pop(field)
        with pytest.raises(RecorderContractError, match="missing required field"):
            parse_recorder_contract(document)


def test_an_unknown_field_is_still_refused():
    with pytest.raises(RecorderContractError, match="unknown field"):
        parse_recorder_contract(engineering_payload(prospective_form=None))


def test_an_unparsable_schema_is_refused_rather_than_guessed():
    with pytest.raises(RecorderContractError, match="this build parses"):
        parse_recorder_contract(engineering_payload(contract_schema="chimera.something/9"))


def test_the_inherited_seals_are_still_carried():
    """P4-HOLD and Styx stay sealed for an engineering acquisition too."""
    contract = parse_recorder_contract(engineering_payload())

    names = {name for name, _ in contract.sealed_regions_inherited}
    assert {"p4_hold", "styx"} <= names


def test_no_engineering_contract_file_is_committed_yet():
    """Deliberate, and worth pinning so a later commit has to be a decision.

    The schema is what R2 needs before it can instantiate its contract. The
    INSTANCE names an exact symbol universe, and §22 of the master roadmap lists
    the exact gen4 symbol count and universe among the decisions that must not
    be guessed before the preflight measures them. Writing one here would be
    choosing them.
    """
    import json

    from chimera.recorder.contract import CONTRACTS_DIR

    for path in sorted(CONTRACTS_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["contract_schema"] == CONTRACT_SCHEMA, path
