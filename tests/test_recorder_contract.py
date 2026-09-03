"""The gen3 prospective recorder contract, and what it refuses.

Three properties are load-bearing and each is asserted from both sides here:

* **identity is semantic.** Reformatting the file, reordering its keys, its
  streams or its markets, and rewriting its description all leave the hash
  alone. Changing what is acquired, how it is stamped, how it is stored, or
  where the prospective boundary is all move it.
* **the boundary is unset, and immutable once set.** The committed contract
  carries ``prospective_from: null``. Setting it is a pure operation that
  refuses anything but an exact UTC midnight and refuses to happen twice.
* **nothing machine-specific is in the identity.** No absolute path, no drive
  letter, no user name, no hostname — the same contract hashes the same on the
  host that records and the host that reviews.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chimera.recorder.contract import (
    CONTRACT_SCHEMA,
    CONTRACTS_DIR,
    DOCUMENTARY_FIELDS,
    GEN3_CONTRACT_ID,
    REQUIRED_FIELDS,
    STORAGE_LAYOUT_VERSION,
    ProspectiveBoundaryError,
    RecorderContractError,
    available_recorder_contract_ids,
    canonical_material,
    contract_hash,
    load_recorder_contract,
    parse_recorder_contract,
    read_recorder_contract_file,
)
from chimera.recorder.events import (
    SPOT_BOOK_TICKER,
    SPOT_KLINE_1M,
    UM_BOOK_TICKER,
    UM_FUNDING,
    UM_KLINE_1M,
    UM_MARK_PRICE,
)

CONTRACT_PATH = CONTRACTS_DIR / f"{GEN3_CONTRACT_ID}.json"
RAW = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

#: A boundary that is a legal UTC midnight. Not the real one: PR-04 fixes no
#: boundary, and every use of this below is a pure in-memory operation.
MIDNIGHT = datetime(2026, 9, 20, tzinfo=timezone.utc)


def payload(**overrides: object) -> dict[str, object]:
    """The committed document with named fields replaced."""
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    document.update(overrides)
    return document


# --- A. the committed contract ------------------------------------------------
def test_the_only_committed_contract_is_gen3():
    assert available_recorder_contract_ids() == [GEN3_CONTRACT_ID]


def test_the_committed_contract_describes_the_streams_the_plan_names():
    contract = load_recorder_contract()
    assert RAW["contract_schema"] == CONTRACT_SCHEMA
    assert set(contract.streams) == {
        UM_KLINE_1M,
        UM_MARK_PRICE,
        UM_BOOK_TICKER,
        UM_FUNDING,
        SPOT_KLINE_1M,
        SPOT_BOOK_TICKER,
    }
    assert set(contract.required_for_coverage) == set(contract.streams) - {SPOT_BOOK_TICKER}
    assert contract.market_keys() == ("spot", "um")
    assert contract.market("um").symbol == "BTCUSDT"
    assert contract.market("um").instrument == "usd-m perpetual"
    assert contract.market("spot").instrument == "spot"
    assert contract.generation == 3
    assert contract.exchange == "binance"
    assert contract.storage_layout_version == STORAGE_LAYOUT_VERSION


def test_the_committed_contract_fixes_no_prospective_boundary():
    """PR-04 commits the contract; the recorder's first run writes the boundary."""
    contract = load_recorder_contract()
    assert RAW["prospective_from"] is None
    assert contract.prospective_from is None
    assert contract.activated is False
    assert "prospective_from" in contract.boundary_rule
    assert contract.provenance()["prospective_from"] is None


def test_the_contract_restates_the_rules_rather_than_pointing_at_a_document():
    """A contract that said "see section 4.9" would not freeze anything."""
    contract = load_recorder_contract()
    for fragment in ("0.995", "1440", "30 consecutive", "RECORDER_OUTAGE"):
        assert fragment in contract.coverage_rule
    for fragment in ("data.binance.vision", "1e-9", "never modify"):
        assert fragment in contract.reconciliation_rule
    assert "source_digest" in contract.recorder_version_policy


def test_the_inherited_seals_are_named_and_not_restated():
    """One source of truth for a sealed instant, and it is not this file.

    ``nn/research_contracts`` and ``data/research/p4_holdout_ledger.json`` hold
    the gen1 boundaries. A copy of one here would be a second constant that
    agrees today and can disagree tomorrow, which is the defect the research
    contract registry already closed once.
    """
    contract = load_recorder_contract()
    assert set(dict(contract.sealed_regions_inherited)) == {"p4_hold", "styx"}
    for note in dict(contract.sealed_regions_inherited).values():
        assert not re.search(r"\d{4}-\d{2}-\d{2}T", note), (
            f"the recorder contract restates a sealed instant in {note!r}; it must point at "
            "the file that owns the boundary instead"
        )
        assert "seal" in note.lower() or "unread" in note.lower()


def test_the_file_name_and_the_declared_id_must_agree(tmp_path):
    document = payload(contract_id="something-else")
    path = tmp_path / f"{GEN3_CONTRACT_ID}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RecorderContractError, match="but is named"):
        read_recorder_contract_file(path)


# --- B. identity is semantic ---------------------------------------------------
def test_the_hash_is_sixty_four_hex_digits_and_does_not_move_between_loads():
    first = load_recorder_contract().contract_hash
    second = parse_recorder_contract(payload()).contract_hash
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert first == second == contract_hash(load_recorder_contract())


def test_canonical_material_is_the_documented_serialization():
    material = canonical_material(load_recorder_contract())
    decoded = json.loads(material)
    assert material == json.dumps(
        decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert set(decoded) == set(REQUIRED_FIELDS)
    assert "description" not in decoded


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"description": "rewritten prose, same acquisition"}, id="description"),
        pytest.param({"streams": list(reversed(RAW["streams"]))}, id="stream-order"),
        pytest.param(
            {"required_for_coverage": list(reversed(RAW["required_for_coverage"]))},
            id="required-order",
        ),
        pytest.param(
            {"markets": {k: RAW["markets"][k] for k in reversed(list(RAW["markets"]))}},
            id="market-order",
        ),
        pytest.param({"exchange": "BINANCE"}, id="exchange-case"),
        pytest.param(
            {
                "markets": {
                    **RAW["markets"],
                    "um": {**RAW["markets"]["um"], "symbol": "btcusdt"},
                }
            },
            id="symbol-case",
        ),
    ],
)
def test_presentation_does_not_change_identity(mutation):
    assert parse_recorder_contract(payload(**mutation)).contract_hash == (
        load_recorder_contract().contract_hash
    )


def test_reordering_the_keys_of_the_file_does_not_change_identity(tmp_path):
    reordered = {key: RAW[key] for key in reversed(list(RAW))}
    path = tmp_path / f"{GEN3_CONTRACT_ID}.json"
    path.write_text(json.dumps(reordered, indent=4), encoding="utf-8")
    assert read_recorder_contract_file(path).contract_hash == (
        load_recorder_contract().contract_hash
    )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"contract_id": "btcusdt-prospective-gen4"}, id="id"),
        pytest.param({"generation": 4}, id="generation"),
        pytest.param({"exchange": "bybit"}, id="exchange"),
        pytest.param({"streams": RAW["streams"][:-1]}, id="one-stream-fewer"),
        pytest.param(
            {"required_for_coverage": RAW["required_for_coverage"][:-1]},
            id="one-required-fewer",
        ),
        pytest.param(
            {"minute_key": "kline close time (ms since epoch, UTC)"}, id="minute-key"
        ),
        pytest.param({"boundary_rule": "anything goes"}, id="boundary-rule"),
        pytest.param({"checksum_scheme": "md5"}, id="checksum-scheme"),
        pytest.param({"coverage_rule": "0.900 is enough"}, id="coverage-rule"),
        pytest.param({"reconciliation_rule": "trust the recorder"}, id="reconciliation-rule"),
        pytest.param({"recorder_version_policy": "none"}, id="version-policy"),
        pytest.param(
            {"sealed_regions_inherited": {"p4_hold": "reopened"}}, id="sealed-regions"
        ),
        pytest.param(
            {
                "markets": {
                    **RAW["markets"],
                    "um": {**RAW["markets"]["um"], "symbol": "ETHUSDT"},
                }
            },
            id="symbol",
        ),
    ],
)
def test_changing_what_is_acquired_changes_identity(mutation):
    assert parse_recorder_contract(payload(**mutation)).contract_hash != (
        load_recorder_contract().contract_hash
    )


def test_to_dict_round_trips_to_the_same_identity():
    contract = load_recorder_contract()
    assert parse_recorder_contract(contract.to_dict()).contract_hash == contract.contract_hash


# --- C. the prospective boundary ----------------------------------------------
def test_setting_the_boundary_is_pure_and_moves_the_hash():
    before = CONTRACT_PATH.read_bytes()
    contract = load_recorder_contract()
    activated = contract.with_prospective_from(MIDNIGHT)

    assert activated.prospective_from == MIDNIGHT
    assert activated.activated is True
    assert contract.prospective_from is None, "the original contract was mutated"
    assert (
        activated.contract_hash != contract.contract_hash
    ), "the boundary is scientifically meaningful and must be inside the identity"
    assert CONTRACT_PATH.read_bytes() == before, "with_prospective_from wrote to disk"


def test_the_boundary_cannot_be_moved_once_set():
    activated = load_recorder_contract().with_prospective_from(MIDNIGHT)
    with pytest.raises(ProspectiveBoundaryError, match="already fixes prospective_from"):
        activated.with_prospective_from(MIDNIGHT + timedelta(days=1))


@pytest.mark.parametrize(
    "instant, expected",
    [
        pytest.param(datetime(2026, 9, 20), "no UTC offset", id="naive"),
        pytest.param(
            datetime(2026, 9, 20, 0, 0, 1, tzinfo=timezone.utc),
            "not a UTC midnight",
            id="second",
        ),
        pytest.param(
            datetime(2026, 9, 20, 12, tzinfo=timezone.utc), "not a UTC midnight", id="midday"
        ),
        pytest.param(
            datetime(2026, 9, 20, tzinfo=timezone(timedelta(hours=2))),
            "not a UTC midnight",
            id="local-midnight-elsewhere",
        ),
        pytest.param("2026-09-20T00:00:00+00:00", "must be a datetime", id="string"),
    ],
)
def test_the_boundary_refuses_anything_it_cannot_derive(instant, expected):
    with pytest.raises(ProspectiveBoundaryError, match=expected):
        load_recorder_contract().with_prospective_from(instant)


def test_an_activated_boundary_parses_back_to_the_same_instant():
    activated = load_recorder_contract().with_prospective_from(MIDNIGHT)
    reparsed = parse_recorder_contract(activated.to_dict())
    assert reparsed.prospective_from == MIDNIGHT
    assert reparsed.contract_hash == activated.contract_hash


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param("2026-09-20T00:00:00", "no UTC offset", id="naive"),
        pytest.param("2026-09-20T04:00:00+00:00", "not a UTC midnight", id="not-midnight"),
        pytest.param("not-a-date", "not an ISO-8601 instant", id="nonsense"),
        pytest.param(20260920, "must be null or an ISO-8601 string", id="integer"),
    ],
)
def test_a_file_cannot_declare_a_boundary_that_is_not_one(value, expected):
    with pytest.raises(RecorderContractError, match=expected):
        parse_recorder_contract(payload(prospective_from=value))


def test_a_midnight_written_as_z_is_the_same_boundary():
    with_offset = parse_recorder_contract(
        payload(prospective_from="2026-09-20T00:00:00+00:00")
    )
    with_zulu = parse_recorder_contract(payload(prospective_from="2026-09-20T00:00:00Z"))
    assert with_offset.contract_hash == with_zulu.contract_hash


# --- D. strict validation ------------------------------------------------------
@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_required_field_is_required(field):
    document = payload()
    document.pop(field)
    with pytest.raises(RecorderContractError, match="missing required field"):
        parse_recorder_contract(document)


def test_an_unknown_field_is_refused_rather_than_ignored():
    with pytest.raises(RecorderContractError, match="unknown field"):
        parse_recorder_contract(payload(prospective_form=None))


def test_the_documentary_field_is_the_only_optional_one():
    document = payload()
    for field in DOCUMENTARY_FIELDS:
        document.pop(field)
    assert parse_recorder_contract(document).description == ""


@pytest.mark.parametrize(
    "mutation, expected",
    [
        pytest.param(
            {"contract_schema": "chimera.recorder-contract/2"}, "contract_schema", id="schema"
        ),
        pytest.param({"timezone": "Europe/Moscow"}, "timezone", id="timezone"),
        pytest.param({"storage_layout_version": 2}, "storage_layout_version", id="layout"),
        pytest.param(
            {"storage_layout_version": 0}, "must be an integer >= 1", id="layout-zero"
        ),
        pytest.param({"generation": 0}, "must be an integer >= 1", id="generation-zero"),
        pytest.param({"generation": True}, "must be an integer >= 1", id="generation-bool"),
        pytest.param({"markets": {}}, "non-empty object", id="no-markets"),
        pytest.param({"streams": []}, "non-empty list", id="no-streams"),
        pytest.param(
            {"streams": RAW["streams"] + ["um.kline_1m"]}, "more than once", id="dup-stream"
        ),
        pytest.param(
            {"streams": ["umkline"]}, "not a <market>.<stream> id", id="no-market-prefix"
        ),
        pytest.param(
            {"streams": ["um.kline/1m"]}, "outside \\[A-Za-z0-9_\\]", id="path-separator"
        ),
        pytest.param(
            {"streams": ["um..kline"]}, "not a <market>.<stream> id", id="double-dot"
        ),
        pytest.param(
            {"streams": RAW["streams"] + ["cm.kline_1m"]},
            "name a market this contract does not declare",
            id="orphan-stream",
        ),
        pytest.param(
            {"required_for_coverage": RAW["required_for_coverage"] + ["um.trades"]},
            "which is not in streams",
            id="required-not-recorded",
        ),
        pytest.param(
            {"markets": {"um": {"symbol": "BTCUSDT", "quote": "USDT"}}},
            "must carry exactly",
            id="market-missing-field",
        ),
        pytest.param(
            {"markets": {"um": {**RAW["markets"]["um"], "tick": "0.1"}}},
            "must carry exactly",
            id="market-unknown-field",
        ),
        pytest.param(
            {"sealed_regions_inherited": []}, "must be an object", id="seals-not-object"
        ),
        pytest.param(
            {"sealed_regions_inherited": {"styx": ""}}, "non-empty note", id="seal-empty"
        ),
        pytest.param({"minute_key": "   "}, "non-empty string", id="blank-minute-key"),
        pytest.param(
            {"description": 5}, "description must be a string", id="description-type"
        ),
    ],
)
def test_a_malformed_field_is_refused(mutation, expected):
    with pytest.raises(RecorderContractError, match=expected):
        parse_recorder_contract(payload(**mutation))


def test_a_contract_that_is_not_an_object_is_refused():
    with pytest.raises(RecorderContractError, match="must be a JSON object"):
        parse_recorder_contract(["not", "a", "contract"])


def test_an_unreadable_file_is_refused(tmp_path):
    path = tmp_path / f"{GEN3_CONTRACT_ID}.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RecorderContractError, match="not readable JSON"):
        read_recorder_contract_file(path)


def test_an_unknown_contract_id_does_not_fall_back_to_the_default():
    with pytest.raises(RecorderContractError, match="unknown recorder contract"):
        load_recorder_contract("btcusdt-prospective-gen9")


def test_an_undeclared_market_is_refused():
    with pytest.raises(RecorderContractError, match="declares markets"):
        load_recorder_contract().market("cm")


# --- E. nothing machine-specific ----------------------------------------------
def test_the_identity_carries_no_machine_specific_value():
    material = canonical_material(load_recorder_contract())
    assert not re.search(r"[A-Za-z]:[\\/]", material), "a drive letter is in the identity"
    assert "\\" not in material, "a Windows path separator is in the identity"
    for token in ("/home/", "/Users/", "C:", "F:", "tmp", "Temp", "localhost"):
        assert token not in material, f"{token!r} is in the contract's identity"


def test_the_identity_does_not_depend_on_where_the_file_was_read_from(tmp_path):
    copied = tmp_path / "deeply" / "nested" / f"{GEN3_CONTRACT_ID}.json"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(CONTRACT_PATH.read_bytes())
    from_copy = read_recorder_contract_file(copied)
    assert from_copy.contract_hash == load_recorder_contract().contract_hash
    assert from_copy == load_recorder_contract(), "source is not part of equality"


def test_the_storage_root_is_injected_and_uses_forward_slashes(tmp_path):
    contract = load_recorder_contract()
    root = contract.storage_root(tmp_path / "data")
    assert root.name == "gen3"
    assert root.parent.name == "prospective"
    assert Path(contract.storage_root("data")).as_posix() == "data/prospective/gen3"


def test_the_hash_does_not_depend_on_the_file_s_newline_style(tmp_path):
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    crlf = tmp_path / f"{GEN3_CONTRACT_ID}.json"
    crlf.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert read_recorder_contract_file(crlf).contract_hash == (
        load_recorder_contract().contract_hash
    )


def test_the_label_names_the_id_and_the_short_hash():
    contract = load_recorder_contract()
    assert contract.label == f"{contract.contract_id}@{contract.contract_hash[:16]}"
    assert contract.provenance()["contract_hash"] == contract.contract_hash
    assert contract.provenance()["contract_schema"] == CONTRACT_SCHEMA
