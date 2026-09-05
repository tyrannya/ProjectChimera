"""The demo campaign configuration: what it refuses, and what its hash means.

Two properties carry the weight and each is asserted from both sides.

*Identity is semantic.* Reformatting the file, reordering its keys, changing its
indentation, saving it with CRLF, rewriting its ``description`` and reading it
from a different directory all leave ``config_hash`` alone; changing a limit, the
campaign id, the profile or freezing ``protocol_hash`` all move it. Both halves
matter: a hash that never moves proves nothing, and one that moves when the file
is merely reformatted would make two identical campaigns look like two campaigns.

*Nothing is ignored.* An unrecognised key, a missing limit, a count written as a
float and — for a campaign — a ``faults`` key at any depth are all refused, each
with a passing case beside it so the test says where the boundary is.

The committed skeleton is checked against section 7.4's proposed values and
against the two things PR-09 must not be mistaken for: it freezes no scientific
protocol, and PR-14 owns the prospective protocol and its hash.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from chimera.demo.config import (
    CONFIG_SCHEMA,
    COUNT_LIMITS,
    DOCUMENTARY_FIELDS,
    LIMIT_FIELDS,
    RATIO_LIMITS,
    REQUIRED_FIELDS,
    ConfigProfile,
    DemoConfig,
    DemoConfigError,
    canonical_material,
    config_hash,
    load_demo_config,
    parse_demo_config,
    parse_demo_limits,
    read_demo_config_file,
)

REPO = Path(__file__).resolve().parents[1]
PVC1 = REPO / "conf" / "demo" / "pvc1.json"

#: Section 7.4's proposed demo limits, written out here rather than read from the
#: file the tests are checking.
SECTION_7_4_LIMITS: dict[str, Any] = {
    "max_drawdown_pct": 0.05,
    "max_daily_loss_pct": 0.02,
    "max_open_positions": 2,
    "max_total_exposure_pct": 1.0,
    "max_exposure_per_asset_pct": 1.0,
    "max_leverage": 1.0,
    "max_orders_per_minute": 4,
    "loss_streak_limit": 3,
    "cooldown_seconds": 3600,
    "max_data_delay_s": 180,
    "max_funding_cost_rate": 0.0005,
    "funding_adverse_streak_limit": 3,
    "min_liquidation_distance_pct": 0.5,
}

A_HASH = "sha256:" + "a" * 64


def document(**overrides: Any) -> dict[str, Any]:
    """A valid campaign document. Overridden per test."""
    payload: dict[str, Any] = {
        "config_schema": CONFIG_SCHEMA,
        "campaign_id": "pvc1",
        "profile": "CAMPAIGN",
        "protocol_hash": None,
        "limits": dict(SECTION_7_4_LIMITS),
        "description": "a test document",
    }
    payload.update(overrides)
    return payload


# --- the committed skeleton -------------------------------------------------
def test_the_committed_skeleton_parses_as_a_campaign() -> None:
    config = load_demo_config(PVC1)
    assert config.campaign_id == "pvc1"
    assert config.profile is ConfigProfile.CAMPAIGN
    assert config.faults is None
    assert config.config_hash.startswith("sha256:")
    assert len(config.config_hash) == len("sha256:") + 64


def test_the_skeleton_carries_section_7_4s_proposed_limits() -> None:
    limits = load_demo_config(PVC1).limits
    for name, value in SECTION_7_4_LIMITS.items():
        assert getattr(limits, name) == pytest.approx(value), name
    assert set(SECTION_7_4_LIMITS) == set(LIMIT_FIELDS)


def test_the_skeleton_freezes_no_scientific_protocol() -> None:
    """PR-09 is plumbing. PR-14 owns the prospective protocol and its hash."""
    config = load_demo_config(PVC1)
    assert config.protocol_hash is None
    assert config.protocol_frozen is False
    raw = json.loads(PVC1.read_text(encoding="utf-8"))
    assert "protocol_hash" in raw and raw["protocol_hash"] is None
    prose = raw["description"]
    assert "PR-09 does not freeze the S2 scientific protocol" in prose
    assert "PR-14" in prose


def test_the_skeleton_carries_no_faults_key() -> None:
    def keys(node: Any) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {k for v in node.values() for k in keys(v)}
        if isinstance(node, list):
            return {k for v in node for k in keys(v)}
        return set()

    raw = json.loads(PVC1.read_text(encoding="utf-8"))
    assert not {name for name in keys(raw) if name.casefold() == "faults"}
    # And the parser would have refused it, not merely gone along with the file.
    assert load_demo_config(PVC1).faults is None


# --- identity ---------------------------------------------------------------
def test_the_hash_is_the_same_for_the_same_campaign_written_differently(
    tmp_path: Path,
) -> None:
    payload = document()
    reference = config_hash(parse_demo_config(payload))

    reordered = {key: payload[key] for key in reversed(list(payload))}
    reordered["limits"] = {
        key: payload["limits"][key] for key in reversed(list(payload["limits"]))
    }
    assert config_hash(parse_demo_config(reordered)) == reference

    # Formatting, indentation and line endings are not identity.
    pretty = tmp_path / "pretty.json"
    pretty.write_bytes(json.dumps(payload, indent=4).encode("utf-8") + b"\n")
    packed = tmp_path / "packed.json"
    packed.write_bytes(
        json.dumps(reordered, separators=(",", ":")).replace("\n", "\r\n").encode("utf-8")
    )
    assert load_demo_config(pretty).config_hash == reference
    assert load_demo_config(packed).config_hash == reference


def test_the_hash_does_not_depend_on_where_the_file_lives(tmp_path: Path) -> None:
    body = PVC1.read_bytes()
    here = tmp_path / "here" / "pvc1.json"
    there = tmp_path / "somewhere" / "else" / "renamed.json"
    for path in (here, there):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    assert load_demo_config(here).config_hash == load_demo_config(there).config_hash
    assert load_demo_config(here).config_hash == load_demo_config(PVC1).config_hash
    # And no path, host or user name is in the material at all: the only "/" it
    # holds is the one inside the schema string.
    material = canonical_material(load_demo_config(PVC1))
    assert str(tmp_path) not in material
    assert "\\" not in material
    assert material.count("/") == 1 and CONFIG_SCHEMA in material
    assert str(PVC1) not in material and PVC1.name not in material


def test_the_documentary_prose_is_not_part_of_the_identity() -> None:
    assert DOCUMENTARY_FIELDS == ("description",)
    one = parse_demo_config(document(description="one wording"))
    two = parse_demo_config(document(description="a completely different wording"))
    assert one.config_hash == two.config_hash


def test_an_integer_and_a_float_spelling_of_one_limit_are_one_identity() -> None:
    limits = dict(SECTION_7_4_LIMITS)
    limits["max_leverage"] = 1
    assert (
        parse_demo_config(document(limits=limits)).config_hash
        == parse_demo_config(document()).config_hash
    )


@pytest.mark.parametrize(
    "change",
    [
        {"campaign_id": "pvc2"},
        {"protocol_hash": A_HASH},
        {"limits": {**SECTION_7_4_LIMITS, "max_drawdown_pct": 0.06}},
        {"limits": {**SECTION_7_4_LIMITS, "max_open_positions": 3}},
    ],
)
def test_a_semantically_meaningful_change_moves_the_hash(change: dict[str, Any]) -> None:
    reference = parse_demo_config(document()).config_hash
    assert parse_demo_config(document(**change)).config_hash != reference


def test_the_profile_is_part_of_the_identity() -> None:
    campaign = parse_demo_config(document())
    soak = parse_demo_config(document(profile="SOAK"), expected_profile=ConfigProfile.SOAK)
    assert campaign.config_hash != soak.config_hash


def test_a_configured_fault_schedule_is_in_the_identity() -> None:
    """A soak run's identity records that faults were configured."""
    without = parse_demo_config(document(profile="SOAK"), expected_profile=ConfigProfile.SOAK)
    with_faults = parse_demo_config(
        document(profile="SOAK", faults={"drop_quote": [7]}),
        expected_profile=ConfigProfile.SOAK,
    )
    assert without.config_hash != with_faults.config_hash
    assert with_faults.faults == {"drop_quote": [7]}


def test_the_hash_form_is_sha256_prefixed_hex() -> None:
    value = parse_demo_config(document()).config_hash
    assert value.startswith("sha256:")
    digest = value[len("sha256:") :]
    assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")


# --- the faults refusal -----------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        document(faults={"drop_quote": [7]}),
        document(limits={**SECTION_7_4_LIMITS, "faults": {}}),
        document(protocol_hash=None, description={"faults": {}}),
        document(limits={**SECTION_7_4_LIMITS, "extra": [{"nested": {"faults": 1}}]}),
        document(FAULTS={"drop_quote": [7]}),
        document(Faults={}),
    ],
)
def test_a_campaign_configuration_refuses_a_faults_key_at_any_depth(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(DemoConfigError, match="carries no 'faults' key"):
        parse_demo_config(payload)


def test_the_faults_refusal_names_where_it_found_the_key() -> None:
    with pytest.raises(DemoConfigError, match=r"limits\.extra\[0\]\.nested\.faults"):
        parse_demo_config(
            document(limits={**SECTION_7_4_LIMITS, "extra": [{"nested": {"faults": 1}}]})
        )


def test_a_soak_or_test_configuration_may_carry_one() -> None:
    for profile in (ConfigProfile.SOAK, ConfigProfile.TEST):
        config = parse_demo_config(
            document(profile=profile.value, faults={"forced_restart": [11]}),
            expected_profile=profile,
        )
        assert config.profile is profile
        assert config.faults == {"forced_restart": [11]}


def test_reaching_the_fault_schema_takes_a_deliberate_act_on_both_sides() -> None:
    # The file says CAMPAIGN, so a soak parse refuses it.
    with pytest.raises(DemoConfigError, match="declares profile CAMPAIGN"):
        parse_demo_config(document(), expected_profile=ConfigProfile.SOAK)
    # The file says SOAK, so the default campaign parse refuses it.
    with pytest.raises(DemoConfigError):
        parse_demo_config(document(profile="SOAK"))


def test_a_faults_block_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(DemoConfigError, match="must be a JSON object"):
        parse_demo_config(
            document(profile="SOAK", faults=["drop_quote"]),
            expected_profile=ConfigProfile.SOAK,
        )


# --- strict parsing ---------------------------------------------------------
def test_an_unknown_top_level_key_is_refused_not_ignored() -> None:
    with pytest.raises(DemoConfigError, match=r"unknown field\(s\) \['stake_amount'\]"):
        parse_demo_config(document(stake_amount=200))


def test_an_unknown_limit_is_refused_not_ignored() -> None:
    """The failure this prevents: a misspelled bound silently not in force."""
    with pytest.raises(DemoConfigError, match="max_drawdown_pctt"):
        parse_demo_config(document(limits={**SECTION_7_4_LIMITS, "max_drawdown_pctt": 0.05}))


@pytest.mark.parametrize("missing", REQUIRED_FIELDS)
def test_every_required_field_is_required(missing: str) -> None:
    payload = document()
    del payload[missing]
    with pytest.raises(DemoConfigError, match="missing required field"):
        parse_demo_config(payload)


@pytest.mark.parametrize("missing", LIMIT_FIELDS)
def test_every_limit_is_required(missing: str) -> None:
    limits = dict(SECTION_7_4_LIMITS)
    del limits[missing]
    with pytest.raises(DemoConfigError, match="is missing"):
        parse_demo_config(document(limits=limits))


def test_the_description_is_optional_and_the_hash_is_unaffected() -> None:
    payload = document()
    del payload["description"]
    assert parse_demo_config(payload).description == ""
    assert parse_demo_config(payload).config_hash == parse_demo_config(document()).config_hash


@pytest.mark.parametrize("name", COUNT_LIMITS)
def test_a_count_written_as_a_float_is_refused(name: str) -> None:
    limits = dict(SECTION_7_4_LIMITS)
    # Exactly at, as an integer: accepted.
    limits[name] = 2
    assert parse_demo_config(document(limits=limits))
    # The same value as a float: refused, because 2 and 2.0 would be two
    # identities for one campaign.
    limits[name] = 2.0
    with pytest.raises(DemoConfigError, match="must be an integer"):
        parse_demo_config(document(limits=limits))


@pytest.mark.parametrize("name", COUNT_LIMITS)
def test_a_count_below_one_is_refused(name: str) -> None:
    limits = dict(SECTION_7_4_LIMITS)
    limits[name] = 1
    assert parse_demo_config(document(limits=limits))
    limits[name] = 0
    with pytest.raises(DemoConfigError, match="not a limit"):
        parse_demo_config(document(limits=limits))


@pytest.mark.parametrize("bad", [0, -0.01, float("nan"), float("inf"), True, "0.05", None])
def test_a_ratio_limit_that_is_not_a_positive_finite_number_is_refused(bad: Any) -> None:
    limits = dict(SECTION_7_4_LIMITS)
    limits["max_drawdown_pct"] = bad
    with pytest.raises(DemoConfigError):
        parse_demo_config(document(limits=limits))
    # Just above zero is a limit, however small.
    limits["max_drawdown_pct"] = 1e-9
    assert parse_demo_config(document(limits=limits))


def test_the_limit_names_partition_into_counts_and_ratios() -> None:
    assert set(COUNT_LIMITS) | set(RATIO_LIMITS) == set(LIMIT_FIELDS)
    assert not set(COUNT_LIMITS) & set(RATIO_LIMITS)


def test_a_schema_this_build_does_not_know_is_refused() -> None:
    with pytest.raises(DemoConfigError, match="config_schema"):
        parse_demo_config(document(config_schema="chimera.demo-campaign-config/2"))


@pytest.mark.parametrize("bad", ["", "PVC1", "pvc 1", "pvc/1", "a" * 65, 1, None])
def test_a_campaign_id_that_could_not_name_a_directory_is_refused(bad: Any) -> None:
    with pytest.raises(DemoConfigError, match="campaign_id"):
        parse_demo_config(document(campaign_id=bad))


def test_a_campaign_id_of_the_permitted_alphabet_is_accepted() -> None:
    assert parse_demo_config(document(campaign_id="pvc-1_v2")).campaign_id == "pvc-1_v2"


@pytest.mark.parametrize("bad", ["a" * 64, "sha256:" + "A" * 64, "sha256:", 1, "sha256:zz"])
def test_a_protocol_hash_that_is_not_the_one_form_is_refused(bad: Any) -> None:
    with pytest.raises(DemoConfigError, match="protocol_hash"):
        parse_demo_config(document(protocol_hash=bad))


def test_a_frozen_protocol_hash_is_accepted_and_says_so() -> None:
    config = parse_demo_config(document(protocol_hash=A_HASH))
    assert config.protocol_hash == A_HASH
    assert config.protocol_frozen is True


@pytest.mark.parametrize("bad", ["OPERATOR", "campaign", "", None])
def test_an_unknown_profile_is_refused(bad: Any) -> None:
    with pytest.raises(DemoConfigError, match="profile"):
        parse_demo_config(document(profile=bad))


def test_a_document_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DemoConfigError, match="must be a JSON object"):
        parse_demo_config([1, 2, 3])  # type: ignore[arg-type]
    broken = tmp_path / "broken.json"
    broken.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(DemoConfigError, match="must hold a JSON object"):
        load_demo_config(broken)
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(DemoConfigError, match="not valid JSON"):
        load_demo_config(broken)
    with pytest.raises(DemoConfigError, match="could not read"):
        load_demo_config(tmp_path / "absent.json")


def test_a_config_file_is_read_as_utf8_whatever_the_host_default(tmp_path: Path) -> None:
    payload = document(description="é中")
    path = tmp_path / "utf8.json"
    path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    assert load_demo_config(path).description == "é中"


# --- the campaign/faults rule belongs to the type, not only to the parser ----
def limits() -> Any:
    """Section 7.4's limits as the parsed object, for building a config by hand."""
    return parse_demo_limits(dict(SECTION_7_4_LIMITS))


def test_a_campaign_config_built_without_the_parser_still_refuses_faults() -> None:
    """The rule PR-10 has to be able to rely on, held by the type it is handed.

    ``parse_demo_config`` is not the only way a ``DemoConfig`` comes into being:
    a fixture, a CLI override merge or a ``dataclasses.replace`` that changes one
    field of a soak configuration all bypass it. Without this the campaign whose
    evidence was produced under an injected fault schedule would have a
    well-formed ``config_hash`` and nothing would say so.
    """
    with pytest.raises(DemoConfigError, match="carries no 'faults'"):
        DemoConfig(
            campaign_id="pvc1",
            profile=ConfigProfile.CAMPAIGN,
            protocol_hash=None,
            limits=limits(),
            faults={"drop_quote": [7]},
        )


def test_replacing_the_profile_of_a_soak_config_cannot_smuggle_a_fault_schedule() -> None:
    soak = parse_demo_config(
        document(profile="SOAK", faults={"drop_quote": [7]}),
        expected_profile=ConfigProfile.SOAK,
    )
    with pytest.raises(DemoConfigError, match="carries no 'faults'"):
        dataclasses.replace(soak, profile=ConfigProfile.CAMPAIGN)
    # Dropping the schedule at the same time is what makes it a campaign.
    promoted = dataclasses.replace(soak, profile=ConfigProfile.CAMPAIGN, faults=None)
    assert promoted.faults is None
    assert promoted.config_hash != soak.config_hash


def test_a_soak_config_built_by_hand_carries_its_schedule_into_the_hash() -> None:
    """The other side of the rule: outside a campaign the block is ordinary."""
    without = DemoConfig(
        campaign_id="pvc1",
        profile=ConfigProfile.SOAK,
        protocol_hash=None,
        limits=limits(),
    )
    with_faults = DemoConfig(
        campaign_id="pvc1",
        profile=ConfigProfile.SOAK,
        protocol_hash=None,
        limits=limits(),
        faults={"drop_quote": [7]},
    )
    assert with_faults.faults == {"drop_quote": [7]}
    assert with_faults.config_hash != without.config_hash
    assert "faults" in canonical_material(with_faults)
    assert "faults" not in canonical_material(without)


def test_a_campaign_config_built_by_hand_without_faults_is_fine() -> None:
    config = DemoConfig(
        campaign_id="pvc1",
        profile=ConfigProfile.CAMPAIGN,
        protocol_hash=None,
        limits=limits(),
    )
    assert config.faults is None
    assert config.config_hash == config_hash(config)


# --- what the file reader refuses -------------------------------------------
@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_a_config_file_carrying_a_non_json_token_is_refused(
    tmp_path: Path, token: str
) -> None:
    """Python's decoder accepts these bare tokens and no other reader does."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps(document())[:-1] + f', "extra": {token}}}', encoding="utf-8")
    with pytest.raises(DemoConfigError, match="not valid JSON"):
        read_demo_config_file(path)
    # The same file with a finite number parses, and is then refused for the
    # reason a reader would expect: an unknown field.
    path.write_text(json.dumps(document())[:-1] + ', "extra": 1}', encoding="utf-8")
    assert read_demo_config_file(path)["extra"] == 1
    with pytest.raises(DemoConfigError, match=r"unknown field\(s\) \['extra'\]"):
        load_demo_config(path)


def test_a_repeated_key_is_refused_rather_than_silently_resolved(tmp_path: Path) -> None:
    """A reader sees the first value and Python's decoder keeps the last."""
    path = tmp_path / "c.json"
    body = json.dumps(document())
    doubled = body[:-1] + ', "campaign_id": "somethingelse"}'
    path.write_text(doubled, encoding="utf-8")
    assert json.loads(doubled)["campaign_id"] == "somethingelse"
    with pytest.raises(DemoConfigError, match="appears more than once"):
        read_demo_config_file(path)
    # Written once, the same file loads and the value is unambiguous.
    path.write_text(body, encoding="utf-8")
    assert read_demo_config_file(path)["campaign_id"] == "pvc1"


def test_a_faults_block_that_cannot_be_hashed_fails_when_the_file_is_read() -> None:
    """Its identity is its hash, so a block with no canonical form has none."""
    with pytest.raises(DemoConfigError, match="cannot be serialised canonically"):
        parse_demo_config(
            document(profile="SOAK", faults={"drop_quote": {1, 2}}),
            expected_profile=ConfigProfile.SOAK,
        )
    # A block made of JSON values hashes, and the hash moves with its contents.
    one = parse_demo_config(
        document(profile="SOAK", faults={"drop_quote": [7]}),
        expected_profile=ConfigProfile.SOAK,
    )
    two = parse_demo_config(
        document(profile="SOAK", faults={"drop_quote": [8]}),
        expected_profile=ConfigProfile.SOAK,
    )
    assert one.config_hash != two.config_hash
