"""R1-o: §37.1's classes as identifiers, and the three refusals they buy.

The checks here are about CITATION, which is what §37.1 actually asks for: a
front-door document may not cite a non-PROSPECTIVE artifact as prospective
evidence, nor a non-CONFIRMATORY one as confirmatory evidence, nor a block the
verifier marked INVALID as a result. Every one of them is two-sided, because the
dangerous failure is not a missed violation -- it is a check so eager that a true
sentence trips it, after which the check gets loosened until it catches nothing.

That risk is concrete and measured, not hypothetical. These documents discuss
these words constantly and mostly to DENY them:

* ``README.md``: "no prospective evidence exists"
* ``docs/current_development_plan.md``: "never as pristine prospective evidence"
* ``docs/p4_preregistration.md``: "cannot produce confirmatory evidence"

All three are true, and a guard that fired on the phrase alone would forbid the
truth along with the error. So the real sentences get their own regression test
below, by content rather than by hope.

The vocabulary test compares the code against §37.1's own table in
``docs/master_roadmap_r0_r18.md``. That is not the tautology this repository
guards against elsewhere -- the document and the module are two independent
artifacts, and the point is that editing one without the other fails here.
Nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nn.evidence_class import (
    ADAPTIVE,
    BLOCK_STATUS_KEY,
    BURNED,
    CAMPAIGN_OUTCOMES,
    CONFIRMATORY,
    DECIDING_CLASSES,
    DIAGNOSTIC,
    ENGINEERING,
    EVIDENCE_CLASS_KEY,
    EXPLORATORY,
    FREEZE_CLASSES,
    GOVERNANCE_CLASSES,
    INVALID,
    PREREGISTERED,
    PROSPECTIVE,
    RECORDER_CLASS_EQUIVALENTS,
    SEALED_DIAGNOSTIC,
    VALID,
    citation_problems,
    declared_block_status,
    declared_class,
    deciding_artifacts,
    is_deciding,
    normalise,
    undeclared_artifacts,
)

#: Every artifact directory that exists at this commit and declares no §37.1
#: class. This is a GRANDFATHERING list, generated from the tree rather than
#: written from memory, and its job is to make the requirement enforceable
#: without retro-classifying anything: a directory here may stay undeclared,
#: and any directory NOT here must declare. R1 may not touch a scientific
#: artifact, and choosing between BURNED, ADAPTIVE and EXPLORATORY for a
#: committed run is a scientific judgement, not an engineering one -- so the
#: 113 below are left exactly as their runs wrote them.
#:
#: The list may only SHRINK. An entry that has since started declaring a class
#: fails the guard below, so classifying one is a change to this list too, and
#: nobody can add a new unclassified artifact by appending to it quietly --
#: appending is the reviewable act.
UNDECLARED_AT_R1O: frozenset[str] = frozenset(
    {
        "artifacts/benchmark",
        "artifacts/benchmark/btc_p13_a2r2_source_acquisition",
        "artifacts/benchmark/btc_p13_carry",
        "artifacts/benchmark/btc_p2a_comparison",
        "artifacts/benchmark/btc_p2a_seed_142",
        "artifacts/benchmark/btc_p2a_seed_242",
        "artifacts/benchmark/btc_p2a_seed_342",
        "artifacts/benchmark/btc_p2a_seed_42",
        "artifacts/benchmark/btc_p2a_seed_442",
        "artifacts/benchmark/btc_p2b_ablation_xgboost",
        "artifacts/benchmark/btc_p2b_comparison",
        "artifacts/benchmark/btc_p2b_ohlcv14_lightgbm",
        "artifacts/benchmark/btc_p2b_ohlcv14_logistic_regression",
        "artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_lightgbm",
        "artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_logistic_regression",
        "artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_minus_breaks_xgboost",
        "artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_minus_displacement_xgboost",
        "artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_minus_fvg_xgboost",
        "artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_minus_liquidity_xgboost",
        "artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_minus_structure_xgboost",
        "artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_minus_sweeps_xgboost",
        "artifacts/benchmark/btc_p2b_ohlcv14_plus_smc_v1_xgboost",
        "artifacts/benchmark/btc_p2b_ohlcv14_xgboost",
        "artifacts/benchmark/btc_p2b_regimes",
        "artifacts/benchmark/btc_p2b_repro_ohlcv14_plus_smc_v1_xgboost",
        "artifacts/benchmark/btc_p2b_smc_v1_lightgbm",
        "artifacts/benchmark/btc_p2b_smc_v1_logistic_regression",
        "artifacts/benchmark/btc_p2b_smc_v1_xgboost",
        "artifacts/benchmark/btc_p2c_chart_structure_v1_lightgbm",
        "artifacts/benchmark/btc_p2c_chart_structure_v1_logistic_regression",
        "artifacts/benchmark/btc_p2c_chart_structure_v1_xgboost",
        "artifacts/benchmark/btc_p2c_comparison",
        "artifacts/benchmark/btc_p2c_ohlcv14_lightgbm",
        "artifacts/benchmark/btc_p2c_ohlcv14_logistic_regression",
        "artifacts/benchmark/btc_p2c_ohlcv14_plus_chart_structure_v1_lightgbm",
        "artifacts/benchmark/btc_p2c_ohlcv14_plus_chart_structure_v1_logistic_regression",
        "artifacts/benchmark/btc_p2c_ohlcv14_plus_chart_structure_v1_xgboost",
        "artifacts/benchmark/btc_p2c_ohlcv14_xgboost",
        "artifacts/benchmark/btc_p3_comparison",
        "artifacts/benchmark/btc_p3_microstructure_v1_lightgbm",
        "artifacts/benchmark/btc_p3_microstructure_v1_logistic_regression",
        "artifacts/benchmark/btc_p3_microstructure_v1_xgboost",
        "artifacts/benchmark/btc_p3_ohlcv14_lightgbm",
        "artifacts/benchmark/btc_p3_ohlcv14_logistic_regression",
        "artifacts/benchmark/btc_p3_ohlcv14_plus_microstructure_v1_lightgbm",
        "artifacts/benchmark/btc_p3_ohlcv14_plus_microstructure_v1_logistic_regression",
        "artifacts/benchmark/btc_p3_ohlcv14_plus_microstructure_v1_xgboost",
        "artifacts/benchmark/btc_p3_ohlcv14_xgboost",
        "artifacts/benchmark/btc_p4_comparison",
        "artifacts/benchmark/btc_p4_derivatives_v1_lightgbm",
        "artifacts/benchmark/btc_p4_derivatives_v1_logistic_regression",
        "artifacts/benchmark/btc_p4_derivatives_v1_xgboost",
        "artifacts/benchmark/btc_p4_ohlcv14_lightgbm",
        "artifacts/benchmark/btc_p4_ohlcv14_logistic_regression",
        "artifacts/benchmark/btc_p4_ohlcv14_plus_derivatives_v1_lightgbm",
        "artifacts/benchmark/btc_p4_ohlcv14_plus_derivatives_v1_logistic_regression",
        "artifacts/benchmark/btc_p4_ohlcv14_plus_derivatives_v1_xgboost",
        "artifacts/benchmark/btc_p4_ohlcv14_xgboost",
        "artifacts/benchmark/btc_p4_stage1",
        "artifacts/benchmark/btc_p5_comparison",
        "artifacts/benchmark/btc_p5_decision",
        "artifacts/benchmark/btc_p5_mtf_v1_lightgbm",
        "artifacts/benchmark/btc_p5_mtf_v1_logistic_regression",
        "artifacts/benchmark/btc_p5_mtf_v1_xgboost",
        "artifacts/benchmark/btc_p5_ohlcv14_lightgbm",
        "artifacts/benchmark/btc_p5_ohlcv14_logistic_regression",
        "artifacts/benchmark/btc_p5_ohlcv14_plus_mtf_v1_lightgbm",
        "artifacts/benchmark/btc_p5_ohlcv14_plus_mtf_v1_logistic_regression",
        "artifacts/benchmark/btc_p5_ohlcv14_plus_mtf_v1_xgboost",
        "artifacts/benchmark/btc_p5_ohlcv14_xgboost",
        "artifacts/benchmark/btc_p6_15m_lightgbm",
        "artifacts/benchmark/btc_p6_15m_logistic_regression",
        "artifacts/benchmark/btc_p6_15m_xgboost",
        "artifacts/benchmark/btc_p6_1h_lightgbm",
        "artifacts/benchmark/btc_p6_1h_logistic_regression",
        "artifacts/benchmark/btc_p6_1h_xgboost",
        "artifacts/benchmark/btc_p6_1m_lightgbm",
        "artifacts/benchmark/btc_p6_1m_logistic_regression",
        "artifacts/benchmark/btc_p6_1m_xgboost",
        "artifacts/benchmark/btc_p6_30m_lightgbm",
        "artifacts/benchmark/btc_p6_30m_logistic_regression",
        "artifacts/benchmark/btc_p6_30m_xgboost",
        "artifacts/benchmark/btc_p6_5m_lightgbm",
        "artifacts/benchmark/btc_p6_5m_logistic_regression",
        "artifacts/benchmark/btc_p6_5m_xgboost",
        "artifacts/benchmark/btc_p6_decision",
        "artifacts/benchmark/btc_p6ext_1d_lightgbm",
        "artifacts/benchmark/btc_p6ext_1d_logistic_regression",
        "artifacts/benchmark/btc_p6ext_1d_xgboost",
        "artifacts/benchmark/btc_p6ext_4h_lightgbm",
        "artifacts/benchmark/btc_p6ext_4h_logistic_regression",
        "artifacts/benchmark/btc_p6ext_4h_xgboost",
        "artifacts/benchmark/btc_p6ext_decision",
        "artifacts/benchmark/btc_p7_day_trading",
        "artifacts/benchmark/btc_p7_decision",
        "artifacts/benchmark/btc_p7_scalping",
        "artifacts/diagnostics",
        "artifacts/diagnostics/btc_regimes_v2",
        "artifacts/diagnostics/btc_regimes_v3",
        "artifacts/diagnostics/btc_regimes_v4",
        "artifacts/futures_dry_run_v1",
        "artifacts/paper_smoke",
        "artifacts/walkforward",
        "artifacts/walkforward/btc_nested_seed_142",
        "artifacts/walkforward/btc_nested_seed_242",
        "artifacts/walkforward/btc_nested_seed_342",
        "artifacts/walkforward/btc_nested_seed_442",
        "artifacts/walkforward/btc_nested_v1",
        "artifacts/walkforward/btc_nested_v4_seed_142",
        "artifacts/walkforward/btc_nested_v4_seed_242",
        "artifacts/walkforward/btc_nested_v4_seed_342",
        "artifacts/walkforward/btc_nested_v4_seed_42",
        "artifacts/walkforward/btc_nested_v4_seed_442",
    }
)

REPO = Path(__file__).resolve().parents[1]
ROADMAP = REPO / "docs/master_roadmap_r0_r18.md"
DOCS = ("doc.md",)


def write_artifact(root: Path, relative: str, payload: dict[str, object]) -> Path:
    """One artifact directory carrying one JSON, under ``root``."""
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aggregate.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def write_doc(root: Path, body: str) -> None:
    (root / "doc.md").write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------- #
# the vocabulary, against the roadmap's own table
# --------------------------------------------------------------------------- #


def test_the_nine_classes_are_exactly_section_37_1s_table() -> None:
    """Two independent artifacts, compared: editing one alone fails here."""
    text = ROADMAP.read_text(encoding="utf-8")
    section = text.split("### 37.1", 1)[1].split("### 37.2", 1)[0]
    rows = re.findall(r"^\| ([A-Z][A-Z-]+) \|", section, re.MULTILINE)
    assert len(rows) == 9, f"parsed {len(rows)} class rows from 37.1, not 9: {rows}"
    assert tuple(rows) == GOVERNANCE_CLASSES


def test_the_block_and_campaign_labels_are_the_ones_37_1_names() -> None:
    section = ROADMAP.read_text(encoding="utf-8").split("### 37.1", 1)[1]
    section = section.split("### 37.2", 1)[0]
    for label in (*(VALID, INVALID), *CAMPAIGN_OUTCOMES):
        spelled = label.replace("_", " ")
        assert spelled in section, f"37.1 does not name {spelled}"


def test_only_two_classes_may_decide() -> None:
    assert DECIDING_CLASSES == (PROSPECTIVE, CONFIRMATORY)
    for label in GOVERNANCE_CLASSES:
        assert is_deciding(label) is (label in DECIDING_CLASSES), label
    assert not is_deciding(None)


# --------------------------------------------------------------------------- #
# normalise: both sides of every rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", GOVERNANCE_CLASSES)
def test_a_canonical_label_is_itself(label: str) -> None:
    assert normalise(label) == label
    assert normalise(f"  {label}  ") == label


def test_the_recorders_lowercase_engineering_is_the_governance_class() -> None:
    assert RECORDER_CLASS_EQUIVALENTS == {"engineering": ENGINEERING}
    assert normalise("engineering") == ENGINEERING
    assert normalise("ENGINEERING") == ENGINEERING


def test_a_lowercase_deciding_class_is_refused() -> None:
    """The deliberate asymmetry, and the reason R1-o is worth having.

    ``chimera/recorder/health.py`` writes ``prospective`` to mean "this feed is
    past its boundary". If that spelling also promoted an artifact into evidence
    a real-money decision may rest on, the most consequential label in the
    programme would be reachable by a lowercase string that already exists in the
    tree for an unrelated purpose.
    """
    assert normalise("prospective") is None
    assert normalise("confirmatory") is None
    assert normalise(PROSPECTIVE) == PROSPECTIVE


@pytest.mark.parametrize("value", FREEZE_CLASSES)
def test_the_freezers_values_are_not_governance_classes(value: str) -> None:
    """``primary``/``derived`` answer a different question entirely."""
    assert normalise(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "one native-timeframe specialist, fitted once; P6 primary evidence",
        "exploratory screen on burned blocks; not a research result",
        "operational",
        "",
        None,
        17,
        ["PROSPECTIVE"],
    ],
)
def test_prose_and_non_strings_declare_nothing(value: object) -> None:
    """The 91 committed artifacts that carry free text read as undeclared.

    Undeclared is the safe answer and the honest one: R1 may not retro-classify
    a scientific artifact, and every refusal here only needs to know that
    something is not a deciding class.
    """
    assert normalise(value) is None


# --------------------------------------------------------------------------- #
# declared_class, over a directory
# --------------------------------------------------------------------------- #


def test_a_directory_declares_what_its_json_says(tmp_path: Path) -> None:
    directory = write_artifact(tmp_path, "artifacts/x", {EVIDENCE_CLASS_KEY: DIAGNOSTIC})
    assert declared_class(directory) == DIAGNOSTIC


@pytest.mark.parametrize(
    "payload",
    [
        {EVIDENCE_CLASS_KEY: "primary"},
        {EVIDENCE_CLASS_KEY: "prospective"},
        {"something_else": PROSPECTIVE},
        {},
    ],
)
def test_a_directory_that_declares_no_governance_class(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    directory = write_artifact(tmp_path, "artifacts/x", payload)
    assert declared_class(directory) is None


def test_malformed_and_missing_declare_nothing(tmp_path: Path) -> None:
    """A check over a tree someone else wrote does not crash on a bad file."""
    broken = tmp_path / "artifacts/broken"
    broken.mkdir(parents=True)
    (broken / "a.json").write_text("{not json", encoding="utf-8")
    assert declared_class(broken) is None
    assert declared_class(tmp_path / "artifacts/absent") is None

    empty = tmp_path / "artifacts/empty"
    empty.mkdir(parents=True)
    assert declared_class(empty) is None


# --------------------------------------------------------------------------- #
# the premise: nothing in this repository is a deciding class, and why
# --------------------------------------------------------------------------- #


def test_no_artifact_in_this_repository_declares_a_deciding_class() -> None:
    assert deciding_artifacts(REPO) == {}


def test_the_premise_is_checked_rather_than_assumed() -> None:
    """Nothing can honestly be PROSPECTIVE because no boundary was ever activated.

    Asserted from the committed contract rather than stated in a comment: the
    emptiness above is a consequence of this, and if a boundary is ever fixed
    this test is the one that says the reasoning has moved on.
    """
    contracts = sorted((REPO / "chimera/recorder/contracts").glob("*.json"))
    assert contracts, "no recorder contract found -- the premise cannot be checked"
    for path in contracts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("prospective_from") is None, path.name


def test_a_declared_deciding_artifact_is_found_when_one_exists(tmp_path: Path) -> None:
    """The positive control. An empty result from a walker that found nothing
    would satisfy the test above while proving nothing at all."""
    write_artifact(tmp_path, "artifacts/benchmark/real", {EVIDENCE_CLASS_KEY: PROSPECTIVE})
    write_artifact(tmp_path, "artifacts/benchmark/other", {EVIDENCE_CLASS_KEY: BURNED})
    assert deciding_artifacts(tmp_path) == {"artifacts/benchmark/real": PROSPECTIVE}


# --------------------------------------------------------------------------- #
# refusal 1 and 2: prospective and confirmatory citations
# --------------------------------------------------------------------------- #


def test_the_real_tree_has_no_citation_problem() -> None:
    """The load-bearing true negative: the repository as it stands passes."""
    from nn.research_state import FRONT_DOOR_DOCUMENTS

    assert citation_problems(REPO, FRONT_DOOR_DOCUMENTS) == []


@pytest.mark.parametrize(
    "claim, wanted",
    [
        ("artifacts/x/a.json is prospective evidence for the carry lane.", "prospective"),
        ("The gate rests on artifacts/x as prospective evidence.", "prospective"),
        ("artifacts/x/a.json is confirmatory evidence of the first period.", "confirmatory"),
        ("Read artifacts/x as confirmatory evidence.", "confirmatory"),
    ],
)
def test_citing_an_undeclared_artifact_as_deciding_evidence_is_refused(
    tmp_path: Path, claim: str, wanted: str
) -> None:
    write_artifact(tmp_path, "artifacts/x", {EVIDENCE_CLASS_KEY: "primary"})
    write_doc(tmp_path, claim)
    problems = citation_problems(tmp_path, DOCS)
    assert len(problems) == 1, problems
    assert f"as {wanted} evidence" in problems[0]
    assert "no governance class" in problems[0]


@pytest.mark.parametrize(
    "declared, claim",
    [
        (PROSPECTIVE, "artifacts/x/a.json is prospective evidence."),
        (CONFIRMATORY, "artifacts/x/a.json is confirmatory evidence."),
    ],
)
def test_citing_a_correctly_declared_artifact_is_allowed(
    tmp_path: Path, declared: str, claim: str
) -> None:
    """The other side. A check that refused the true citation too would make the
    class system unusable the moment real prospective evidence exists."""
    write_artifact(tmp_path, "artifacts/x", {EVIDENCE_CLASS_KEY: declared})
    write_doc(tmp_path, claim)
    assert citation_problems(tmp_path, DOCS) == []


def test_citing_an_artifact_as_the_WRONG_deciding_class_is_refused(tmp_path: Path) -> None:
    """PROSPECTIVE is not CONFIRMATORY: a second period is a separate fact."""
    write_artifact(tmp_path, "artifacts/x", {EVIDENCE_CLASS_KEY: PROSPECTIVE})
    write_doc(tmp_path, "artifacts/x/a.json is confirmatory evidence.")
    problems = citation_problems(tmp_path, DOCS)
    assert len(problems) == 1
    assert "declares PROSPECTIVE" in problems[0]


@pytest.mark.parametrize(
    "line",
    [
        "artifacts/x is never prospective evidence.",
        "artifacts/x may not be cited as prospective evidence.",
        "artifacts/x cannot produce confirmatory evidence.",
        "No part of artifacts/x is prospective evidence.",
        "artifacts/x is a non-PROSPECTIVE artifact, so it is not prospective evidence.",
        "The verifier refuses artifacts/x as prospective evidence.",
    ],
)
def test_a_denial_is_not_a_claim(tmp_path: Path, line: str) -> None:
    """Every one of these is TRUE today, and a guard that refused them would be
    loosened until it refused nothing."""
    write_artifact(tmp_path, "artifacts/x", {})
    write_doc(tmp_path, line)
    assert citation_problems(tmp_path, DOCS) == []


def test_the_sentences_the_real_documents_actually_contain_still_pass() -> None:
    """A regression guard on the measured phrases, not on my memory of them."""
    from nn.research_state import FRONT_DOOR_DOCUMENTS

    phrases = {
        "README.md": "no prospective evidence exists",
        "docs/current_development_plan.md": "never as pristine prospective evidence",
        "docs/p4_preregistration.md": "cannot produce confirmatory evidence",
    }
    for name, phrase in phrases.items():
        text = (REPO / name).read_text(encoding="utf-8")
        assert phrase in " ".join(text.split()), f"{name} no longer says {phrase!r}"
    assert citation_problems(REPO, FRONT_DOOR_DOCUMENTS) == []


def test_a_line_with_a_claim_but_no_artifact_path_is_not_a_citation(tmp_path: Path) -> None:
    """§37.1 asks about citing an ARTIFACT. Governance prose that discusses the
    class without naming a directory is not a citation of anything."""
    write_doc(tmp_path, "Only a second period is confirmatory evidence under this rule.")
    assert citation_problems(tmp_path, DOCS) == []


# --------------------------------------------------------------------------- #
# refusal 3: an INVALID block cited as a result
# --------------------------------------------------------------------------- #


def test_an_invalid_block_cited_as_a_result_is_refused(tmp_path: Path) -> None:
    write_artifact(tmp_path, "artifacts/campaign/block_03", {BLOCK_STATUS_KEY: INVALID})
    write_doc(tmp_path, "artifacts/campaign/block_03 is the result of the third week.")
    problems = citation_problems(tmp_path, DOCS)
    assert len(problems) == 1, problems
    assert "marked that block INVALID" in problems[0]
    assert "never a result" in problems[0]


@pytest.mark.parametrize(
    "payload",
    [
        {BLOCK_STATUS_KEY: VALID},
        {BLOCK_STATUS_KEY: "nonsense"},
        {},
    ],
)
def test_a_valid_or_unlabelled_block_may_be_cited_as_a_result(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    """The other side, and the reason the check is inert today: no block in this
    repository carries a verdict, so nothing is refused until R10 writes one."""
    write_artifact(tmp_path, "artifacts/campaign/block_03", payload)
    write_doc(tmp_path, "artifacts/campaign/block_03 is the result of the third week.")
    assert citation_problems(tmp_path, DOCS) == []


def test_the_block_status_reader_accepts_only_the_two_labels(tmp_path: Path) -> None:
    for value, expected in ((VALID, VALID), (INVALID, INVALID), ("maybe", None)):
        directory = write_artifact(tmp_path, f"artifacts/b_{value}", {BLOCK_STATUS_KEY: value})
        assert declared_block_status(directory) == expected


# --------------------------------------------------------------------------- #
# the collision this module exists to keep apart
# --------------------------------------------------------------------------- #


def test_the_three_vocabularies_under_one_key_stay_distinguishable() -> None:
    """A later "unification" that made the freezer's values governance classes,
    or the recorder's lowercase ``prospective`` a deciding one, fails here."""
    from tools.freeze_evidence import DERIVED, EVIDENCE_CLASS_KEY as FREEZE_KEY, PRIMARY

    assert FREEZE_KEY == EVIDENCE_CLASS_KEY, "the two readers must share one key"
    assert (PRIMARY, DERIVED) == FREEZE_CLASSES
    assert normalise(PRIMARY) is None and normalise(DERIVED) is None

    from chimera.recorder.health import RecorderHealth

    engineering = RecorderHealth(contract_id="x", contract_hash="y", prospective_from=None)
    assert normalise(engineering.evidence_class) == ENGINEERING

    prospective = RecorderHealth(
        contract_id="x",
        contract_hash="y",
        prospective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert prospective.evidence_class == "prospective"
    assert normalise(prospective.evidence_class) is None, (
        "the recorder's lowercase spelling must not reach a deciding class: the "
        "feed being past its boundary is not an artifact's eligibility"
    )


def test_every_label_is_a_distinct_identifier() -> None:
    """Catches a copy-paste that gave two classes the same string."""
    assert len(set(GOVERNANCE_CLASSES)) == len(GOVERNANCE_CLASSES)
    assert len(set(CAMPAIGN_OUTCOMES)) == len(CAMPAIGN_OUTCOMES)
    named = {
        ENGINEERING,
        DIAGNOSTIC,
        EXPLORATORY,
        PREREGISTERED,
        PROSPECTIVE,
        CONFIRMATORY,
        ADAPTIVE,
        BURNED,
        SEALED_DIAGNOSTIC,
    }
    assert named == set(GOVERNANCE_CLASSES)


# --------------------------------------------------------------------------- #
# the manifest-field requirement, grandfathered
# --------------------------------------------------------------------------- #


def test_every_artifact_directory_either_declares_a_class_or_is_grandfathered() -> None:
    """§37.1's "every artifact directory carries the field", made enforceable.

    The 113 directories that predate R1-o are listed, not classified. What this
    buys is the half of the requirement that can be enforced from here: a NEW
    artifact directory must declare a class, because it cannot be on a list
    written at this commit.
    """
    directories = sorted(
        p.relative_to(REPO).as_posix() for p in (REPO / "artifacts").rglob("*") if p.is_dir()
    )
    assert directories, "no artifact directories found -- this guard proves nothing"

    missing = undeclared_artifacts(REPO, UNDECLARED_AT_R1O)
    assert not missing, (
        "a new artifact directory declares no evidence_class. Every directory "
        "created from R1-o onward carries one of "
        f"{GOVERNANCE_CLASSES}: {missing}"
    )


def test_the_grandfathering_list_only_shrinks() -> None:
    """An entry that has since been classified must leave the list.

    Without this the list would be write-only: a directory could be classified
    and still sit here, and the next reader could not tell which entries are
    still true. It also catches a path renamed out from under the list.
    """
    stale = [d for d in sorted(UNDECLARED_AT_R1O) if declared_class(REPO / d) is not None]
    assert not stale, (
        "these are listed as undeclared but now declare a class; remove them "
        f"from UNDECLARED_AT_R1O: {stale}"
    )
    gone = [d for d in sorted(UNDECLARED_AT_R1O) if not (REPO / d).is_dir()]
    assert not gone, f"these listed directories no longer exist: {gone}"


def test_the_checks_reach_verify(tmp_path: Path) -> None:
    """The wiring, asserted rather than assumed.

    `citation_problems` could be perfect and unreachable: `verify` is what
    `tools.verify_research_state` and `make check` call, so a check that is not
    in its list is a check that never runs. A synthetic root is used because the
    real one is clean by construction -- the point here is that a violation
    arrives through `verify`, not that one exists.
    """
    from nn.research_state import verify

    write_artifact(tmp_path, "artifacts/x", {EVIDENCE_CLASS_KEY: "primary"})
    (tmp_path / "README.md").write_text(
        "artifacts/x/a.json is prospective evidence.", encoding="utf-8"
    )
    problems = verify(tmp_path)
    assert any(
        "cites artifacts/x/a.json as prospective evidence" in problem for problem in problems
    ), problems


def test_the_requirement_is_not_vacuous(tmp_path: Path) -> None:
    """The positive control for the guard above, and it is load-bearing.

    A mutation campaign made that guard's predicate always-false and the guard
    still passed: a test standing on an inline predicate cannot notice its own
    predicate being hollowed out. So the predicate is a function, and here it is
    handed a tree where the answer is known -- one declared directory, one not,
    nothing excused -- and must name exactly the undeclared one.
    """
    write_artifact(tmp_path, "artifacts/declared", {EVIDENCE_CLASS_KEY: DIAGNOSTIC})
    write_artifact(tmp_path, "artifacts/silent", {"summary": "declares nothing"})

    assert undeclared_artifacts(tmp_path, frozenset()) == ["artifacts/silent"]
    assert undeclared_artifacts(tmp_path, frozenset({"artifacts/silent"})) == []
