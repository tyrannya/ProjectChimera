"""A completed checkpoint cannot coexist with a page saying it never ran.

At revision `1183e592` this repository held nine frozen P3 cells, their
comparison, and an artifact index listing all ten as CURRENT — while `README.md`
said P3 "has **not run**", `docs/research_roadmap.md` said "no P3 cell has been
fitted, no P3 number exists, and `artifacts/` contains no P3 evidence", and
`docs/microstructure_v1.md` said the same. Nothing failed, because nothing had
ever been told what those documents were claiming.

What is asserted here is a *declaration*, not a set of forbidden sentences: the
state of each checkpoint is derived from the artifact tree, rendered into one
block, and every front-door document must carry that block byte for byte. A
prose guard would pin today's wording and miss tomorrow's rewrite; a generated
block cannot be talked out of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nn import research_state
from nn.research_state import (
    ANSWERED,
    CHECKPOINTS,
    Checkpoint,
    DECLINED,
    FRONT_DOOR_DOCUMENTS,
    UNRUN,
    WITHDRAWN,
    checkpoint_states,
    existing_block,
    render_block,
    result_claims,
    terminal_contradictions,
    unrun_claims,
    verify,
)
from tools.verify_research_state import main, write_blocks

ROOT = Path(__file__).resolve().parent.parent


def test_this_repository_is_consistent_with_its_own_evidence():
    assert verify(ROOT) == []


def test_the_verifier_exits_zero_here():
    assert main(["--root", str(ROOT)]) == 0


def test_every_front_door_document_carries_the_block():
    expected = render_block(checkpoint_states(ROOT))
    for name in FRONT_DOOR_DOCUMENTS:
        assert existing_block((ROOT / name).read_text()) == expected, name


def test_the_state_comes_from_the_artifacts_and_not_from_a_hand_written_list():
    """Every `answered` checkpoint names an aggregate that is actually there."""
    states = checkpoint_states(ROOT)
    for checkpoint in CHECKPOINTS:
        present = (ROOT / checkpoint.evidence).is_file()
        assert (states[checkpoint.name] == ANSWERED) == present, checkpoint.name


def test_p3_is_answered_here_because_its_comparison_is_committed():
    """The specific regression: the audited revision said otherwise."""
    assert checkpoint_states(ROOT)["P3"] == ANSWERED


def test_p4_is_answered_here_because_stage1_evidence_exists():
    """P4 ran its preregistered Stage 1 and screened out before P4-HOLD."""
    states = checkpoint_states(ROOT)
    assert states["P4"] == ANSWERED
    assert (
        ROOT / "artifacts" / "benchmark" / "btc_p4_comparison" / "p2b_comparison.json"
    ).is_file()


# --- the guard has to be able to fail ----------------------------------------
def _clone(tmp_path: Path) -> Path:
    """A tree with the front-door documents and the artifacts they describe."""
    root = tmp_path / "repo"
    for name in FRONT_DOOR_DOCUMENTS:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text((ROOT / name).read_text())
    for checkpoint in CHECKPOINTS:
        source = ROOT / checkpoint.evidence
        if source.is_file():
            target = root / checkpoint.evidence
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}")
        if checkpoint.preregistration and (ROOT / checkpoint.preregistration).is_file():
            target = root / checkpoint.preregistration
            target.parent.mkdir(parents=True, exist_ok=True)
            # Copied rather than stubbed when it is itself a front-door document:
            # replacing it with a placeholder would delete the block the clone is
            # supposed to start out carrying.
            if not target.is_file():
                target.write_text("prereg")
    return root


def test_the_clone_is_clean_before_it_is_broken(tmp_path):
    """Falsifies everything below: a guard that always failed would pass them."""
    assert verify(_clone(tmp_path)) == []


def test_a_document_whose_block_is_stale_is_rejected(tmp_path):
    root = _clone(tmp_path)
    readme = root / "README.md"
    readme.write_text(readme.read_text().replace("| **answered** |", "| **unrun** |", 1))
    problems = verify(root)
    assert any("does not match the artifact tree" in problem for problem in problems)


def _with_a_finished_checkpoint(monkeypatch, root: Path) -> None:
    """Register one more checkpoint and give it evidence, inside ``root``.

    Synthetic rather than "whichever checkpoint happens to be last", so this
    asserts the invariant and not the contents of `CHECKPOINTS` on the day it
    was written.
    """
    extra = Checkpoint("P9", "btc_p9_probe", "artifacts/benchmark/btc_p9_comparison/p9.json")
    monkeypatch.setattr(research_state, "CHECKPOINTS", CHECKPOINTS + (extra,))
    aggregate = root / extra.evidence
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    aggregate.write_text("{}")


def test_a_new_checkpoints_evidence_makes_every_document_fail(tmp_path, monkeypatch):
    """The invariant, stated as the event it exists for.

    A checkpoint finishing is exactly when the documents go stale, so that is
    when they must stop passing.
    """
    root = _clone(tmp_path)
    assert verify(root) == []
    _with_a_finished_checkpoint(monkeypatch, root)

    problems = verify(root)
    assert len(problems) >= len(FRONT_DOOR_DOCUMENTS)
    assert all(any(name in problem for problem in problems) for name in FRONT_DOOR_DOCUMENTS)


def test_the_writer_reconciles_what_the_checker_rejected(tmp_path, monkeypatch):
    root = _clone(tmp_path)
    _with_a_finished_checkpoint(monkeypatch, root)
    assert verify(root) != []

    written = write_blocks(root)
    assert sorted(written) == sorted(FRONT_DOOR_DOCUMENTS)
    assert verify(root) == []


def test_a_document_with_no_block_at_all_is_rejected(tmp_path):
    """Deleting the declaration is not a way to stop declaring."""
    root = _clone(tmp_path)
    readme = root / "README.md"
    block = existing_block(readme.read_text())
    readme.write_text(readme.read_text().replace(block, ""))
    assert any("carries no research-state block" in p for p in verify(root))


@pytest.mark.parametrize(
    "sentence",
    [
        "no P3 cell has been fitted",
        "no P3 number exists",
        "`artifacts/` contains no P3 evidence",
        "A third checkpoint, **P3**, is declared and implemented but has **not run**",
        "the causal trade-flow information set (checkpoint P3, unrun)",
    ],
)
def test_the_exact_claims_the_audited_revision_made_are_now_rejected(tmp_path, sentence):
    """Every one of these was in a front-door document at `1183e592`."""
    root = _clone(tmp_path)
    readme = root / "README.md"
    readme.write_text(readme.read_text() + "\n" + sentence + "\n")
    problems = verify(root)
    assert any("whose evidence is committed" in problem for problem in problems)


def test_the_same_claims_about_an_unrun_checkpoint_are_allowed(monkeypatch, tmp_path):
    """The guard is about contradicting the evidence, not about vocabulary.

    A checkpoint with no evidence has no evidence, and saying so is the truth.
    """
    extra = Checkpoint("P9", "btc_p9_probe", "artifacts/benchmark/btc_p9_comparison/p9.json")
    monkeypatch.setattr(research_state, "CHECKPOINTS", CHECKPOINTS + (extra,))
    root = _clone(tmp_path)
    states = checkpoint_states(root)
    assert states["P9"] == UNRUN
    assert unrun_claims("There is no P9 evidence and no P9 number exists.", states) == []


# --- ending without a result -------------------------------------------------
# `withdrawn` and `declined` are the two states no artifact tree can express. A
# checkpoint nobody will ever open and a checkpoint waiting its turn leave the
# same empty directory, so the difference has to be declared — and a declaration
# is exactly the kind of thing that goes stale silently, which is what the tests
# below are for.


def test_p8_is_withdrawn_here_because_its_precondition_cannot_be_met():
    """Never opened, and no longer pending: two eligible modes cannot happen."""
    assert checkpoint_states(ROOT)["P8"] == WITHDRAWN
    assert not (ROOT / "artifacts" / "benchmark" / "btc_p8_decision").exists()


def test_p14_is_declined_here_and_its_design_is_not_in_this_tree():
    """Declined before opening, on a branch that was closed without merging.

    The missing `preregistration` path is the accurate record rather than an
    omission: the design lives on `claude/p14-native-tradeflow-prereg`, so
    naming a file here would point a reader at something that is not there.
    """
    assert checkpoint_states(ROOT)["P14"] == DECLINED
    p14 = next(c for c in CHECKPOINTS if c.name == "P14")
    assert p14.preregistration is None
    assert not (ROOT / "docs" / "p14_preregistration.md").exists()
    assert not (ROOT / "artifacts" / "benchmark" / "btc_p14_decision").exists()


def test_no_checkpoint_here_is_both_closed_without_a_result_and_holding_one():
    assert terminal_contradictions(ROOT) == []


@pytest.mark.parametrize(
    ("terminal", "expected"),
    [(None, UNRUN), (WITHDRAWN, WITHDRAWN), (DECLINED, DECLINED)],
)
def test_the_declaration_is_the_only_thing_separating_these_states(
    tmp_path, monkeypatch, terminal, expected
):
    """One empty tree, three readings, and only the declaration differs.

    This is why the state cannot be derived: nothing under `artifacts/`
    distinguishes a checkpoint waiting its turn from one nobody will ever open.
    """
    probe = Checkpoint(
        "P9", "btc_p9_probe", "artifacts/benchmark/btc_p9/p9.json", terminal=terminal
    )
    monkeypatch.setattr(research_state, "CHECKPOINTS", (probe,))
    assert not (tmp_path / probe.evidence).exists()
    assert checkpoint_states(tmp_path)["P9"] == expected


def test_a_terminal_declaration_may_not_outrank_evidence(tmp_path, monkeypatch):
    """The failure a declaration makes possible, and the guard against it.

    Declaring a checkpoint closed-without-a-result must never be a way to stop
    a committed number from being reported. So evidence wins the state, *and*
    the contradiction is named.
    """
    root = _clone(tmp_path)
    closed = Checkpoint(
        "P9", "btc_p9_probe", "artifacts/benchmark/btc_p9/p9.json", terminal=DECLINED
    )
    monkeypatch.setattr(research_state, "CHECKPOINTS", CHECKPOINTS + (closed,))
    assert checkpoint_states(root)["P9"] == DECLINED
    assert terminal_contradictions(root) == []

    aggregate = root / closed.evidence
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    aggregate.write_text("{}")

    assert checkpoint_states(root)["P9"] == ANSWERED
    problems = terminal_contradictions(root)
    assert len(problems) == 1 and "declared declined" in problems[0]
    assert any("declared declined" in problem for problem in verify(root))


def test_a_state_that_is_not_a_terminal_state_is_refused():
    """`terminal` is a small closed vocabulary, not free text."""
    with pytest.raises(ValueError, match="not a terminal state"):
        Checkpoint(
            "P9", "btc_p9_probe", "artifacts/benchmark/btc_p9/p9.json", terminal="negative"
        )


@pytest.mark.parametrize(
    "sentence",
    [
        "P14 is answered",
        "P14 was answered",
        "P14 is negative",
        "P14 answered negative",
        "P14 returned negative",
        "P14's verdict is recorded below",
        "P8 failed",
        "P8 is inconclusive",
        "P8's result is recorded below",
    ],
)
def test_a_document_claiming_an_outcome_for_a_checkpoint_that_has_none_is_rejected(
    tmp_path, sentence
):
    """The mirror of the unrun guard, and the reason both exist.

    `withdrawn` and `declined` mean NO NUMBER WAS EVER PRODUCED. Every sentence
    here presupposes a statistic, so every one of them is false by construction
    — which is exactly the kind of false a summary drifts into.
    """
    root = _clone(tmp_path)
    readme = root / "README.md"
    readme.write_text(readme.read_text() + "\n" + sentence + "\n")
    problems = verify(root)
    assert any("closed without ever producing a number" in problem for problem in problems)


@pytest.mark.parametrize(
    "sentence", ["P3 is negative", "P3 answered negative", "P3's verdict"]
)
def test_the_same_words_about_a_checkpoint_that_does_have_a_result_are_allowed(
    tmp_path, sentence
):
    """Falsifies the test above: the guard is about the state, not the words.

    P3 ran and answered negative. Saying so is the truth, and a guard that
    forbade it would be forbidding the repository's own findings.
    """
    root = _clone(tmp_path)
    readme = root / "README.md"
    readme.write_text(readme.read_text() + "\n" + sentence + "\n")
    assert verify(root) == []


def test_the_real_documents_describe_p8_and_p14_as_the_non_results_they_are():
    """Not a clone: the front doors this repository actually ships."""
    states = checkpoint_states(ROOT)
    for name in FRONT_DOOR_DOCUMENTS:
        assert result_claims((ROOT / name).read_text(encoding="utf-8"), states) == [], name


def test_the_writer_reconciles_a_checkpoint_that_was_closed_without_a_result(
    tmp_path, monkeypatch
):
    """A withdrawal goes stale in the documents exactly as a result would."""
    root = _clone(tmp_path)
    assert verify(root) == []
    closed = Checkpoint(
        "P9", "btc_p9_probe", "artifacts/benchmark/btc_p9/p9.json", terminal=WITHDRAWN
    )
    monkeypatch.setattr(research_state, "CHECKPOINTS", CHECKPOINTS + (closed,))
    assert verify(root) != []

    written = write_blocks(root)
    assert sorted(written) == sorted(FRONT_DOOR_DOCUMENTS)
    assert verify(root) == []
    assert "**withdrawn**" in (root / "README.md").read_text()
