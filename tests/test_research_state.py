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
    FRONT_DOOR_DOCUMENTS,
    PREREGISTERED,
    UNRUN,
    checkpoint_states,
    existing_block,
    render_block,
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


def test_p4_is_preregistered_and_not_answered():
    """Its design is committed; its evidence is not, and must not be."""
    states = checkpoint_states(ROOT)
    assert states["P4"] == PREREGISTERED
    assert not (ROOT / "artifacts" / "benchmark" / "btc_p4_comparison").exists()


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
