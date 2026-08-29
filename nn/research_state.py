"""Which research checkpoints have evidence, derived from the evidence itself.

A repository can hold a frozen, committed, CURRENT result and a front page that
says the checkpoint has not run. This one did: at ``1183e592`` the P3 comparison
and its nine frozen cells were committed and indexed as CURRENT while
``README.md``, ``docs/research_roadmap.md`` and ``docs/microstructure_v1.md``
all still said no P3 cell had been fitted and no P3 number existed. Nothing was
in a position to notice, because nothing had ever been told what the documents
were claiming.

The fix here is a *declaration*, not a prose test. The authoritative state of a
checkpoint is a fact about the artifact tree — its aggregate either exists or it
does not — and :func:`checkpoint_states` reads it from there. Every front-door
document then carries the rendered state block, byte for byte, so a checkpoint
that gains evidence makes those documents fail until they are updated. A guard
that matched sentences would pin today's wording and miss tomorrow's; a guard
that regenerates a block cannot be talked out of it by a rewrite.

The one thing a block cannot catch is prose *around* it that still narrates the
old state, so :func:`unrun_claims` adds a narrow second net: a small, named list
of ways to say "this checkpoint has no evidence", forbidden in a front-door
document for a checkpoint that has some. It is deliberately about a class of
claim rather than about a particular sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: A checkpoint has evidence, or a preregistration and no evidence, or neither.
ANSWERED = "answered"
PREREGISTERED = "preregistered"
UNRUN = "unrun"

MARKER_BEGIN = "<!-- research-state:begin -->"
MARKER_END = "<!-- research-state:end -->"

#: How a document may *not* describe a checkpoint that has evidence. Short and
#: named on purpose: each entry is a claim that no result exists, which is the
#: one thing a completed checkpoint makes false. Matched case-insensitively
#: against the checkpoint's id substituted into the pattern.
UNRUN_CLAIM_PATTERNS: tuple[str, ...] = (
    r"no {cp} cell has been fitted",
    r"no {cp} number exists",
    r"no {cp} evidence",
    r"contains no {cp} evidence",
    r"{cp}[^.\n]{{0,40}}has \*\*not run\*\*",
    r"{cp}[^.\n]{{0,40}}has not run",
    r"{cp}, unrun",
)


@dataclass(frozen=True)
class Checkpoint:
    """One research checkpoint and where its answer would live."""

    #: The identity used in documents and in the state block.
    name: str
    #: The research question `artifacts/README.md` scopes CURRENT to.
    question: str
    #: The aggregate whose existence *is* the checkpoint having an answer.
    evidence: str
    #: A committed preregistration, for a checkpoint designed before it runs.
    preregistration: str | None = None


#: Every checkpoint this repository has asked, oldest first. A new checkpoint is
#: added here when it is preregistered, not when it produces a number.
CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(
        "v4",
        "btc_ohlcv14_mtst_baseline",
        "artifacts/diagnostics/btc_regimes_v4/regime_diagnostics.json",
    ),
    Checkpoint(
        "P2a",
        "btc_p2a_model_family_benchmark",
        "artifacts/benchmark/btc_p2a_comparison/p2a_comparison.json",
    ),
    Checkpoint(
        "P2b",
        "btc_p2b_information_set_benchmark",
        "artifacts/benchmark/btc_p2b_comparison/p2b_comparison.json",
    ),
    Checkpoint(
        "P2c",
        "btc_p2c_information_set_benchmark",
        "artifacts/benchmark/btc_p2c_comparison/p2b_comparison.json",
    ),
    Checkpoint(
        "P3",
        "btc_p3_information_set_benchmark",
        "artifacts/benchmark/btc_p3_comparison/p2b_comparison.json",
    ),
    Checkpoint(
        "P4",
        "btc_p4_derivatives_positioning_benchmark",
        "artifacts/benchmark/btc_p4_comparison/p2b_comparison.json",
        preregistration="docs/p4_preregistration.md",
    ),
    Checkpoint(
        "P5",
        "btc_p5_information_set_benchmark",
        "artifacts/benchmark/btc_p5_comparison/p2b_comparison.json",
        preregistration="docs/p5_preregistration.md",
    ),
)

#: The documents a reader meets before any artifact. Each must carry the block.
FRONT_DOOR_DOCUMENTS: tuple[str, ...] = (
    "README.md",
    "artifacts/README.md",
    "docs/research_roadmap.md",
    "docs/research_reproduction.md",
    "docs/microstructure_v1.md",
    "docs/p4_preregistration.md",
    "docs/p5_preregistration.md",
)


class ResearchStateError(RuntimeError):
    """A document and the evidence tree disagree about what has been run."""


def checkpoint_states(root: Path) -> dict[str, str]:
    """The state of every checkpoint, read from the artifact tree.

    Evidence beats preregistration: a checkpoint that produced its aggregate is
    ``answered`` whatever else is committed beside it.
    """
    root = Path(root)
    states: dict[str, str] = {}
    for checkpoint in CHECKPOINTS:
        if (root / checkpoint.evidence).is_file():
            states[checkpoint.name] = ANSWERED
        elif checkpoint.preregistration and (root / checkpoint.preregistration).is_file():
            states[checkpoint.name] = PREREGISTERED
        else:
            states[checkpoint.name] = UNRUN
    return states


def render_block(states: dict[str, str]) -> str:
    """The exact text every front-door document has to contain."""
    lines = [
        MARKER_BEGIN,
        "<!--",
        "  Generated by nn.research_state from the artifact tree; do not edit by hand.",
        "  Regenerate with: python -m tools.verify_research_state --write",
        "-->",
        "",
        "| checkpoint | research question | state |",
        "| --- | --- | --- |",
    ]
    for checkpoint in CHECKPOINTS:
        lines.append(
            f"| `{checkpoint.name}` | `{checkpoint.question}` | "
            f"**{states[checkpoint.name]}** |"
        )
    lines += ["", MARKER_END]
    return "\n".join(lines)


def existing_block(text: str) -> str | None:
    """The state block a document currently carries, if it carries one."""
    start = text.find(MARKER_BEGIN)
    if start < 0:
        return None
    end = text.find(MARKER_END, start)
    if end < 0:
        return None
    return text[start : end + len(MARKER_END)]


def unrun_claims(text: str, states: dict[str, str]) -> list[str]:
    """Sentences claiming no evidence exists for a checkpoint that has some."""
    found: list[str] = []
    for checkpoint in CHECKPOINTS:
        if states[checkpoint.name] != ANSWERED:
            continue
        for pattern in UNRUN_CLAIM_PATTERNS:
            expression = pattern.format(cp=re.escape(checkpoint.name))
            for match in re.finditer(expression, text, flags=re.IGNORECASE):
                found.append(match.group(0))
    return found


def verify(root: Path) -> list[str]:
    """Every disagreement between the documents and the evidence tree."""
    root = Path(root)
    states = checkpoint_states(root)
    expected = render_block(states)
    problems: list[str] = []
    for name in FRONT_DOOR_DOCUMENTS:
        path = root / name
        if not path.is_file():
            problems.append(f"{name}: front-door document is missing")
            continue
        text = path.read_text()
        block = existing_block(text)
        if block is None:
            problems.append(
                f"{name}: carries no research-state block. Every front-door document "
                "declares which checkpoints have evidence, so a completed checkpoint "
                "cannot coexist with a page that never mentions it."
            )
        elif block != expected:
            problems.append(
                f"{name}: its research-state block does not match the artifact tree. "
                "Run `python -m tools.verify_research_state --write` and reconcile the "
                "prose around it."
            )
        for claim in unrun_claims(text, states):
            problems.append(
                f"{name}: says {claim!r} about a checkpoint whose evidence is committed"
            )
    return problems
