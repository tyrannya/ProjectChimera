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

Two states cannot be derived this way, because no absence of evidence carries
their meaning. A checkpoint whose question became moot and a checkpoint whose
design was refused before it opened both leave exactly the artifact tree of a
checkpoint that is merely waiting its turn, and reading them as "waiting" is
wrong in a way that compounds: the programme keeps a decision on its list of
things still to do. So ``withdrawn`` and ``declined`` are *declared* on the
checkpoint, evidence still overrides them (:func:`terminal_contradictions`
reports the contradiction rather than letting a declaration bury a number), and
:func:`result_claims` is the mirror of :func:`unrun_claims`: a checkpoint that
never produced a number may not be described in a front-door document as
answered, negative, positive or failed. Those words all presuppose a statistic
that does not exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: A checkpoint has evidence, or a preregistration and no evidence, or neither.
ANSWERED = "answered"
PREREGISTERED = "preregistered"
UNRUN = "unrun"

#: The two ways a checkpoint ends without ever producing a number. Both are
#: decisions a person recorded, never facts about the artifact tree.
#:
#: ``withdrawn`` — the question became moot and the checkpoint was never opened.
#: ``declined`` — the design was reviewed and refused before it was opened.
#:
#: Neither is an answer, a negative, a positive, a failure or an inconclusive
#: result. Each means NO RESULT EXISTS, arrived at deliberately rather than by
#: not having got round to it yet.
WITHDRAWN = "withdrawn"
DECLINED = "declined"

#: The states that mean "closed, and no number was ever produced".
TERMINAL_STATES: tuple[str, ...] = (WITHDRAWN, DECLINED)

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

#: The mirror list: how a document may *not* describe a checkpoint that was
#: closed without a result. Each entry asserts an outcome, which is the one
#: thing ``withdrawn`` and ``declined`` make false — there is no statistic to be
#: negative, positive, failed or inconclusive about. Kept literal and short for
#: the same reason as the list above: this is a guard against a class of claim,
#: not a style checker, and a pattern loose enough to swallow "its eligibility
#: precondition failed" would forbid the truth along with the error.
RESULT_CLAIM_PATTERNS: tuple[str, ...] = (
    r"{cp} is answered",
    r"{cp} was answered",
    r"{cp} is negative",
    r"{cp} was negative",
    r"{cp} is positive",
    r"{cp} was positive",
    r"{cp} is inconclusive",
    r"{cp} was inconclusive",
    r"{cp} answered negative",
    r"{cp} answered positive",
    r"{cp} returned negative",
    r"{cp} returned positive",
    r"{cp} failed",
    r"{cp} has failed",
    r"{cp}'s (?:result|verdict|answer|number|finding)",
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
    #: ``WITHDRAWN`` or ``DECLINED`` when the checkpoint was closed by a recorded
    #: decision instead of by evidence; ``None`` while it can still produce a
    #: number. This is the only field here that is not a path, because it is the
    #: only fact about a checkpoint that the repository cannot look up.
    terminal: str | None = None

    def __post_init__(self) -> None:
        if self.terminal is not None and self.terminal not in TERMINAL_STATES:
            raise ValueError(
                f"{self.name}: {self.terminal!r} is not a terminal state. "
                f"Expected one of {TERMINAL_STATES}, or None."
            )


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
    Checkpoint(
        "P6",
        "btc_p6_multiclock_specialist_screen",
        "artifacts/benchmark/btc_p6_decision/decision.json",
        preregistration="docs/p6_preregistration.md",
    ),
    Checkpoint(
        "P6-EXT",
        "btc_p6ext_swing_clock_specialist_screen",
        "artifacts/benchmark/btc_p6ext_decision/decision.json",
        preregistration="docs/p6_extension_preregistration.md",
    ),
    Checkpoint(
        "P7",
        "btc_p7_cross_timeframe_consensus",
        "artifacts/benchmark/btc_p7_decision/decision.json",
        preregistration="docs/p7_preregistration.md",
    ),
    Checkpoint(
        "P8",
        "btc_p8_automatic_trading_mode_router",
        "artifacts/benchmark/btc_p8_decision/decision.json",
        preregistration="docs/p8_preregistration.md",
        # Withdrawn as moot, never opened. Its own eligibility precondition is
        # two eligible trading modes; P6, P6-EXT and P7 left none, and the only
        # route to a second one is refitting clocks those checkpoints screened
        # out, which its rules forbid. The design stays committed and readable.
        terminal=WITHDRAWN,
    ),
    Checkpoint(
        "P13",
        "btc_p13_structural_carry_feasibility",
        "artifacts/benchmark/btc_p13_decision/decision.json",
        preregistration="docs/p13_preregistration.md",
    ),
    Checkpoint(
        "P14",
        "btc_p14_native_tradeflow_screen",
        "artifacts/benchmark/btc_p14_decision/decision.json",
        # No `preregistration` path, and the omission is the accurate record
        # rather than an oversight: P14 was preregistered on the branch
        # `claude/p14-native-tradeflow-prereg` at
        # `36cdae48877b1d5fa88b2664c127b5307a917751` (PR #67), which was closed
        # without merging, so `docs/p14_preregistration.md` is not in this tree
        # and naming it here would point a reader at a file that is not there.
        # The branch is retained as historical design evidence.
        terminal=DECLINED,
    ),
)

#: The documents a reader meets before any artifact. Each must carry the block.
#:
#: A preregistration joins this list in the commit that creates it, not before:
#: a front-door document that does not exist yet is a missing file rather than a
#: silent omission, and the verifier says so either way.
FRONT_DOOR_DOCUMENTS: tuple[str, ...] = (
    "README.md",
    "artifacts/README.md",
    "docs/research_roadmap.md",
    "docs/research_reproduction.md",
    "docs/current_development_plan.md",
    "docs/proposed_development_plan_post_fable_5_1_audit.md",
    "docs/microstructure_v1.md",
    "docs/p4_preregistration.md",
    "docs/p5_preregistration.md",
    "docs/p6_preregistration.md",
    "docs/p6_extension_preregistration.md",
    "docs/p7_preregistration.md",
    "docs/p8_preregistration.md",
    "docs/p13_preregistration.md",
)


class ResearchStateError(RuntimeError):
    """A document and the evidence tree disagree about what has been run."""


def checkpoint_states(root: Path) -> dict[str, str]:
    """The state of every checkpoint, read from the artifact tree.

    Evidence beats everything: a checkpoint that produced its aggregate is
    ``answered`` whatever else is committed or declared beside it. That ordering
    is deliberate — a declaration is allowed to say a checkpoint was closed
    without a number, and is never allowed to conceal one that exists. When the
    two disagree, :func:`terminal_contradictions` says so.

    A terminal declaration then beats a committed preregistration, because P8's
    design being committed is exactly why it would otherwise read
    ``preregistered`` forever: the file is still there, and it is still the
    right file to read. What changed is that nobody is going to open it.
    """
    root = Path(root)
    states: dict[str, str] = {}
    for checkpoint in CHECKPOINTS:
        if (root / checkpoint.evidence).is_file():
            states[checkpoint.name] = ANSWERED
        elif checkpoint.terminal is not None:
            states[checkpoint.name] = checkpoint.terminal
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


def result_claims(text: str, states: dict[str, str]) -> list[str]:
    """Sentences claiming an outcome for a checkpoint that produced none."""
    found: list[str] = []
    for checkpoint in CHECKPOINTS:
        if states[checkpoint.name] not in TERMINAL_STATES:
            continue
        for pattern in RESULT_CLAIM_PATTERNS:
            expression = pattern.format(cp=re.escape(checkpoint.name))
            for match in re.finditer(expression, text, flags=re.IGNORECASE):
                found.append(match.group(0))
    return found


def terminal_contradictions(root: Path) -> list[str]:
    """Checkpoints declared closed without a result that nevertheless have one.

    The failure mode a declaration introduces: someone writes ``withdrawn`` on a
    checkpoint and later the checkpoint produces an aggregate anyway. The block
    would say ``answered`` — :func:`checkpoint_states` refuses to let a
    declaration outrank evidence — and the declaration would sit in the source
    quietly contradicting it. This is the second half of that guarantee.
    """
    root = Path(root)
    problems: list[str] = []
    for checkpoint in CHECKPOINTS:
        if checkpoint.terminal is None:
            continue
        if (root / checkpoint.evidence).is_file():
            problems.append(
                f"{checkpoint.name}: declared {checkpoint.terminal} — closed with no "
                f"result — but {checkpoint.evidence} exists. A checkpoint that "
                "produced an aggregate was opened; remove the declaration and "
                "record what it actually did."
            )
    return problems


def verify(root: Path) -> list[str]:
    """Every disagreement between the documents and the evidence tree."""
    root = Path(root)
    states = checkpoint_states(root)
    expected = render_block(states)
    problems: list[str] = terminal_contradictions(root)
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
        for claim in result_claims(text, states):
            problems.append(
                f"{name}: says {claim!r} about a checkpoint that was closed without "
                "ever producing a number. There is no result to be answered, "
                "negative, positive or failed."
            )
    return problems
