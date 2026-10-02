"""§37.1's governance classes, as names a program can check.

Section 37.1 of the master roadmap fixes the vocabulary the whole programme
reasons in: what a piece of evidence IS, and therefore what it may be reused
for. The table is prose there, which means nothing stops a document from citing
an engineering soak log as prospective evidence, or an adaptive re-read of a
burned block as a result. R1-o turns the load-bearing half of that table into
identifiers and three refusals.

THE KEY `evidence_class` ALREADY MEANS THREE DIFFERENT THINGS in this
repository, and that is the first thing to understand before changing anything:

* ``tools/freeze_evidence.py`` writes ``primary`` / ``derived``. That is a
  FREEZING distinction -- is this file covered by an immutable manifest, or is
  it pinned by regenerating it? It says nothing about scientific standing;
* ``chimera/recorder/health.py`` writes ``prospective`` / ``engineering``. That
  is a question about the FEED -- what would a minute recorded right now be?
* 91 of the 112 committed artifact JSONs carry the key with their own wording,
  from ``primary`` to whole sentences like "one native-timeframe specialist,
  fitted once; P6 primary evidence". 21 carry nothing.

So this module does not redefine the key. It defines the §37.1 VALUES, and
:func:`normalise` answers one question about a value found in the wild: is this
a governance class, and which? Everything else -- including both legacy
spellings above -- reads as *undeclared*, which is the safe answer, because
every refusal below only needs to know that something is NOT a deciding class.

WHY NOTHING IS RETRO-CLASSIFIED. The roadmap asks that every artifact directory
carry the field. 112 committed artifact files predate it, and R1's own PR rules
forbid a PR here from touching a scientific artifact. Guessing their classes in
code would be worse than leaving them undeclared: choosing between BURNED,
ADAPTIVE and EXPLORATORY for a committed run is a scientific judgement about
what that evidence may be reused for, and R1 creates no scientific evidence and
alters no scientific interpretation. They are therefore UNDECLARED, the checks
treat undeclared as "not a deciding class", and that is strictly stronger than a
guess: an undeclared artifact can never be cited as prospective or confirmatory
evidence, which is precisely what §37.1 asks the checks to prevent.

A DECIDING CLASS MUST BE SPELLED CANONICALLY. ``PROSPECTIVE`` and
``CONFIRMATORY`` are the two classes a promotion decision may rest on, so they
are recognised only in their exact uppercase form. The lowercase spellings the
recorder already uses are accepted for the rest, where the stakes are a label on
a soak log rather than eligibility for real money.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# 37.1: the governance classes
# --------------------------------------------------------------------------- #

#: tests, invariants, soak logs, replay parity, coverage.
ENGINEERING = "ENGINEERING"
#: descriptive statistics on engineering-root or pre-activation data.
DIAGNOSTIC = "DIAGNOSTIC"
#: any model/feature/rule evaluation on pre-boundary or engineering-root data.
EXPLORATORY = "EXPLORATORY"
#: a hashed design committed and pushed before the data it governs is read.
PREREGISTERED = "PREREGISTERED"
#: evidence accrued at or after an activated ``prospective_from``.
PROSPECTIVE = "PROSPECTIVE"
#: a second, independent prospective period under the same frozen candidate.
CONFIRMATORY = "CONFIRMATORY"
#: any evaluation whose design was chosen after seeing outcomes from the data.
ADAPTIVE = "ADAPTIVE"
#: the four outer historical blocks; P4-HOLD; anything read twice to decide.
BURNED = "BURNED"
#: a hashed artifact that may not be read before a named event.
SEALED_DIAGNOSTIC = "SEALED-DIAGNOSTIC"

#: Every class §37.1 defines, in the table's order.
GOVERNANCE_CLASSES: tuple[str, ...] = (
    ENGINEERING,
    DIAGNOSTIC,
    EXPLORATORY,
    PREREGISTERED,
    PROSPECTIVE,
    CONFIRMATORY,
    ADAPTIVE,
    BURNED,
    SEALED_DIAGNOSTIC,
)

#: The two a promotion decision may rest on. Written out rather than derived,
#: because "which classes can decide" is the question the whole table exists to
#: answer and it should be one line to read, not a filter to evaluate.
DECIDING_CLASSES: tuple[str, ...] = (PROSPECTIVE, CONFIRMATORY)

# --------------------------------------------------------------------------- #
# 37.1: block and campaign status labels (R10/R11)
# --------------------------------------------------------------------------- #

#: A scored block is one or the other, decided mechanically and blind to
#: outcomes. An INVALID block stays in chronology and reporting for ever and is
#: never a negative result.
VALID = "VALID"
INVALID = "INVALID"
BLOCK_STATUSES: tuple[str, ...] = (VALID, INVALID)

PASS = "PASS"
NEGATIVE = "NEGATIVE"
NOT_EVALUABLE = "NOT_EVALUABLE"
ABORTED = "ABORTED"
DECLINED = "DECLINED"

#: How a campaign ends. ``NOT_EVALUABLE`` is reserved for exhausting the
#: extension before N valid blocks; ``ABORTED`` is an owner stop or a change to
#: a contract-hashed module during accrual. Neither is a negative result, and
#: §37.1 is explicit that they must not be recorded as one another.
CAMPAIGN_OUTCOMES: tuple[str, ...] = (PASS, NEGATIVE, NOT_EVALUABLE, ABORTED, DECLINED)

# --------------------------------------------------------------------------- #
# the spellings already in the tree
# --------------------------------------------------------------------------- #

#: ``tools/freeze_evidence.py``'s values. Not governance classes at all -- they
#: answer "immutable manifest or regenerated?" -- and listed here so that
#: :func:`normalise` refuses them deliberately rather than by falling through.
FREEZE_CLASSES: tuple[str, ...] = ("primary", "derived")

#: ``chimera/recorder/health.py``'s values, and what they mean in §37.1 terms.
#: ``prospective`` is deliberately absent: a deciding class is recognised only
#: in its canonical uppercase form, so a lowercase spelling cannot promote a
#: soak log into evidence a real-money decision may rest on.
RECORDER_CLASS_EQUIVALENTS: dict[str, str] = {"engineering": ENGINEERING}

#: The field, under the name §37.1 gives it.
EVIDENCE_CLASS_KEY = "evidence_class"


def normalise(raw: object) -> str | None:
    """The §37.1 class a declared value names, or ``None``.

    ``None`` means *undeclared as a governance class* and covers four different
    situations on purpose, because every check here needs the same answer from
    all of them: nothing was declared; a freezing value (``primary``) was;
    free-form prose was; or a deciding class was spelled in a form that does not
    count. Undeclared is never a deciding class, which is the only property the
    refusals rest on.
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if value in GOVERNANCE_CLASSES:
        return value
    return RECORDER_CLASS_EQUIVALENTS.get(value.lower())


def is_deciding(label: str | None) -> bool:
    """Whether evidence of this class may back a promotion decision."""
    return label in DECIDING_CLASSES


def declared_class(directory: Path) -> str | None:
    """The §37.1 class an artifact directory declares, or ``None``.

    Read from the JSON the run wrote, under the same key and by the same
    first-match rule ``tools.freeze_evidence.evidence_class_of`` uses, so that
    one directory does not answer two different questions depending on which
    tool asked. Unreadable and non-object JSON is skipped rather than raising:
    this is a check over a tree someone else wrote, and a malformed file is a
    directory that declares nothing, not a crash.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(payload, dict) and EVIDENCE_CLASS_KEY in payload:
            return normalise(payload[EVIDENCE_CLASS_KEY])
    return None


def deciding_artifacts(root: Path) -> dict[str, str]:
    """Every artifact directory under ``root`` that declares a deciding class.

    Returns repo-relative directory path -> class. Today this is empty, and it
    is empty for a reason that can be stated rather than hoped: the only
    recorder contract that could carry a boundary has ``prospective_from: null``,
    so no minute in this repository was recorded at or after an activated
    boundary, so nothing can honestly be PROSPECTIVE. The function is not
    hard-coded to agree with that -- it reads the tree -- which is what makes it
    still correct on the day the first one appears.
    """
    root = Path(root)
    artifacts = root / "artifacts"
    found: dict[str, str] = {}
    if not artifacts.is_dir():
        return found
    for directory in sorted(p for p in artifacts.rglob("*") if p.is_dir()):
        label = declared_class(directory)
        if is_deciding(label):
            assert label is not None  # narrowed by is_deciding
            found[directory.relative_to(root).as_posix()] = label
    return found


# --------------------------------------------------------------------------- #
# the three refusals §37.1 asks for
# --------------------------------------------------------------------------- #

#: The key a block's committed validity verdict is written under. §37.1 puts
#: that verdict in the block's own evidence directory, computed by the verifier
#: job and never by hand. No block exists yet, so nothing in the tree carries
#: this today and the check below is inert until R10 -- defined now, with
#: synthetic fixtures standing in for a campaign, because the alternative is
#: writing it during the first campaign it would govern.
BLOCK_STATUS_KEY = "block_status"

#: An artifact path, as a document writes one. Matching the path and the claim
#: on the SAME LINE is deliberate: these documents are markdown, their claims
#: live in table rows and bullets at least as often as in sentences, and a
#: sentence splitter over a table produces windows that span unrelated rows.
ARTIFACT_PATH = r"artifacts/[A-Za-z0-9_./-]+"

#: How a document may *not* describe an artifact that is not PROSPECTIVE. Each
#: entry is an affirmative claim that the cited artifact IS prospective
#: evidence, which is the one thing a non-deciding class makes false.
#:
#: Kept short, literal and affirmative for the reason the lists in
#: ``nn.research_state`` are: these documents discuss the word constantly and
#: mostly to DENY it. ``README.md`` says "no prospective evidence exists",
#: ``current_development_plan.md`` says "never as pristine prospective
#: evidence", ``p4_preregistration.md`` says "cannot produce confirmatory
#: evidence" -- every one of them true, and a pattern that fired on the phrase
#: alone would forbid the truth along with the error. Hence :data:`NEGATORS`.
PROSPECTIVE_CITATION_PATTERNS: tuple[str, ...] = (
    r"is prospective evidence",
    r"are prospective evidence",
    r"as prospective evidence",
    r"counts as prospective",
    r"provides prospective evidence",
)

#: The mirror list for CONFIRMATORY.
CONFIRMATORY_CITATION_PATTERNS: tuple[str, ...] = (
    r"is confirmatory evidence",
    r"are confirmatory evidence",
    r"as confirmatory evidence",
    r"counts as confirmatory",
    r"provides confirmatory evidence",
)

#: How a document may *not* describe a block the verifier marked INVALID. An
#: INVALID block stays in chronology for ever and is never a result -- not a
#: negative one, not a positive one.
RESULT_CITATION_PATTERNS: tuple[str, ...] = (
    r"is the result",
    r"is a result",
    r"the result is",
    r"shows that",
    r"demonstrates",
    r"its result",
)

#: Words that turn any claim above into its denial. Checked anywhere on the
#: line rather than only before the claim: in these documents the denial often
#: trails it ("prospective evidence: none"), and a guard that refuses a true
#: sentence gets loosened until it refuses nothing.
NEGATORS: tuple[str, ...] = (
    r"\bno\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bcannot\b",
    r"\bnone\b",
    r"\bneither\b",
    r"\bwithout\b",
    r"\bnon-",
    r"\bmay not\b",
    r"\bforbid",
    r"\brefuses?\b",
)


def _negated(line: str) -> bool:
    """Whether the line denies rather than asserts."""
    return any(re.search(pattern, line, re.IGNORECASE) for pattern in NEGATORS)


def _citations(text: str, patterns: tuple[str, ...]) -> list[tuple[str, str]]:
    """Every (artifact path, line) where the line affirmatively makes the claim."""
    found: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not any(re.search(p, line, re.IGNORECASE) for p in patterns):
            continue
        if _negated(line):
            continue
        for path in re.findall(ARTIFACT_PATH, line):
            found.append((path, " ".join(line.split())))
    return found


def declared_block_status(directory: Path) -> str | None:
    """The validity verdict a block directory carries, or ``None``."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(payload, dict) and BLOCK_STATUS_KEY in payload:
            value = payload[BLOCK_STATUS_KEY]
            if isinstance(value, str) and value.strip() in BLOCK_STATUSES:
                return value.strip()
    return None


def citation_problems(root: Path, documents: tuple[str, ...]) -> list[str]:
    """§37.1's three refusals, over the front-door documents.

    A document may not cite an artifact that is not PROSPECTIVE as prospective
    evidence, nor one that is not CONFIRMATORY as confirmatory evidence, nor a
    block the verifier marked INVALID as a result.

    The class comes from the cited directory itself, never from the citing
    sentence: a document asserting a class is exactly the thing being checked,
    so trusting it would make the check agree with whatever it was told.
    """
    root = Path(root)
    problems: list[str] = []
    for name in documents:
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")

        for wanted, patterns in (
            (PROSPECTIVE, PROSPECTIVE_CITATION_PATTERNS),
            (CONFIRMATORY, CONFIRMATORY_CITATION_PATTERNS),
        ):
            for cited, line in _citations(text, patterns):
                actual = declared_class(_directory_of(root, cited))
                if actual == wanted:
                    continue
                problems.append(
                    f"{name}: cites {cited} as {wanted.lower()} evidence, but that "
                    f"artifact declares {actual or 'no governance class'}. "
                    f"Only a directory declaring {wanted} may be cited that way "
                    f"({EVIDENCE_CLASS_KEY} in its manifest). The line reads: {line!r}"
                )

        for cited, line in _citations(text, RESULT_CITATION_PATTERNS):
            if declared_block_status(_directory_of(root, cited)) != INVALID:
                continue
            problems.append(
                f"{name}: cites {cited} as a result, but the verifier marked that "
                "block INVALID. An INVALID block stays in chronology and reporting "
                "for ever and is never a result -- not negative, not positive. "
                f"The line reads: {line!r}"
            )
    return problems


def _directory_of(root: Path, cited: str) -> Path:
    """The directory a cited path names: itself, or the file's parent."""
    target = root / cited
    return target if target.is_dir() else target.parent


def undeclared_artifacts(root: Path, grandfathered: frozenset[str]) -> list[str]:
    """Artifact directories that declare no §37.1 class and are not excused.

    §37.1 asks that every artifact directory carry the field. 112 committed
    artifact files predate the requirement and R1 may not touch a scientific
    artifact, so the enforceable half is this: a directory may be undeclared
    only if it was already undeclared when the rule arrived. Anything created
    afterwards cannot be on a list written before it existed.

    Lives here rather than in the test that uses it for a reason a mutation
    campaign demonstrated: a guard whose predicate is inline in its own test
    cannot catch that predicate being made vacuous. As a function it can be
    handed a synthetic tree and an empty excuse list, which is what the positive
    control does.
    """
    root = Path(root)
    artifacts = root / "artifacts"
    if not artifacts.is_dir():
        return []
    return [
        relative
        for relative in sorted(
            p.relative_to(root).as_posix() for p in artifacts.rglob("*") if p.is_dir()
        )
        if declared_class(root / relative) is None and relative not in grandfathered
    ]
