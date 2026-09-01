"""The six-calendar-block runner, under amendment **A2R2**.

:mod:`nn.p13_carry` evaluates ONE block given a quote series. This module decides
which quote series a block has — which is where ``MARKLESS_LIQUIDATION_VALIDITY_
POLICY`` actually bites, because the frozen distinction between "when may a
position open" and "when must the screen stop" is a distinction about WHEN an
invalid instant is met, not about what it looks like.

**What A2R2 changed here, and why.** A2 and A2R1 made the opening search reject a
candidate instant whose same-bar mark row was absent. That is acausal: a
``markPriceKlines`` row is stamped by candle OPEN ``t``, but the fact that a
completed row for that bar exists in the published archive at all is established
only AFTER the bar completes. Deciding at ``t`` not to open at ``t`` because that
row is absent therefore conditions the ENTRY INSTANT on information that does not
exist at ``t`` — SOURCE-AVAILABILITY LOOK-AHEAD. It read no future PRICE, and the
old implementation was careful about that, but availability is future information
in its own right. A2R2 removes the rule, and this module implements the removal.

So there is now ONE live source-validity state and one live delay reason:

``held_bar_mark_absent`` — the only live markless state
    A HELD bar carrying neither a mark high nor a mark close is TERMINAL and
    SCREEN-WIDE: ``P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT EVALUABLE``. It is
    not skipped, not jumped, not closed before, not reopened after, the affected
    block is not excluded, and — the A2R2 addition — the opening instant is NOT
    moved backwards or forwards to make the bar stop being held. **This includes
    bar 0**, which is where the correction lands: the position opens at bar 0's
    OPEN, so bar 0 is held, so bar 0 needs its mark like every other held bar.

``pre_open_execution_row_absent`` — the only live delay reason
    A leg supplying no row means no fill can be priced at that instant on both
    legs, so the opening search advances — which is ``POSITION_LIFECYCLE
    .open_instant``'s own forward search, not an A2R2 invention. Nothing is
    attributed to the strategy across the skipped interval, which this module
    guarantees the only way that can be guaranteed: the block's quote series
    BEGINS at the valid open, so there is no earlier bar for any accrual, fee,
    touch or excursion to attach to.

``pre_open_mark_absent`` and ``no_valid_opening_instant_in_block`` are RETIRED by
A2R2 and are not reachable from any code path below. Their names survive in the
preregistration as provenance; this module must never produce either.

**What counts as a held bar, and why this module reads it the way it does.**
``MARGIN_AND_LIQUIDATION.liquidation_check`` requires its inequality "evaluated at
every hourly grid instant while the position is open". So the held window is
checked against the CALENDAR grid — every nominal hour from the open to the exit —
rather than against whichever rows happen to exist. Driving it from the rows
instead would make an hour with no row at all invisible to the very check that
exists to notice an untested hour, and it would require this module to invent a
distinction between "no row was published" and "a row was published without a
mark". ``MARKLESS_LIQUIDATION_VALIDITY_POLICY`` authorises no such distinction,
and ``POSITION_LIFECYCLE.validity_definition`` defines one uniform notion of an
invalid instant — "the row is present in every required source ..." — so this
module applies that one notion to every held hour and to every source alike.

The consequence is stated plainly because it is consequential: under this reading
a single unpublished hour inside a holding window terminates the screen. That is
the fail-closed direction, it is what the two frozen sentences say when read
together, and the alternative reading requires a distinction the frozen text does
not make. It is flagged for review rather than buried.

**Object availability is never consulted here.** The funding notional's per-object
fallback lives in :mod:`nn.p13_alignment` and is not reachable from any code path
below. A2 authorises no liquidation surrogate, so there is nothing for this
module to fall back to and no flag that could turn one on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable, Sequence

from nn.p13_alignment import AlignedSources, grid_instants
from nn.p13_carry import (
    NOMINAL_BAR_NS,
    TOUCH_SOURCES,
    Allocation,
    BlockResult,
    CarryError,
    Costs,
    FundingSettlement,
    LiquidationTouchProvenance,
    Quote,
    Venue,
    apply_funding,
    evaluate_block,
    is_liquidated,
    open_carry,
    unclosed_block_result,
)
from nn.p13_preregistration import (
    MARKLESS_STATE_HELD,
    MARKLESS_STATES_RETIRED_BY_A2R2,
    OPENING_DELAY_EXECUTION_ABSENT,
    RESULT_STATES,
    TEMPORAL_PARTITION,
)
from nn.p13_sources import RESEARCH_BOUNDARY_NS

__all__ = [
    "NOT_EVALUABLE",
    "BlockError",
    "SourceInsufficiency",
    "CalendarBlock",
    "OpeningSearch",
    "BlockRun",
    "ScreenRun",
    "calendar_blocks",
    "find_opening_instant",
    "held_and_exit_instants",
    "build_quotes",
    "run_block",
    "run_screen",
    "NoValidExit",
]

#: The one terminal label A2R2's single source-insufficiency branch produces, taken
#: from the frozen ``RESULT_STATES`` rather than spelled again. A literal here
#: could drift from the design; an index into the frozen tuple cannot.
NOT_EVALUABLE = "P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT EVALUABLE"
assert NOT_EVALUABLE in RESULT_STATES, "the terminal label is not a declared result state"

#: The one markless state this module may still produce. Asserted against the
#: frozen design rather than assumed, so that reinstating a retired state here
#: fails at import rather than in an artifact: A2R2 retires the two that could
#: only fire BEFORE an open, and both of them decided the entry instant from a
#: fact established after the bar completed.
assert MARKLESS_STATE_HELD not in MARKLESS_STATES_RETIRED_BY_A2R2
assert OPENING_DELAY_EXECUTION_ABSENT not in MARKLESS_STATES_RETIRED_BY_A2R2


class BlockError(RuntimeError):
    """A block cannot be assembled from the aligned sources."""


class SourceInsufficiency(BlockError):
    """A required risk quantity is not computable, so the SCREEN terminates.

    Carried as an exception because it must be impossible to ignore. A function
    returning a status code can have its status dropped by a caller in a hurry;
    an exception propagating out of :func:`run_screen` cannot become a block
    result, cannot be averaged, and cannot reach a gate.

    :attr:`state` is one of the frozen ``MARKLESS_STATES`` and :attr:`result_state`
    is always the frozen NOT EVALUABLE label — never ``NOT VIABLE``. The reason is
    source insufficiency for a required risk quantity, which says nothing
    whatsoever about carry.
    """

    def __init__(
        self,
        state: str,
        block_label: str,
        message: str,
        *,
        instant_ns: int | None = None,
        missing: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.state = state
        self.block_label = block_label
        self.instant_ns = instant_ns
        self.missing = tuple(missing)
        self.result_state = NOT_EVALUABLE

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "block": self.block_label,
            "instant_ns": self.instant_ns,
            "missing_sources": list(self.missing),
            "result_state": self.result_state,
            "reason": str(self),
            "is_economic_failure": False,
        }


class NoValidExit(BlockError):
    """The block opened and has no permitted closing fill — amendment A1's case.

    Distinct from :class:`SourceInsufficiency` because the outcomes differ: this
    one makes the SCREEN ``INVALID`` through A1's rule, while a source
    insufficiency makes it ``NOT EVALUABLE``. Collapsing them would let a missing
    exit be reported as source insufficiency, or a missing risk quantity as an
    economic-shaped INVALID, and the frozen design distinguishes them for good
    reason.

    It carries the HELD quotes so the runner can report the funding, fees and
    slippage actually incurred "as the facts they are", which is what A1 requires
    of an unclosed block.
    """

    def __init__(
        self, block_label: str, message: str, *, held: Sequence[Quote], instant_ns: int
    ) -> None:
        super().__init__(message)
        self.block_label = block_label
        self.held = tuple(held)
        self.instant_ns = instant_ns


@dataclass(frozen=True)
class CalendarBlock:
    """One inference unit: a UTC calendar year, or the final partial one."""

    label: str
    start_ns: int
    end_exclusive_ns: int

    def contains(self, instant_ns: int) -> bool:
        return self.start_ns <= instant_ns < self.end_exclusive_ns


def calendar_blocks() -> tuple[CalendarBlock, ...]:
    """The six blocks ``TEMPORAL_PARTITION`` froze, derived from it rather than typed.

    The final block's end is the research boundary itself, which is what makes
    2025 partial. Deriving the labels from ``TEMPORAL_PARTITION.blocks`` means a
    seventh block cannot appear here without the preregistration hash moving.
    """
    labels = TEMPORAL_PARTITION["blocks"]
    blocks: list[CalendarBlock] = []
    for label in labels:
        year = int(label.split("-")[0])
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        start_ns = int(start.timestamp() * 1_000_000_000)
        end_ns = min(int(end.timestamp() * 1_000_000_000), RESEARCH_BOUNDARY_NS)
        blocks.append(CalendarBlock(label=label, start_ns=start_ns, end_exclusive_ns=end_ns))
    if len(blocks) != TEMPORAL_PARTITION["inferential_units"]:
        raise BlockError(
            f"{len(blocks)} blocks built from a partition declaring "
            f"{TEMPORAL_PARTITION['inferential_units']} inferential units"
        )
    return tuple(blocks)


@dataclass(frozen=True)
class OpeningSearch:
    """Where a block actually opened, and how much of it was skipped to get there.

    ``MARKLESS_LIQUIDATION_VALIDITY_POLICY.evidence_requirement`` asks for exactly
    these facts — the reason, the skipped time and instant count, and the realised
    opening instant against the calendar boundary — so they are produced as a
    value rather than logged.
    """

    block_label: str
    calendar_start_ns: int
    opened_at_ns: int
    skipped_instants: int
    #: Why the first candidate was not admissible. Under A2R2 this can only ever
    #: be ``OPENING_DELAY_EXECUTION_ABSENT`` — a leg supplied no row — and it is
    #: ``None`` when the block opened at its calendar boundary. A markless state
    #: may never appear here; ``MARKLESS_LIQUIDATION_VALIDITY_POLICY
    #: .evidence_requirement`` says so in terms.
    reason: str | None

    #: **Always False, and reported rather than inferred.** The frozen evidence
    #: requirement asks for "an explicit statement that the opening decision
    #: consulted no mark row, reported as a flag rather than left to be inferred
    #: from a zero count" — because a zero count is equally consistent with "the
    #: mark was consulted and never bound" and with "the mark was never consulted",
    #: and only the second is A2R2.
    OPENING_CONSULTED_MARK = False

    @property
    def skipped_ns(self) -> int:
        return self.opened_at_ns - self.calendar_start_ns

    @property
    def delayed(self) -> bool:
        return self.opened_at_ns != self.calendar_start_ns

    def as_dict(self) -> dict[str, object]:
        return {
            "block": self.block_label,
            "calendar_start_ns": self.calendar_start_ns,
            "opened_at_ns": self.opened_at_ns,
            "delayed": self.delayed,
            "skipped_instants": self.skipped_instants,
            "skipped_ns": self.skipped_ns,
            "reason": self.reason,
            # A2R2's evidence requirement, emitted as a fact rather than left to
            # be deduced from the absence of a mark-shaped field.
            "opening_consulted_mark": self.OPENING_CONSULTED_MARK,
            # Stated in the evidence itself so no reader has to infer it, and so
            # nothing downstream can quietly assume the opposite. An opening
            # delayed for an ABSENT EXECUTION ROW moves basis_at_entry and the
            # accrual window exactly as the retired mark rule would have, and
            # neither direction is claimed.
            "economic_direction": "INDETERMINATE EX ANTE",
        }


@dataclass(frozen=True)
class BlockRun:
    """One block's assembled inputs and the accounting result they produced.

    :attr:`quotes` and :attr:`settlements` are RETAINED rather than discarded, and
    that is what makes the stress variants honest. S1 doubles frictions and S3
    perturbs the basis; neither is a statement about which hours the sources
    published, so neither may move the opening instant or reshape the held window.
    Re-evaluating the retained series guarantees that structurally — a variant
    that re-ran the opening search could silently open somewhere else and report
    the difference as a cost effect.
    """

    block: CalendarBlock
    opening: OpeningSearch
    result: BlockResult
    #: How many settlements fell inside this block's CALENDAR span and were
    #: therefore offered to it. Not the same number as the one below, and
    #: deliberately kept apart from it: a block is offered every settlement in its
    #: year and CHARGED only those inside ``open < settlement <= close``, minus
    #: any after a liquidation trigger.
    settlements_priced: int
    held_instants: int
    quotes: tuple[Quote, ...] = ()
    settlements: tuple[FundingSettlement, ...] = ()

    @property
    def mark_substituted_settlements(self) -> int:
        """Settlements this block ACTUALLY APPLIED on a substituted notional base.

        Read from the accounting result rather than injected, which is the whole
        of the repair. It used to be a screen-wide total handed identically to
        every block, so a screen with ONE substituted settlement reported six —
        the evidence layer sums the per-block figures, and summing one global
        number six times multiplies it by the block count. Now each block reports
        what its own position was charged, and the sum is exact by construction.
        """
        return self.result.mark_substituted_settlements


@dataclass(frozen=True)
class ScreenRun:
    """Every block, or nothing at all.

    On a terminal source insufficiency :attr:`blocks` is EMPTY. That is not
    tidiness: A2 requires that "any block economics computed before the refusal
    fires are NOT a result ... not written as primary evidence, not reported as a
    partial answer, and do not enter any gate". An object that carried the
    survivors would be one lazy caller away from averaging them.
    """

    blocks: tuple[BlockRun, ...]
    openings: tuple[OpeningSearch, ...]
    terminal: SourceInsufficiency | None = None

    @property
    def evaluable(self) -> bool:
        return self.terminal is None

    @property
    def result_state(self) -> str | None:
        """The terminal label, or ``None`` while the screen is still evaluable."""
        return None if self.terminal is None else self.terminal.result_state


# ---------------------------------------------------------------------------
# The opening search
# ---------------------------------------------------------------------------


def find_opening_instant(aligned: AlignedSources, block: CalendarBlock) -> OpeningSearch:
    """The first EXECUTION-VALID instant at or after the block boundary — **A2R2**.

    Forward only, inside this block only, strictly before the research boundary
    only. An admissible instant is one at which BOTH LEGS supply a row, so a fill
    can be priced at their opens. That is the whole predicate.

    **The mark is not consulted, and that is the amendment.** A2 and A2R1 required
    the liquidation mark here too, on the reasoning that bar 0 is a held bar. Bar 0
    *is* a held bar — that has not changed — but the mark row's EXISTENCE is
    established only after bar 0 completes, so requiring it at the opening instant
    decided the entry from a fact unavailable there. A2R2 retires that rule: the
    position opens at the first fillable instant, and bar 0's mark is checked when
    bar 0 is evaluated as a held bar, in :func:`build_quotes`. If it is missing
    there, the SCREEN terminates — the entry does not move.

    The decision reads PRESENCE ONLY, and now only the presence of the two
    execution rows. Nothing here compares a price, and nothing here asks a question
    whose answer arrives after ``t``.

    One failure remains, and it is not A2R2's: a block at which NO instant supplies
    both legs could not be OPENED by the CONSTRUCTION, which is the case
    ``VIABILITY_GATE.excluded_blocks`` already owns — "required source rows absent
    or invalid at every candidate instant". That is signalled by raising
    :class:`BlockError`, and it is NOT a source-insufficiency refusal: nothing
    about the mark has been established, and the excluded-block rule is not
    broadened by A2R2 any more than it was by A2.
    """
    instants = list(
        grid_instants(block.start_ns, min(block.end_exclusive_ns, RESEARCH_BOUNDARY_NS))
    )
    if len(instants) < 2:
        raise BlockError(f"block {block.label} has fewer than two grid instants")
    # The last instant is the INTENDED CLOSE; a position needs an earlier bar to
    # open at, so candidates stop one short of it.
    candidates = instants[:-1]

    skipped = 0
    first_reason: str | None = None
    for instant in candidates:
        validity = aligned.instant_validity(instant)
        if validity.valid_for_opening:
            return OpeningSearch(
                block_label=block.label,
                calendar_start_ns=block.start_ns,
                opened_at_ns=instant,
                skipped_instants=skipped,
                reason=first_reason,
            )
        # The ONLY reason an opening may be delayed under A2R2. It is named from
        # the frozen constant rather than spelled here, so a markless state cannot
        # be substituted into this field by an edit that looks harmless.
        if first_reason is None:
            first_reason = OPENING_DELAY_EXECUTION_ABSENT
        skipped += 1

    raise BlockError(
        f"block {block.label} has no instant at which both legs supply a row, so the "
        "position could not be opened for a reason VIABILITY_GATE.excluded_blocks already "
        "covers. Nothing has been established about the mark: under A2R2 the mark is never "
        "consulted before an open, so this refusal cannot be a mark-coverage refusal "
        "wearing a construction refusal's clothes."
    )


# ---------------------------------------------------------------------------
# The held window
# ---------------------------------------------------------------------------


def held_and_exit_instants(
    block: CalendarBlock, opening_ns: int
) -> tuple[tuple[int, ...], int]:
    """The hourly grid the position is exposed to, and the instant it exits at.

    Held bars are ``opening_ns`` up to but excluding the exit, and the exit is the
    block's LAST hourly grid instant — ``POSITION_LIFECYCLE.close_instant``'s
    "intended close ... strictly before the block end and strictly before the
    research boundary". Both come from the calendar, never from the rows.
    """
    instants = list(
        grid_instants(block.start_ns, min(block.end_exclusive_ns, RESEARCH_BOUNDARY_NS))
    )
    exit_ns = instants[-1]
    if opening_ns >= exit_ns:
        raise BlockError(
            f"block {block.label}: opening instant {opening_ns} is not before the intended "
            f"close {exit_ns}"
        )
    held = tuple(instant for instant in instants if opening_ns <= instant < exit_ns)
    return held, exit_ns


def build_quotes(
    aligned: AlignedSources, block: CalendarBlock, opening_ns: int
) -> tuple[Quote, ...]:
    """Every quote from the open to the exit, with no hour permitted to be missing.

    Each HELD hour must be valid for holding, and **the first held hour is bar 0**
    — the opening instant itself. Under A2R2 the opening search no longer
    guarantees bar 0 carries a mark, so this is where a mark-less bar 0 is caught,
    and it is caught as what it is: a held bar the liquidation model cannot audit.

    The first held hour that is not valid raises the terminal
    :class:`SourceInsufficiency` — it is not skipped, the series is not closed
    early to route around it, and the OPENING INSTANT IS NOT MOVED to make the bar
    stop being held. All three would report a holding period the liquidation model
    never audited, and the third is precisely the acausal rule A2R2 retired.

    The EXIT hour is checked for execution validity only. That exemption is
    ``MARKLESS_LIQUIDATION_VALIDITY_POLICY.exit_bar``: the position closes at that
    bar's OPEN, before its post-open high or close exists, so there is no intra-bar
    window at the exit the position was exposed through. Both legs' opens are
    still required there, and an exit that cannot be priced is not exempted from
    anything — it becomes ``POSITION_LIFECYCLE.close_instant``'s UNCLOSED case,
    which amendment A1 governs, and which :func:`run_block` reports.
    """
    held, exit_ns = held_and_exit_instants(block, opening_ns)
    quotes: list[Quote] = []
    for instant in held:
        validity = aligned.instant_validity(instant)
        if not validity.valid_for_holding:
            raise SourceInsufficiency(
                MARKLESS_STATE_HELD,
                block.label,
                (
                    f"block {block.label}: the position was open across grid instant "
                    f"{instant}, and that hour is missing {list(validity.missing)}. "
                    "MARGIN_AND_LIQUIDATION.liquidation_check requires its inequality "
                    "evaluated at EVERY hourly grid instant while the position is open, and "
                    "A2 authorises exactly two sources for it — the mark HIGH, else the "
                    "mark CLOSE — with no spot, perpetual, REST, cross-venue, reconstructed "
                    "or zero surrogate. The required risk quantity is therefore not "
                    "computable for an hour the position was genuinely exposed through, so "
                    "the SCREEN terminates NOT EVALUABLE. This is source insufficiency, not "
                    "an observed economic failure. If this is bar 0, the opening instant is "
                    "still NOT moved: A2R2 retired the rule that would have moved it, "
                    "because the mark row's existence is not knowable at the opening "
                    "instant."
                ),
                instant_ns=instant,
                missing=validity.missing,
            )
        quotes.append(aligned.quote(instant, require_mark=True))
    exit_validity = aligned.instant_validity(exit_ns)
    if not exit_validity.valid_for_exit:
        # POSITION_LIFECYCLE.close_instant: the search for a valid exit runs
        # forward "at or after" the intended close and is bounded by the block
        # end. The intended close IS the block's last hour, so there is nothing
        # after it to move to, and the block is UNCLOSED — amendment A1's other
        # cause, the one p13_carry records as belonging to this runner.
        raise NoValidExit(
            block.label,
            (
                f"block {block.label}: the intended close at {exit_ns} is missing "
                f"{list(exit_validity.missing)}, and POSITION_LIFECYCLE.close_instant "
                "bounds the search for a later valid exit by the block end, so no "
                "permitted fill exists. The block is UNCLOSED under amendment A1."
            ),
            held=tuple(quotes),
            instant_ns=exit_ns,
        )
    quotes.append(aligned.quote(exit_ns, require_mark=False))
    return tuple(quotes)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_block(
    aligned: AlignedSources,
    block: CalendarBlock,
    *,
    allocation: Allocation,
    costs: Costs,
    venue: Venue,
    min_settlements: int,
    settlements: Sequence[FundingSettlement],
    isolated: bool = False,
) -> BlockRun:
    """Assemble one block and hand it to the accounting engine.

    Raises :class:`SourceInsufficiency` — never returns a degraded result — when
    A2R2's terminal branch fires, so a caller cannot accidentally treat a
    terminated block as a measured one.
    """
    opening = find_opening_instant(aligned, block)
    # Only settlements inside this block's calendar span are this block's; the
    # engine narrows further to the holding window, and applies the frozen
    # boundary tie rule at the close instant.
    within = tuple(s for s in settlements if block.contains(s.instant_ns))
    try:
        quotes = build_quotes(aligned, block, opening.opened_at_ns)
    except NoValidExit as unclosed:
        return _unclosed_run(
            block,
            opening,
            unclosed,
            allocation=allocation,
            costs=costs,
            venue=venue,
            min_settlements=min_settlements,
            settlements=within,
            isolated=isolated,
        )
    result = evaluate_block(
        block.label,
        quotes,
        within,
        allocation,
        costs,
        venue,
        min_settlements,
        isolated=isolated,
    )
    return BlockRun(
        block=block,
        opening=opening,
        result=result,
        settlements_priced=len(within),
        held_instants=len(quotes) - 1,
        quotes=quotes,
        settlements=within,
    )


def _unclosed_run(
    block: CalendarBlock,
    opening: OpeningSearch,
    unclosed: NoValidExit,
    *,
    allocation: Allocation,
    costs: Costs,
    venue: Venue,
    min_settlements: int,
    settlements: Sequence[FundingSettlement],
    isolated: bool,
) -> BlockRun:
    """Amendment A1's UNCLOSED block, accrued over the bars that WERE held.

    ``nn.p13_carry`` records that this cause "belongs to the block runner, which
    does not exist yet" and exposes :func:`~nn.p13_carry.unclosed_block_result` for
    whatever produces it. This is that caller.

    The accrual mirrors the engine's held loop exactly — every held bar applies its
    due settlements, samples the excursion and records which mark series answered
    the liquidation test — because A1 requires the funding, fees and slippage
    "actually incurred up to that instant" reported as facts. What it does NOT do
    is invent a close: exit basis, basis PnL, net PnL and net return come back
    ``NOT_DETERMINABLE``.
    """
    quotes = unclosed.held
    if not quotes:
        raise BlockError(f"block {block.label}: unclosed with no held bars")
    entry = quotes[0]
    try:
        position = open_carry(entry, allocation, costs, venue)
    except CarryError as exc:
        return _excluded_run(block, f"not opened: {exc}")

    due = {
        settlement.instant_ns: settlement
        for settlement in settlements
        if settlement.instant_ns > entry.instant_ns
    }
    worst = position.equity_at(entry.spot_fill, entry.perp_fill) - allocation.total_capital
    touches: dict[str, int] = {name: 0 for name in TOUCH_SOURCES}
    settled = 0
    # Only what was ACCRUED before the terminal held horizon. An UNCLOSED block
    # stops here, so settlements it never reached are neither applied nor counted.
    substituted = 0
    liquidated_at: int | None = None
    for quote in quotes:
        for instant in sorted(k for k in list(due) if k <= quote.instant_ns):
            settlement = due.pop(instant)
            apply_funding(position, settlement)
            settled += 1
            substituted += settlement.notional_substituted
        worst = min(worst, position.equity(quote) - allocation.total_capital)
        touches[quote.liquidation_touch_source] += 1
        if is_liquidated(position, quote, venue, isolated):
            liquidated_at = quote.instant_ns
            break

    result = unclosed_block_result(
        block.label,
        (
            f"UNCLOSED: {unclosed}. Amendment A1 reports the funding, fees and slippage "
            "incurred as facts and the close-dependent economics as NOT DETERMINABLE, and "
            "terminates the screen INVALID."
        ),
        settlements=settled,
        quantity=position.quantity,
        basis_entry=entry.fill_basis,
        funding_received=position.funding_received,
        funding_paid=position.funding_paid,
        fees=position.fees,
        slippage=position.slippage,
        max_adverse_excursion_pnl=worst,
        total_capital=allocation.total_capital,
        thin_sample=settled < min_settlements,
        liquidated=liquidated_at is not None,
        liquidation_instant_ns=liquidated_at,
        liquidation_touch_provenance=LiquidationTouchProvenance(**touches),
        held_bars=len(quotes),
        mark_substituted_settlements=substituted,
    )
    return BlockRun(
        block=block,
        opening=opening,
        result=result,
        settlements_priced=len(settlements),
        held_instants=len(quotes),
        quotes=(),
        settlements=tuple(settlements),
    )


def run_screen(
    aligned: AlignedSources,
    *,
    allocation: Allocation,
    costs: Costs,
    venue: Venue,
    min_settlements: int,
    settlements: Sequence[FundingSettlement],
    blocks: Iterable[CalendarBlock] | None = None,
    isolated: bool = False,
) -> ScreenRun:
    """Every block in chronological order, or a terminal refusal and nothing else.

    A terminal :class:`SourceInsufficiency` anywhere discards every block already
    computed. That is A2's ``partial_numbers_are_not_a_result`` implemented
    rather than promised: the returned :class:`ScreenRun` carries no blocks at
    all, so there is nothing for a gate to read even if one forgot to check
    :attr:`ScreenRun.evaluable`.

    A block that could not be OPENED for a construction reason — no rows at all,
    a quantity flooring to zero, a leg below minimum notional — is NOT terminal.
    It is ``VIABILITY_GATE.excluded_blocks``' own case, and it is carried through
    as a not-opened :class:`~nn.p13_carry.BlockResult` for the gate to exclude.
    """
    chosen = tuple(blocks) if blocks is not None else calendar_blocks()
    runs: list[BlockRun] = []
    openings: list[OpeningSearch] = []
    for block in chosen:
        try:
            run = run_block(
                aligned,
                block,
                allocation=allocation,
                costs=costs,
                venue=venue,
                min_settlements=min_settlements,
                settlements=settlements,
                isolated=isolated,
            )
        except SourceInsufficiency as terminal:
            # Everything computed so far is discarded, deliberately and visibly.
            return ScreenRun(blocks=(), openings=tuple(openings), terminal=terminal)
        except BlockError as excluded:
            runs.append(_excluded_run(block, str(excluded)))
            continue
        runs.append(run)
        openings.append(run.opening)
    return ScreenRun(blocks=tuple(runs), openings=tuple(openings))


def _excluded_run(block: CalendarBlock, reason: str) -> BlockRun:
    """A block the CONSTRUCTION could not open, in the shape the gate excludes.

    Its economics are ``NOT_DETERMINABLE`` rather than zero — the engine's own
    not-opened template — so a gate that forgets ``opened`` raises instead of
    averaging a zero return it never measured.
    """
    from nn.p13_carry import NOT_DETERMINABLE, LiquidationTouchProvenance

    empty = BlockResult(
        label=block.label,
        opened=False,
        reason=f"not opened: {reason}",
        settlements=0,
        quantity=Decimal("0"),
        basis_entry=NOT_DETERMINABLE,
        basis_exit=NOT_DETERMINABLE,
        basis_pnl=NOT_DETERMINABLE,
        funding_received=Decimal("0"),
        funding_paid=Decimal("0"),
        fees=Decimal("0"),
        slippage=Decimal("0"),
        rebalance_cost=Decimal("0"),
        net_pnl=NOT_DETERMINABLE,
        net_return=NOT_DETERMINABLE,
        liquidated=False,
        max_adverse_excursion_pnl=NOT_DETERMINABLE,
        max_adverse_excursion=NOT_DETERMINABLE,
        thin_sample=True,
        liquidation_touch_provenance=LiquidationTouchProvenance(),
    )
    opening = OpeningSearch(
        block_label=block.label,
        calendar_start_ns=block.start_ns,
        opened_at_ns=block.start_ns,
        skipped_instants=0,
        reason=None,
    )
    # ``empty`` carries mark_substituted_settlements=0 by construction: a block
    # that never opened charged no settlement to any position, so it contributes
    # nothing to the screen-wide substitution total.
    return BlockRun(
        block=block,
        opening=opening,
        result=empty,
        settlements_priced=0,
        held_instants=0,
    )


# Re-exported so callers assembling stress variants do not reach into the engine
# for the one constant that defines the grid.
BAR_NS = NOMINAL_BAR_NS
