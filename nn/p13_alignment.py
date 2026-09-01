"""Aligning P13's four source families onto one causal hourly grid.

The accounting engine takes :class:`~nn.p13_carry.Quote` and
:class:`~nn.p13_carry.FundingSettlement`. This module is what turns four
independently published archives into those, and it is where amendment **A2**'s
source-validity vocabulary actually lives, at revision **A2R2**.

**Nothing is ever forward-filled.** A missing hour stays missing. Carrying the
previous candle forward would manufacture an observation the venue never
published and, worse, would manufacture a LIQUIDATION TEST that never happened —
the precise substitution ``MARKLESS_LIQUIDATION_VALIDITY_POLICY`` forbids. The one
"at or immediately preceding" lookup in this module is the funding notional base,
and that phrasing is the frozen design's own
(``BASIS_DEFINITION.which_series_plays_which_role``), not a convenience.

**Two different validities, because the frozen design has two.**

* :func:`execution_validity` — spot and perpetual rows present. This is what a
  FILL needs, at either end of the position.
* :func:`liquidation_validity` — a mark row present. This is what a HELD BAR
  needs, because ``MARGIN_AND_LIQUIDATION.liquidation_check`` evaluates its
  inequality "at every hourly grid instant while the position is open" and A2
  gives that check exactly two authorised sources, the mark HIGH and the mark
  CLOSE.

An OPENING instant needs only the FIRST — this is **amendment A2R2**, and it is
the one thing about this module that changed. A NORMAL EXIT instant likewise needs
only the first, because ``MARKLESS_LIQUIDATION_VALIDITY_POLICY.exit_bar`` records
that the position closes at that bar's OPEN, before its post-open high or close
exists. Bar 0 is still a HELD bar and still needs its mark; what changed is WHEN
that is asked, not whether.

**Why the mark left the opening predicate.** A2 and A2R1 rejected a candidate
opening instant whose mark row was absent. Reading presence rather than a value
was not enough to make that causal. A ``markPriceKlines`` row is stamped by candle
OPEN ``t``, but the fact that a completed row for that bar exists in the published
archive at all is established only AFTER the bar completes — so "at ``t``, refuse
to open because the same-bar mark row is absent" conditions the entry instant on
information that does not exist at ``t``. That is SOURCE-AVAILABILITY LOOK-AHEAD,
and A2R2 removes it. :attr:`InstantValidity.valid_for_opening` is therefore
execution-only, and the mark is checked when bar 0 is EVALUATED as a held bar,
where its absence terminates the screen instead of moving the entry.

**The value guard is still here and still necessary.** :func:`instant_validity`
reads presence only: it never compares, thresholds or otherwise consults a mark
high, a mark close, a spot close or a perpetual close, so no numeric fact about how
a candle turned out can reach any decision. A2R2 does not weaken that; it adds the
second, stricter requirement that the ENTRY decision not read the mark's presence
either. The frozen ``POSITION_LIFECYCLE.validity_definition`` still supplies the
execution predicate — "the row is present in every required source, every
preregistered field is present, every price is strictly positive, and no duplicate
row makes the instant ambiguous" — the last three of which :mod:`nn.p13_sources`
has already applied by withholding unusable rows.

**Object availability is a funding concept and only a funding concept.**
``MARK_PRICE_FALLBACK`` is explicitly "PER ARCHIVE OBJECT, not all-or-nothing", so
:attr:`AlignedSources.published_mark_periods` exists and drives the funding
notional substitution. It is never consulted by the liquidation path, because A2
authorises no liquidation surrogate of any kind, and it is never consulted by the
OPENING path either, because A2R2 lets nothing about the mark reach the entry
decision. The three mechanisms share no code and no state on purpose.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable, Iterator, Mapping, Sequence

from nn.p13_carry import NOMINAL_BAR_NS, FundingSettlement, Quote
from nn.p13_sources import FundingRow, KlineRow, ObjectProvenance

__all__ = [
    "AlignmentError",
    "SPOT",
    "PERPETUAL",
    "MARK",
    "InstantValidity",
    "AlignedSources",
    "FundingBase",
    "grid_instants",
    "monthly_period",
]


class AlignmentError(RuntimeError):
    """The four families cannot be aligned into an honest causal grid."""


SPOT = "spot"
PERPETUAL = "perpetual"
MARK = "mark"

#: The sources a FILL needs, and the source a HELD BAR needs, kept apart because
#: the frozen design keeps them apart. Named rather than inlined so a reader can
#: see that the mark is absent from the first tuple and alone in the second.
EXECUTION_SOURCES: tuple[str, ...] = (SPOT, PERPETUAL)
LIQUIDATION_SOURCES: tuple[str, ...] = (MARK,)


def grid_instants(start_ns: int, end_exclusive_ns: int) -> Iterator[int]:
    """Every nominal hourly grid instant in ``[start, end)``.

    The REFERENCE grid, generated from the calendar rather than read off whichever
    rows happen to exist. That direction matters: a held-window completeness check
    driven by the rows present could never notice an hour for which no row was
    published at all, which is the failure it exists to catch.
    """
    if start_ns > end_exclusive_ns:
        raise AlignmentError(f"grid start {start_ns} is after its end {end_exclusive_ns}")
    instant = start_ns
    while instant < end_exclusive_ns:
        yield instant
        instant += NOMINAL_BAR_NS


@dataclass(frozen=True)
class InstantValidity:
    """Whether one grid instant is usable, and for what.

    Three answers rather than one boolean, because the frozen design asks three
    different questions at three different points in a block's life — may a position
    OPEN here, was it HELD through here, may it CLOSE here — and collapsing them
    would make both the exit-bar exemption and A2R2's opening rule impossible to
    express.
    """

    instant_ns: int
    #: Which of the required sources supply no row here. Empty means all do.
    missing: tuple[str, ...]

    @property
    def has_execution(self) -> bool:
        """Both legs supply a row, so a fill can be priced at their opens."""
        return not any(source in self.missing for source in EXECUTION_SOURCES)

    @property
    def has_liquidation_mark(self) -> bool:
        """A mark row exists, so the frozen liquidation test can be performed."""
        return MARK not in self.missing

    @property
    def valid_for_holding(self) -> bool:
        """Usable as a HELD bar: priceable and testable.

        Bar 0 is a held bar, so this is what bar 0 must satisfy — but it is NOT
        what the OPENING SEARCH may test, because at the opening instant the
        answer is not yet knowable. See :attr:`valid_for_opening`.
        """
        return self.has_execution and self.has_liquidation_mark

    @property
    def valid_for_opening(self) -> bool:
        """Usable as an OPENING instant: priceable, and nothing else — **A2R2**.

        Named separately from :attr:`valid_for_exit` even though the two are the
        same predicate, because they answer different frozen questions and a
        single shared name would make the A2R2 rule invisible at the call site.

        **The mark is deliberately absent from this.** A2R2 retires the rule that
        rejected an opening instant whose same-bar mark row was missing: the mark
        row is stamped by candle OPEN ``t`` but its existence is established only
        after the bar completes, so conditioning the ENTRY on it is
        source-availability look-ahead. Bar 0 still needs its mark — as a HELD
        bar, checked once the bar is complete, by :func:`nn.p13_blocks.build_quotes`
        — and a bar 0 that lacks one terminates the screen rather than moving the
        entry.
        """
        return self.has_execution

    @property
    def valid_for_exit(self) -> bool:
        """Usable as a NORMAL EXIT bar: priceable, and not required to be testable."""
        return self.has_execution


@dataclass(frozen=True)
class FundingBase:
    """The notional base one settlement was charged on, and where it came from."""

    instant_ns: int
    price: Decimal
    #: ``MARK`` when the venue's own notional price was used, ``SPOT`` when
    #: ``MARK_PRICE_FALLBACK`` substituted the spot close because the month's
    #: markPriceKlines OBJECT is unpublished. Recorded per settlement because
    #: ``MARK_PRICE_FALLBACK.reporting_granularity`` asks for "a count of
    #: settlements that used the substituted base".
    source: str
    #: The instant of the candle the base was taken from — "at or immediately
    #: preceding" the settlement, which is usually but not always the settlement
    #: hour itself.
    base_instant_ns: int


@dataclass(frozen=True)
class AlignedSources:
    """The four families, indexed by instant, with nothing filled in.

    Construct with :meth:`build`. The mappings are deliberately plain lookups: an
    instant either has a row or it does not, and there is no accessor that will
    invent one.
    """

    spot: Mapping[int, KlineRow]
    perpetual: Mapping[int, KlineRow]
    mark: Mapping[int, KlineRow]
    funding: tuple[FundingRow, ...]
    #: The monthly periods for which the markPriceKlines OBJECT was published.
    #: Drives ``MARK_PRICE_FALLBACK`` and NOTHING else — see the module docstring.
    published_mark_periods: frozenset[str]
    provenance: tuple[ObjectProvenance, ...] = ()
    _mark_instants: tuple[int, ...] = field(default=(), repr=False)
    _spot_instants: tuple[int, ...] = field(default=(), repr=False)

    @classmethod
    def build(
        cls,
        *,
        spot: Iterable[KlineRow],
        perpetual: Iterable[KlineRow],
        mark: Iterable[KlineRow],
        funding: Iterable[FundingRow],
        published_mark_periods: Iterable[str],
        provenance: Iterable[ObjectProvenance] = (),
    ) -> "AlignedSources":
        spot_rows = {row.instant_ns: row for row in spot}
        perp_rows = {row.instant_ns: row for row in perpetual}
        mark_rows = {row.instant_ns: row for row in mark}
        return cls(
            spot=spot_rows,
            perpetual=perp_rows,
            mark=mark_rows,
            funding=tuple(sorted(funding, key=lambda row: row.instant_ns)),
            published_mark_periods=frozenset(published_mark_periods),
            provenance=tuple(provenance),
            _mark_instants=tuple(sorted(mark_rows)),
            _spot_instants=tuple(sorted(spot_rows)),
        )

    # -- validity ---------------------------------------------------------

    def instant_validity(self, instant_ns: int) -> InstantValidity:
        """Which required sources supply a row at ``instant_ns``.

        **Presence only.** No price value is read, compared or thresholded here,
        so no numeric fact about a candle's eventual high or close can reach the
        decision of whether a position may open at its instant. That is the whole
        of the look-ahead guard, and it is a property of this function's body
        rather than a promise made about it.
        """
        missing = []
        if instant_ns not in self.spot:
            missing.append(SPOT)
        if instant_ns not in self.perpetual:
            missing.append(PERPETUAL)
        if instant_ns not in self.mark:
            missing.append(MARK)
        return InstantValidity(instant_ns=instant_ns, missing=tuple(missing))

    # -- quotes -----------------------------------------------------------

    def quote(self, instant_ns: int, *, require_mark: bool = True) -> Quote:
        """The :class:`~nn.p13_carry.Quote` at ``instant_ns``, or refuse.

        ``require_mark=False`` is the EXIT-BAR case and nothing else — NOT the
        opening case, which is a HELD bar and needs its mark like any other. It does not
        weaken the liquidation ladder: a quote built without a mark simply has no
        authorised touch, and :class:`~nn.p13_carry.Quote` itself raises if
        anything ever asks it for one. The exemption is therefore enforced by the
        engine that already refuses, not by a flag this module is trusted about.
        """
        validity = self.instant_validity(instant_ns)
        if not validity.has_execution:
            raise AlignmentError(
                f"instant {instant_ns} is missing {list(validity.missing)}; both legs must "
                "supply a row before a fill can be priced at their opens"
            )
        if require_mark and not validity.has_liquidation_mark:
            raise AlignmentError(
                f"instant {instant_ns} carries no mark row, so the frozen liquidation test "
                "has neither of its two authorised sources. A2 authorises the mark HIGH "
                "and the mark CLOSE and NOTHING else, so no quote is built here rather than "
                "one that would be tested against a series the design never gave it."
            )
        spot_row = self.spot[instant_ns]
        perp_row = self.perpetual[instant_ns]
        mark_row = self.mark.get(instant_ns)
        return Quote(
            instant_ns=instant_ns,
            spot=spot_row.close,
            perp=perp_row.close,
            spot_open=spot_row.open,
            perp_open=perp_row.open,
            mark=mark_row.close if mark_row is not None else None,
            mark_high=mark_row.high if mark_row is not None else None,
        )

    # -- funding ----------------------------------------------------------

    def funding_base(self, instant_ns: int, *, period: str) -> FundingBase:
        """The notional base for a settlement at ``instant_ns``.

        ``BASIS_DEFINITION.which_series_plays_which_role`` fixes this exactly: the
        MARK candle CLOSE at or immediately preceding the settlement, and under
        ``MARK_PRICE_FALLBACK`` the SPOT candle CLOSE at or immediately preceding
        it instead. Both are knowable at the settlement instant, which is why the
        frozen text can use them without a lookahead.

        The fallback is chosen by whether the month's markPriceKlines OBJECT was
        published — ``MARK_PRICE_FALLBACK``'s own per-object trigger — and never
        by anything economic. A published month whose mark series simply has no
        preceding candle at all is a refusal, not a silent fallback: the frozen
        substitution is authorised by an unpublished OBJECT, and stretching it to
        cover a row-level hole would be extending a frozen rule rather than
        applying it.
        """
        if period in self.published_mark_periods:
            base_instant = _at_or_preceding(self._mark_instants, instant_ns)
            if base_instant is None:
                raise AlignmentError(
                    f"settlement at {instant_ns}: the markPriceKlines object for {period} is "
                    "published, so MARK_PRICE_FALLBACK is not triggered, but no mark candle "
                    "at or preceding the settlement exists to price the notional on"
                )
            return FundingBase(
                instant_ns=instant_ns,
                price=self.mark[base_instant].close,
                source=MARK,
                base_instant_ns=base_instant,
            )
        base_instant = _at_or_preceding(self._spot_instants, instant_ns)
        if base_instant is None:
            raise AlignmentError(
                f"settlement at {instant_ns}: MARK_PRICE_FALLBACK substitutes the spot close "
                "as the funding notional base, and no spot candle at or preceding the "
                "settlement exists either"
            )
        return FundingBase(
            instant_ns=instant_ns,
            price=self.spot[base_instant].close,
            source=SPOT,
            base_instant_ns=base_instant,
        )

    def settlements(
        self, rows: Sequence[FundingRow]
    ) -> tuple[tuple[FundingSettlement, ...], tuple[FundingBase, ...]]:
        """Price a run of funding rows onto their notional bases.

        Returns the settlements and, alongside them, the bases they were priced
        on — so a caller can report how many used the substituted base without
        recovering it from the numbers.

        A settlement's archive period is its own UTC month, computed here rather
        than passed in: the monthly object that publishes a row is the month the
        row falls in, and letting a caller supply a different answer would let the
        funding fallback apply to months it was not triggered for.
        """
        settlements: list[FundingSettlement] = []
        bases: list[FundingBase] = []
        for row in rows:
            base = self.funding_base(row.instant_ns, period=monthly_period(row.instant_ns))
            bases.append(base)
            settlements.append(
                FundingSettlement(
                    instant_ns=row.instant_ns,
                    rate=row.rate,
                    mark_price=base.price,
                    # The substitution travels WITH the settlement, so the count
                    # of settlements charged on a substituted base is taken by
                    # the evaluator that knows which were actually charged.
                    # Deciding it here AND counting it there would be two places
                    # deciding one thing.
                    notional_substituted=base.source != MARK,
                )
            )
        return tuple(settlements), tuple(bases)


def monthly_period(instant_ns: int) -> str:
    """The ``YYYY-MM`` monthly archive period an instant belongs to.

    A named function rather than an inline format so the mapping is nameable in a
    traceback: an off-by-one month here would silently move which settlements the
    funding fallback applied to.
    """
    moment = datetime.fromtimestamp(instant_ns / 1_000_000_000, tz=timezone.utc)
    return f"{moment.year:04d}-{moment.month:02d}"


def _at_or_preceding(sorted_instants: Sequence[int], instant_ns: int) -> int | None:
    """The largest element ``<= instant_ns``, or ``None``.

    Binary search rather than a scan because a block spans thousands of hours and
    this runs once per settlement; and ``bisect_right`` rather than a hand-rolled
    comparison because "at or immediately preceding" includes the instant itself,
    which is the case a strict-inequality search silently gets wrong.
    """
    position = bisect.bisect_right(sorted_instants, instant_ns)
    if position == 0:
        return None
    return sorted_instants[position - 1]
