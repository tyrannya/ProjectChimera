"""The one place a campaign configuration becomes Aegis's limits.

:class:`chimera.demo.config.DemoLimits` and :class:`chimera.risk.RiskLimits` are
separate types on purpose, and ``DemoLimits``' own docstring names the boundary:
"the runner is where the two meet: it is the thing that builds a ``RiskEngine``
from a parsed campaign config, and mapping one to the other in one visible place
is better than sharing a type that fits neither." This module is that one visible
place. Both the operator CLI (``tools/demo_run.py``) and the test harness
(``tests/demo_harness.py``) build their engine here and nowhere else, so the
configuration a campaign is *hashed under* and the configuration a campaign is
*enforced under* cannot be two different things.

They were two different things. ``tools/demo_run.py`` constructed
``RiskLimits(max_position_pct=limits.max_exposure_per_asset_pct,
risk_per_trade_pct=0.5)`` and stopped there, so twelve of section 7.4's thirteen
limits never reached the engine at all: a campaign whose file said
``max_drawdown_pct 0.05`` ran on the ``RiskLimits`` default of ``0.15``, one that
said ``max_leverage 1.0`` ran on ``3.0``, and one that said
``max_orders_per_minute 4`` ran on ``10``. The thirteenth reached the engine
under the wrong name -- ``max_exposure_per_asset_pct`` is a *cumulative per-pair*
ceiling in :meth:`chimera.risk.RiskEngine.evaluate_entry` and
``max_position_pct`` is a *per-order stake* ceiling in
:meth:`chimera.risk.RiskEngine.position_size` -- which left the per-pair ceiling
itself on its ``0.35`` default. ``tests/demo_harness.py`` reproduced the same
partial construction with its own literals, so no test could see any of it.

**Every field is accounted for, and the accounting is checked at import.**
:data:`DIRECT_LIMITS` is name-for-name, :data:`DERIVED_LIMITS` is the fields
section 7.4 does not configure but which a campaign limit must nonetheless move,
and :data:`UNCONFIGURED_LIMITS` is the rest, each with the reason it has no
campaign source. Between them they name every field of both dataclasses exactly
once, and :func:`check_exhaustive` refuses to import if that stops being true.
A limit added to either type is then a loud failure everywhere rather than a
silent default in one runner.

This module opens no socket and makes no market decision. :func:`risk_limits` is
a pure function of its argument; :func:`build_risk_engine` is the one place that
touches the campaign's state directory, and :func:`seed_or_reconcile_equity`
below is the one place that decides what a starting engine believes its equity
to be.
"""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping

from chimera.carry.ledger import CarryLedger, LoadOutcome
from chimera.demo.config import COUNT_LIMITS, DemoConfig, DemoLimits
from chimera.demo.risk_continuity import assess_risk_continuity
from chimera.risk import RiskEngine, RiskLimits, RiskStateLoad


class RiskWiringError(ValueError):
    """The two limit types no longer agree about what this module maps."""


#: Section 7.4's limits, each to the ``RiskLimits`` field that enforces it. Every
#: pair here is name-for-name AND meaning-for-meaning; the names matching is not
#: what makes a pair correct, so each is stated with its enforcement site:
#:
#: ``funding_adverse_streak_limit`` -> ``RiskEngine.note_funding_settlement``
#:     raises ``funding_halt`` at the Nth consecutive paid settlement, and
#:     ``evaluate_entry`` vetoes every increase while it is up.
#: ``loss_streak_limit`` -> ``record_trade_result`` opens a cooldown at the Nth
#:     consecutive loss. (NOT REACHABLE on the demo path today: nothing in
#:     ``chimera/`` or ``tools/`` calls ``record_trade_result``, so
#:     ``consecutive_losses`` never leaves zero. Mapped anyway, so that the
#:     campaign's value is already in force on the day something does.)
#: ``max_open_positions`` -> ``evaluate_entry`` vetoes a new pair once this many
#:     are open. Two, because the hedge is two legs.
#: ``max_orders_per_minute`` -> ``record_order`` HALTS above this many approvals
#:     in the rolling 60-second window.
#: ``cooldown_seconds`` -> the length of the cooldown ``record_trade_result``
#:     opens; ``evaluate_entry`` vetoes until it expires. (NOT REACHABLE on the
#:     demo path today, for the same reason as ``loss_streak_limit``: the
#:     cooldown is never opened, so the gate never closes.)
#: ``max_daily_loss_pct`` -> ``update_equity`` halts on this loss from the day's
#:     starting equity.
#: ``max_data_delay_s`` -> ``note_feed`` marks the feed stale past this delay and
#:     ``evaluate_entry`` vetoes on the mark. (The runner passes no
#:     ``data_delay_s`` to ``evaluate_entry``, so the mark is the demo's
#:     enforcement of this limit, not the direct comparison beside it.)
#: ``max_drawdown_pct`` -> ``update_equity`` halts on this drawdown from peak.
#: ``max_exposure_per_asset_pct`` -> ``evaluate_entry`` vetoes when this pair's
#:     CUMULATIVE exposure plus the new stake would pass the fraction of equity.
#: ``max_funding_cost_rate`` -> ``evaluate_entry``'s side-aware funding veto,
#:     ``sign(side) * rate`` (amendment A10). (NOT REACHABLE on the demo path
#:     today: ``chimera/carry/hedge.py`` calls ``execute_target`` without a
#:     ``funding_rate``, so ``evaluate_entry`` skips the whole funding branch.
#:     What the demo DOES enforce against adverse funding is
#:     ``funding_adverse_streak_limit``, whose settlements the runner really
#:     does report.)
#: ``max_leverage`` -> ``evaluate_entry`` vetoes above it, and ``position_size``
#:     caps the leverage it divides by.
#: ``max_total_exposure_pct`` -> ``evaluate_entry`` vetoes when the sum of all
#:     exposures plus the new stake would pass the fraction of equity.
#: ``min_liquidation_distance_pct`` -> ``evaluate_entry`` vetoes when the
#:     liquidation price is nearer than this fraction of the entry price.
DIRECT_LIMITS: Mapping[str, str] = {
    "cooldown_seconds": "cooldown_seconds",
    "funding_adverse_streak_limit": "funding_adverse_streak_limit",
    "loss_streak_limit": "loss_streak_limit",
    "max_daily_loss_pct": "max_daily_loss_pct",
    "max_data_delay_s": "max_data_delay_s",
    "max_drawdown_pct": "max_drawdown_pct",
    "max_exposure_per_asset_pct": "max_exposure_per_asset_pct",
    "max_funding_cost_rate": "max_funding_cost_rate",
    "max_leverage": "max_leverage",
    "max_open_positions": "max_open_positions",
    "max_orders_per_minute": "max_orders_per_minute",
    "max_total_exposure_pct": "max_total_exposure_pct",
    "min_liquidation_distance_pct": "min_liquidation_distance_pct",
}

#: ``RiskLimits`` fields section 7.4 does not name, but which a campaign limit
#: must still move -- because leaving them on a default would let the engine
#: enforce something the campaign file does not say.
#:
#: ``max_position_pct`` <- ``max_exposure_per_asset_pct``. The per-order stake
#:     ceiling. Set to the per-pair cumulative ceiling so that it is never the
#:     tighter of the two: the cumulative check in ``evaluate_entry`` is
#:     ``existing + stake``, so at equal fractions it binds first and the
#:     campaign's exposure limit is what actually decides. Left on its ``0.25``
#:     default it would silently cap every order at a quarter of equity, which
#:     section 7.4's ``1.0`` (`"the hedge holds equal notional on both legs"`)
#:     plainly does not intend. This is the mapping ``tools/demo_run.py`` had --
#:     correct in itself, and wrong only because it was made INSTEAD OF the
#:     name-for-name one rather than beside it.
#: ``max_funding_rate`` <- ``max_funding_cost_rate``. The SIGN-BLIND funding
#:     ceiling, applied when a caller names no position side. Section 7.2 fixes
#:     the relationship: the sign-blind check "vetoes a superset and so cannot
#:     approve anything the side-aware check would refuse". ``abs(rate) > c``
#:     is a superset of ``sign(side) * rate > c`` only while the two ceilings are
#:     equal or the sign-blind one is tighter, so it has to move with the
#:     campaign's ``max_funding_cost_rate`` rather than sit on a default that a
#:     tightened campaign would leave behind. Like the side-aware ceiling it
#:     shadows, this is NOT REACHABLE on the demo path today -- no caller passes
#:     a ``funding_rate`` at all, let alone one without a side -- so the
#:     invariant is maintained here for the day one does, not for today.
DERIVED_LIMITS: Mapping[str, str] = {
    "max_position_pct": "max_exposure_per_asset_pct",
    "max_funding_rate": "max_funding_cost_rate",
}

#: ``RiskLimits`` fields with no campaign source at all, and why. Each keeps its
#: ``RiskLimits`` default; naming them here is what makes that a decision rather
#: than an omission.
UNCONFIGURED_LIMITS: Mapping[str, str] = {
    "risk_per_trade_pct": (
        "there is no stop on the demo path for it to be about. A carry hedge has "
        "no stop-loss, and FuturesExecutor._implied_stop fabricates one at the "
        "midpoint of the sizing band purely so position_size's arithmetic is "
        "defined. See DEMO_RISK_PER_TRADE_PCT for the invariant that replaces it."
    ),
    "min_stop_distance_pct": (
        "the sizing band's lower edge. Section 7.4 configures no stop, and this "
        "edge is half of what _implied_stop takes its midpoint from."
    ),
    "max_stop_distance_pct": (
        "the sizing band's upper edge; the other half of _implied_stop's midpoint."
    ),
    "max_inference_staleness_s": (
        "section 7.2: 'stale inference | n/a in the demo (no model) | pass None'. "
        "Nothing on the demo path passes inference_age_s, so the guard is "
        "unreachable and any value configured for it would be a fiction."
    ),
}

#: What replaces a stop-loss fraction the demo cannot have.
#:
#: ``position_size`` computes ``equity * risk_per_trade_pct / stop_distance /
#: leverage`` and caps it at ``equity * max_position_pct``. On the demo path the
#: stop distance is invented by ``_implied_stop``, so the first term carries no
#: risk meaning; the campaign's exposure ceilings are what may decide a size. The
#: invariant is therefore that the CAP binds and the invented stop never does,
#: which holds whenever
#:
#:     risk_per_trade_pct >= max_position_pct * max_stop_distance_pct
#:                           * max(1.0, max_leverage)
#:
#: -- the worst case being the widest stop the band admits at the HIGHEST
#: effective leverage ``position_size`` will divide by. The leverage factor is
#: not decoration and the direction of it is easy to get backwards: dividing by
#: a larger number makes the stop-based stake SMALLER, so a higher ceiling is
#: what makes the invented stop, not the cap, decide. ``position_size`` clamps
#: with ``max(1.0, min(leverage, max_leverage))`` and ``evaluate_entry`` refuses
#: any ``leverage > max_leverage``, so ``max(1.0, max_leverage)`` is exactly the
#: largest divisor reachable. For the committed campaign, whose ``max_leverage``
#: is ``1.0``, that floor is ``1.0 * 0.15 * 1.0 = 0.15``; at ``max_leverage 4``
#: it would be ``0.60`` and this constant would no longer satisfy it.
#:
#: The value below is the one the demo path has always used, in
#: ``tools/demo_run.py``, ``tools/futures_dry_run.py`` and every carry
#: test; it is kept rather than retuned because retuning it would change what the
#: engine enforces, and it is stated here rather than in a runner because a
#: number that decides a size belongs where a reviewer looking for one will look.
#: :func:`sizing_cap_binds` is the machine-checked form, asserted by the tests.
#:
#: Section 7.4 does not name this quantity. If S2 wants it configured, that is a
#: campaign-config key and a preregistration decision, not a runner's to invent.
DEMO_RISK_PER_TRADE_PCT = 0.5

#: The four ``DemoLimits`` fields carried as counts. Imported rather than
#: restated so the two modules cannot disagree about which limits are integers.
_COUNTS = frozenset(COUNT_LIMITS)


def sizing_cap_binds(limits: RiskLimits) -> bool:
    """Whether ``max_position_pct`` -- not the invented stop -- caps a stake.

    See :data:`DEMO_RISK_PER_TRADE_PCT`. True when stop-based sizing cannot be
    the binding constraint for any stop distance the band admits, at any leverage
    the engine will accept.

    The three factors mirror ``RiskEngine.position_size`` exactly: it divides by
    ``max(1.0, min(leverage, max_leverage))``, so the largest divisor a caller can
    reach is ``max(1.0, max_leverage)`` and that is the case the cap has to
    survive.
    """
    return limits.risk_per_trade_pct >= (
        limits.max_position_pct * limits.max_stop_distance_pct * max(1.0, limits.max_leverage)
    )


def risk_limits(limits: DemoLimits) -> RiskLimits:
    """One campaign's limits as Aegis enforces them. Pure; no I/O, no clock.

    Every ``RiskLimits`` field is passed explicitly, including the ones that keep
    a default: ``RiskLimits.from_dict`` drops keys it does not recognise, and a
    mapping built by ``**kwargs`` off a dict would turn a renamed field into a
    silently defaulted one. Here a renamed field is a ``TypeError``.
    """
    mapped: dict[str, Any] = {
        risk_field: _coerce(demo_field, getattr(limits, demo_field))
        for demo_field, risk_field in DIRECT_LIMITS.items()
    }
    mapped.update(
        {
            risk_field: float(getattr(limits, demo_field))
            for risk_field, demo_field in DERIVED_LIMITS.items()
        }
    )
    return RiskLimits(risk_per_trade_pct=DEMO_RISK_PER_TRADE_PCT, **mapped)


def _coerce(demo_field: str, value: float) -> int | float:
    """A count as an ``int``, a ratio as a ``float``. The config's own split."""
    return int(value) if demo_field in _COUNTS else float(value)


#: What a restart's reconciliation halts on when the two files state different
#: equities. A prefix, because the reason names both numbers after it.
EQUITY_DISPUTE_PREFIX = "equity_dispute:"

#: What it halts on when the ledger could not be read at all, so the persisted
#: claim cannot be checked against anything.
EQUITY_RECONCILIATION_PREFIX = "equity_reconciliation:"


def is_equity_reconciliation_halt(reason: str) -> bool:
    """Whether ``reason`` is one of the two halts THIS module raises on a restart.

    The clearing path (``DemoRunner.resolve_equity``) asks this rather than
    matching the strings itself, so the reason a restart raises and the reason an
    operator may settle cannot drift apart: a rewording here would otherwise
    leave the campaign holding a halt no command would admit to recognising.
    """
    return reason.startswith(EQUITY_DISPUTE_PREFIX) or reason.startswith(
        EQUITY_RECONCILIATION_PREFIX
    )


def ledger_equity(state_dir: Path | str, *, capital: Decimal | float) -> Decimal | None:
    """What the campaign's accounting says the account is worth, or ``None``.

    The carry ledger is the demo's accounting authority and this reads it through
    its own loader, exactly as :func:`chimera.demo.inspection.inspect_demo_state`
    and :func:`chimera.carry.factory.build_hedged_position` do. No second
    accounting authority is created here and none may be: Aegis is the central
    risk authority, the ledger is where the cash and the marks live, and this
    function only asks the second what it already recorded.

    ``None`` means the ledger could not speak -- it was UNREADABLE -- and is a
    different answer from "the ledger has never been marked", which is a loaded
    or absent ledger whose ``last_equity`` is ``None`` and whose equity is
    therefore the configured capital. :func:`chimera.demo.runner.DemoRunner._portfolio`
    already reads it that way (``ledger.last_equity if ... is not None else
    self.capital``) and this repeats that contract rather than inventing a
    second one.

    That split is only worth as much as the loader's ability to tell the two
    apart, and this is the caller that made it safety-relevant. ``CarryLedger.open``
    used to decide "absent" with ``Path.exists()``, which answers ``False`` for a
    path it merely could not EXAMINE -- so a ledger on a degraded mount read as
    MISSING, which reads here as "worth the configured capital", which is an
    accounting claim nothing had made. It now lets the read decide, as
    :meth:`chimera.risk.RiskEngine._load_state` does, and an unexaminable file
    reaches this function as ``None`` rather than as a number.
    """
    root = Path(state_dir)
    ledger = CarryLedger.open(root / "carry_ledger.json", capital=Decimal(capital))
    if ledger.outcome is LoadOutcome.UNREADABLE:
        return None
    last = ledger.state.last_equity
    return Decimal(capital) if last is None else last


def seed_or_reconcile_equity(
    engine: RiskEngine, *, capital: Decimal | float, state_dir: Path | str
) -> None:
    """Seed equity on a first start; on a restart, reconcile instead of re-seeding.

    ``build_risk_engine`` used to end with an unconditional
    ``engine.update_equity(float(capital))``, which ran *after* the constructor
    had already restored the persisted state. That one line is AEG-1. It did not
    merely record a number: ``update_equity`` is a guard, and feeding it the
    configured capital on every start made a restart lie to both account guards
    and to the evidence at once.

    * **The drawdown guard.** ``peak_equity`` survives a restart on purpose --
      that is what `tests/test_risk_state_persistence.py` calls "the peak equity
      survives a restart between the peak and the breach" -- while ``equity`` was
      being put back to ``capital``. A campaign that had genuinely earned a peak
      of ``capital / (1 - max_drawdown_pct)`` or better therefore halted on its
      own startup, with a drawdown it had never taken, measured from a peak it
      really had against an equity it did not have.
    * **The daily-loss guard.** ``update_equity`` rolls the day when the UTC date
      has changed and sets ``day_start_equity`` to the equity it was handed. A
      restart across a UTC midnight thus opened the new day's loss budget at
      ``capital`` rather than at what the account was actually worth. An account
      legitimately down a few percent over several days then had its whole
      cumulative fall re-measured as one day's loss, and halted on the first
      ordinary minute of the new day.
    * **The reported state.** ``equity``, ``peak_equity``, ``day_start_equity``
      and ``daily_pnl`` are all hashed into ``risk.state_hash``, so a restart
      moved the hash for a reason that was not a decision. Two runs over the same
      recorded minutes stopped being comparable across a restart.

    So capital seeds equity **only on a genuine first start**, and "genuine" is
    read off :attr:`chimera.risk.RiskEngine.load_outcome` -- the read itself --
    rather than inferred from an equity that happens to equal ``capital``, which
    is also what a restarted campaign looks like after giving back exactly its
    gains.

    The four outcomes, and why each gets what it gets:

    ``MISSING``
        No file: nothing has ever been persisted, so there is nothing to
        preserve and ``capital`` is the only claim available. Seed.

        Whether a missing ``risk.json`` is suspicious -- because the decision log
        already holds records for this campaign -- is **canonical R1-c's**
        question, and it is now answered one level up, in
        :func:`build_risk_engine`, BEFORE this function is reached. That is
        where it has to be: seeding is a write, and a file created from the
        configured capital cannot be un-created afterwards. So by the time
        ``MISSING`` arrives here it is a first start that R1-c has checked
        against the log and found to be one; this function still reads no log
        and writes no record.
    ``LEGACY``
        The pre-schema document carried ``halted`` and nothing else. It makes no
        claim about equity, so there is again nothing to preserve, and seeding is
        exactly what this build did before. Its halt, if it had one, is already
        restored and is untouched by the seed. R1-c refuses this combination
        against a log that holds records -- a campaign with a history had an
        account, and a document that claims none cannot be its state -- so what
        reaches here is a legacy file on a campaign with no committed record.
    ``UNREADABLE``
        The engine has failed closed onto a deliberately empty state, is halted,
        and will not write that state over the file it could not read. Seeding
        would put a fabricated equity into the one state whose whole meaning is
        *the absence of a claim*. Do neither: no seed, and no reconciliation
        against an account state that does not exist.
    ``LOADED``
        A restart carrying real persisted history. Do not touch the equity;
        reconcile it instead.

    **Reconciliation.** The comparison is exact and invents no tolerance,
    because it is made in the domain the value actually travelled through. The
    runner's only equity writer is
    ``self.risk.update_equity(float(mark.equity))``, and ``mark.equity`` is the
    same ``Decimal`` that ``CarryLedger.mark`` stored as ``last_equity`` one
    statement earlier. The ledger round-trips its decimals through ``str``, so
    ``float(ledger_equity)`` is bit-for-bit the number the engine was handed. A
    tolerance here would not absorb rounding -- there is none to absorb -- it
    would only decide how large a real disagreement may be before anybody is
    told about it.

    A disagreement is a **dispute**: the engine halts, naming both numbers, and
    neither side is overwritten to make them agree. The persisted risk state
    keeps every field it loaded (``halt`` writes the halt beside them, it does
    not replace them) and the ledger file is not written at all. An operator
    resolves it; this function never does.

    An UNREADABLE ledger is the same dispute for a different reason: the
    accounting truth is unavailable, so the reconciliation cannot be performed.
    Skipping it in that case would make the guard removable by exactly the
    failure it guards against. `DemoRunner.self_check` already refuses to start
    on that ledger, so this adds no new refusal to the runner -- it makes
    ``build_risk_engine`` itself honest for the callers that build an engine
    without a runner.

    Idempotent across repeated restarts: ``halt`` returns early when the engine
    is already halted, so a second restart against the same disagreement writes
    nothing and leaves ``risk.state_hash`` where the first one left it.

    WHERE A DISAGREEMENT CAN COME FROM. The runner has exactly one writer into
    Aegis's equity -- ``DemoRunner.tick``'s ``update_equity(float(mark.equity))``
    -- and five places that mark the ledger and persist it. A crash is therefore
    not the only way the two can part, and an earlier revision of this docstring
    said it was. Enumerated against the current runner:

    * ``tick``'s own mark is followed by that one writer, so an uninterrupted
      minute leaves the two equal. A process killed BETWEEN the ledger write and
      ``update_equity`` leaves the risk state one mark behind -- a real
      disagreement, reported here. Closing that window means changing the
      runner's persistence ordering and its crash semantics, which is canonical
      **R1-i**, not this item; R1-b does not reorder the runner's writes.
    * ``_funding``'s mark either falls through to that same writer later in the
      tick, or the tick halts.
    * both marks on the liquidation-touch path are followed by a halt, and
      ``halt`` keeps the FIRST reason, so the restart reports the touch rather
      than the equity. The disagreement is still on disk underneath it and
      surfaces if the touch is ever resumed.
    * ``DemoRunner.flatten`` marked, persisted, and told Aegis nothing -- so an
      ordinary operator flatten, with no crash anywhere, left the two files
      stating different equities and the campaign halting on its next start. That
      was a real defect and it is fixed in the runner rather than tolerated here:
      ``flatten`` now hands the flattened equity to the same single writer, under
      the same ledger-may-speak condition as the record it writes.

    So a disagreement reaching this function now means a crash in that one
    window, a state directory restored in pieces, a file copied between hosts, or
    a truncated ledger -- and not the routine operation of the campaign.
    """
    outcome = engine.load_outcome
    if outcome is RiskStateLoad.UNREADABLE:
        return
    if outcome in (RiskStateLoad.MISSING, RiskStateLoad.LEGACY):
        engine.update_equity(float(capital))
        return

    accounted = ledger_equity(state_dir, capital=capital)
    if accounted is None:
        engine.halt(
            f"{EQUITY_RECONCILIATION_PREFIX} the persisted risk state was restored and "
            "the carry ledger could not be read, so the equity it claims cannot be "
            "checked against the campaign's accounting"
        )
        return
    if float(accounted) != engine.state.equity:
        engine.halt(
            f"{EQUITY_DISPUTE_PREFIX} the persisted risk state says equity is "
            f"{engine.state.equity!r} and the carry ledger accounts for {accounted}. "
            "Neither is overwritten; an operator decides which is right, and "
            '`demo_run resolve --equity --note "..."` is how they say so '
            "(docs/demo_runbook.md, section 4, for the full invocation: the "
            "subcommand named here still needs --config and --root)"
        )


def build_risk_engine(
    config: DemoConfig,
    *,
    capital: Decimal | float,
    state_dir: Path | str | None = None,
    clock: Callable[[], float] | None = None,
) -> RiskEngine:
    """The demo's Aegis: campaign limits, the campaign's state files, its equity.

    ``state_dir`` defaults to the configuration's own ``state_dir`` runner
    setting, which is where the runner keeps everything else. Both entry points
    call this, so the engine a test drives is built exactly as the engine an
    operator runs.

    ``capital`` is what a *first* start is worth and nothing else;
    :func:`seed_or_reconcile_equity` holds the line between that and a restart.

    **Canonical R1-c (AEG-4) is checked first, and the order is the point.**
    :func:`chimera.demo.risk_continuity.assess_risk_continuity` asks whether the
    state this engine just read can be the continuation of the campaign's
    decision log. It is asked HERE rather than in the runner because the very
    next statement is a write: on a ``MISSING`` load
    :func:`seed_or_reconcile_equity` seeds the configured capital and persists
    it, so a campaign whose log holds a month of records would have a brand-new
    ``risk.json`` on disk before anything had a chance to notice the month was
    missing -- and the restart after that one would find a ``LOADED`` state this
    process invented. A check made afterwards cannot un-make that file.

    The engine is built with ``check_kill_switch_at_construction=False`` for the
    same reason. ``RiskEngine.__init__`` otherwise ends by consulting the kill
    switch itself, and that consultation can halt-and-persist -- so a kill
    switch left engaged beside a MISSING or MISMATCHED ``risk.json`` used to
    create or overwrite the very file this function has not yet had a chance to
    dispute, before ``assess_risk_continuity`` below ever ran. The engine is
    still checked against the switch, explicitly, but only once it is known
    that nothing here is about to seal it: see the ``check_kill_switch`` call
    below.

    The three faults no crash can produce are the three this stops for, and they
    are the same three where the next statement would fabricate something: an
    absent file and a pre-schema one are what
    :func:`seed_or_reconcile_equity` seeds the configured capital for, and an
    unreadable one is already sealed by R1-b. The fourth --
    ``RISK_STATE_MISMATCH``, a readable current-schema state whose hash is not
    the one the log last restated -- is deliberately NOT stopped here: a crash
    inside a tick produces it too
    (:attr:`chimera.demo.risk_continuity.RiskContinuity.crash_could_explain`),
    only the runner can tell the two apart (its crash triage, and the
    crash-window proofs that need the limits and ledger it reads), and nothing is
    fabricated by letting it through -- the state is ``LOADED``, so what happens
    next is R1-b's reconciliation against the campaign's accounting rather than
    a seed. :meth:`chimera.demo.runner.DemoRunner._risk_continuity_stands`
    decides that one, and seals the engine there before startup can write.

    When the verdict disputes, the engine is halted and SEALED
    (:meth:`chimera.risk.RiskEngine.halt_for_continuity_dispute`): nothing is
    seeded, nothing is reconciled, and nothing is written over -- or into the
    place of -- the file an operator has to look at. The campaign's own record
    of the finding is the ``RECOVERY`` record :class:`chimera.demo.runner.DemoRunner`
    writes from the same verdict; this function appends nothing, because a
    function that builds a risk engine has no business opening a log for write.

    The reconciliation is skipped on a dispute rather than being run first, and
    deliberately: R1-b's question is *does the persisted equity match the
    campaign's accounting*, and it presumes the persisted state is this
    campaign's. When that presumption is what is in dispute, an answer to it
    would be an answer about the wrong file.
    """
    root = Path(state_dir if state_dir is not None else config.runner_setting("state_dir"))

    kwargs: dict[str, Any] = {}
    if clock is not None:
        kwargs["clock"] = clock

    engine = RiskEngine(
        risk_limits(config.limits),
        state_path=root / "risk.json",
        kill_switch_path=root / "KILL_SWITCH",
        check_kill_switch_at_construction=False,
        **kwargs,
    )
    continuity = assess_risk_continuity(
        load=engine.load_outcome, snapshot=engine.loaded_snapshot, state_dir=root
    )
    if continuity.disputed and not continuity.crash_could_explain:
        engine.halt_for_continuity_dispute(continuity.halt_reason)
        return engine
    if not continuity.disputed:
        # Deliberately NOT for a crash-could-explain ``RISK_STATE_MISMATCH``.
        # That fault is deferred to `chimera.demo.runner.DemoRunner`'s own
        # triage, which has not run yet -- it needs the log read before
        # anything here writes -- so a kill switch present beside a disputed,
        # not-yet-triaged file must not halt-and-persist over the disputed
        # bytes before the runner decides whether the finding stands. The
        # runner checks the switch itself, later, once that decision is made
        # (`DemoRunner.start`); an undisputed engine has no such decision
        # pending and is checked here as it always was.
        engine.check_kill_switch()
    seed_or_reconcile_equity(engine, capital=capital, state_dir=root)
    return engine


def check_exhaustive(
    direct: Mapping[str, str] | None = None,
    derived: Mapping[str, str] | None = None,
    unconfigured: Mapping[str, str] | None = None,
    *,
    demo_fields: set[str] | None = None,
    risk_fields: set[str] | None = None,
) -> None:
    """Refuse to import while any limit is unaccounted for.

    A field added to ``DemoLimits`` that nothing maps would be hashed into a
    campaign's identity and enforced by nothing; a field added to ``RiskLimits``
    that nothing names would be enforced on a default nobody chose. Both are the
    defect this module exists to close, so both fail here rather than in whichever
    campaign noticed first.

    Every input is injectable, defaulting to this module's own tables, so the
    tests can drive each refusal with a table that really is broken. A guard
    asserted only against the correct tables is a guard that cannot fail.
    """
    direct = DIRECT_LIMITS if direct is None else direct
    derived = DERIVED_LIMITS if derived is None else derived
    unconfigured = UNCONFIGURED_LIMITS if unconfigured is None else unconfigured
    demo_fields = {f.name for f in fields(DemoLimits)} if demo_fields is None else demo_fields
    risk_fields = {f.name for f in fields(RiskLimits)} if risk_fields is None else risk_fields

    if missing := demo_fields - set(direct):
        raise RiskWiringError(
            f"DemoLimits fields reach no RiskLimits field: {sorted(missing)}. A campaign "
            "limit that nothing maps is hashed into the campaign's identity and enforced "
            "by nothing"
        )
    if unknown := set(direct) - demo_fields:
        raise RiskWiringError(
            f"DIRECT_LIMITS names no such DemoLimits field: {sorted(unknown)}"
        )

    claimed = set(direct.values()) | set(derived) | set(unconfigured)
    if unknown := claimed - risk_fields:
        raise RiskWiringError(f"no such RiskLimits field: {sorted(unknown)}")
    if missing := risk_fields - claimed:
        raise RiskWiringError(
            f"RiskLimits fields this module does not account for: {sorted(missing)}. Add "
            "each to DIRECT_LIMITS, DERIVED_LIMITS or UNCONFIGURED_LIMITS with its reason; "
            "a field left out is one the demo enforces on a default nobody chose"
        )
    overlap = (
        (set(direct.values()) & set(derived))
        | (set(direct.values()) & set(unconfigured))
        | (set(derived) & set(unconfigured))
    )
    if overlap:
        raise RiskWiringError(f"RiskLimits fields claimed twice: {sorted(overlap)}")

    if unknown := set(derived.values()) - demo_fields:
        raise RiskWiringError(
            f"DERIVED_LIMITS derives from no such DemoLimits field: {sorted(unknown)}"
        )


check_exhaustive()
