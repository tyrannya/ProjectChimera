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

This module opens no socket, reads no clock and makes no decision.
:func:`risk_limits` is a pure function of its argument.
"""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping

from chimera.demo.config import COUNT_LIMITS, DemoConfig, DemoLimits
from chimera.risk import RiskEngine, RiskLimits


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
    """
    root = Path(state_dir if state_dir is not None else config.runner_setting("state_dir"))

    kwargs: dict[str, Any] = {}
    if clock is not None:
        kwargs["clock"] = clock

    engine = RiskEngine(
        risk_limits(config.limits),
        state_path=root / "risk.json",
        kill_switch_path=root / "KILL_SWITCH",
        **kwargs
    )
    engine.update_equity(float(capital))
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
