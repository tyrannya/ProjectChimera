"""Information sets, and the proof that two of them were compared fairly.

Research checkpoint P2b asks whether causal market structure adds information
beyond OHLCV14; P2c asks the same of causal classical chart structure. Either
question only has an answer if the only thing that differs between the arms is
the *information*. Everything else — which rows are
scored, what the label is, where the folds fall, what a trade costs — has to be
identical, and "identical" has to be checked rather than intended.

The failure this module exists to prevent is specific and quiet. A new feature
family warms up over its own first rows. Drop those rows and every later row
shifts by the number dropped, so fold boundaries resolved by row index land on
different candles, and the two arms are then measured over different market
periods. Nothing raises. The report still prints four folds. The comparison is
simply no longer about the features.

The defence has two halves:

*one spine, many views*
    :func:`build_information_set_views` builds one :class:`nn.train.ResearchData`
    per information set that share — by object identity, not by equality — the
    dates, targets, future returns, closes and segment ids. Only ``features``
    differs. Since :func:`nn.dataset.sample_indices` is a function of the split,
    the sequence length, the horizon and the segment ids alone, the three views
    cannot select different rows.

*and say so out loud*
    :meth:`AlignedResearchSamples.prove_alignment` re-derives the sample index,
    timestamps, labels and costs per fold per view and asserts they match, then
    returns the evidence for the artifact. A claim of alignment that nothing
    recomputes is a comment, not a guarantee.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chimera.features import feature_columns
from nn.data_pipeline import DatasetMetadata, _contiguous_segment_ids
from nn.dataset import sample_indices
from nn.derivatives import (
    DERIVATIVES_FEATURE_FAMILIES,
    DERIVATIVES_SPEC_VERSION,
    DerivativesSpec,
    derivatives_feature_columns,
)
from nn.p4_preregistration import (
    ARMS as P4_ARMS,
    COMBINED as P4_COMBINED,
    DERIVATIVES_V1 as P4_DERIVATIVES_V1,
    RESEARCH_CLASSIFICATION as P4_RESEARCH_CLASSIFICATION,
)
from nn.chart_structure import (
    CHART_FEATURE_FAMILIES,
    CHART_SPEC_VERSION,
    ChartSpec,
    chart_feature_columns,
    compute_chart_features_segmented,
)
from nn.mtf import (
    MTF_FEATURE_FAMILIES,
    MtfSpec,
    build_mtf_context,
    mtf_feature_columns,
)
from nn.p5_preregistration import (
    ARMS as P5_ARMS,
    COMBINED as P5_COMBINED,
    FAMILY as P5_MTF_V1,
    RESEARCH_CLASSIFICATION as P5_RESEARCH_CLASSIFICATION,
)
from nn.microstructure import (
    MICROSTRUCTURE_FEATURE_FAMILIES,
    MICROSTRUCTURE_SPEC_VERSION,
    MicrostructureSpec,
    compute_microstructure_features,
    microstructure_feature_columns,
)
from nn.smc import (
    SMC_FEATURE_FAMILIES,
    SMC_SPEC_VERSION,
    SMCSpec,
    compute_smc_features_segmented,
    smc_feature_columns,
)
from nn.train import ResearchData
from nn.walkforward import FoldPlan

#: The OHLCV14 information set: what P2a's control was measured on, unchanged.
OHLCV14 = "ohlcv14"
#: Causal market structure alone, with no OHLCV14 column present.
SMC_V1 = "smc_v1"
#: Both, concatenated in that order.
COMBINED = "ohlcv14_plus_smc_v1"
#: Causal classical chart structure alone (docs/chart_structure_v1.md).
CHART_V1 = "chart_structure_v1"
#: OHLCV14 plus chart structure — P2c's combined arm.
OHLCV14_PLUS_CHART = "ohlcv14_plus_chart_structure_v1"
#: Causal trade-flow microstructure alone (docs/microstructure_v1.md). The first
#: information set in this repository that is not a function of the hourly candle.
MICROSTRUCTURE_V1 = "microstructure_v1"

#: P4's family, and the arm that adds it to the control. Both names come from
#: the preregistration rather than being spelled again here: a run whose arm was
#: called something else would be answering a question nobody registered.
DERIVATIVES_V1 = P4_DERIVATIVES_V1
OHLCV14_PLUS_DERIVATIVES = P4_COMBINED
#: OHLCV14 plus trade-flow microstructure — P3's combined arm.
OHLCV14_PLUS_MICROSTRUCTURE = "ohlcv14_plus_microstructure_v1"

#: P5's family, and the arm that adds it to the control. Both names come from the
#: preregistration rather than being spelled again here, for the reason P4's do: a
#: run whose arm was called something else would be answering a question nobody
#: registered.
MTF_V1 = P5_MTF_V1
OHLCV14_PLUS_MTF = P5_COMBINED


class AlignmentError(AssertionError):
    """Raised when two information sets would not be compared on the same rows."""


@dataclass(frozen=True)
class InformationSet:
    """A named feature column list, plus the families an ablation can remove.

    ``families`` is not decoration: the leave-one-family-out robustness study
    drops one at a time, and a family that is not declared here cannot be ablated —
    which is the point, because choosing the groups after seeing which columns
    mattered would make the ablation a search.
    """

    name: str
    columns: tuple[str, ...]
    families: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError(f"{self.name} lists a column twice")
        covered = [column for group in self.families.values() for column in group]
        if sorted(covered) != sorted(self.columns):
            raise ValueError(
                f"{self.name}'s families do not partition its columns: "
                f"{len(covered)} covered vs {len(self.columns)} declared"
            )

    def without(self, family: str) -> "InformationSet":
        """This set minus one family, for the leave-one-family-out ablation."""
        if family not in self.families:
            raise KeyError(f"{self.name} has no family {family!r}")
        removed = set(self.families[family])
        return InformationSet(
            name=f"{self.name}_minus_{family}",
            columns=tuple(c for c in self.columns if c not in removed),
            families={k: v for k, v in self.families.items() if k != family},
        )


def ohlcv14_set() -> InformationSet:
    """The frozen 14-column set, as one family — it is the control, not a subject."""
    columns = tuple(feature_columns())
    return InformationSet(name=OHLCV14, columns=columns, families={"ohlcv14": columns})


def smc_v1_set() -> InformationSet:
    return InformationSet(
        name=SMC_V1,
        columns=tuple(smc_feature_columns()),
        families={k: tuple(v) for k, v in SMC_FEATURE_FAMILIES.items()},
    )


def chart_v1_set() -> InformationSet:
    return InformationSet(
        name=CHART_V1,
        columns=tuple(chart_feature_columns()),
        families={k: tuple(v) for k, v in CHART_FEATURE_FAMILIES.items()},
    )


def ohlcv14_plus_chart_set() -> InformationSet:
    ohlcv = ohlcv14_set()
    chart = chart_v1_set()
    return InformationSet(
        name=OHLCV14_PLUS_CHART,
        columns=ohlcv.columns + chart.columns,
        families={**ohlcv.families, **chart.families},
    )


def microstructure_v1_set() -> InformationSet:
    return InformationSet(
        name=MICROSTRUCTURE_V1,
        columns=tuple(microstructure_feature_columns()),
        families={k: tuple(v) for k, v in MICROSTRUCTURE_FEATURE_FAMILIES.items()},
    )


def ohlcv14_plus_microstructure_set() -> InformationSet:
    ohlcv = ohlcv14_set()
    micro = microstructure_v1_set()
    return InformationSet(
        name=OHLCV14_PLUS_MICROSTRUCTURE,
        columns=ohlcv.columns + micro.columns,
        families={**ohlcv.families, **micro.families},
    )


def derivatives_v1_set() -> InformationSet:
    return InformationSet(
        name=DERIVATIVES_V1,
        columns=tuple(derivatives_feature_columns()),
        families={k: tuple(v) for k, v in DERIVATIVES_FEATURE_FAMILIES.items()},
    )


def ohlcv14_plus_derivatives_set() -> InformationSet:
    ohlcv = ohlcv14_set()
    derivatives = derivatives_v1_set()
    return InformationSet(
        name=OHLCV14_PLUS_DERIVATIVES,
        columns=ohlcv.columns + derivatives.columns,
        families={**ohlcv.families, **derivatives.families},
    )


def mtf_v1_set() -> InformationSet:
    return InformationSet(
        name=MTF_V1,
        columns=tuple(mtf_feature_columns()),
        families={k: tuple(v) for k, v in MTF_FEATURE_FAMILIES.items()},
    )


def ohlcv14_plus_mtf_set() -> InformationSet:
    ohlcv = ohlcv14_set()
    mtf = mtf_v1_set()
    return InformationSet(
        name=OHLCV14_PLUS_MTF,
        columns=ohlcv.columns + mtf.columns,
        families={**ohlcv.families, **mtf.families},
    )


def combined_set() -> InformationSet:
    ohlcv = ohlcv14_set()
    smc = smc_v1_set()
    return InformationSet(
        name=COMBINED,
        columns=ohlcv.columns + smc.columns,
        families={**ohlcv.families, **smc.families},
    )


#: The three arms of P2b, in report order. Predeclared.
P2B_INFORMATION_SETS = (OHLCV14, SMC_V1, COMBINED)

#: The three arms of P2c, in report order. The control is `ohlcv14` rather than
#: P2b's combined arm: P2b found that market structure did not improve on
#: OHLCV14, so OHLCV14 remains the best information set this repository has — by
#: default rather than by merit, which is exactly why the next family is worth
#: measuring against it.
P2C_INFORMATION_SETS = (OHLCV14, CHART_V1, OHLCV14_PLUS_CHART)

#: The three arms of P3, in report order. The control is `ohlcv14` again, and for
#: the same reason P2c used it: three families have now failed to improve on it,
#: so it remains the best information set this repository has — by default rather
#: than by merit, which is exactly why the next family is measured against it.
P3_INFORMATION_SETS = (OHLCV14, MICROSTRUCTURE_V1, OHLCV14_PLUS_MICROSTRUCTURE)

#: The three arms of P4, in report order, taken from the preregistration. The
#: control is `ohlcv14` for the fourth time, and for the reason P2c and P3 gave:
#: four families have now failed to improve on it, so it remains the best
#: information set this repository has — by default rather than by merit.
#:
#: Unlike every earlier checkpoint, this control is **re-run on P4's own sample
#: universe** rather than reproduced on the full spine: `derivatives_v1` is not
#: defined on every row, and comparing a derivatives arm scored where the data
#: exists against a control scored everywhere would measure two market periods
#: and report the difference as an information set.
P4_INFORMATION_SETS = tuple(P4_ARMS)

#: The three arms of P5, in report order, taken from the preregistration. The
#: control is `ohlcv14` for the fifth time, and for the reason P2c, P3 and P4
#: gave: four families have now failed to improve on it.
#:
#: Like P4 this control is **re-run on P5's own sample universe** rather than
#: reproduced on the full spine. `mtf_v1` is undefined until each higher clock has
#: warmed up, and comparing an arm scored where its data exists against a control
#: scored everywhere would measure two market periods and report the difference as
#: an information set (`docs/p5_preregistration.md` §6.1).
P5_INFORMATION_SETS = tuple(P5_ARMS)


#: Separator between the combined set and the family an ablation removed.
ABLATION_PREFIX = f"{COMBINED}_minus_"

#: The families the ablation may remove, in report order. Fixed here rather
#: than derived from whatever columns turned out to matter: an ablation whose
#: groups were chosen after seeing the result is a search wearing a diagnostic's
#: name.
ABLATABLE_FAMILIES = tuple(SMC_FEATURE_FAMILIES)


def information_set(name: str) -> InformationSet:
    """Any checkpoint's three arms, or one ablation of P2b's combined arm."""
    builders = {
        OHLCV14: ohlcv14_set,
        SMC_V1: smc_v1_set,
        COMBINED: combined_set,
        CHART_V1: chart_v1_set,
        OHLCV14_PLUS_CHART: ohlcv14_plus_chart_set,
        MICROSTRUCTURE_V1: microstructure_v1_set,
        OHLCV14_PLUS_MICROSTRUCTURE: ohlcv14_plus_microstructure_set,
        DERIVATIVES_V1: derivatives_v1_set,
        OHLCV14_PLUS_DERIVATIVES: ohlcv14_plus_derivatives_set,
        MTF_V1: mtf_v1_set,
        OHLCV14_PLUS_MTF: ohlcv14_plus_mtf_set,
    }
    if name in builders:
        return builders[name]()
    if name.startswith(ABLATION_PREFIX):
        family = name[len(ABLATION_PREFIX) :]
        if family in ABLATABLE_FAMILIES:
            return combined_set().without(family)
    known = sorted(builders) + [f"{ABLATION_PREFIX}{f}" for f in ABLATABLE_FAMILIES]
    raise KeyError(f"unknown information set {name!r}; known: {known}")


def ablation_set_names() -> list[str]:
    """The six leave-one-family-out arms, in family order."""
    return [f"{ABLATION_PREFIX}{family}" for family in ABLATABLE_FAMILIES]


@dataclass(frozen=True)
class Checkpoint:
    """Which research question a run is answering, as data rather than as prose.

    Named after the checkpoint in ``docs/research_roadmap.md``. Everything a
    reader needs to tell two checkpoints apart hangs off this: the family under
    test, the arms that are allowed to appear, the question the artifact states,
    and how adaptive the evidence is.

    It exists because the runner was a module constant. ``nn.p2b`` hard-coded
    ``CHECKPOINT = "P2b"`` and a purpose string naming market structure, so a P2c
    run — different family, different question, different adaptive status —
    wrote nine artifacts and a comparison that all called themselves P2b and
    asked about ``smc_v1``. Nothing failed: the numbers were P2c's throughout and
    only the identity was wrong, which is the kind of error a green run cannot
    surface. Making the identity an input means a cell must say which question it
    is answering before it may answer one.
    """

    name: str
    #: The information family under test. The control is not it.
    family: str
    control: str
    #: The three arms, in report order: control, family alone, both.
    arms: tuple[str, ...]
    #: Extra arms this checkpoint accepts that are not canonical evidence.
    diagnostic_arms: tuple[str, ...] = ()
    #: Whether this checkpoint's evidence is defined against the trade source.
    #:
    #: True for every arm of such a checkpoint, including its OHLCV14 control.
    #: The control is not a P2 cell that happens to be re-run here: it is P3
    #: evidence, it must carry the same trade-source identity as the arms it is
    #: compared against, and `nn.p2b_compare` refuses a batch whose cells
    #: disagree about which source they were measured under.
    trade_source: bool = False
    #: Whether this checkpoint's evidence is defined against the derivatives
    #: source, and therefore against a **restricted sample universe**.
    #:
    #: True for every arm of such a checkpoint, its OHLCV14 control included.
    #: That is not a formality: `derivatives_v1` is undefined wherever the
    #: archive has no observation, so the control has to be re-run on the
    #: intersection rather than reproduced from an earlier checkpoint's numbers
    #: — otherwise the delta measures two market periods
    #: (`docs/p4_preregistration.md` §6.2).
    derivatives_source: bool = False
    #: Whether this checkpoint's arms are functions of higher-timeframe bars, and
    #: therefore against a **restricted sample universe**.
    #:
    #: True for every arm of such a checkpoint, its OHLCV14 control included, for
    #: the reason `derivatives_source` gives: `mtf_v1` is undefined until each
    #: higher clock has warmed up, so the control has to be re-run on the same
    #: rows rather than reproduced from an earlier checkpoint's numbers
    #: (`docs/p5_preregistration.md` §6.1).
    #:
    #: Unlike the other two source flags this needs no second snapshot: the bars
    #: are cut from the candles the control already reads.
    mtf_source: bool = False
    #: What had already been read when this checkpoint was designed.
    adaptive_status: str = ""

    def __post_init__(self) -> None:
        if self.control not in self.arms:
            raise ValueError(f"{self.name}'s control {self.control!r} is not one of its arms")
        if self.family in (self.control, ""):
            raise ValueError(f"{self.name} has no family under test distinct from its control")

    @property
    def question(self) -> str:
        """The question every artifact of this checkpoint states, generated once.

        Generated rather than written per artifact, because the two P2b strings
        that existed before this — one in the cell runner, one in the comparison
        — disagreed with each other in wording and with P2c's arms entirely.
        """
        control = self.control.upper()
        return (
            f"does causal {self.family}, alone or combined with {control}, add usable "
            f"information beyond {control}?"
        )

    def accepts(self, information_set: str) -> bool:
        return information_set in self.arms or information_set in self.diagnostic_arms


#: P2b. Declared before any P2b cell ran; adaptive with respect to P2a.
P2B = Checkpoint(
    name="P2b",
    family=SMC_V1,
    control=OHLCV14,
    arms=P2B_INFORMATION_SETS,
    diagnostic_arms=tuple(ablation_set_names()),
    adaptive_status=(
        "P2b was designed after P2a's outer results had been seen, so its outer blocks "
        "are adaptive research evidence rather than a pristine out-of-sample test"
    ),
)

#: P2c. Designed after P2b's outer results had been seen, which makes it more
#: adaptive than P2b and not a confirmation of anything.
P2C = Checkpoint(
    name="P2c",
    family=CHART_V1,
    control=OHLCV14,
    arms=P2C_INFORMATION_SETS,
    adaptive_status=(
        "P2c was designed after P2b's outer results had been seen, and by the time it "
        "ran these four outer blocks had already been read by v4, P2a, P2b, the P2b "
        "ablation and the P2b regime description. Its constants were fixed before its "
        "own outer results were read, but the family was chosen because the previous "
        "one failed, so this is exploratory adaptive evidence: it generates hypotheses "
        "and cannot confirm one"
    ),
)

#: P3. The first checkpoint whose input is not the hourly candle.
#:
#: Designed after P2b's and P2c's outer results had been seen, so it is at least
#: as adaptive as P2c. What changes is the *source*: `microstructure_v1` is
#: computed from Binance's public aggTrades archive rather than from OHLCV, which
#: is why this checkpoint was worth spending on after two negative feature
#: families rather than fitting a third transformation of the same five numbers.
P3 = Checkpoint(
    name="P3",
    family=MICROSTRUCTURE_V1,
    control=OHLCV14,
    arms=P3_INFORMATION_SETS,
    trade_source=True,
    adaptive_status=(
        "P3 was designed after P2b's and P2c's outer results had been seen, and by the "
        "time it ran these four outer blocks had already been read by v4, P2a, P2b, the "
        "P2b ablation, the P2b regime description and P2c. Its constants were fixed "
        "before its own outer results were read, and its information source is new "
        "rather than another transformation of the same candles — but the blocks are "
        "the same blocks, so this is exploratory adaptive evidence: it generates "
        "hypotheses and cannot confirm one"
    ),
)

#: P4. The first checkpoint whose input is not a function of the spot tape.
#:
#: Registered here because `docs/p4_preregistration.md` §13 requires it before
#: P4 may run — and registration is *not* permission to run. `nn.p4_stage1`
#: holds the execution interlock, and every P4 fit goes through it; a checkpoint
#: that could be run by naming it would make the interlock a convention.
P4 = Checkpoint(
    name="P4",
    family=DERIVATIVES_V1,
    control=OHLCV14,
    arms=P4_INFORMATION_SETS,
    derivatives_source=True,
    adaptive_status=P4_RESEARCH_CLASSIFICATION,
)

#: P5. The first checkpoint whose input is a different *clock* rather than a
#: different transformation, a different source, or a different market.
#:
#: Its family is a function of the same candles the control reads, so it needs no
#: new snapshot and no acquisition interlock — but it *is* defined against a
#: restricted sample universe, because neither higher clock has a usable context
#: until it has warmed up. That mask is computed once and applied to every arm by
#: object identity, exactly as P4's is.
P5 = Checkpoint(
    name="P5",
    family=MTF_V1,
    control=OHLCV14,
    arms=P5_INFORMATION_SETS,
    mtf_source=True,
    adaptive_status=P5_RESEARCH_CLASSIFICATION,
)

#: Every checkpoint this runner can execute, by name.
CHECKPOINTS: dict[str, Checkpoint] = {c.name: c for c in (P2B, P2C, P3, P4, P5)}


def checkpoint(name: str) -> Checkpoint:
    """Look one up, or say which names exist rather than raise a bare KeyError."""
    try:
        return CHECKPOINTS[name]
    except KeyError:
        raise KeyError(f"unknown checkpoint {name!r}; known: {sorted(CHECKPOINTS)}") from None


@dataclass(frozen=True)
class AlignedResearchSamples:
    """One :class:`ResearchData` per information set over one shared row spine.

    ``spine`` is the processed research frame every view was built from. The
    views hold *the same array objects* for everything except ``features``, so
    the alignment this class proves is a property of how it was constructed and
    not something a caller could break by passing the wrong thing.
    """

    views: dict[str, ResearchData]
    sets: dict[str, InformationSet]
    spine_dates: np.ndarray
    spine_targets: np.ndarray
    smc_spec: SMCSpec
    chart_spec: ChartSpec
    #: Rows the common-validity mask kept. All of them, or construction failed.
    n_rows: int
    #: The column each view was assembled from, by name. Kept so the alignment
    #: proof can check the one array that actually differs between views.
    columns: dict[str, np.ndarray]
    #: Independent re-derivation of the join, from :func:`_join_evidence`.
    join_evidence: dict[str, Any]
    #: The microstructure constants this run used, when a trade source was
    #: supplied. ``None`` for a P2b or P2c run, which reads no trade data at all
    #: and must not record a spec it never applied.
    micro_spec: MicrostructureSpec | None = None
    #: The derivatives constants this run used, when a derivatives source was
    #: supplied. ``None`` for every checkpoint before P4, which reads none.
    derivatives_spec: DerivativesSpec | None = None
    #: The higher-timeframe constants this run used. ``None`` for every checkpoint
    #: before P5, which cuts no bar wider than the base clock.
    mtf_spec: MtfSpec | None = None
    #: P4's sample universe: one flag per spine row, shared by every view **by
    #: object identity** so that "the arms were scored on the same rows" is a
    #: property of construction rather than a claim. ``None`` before P4, which is
    #: how a run scored on the whole spine says so.
    eligible: np.ndarray | None = None

    def names(self) -> list[str]:
        return list(self.views)

    def prove_alignment(self, folds: Sequence[FoldPlan], seq_len: int) -> dict[str, Any]:
        """Recompute what each view would score, per fold, and assert it matches.

        Returns the evidence rather than just a boolean: the artifact records the
        sample counts, the first and last timestamp of every block, and a hash
        over the sample indices, so a later reader can check the claim against
        the persisted predictions instead of believing this docstring.
        """
        names = self.names()
        reference = self.views[names[0]]
        horizon = reference.target_spec.horizon

        for name in names[1:]:
            view = self.views[name]
            if view.targets is not reference.targets:
                raise AlignmentError(f"{name} does not share the spine's target array")
            if view.dates is not reference.dates:
                raise AlignmentError(f"{name} does not share the spine's date array")
            if view.future_return is not reference.future_return:
                raise AlignmentError(f"{name} does not share the spine's return array")
            if view.close is not reference.close:
                raise AlignmentError(f"{name} does not share the spine's close array")
            if view.segment_ids is not reference.segment_ids:
                raise AlignmentError(f"{name} does not share the spine's segment ids")
            if view.target_spec != reference.target_spec:
                raise AlignmentError(
                    f"{name} has target spec {view.target_spec} but the control has "
                    f"{reference.target_spec}: horizon or costs differ"
                )
            if view.n_rows != reference.n_rows:
                raise AlignmentError(
                    f"{name} has {view.n_rows} rows but the control has {reference.n_rows}"
                )

        # The checks above are about the arrays every view *shares*, and after
        # the identity assertions they can only succeed — `a[i] == a[i]` is not
        # evidence. `features` is the one array that differs between views and
        # the one a bad join would corrupt, so it is checked against the columns
        # it was assembled from, and those columns were checked against an
        # independent re-derivation when the views were built.
        for name in names:
            view = self.views[name]
            for position, column in enumerate(view.feature_names):
                if column not in self.columns:
                    raise AlignmentError(f"{name} holds unknown column {column!r}")
                if not np.array_equal(view.features[:, position], self.columns[column]):
                    raise AlignmentError(
                        f"{name}'s {column!r} column does not match the column it was "
                        "assembled from; the feature matrix and the join disagree"
                    )

        evidence: list[dict[str, Any]] = []
        for fold, plan in enumerate(folds):
            blocks: dict[str, Any] = {}
            for block, split in (
                ("train", plan.train),
                ("inner_validation", plan.inner),
                ("outer_validation", plan.outer),
            ):
                per_view = {
                    name: sample_indices(
                        split,
                        seq_len,
                        horizon,
                        segment_ids=self.views[name].segment_ids,
                        eligible=self.eligible,
                    )
                    for name in names
                }
                control = per_view[names[0]]
                for name in names[1:]:
                    if not np.array_equal(per_view[name], control):
                        raise AlignmentError(
                            f"fold {fold} {block}: {name} would score "
                            f"{len(per_view[name])} rows against the control's "
                            f"{len(control)}, or the same count at different rows"
                        )
                    if not np.array_equal(
                        self.views[name].targets[control], reference.targets[control]
                    ):
                        raise AlignmentError(f"fold {fold} {block}: {name} labels differ")
                    if not np.array_equal(
                        self.views[name].dates[control], reference.dates[control]
                    ):
                        raise AlignmentError(f"fold {fold} {block}: {name} timestamps differ")
                if len(control) == 0:
                    raise AlignmentError(f"fold {fold} {block} produced no samples")
                blocks[block] = {
                    "row_range": [split.start, split.end],
                    "samples": int(len(control)),
                    "first_row": int(control[0]),
                    "last_row": int(control[-1]),
                    "start": str(reference.dates[control[0]]),
                    "end": str(reference.dates[control[-1]]),
                    "sample_index_sha256": _index_hash(control),
                }
            evidence.append({"fold": fold, **blocks})

        shared = [
            "shared spine arrays by object identity",
            "identical fold row ranges",
            "identical label horizon",
            "identical fee, slippage and cost threshold",
            "identical sample-index arrays per fold and block",
        ]
        if self.eligible is not None:
            shared.append(
                "one sample universe, applied to every arm from the same array object"
            )
        # Said separately because they mean different things. The checks above
        # are strong for two or more views and vacuous for one, and a control
        # cell that runs a single view should not report seven comparisons it
        # never made.
        if len(names) < 2:
            shared = [
                "one information set in this run, so no cross-set comparison was made",
                "identical fold row ranges",
                "identical label horizon",
                "identical fee, slippage and cost threshold",
            ]
        return {
            "information_sets": names,
            "checked": shared,
            "checked_per_view": [
                "every feature column equals the column it was assembled from",
                "the join was re-derived independently — see join_evidence",
            ],
            "join_evidence": self.join_evidence,
            "horizon": horizon,
            "costs": reference.target_spec.to_dict(),
            "seq_len": seq_len,
            "rows": self.n_rows,
            "sample_universe": (
                None
                if self.eligible is None
                else {
                    "rows_eligible": int(np.count_nonzero(self.eligible)),
                    "rows_total": int(len(self.eligible)),
                    "universe_sha256": _universe_hash(self.eligible),
                    "applied_to": list(names),
                }
            ),
            "folds": evidence,
        }


def _index_hash(idx: np.ndarray) -> str:
    """A stable digest of which rows were selected, cheap enough to store."""
    return hashlib.sha256(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _universe_hash(mask: np.ndarray) -> str:
    """A stable digest of a sample universe. The same value `nn.p4_universe` records."""
    packed = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    return hashlib.sha256(
        np.packbits(packed).tobytes() + str(len(packed)).encode()
    ).hexdigest()


def build_information_set_views(
    spine: pd.DataFrame,
    ds_meta: DatasetMetadata,
    raw_candles: pd.DataFrame,
    names: Sequence[str] = P2B_INFORMATION_SETS,
    *,
    smc_spec: SMCSpec | None = None,
    chart_spec: ChartSpec | None = None,
    micro_spec: MicrostructureSpec | None = None,
    trade_aggregates: pd.DataFrame | None = None,
    derivatives: Any = None,
    mtf_spec: MtfSpec | None = None,
    extra_sets: Mapping[str, InformationSet] | None = None,
) -> AlignedResearchSamples:
    """Build one aligned view per named information set.

    ``spine`` is the processed research dataset — the rows that already survived
    OHLCV14 warm-up, NaN removal and horizon trimming, and whose row indices the
    fold plan is expressed in. ``raw_candles`` is the unprocessed OHLCV history
    the spine was derived from; it must cover every spine timestamp and the
    candles before the first one, because market structure entering the spine's
    first row has to have been built from the same history OHLCV14's indicators
    warmed up over.

    ``trade_aggregates`` is P3's second source: the hourly trade table from the
    committed trade snapshot, covering every spine timestamp and the warm-up
    hours before the first one. It is optional because P2b and P2c read no trade
    data at all, and a run that was handed none may not produce a microstructure
    column — asking for a P3 arm without a trade source is a refusal, not an
    empty matrix.

    ``mtf_spec`` turns P5's family on. It is a *spec* rather than a source because
    ``mtf_v1`` has no second source: its 4h and 1d bars are cut from ``raw_candles``,
    the same history OHLCV14 warms up over. Passing it also produces P5's sample
    universe, because neither higher clock has a usable context until it has warmed
    up, and that mask travels into every view by object identity for the reason
    P4's does.

    ``derivatives`` is P4's second source, and it arrives as a
    :class:`nn.p4_universe.Universe` rather than as a table. That is deliberate:
    P4's columns and P4's *sample universe* are computed together from the same
    join, and handing the runner the two separately would let a caller pair one
    checkpoint's columns with another's mask. The universe travels into every
    view by object identity, so §6.2's "computed once and applied to every arm"
    is a property of this function rather than of its callers.

    Fails closed on anything that would move a row: a spine timestamp missing
    from the raw candles or from the trade aggregates, a NaN or infinity in a
    computed feature, a spine segment whose rows are not contiguous in the raw
    candles or in the trade hours.
    """
    smc_spec = smc_spec or SMCSpec()
    chart_spec = chart_spec or ChartSpec()
    timeframe = ds_meta.timeframe or "1h"

    spine = spine.reset_index(drop=True)
    if "date" not in spine.columns:
        raise ValueError("the research spine has no 'date' column")
    if "segment_id" not in spine.columns:
        # Checked here rather than only in `_spine_arrays`, which raises the same
        # thing further down: `_check_segment_contiguity` reads the column on the
        # way past and would otherwise stop the run with a bare KeyError, telling
        # the operator a column name instead of what to do about it.
        raise ValueError(
            "the research spine has no 'segment_id' column, so windows could bridge a "
            "market-data gap. Rebuild it with the current tools.build_features."
        )
    spine_dates = pd.to_datetime(spine["date"], utc=True)

    raw = raw_candles.reset_index(drop=True).copy()
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    if not raw["date"].is_monotonic_increasing or not raw["date"].is_unique:
        raise ValueError("raw candles must be sorted and free of duplicate timestamps")

    # Positional join, not a merge: a merge that silently dropped or duplicated
    # a row would be indistinguishable from a clean one in the result's shape.
    raw_row_of = pd.Series(np.arange(len(raw), dtype=np.int64), index=raw["date"].to_numpy())
    try:
        raw_rows = raw_row_of.reindex(spine_dates.to_numpy())
    except Exception as exc:  # pragma: no cover - index construction is total
        raise ValueError(f"could not locate spine timestamps in the raw candles: {exc}")
    if raw_rows.isna().any():
        missing = int(raw_rows.isna().sum())
        first = spine_dates[raw_rows.isna().to_numpy()].iloc[0]
        raise ValueError(
            f"{missing} research row(s) have no raw candle (first: {first}). The raw "
            "history does not cover the processed dataset it was supposed to produce."
        )
    raw_rows = raw_rows.to_numpy(dtype=np.int64)

    # The engine never sees a candle later than the last row the spine uses.
    # The committed raw file runs three months past the spine's end, and while
    # the engine is causal — verified bit-for-bit, the truncated and full passes
    # agree on every spine row — that is a property of today's implementation,
    # not of this call. One future feature built with a centred window or a
    # backfill would put three months of future market structure into every
    # training row of every fold, and nothing downstream would raise: the
    # alignment proof would pass, every cell would share the leak so the parity
    # check would pass, and recomputation reads the same poisoned predictions.
    # Truncating makes causality structural here instead of inherited.
    horizon_bound = int(raw_rows[-1]) + 1
    raw = raw.iloc[:horizon_bound].reset_index(drop=True)

    smc = compute_smc_features_segmented(raw, timeframe, smc_spec)
    smc.insert(0, "date", raw["date"].to_numpy())

    _check_segment_contiguity(spine, raw_rows, raw["date"], timeframe)

    smc_values = smc[smc_feature_columns()].to_numpy(dtype=np.float64)[raw_rows]
    if not np.isfinite(smc_values).all():
        bad = np.argwhere(~np.isfinite(smc_values))
        column = smc_feature_columns()[int(bad[0][1])]
        raise ValueError(
            f"{len(bad)} non-finite market-structure value(s), first in column "
            f"{column!r} at research row {int(bad[0][0])}. smc_v1 guarantees a finite "
            "value on every row (docs/smc_v1.md §5); a NaN here would silently change "
            "which rows the information sets are compared on."
        )

    columns: dict[str, np.ndarray] = {
        name: spine[name].to_numpy(dtype=np.float64) for name in feature_columns()
    }
    for position, name in enumerate(smc_feature_columns()):
        columns[name] = smc_values[:, position]

    chart = compute_chart_features_segmented(raw, timeframe, chart_spec)
    chart_values = chart[chart_feature_columns()].to_numpy(dtype=np.float64)[raw_rows]
    if not np.isfinite(chart_values).all():
        bad = np.argwhere(~np.isfinite(chart_values))
        column = chart_feature_columns()[int(bad[0][1])]
        raise ValueError(
            f"{len(bad)} non-finite chart-structure value(s), first in column "
            f"{column!r} at research row {int(bad[0][0])}. chart_structure_v1 guarantees a "
            "finite value on every row (docs/chart_structure_v1.md §5)."
        )
    for position, name in enumerate(chart_feature_columns()):
        columns[name] = chart_values[:, position]

    micro_evidence: dict[str, Any] = {"trade_source": "absent; no P3 arm may be built"}
    applied_micro_spec: MicrostructureSpec | None = None
    if trade_aggregates is not None:
        applied_micro_spec = micro_spec or MicrostructureSpec()
        columns, micro_evidence = _join_microstructure(
            spine, spine_dates, trade_aggregates, applied_micro_spec, columns
        )

    derivatives_evidence: dict[str, Any] = {
        "derivatives_source": "absent; no P4 arm may be built"
    }
    applied_derivatives_spec: DerivativesSpec | None = None
    eligible: np.ndarray | None = None
    if derivatives is not None:
        if len(derivatives.eligible) != len(spine):
            raise ValueError(
                f"the P4 sample universe covers {len(derivatives.eligible)} rows and the "
                f"research spine has {len(spine)}. A mask built against another spine "
                "would move every fold's rows without moving a single number."
            )
        if tuple(derivatives.feature_names) != tuple(derivatives_feature_columns()):
            raise ValueError(
                "the P4 sample universe carries columns "
                f"{list(derivatives.feature_names)}, not {derivatives_feature_columns()}"
            )
        applied_derivatives_spec = DerivativesSpec()
        if derivatives.spec_hash != applied_derivatives_spec.spec_hash():
            raise ValueError(
                f"the P4 sample universe was built under derivatives spec "
                f"{derivatives.spec_hash} and this run computes "
                f"{applied_derivatives_spec.spec_hash()}. Two spec hashes are two "
                "feature families; they must not share an arm name."
            )
        eligible = np.asarray(derivatives.eligible, dtype=bool)
        for position, name in enumerate(derivatives.feature_names):
            columns[name] = derivatives.features[:, position]
        derivatives_evidence = dict(derivatives.join_evidence)
        derivatives_evidence["universe"] = derivatives.to_dict()

    mtf_evidence: dict[str, Any] = {"mtf_source": "absent; no P5 arm may be built"}
    applied_mtf_spec: MtfSpec | None = None
    mtf_universe: dict[str, Any] | None = None
    if mtf_spec is not None:
        applied_mtf_spec = mtf_spec
        context = build_mtf_context(raw, applied_mtf_spec)
        row_eligible = np.asarray(context.eligible, dtype=bool)[raw_rows]
        values = context.values[raw_rows]
        if not np.isfinite(values).all():
            bad = np.argwhere(~np.isfinite(values))
            column = mtf_feature_columns()[int(bad[0][1])]
            raise ValueError(
                f"{len(bad)} non-finite mtf_v1 value(s), first in column {column!r} at "
                f"research row {int(bad[0][0])}. Every row is either eligible with a "
                "finite context on both clocks or ineligible and filled; a NaN here "
                "would silently change which rows the information sets are compared on."
            )
        if not np.isfinite(values[row_eligible]).all():  # pragma: no cover - implied above
            raise ValueError("an eligible mtf_v1 row is not finite")
        for position, name in enumerate(mtf_feature_columns()):
            columns[name] = values[:, position]
        mtf_universe = {
            "rows": int(len(row_eligible)),
            "rows_eligible": int(np.count_nonzero(row_eligible)),
            "eligible_fraction": round(float(row_eligible.mean()), 6),
            "universe_sha256": _universe_hash(row_eligible),
            "first_eligible_row": int(np.argmax(row_eligible)),
            "spec_hash": applied_mtf_spec.spec_hash(),
        }
        mtf_evidence = dict(context.evidence)
        mtf_evidence["universe"] = mtf_universe
        mtf_evidence["context"] = {
            k: v for k, v in context.to_dict().items() if k != "evidence"
        }
        if eligible is not None:
            raise ValueError(
                "two sample universes were supplied. A run may be restricted by one "
                "checkpoint's rule or another's, never by both at once: the arms would "
                "then be scored on an intersection nobody preregistered."
            )
        eligible = row_eligible

    evidence = _join_evidence(spine, raw, raw_rows, columns, timeframe)
    evidence["microstructure"] = micro_evidence
    evidence["derivatives"] = derivatives_evidence
    evidence["mtf"] = mtf_evidence
    degenerate = [
        evidence["matches_under_plus_one_shift"],
        evidence["matches_under_minus_one_shift"],
        micro_evidence.get("matches_under_plus_one_shift", False),
        micro_evidence.get("matches_under_minus_one_shift", False),
    ]
    if any(degenerate):
        raise ValueError(
            "the join check cannot tell a correct join from a shifted one on this data, "
            "so it is not evidence; it must not be reported as though it were"
        )

    base = _spine_arrays(spine, ds_meta)
    available = {name: information_set(name) for name in names}
    if extra_sets:
        available.update(extra_sets)

    views: dict[str, ResearchData] = {}
    micro_columns = set(microstructure_feature_columns())
    derivatives_columns = set(derivatives_feature_columns())
    mtf_columns_set = set(mtf_feature_columns())
    for name, spec in available.items():
        missing = [c for c in spec.columns if c not in columns]
        if missing and set(missing) <= mtf_columns_set:
            raise ValueError(
                f"information set {name!r} needs {len(missing)} higher-timeframe "
                "column(s) and this run was given no mtf spec. A P5 arm is built from "
                "the committed candles through nn.mtf; pass mtf_spec=, or run a "
                "checkpoint whose arms are functions of the 1h bar alone."
            )
        if missing and set(missing) <= derivatives_columns:
            raise ValueError(
                f"information set {name!r} needs {len(missing)} derivatives column(s) "
                "and this run was given no derivatives source. A P4 arm is built from "
                "the committed derivatives snapshot through nn.p4_universe; pass "
                "derivatives=, or run a checkpoint whose arms are functions of the "
                "candles alone."
            )
        if missing and set(missing) <= micro_columns:
            raise ValueError(
                f"information set {name!r} needs {len(missing)} microstructure column(s) "
                "and this run was given no trade source. A P3 arm is built from the "
                "committed trade snapshot; pass trade_aggregates, or run a checkpoint "
                "whose arms are functions of the candles alone."
            )
        if missing:
            raise ValueError(f"information set {name!r} needs unavailable columns: {missing}")
        views[name] = ResearchData(
            **base,
            feature_names=list(spec.columns),
            features=np.column_stack([columns[c] for c in spec.columns]),
        )

    return AlignedResearchSamples(
        views=views,
        sets=available,
        spine_dates=base["dates"],
        spine_targets=base["targets"],
        smc_spec=smc_spec,
        chart_spec=chart_spec,
        n_rows=len(spine),
        columns=columns,
        join_evidence=evidence,
        micro_spec=applied_micro_spec,
        derivatives_spec=applied_derivatives_spec,
        mtf_spec=applied_mtf_spec,
        eligible=eligible,
    )


def _join_evidence(
    spine: pd.DataFrame,
    raw: pd.DataFrame,
    raw_rows: np.ndarray,
    columns: dict[str, np.ndarray],
    timeframe: str,
) -> dict[str, Any]:
    """Re-derive enough of the join, independently, to catch a shifted row.

    The join is the step where a market-structure value could end up on the
    wrong candle, and a shift of one row is both the most likely mistake and
    the most damaging — it is a one-candle look-ahead on all 39 columns at once.
    Nothing else in the alignment proof can see it, because every other array is
    shared between the views by object identity.

    So two columns are rebuilt here from the source rather than from the engine:

    * ``smc_body_ratio`` — ``|c-o| / (h-l)``, purely local to one candle, no ATR
      and no state, so it can be recomputed from the raw candles at ``raw_rows``
      in one line and cannot agree under a shift;
    * an OHLCV14 column, taken straight from the spine, which pins the other
      half of the matrix to the rows the fold plan is expressed in.

    A ``+1`` and a ``-1`` shift are both scored so the evidence records not just
    that the join matched but that it would not have matched if it were wrong.
    """
    from chimera.features import _true_range, _wilder

    high = raw["high"].to_numpy(dtype=np.float64)[raw_rows]
    low = raw["low"].to_numpy(dtype=np.float64)[raw_rows]
    open_ = raw["open"].to_numpy(dtype=np.float64)[raw_rows]
    close = raw["close"].to_numpy(dtype=np.float64)[raw_rows]
    # Whole-series quantities, indexed down afterwards: `chart_pole_atr` spans
    # 60 candles back, so it cannot be rebuilt from the joined rows alone.
    all_close = raw["close"].to_numpy(dtype=np.float64)
    atr_den = np.empty(len(raw), dtype=np.float64)
    close_lag = np.full(len(raw), np.nan)
    close_full = np.full(len(raw), np.nan)
    # Per segment, because the engine resets at a market-data gap and a
    # re-derivation that did not would disagree by a real amount near every
    # boundary — which is a difference in the reference, not in the join.
    segments = _contiguous_segment_ids(raw["date"], timeframe).to_numpy()
    for segment in np.unique(segments):
        block = np.flatnonzero(segments == segment)
        frame = raw.iloc[block]
        closes = all_close[block]
        atr = _wilder(_true_range(frame), 14).to_numpy(dtype=np.float64)
        atr_den[block] = np.maximum(atr, 1e-12 * closes)
        if len(block) > 60:
            close_lag[block[60:]] = closes[:-60]
        if len(block) > 20:
            close_full[block[20:]] = closes[:-20]
    span = high - low
    expected = np.where(
        span > 0.0, np.abs(close - open_) / np.where(span > 0.0, span, 1.0), 0.0
    )

    joined = columns["smc_body_ratio"]
    if not np.allclose(joined, expected, rtol=0.0, atol=1e-12):
        worst = int(np.argmax(np.abs(joined - expected)))
        raise ValueError(
            f"the market-structure join is wrong: smc_body_ratio at research row {worst} "
            f"is {joined[worst]!r} but the raw candle there gives {expected[worst]!r}. A "
            "value is sitting on the wrong candle."
        )

    # A chart column too, on the same pattern. `chart_pole_atr` is the cheapest
    # one to rebuild from source — two closes and the ATR — and P2c's arms
    # contain no SMC column at all, so without this the join evidence for a
    # chart run re-derived nothing that run actually used. Rolling all 30 chart
    # columns one candle forward left the old check reporting `matches: true`.
    pole = (close_full[raw_rows] - close_lag[raw_rows]) / atr_den[raw_rows]
    joined_pole = columns["chart_pole_atr"]
    # Rows too early in their segment to have a 60-candle lag are `nan` in the
    # re-derivation and a declared default in the engine; they are not evidence
    # either way and are excluded rather than fudged with a tolerance.
    known = np.isfinite(pole)
    if not np.allclose(joined_pole[known], pole[known], rtol=0.0, atol=1e-9):
        worst = int(np.nanargmax(np.abs(joined_pole - pole)))
        raise ValueError(
            f"the chart-structure join is wrong: chart_pole_atr at research row {worst} "
            f"is {joined_pole[worst]!r} but the raw candles there give {pole[worst]!r}"
        )

    def chart_shifted_matches(offset: int) -> bool:
        rolled = np.roll(pole, offset)
        both = known & np.isfinite(rolled)
        return bool(np.allclose(joined_pole[both], rolled[both], rtol=0.0, atol=1e-9))

    def shifted_matches(offset: int) -> bool:
        rolled = np.roll(expected, offset)
        return bool(np.allclose(joined[1:-1], rolled[1:-1], rtol=0.0, atol=1e-12))

    return {
        "recomputed": [
            "smc_body_ratio from the raw candles",
            "chart_pole_atr from the raw candles",
        ],
        "not_evidence": (
            "an OHLCV14 column was previously compared against the spine column it was "
            "copied from, which could not fail; dropped rather than kept as decoration"
        ),
        "rows": int(len(raw_rows)),
        "matches": True,
        "matches_under_plus_one_shift": shifted_matches(1) or chart_shifted_matches(1),
        "matches_under_minus_one_shift": shifted_matches(-1) or chart_shifted_matches(-1),
        "chart_rows_compared": int(known.sum()),
        "raw_rows_first": int(raw_rows[0]),
        "raw_rows_last": int(raw_rows[-1]),
        "raw_candles_seen_by_the_engine": int(raw_rows[-1]) + 1,
    }


def _join_microstructure(
    spine: pd.DataFrame,
    spine_dates: pd.Series,
    trade_aggregates: pd.DataFrame,
    spec: MicrostructureSpec,
    columns: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute ``microstructure_v1`` and put each value on the candle it describes.

    The join is the step where a trade-flow value could end up on the wrong hour,
    and a shift of one row is both the most likely mistake and the most damaging:
    it is a one-hour look-ahead on all 32 columns at once, and nothing else in
    the alignment proof can see it, because every other array is shared between
    the views by object identity.

    Causality is made structural rather than inherited, the same way the raw
    candles are truncated above. The aggregate table is cut at the last hour the
    spine uses *before* the engine runs, so a future feature written with a
    centred window could not reach a later hour even if it tried. Trailing
    windows are unaffected: they only ever look backwards.
    """
    aggregates = trade_aggregates.reset_index(drop=True).copy()
    if "date" not in aggregates.columns:
        raise ValueError("the trade aggregate table has no 'date' column")
    aggregates["date"] = pd.to_datetime(aggregates["date"], utc=True)
    if not aggregates["date"].is_monotonic_increasing or not aggregates["date"].is_unique:
        raise ValueError("trade aggregate hours must be sorted and free of duplicates")

    last_used = spine_dates.iloc[-1]
    aggregates = aggregates.loc[aggregates["date"] <= last_used].reset_index(drop=True)
    if aggregates.empty:
        raise ValueError(
            "no trade aggregate hour is at or before the spine's last row; the trade "
            "source and the candles do not describe the same period"
        )

    # Positional join, not a merge: a merge that silently dropped or duplicated a
    # row would be indistinguishable from a clean one in the result's shape.
    agg_row_of = pd.Series(
        np.arange(len(aggregates), dtype=np.int64), index=aggregates["date"].to_numpy()
    )
    agg_rows = agg_row_of.reindex(spine_dates.to_numpy())
    if agg_rows.isna().any():
        missing = int(agg_rows.isna().sum())
        first = spine_dates[agg_rows.isna().to_numpy()].iloc[0]
        raise ValueError(
            f"{missing} research row(s) have no trade-aggregate hour (first: {first}). "
            "The trade source does not cover the candles it is being aligned to, and a "
            "microstructure row for those candles would have to be invented."
        )
    agg_rows = agg_rows.to_numpy(dtype=np.int64)

    micro = compute_microstructure_features(aggregates, spec)
    names = microstructure_feature_columns()
    values = micro[names].to_numpy(dtype=np.float64)[agg_rows]
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))
        column = names[int(bad[0][1])]
        raise ValueError(
            f"{len(bad)} non-finite microstructure value(s), first in column {column!r} at "
            "research row "
            f"{int(bad[0][0])}. microstructure_v1 guarantees a finite value on every row "
            "(docs/microstructure_v1.md §5); a NaN here would silently change which rows "
            "the information sets are compared on."
        )
    for position, name in enumerate(names):
        columns[name] = values[:, position]

    _check_trade_segment_contiguity(spine, agg_rows)
    return columns, _microstructure_join_evidence(aggregates, agg_rows, spine_dates, columns)


def _microstructure_join_evidence(
    aggregates: pd.DataFrame,
    agg_rows: np.ndarray,
    spine_dates: pd.Series,
    columns: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Re-derive two microstructure columns from the source, two different ways.

    Two, and by two different join mechanisms, because they fail differently:

    * ``ms_qty_imbalance`` is rebuilt **positionally**, from the aggregate rows
      the join selected. It is local to one hour — ``(2·buy_qty − qty) / qty``,
      no window and no state — so it can be recomputed in one line and cannot
      agree under a shift.
    * ``ms_log_trade_count`` is rebuilt **by timestamp**, through a lookup keyed
      on the hour itself rather than on a row number. A positional join that was
      off by one would satisfy neither, but a positional check alone would also
      pass if ``agg_rows`` and the re-derivation were shifted together.

    A ``+1`` and a ``−1`` shift are both scored, so the evidence records not just
    that the join matched but that it would not have matched if it were wrong.
    """
    eps = MicrostructureSpec().eps
    qty = aggregates["qty"].to_numpy(dtype=np.float64)
    buy_qty = aggregates["buy_qty"].to_numpy(dtype=np.float64)
    expected = ((2.0 * buy_qty - qty) / np.maximum(qty, eps))[agg_rows]
    joined = columns["ms_qty_imbalance"]
    if not np.allclose(joined, expected, rtol=0.0, atol=1e-12):
        worst = int(np.argmax(np.abs(joined - expected)))
        raise ValueError(
            f"the microstructure join is wrong: ms_qty_imbalance at research row {worst} "
            f"is {joined[worst]!r} but the trade aggregate there gives {expected[worst]!r}. "
            "A value is sitting on the wrong hour."
        )

    by_hour = pd.Series(
        np.log1p(aggregates["n_trades"].to_numpy(dtype=np.float64)),
        index=pd.DatetimeIndex(aggregates["date"]),
    )
    looked_up = by_hour.reindex(pd.DatetimeIndex(spine_dates)).to_numpy(dtype=np.float64)
    if not np.allclose(columns["ms_log_trade_count"], looked_up, rtol=0.0, atol=1e-12):
        worst = int(np.argmax(np.abs(columns["ms_log_trade_count"] - looked_up)))
        raise ValueError(
            "the microstructure join is wrong: ms_log_trade_count at research row "
            f"{worst} does not match the aggregate hour with that timestamp"
        )

    def shifted_matches(offset: int) -> bool:
        rolled = np.roll(expected, offset)
        return bool(np.allclose(joined[1:-1], rolled[1:-1], rtol=0.0, atol=1e-12))

    return {
        "recomputed": [
            "ms_qty_imbalance from the trade aggregates, positionally",
            "ms_log_trade_count from the trade aggregates, by timestamp",
        ],
        "rows": int(len(agg_rows)),
        "matches": True,
        "matches_under_plus_one_shift": shifted_matches(1),
        "matches_under_minus_one_shift": shifted_matches(-1),
        "aggregate_rows_first": int(agg_rows[0]),
        "aggregate_rows_last": int(agg_rows[-1]),
        "aggregate_hours_seen_by_the_engine": int(len(aggregates)),
        "warmup_hours_before_the_spine": int(agg_rows[0]),
    }


def _check_trade_segment_contiguity(spine: pd.DataFrame, agg_rows: np.ndarray) -> None:
    """Two spine rows in one segment must be adjacent hours in the trade source.

    The spine's segment ids decide which windows may span which rows.
    ``microstructure_v1`` resets its trailing windows on the *trade table's* own
    gaps. If the two disagreed — if the spine called two rows contiguous that the
    trade hours separate — a window would carry trailing state across a gap the
    windowing believed was not there, which is the bridging the gap rule forbids.

    The converse is allowed and is not checked, for the reason
    :func:`_check_segment_contiguity` gives: a spine boundary with no trade gap
    behind it makes the windowing stricter than the feature state, and that
    direction cannot leak.
    """
    segments = spine["segment_id"].to_numpy(dtype=np.int64)
    same_segment = segments[1:] == segments[:-1]
    step = np.diff(agg_rows)
    offenders = np.flatnonzero(same_segment & (step != 1))
    if offenders.size:
        row = int(offenders[0])
        raise ValueError(
            f"research rows {row} and {row + 1} share segment {int(segments[row])} but are "
            f"{int(step[row])} hours apart in the trade source. The processed dataset's "
            "gap structure and the trade table's gap structure disagree, so trailing "
            "microstructure state would bridge a gap the windowing does not know about."
        )


def _spine_arrays(spine: pd.DataFrame, ds_meta: DatasetMetadata) -> dict[str, Any]:
    """The non-feature arrays every view shares, built exactly once.

    Returned as a kwargs dict so each :class:`ResearchData` receives the *same*
    array objects. Copying them per view would still be correct today and would
    remove the identity check :meth:`AlignedResearchSamples.prove_alignment`
    relies on to prove it tomorrow.
    """
    from chimera.contracts import TargetSpec
    from chimera.features import FeatureSpec
    from nn.data_pipeline import timeframe_to_minutes

    segment_ids = (
        spine["segment_id"].to_numpy(dtype=np.int64) if "segment_id" in spine.columns else None
    )
    if segment_ids is None:
        raise ValueError(
            "the research spine has no 'segment_id' column, so windows could bridge a "
            "market-data gap. Rebuild it with the current tools.build_features."
        )
    timeframe = ds_meta.timeframe or "1h"
    return {
        "ds_meta": ds_meta,
        "feature_spec": FeatureSpec.from_dict(ds_meta.feature_spec),
        "target_spec": TargetSpec.from_dict(ds_meta.target_spec),
        "targets": spine["target"].to_numpy(dtype=np.int64),
        "future_return": spine["future_return"].to_numpy(dtype=np.float64),
        "close": spine["close"].to_numpy(dtype=np.float64),
        "segment_ids": segment_ids,
        "dates": spine["date"].to_numpy(),
        "candles_per_year": 365 * 24 * 60 / timeframe_to_minutes(timeframe),
    }


def _check_segment_contiguity(
    spine: pd.DataFrame, raw_rows: np.ndarray, raw_dates: pd.Series, timeframe: str
) -> None:
    """Two spine rows in one segment must be adjacent candles in the raw history.

    The spine's segment ids decide which windows are allowed to span which rows.
    Market structure resets on the *raw* history's gaps. If the two disagreed —
    if the spine called two rows contiguous that the raw candles separate — a
    window would carry structure state across a gap the windowing believed was
    not there, which is exactly the bridging the gap rule forbids.

    The converse is allowed and is not checked. A spine boundary with no raw gap
    behind it happens when :func:`nn.data_pipeline.build_dataset` drops a row for
    a NaN feature and re-derives the segments over what is left. There the
    windowing is *stricter* than the market-structure state: no window spans the
    dropped row, while the structure entering the next row was built from the
    real, contiguous, past candles on either side of it. That direction cannot
    leak, so it is left alone rather than forced into agreement.
    """
    segments = spine["segment_id"].to_numpy(dtype=np.int64)
    same_segment = segments[1:] == segments[:-1]
    raw_step = np.diff(raw_rows)
    offenders = np.flatnonzero(same_segment & (raw_step != 1))
    if offenders.size:
        row = int(offenders[0])
        raise ValueError(
            f"research rows {row} and {row + 1} share segment {int(segments[row])} but are "
            f"{int(raw_step[row])} candles apart in the raw history. The processed "
            "dataset's gap structure and the raw candles' gap structure disagree, so "
            "market-structure state would bridge a gap the windowing does not know about."
        )
    raw_segments = _contiguous_segment_ids(raw_dates, timeframe).to_numpy()
    aligned_segments = raw_segments[raw_rows]
    crossing = np.flatnonzero(same_segment & (aligned_segments[1:] != aligned_segments[:-1]))
    if crossing.size:
        row = int(crossing[0])
        raise ValueError(
            f"research rows {row} and {row + 1} share a processed segment but fall in "
            "different raw segments; market-structure state would bridge a market-data gap"
        )


def feature_spec_identity(
    smc_spec: SMCSpec, ds_meta: DatasetMetadata, chart_spec: ChartSpec | None = None
) -> dict[str, Any]:
    """What produced the columns, recorded in every information-set artifact.

    Both halves, because both can change independently: the OHLCV14 window
    lengths come from the dataset that was built months ago, and the market
    structure constants come from the spec this run was written against.
    """
    chart_spec = chart_spec or ChartSpec()
    return {
        "ohlcv14_feature_spec": dict(ds_meta.feature_spec),
        "chart_spec_version": CHART_SPEC_VERSION,
        "chart_spec": chart_spec.to_dict(),
        "chart_spec_hash": chart_spec.spec_hash(),
        "chart_feature_families": {k: list(v) for k, v in CHART_FEATURE_FAMILIES.items()},
        "smc_spec_version": SMC_SPEC_VERSION,
        "smc_spec": smc_spec.to_dict(),
        "smc_spec_hash": smc_spec.spec_hash(),
        "smc_feature_families": {k: list(v) for k, v in SMC_FEATURE_FAMILIES.items()},
        "combined_spec_hash": hashlib.sha256(
            json.dumps(
                {
                    "ohlcv14": dict(ds_meta.feature_spec),
                    "smc": smc_spec.to_dict(),
                    "smc_version": SMC_SPEC_VERSION,
                    "chart": chart_spec.to_dict(),
                    "chart_version": CHART_SPEC_VERSION,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }


def derivatives_spec_identity(
    derivatives_spec: DerivativesSpec,
    derivatives_source: Mapping[str, Any],
    universe: Mapping[str, Any],
) -> dict[str, Any]:
    """What produced the P4 columns and which rows they were scored on.

    A third block rather than a widening of the two above, for the reason
    :func:`microstructure_spec_identity` gives: ``combined_spec_hash`` is
    recorded in every committed P2b, P2c and P3 cell, and widening what it covers
    would give the same research a different identity.

    The **universe** is in here and not beside it. P4 is the first checkpoint
    whose arms are not scored on the whole spine, so "which rows" is part of what
    a P4 number means in exactly the way the feature constants are: two cells
    built over different intersections are not two measurements of one thing, and
    a comparison must be able to refuse to join them without recomputing the mask.
    """
    return {
        "derivatives_spec_version": DERIVATIVES_SPEC_VERSION,
        "derivatives_spec": derivatives_spec.to_dict(),
        "derivatives_spec_hash": derivatives_spec.spec_hash(),
        "derivatives_feature_families": {
            k: list(v) for k, v in DERIVATIVES_FEATURE_FAMILIES.items()
        },
        "derivatives_source": dict(derivatives_source),
        "universe": dict(universe),
        "combined_derivatives_hash": hashlib.sha256(
            json.dumps(
                {
                    "spec": derivatives_spec.to_dict(),
                    "version": DERIVATIVES_SPEC_VERSION,
                    "columns": list(derivatives_feature_columns()),
                    "source": dict(derivatives_source),
                    "universe_hash": universe.get("universe_hash"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }


def microstructure_spec_identity(
    micro_spec: MicrostructureSpec, trade_source: Mapping[str, Any]
) -> dict[str, Any]:
    """What produced the P3 columns, recorded in every P3 artifact.

    Kept apart from :func:`feature_spec_identity` rather than folded into it, and
    deliberately. That function's ``combined_spec_hash`` is recorded in every
    committed P2b and P2c cell; widening what it covers would mean a re-run of
    those checkpoints produced a different identity for the same research, which
    is precisely the silent reinterpretation the hash exists to prevent. P3 is a
    new source, so it gets a new block.

    Both halves are here because both can change independently: the feature
    constants come from the spec this run was written against, and the source
    identity comes from the trade snapshot it read.
    """
    return {
        "microstructure_spec_version": MICROSTRUCTURE_SPEC_VERSION,
        "microstructure_spec": micro_spec.to_dict(),
        "microstructure_spec_hash": micro_spec.spec_hash(),
        "microstructure_feature_families": {
            k: list(v) for k, v in MICROSTRUCTURE_FEATURE_FAMILIES.items()
        },
        "trade_source": dict(trade_source),
        "combined_microstructure_hash": hashlib.sha256(
            json.dumps(
                {
                    "spec": micro_spec.to_dict(),
                    "version": MICROSTRUCTURE_SPEC_VERSION,
                    "columns": list(microstructure_feature_columns()),
                    "source": dict(trade_source),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
