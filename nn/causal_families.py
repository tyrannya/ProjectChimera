"""Every causal feature family, in one table the generic tests iterate over.

``nn/smc.py`` and ``nn/chart_structure.py`` promise the same things — append
invariance, no NaN, a declared column order, a family partition, a reset at a
market-data gap, a spec hash over predeclared constants — and each proves them
in its own file, ``tests/test_smc_features.py`` and
``tests/test_chart_structure.py``. Those tests are the ones worth having: their
expected numbers were derived from the specs by hand, and only a per-family test
can say what ``smc_bos_bull`` should equal on a particular candle.

What they cannot do is notice an omission. A third family inherits none of them,
and nothing in this repository fails if its author simply does not write the
append-invariance test — the property most likely to be quietly false, because a
``center=True`` or a ``shift(-1)`` yields a frame that looks right and silently
re-dates history every time a candle arrives. Registering a family here makes
``tests/test_causal_families.py`` check all of it, so the cost of an unwritten
test is a red suite rather than a backtest nobody can falsify.

The registry is additive. ``nn/information_sets.py`` keeps importing the two
engines directly: it is imported by the P2b/P2c benchmark runner and its source
digest is recorded in frozen evidence, so rewiring it through this table would
invalidate artifacts that already answered a question, in exchange for nothing.

``spec`` holds the frozen constants object itself, not just a hash callable. A
test can show a hash is stable from the hash alone, but it can only show the
hash *moves* when a constant moves if it can build a perturbed spec from the
registered one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import pandas as pd

from nn.chart_structure import (
    CHART_FEATURE_FAMILIES,
    CHART_FEATURE_NAMES,
    CHART_SPEC_VERSION,
    ChartSpec,
    compute_chart_features,
    compute_chart_features_segmented,
)
from nn.smc import (
    SMC_FEATURE_FAMILIES,
    SMC_FEATURE_NAMES,
    SMC_SPEC_VERSION,
    SMCSpec,
    compute_smc_features,
    compute_smc_features_segmented,
)


@dataclass(frozen=True)
class CausalFeatureFamily:
    """One registered engine: what it emits, how to call it, where it is defined."""

    #: Registry key, and the information-set name in :mod:`nn.information_sets`.
    #: It repeats ``version`` for both families today because each set is named
    #: after the spec version it computes.
    name: str
    version: str
    columns: tuple[str, ...]
    families: dict[str, tuple[str, ...]]
    #: ONE contiguous segment. Bound to ``spec``, so the table cannot declare one
    #: set of constants and compute another.
    compute: Callable[[pd.DataFrame], pd.DataFrame]
    #: ``(candles, timeframe)``; splits on market-data gaps first.
    compute_segmented: Callable[[pd.DataFrame, str], pd.DataFrame]
    #: The frozen dataclass of predeclared constants (``SMCSpec``, ``ChartSpec``).
    spec: Any
    #: Repository-relative path of the document that defines the family.
    spec_document: str

    def spec_hash(self) -> str:
        return self.spec.spec_hash()


_SMC_SPEC = SMCSpec()
_CHART_SPEC = ChartSpec()

CAUSAL_FEATURE_FAMILIES: dict[str, CausalFeatureFamily] = {
    SMC_SPEC_VERSION: CausalFeatureFamily(
        name=SMC_SPEC_VERSION,
        version=SMC_SPEC_VERSION,
        columns=SMC_FEATURE_NAMES,
        families=SMC_FEATURE_FAMILIES,
        compute=partial(compute_smc_features, spec=_SMC_SPEC),
        compute_segmented=partial(compute_smc_features_segmented, spec=_SMC_SPEC),
        spec=_SMC_SPEC,
        spec_document="docs/smc_v1.md",
    ),
    CHART_SPEC_VERSION: CausalFeatureFamily(
        name=CHART_SPEC_VERSION,
        version=CHART_SPEC_VERSION,
        columns=CHART_FEATURE_NAMES,
        families=CHART_FEATURE_FAMILIES,
        compute=partial(compute_chart_features, spec=_CHART_SPEC),
        compute_segmented=partial(compute_chart_features_segmented, spec=_CHART_SPEC),
        spec=_CHART_SPEC,
        spec_document="docs/chart_structure_v1.md",
    ),
}
