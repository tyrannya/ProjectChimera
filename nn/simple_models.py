"""The predeclared simple-model family for research checkpoint P2a.

P2a asks one diagnostic question: **does model complexity add predictive or
economic value when simple models receive exactly the same information as the
MTST Transformer?** It changes one thing — the model family — and holds the
information set, the research generation, the fold geometry, the labels, the
costs and the threshold rule fixed.

**Information parity is the whole experiment.** MTST reads a ``(seq_len,
n_features)`` sequence per sample; so do the models here. The sequence is not
approximated, re-derived or truncated to the latest candle: it is the very array
:func:`nn.dataset.build_windows` produced for MTST, flattened by
:func:`flatten_windows` in one declared order. Giving a tabular model only the
last row would answer a different question — "is one candle enough?" — and would
make any gap between the families uninterpretable.

**The flattening order is part of the contract.** ``X[n, t, f]`` becomes column
``t * n_features + f`` (C order), so a row reads oldest timestep first and, within
each timestep, the dataset's own feature order. :func:`flattened_column_names`
names every column of the 896-wide row, and a test pins both the order and the
values against a hand-built extraction.

**Scaling.** All three models receive the *scaled* windows, standardised by the
train-only :class:`nn.dataset.StandardScaler` MTST uses. Trees do not need
scaling — a per-feature affine map with positive slope cannot change which rows
fall on which side of a split, so the fitted trees are the same either way — but
feeding them a different array than MTST saw would weaken the parity claim for
no benefit, so one scaled array is used for every family.

**The configurations below are fixed and predeclared.** No Optuna, no Ray Tune,
no grid or random search, no early stopping, no post-hoc edit after seeing an
outer result. They are conservative library defaults with the depth/estimator
budget stated up front, and the only permitted change is a compatibility
adaptation when a named parameter no longer exists in the installed library —
recorded, when it happens, in the provenance the runner writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable

import numpy as np

from chimera.contracts import CLASS_ORDER

N_CLASSES = len(CLASS_ORDER)

#: How ``X[n, t, f]`` is laid out in the flattened row. Recorded in provenance.
#:
#: Stated as a formula rather than "C order" because the reader of an artifact
#: has to be able to reconstruct the mapping without knowing NumPy's conventions,
#: and because a silent change of order would leave every recorded number intact
#: while meaning something else.
FLATTEN_ORDER = "column j = timestep t * n_features + feature f, t oldest first"

#: Separator between the timestep offset and the feature name in a column name.
COLUMN_SEPARATOR = "__"


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """Flatten MTST's ``(n, seq_len, n_features)`` windows to ``(n, seq_len*n_features)``.

    The one place the tabular row is built. It takes the already-windowed,
    already-scaled array rather than a dataset and a split, so there is no
    second sample-construction path that could drift from MTST's: whatever
    :func:`nn.dataset.build_windows` emitted is what gets flattened, or this
    raises.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (n, seq_len, n_features) windows, got shape {X.shape}")
    n, seq_len, n_features = X.shape
    # `reshape` on a C-contiguous gather is a view where possible and a copy
    # otherwise; either way the element order is the one FLATTEN_ORDER states.
    return np.ascontiguousarray(X).reshape(n, seq_len * n_features)


def flattened_column_names(feature_names: list[str], seq_len: int) -> list[str]:
    """Name every flattened column, in the order :func:`flatten_windows` emits.

    ``t-63__ret_1`` is the ``ret_1`` value 63 candles before the decision row;
    ``t-0__volume_z`` is ``volume_z`` on the decision row itself. Offsets are
    relative to the sample's own row, so the names are true of every sample and
    a reader can check the ordering claim without the arrays.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if not feature_names:
        raise ValueError("feature_names must not be empty")
    return [
        f"t-{seq_len - 1 - step}{COLUMN_SEPARATOR}{name}"
        for step in range(seq_len)
        for name in feature_names
    ]


def align_class_columns(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Reorder a classifier's probability columns into :data:`CLASS_ORDER`.

    Every scikit-learn-style estimator orders ``predict_proba`` columns by its
    own ``classes_``, which is the *sorted set of labels present in the training
    rows* — not the project's class encoding. The two agree only as long as all
    three classes appear in every training block. Rather than depend on that,
    the columns are placed by label and an absent class gets probability zero,
    which is what "the model never learned this class" means.
    """
    proba = np.asarray(proba, dtype=np.float64)
    classes = np.asarray(classes)
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError(
            f"probabilities of shape {proba.shape} do not match {len(classes)} classes"
        )
    out = np.zeros((len(proba), N_CLASSES), dtype=np.float64)
    for position, label in enumerate(classes):
        index = int(label)
        if not 0 <= index < N_CLASSES:
            raise ValueError(f"classifier emitted unknown class label {label!r}")
        out[:, index] = proba[:, position]
    return out


@dataclass(frozen=True)
class SimpleModelSpec:
    """One predeclared model: how to build it, and what to record about it.

    ``params`` is the complete fixed configuration; nothing else is passed to
    the constructor, so the artifact's record of the configuration and the
    object actually fitted cannot diverge. ``seeded`` names the constructor
    argument the per-fold seed goes into, or is empty for a deterministic
    estimator that takes none.
    """

    name: str
    module: str
    factory: str
    params: dict[str, Any]
    seeded: str = ""
    #: Any minimal adaptation made for the installed library version, recorded
    #: in provenance. Empty when the declared configuration was used verbatim.
    compatibility_notes: tuple[str, ...] = field(default_factory=tuple)

    def config(self, seed: int) -> dict[str, Any]:
        """The exact keyword arguments this model is constructed with."""
        params = dict(self.params)
        if self.seeded:
            params[self.seeded] = seed
        return params

    def build(self, seed: int) -> Any:
        """Instantiate the estimator, importing its library only when needed."""
        try:
            module = import_module(self.module)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                f"{self.name} needs the '{self.module}' package, which is not installed. "
                "Install the benchmark dependency group: pip install -e '.[benchmark]'"
            ) from exc
        return getattr(module, self.factory)(**self.config(seed))

    def version(self) -> str:
        """The installed version of the library this model comes from."""
        root = import_module(self.module.split(".")[0])
        return str(getattr(root, "__version__", "unknown"))


#: Multinomial logistic regression on the flattened window.
#:
#: ``multi_class`` is deliberately not passed: scikit-learn deprecated it in 1.5
#: and removed it in 1.7, and ``lbfgs`` has fit the multinomial loss by default
#: throughout. Naming it would either warn or fail depending on the installed
#: version while changing nothing about what is fitted.
LOGISTIC_REGRESSION = SimpleModelSpec(
    name="logistic_regression",
    module="sklearn.linear_model",
    factory="LogisticRegression",
    params={
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        # Stated rather than omitted: "no class weighting" is a declared choice
        # of this benchmark, not an accident of the library default.
        "class_weight": None,
    },
)

#: Gradient-boosted trees, LightGBM.
#:
#: ``deterministic=True`` needs an explicit histogram-construction mode, so
#: ``force_row_wise=True`` is set with it; ``n_jobs=1`` removes the remaining
#: thread-scheduling non-determinism. ``subsample=1.0`` is the declared value
#: and ``subsample_freq`` stays at its default of 0, which is what makes it
#: mean "no bagging" rather than "bag the whole set".
LIGHTGBM = SimpleModelSpec(
    name="lightgbm",
    module="lightgbm",
    factory="LGBMClassifier",
    params={
        "objective": "multiclass",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    },
    seeded="random_state",
    compatibility_notes=(
        "deterministic=True requires an explicit histogram mode, so force_row_wise=True "
        "is set alongside it; n_jobs=1 for reproducible CPU behaviour",
    ),
)

#: Gradient-boosted trees, XGBoost, on multiclass soft probabilities.
XGBOOST = SimpleModelSpec(
    name="xgboost",
    module="xgboost",
    factory="XGBClassifier",
    params={
        "objective": "multi:softprob",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "device": "cpu",
        "n_jobs": 1,
    },
    seeded="random_state",
    compatibility_notes=(
        "device='cpu' is XGBoost 2.x's replacement for the removed gpu_id/predictor "
        "arguments; n_jobs=1 for reproducible CPU behaviour",
    ),
)

#: The predeclared benchmark, in report order. Fixed before any BTC result was
#: seen, and not to be extended or reordered on the strength of one.
SIMPLE_MODELS: tuple[SimpleModelSpec, ...] = (LOGISTIC_REGRESSION, LIGHTGBM, XGBOOST)

#: Model names this benchmark reports, in table order.
SIMPLE_MODEL_NAMES = tuple(spec.name for spec in SIMPLE_MODELS)


@dataclass(frozen=True)
class FittedSimpleModel:
    """A fitted estimator plus the provenance of how it was fitted.

    ``predict`` takes MTST-shaped windows, not flattened rows: the flattening
    happens inside, so a caller cannot accidentally hand a differently-built
    tabular row to a model that was fitted on this one.
    """

    spec: SimpleModelSpec
    estimator: Any
    seed: int
    library_version: str
    n_features_in: int
    train_class_counts: dict[str, int]

    @property
    def name(self) -> str:
        return self.spec.name

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities in :data:`CLASS_ORDER` for MTST-shaped windows."""
        flat = flatten_windows(X)
        if flat.shape[1] != self.n_features_in:
            raise ValueError(
                f"{self.name} was fitted on {self.n_features_in} flattened features but "
                f"was given {flat.shape[1]}; the two families are not seeing one "
                "information set"
            )
        return align_class_columns(self.estimator.predict_proba(flat), self.estimator.classes_)

    def provenance(self) -> dict[str, Any]:
        """Everything needed to rebuild this fit, minus the data itself."""
        return {
            "model": self.name,
            "library": self.spec.module.split(".")[0],
            "library_version": self.library_version,
            "estimator": f"{self.spec.module}.{self.spec.factory}",
            "config": self.spec.config(self.seed),
            "seed": self.seed,
            "seed_parameter": self.spec.seeded or None,
            "flattened_input_dim": self.n_features_in,
            "train_class_counts": self.train_class_counts,
            "compatibility_notes": list(self.spec.compatibility_notes),
            "early_stopping": False,
            "hyperparameter_search": "none (predeclared fixed configuration)",
        }


def fit_simple_model(
    spec: SimpleModelSpec, X_train: np.ndarray, y_train: np.ndarray, *, seed: int
) -> FittedSimpleModel:
    """Fit one predeclared model on one fold's **training** windows.

    ``X_train`` is MTST's own ``(n, seq_len, n_features)`` training array. No
    validation array is accepted — not "not used by default": there is no
    parameter here an early-stopping or calibration set could arrive through,
    which is what makes "nothing was fitted on inner or outer validation" a
    property of the signature rather than a promise about the caller.
    """
    flat = flatten_windows(X_train)
    y_train = np.asarray(y_train, dtype=np.int64)
    if len(flat) != len(y_train):
        raise ValueError("training windows and labels must have the same length")
    estimator = spec.build(seed)
    estimator.fit(flat, y_train)
    counts = np.bincount(y_train, minlength=N_CLASSES)
    return FittedSimpleModel(
        spec=spec,
        estimator=estimator,
        seed=seed,
        library_version=spec.version(),
        n_features_in=flat.shape[1],
        train_class_counts={
            signal.value: int(counts[index]) for index, signal in enumerate(CLASS_ORDER)
        },
    )


def proba_function(fitted: FittedSimpleModel) -> Callable[[np.ndarray], np.ndarray]:
    """Adapt a fitted model to the frozen-scoring hook in :mod:`nn.train`."""
    return fitted.predict_proba
