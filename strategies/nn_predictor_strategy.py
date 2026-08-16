"""Freqtrade strategy driven by the inference service.

What it does, per candle:

1. compute features from the real dataframe with :func:`chimera.features.compute_features`
   — the same function the training pipeline uses;
2. take the window of ``sequence_length`` rows ending at the last **closed**
   candle;
3. ``POST /predict`` with those raw features (the service applies the scaler);
4. validate the response;
5. compare the winning probability with the model's own decision threshold;
6. emit an entry or exit signal.

What it explicitly does not do:

* no ``GET /predict`` without data;
* no comparison of a price against a probability;
* **no network call per dataframe row.** In live and dry-run only the final
  closed candle is sent, once, cached on pair + timestamp + model version.
  In backtest the service is not called at all — see below.

Backtesting
-----------
Calling an HTTP service once per historical row is both unusably slow and
dishonest, because the answer would come from today's model applied to a past
candle regardless of what was known then. So in backtest and hyperopt this
strategy predicts **in-process** from a local model artifact, batching the
whole dataframe through it. Features are causal, so batching introduces no
look-ahead. If no local artifact is configured, the strategy produces no
signals at all and says so — rather than quietly backtesting a different
strategy.

Failure behaviour
-----------------
Service unreachable, timeout, malformed response, feature-contract mismatch, or
too little history all resolve to HOLD, which means no new entry. The fallback
is never "become a rule-based strategy": that would silently backtest and trade
something other than what was intended. A rule-based fallback can be enabled
explicitly with ``nn_fallback_strategy``, and it is off by default.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from freqtrade.enums import RunMode

from chimera import metrics
from chimera.contracts import Signal
from chimera.features import FeatureSpec, compute_features, feature_columns
from chimera.inference_client import InferenceClient
from strategies.common.risk_manager import RiskAwareStrategy

logger = logging.getLogger(__name__)

#: Numeric encoding of Signal for the dataframe columns.
_SIGNAL_CODE = {Signal.SHORT: -1, Signal.HOLD: 0, Signal.LONG: 1}


class NNPredictorStrategy(RiskAwareStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.03}
    stoploss = -0.05
    startup_candle_count: int = 400
    can_short = False
    process_only_new_candles = True

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.client = InferenceClient(
            config.get("nn_infer_url", "http://nn_infer:3000"),
            timeout_s=float(config.get("nn_timeout_s", 2.0)),
            retries=int(config.get("nn_retries", 1)),
        )
        self.local_model_dir: str | None = config.get("nn_local_model_dir")
        self.enable_rule_fallback = bool(config.get("nn_fallback_strategy", False))
        self._last_prediction_at: dict[str, float] = {}
        self._local: Any = None

    # ------------------------------------------------------------------
    def _feature_frame(self, dataframe: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
        return compute_features(dataframe, spec)

    def _contract_ok(self, names: tuple[str, ...]) -> bool:
        """Refuse to serve a model whose feature list differs from our code."""
        if not names:
            return True  # older service that does not publish the contract
        if list(names) == feature_columns():
            return True
        logger.error(
            "Feature contract mismatch: the served model expects %s but this "
            "strategy computes %s. Holding.",
            list(names),
            feature_columns(),
        )
        return False

    def _predict_live(self, dataframe: pd.DataFrame, pair: str) -> tuple[int, float]:
        """One prediction for the final closed candle. ``(code, confidence)``."""
        info = self.client.describe()
        if info is None:
            logger.warning("Inference service unavailable for %s; holding", pair)
            metrics.mark_inference_failure("unavailable")
            return 0, 0.0

        if not self._contract_ok(info.feature_names):
            metrics.mark_inference_failure("contract_mismatch")
            return 0, 0.0

        spec = FeatureSpec.from_dict(info.feature_spec) if info.feature_spec else FeatureSpec()
        features = self._feature_frame(dataframe, spec)

        window = features.iloc[-info.sequence_length :]
        if len(window) < info.sequence_length or window.isna().to_numpy().any():
            logger.warning(
                "Not enough clean history for %s (%d/%d rows); holding",
                pair,
                len(window.dropna()),
                info.sequence_length,
            )
            return 0, 0.0

        if window.shape[1] != info.n_features:
            logger.error(
                "Computed %d features but the model expects %d; holding",
                window.shape[1],
                info.n_features,
            )
            metrics.mark_inference_failure("contract_mismatch")
            return 0, 0.0

        candle_time = dataframe["date"].iloc[-1]
        prediction = self.client.predict(pair, self.timeframe, candle_time, window.to_numpy())
        if prediction is None:
            metrics.mark_inference_failure("no_prediction")
            return 0, 0.0

        self._last_prediction_at[pair] = prediction.received_at
        signal = Signal(prediction.signal)
        logger.info(
            "%s @ %s -> %s (confidence %.3f, model %s)",
            pair,
            candle_time,
            signal.value,
            prediction.confidence,
            prediction.model_version,
        )
        return _SIGNAL_CODE[signal], prediction.confidence

    def _predict_offline(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Batch predictions from a local artifact, for backtests.

        Returns a frame with ``nn_signal`` and ``nn_confidence`` aligned to
        ``dataframe``'s index. All-HOLD when no artifact is configured.
        """
        result = pd.DataFrame({"nn_signal": 0, "nn_confidence": 0.0}, index=dataframe.index)
        if not self.local_model_dir:
            logger.warning(
                "Backtesting %s without nn_local_model_dir: this strategy produces "
                "no signals. Set nn_local_model_dir to a model artifact directory.",
                pair,
            )
            return result

        if self._local is None:
            # Imported lazily: torch is only needed for offline backtests, not
            # for live trading against the service.
            from nn.registry import load_model

            model, metadata = load_model(self.local_model_dir)
            self._local = (model, metadata)
            logger.info("Loaded local model %s for backtesting", metadata.model_version)

        import torch

        model, metadata = self._local
        if not self._contract_ok(tuple(metadata.feature_names)):
            return result

        features = self._feature_frame(dataframe, metadata.feature_spec)
        values = features.to_numpy(dtype=np.float64)
        valid = np.isfinite(values).all(axis=1)

        seq_len = metadata.sequence_length
        mean = np.asarray(metadata.scaler_mean)
        std = np.asarray(metadata.scaler_std)
        scaled = (values - mean) / std

        # Row i is predictable when the whole window [i-seq_len+1, i] is finite.
        window_ok = pd.Series(valid).rolling(seq_len).min().fillna(0).to_numpy().astype(bool)
        rows = np.flatnonzero(window_ok)
        if rows.size == 0:
            return result

        offsets = np.arange(-seq_len + 1, 1)
        windows = scaled[rows[:, None] + offsets[None, :]].astype(np.float32)

        probabilities = []
        with torch.no_grad():
            for start in range(0, len(windows), 512):
                batch = torch.from_numpy(windows[start : start + 512])
                probabilities.append(torch.softmax(model(batch).float(), dim=-1).numpy())
        proba = np.concatenate(probabilities)

        from nn.evaluate import signals_from_proba

        classes = signals_from_proba(proba, metadata.decision_threshold)
        # signals_from_proba returns class indices (0=SHORT, 1=HOLD, 2=LONG);
        # the dataframe uses -1/0/+1.
        result.iloc[rows, result.columns.get_loc("nn_signal")] = classes - 1
        result.iloc[rows, result.columns.get_loc("nn_confidence")] = proba.max(axis=1)
        return result

    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]
        dataframe["nn_signal"] = 0
        dataframe["nn_confidence"] = 0.0

        if dataframe.empty:
            return dataframe

        run_mode = self.dp.runmode if self.dp else RunMode.OTHER
        if run_mode in (RunMode.LIVE, RunMode.DRY_RUN):
            code, confidence = self._predict_live(dataframe, pair)
            # Only the last row carries a live prediction. Broadcasting one
            # scalar across the whole history is exactly the look-ahead bug
            # this rewrite exists to remove.
            dataframe.loc[dataframe.index[-1], "nn_signal"] = code
            dataframe.loc[dataframe.index[-1], "nn_confidence"] = confidence
        else:
            offline = self._predict_offline(dataframe, pair)
            dataframe["nn_signal"] = offline["nn_signal"].to_numpy()
            dataframe["nn_confidence"] = offline["nn_confidence"].to_numpy()

        if self.enable_rule_fallback:
            self._add_fallback_columns(dataframe)
        return dataframe

    def _add_fallback_columns(self, dataframe: pd.DataFrame) -> None:
        """Opt-in rule-based columns, used only where the model said nothing.

        Off by default. When on, this genuinely is a different strategy, which
        is why it must be requested explicitly in the config rather than being
        reached by accident when the network blips.
        """
        dataframe["fb_ema_fast"] = dataframe["close"].ewm(span=21, adjust=False).mean()
        dataframe["fb_ema_slow"] = dataframe["close"].ewm(span=55, adjust=False).mean()

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        if dataframe.empty:
            return dataframe

        long_signal = dataframe["nn_signal"] == 1
        if self.enable_rule_fallback:
            long_signal |= (dataframe["nn_signal"] == 0) & (
                dataframe["fb_ema_fast"] > dataframe["fb_ema_slow"]
            )
        dataframe.loc[long_signal, "enter_long"] = 1
        dataframe.loc[long_signal, "enter_tag"] = "nn_long"

        if self.can_short:
            short_signal = dataframe["nn_signal"] == -1
            dataframe.loc[short_signal, "enter_short"] = 1
            dataframe.loc[short_signal, "enter_tag"] = "nn_short"
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        if dataframe.empty:
            return dataframe

        # Exit on an opposing call only. HOLD means "no opinion", and exiting on
        # HOLD would close every position the moment the service went quiet.
        dataframe.loc[dataframe["nn_signal"] == -1, "exit_long"] = 1
        if self.can_short:
            dataframe.loc[dataframe["nn_signal"] == 1, "exit_short"] = 1
        return dataframe

    # ------------------------------------------------------------------
    def data_age_seconds(self, pair: str, current_time: datetime) -> float | None:
        """Age of the newest candle, for the risk engine's staleness guard."""
        if self.dp is None:
            return None
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except Exception:  # noqa: BLE001
            return None
        if dataframe is None or dataframe.empty:
            return None
        last = pd.Timestamp(dataframe["date"].iloc[-1]).to_pydatetime()
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age = (current_time - last).total_seconds()
        metrics.DATA_STALENESS.labels(pair=pair).set(age)
        return age

    def inference_age_seconds(self, pair: str) -> float | None:
        received = self._last_prediction_at.get(pair)
        if received is None:
            return None
        return max(0.0, time.time() - received)
