"""Shared, dependency-light building blocks used by the data pipeline, the
training code, the inference service and the Freqtrade strategies.

Nothing in this package imports torch, freqtrade or ccxt, so it can be loaded
in every container of the stack.
"""

__all__ = ["contracts", "features", "risk", "safety"]
