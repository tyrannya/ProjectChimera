"""Shared, dependency-light building blocks used by the data pipeline, the
training code, the inference service, the recorder and the demo runtime.

Nothing in this package imports torch or ccxt, so it can be loaded in every
container of the stack. Freqtrade was one of the consumers until R1-m deleted
that path; no code in this repository imports it any more.
"""

__all__ = ["contracts", "features", "risk", "safety"]
