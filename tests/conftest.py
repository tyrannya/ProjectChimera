"""Shared fixtures.

Two things are deliberately global here:

* the repository root goes on ``sys.path`` so ``chimera``/``nn``/``strategies``
  import the same way they do at runtime;
* nothing stubs out a third-party module. The previous suite replaced
  ``freqtrade`` with a hand-written ``types.ModuleType`` that defined a
  ``TemporaryStopException`` the real library does not have, so the tests
  asserted against a fictional dependency and stayed green while the code was
  unimportable. Tests that need Freqtrade import the real Freqtrade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.make_sample_data import generate_candles  # noqa: E402


@pytest.fixture
def candles():
    """1000 deterministic synthetic candles satisfying the OHLC invariants."""
    return generate_candles(rows=1000, seed=1234)


@pytest.fixture
def small_candles():
    """400 candles: enough for a short window, small enough to stay fast."""
    return generate_candles(rows=400, seed=99)
